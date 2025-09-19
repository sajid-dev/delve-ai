"""Service encapsulating interactions with the language model.

Uses LangChain's ChatOpenAI integration to communicate with the
OpenAI Chat API.  The service accepts a prompt and a memory
instance and returns the generated response.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import defaultdict
from typing import Any

from loguru import logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from mcp import ClientSession
from mcp import types as mcp_types
from mcp.client.stdio import StdioServerParameters, stdio_client

from ..config.llm_config import LlmConfig, get_llm_config
from ..memory.chat_memory import ChatMemory
from ..models.chat_message import ChatMessage
from ..models.enums import MessageRole


class LLMService:
    """Service for generating responses from the language model.

    This service wraps LangChain's :class:`~langchain_openai.ChatOpenAI` class and
    constructs it based solely on a unified :class:`LlmConfig` instance.  All
    provider‑specific details (such as API keys, base URL and model name) are
    defined in a single configuration.  There is no longer any distinction
    between OpenAI and other providers – if a ``base_url`` is provided the
    client will communicate with the supplied endpoint, otherwise it will use
    the default OpenAI endpoint.  Additional parameters such as ``temperature``,
    ``max_tokens`` and ``timeout`` are forwarded verbatim to the underlying
    ChatOpenAI constructor.
    """

    def __init__(self, llm_config: LlmConfig | None = None) -> None:
        """Initialise the LLM service with the provided configuration.

        Parameters
        ----------
        llm_config: LlmConfig, optional
            A configuration instance specifying API credentials, base URL,
            model name and tuning parameters.  If omitted, the configuration
            will be loaded from environment variables via :func:`get_llm_config`.
        """
        # Load LLM configuration if none supplied
        self.llm_config = llm_config or get_llm_config()

        # Build keyword arguments for ChatOpenAI using the unified config.  At a
        # minimum the API key, model name and temperature are provided.  If a
        # base URL, max_tokens or timeout are specified they are also passed
        # through.  These keys correspond to aliases documented in the
        # ``langchain_openai`` API reference.
        self._llm_kwargs: dict[str, object] = {
            "api_key": self.llm_config.api_key,
            "model": self.llm_config.model,
            "temperature": self.llm_config.temperature,
        }
        if self.llm_config.base_url:
            self._llm_kwargs["base_url"] = self.llm_config.base_url
        # Pass optional tuning parameters only if they differ from defaults
        if self.llm_config.max_tokens:
            self._llm_kwargs["max_tokens"] = self.llm_config.max_tokens
        if self.llm_config.timeout:
            self._llm_kwargs["timeout"] = self.llm_config.timeout

        # Optional keyword arguments forwarded to the model constructor (e.g. user-level tags).
        self._model_kwargs: dict[str, object] = {}

        init_kwargs: dict[str, object] = dict(self._llm_kwargs)
        if self._model_kwargs:
            init_kwargs["model_kwargs"] = dict(self._model_kwargs)

        # Initialise the ChatOpenAI model with the assembled kwargs.  Unspecified
        # parameters will fall back to library defaults.
        self.llm = ChatOpenAI(**init_kwargs)

        # Base system prompt used for every interaction.  Additional context is
        # appended dynamically from the conversation memory on each request.
        self._system_prompt = (
            "You are a helpful AI assistant. Use the provided conversation context "
            "when it is available. If the context is empty, respond using only the "
            "latest user message."
        )

    def generate(self, prompt: str, memory: ChatMemory, user_id: str | None = None) -> str:
        """Generate a response using the provided prompt and memory.

        This method wraps the LangChain call in a try/except block so that
        any exceptions are converted into a ChatError.  This ensures
        upstream callers can handle failures uniformly.

        Parameters
        ----------
        prompt: str
            The user's input message.
        memory: ChatMemory
            A chat memory instance providing conversation history.
        user_id: str, optional
            Identifier of the end user for the request.  When provided the
            identifier is forwarded to the underlying LLM so usage can be
            tracked per user.

        Returns
        -------
        str
            The assistant's reply.

        Raises
        ------
        ChatError
            If an unexpected error occurs during generation.
        """
        logger.debug("Generating response for prompt: {!r}", prompt)
        try:
            llm = self.llm
            if user_id:
                # Recreate the LLM with the user identifier, forwarding via model_kwargs
                # to avoid warnings from the underlying client.
                model_kwargs = {**self._model_kwargs, "user": user_id}
                llm = ChatOpenAI(**self._llm_kwargs, model_kwargs=model_kwargs)

            system_message = self._build_system_message(prompt, memory)

            tool_context: str | None = None
            if self.llm_config.mcp_enabled and self._should_use_mcp(prompt):
                try:
                    tool_context = self._collect_mcp_context(prompt)
                except Exception:
                    logger.exception("Failed to collect MCP tool context")

            return self._invoke_llm(llm, system_message, prompt, tool_context)
        except Exception as exc:
            logger.exception("LLM generation failed")
            from ..utils.error_handler import ChatError

            raise ChatError("LLM generation failed") from exc

    # ------------------------------------------------------------------
    # Token accounting helpers

    def count_tokens(self, messages: list[ChatMessage]) -> int:
        """Return token usage for a sequence of chat messages.

        Falls back to a simple character-based heuristic if the underlying
        model does not expose token counting utilities.
        """
        if not messages:
            return 0

        if hasattr(self.llm, "get_num_tokens_from_messages"):
            lc_messages = []
            for message in messages:
                role = message.role
                content = message.content
                if role == MessageRole.USER:
                    lc_messages.append(HumanMessage(content=content))
                elif role == MessageRole.ASSISTANT:
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(SystemMessage(content=content))
            try:
                return int(self.llm.get_num_tokens_from_messages(lc_messages))
            except Exception:
                # Fall through to heuristic if token counting fails
                pass

        # Simple heuristic: average of 4 characters per token with minimum of 1
        return sum(max(1, math.ceil(len(message.content) / 4)) for message in messages)

    # ------------------------------------------------------------------
    # Internal helpers

    def _build_system_message(self, prompt: str, memory: ChatMemory) -> str:
        """Construct the system prompt including contextual memory."""
        history_snippets = memory.get_relevant_history(prompt)
        if history_snippets:
            return f"{self._system_prompt}\n\nContext:\n{history_snippets}"
        return f"{self._system_prompt}\n\nContext: <none>"

    def _invoke_llm(self, llm: ChatOpenAI, system_message: str, prompt: str, tool_context: str | None = None) -> str:
        """Invoke the base LLM with the provided system and user messages."""
        human_content = prompt
        if tool_context:
            human_content = (
                f"{prompt}\n\n"
                f"Additional data gathered from MCP tools:\n{tool_context}"
            )
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_content),
        ]
        response = llm.invoke(messages)
        content = getattr(response, "content", str(response))
        return content.strip()

    def _collect_mcp_context(self, prompt: str) -> str | None:
        """Synchronously collect additional tool context via the configured MCP transport."""
        if self.llm_config.mcp_transport != "stdio":
            raise ValueError("Only the 'stdio' MCP transport is currently supported")

        try:
            return asyncio.run(self._acollect_mcp_context(prompt))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._acollect_mcp_context(prompt))
            finally:
                loop.close()

    async def _acollect_mcp_context(self, prompt: str) -> str | None:
        """Async helper that launches the MCP server, selects tools and refines their outputs."""
        server_params = StdioServerParameters(
            command=self.llm_config.mcp_server_command or "",
            args=self.llm_config.mcp_server_args,
            env=self.llm_config.mcp_server_env,
            cwd=self.llm_config.mcp_server_cwd,
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            session = ClientSession(read_stream, write_stream)
            await session.initialize()

            tools_result = await session.list_tools()
            available_tools = list(tools_result.tools)
            if not available_tools:
                logger.info("MCP server exposed no tools for prompt")
                return None

            selected_tools = self._select_mcp_tools(prompt, available_tools)
            if not selected_tools:
                logger.info("No MCP tools matched the current prompt; skipping tool calls")
                return None

            refined_results: list[dict[str, Any]] = []
            for tool_info in selected_tools:
                arguments = self._prepare_tool_arguments(tool_info, prompt)
                if arguments is None:
                    continue

                try:
                    tool_result = await session.call_tool(tool_info.name, arguments=arguments)
                except Exception:
                    logger.exception("MCP tool {} invocation failed", tool_info.name)
                    continue

                if tool_result.isError:
                    logger.warning("MCP tool {} returned an error payload", tool_info.name)
                    continue

                refined = self._refine_tool_output(tool_info, tool_result)
                if refined:
                    refined_results.append(refined)

            if not refined_results:
                return None

            return self._format_tool_context(refined_results)

    def _should_use_mcp(self, prompt: str) -> bool:
        """Return True when the prompt should trigger MCP tool usage."""
        keywords = [kw.lower() for kw in self.llm_config.mcp_trigger_keywords if kw]
        if not keywords:
            return True
        lowered_prompt = prompt.lower()
        return any(keyword in lowered_prompt for keyword in keywords)

    def _select_mcp_tools(self, prompt: str, tools: list[mcp_types.Tool]) -> list[mcp_types.Tool]:
        """Select tools that appear relevant to the prompt using heuristics."""
        if not tools:
            return []

        keywords = [kw.lower() for kw in self.llm_config.mcp_trigger_keywords if kw]
        lowered_prompt = prompt.lower()
        selected: list[mcp_types.Tool] = []
        seen: set[str] = set()

        def add(tool: mcp_types.Tool) -> None:
            if tool.name not in seen:
                selected.append(tool)
                seen.add(tool.name)

        if keywords:
            for tool in tools:
                haystack = f"{tool.name} {(tool.description or '')}".lower()
                if any(keyword in lowered_prompt and keyword in haystack for keyword in keywords):
                    add(tool)
            if selected:
                return selected

        for tool in tools:
            tokens = tool.name.lower().replace("_", " ").split()
            if any(token and token in lowered_prompt for token in tokens):
                add(tool)
        if selected:
            return selected

        return tools

    def _prepare_tool_arguments(self, tool: mcp_types.Tool, prompt: str) -> dict[str, Any] | None:
        """Populate tool arguments using the prompt when possible."""
        schema = tool.inputSchema or {}
        properties = schema.get("properties", {})
        if not properties:
            return {}

        arguments: dict[str, Any] = {}
        for name, meta in properties.items():
            field_type = meta.get("type")
            if field_type == "string":
                arguments[name] = prompt
            elif field_type == "array" and meta.get("items", {}).get("type") == "string":
                arguments[name] = [prompt]

        required = schema.get("required", [])
        missing = [name for name in required if name not in arguments]
        if missing:
            logger.debug("Skipping MCP tool {} due to unsupported required arguments {}", tool.name, missing)
            return None

        return arguments

    def _refine_tool_output(
        self,
        tool_info: mcp_types.Tool,
        tool_result: mcp_types.CallToolResult,
    ) -> dict[str, Any] | None:
        """Extract text/structured content and apply business logic."""
        text_output = self._render_text_content(tool_result.content)
        summary, metrics, preview = self._apply_business_logic(
            tool_info.name,
            text_output,
            tool_result.structuredContent,
        )

        if not summary and not metrics and not preview:
            return None

        refined: dict[str, Any] = {
            "name": tool_info.name,
            "description": tool_info.description or "",
            "summary": summary or "",
        }
        if metrics:
            refined["metrics"] = metrics
        if preview:
            refined["raw_preview"] = preview
        return refined

    @staticmethod
    def _render_text_content(content: list[mcp_types.TextContent | mcp_types.EmbeddedResource | mcp_types.ImageContent]) -> str:
        """Flatten textual blocks from a tool response."""
        if not content:
            return ""

        fragments: list[str] = []
        for block in content:
            if isinstance(block, mcp_types.TextContent):
                fragments.append(block.text)
        return "\n".join(fragment.strip() for fragment in fragments if fragment.strip())

    def _apply_business_logic(
        self,
        tool_name: str,
        text_output: str,
        structured_payload: dict[str, Any] | None,
    ) -> tuple[str | None, dict[str, Any] | None, str | None]:
        """Apply basic aggregation and summarisation rules to tool output."""
        payload: Any | None = structured_payload
        if payload is None:
            payload = self._try_parse_json(text_output)

        if payload is not None:
            summary, metrics = self._summarize_structured_data(payload)
            preview = self._truncate(json.dumps(payload, ensure_ascii=False)) if isinstance(payload, (dict, list)) else None
            if not preview and text_output:
                preview = self._truncate(text_output)
            return summary, metrics, preview

        preview = self._truncate(text_output) if text_output else None
        summary = None
        if preview:
            summary = preview.splitlines()[0]
        return summary, None, preview

    def _summarize_structured_data(self, payload: Any) -> tuple[str, dict[str, Any] | None]:
        """Return a textual summary and numeric aggregations for structured payloads."""
        if isinstance(payload, list):
            if not payload:
                return "Tool returned an empty list.", None

            if all(isinstance(item, (int, float)) for item in payload):
                metrics = self._aggregate_numeric_values([float(item) for item in payload])
                summary = f"Processed {len(payload)} numeric values from MCP tool."
                return summary, metrics

            if all(isinstance(item, dict) for item in payload):
                aggregates: dict[str, list[float]] = defaultdict(list)
                for item in payload:
                    for key, value in item.items():
                        if isinstance(value, (int, float)):
                            aggregates[key].append(float(value))

                if aggregates:
                    metrics = {
                        key: self._aggregate_numeric_values(values)
                        for key, values in aggregates.items()
                    }
                    summary = (
                        f"Aggregated {len(payload)} records across {len(metrics)} numeric field(s)."
                    )
                    return summary, metrics

                summary = f"Processed {len(payload)} records without numeric fields to aggregate."
                return summary, None

        if isinstance(payload, dict):
            numeric_fields = {
                key: float(value)
                for key, value in payload.items()
                if isinstance(value, (int, float))
            }
            if numeric_fields:
                metrics = {
                    key: self._aggregate_numeric_values([value])
                    for key, value in numeric_fields.items()
                }
                summary = "Extracted numeric metrics from MCP tool payload."
                return summary, metrics

        preview = json.dumps(payload, ensure_ascii=False)
        summary = "Structured data returned; no numeric aggregations available."
        return summary, {"data_preview": self._truncate(preview)}

    @staticmethod
    def _aggregate_numeric_values(values: list[float]) -> dict[str, float]:
        """Return standard aggregate statistics for numeric collections."""
        if not values:
            return {}

        total = sum(values)
        count = len(values)
        average = total / count
        return {
            "count": count,
            "sum": round(total, 3),
            "average": round(average, 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }

    def _format_tool_context(self, results: list[dict[str, Any]]) -> str:
        """Combine refined tool results into a prompt-friendly text block."""
        sections: list[str] = []
        for result in results:
            lines = [f"Tool {result['name']}: {result['summary']}"]
            description = result.get("description")
            if description:
                lines.append(f"Description: {description}")
            metrics = result.get("metrics")
            if metrics:
                metrics_str = self._truncate(self._stringify_metrics(metrics))
                lines.append(f"Metrics: {metrics_str}")
            preview = result.get("raw_preview")
            if preview:
                lines.append(f"Preview: {preview}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    @staticmethod
    def _truncate(text: str, limit: int = 600) -> str:
        """Limit text length to ensure the prompt stays compact."""
        if text is None:
            return ""
        stripped = text.strip()
        if len(stripped) <= limit:
            return stripped
        return stripped[:limit].rstrip() + "…"

    @staticmethod
    def _stringify_metrics(metrics: dict[str, Any]) -> str:
        """Serialise metrics to JSON for inclusion in prompts."""
        try:
            return json.dumps(metrics, ensure_ascii=False)
        except Exception:
            return str(metrics)

    @staticmethod
    def _try_parse_json(candidate: str) -> Any | None:
        """Attempt to parse a string as JSON, returning None on failure."""
        if not candidate:
            return None
        candidate = candidate.strip()
        if not candidate:
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
