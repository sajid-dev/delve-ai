from __future__ import annotations

import asyncio
import json
import threading
from collections import defaultdict
from typing import Any

from loguru import logger
from mcp import types as mcp_types
from langchain_mcp import MCPToolkit
from langchain_mcp_adapters.client import MultiServerMCPClient

from ..config.mcp_config import McpConfig, McpServerConfig


class MCPContextCollector:
    """Collect and post-process context from MCP tools for LLM prompts."""

    def __init__(self, mcp_config: McpConfig) -> None:
        self._config = mcp_config

    def should_use_mcp(self, prompt: str) -> bool:
        """Return True when the prompt should trigger MCP tool usage."""
        if not self._config.enabled or not self._config.servers:
            return False

        lowered_prompt = prompt.lower()
        for server in self._config.servers:
            keywords = self._keywords_for_server(server)
            if not keywords or any(keyword in lowered_prompt for keyword in keywords):
                return True
        return False

    def collect_context(self, prompt: str, session_id: str | None = None) -> str | None:
        """Synchronously collect additional tool context via the configured MCP transport."""
        if self._config.transport != "stdio":
            raise ValueError("Only the 'stdio' MCP transport is currently supported")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._acollect_context(prompt, session_id=session_id))

        result: dict[str, Any] = {}
        error: list[BaseException] = []

        def runner() -> None:
            local_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(local_loop)
                result["value"] = local_loop.run_until_complete(
                    self._acollect_context(prompt, session_id=session_id)
                )
            except BaseException as exc:  # pragma: no cover - pass-through for diagnostics
                error.append(exc)
            finally:
                asyncio.set_event_loop(None)
                local_loop.close()

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()

        if error:
            raise error[0]
        return result.get("value")

    async def _acollect_context(self, prompt: str, session_id: str | None = None) -> str | None:
        """Async helper that launches MCP servers, selects tools and aggregates results."""

        if not self._config.enabled:
            return None

        relevant_servers = [
            server
            for server in self._config.servers
            if self._should_query_server(server, prompt)
        ]

        if not relevant_servers:
            logger.debug(
                f"No MCP servers matched the current prompt for session={session_id}"
            )
            return None

        logger.debug(
            f"Selected MCP servers {[server.label for server in relevant_servers]} for session={session_id}"
        )

        client = self._build_multi_server_client(relevant_servers)
        aggregated_sections: list[str] = []
        offline_servers: list[str] = []
        for server in relevant_servers:
            try:
                section = await self._collect_from_server(
                    client,
                    server,
                    prompt,
                    session_id,
                )
            except Exception:
                logger.exception(
                    "Failed collecting MCP context from server={}",
                    server.label,
                )
                offline_servers.append(server.label)
                continue

            if section:
                aggregated_sections.append(section)

        if not aggregated_sections:
            if offline_servers:
                notice = self._format_offline_notice(offline_servers)
                logger.warning(
                    f"MCP servers unavailable for session={session_id}: {offline_servers}"
                )
                return notice

            logger.debug(
                f"No contextual data returned from MCP servers for session={session_id}"
            )
            return None

        merged = "\n\n".join(aggregated_sections)
        logger.debug(
            f"Aggregated MCP context for session={session_id} (length={len(merged)})"
        )
        return merged

    async def _collect_from_server(
        self,
        client: MultiServerMCPClient,
        server: McpServerConfig,
        prompt: str,
        session_id: str | None,
    ) -> str | None:
        """Collect context from a specific MCP server definition."""

        server_id = self._server_identifier(server)
        command_repr = self._describe_server_command(server)

        async with client.session(server_id) as session:
            toolkit = MCPToolkit(session=session)
            await toolkit.initialize()

            raw_tools: list[mcp_types.Tool] = list(toolkit._tools.tools) if toolkit._tools else []
            logger.debug(
                f"Initialized LangChain MCP toolkit for server={server.label} exposing {len(raw_tools)} tool(s)"
            )

            logger.debug(
                "Collecting MCP context from server={server} using command={command} for session={session} prompt_snippet={snippet}".format(
                    server=server.label,
                    command=command_repr,
                    session=session_id,
                    snippet=prompt[:80],
                )
            )

            available_tools = raw_tools
            if not available_tools:
                logger.info(
                    "MCP server {} exposed no tools for prompt",
                    server.label,
                )
                return None

            selected_tools = self._select_tools(prompt, available_tools, server)
            if not selected_tools:
                logger.info(
                    "No MCP tools on server {} matched the current prompt; skipping tool calls",
                    server.label,
                )
                return None

            refined_results: list[dict[str, Any]] = []
            for tool_info in selected_tools:
                arguments = self._prepare_tool_arguments(tool_info.name, tool_info.inputSchema, prompt)
                if arguments is None:
                    continue

                try:
                    tool_result = await session.call_tool(tool_info.name, arguments=arguments)
                except Exception:
                    logger.exception(
                        "MCP tool {} invocation failed on server={}",
                        tool_info.name,
                        server.label,
                    )
                    continue

                if tool_result.isError:
                    logger.warning(
                        "MCP tool {} returned an error payload on server={}",
                        tool_info.name,
                        server.label,
                    )
                    continue

                refined = self._refine_tool_output(tool_info, tool_result)
                if refined:
                    refined["server"] = server.label
                    refined_results.append(refined)

            if not refined_results:
                return None

            section = self._format_tool_context(refined_results)
            logger.debug(
                f"Server {server.label} produced {len(refined_results)} tool result(s) (length={len(section)})"
            )
            return section

    def _select_tools(
        self,
        prompt: str,
        tools: list[mcp_types.Tool],
        server: McpServerConfig,
    ) -> list[mcp_types.Tool]:
        """Select tools that appear relevant to the prompt using heuristics."""
        if not tools:
            return []

        keywords = self._keywords_for_server(server)
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

    def _prepare_tool_arguments(
        self,
        tool_name: str,
        schema: dict[str, Any] | None,
        prompt: str,
    ) -> dict[str, Any] | None:
        """Populate tool arguments using the prompt when possible."""
        schema = schema or {}
        properties = schema.get("properties", {})
        if not properties:
            return {}

        arguments: dict[str, Any] = {}
        for name, meta in properties.items():
            field_type = meta.get("type")
            enum_values = meta.get("enum") or []
            if enum_values:
                arguments[name] = enum_values[0]
                continue

            if field_type == "string":
                arguments[name] = prompt
            elif field_type == "array":
                items = meta.get("items", {})
                item_enum = items.get("enum") or []
                if item_enum:
                    arguments[name] = [item_enum[0]]
                elif items.get("type") == "string":
                    arguments[name] = [prompt]

        required = schema.get("required", [])
        missing = [name for name in required if name not in arguments]
        if missing:
            logger.debug(
                "Skipping MCP tool {} due to unsupported required arguments {}",
                tool_name,
                missing,
            )
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
    def _render_text_content(
        content: list[mcp_types.TextContent | mcp_types.EmbeddedResource | mcp_types.ImageContent],
    ) -> str:
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
            server_label = result.get("server")
            if server_label:
                lines.append(f"Server: {server_label}")
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

    def _should_query_server(self, server: McpServerConfig, prompt: str) -> bool:
        keywords = self._keywords_for_server(server)
        if not keywords:
            return True
        lowered_prompt = prompt.lower()
        return any(keyword in lowered_prompt for keyword in keywords)

    def _keywords_for_server(self, server: McpServerConfig) -> list[str]:
        keywords = server.trigger_keywords or self._config.trigger_keywords
        return [kw.lower() for kw in keywords if kw]

    @staticmethod
    def _describe_server_command(server: McpServerConfig) -> str:
        parts = [server.command]
        parts.extend(server.args)
        return " ".join(str(part) for part in parts if part)

    @staticmethod
    def _format_offline_notice(servers: list[str]) -> str:
        if not servers:
            return ""
        if len(servers) == 1:
            return f"MCP server '{servers[0]}' is currently unavailable."
        joined = ", ".join(servers)
        return f"MCP servers {joined} are currently unavailable."

    def _build_multi_server_client(
        self,
        servers: list[McpServerConfig],
    ) -> MultiServerMCPClient:
        connections: dict[str, dict[str, Any]] = {}
        for server in servers:
            server_id = self._server_identifier(server)
            connection: dict[str, Any] = {
                "transport": "stdio",
                "command": server.command,
                "args": list(server.args),
            }
            if server.env is not None:
                connection["env"] = server.env
            if server.cwd is not None:
                connection["cwd"] = server.cwd
            connections[server_id] = connection
        return MultiServerMCPClient(connections)

    @staticmethod
    def _server_identifier(server: McpServerConfig) -> str:
        return server.name or server.command
