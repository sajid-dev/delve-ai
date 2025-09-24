from __future__ import annotations

import asyncio
import json
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from loguru import logger
from mcp import types as mcp_types
from langchain_mcp import MCPToolkit
from langchain_mcp_adapters.client import MultiServerMCPClient

from ..config.mcp_config import McpConfig, McpServerConfig


@dataclass(slots=True)
class ToolCallPlan:
    """Plan describing which tool to call and which arguments to supply."""

    tool: mcp_types.Tool
    arguments: dict[str, Any]


class RouterChain:
    """Decide which MCP servers are relevant for a given prompt."""

    def __init__(self, servers: Iterable[McpServerConfig], fallback_keywords: list[str]) -> None:
        self._servers = list(servers)
        self._fallback_keywords = [kw.lower() for kw in fallback_keywords if kw]

    def select(self, prompt: str) -> list[McpServerConfig]:
        """Return the servers whose keywords appear in the prompt."""

        lowered = prompt.lower()
        selected: list[McpServerConfig] = []
        for server in self._servers:
            keywords = [kw.lower() for kw in server.trigger_keywords if kw]
            if not keywords:
                keywords = list(self._fallback_keywords)
            if not keywords or any(keyword in lowered for keyword in keywords):
                selected.append(server)
        return selected


class ServerSchemaMap:
    """Provide simple argument schemas for each configured server."""

    def __init__(self, servers: Iterable[McpServerConfig]) -> None:
        self._schema: dict[str, dict[str, Any]] = {}
        for server in servers:
            self._schema[self._identifier(server)] = {
                "query": {
                    "type": "string",
                    "description": "Original user request passed to the MCP tool.",
                }
            }

    def schema_for(self, server: McpServerConfig) -> dict[str, Any]:
        """Return the schema describing expected arguments for the server."""

        return self._schema.get(self._identifier(server), {"query": {"type": "string"}})

    @staticmethod
    def _identifier(server: McpServerConfig) -> str:
        return server.name or server.command


class ArgumentExtractor:
    """Convert natural language prompts into JSON arguments for MCP tools."""

    def __init__(self, schema_map: ServerSchemaMap) -> None:
        self._schema_map = schema_map

    def build_plans(
        self,
        server: McpServerConfig,
        prompt: str,
        tools: list[mcp_types.Tool],
    ) -> list[ToolCallPlan]:
        if not tools:
            return []

        schema = self._schema_map.schema_for(server)
        arguments = self._populate_arguments(schema, prompt)
        if arguments is None:
            return []

        # Use the first available tool by default. Servers can expose a single
        # entry point that understands the "query" argument containing the user request.
        primary_tool = tools[0]
        return [ToolCallPlan(tool=primary_tool, arguments=arguments)]

    def _populate_arguments(
        self, schema: dict[str, Any], prompt: str
    ) -> dict[str, Any] | None:
        if not schema:
            return {}

        arguments: dict[str, Any] = {}
        for name, meta in schema.items():
            field_type = meta.get("type")
            if field_type == "string":
                arguments[name] = prompt
            elif "default" in meta:
                arguments[name] = meta["default"]
        return arguments


class QueryCapableMultiServerMCPClient(MultiServerMCPClient):
    """Extend the default client with helper methods for orchestration."""

    async def list_tools(self, server_id: str) -> list[mcp_types.Tool]:
        async with self.session(server_id) as session:
            toolkit = MCPToolkit(session=session)
            await toolkit.initialize()
            return list(toolkit._tools.tools) if toolkit._tools else []

    async def query(
        self,
        server_id: str,
        *,
        tool: str,
        arguments: dict[str, Any] | None = None,
    ) -> mcp_types.CallToolResult:
        async with self.session(server_id) as session:
            return await session.call_tool(tool, arguments=arguments or {})


class MCPContextCollector:
    """Collect and post-process context from MCP tools for LLM prompts."""

    def __init__(self, mcp_config: McpConfig) -> None:
        self._config = mcp_config
        self._router = RouterChain(mcp_config.servers, mcp_config.trigger_keywords)
        self._schema_map = ServerSchemaMap(mcp_config.servers)
        self._argument_extractor = ArgumentExtractor(self._schema_map)

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

        selected_servers = self._router.select(prompt)
        if not selected_servers:
            logger.debug(
                "RouterChain found no relevant MCP servers for session={}",
                session_id,
            )
            return None

        multi_client = self._build_multi_server_client(selected_servers)
        aggregated_sections: list[str] = []
        offline_servers: list[str] = []

        for server in selected_servers:
            server_id = self._server_identifier(server)
            try:
                available_tools = await multi_client.list_tools(server_id)
            except Exception:
                logger.exception(
                    "Failed to initialise MCP server=%s for session=%s",
                    server.label,
                    session_id,
                )
                offline_servers.append(server.label)
                continue

            plans = self._argument_extractor.build_plans(server, prompt, available_tools)
            if not plans:
                logger.info(
                    "Argument extractor produced no plan for server=%s; skipping",
                    server.label,
                )
                continue

            refined_results: list[dict[str, Any]] = []
            for plan in plans:
                try:
                    tool_result = await multi_client.query(
                        server_id,
                        tool=plan.tool.name,
                        arguments=plan.arguments,
                    )
                except Exception:
                    logger.exception(
                        "MCP tool %s invocation failed on server=%s",
                        plan.tool.name,
                        server.label,
                    )
                    continue

                if tool_result.isError:
                    logger.warning(
                        "MCP tool %s returned an error payload on server=%s",
                        plan.tool.name,
                        server.label,
                    )
                    continue

                refined = self._refine_tool_output(plan.tool, tool_result, server.label)
                if refined:
                    refined_results.append(refined)

            if refined_results:
                aggregated_sections.append(self._format_tool_context(refined_results))
                logger.debug(
                    "Server %s produced %d refined MCP result(s)",
                    server.label,
                    len(refined_results),
                )
            else:
                logger.debug(
                    "Server %s returned no actionable MCP context for session=%s",
                    server.label,
                    session_id,
                )

        if aggregated_sections:
            merged = "\n\n".join(aggregated_sections)
            logger.debug(
                "Aggregated MCP context for session={} (length={})",
                session_id,
                len(merged),
            )
            return merged

        if offline_servers:
            notice = self._format_offline_notice(offline_servers)
            logger.warning(
                "MCP servers unavailable for session={}: {}",
                session_id,
                offline_servers,
            )
            return notice

        logger.debug(
            "No contextual data returned from MCP servers for session={}",
            session_id,
        )
        return None

    def _refine_tool_output(
        self,
        tool_info: mcp_types.Tool,
        tool_result: mcp_types.CallToolResult,
        server_label: str,
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
            "server": server_label,
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
            preview = (
                self._truncate(json.dumps(payload, ensure_ascii=False))
                if isinstance(payload, (dict, list))
                else None
            )
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
    ) -> QueryCapableMultiServerMCPClient:
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
        return QueryCapableMultiServerMCPClient(connections)

    @staticmethod
    def _server_identifier(server: McpServerConfig) -> str:
        return server.name or server.command
