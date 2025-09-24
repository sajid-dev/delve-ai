from __future__ import annotations

from functools import lru_cache
import json
import shlex
from typing import Optional

from dotenv import load_dotenv
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .mcp_servers import DEFAULT_MCP_SERVERS

load_dotenv()


SUPPORTED_TRANSPORTS = {"stdio", "sse", "streamable_http", "websocket"}
TRANSPORT_ALIASES = {
    "streamable-http": "streamable_http",
    "streamablehttp": "streamable_http",
    "http": "streamable_http",
}


def _normalise_transport(value: str | None) -> str | None:
    """Return a canonical transport string or None when not provided."""

    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip().lower()
        if not candidate:
            return None
        candidate = TRANSPORT_ALIASES.get(candidate, candidate)
        if candidate not in SUPPORTED_TRANSPORTS:
            raise ValueError(
                "Unsupported MCP transport '{candidate}'. Supported transports are: {supported}".format(
                    candidate=value,
                    supported=", ".join(sorted(SUPPORTED_TRANSPORTS | set(TRANSPORT_ALIASES))),
                )
            )
        return candidate
    raise TypeError("MCP transport must be provided as a string or null")


class McpServerConfig(BaseModel):
    """Definition for a single MCP server instance."""

    name: Optional[str] = None
    command: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    env: Optional[dict[str, str]] = None
    cwd: Optional[str] = None
    trigger_keywords: list[str] = Field(default_factory=list)
    transport: Optional[str] = None
    url: Optional[str] = None
    headers: Optional[dict[str, str]] = None
    timeout: Optional[float] = None
    sse_read_timeout: Optional[float] = None
    terminate_on_close: Optional[bool] = None

    @field_validator("args", mode="before")
    @classmethod
    def parse_args(cls, value: list[str] | str) -> list[str]:
        if isinstance(value, list):
            return value
        if not value:
            return []
        return shlex.split(value)

    @field_validator("trigger_keywords", mode="before")
    @classmethod
    def parse_keywords(cls, value: list[str] | str) -> list[str]:
        if isinstance(value, list):
            return value
        if not value:
            return []
        return [part.strip() for part in value.replace(",", " ").split() if part.strip()]

    @field_validator("headers", mode="before")
    @classmethod
    def parse_headers(cls, value: dict[str, str] | str | None) -> dict[str, str] | None:
        if value is None or value == "":
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("MCP server headers must be valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError("MCP server headers must decode to a JSON object")
            return {str(key): str(val) for key, val in parsed.items()}
        raise TypeError("MCP server headers must be provided as a mapping or JSON string")

    @field_validator("transport", mode="before")
    @classmethod
    def normalise_transport(cls, value: Optional[str]) -> Optional[str]:
        return _normalise_transport(value)

    @property
    def label(self) -> str:
        """Return a display name for logging and context output."""

        return self.name or self.command or (self.url or "<unnamed>")


class McpConfig(BaseSettings):
    """Settings controlling MCP tool discovery and invocation."""

    enabled: bool = Field(
        False,
        alias="MCP_ENABLED",
        validation_alias=AliasChoices("MCP_ENABLED", "LLM_MCP_ENABLED"),
    )
    transport: str = Field(
        "stdio",
        alias="MCP_TRANSPORT",
        validation_alias=AliasChoices("MCP_TRANSPORT", "LLM_MCP_TRANSPORT"),
    )
    trigger_keywords: list[str] = Field(
        default_factory=list,
        alias="MCP_TRIGGER_KEYWORDS",
        validation_alias=AliasChoices("MCP_TRIGGER_KEYWORDS", "LLM_MCP_TRIGGER_KEYWORDS"),
    )
    servers: list[McpServerConfig] = Field(
        default_factory=list,
        alias="MCP_SERVERS",
        validation_alias=AliasChoices("MCP_SERVERS", "LLM_MCP_SERVERS"),
    )

    # Backwards compatibility for single-server env variables
    server_command: Optional[str] = Field(
        None,
        alias="MCP_SERVER_COMMAND",
        validation_alias=AliasChoices("MCP_SERVER_COMMAND", "LLM_MCP_SERVER_COMMAND"),
    )
    server_transport: Optional[str] = Field(
        None,
        alias="MCP_SERVER_TRANSPORT",
        validation_alias=AliasChoices("MCP_SERVER_TRANSPORT", "LLM_MCP_SERVER_TRANSPORT"),
    )
    server_args: list[str] = Field(
        default_factory=list,
        alias="MCP_SERVER_ARGS",
        validation_alias=AliasChoices("MCP_SERVER_ARGS", "LLM_MCP_SERVER_ARGS"),
    )
    server_env: Optional[dict[str, str]] = Field(
        None,
        alias="MCP_SERVER_ENV",
        validation_alias=AliasChoices("MCP_SERVER_ENV", "LLM_MCP_SERVER_ENV"),
    )
    server_cwd: Optional[str] = Field(
        None,
        alias="MCP_SERVER_CWD",
        validation_alias=AliasChoices("MCP_SERVER_CWD", "LLM_MCP_SERVER_CWD"),
    )
    server_url: Optional[str] = Field(
        None,
        alias="MCP_SERVER_URL",
        validation_alias=AliasChoices("MCP_SERVER_URL", "LLM_MCP_SERVER_URL"),
    )
    server_headers: Optional[dict[str, str]] = Field(
        None,
        alias="MCP_SERVER_HEADERS",
        validation_alias=AliasChoices("MCP_SERVER_HEADERS", "LLM_MCP_SERVER_HEADERS"),
    )
    server_timeout: Optional[float] = Field(
        None,
        alias="MCP_SERVER_TIMEOUT",
        validation_alias=AliasChoices("MCP_SERVER_TIMEOUT", "LLM_MCP_SERVER_TIMEOUT"),
    )
    server_sse_read_timeout: Optional[float] = Field(
        None,
        alias="MCP_SERVER_SSE_READ_TIMEOUT",
        validation_alias=AliasChoices(
            "MCP_SERVER_SSE_READ_TIMEOUT",
            "LLM_MCP_SERVER_SSE_READ_TIMEOUT",
        ),
    )
    server_terminate_on_close: Optional[bool] = Field(
        None,
        alias="MCP_SERVER_TERMINATE_ON_CLOSE",
        validation_alias=AliasChoices(
            "MCP_SERVER_TERMINATE_ON_CLOSE",
            "LLM_MCP_SERVER_TERMINATE_ON_CLOSE",
        ),
    )

    @field_validator("transport", mode="before")
    @classmethod
    def normalise_transport(cls, value: Optional[str]) -> str:
        normalised = _normalise_transport(value) or "stdio"
        return normalised

    @field_validator("server_transport", mode="before")
    @classmethod
    def normalise_server_transport(cls, value: Optional[str]) -> Optional[str]:
        return _normalise_transport(value)

    @field_validator("trigger_keywords", mode="before")
    @classmethod
    def parse_trigger_keywords(cls, value: list[str] | str) -> list[str]:
        if isinstance(value, list):
            return value
        if not value:
            return []
        return [part.strip() for part in value.replace(",", " ").split() if part.strip()]

    @field_validator("server_args", mode="before")
    @classmethod
    def parse_server_args(cls, value: list[str] | str) -> list[str]:
        if isinstance(value, list):
            return value
        if not value:
            return []
        return shlex.split(value)

    @field_validator("server_headers", mode="before")
    @classmethod
    def parse_server_headers(
        cls, value: dict[str, str] | str | None
    ) -> dict[str, str] | None:
        return McpServerConfig.parse_headers(value)

    @field_validator("servers", mode="before")
    @classmethod
    def parse_servers(cls, value: list[dict[str, object]] | str | None) -> list[dict[str, object]]:
        if value in (None, ""):
            return []
        if isinstance(value, list):
            return value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("MCP_SERVERS must contain valid JSON") from exc
        if not isinstance(parsed, list):
            raise ValueError("MCP_SERVERS must decode to a list of server definitions")
        return parsed

    @model_validator(mode="after")
    def validate_mcp_config(self) -> "McpConfig":
        servers: list[McpServerConfig] = list(self.servers)

        if not servers and DEFAULT_MCP_SERVERS:
            servers = [McpServerConfig(**definition) for definition in DEFAULT_MCP_SERVERS]

        if any(
            value is not None
            for value in (
                self.server_command,
                self.server_url,
                self.server_transport,
                self.server_headers,
                self.server_timeout,
                self.server_sse_read_timeout,
                self.server_terminate_on_close,
            )
        ) or self.server_args or self.server_env or self.server_cwd:
            primary_kwargs: dict[str, object] = {
                "name": "default",
                "command": self.server_command,
                "args": self.server_args,
                "env": self.server_env,
                "cwd": self.server_cwd,
                "trigger_keywords": self.trigger_keywords,
            }
            if self.server_transport:
                primary_kwargs["transport"] = self.server_transport
            if self.server_url:
                primary_kwargs["url"] = self.server_url
            if self.server_headers is not None:
                primary_kwargs["headers"] = self.server_headers
            if self.server_timeout is not None:
                primary_kwargs["timeout"] = self.server_timeout
            if self.server_sse_read_timeout is not None:
                primary_kwargs["sse_read_timeout"] = self.server_sse_read_timeout
            if self.server_terminate_on_close is not None:
                primary_kwargs["terminate_on_close"] = self.server_terminate_on_close

            primary = McpServerConfig(**primary_kwargs)
            servers.insert(0, primary)

        if self.enabled and not servers:
            raise ValueError("At least one MCP server must be configured when MCP is enabled")

        resolved_servers: list[McpServerConfig] = []
        for server in servers:
            transport = server.transport or self.transport
            if transport not in SUPPORTED_TRANSPORTS:
                raise ValueError(
                    f"Unsupported MCP transport '{transport}' configured for server '{server.label}'"
                )
            if transport == "stdio" and not server.command:
                raise ValueError(
                    f"MCP server '{server.label}' requires a command when using stdio transport"
                )
            if transport != "stdio" and not server.url:
                raise ValueError(
                    f"MCP server '{server.label}' requires a URL when using {transport} transport"
                )
            # Replace the server transport with the resolved canonical value for consistency.
            if server.transport != transport:
                server = server.model_copy(update={"transport": transport})
            resolved_servers.append(server)

        if resolved_servers:
            primary = resolved_servers[0]
            object.__setattr__(self, "server_command", primary.command)
            object.__setattr__(self, "server_transport", primary.transport)
            object.__setattr__(self, "server_args", primary.args)
            object.__setattr__(self, "server_env", primary.env)
            object.__setattr__(self, "server_cwd", primary.cwd)
            object.__setattr__(self, "server_url", primary.url)
            object.__setattr__(self, "server_headers", primary.headers)
            object.__setattr__(self, "server_timeout", primary.timeout)
            object.__setattr__(self, "server_sse_read_timeout", primary.sse_read_timeout)
            object.__setattr__(
                self,
                "server_terminate_on_close",
                primary.terminate_on_close,
            )
            if not self.trigger_keywords:
                object.__setattr__(self, "trigger_keywords", primary.trigger_keywords)

        object.__setattr__(self, "servers", resolved_servers)
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def primary_server(self) -> Optional[McpServerConfig]:
        """Return the first configured server, if any."""

        return self.servers[0] if self.servers else None



@lru_cache()
def get_mcp_config() -> McpConfig:
    """Return a cached MCP configuration instance."""

    return McpConfig()
