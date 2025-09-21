from __future__ import annotations

from functools import lru_cache
import json
import shlex
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .mcp_servers import DEFAULT_MCP_SERVERS

load_dotenv()


class McpServerConfig(BaseModel):
    """Definition for a single MCP server instance."""

    name: Optional[str] = None
    command: str
    args: list[str] = Field(default_factory=list)
    env: Optional[dict[str, str]] = None
    cwd: Optional[str] = None
    trigger_keywords: list[str] = Field(default_factory=list)

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

    @property
    def label(self) -> str:
        """Return a display name for logging and context output."""

        return self.name or self.command


class McpConfig(BaseSettings):
    """Settings controlling MCP tool discovery and invocation."""

    enabled: bool = Field(
        False,
        alias="MCP_ENABLED",
        validation_alias=AliasChoices("MCP_ENABLED", "LLM_MCP_ENABLED"),
    )
    transport: Literal["stdio"] = Field(
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

        if self.server_command:
            primary = McpServerConfig(
                name="default",
                command=self.server_command,
                args=self.server_args,
                env=self.server_env,
                cwd=self.server_cwd,
                trigger_keywords=self.trigger_keywords,
            )
            servers.insert(0, primary)

        if self.enabled:
            if self.transport != "stdio":
                raise ValueError("Only 'stdio' MCP transport is currently supported")
            if not servers:
                raise ValueError("At least one MCP server must be configured when MCP is enabled")

        if servers:
            primary = servers[0]
            object.__setattr__(self, "server_command", primary.command)
            object.__setattr__(self, "server_args", primary.args)
            object.__setattr__(self, "server_env", primary.env)
            object.__setattr__(self, "server_cwd", primary.cwd)
            if not self.trigger_keywords:
                object.__setattr__(self, "trigger_keywords", primary.trigger_keywords)

        object.__setattr__(self, "servers", servers)
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
