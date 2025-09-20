from __future__ import annotations

from functools import lru_cache
import json
import shlex
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    @field_validator("trigger_keywords", mode="before")
    @classmethod
    def parse_trigger_keywords(cls, value: list[str] | str) -> list[str]:
        if isinstance(value, list):
            return value
        if not value:
            return []
        return [part.strip() for part in value.replace(",", " ").split() if part.strip()]

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

        if self.enabled:
            if self.transport != "stdio":
                raise ValueError("Only 'stdio' MCP transport is currently supported")
            if not servers:
                raise ValueError("At least one MCP server must be configured when MCP is enabled")

        if servers:
            primary = servers[0]
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



@lru_cache()
def get_mcp_config() -> McpConfig:
    """Return a cached MCP configuration instance."""

    return McpConfig()
