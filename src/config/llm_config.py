from functools import lru_cache
import shlex
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class LlmConfig(BaseSettings):
    """Configuration settings for language model integrations."""

    api_key: str = Field(..., alias="LLM_API_KEY")
    base_url: Optional[str] = Field(default=None, alias="LLM_BASE_URL")
    model: str = Field("gpt-3.5-turbo", alias="LLM_MODEL")
    temperature: float = Field(0.7, alias="LLM_TEMPERATURE")
    max_tokens: Optional[int] = Field(None, alias="LLM_MAX_TOKENS")
    timeout: int = Field(30, alias="LLM_TIMEOUT")
    mcp_enabled: bool = Field(False, alias="LLM_MCP_ENABLED")
    mcp_transport: Literal["stdio"] = Field("stdio", alias="LLM_MCP_TRANSPORT")
    mcp_server_command: Optional[str] = Field(None, alias="LLM_MCP_SERVER_COMMAND")
    mcp_server_args: list[str] = Field(default_factory=list, alias="LLM_MCP_SERVER_ARGS")
    mcp_server_env: Optional[dict[str, str]] = Field(None, alias="LLM_MCP_SERVER_ENV")
    mcp_server_cwd: Optional[str] = Field(None, alias="LLM_MCP_SERVER_CWD")
    mcp_trigger_keywords: list[str] = Field(default_factory=list, alias="LLM_MCP_TRIGGER_KEYWORDS")

    @field_validator("temperature")
    def validate_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("LLM_TEMPERATURE must be between 0.0 and 1.0")
        return value

    @field_validator("timeout")
    def validate_timeout(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("LLM_TIMEOUT must be positive")
        return value

    @field_validator("max_tokens")
    def validate_max_tokens(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("LLM_MAX_TOKENS must be positive")
        return value

    @field_validator("mcp_server_args", mode="before")
    def parse_mcp_args(cls, value: list[str] | str) -> list[str]:
        if isinstance(value, list):
            return value
        if not value:
            return []
        return shlex.split(value)

    @field_validator("mcp_trigger_keywords", mode="before")
    def parse_trigger_keywords(cls, value: list[str] | str) -> list[str]:
        if isinstance(value, list):
            return value
        if not value:
            return []
        return [part.strip() for part in value.replace(",", " ").split() if part.strip()]

    @model_validator(mode="after")
    def validate_mcp_config(self) -> "LlmConfig":
        if self.mcp_enabled:
            if self.mcp_transport != "stdio":
                raise ValueError("Only 'stdio' MCP transport is currently supported")
            if not self.mcp_server_command:
                raise ValueError("LLM_MCP_SERVER_COMMAND must be provided when MCP is enabled")
        return self

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


@lru_cache()
def get_llm_config() -> LlmConfig:
    """Return a cached language model configuration."""

    return LlmConfig()


# Global LLM configuration instance for convenience
llm_config = get_llm_config()
