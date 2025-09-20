from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .mcp_config import McpConfig, get_mcp_config

load_dotenv()


class LlmConfig(BaseSettings):
    """Configuration settings for language model integrations."""

    api_key: str = Field(..., alias="LLM_API_KEY")
    base_url: Optional[str] = Field(default=None, alias="LLM_BASE_URL")
    model: str = Field("gpt-3.5-turbo", alias="LLM_MODEL")
    temperature: float = Field(0.7, alias="LLM_TEMPERATURE")
    max_tokens: Optional[int] = Field(None, alias="LLM_MAX_TOKENS")
    timeout: int = Field(30, alias="LLM_TIMEOUT")
    mcp: McpConfig = Field(default_factory=get_mcp_config)

    def _update_mcp_fields(self, **updates: object) -> None:
        data = self.mcp.model_dump()
        data.update(updates)
        self.mcp = McpConfig(**data)

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

    @property
    def mcp_enabled(self) -> bool:
        return self.mcp.enabled

    @mcp_enabled.setter
    def mcp_enabled(self, value: bool) -> None:
        self._update_mcp_fields(enabled=value)

    @property
    def mcp_transport(self) -> str:
        return self.mcp.transport

    @mcp_transport.setter
    def mcp_transport(self, value: str) -> None:
        self._update_mcp_fields(transport=value)

    @property
    def mcp_trigger_keywords(self) -> list[str]:
        return self.mcp.trigger_keywords

    @mcp_trigger_keywords.setter
    def mcp_trigger_keywords(self, value: list[str]) -> None:
        self._update_mcp_fields(trigger_keywords=value)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


@lru_cache()
def get_llm_config() -> LlmConfig:
    """Return a cached language model configuration."""

    return LlmConfig()


# Global LLM configuration instance for convenience
llm_config = get_llm_config()
