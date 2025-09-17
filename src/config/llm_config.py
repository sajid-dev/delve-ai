"""LLM configuration definitions.

This module defines a `LlmConfig` class representing the unified
configuration required to initialise a ChatGPT‑compatible model using
LangChain's `ChatOpenAI` integration.  The configuration no longer
distinguishes between OpenAI and other providers; instead a single
set of fields covers API credentials, endpoint configuration and
tuning parameters.  The `get_llm_config` helper caches an instance so
it can be re‑used throughout the application.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class LlmConfig(BaseSettings):
    """Unified configuration for interacting with a language model API.

    This settings class defines a single set of fields for
    communicating with a ChatGPT‑compatible endpoint.  It no longer
    distinguishes between OpenAI and other providers.  Instead,
    applications should set `LLM_API_KEY`, `LLM_BASE_URL` (optional)
    and `LLM_MODEL` in the environment.  Additional parameters such as
    temperature, max tokens and timeout control the behaviour of the
    language model.
    """

    # API key used by the LLM provider (e.g., OpenAI, LLaMA)
    api_key: str
    # Base URL for the API.  If left blank, the default OpenAI endpoint will be used.
    base_url: str | None = None
    # Model name to use for chat (e.g., gpt-3.5-turbo or llama-2-7b-chat)
    model: str = "gpt-3.5-turbo"
    # Temperature controlling randomness of responses
    temperature: float = 0.7
    # Maximum number of tokens to generate
    max_tokens: int = 2048
    # Timeout in seconds for API calls
    timeout: int = 30

    # Path to the `.env` file; Pydantic will read variables from this file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_llm_config() -> LlmConfig:
    """Return a cached LlmConfig instance.

    The result is cached so subsequent calls reuse the same configuration.
    """
    return LlmConfig()  # type: ignore[arg-type]