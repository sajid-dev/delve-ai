"""Application settings loaded from environment variables.

This module defines a singleton `Settings` class that reads
configuration values from a `.env` file or the host environment.
Using Pydantic's `BaseSettings` makes environment management
typesafe and allows default values.  This file was moved into the
`src/config` package so that all configuration lives under the `src`
directory, as requested.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration for the application.

    This settings class centralises all configuration values loaded from
    environment variables.  Default values are provided for optional
    settings so the application can start even if some variables are
    absent.  Both OpenAI and LLaMA API credentials are supported.
    """

    # ---------------------------------------------------------------------
    # OpenAI configuration
    openai_api_key: str
    openai_model_name: str = "gpt-3.5-turbo"
    openai_temperature: float = 0.7

    # ---------------------------------------------------------------------
    # LLaMA API configuration (compatible with ChatOpenAI via base_url)
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    llm_timeout: int = 30

    # ---------------------------------------------------------------------
    # Application settings
    app_env: str = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8501

    # ---------------------------------------------------------------------
    # Memory configuration
    memory_type: str = "in_memory"
    redis_url: str | None = None

    # ---------------------------------------------------------------------
    # Logging configuration
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    # Path to the `.env` file; Pydantic will read variables from this file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Pydantic will read environment variables on instantiation; the result
    is cached so subsequent calls return the same object.  This helper
    ensures settings are only loaded once per process.
    """
    return Settings()  # type: ignore[arg-type]
