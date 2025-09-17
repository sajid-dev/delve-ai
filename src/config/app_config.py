"""Application configuration definitions.

This module defines simple Pydantic models for high‑level
application settings.  They are separated from environment
loading logic so that values can be composed from multiple
sources if desired.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Application-level configuration settings.

    This class centralises settings that apply to the overall
    application rather than the LLM.  These include environment
    settings, host/port configuration, memory backend selection and
    logging.  Values are loaded from environment variables or a
    `.env` file and can be overridden via environment.
    """

    # Metadata
    app_name: str = "LLM Chat App"
    api_prefix: str = ""

    # Application environment settings
    app_env: str = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8501

    # Memory backend configuration
    memory_type: str = "in_memory"
    redis_url: str | None = None

    # Logging configuration
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_app_config() -> AppConfig:
    """Return a cached AppConfig instance.

    The result is cached to avoid re-reading environment variables on each call.
    """
    return AppConfig()  # type: ignore[arg-type]