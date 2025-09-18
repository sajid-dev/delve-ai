from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class AppConfig(BaseSettings):
    """Application configuration settings."""

    # Environment
    app_env: str = Field("development")
    app_debug: bool = Field(False)
    app_host: str = Field("0.0.0.0")
    app_port: int = Field(8501)

    # Logging
    log_level: str = Field("INFO")
    log_file: Optional[str] = Field(None)

    # Memory
    memory_type: str = Field("in_memory")
    redis_url: Optional[str] = Field(None)

    @field_validator("app_env")
    def validate_app_env(cls, value: str) -> str:
        if value not in ["development", "staging", "production"]:
            raise ValueError("APP_ENV must be development, staging, or production")
        return value

    @field_validator("memory_type")
    def validate_memory_type(cls, value: str) -> str:
        if value not in ["in_memory", "redis"]:
            raise ValueError("MEMORY_TYPE must be in_memory or redis")
        return value

    @field_validator("log_level")
    def validate_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("LOG_LEVEL must be a valid Loguru level")
        return level

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Global configuration instance
app_config = AppConfig()


@lru_cache()
def get_app_config() -> AppConfig:
    """Return a cached application configuration instance."""

    return AppConfig()
