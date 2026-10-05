from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Core Application
    PROJECT_NAME: str = Field(default="ModelOps Sentinel")
    VERSION: str = Field(default="0.1.0")
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(default="development")
    DEBUG: bool = Field(default=False)
    API_V1_PREFIX: str = Field(default="/api/v1")
    SECRET_KEY: str = Field(default="dev_secret_key_change_in_production")

    # Server Binding
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)

    # PostgreSQL Database
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="modelops_db")
    POSTGRES_USER: str = Field(default="modelops_user")
    POSTGRES_PASSWORD: str = Field(default="modelops_password")
    POSTGRES_POOL_SIZE: int = Field(default=10)
    POSTGRES_MAX_OVERFLOW: int = Field(default=5)

    # Redis Cache & Task Broker
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0)
    REDIS_PASSWORD: str | None = Field(default=None)

    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    LOG_FORMAT: Literal["text", "json"] = Field(default="text")

    # Local Artifact Directory
    ARTIFACT_DIR: str = Field(default="./artifacts")

    @property
    def async_database_url(self) -> str:
        """Construct async PostgreSQL connection URL for asyncpg."""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def redis_url(self) -> str:
        """Construct Redis connection URL."""
        auth_part = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance (singleton pattern)."""
    return Settings()
