"""Application settings sourced from environment variables.

All configuration is read through ``pydantic-settings`` so that a single
instantiation (``get_settings``) is enough for the whole app and can be
overridden in tests via environment variables or the ``lru_cache`` reset.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Model store ---
    model_store: str = Field(default="local")
    model_dir: str = Field(default="models")
    s3_bucket: str = Field(default="ml-serving-drift-platform-models")
    s3_prefix: str = Field(default="models")
    model_version: str = Field(default="latest")
    aws_region: str = Field(default="us-east-1")

    # --- API ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    batch_max_records: int = Field(default=1000)

    # --- Database ---
    db_enabled: bool = Field(default=True)
    database_url: str | None = Field(default=None)
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="mlmonitor")
    postgres_user: str = Field(default="mluser")
    postgres_password: str = Field(default="changeme")

    # --- Drift ---
    psi_threshold: float = Field(default=0.2)
    ks_alpha: float = Field(default=0.05)
    drift_feature_fraction: float = Field(default=0.2)
    drift_bin_count: int = Field(default=10)
    drift_epsilon: float = Field(default=1e-6)

    @field_validator("model_store")
    @classmethod
    def _validate_store(cls, v: str) -> str:
        v = v.lower()
        if v not in {"local", "s3"}:
            raise ValueError("MODEL_STORE must be 'local' or 's3'")
        return v

    @property
    def sqlalchemy_url(self) -> str:
        """Resolved SQLAlchemy DSN for PostgreSQL."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
