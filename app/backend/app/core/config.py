from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = Field(alias="APP_ENV")
    app_profile: Literal["gtx1650", "rtx3060"] = Field(
        default="gtx1650",
        alias="APP_PROFILE",
    )
    database_url: str = Field(alias="DATABASE_URL", min_length=1)
    database_password: SecretStr | None = Field(default=None, alias="DATABASE_PASSWORD")
    jwt_secret: SecretStr = Field(alias="JWT_SECRET")
    jwt_algorithm: Literal["HS256"] = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=480,
        alias="ACCESS_TOKEN_EXPIRE_MINUTES",
        gt=0,
    )
    upload_root: Path = Field(alias="UPLOAD_ROOT")
    evidence_root: Path = Field(alias="EVIDENCE_ROOT")
    model_root: Path = Field(alias="MODEL_ROOT")
    config_root: Path = Field(default=Path("/app/configs"), alias="CONFIG_ROOT")
    max_upload_bytes: int = Field(
        default=4 * 1024 * 1024 * 1024,
        alias="MAX_UPLOAD_BYTES",
        gt=0,
    )
    ffprobe_timeout_seconds: int = Field(
        default=30,
        alias="FFPROBE_TIMEOUT_SECONDS",
        gt=0,
        le=300,
    )

    admin_username: str | None = Field(default=None, alias="ADMIN_USERNAME")
    admin_password: SecretStr | None = Field(default=None, alias="ADMIN_PASSWORD")
    admin_full_name: str | None = Field(default=None, alias="ADMIN_FULL_NAME")
    # Built frontend (frontend/dist) served by the backend: one process, no nginx needed.
    frontend_dist: Path | None = Field(default=None, alias="FRONTEND_DIST")
    # Auth cookies (media, WebSocket) are Secure in production. Set false only to use a production
    # server over plain HTTP on a trusted LAN.
    cookie_secure: bool | None = Field(default=None, alias="COOKIE_SECURE")

    @property
    def secure_cookies(self) -> bool:
        return self.app_env == "production" if self.cookie_secure is None else self.cookie_secure

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        if not value.startswith("postgresql+psycopg://"):
            raise ValueError("DATABASE_URL must use postgresql+psycopg")
        return value

    @field_validator("admin_username")
    @classmethod
    def normalize_optional_username(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        return normalized or None

    @field_validator("admin_full_name")
    @classmethod
    def normalize_optional_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("admin_password", mode="before")
    @classmethod
    def normalize_optional_password(cls, value: object) -> object | None:
        if isinstance(value, str) and not value:
            return None
        return value

    @model_validator(mode="after")
    def validate_secrets(self) -> "Settings":
        secret = self.jwt_secret.get_secret_value()
        if len(secret) < 32:
            raise ValueError("JWT_SECRET must contain at least 32 characters")

        if self.app_env == "production":
            lowered_secret = secret.lower()
            if "change-me" in lowered_secret or "changeme" in lowered_secret:
                raise ValueError("JWT_SECRET cannot be a placeholder in production")

            if self.admin_password is not None:
                password = self.admin_password.get_secret_value()
                lowered_password = password.lower()
                invalid_password = (
                    len(password) < 12
                    or "change-me" in lowered_password
                    or "changeme" in lowered_password
                )
                if invalid_password:
                    raise ValueError("ADMIN_PASSWORD is not acceptable for production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
