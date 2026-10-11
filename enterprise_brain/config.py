"""Runtime configuration, read from environment variables (and an optional .env file).

Every setting has a safe development default. When ``ENVIRONMENT=production`` the
settings are validated strictly at startup: a weak or missing ``SECRET_KEY`` or a
default admin password refuses to boot rather than running insecurely.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Known placeholder values that must never be used in production.
INSECURE_SECRETS = frozenset({"", "change-me", "change-me-in-production", "secret", "changeme"})
INSECURE_PASSWORDS = frozenset({"ChangeMe123!", "changeme", "password", "admin", "admin123"})
MIN_SECRET_KEY_LENGTH = 32
MIN_PROD_ADMIN_PASSWORD_LENGTH = 12


class ConfigError(ValueError):
    """Raised when the configuration is unsafe or inconsistent."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=True)

    # --- application ---
    ENVIRONMENT: Literal["development", "test", "production"] = "development"
    APP_NAME: str = "Enterprise Brain"
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8002, ge=1, le=65535)
    WORKERS: int = Field(default=1, ge=1, le=64)

    # --- storage ---
    DATABASE_URL: str = "sqlite:///./data/enterprise_brain.db"
    DATA_DIR: Path = Path("data")
    STAGING_DIR: Path | None = None  # defaults to DATA_DIR/staging
    AUTO_MIGRATE: bool = True

    # --- security ---
    SECRET_KEY: SecretStr = SecretStr("")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=720, ge=5, le=60 * 24 * 30)
    COOKIE_SECURE: bool | None = None  # None = secure cookies in production only
    BCRYPT_ROUNDS: int = Field(default=12, ge=4, le=16)
    LOGIN_RATE_LIMIT: int = Field(default=10, ge=1)
    LOGIN_RATE_WINDOW_SECONDS: int = Field(default=300, ge=1)
    CORS_ORIGINS: str = ""  # comma-separated list; empty disables CORS
    MIN_PASSWORD_LENGTH: int = Field(default=10, ge=8, le=128)

    # --- bootstrap ---
    ADMIN_EMAIL: str = ""
    ADMIN_PASSWORD: SecretStr = SecretStr("")
    DEMO_DATA: bool = False

    # --- answering / LLM ---
    LLM_PROVIDER: Literal["auto", "none", "mock", "openai"] = "auto"
    LLM_BASE_URL: str = ""
    LLM_API_KEY: SecretStr = SecretStr("")
    LLM_CHAT_MODEL: str = "llama-3.1-8b-instant"
    LLM_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0, le=600)
    LLM_MAX_RETRIES: int = Field(default=2, ge=0, le=10)
    LLM_TEMPERATURE: float = Field(default=0.1, ge=0, le=2)
    LLM_MAX_TOKENS: int = Field(default=800, ge=16, le=32000)

    # --- embeddings ---
    EMBEDDINGS: Literal["hashed", "openai", "st"] = "hashed"
    HASHED_EMBED_DIM: int = Field(default=256, ge=16, le=8192)
    EMBED_BASE_URL: str = ""  # defaults to LLM_BASE_URL
    EMBED_API_KEY: SecretStr = SecretStr("")  # defaults to LLM_API_KEY
    EMBED_MODEL: str = "text-embedding-3-small"
    ST_MODEL: str = "all-MiniLM-L6-v2"

    # --- retrieval / ingest limits ---
    CHUNK_SIZE: int = Field(default=600, ge=100, le=8000)
    CHUNK_OVERLAP: int = Field(default=120, ge=0, le=2000)
    TOP_K: int = Field(default=5, ge=1, le=50)
    MAX_UPLOAD_MB: int = Field(default=25, ge=1, le=1024)
    MAX_QUESTION_CHARS: int = Field(default=2000, ge=10, le=20000)

    # --- background sync ---
    SYNC_INTERVAL_MINUTES: int = Field(default=0, ge=0)

    # --- logging ---
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_FORMAT: Literal["auto", "json", "text"] = "auto"

    @field_validator("DATABASE_URL")
    @classmethod
    def _normalise_db_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("DATABASE_URL must not be empty")
        # Heroku/Render style URLs use the legacy scheme.
        if v.startswith("postgres://"):
            v = "postgresql+psycopg2://" + v[len("postgres://") :]
        return v

    @field_validator("LLM_BASE_URL", "EMBED_BASE_URL")
    @classmethod
    def _strip_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.LLM_PROVIDER == "openai" and not self.LLM_BASE_URL:
            raise ValueError("LLM_PROVIDER=openai requires LLM_BASE_URL")
        if self.EMBEDDINGS == "openai" and not self.embed_base_url:
            raise ValueError("EMBEDDINGS=openai requires EMBED_BASE_URL or LLM_BASE_URL")
        if self.ENVIRONMENT == "production":
            self._check_production()
        elif self.SECRET_KEY.get_secret_value() in INSECURE_SECRETS:
            # Development convenience: an ephemeral key (sessions reset on restart).
            object.__setattr__(self, "SECRET_KEY", SecretStr(secrets.token_urlsafe(48)))
        return self

    def _check_production(self) -> None:
        key = self.SECRET_KEY.get_secret_value()
        if key in INSECURE_SECRETS or len(key) < MIN_SECRET_KEY_LENGTH:
            raise ConfigError(
                f"SECRET_KEY must be set to a random value of at least {MIN_SECRET_KEY_LENGTH} "
                "characters in production (python -c 'import secrets; print(secrets.token_urlsafe(48))')"
            )
        pw = self.ADMIN_PASSWORD.get_secret_value()
        if pw and (pw in INSECURE_PASSWORDS or len(pw) < MIN_PROD_ADMIN_PASSWORD_LENGTH):
            raise ConfigError(
                f"ADMIN_PASSWORD must be at least {MIN_PROD_ADMIN_PASSWORD_LENGTH} characters and "
                "not a known default in production"
            )

    # --- derived values ---
    @property
    def staging_dir(self) -> Path:
        return self.STAGING_DIR if self.STAGING_DIR is not None else self.DATA_DIR / "staging"

    @property
    def upload_dir(self) -> Path:
        return self.DATA_DIR / "uploads"

    @property
    def cookie_secure(self) -> bool:
        if self.COOKIE_SECURE is None:
            return self.ENVIRONMENT == "production"
        return self.COOKIE_SECURE

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def llm_mode(self) -> Literal["none", "mock", "openai"]:
        if self.LLM_PROVIDER == "auto":
            return "openai" if self.LLM_BASE_URL else "none"
        return self.LLM_PROVIDER

    @property
    def embed_base_url(self) -> str:
        return self.EMBED_BASE_URL or self.LLM_BASE_URL

    @property
    def embed_api_key(self) -> str:
        return self.EMBED_API_KEY.get_secret_value() or self.LLM_API_KEY.get_secret_value()

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    @property
    def log_format(self) -> Literal["json", "text"]:
        if self.LOG_FORMAT == "auto":
            return "json" if self.ENVIRONMENT == "production" else "text"
        return self.LOG_FORMAT

    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.startswith("sqlite")

    @property
    def is_postgres(self) -> bool:
        return self.DATABASE_URL.startswith("postgresql")

    def public_summary(self) -> dict[str, str]:
        """Non-secret configuration shown to admins on the Settings page."""
        db = self.DATABASE_URL
        if "@" in db:  # hide credentials
            scheme = db.split("://", 1)[0]
            db = f"{scheme}://***@{db.rsplit('@', 1)[1]}"
        return {
            "Environment": self.ENVIRONMENT,
            "Application name": self.APP_NAME,
            "Answer mode": {
                "none": "extractive (local, no LLM)",
                "mock": "mock LLM (deterministic, offline)",
                "openai": f"LLM via OpenAI-compatible API ({self.LLM_CHAT_MODEL})",
            }[self.llm_mode],
            "Embeddings": {
                "hashed": f"hashed ({self.HASHED_EMBED_DIM}-dim, local)",
                "openai": f"OpenAI-compatible ({self.EMBED_MODEL})",
                "st": f"sentence-transformers ({self.ST_MODEL})",
            }[self.EMBEDDINGS],
            "Database": db,
            "Upload directory": str(self.upload_dir),
            "Connector staging directory": str(self.staging_dir),
            "Background sync": (f"every {self.SYNC_INTERVAL_MINUTES} min" if self.SYNC_INTERVAL_MINUTES else "off"),
            "Max upload size": f"{self.MAX_UPLOAD_MB} MB",
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
