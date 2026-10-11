from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from enterprise_brain.config import Settings, get_settings

STRONG = "k" * 48


def make(**kw: Any) -> Settings:
    return Settings(_env_file=None, **kw)


def test_defaults() -> None:
    s = make()
    assert s.ENVIRONMENT == "development"
    assert s.PORT == 8002
    assert s.DATABASE_URL == "sqlite:///./data/enterprise_brain.db"
    assert s.llm_mode == "none"
    assert s.EMBEDDINGS == "hashed"
    assert s.cookie_secure is False
    assert s.log_format == "text"
    assert s.staging_dir == Path("data/staging")
    assert s.upload_dir == Path("data/uploads")
    assert s.max_upload_bytes == 25 * 1024 * 1024
    assert s.cors_origins == []
    assert s.is_sqlite and not s.is_postgres


def test_dev_generates_ephemeral_secret() -> None:
    a, b = make(), make()
    assert len(a.SECRET_KEY.get_secret_value()) >= 32
    assert a.SECRET_KEY.get_secret_value() != b.SECRET_KEY.get_secret_value()
    assert make(SECRET_KEY="change-me").SECRET_KEY.get_secret_value() != "change-me"
    assert make(SECRET_KEY="mine").SECRET_KEY.get_secret_value() == "mine"


@pytest.mark.parametrize("key", ["", "change-me-in-production", "short", "x" * 31])
def test_production_rejects_weak_secret(key: str) -> None:
    with pytest.raises(ValidationError, match="SECRET_KEY"):
        make(ENVIRONMENT="production", SECRET_KEY=key)


def test_production_accepts_strong_secret() -> None:
    s = make(ENVIRONMENT="production", SECRET_KEY="x" * 32)
    assert s.cookie_secure is True
    assert s.log_format == "json"
    assert s.SECRET_KEY.get_secret_value() == "x" * 32


@pytest.mark.parametrize("pw", ["ChangeMe123!", "short1!", "password", "x" * 11])
def test_production_rejects_weak_admin_password(pw: str) -> None:
    with pytest.raises(ValidationError, match="ADMIN_PASSWORD"):
        make(ENVIRONMENT="production", SECRET_KEY=STRONG, ADMIN_PASSWORD=pw)


def test_production_admin_password_ok() -> None:
    assert make(ENVIRONMENT="production", SECRET_KEY=STRONG, ADMIN_PASSWORD="a-very-long-pw-1")
    assert make(ENVIRONMENT="production", SECRET_KEY=STRONG, ADMIN_PASSWORD="").ADMIN_EMAIL == ""
    assert make(ENVIRONMENT="production", SECRET_KEY=STRONG, ADMIN_PASSWORD="x" * 12)


def test_dev_allows_default_admin_password() -> None:
    assert make(ADMIN_PASSWORD="ChangeMe123!").ADMIN_PASSWORD.get_secret_value() == "ChangeMe123!"


def test_postgres_url_normalised() -> None:
    s = make(DATABASE_URL="  postgres://u:p@h:5432/db ")
    assert s.DATABASE_URL == "postgresql+psycopg2://u:p@h:5432/db"
    assert s.is_postgres and not s.is_sqlite
    assert make(DATABASE_URL="postgresql://u@h/db").DATABASE_URL == "postgresql://u@h/db"


def test_empty_db_url_rejected() -> None:
    with pytest.raises(ValidationError, match="DATABASE_URL"):
        make(DATABASE_URL="  ")


def test_llm_urls() -> None:
    s = make(LLM_BASE_URL=" https://api.groq.com/openai/v1/ ")
    assert s.LLM_BASE_URL == "https://api.groq.com/openai/v1"
    assert s.llm_mode == "openai"
    assert s.embed_base_url == "https://api.groq.com/openai/v1"
    with pytest.raises(ValidationError, match="http"):
        make(LLM_BASE_URL="ftp://x")
    with pytest.raises(ValidationError, match="http"):
        make(EMBED_BASE_URL="localhost:11434")
    assert make(LLM_BASE_URL="http://localhost:11434/v1").LLM_BASE_URL == "http://localhost:11434/v1"


def test_llm_mode_explicit() -> None:
    assert make(LLM_PROVIDER="mock").llm_mode == "mock"
    assert make(LLM_PROVIDER="none", LLM_BASE_URL="http://x").llm_mode == "none"
    assert make(LLM_PROVIDER="openai", LLM_BASE_URL="http://x").llm_mode == "openai"
    with pytest.raises(ValidationError, match="LLM_BASE_URL"):
        make(LLM_PROVIDER="openai")


def test_embedding_requires_url() -> None:
    with pytest.raises(ValidationError, match="EMBED_BASE_URL"):
        make(EMBEDDINGS="openai")
    assert make(EMBEDDINGS="openai", EMBED_BASE_URL="http://e").embed_base_url == "http://e"


def test_embed_key_fallback() -> None:
    assert make(LLM_API_KEY="a").embed_api_key == "a"
    assert make(LLM_API_KEY="a", EMBED_API_KEY="b").embed_api_key == "b"
    assert make().embed_api_key == ""


def test_chunk_overlap_must_be_smaller() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        make(CHUNK_SIZE=200, CHUNK_OVERLAP=200)
    assert make(CHUNK_SIZE=200, CHUNK_OVERLAP=199).CHUNK_OVERLAP == 199


def test_derived_paths_and_cookie_override() -> None:
    s = make(DATA_DIR="/srv/d", STAGING_DIR="/mnt/exports", COOKIE_SECURE=True, MAX_UPLOAD_MB=2)
    assert s.staging_dir == Path("/mnt/exports")
    assert s.upload_dir == Path("/srv/d/uploads")
    assert s.cookie_secure is True
    assert s.max_upload_bytes == 2 * 1024 * 1024
    assert make(ENVIRONMENT="production", SECRET_KEY=STRONG, COOKIE_SECURE=False).cookie_secure is False


def test_cors_and_log_format() -> None:
    assert make(CORS_ORIGINS=" https://a.com, ,https://b.com ").cors_origins == ["https://a.com", "https://b.com"]
    assert make(LOG_FORMAT="json").log_format == "json"
    assert make(ENVIRONMENT="production", SECRET_KEY=STRONG, LOG_FORMAT="text").log_format == "text"


def test_public_summary_hides_credentials() -> None:
    s = make(DATABASE_URL="postgresql://user:hunter2@db:5432/eb", LLM_BASE_URL="http://x", LLM_API_KEY="sk-secret")
    summary = s.public_summary()
    assert summary["Database"] == "postgresql://***@db:5432/eb"
    assert "hunter2" not in str(summary) and "sk-secret" not in str(summary)
    assert summary["Answer mode"] == "LLM via OpenAI-compatible API (llama-3.1-8b-instant)"
    assert summary["Background sync"] == "off"
    assert summary["Max upload size"] == "25 MB"
    assert summary["Environment"] == "development"
    assert make(SYNC_INTERVAL_MINUTES=15).public_summary()["Background sync"] == "every 15 min"
    assert make().public_summary()["Database"] == "sqlite:///./data/enterprise_brain.db"
    assert make().public_summary()["Answer mode"] == "extractive (local, no LLM)"
    assert make(LLM_PROVIDER="mock").public_summary()["Answer mode"] == "mock LLM (deterministic, offline)"
    assert make().public_summary()["Embeddings"] == "hashed (256-dim, local)"
    emb = make(EMBEDDINGS="openai", EMBED_BASE_URL="http://e").public_summary()["Embeddings"]
    assert emb == "OpenAI-compatible (text-embedding-3-small)"


def test_get_settings_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_NAME", "Acme Brain")
    get_settings.cache_clear()
    try:
        assert get_settings().APP_NAME == "Acme Brain"
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
