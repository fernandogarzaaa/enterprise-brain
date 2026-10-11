"""Shared fixtures.

Every test gets an isolated database and data directory. By default that is a fresh
SQLite file; set ``TEST_DATABASE_URL`` to a PostgreSQL URL to run the same suite
against Postgres (+pgvector), which CI does.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from enterprise_brain.config import Settings
from enterprise_brain.deps import AppContext
from enterprise_brain.main import create_app
from enterprise_brain.vectorstore import reset_pgvector_cache

ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "Adm1n-Passw0rd!"
TEST_SECRET = "test-secret-key-" + "x" * 40
PG_URL = os.environ.get("TEST_DATABASE_URL", "")

SettingsFactory = Callable[..., Settings]


def reset_postgres(url: str) -> None:
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    reset_pgvector_cache()


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    db_url = PG_URL or f"sqlite:///{tmp_path / 'test.db'}"
    values: dict[str, Any] = {
        "ENVIRONMENT": "test",
        "DATABASE_URL": db_url,
        "DATA_DIR": tmp_path / "data",
        "SECRET_KEY": TEST_SECRET,
        "ADMIN_EMAIL": ADMIN_EMAIL,
        "ADMIN_PASSWORD": ADMIN_PASSWORD,
        "BCRYPT_ROUNDS": 4,
        "LLM_PROVIDER": "none",
        "LOG_FORMAT": "text",
        "LOG_LEVEL": "WARNING",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture(autouse=True)
def _fresh_postgres() -> None:
    if PG_URL:
        reset_postgres(PG_URL)


@pytest.fixture
def settings_factory(tmp_path: Path) -> SettingsFactory:
    def factory(**overrides: Any) -> Settings:
        return make_settings(tmp_path, **overrides)

    return factory


@pytest.fixture
def settings(settings_factory: SettingsFactory) -> Settings:
    return settings_factory()


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def ctx(app: FastAPI) -> AppContext:
    c: AppContext = app.state.ctx
    return c


@pytest.fixture
def db(ctx: AppContext) -> Iterator[Session]:
    session = ctx.sessions()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def login(client: TestClient, email: str = ADMIN_EMAIL, password: str = ADMIN_PASSWORD) -> dict[str, str]:
    r = client.post("/api/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    client.cookies.clear()  # authenticate with the bearer token explicitly
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def admin_headers(client: TestClient) -> dict[str, str]:
    return login(client)


@pytest.fixture
def member(client: TestClient, admin_headers: dict[str, str]) -> dict[str, Any]:
    r = client.post("/api/team/invite", headers=admin_headers, json={"email": "member@example.com", "name": "Mem"})
    assert r.status_code == 201, r.text
    data: dict[str, Any] = r.json()
    data["headers"] = login(client, "member@example.com", data["temporary_password"])
    return data


@pytest.fixture
def demo_app(settings_factory: SettingsFactory) -> FastAPI:
    return create_app(settings_factory(DEMO_DATA=True))


@pytest.fixture
def demo_client(demo_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(demo_app) as c:
        yield c


@pytest.fixture
def demo_headers(demo_client: TestClient) -> dict[str, str]:
    return login(demo_client)
