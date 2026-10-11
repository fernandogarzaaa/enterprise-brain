"""Schema migrations: fresh install, downgrade, and in-place upgrade of a 0.x beta database."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from alembic import command
from fastapi.testclient import TestClient
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    inspect,
    text,
)

from enterprise_brain.db import alembic_config, run_migrations
from enterprise_brain.llm import HashedEmbedder
from enterprise_brain.main import create_app
from enterprise_brain.security import hash_password
from tests.conftest import SettingsFactory

TABLES = {"users", "api_keys", "sources", "documents", "chunks", "queries", "saved_answers"}


def test_fresh_migration_creates_schema(settings_factory: SettingsFactory) -> None:
    s = settings_factory()
    run_migrations(s)
    run_migrations(s)  # idempotent
    engine = create_engine(s.DATABASE_URL)
    insp = inspect(engine)
    assert set(insp.get_table_names()) >= TABLES
    with engine.connect() as conn:
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0001"
    fks = {fk["referred_table"]: fk["options"].get("ondelete") for fk in insp.get_foreign_keys("chunks")}
    assert fks == {"sources": "CASCADE", "documents": "CASCADE"}
    engine.dispose()


def test_downgrade_removes_schema(settings_factory: SettingsFactory) -> None:
    s = settings_factory()
    run_migrations(s)
    command.downgrade(alembic_config(s), "base")
    engine = create_engine(s.DATABASE_URL)
    assert not (TABLES & set(inspect(engine).get_table_names()))
    engine.dispose()


def _beta_metadata() -> MetaData:
    md = MetaData()
    Table(
        "users",
        md,
        Column("id", Integer, primary_key=True),
        Column("email", String(255), unique=True, index=True, nullable=False),
        Column("name", String(255), nullable=False),
        Column("hashed_password", String(255), nullable=False),
        Column("role", String(32), nullable=False),
        Column("created_at", DateTime, nullable=False),
    )
    Table(
        "sources",
        md,
        Column("id", Integer, primary_key=True),
        Column("name", String(255), nullable=False),
        Column("kind", String(32), nullable=False),
        Column("connector_type", String(64)),
        Column("status", String(32), nullable=False),
        Column("last_sync_at", DateTime),
        Column("created_at", DateTime, nullable=False),
    )
    Table(
        "chunks",
        md,
        Column("id", Integer, primary_key=True),
        Column("source_id", Integer, ForeignKey("sources.id"), nullable=False, index=True),
        Column("text", Text, nullable=False),
        Column("embedding_json", Text, nullable=False),
        Column("created_at", DateTime, nullable=False),
    )
    Table(
        "queries",
        md,
        Column("id", Integer, primary_key=True),
        Column("question", Text, nullable=False),
        Column("answer", Text, nullable=False),
        Column("citations_json", Text, nullable=False),
        Column("created_at", DateTime, nullable=False),
    )
    Table(
        "saved_answers",
        md,
        Column("id", Integer, primary_key=True),
        Column("query_id", Integer, ForeignKey("queries.id"), nullable=False, unique=True),
        Column("note", String(255), nullable=False),
        Column("created_at", DateTime, nullable=False),
    )
    return md


def test_upgrade_from_beta_database(settings_factory: SettingsFactory) -> None:
    s = settings_factory(ADMIN_EMAIL="", ADMIN_PASSWORD="")
    engine = create_engine(s.DATABASE_URL)
    md = _beta_metadata()
    md.create_all(engine)
    now = datetime(2026, 9, 1, 12, 0, 0)
    text_body = "The Acme Corp contract is active until December 2027 and billed quarterly."
    (vec,) = HashedEmbedder(256).embed([text_body])  # beta used identical md5-hashed vectors
    rows: dict[str, list[dict[str, Any]]] = {
        "users": [
            {
                "id": 1,
                "email": "Admin@ClusterX.local",
                "name": "Admin",
                "hashed_password": hash_password("Beta-pass-123", rounds=4),
                "role": "admin",
                "created_at": now,
            }
        ],
        "sources": [
            {
                "id": 1,
                "name": "Acme MSA",
                "kind": "document",
                "connector_type": None,
                "status": "ready",
                "last_sync_at": now,
                "created_at": now,
            }
        ],
        "chunks": [{"id": 1, "source_id": 1, "text": text_body, "embedding_json": json.dumps(vec), "created_at": now}],
        "queries": [{"id": 1, "question": "acme?", "answer": "active [1]", "citations_json": "[]", "created_at": now}],
        "saved_answers": [{"id": 1, "query_id": 1, "note": "pin", "created_at": now}],
    }
    with engine.begin() as conn:
        for table, values in rows.items():
            conn.execute(md.tables[table].insert(), values)
    engine.dispose()

    app = create_app(s)  # AUTO_MIGRATE upgrades the beta schema in place
    engine = create_engine(s.DATABASE_URL)
    insp = inspect(engine)
    assert set(insp.get_table_names()) >= TABLES
    assert {"is_active", "token_version", "last_login_at"} <= {c["name"] for c in insp.get_columns("users")}
    with engine.connect() as conn:
        email, active, ver = conn.execute(text("SELECT email, is_active, token_version FROM users")).one()
        assert (email, bool(active), ver) == ("admin@clusterx.local", True, 0)
        model, dim = conn.execute(text("SELECT embedding_model, embedding_dim FROM chunks")).one()
        assert (model, dim) == ("hashed-256", 256)
        assert conn.execute(text("SELECT mode FROM queries")).scalar() == "extractive"
        assert conn.execute(text("SELECT note FROM saved_answers")).scalar() == "pin"
    engine.dispose()

    with TestClient(app) as client:
        r = client.post("/api/auth/login", json={"email": "admin@clusterx.local", "password": "Beta-pass-123"})
        assert r.status_code == 200, r.text
        headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
        client.cookies.clear()
        ans = client.post("/api/ask", headers=headers, json={"question": "Is the Acme Corp contract active?"})
        assert ans.status_code == 200
        assert ans.json()["citations"][0]["source_name"] == "Acme MSA"
        # cascading delete works on the upgraded foreign keys
        assert client.delete("/api/sources/1", headers=headers).status_code == 200
        assert client.get("/api/insights", headers=headers).json()["total_chunks"] == 0
        # inserts keep working on the upgraded tables (sequences / rowids intact)
        r2 = client.post("/api/connectors", headers=headers, json={"name": "New", "type": "crm"})
        assert r2.status_code == 201
        r3 = client.post("/api/team/invite", headers=headers, json={"email": "new@example.com"})
        assert r3.status_code == 201 and r3.json()["id"] == 2
