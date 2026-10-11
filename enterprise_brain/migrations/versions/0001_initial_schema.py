"""Initial 1.0 schema (also upgrades a 0.x beta database in place).

Revision ID: 0001
Revises:
Create Date: 2026-10-11

The 0.x beta created its tables with ``create_all`` and had no migration history. When
those tables are found they are altered in place to the 1.0 schema (new columns,
ON DELETE CASCADE foreign keys, the new ``documents`` and ``api_keys`` tables). Beta
password hashes (bcrypt) and hashed embeddings remain valid.

On PostgreSQL the ``vector`` extension and the ``chunk_vectors`` mirror table are
created when available; if the extension cannot be installed the app falls back to
in-process cosine search.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _ts(name: str, nullable: bool = False) -> sa.Column[Any]:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=nullable)


def _create_api_keys() -> None:
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("prefix", sa.String(16), nullable=False),
        sa.Column("key_hash", sa.String(64), nullable=False),
        _ts("created_at"),
        _ts("last_used_at", True),
        _ts("revoked_at", True),
    )
    op.create_index("ix_api_keys_prefix", "api_keys", ["prefix"], unique=True)
    op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"])


def _create_documents() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_id", sa.Integer(), sa.ForeignKey("sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("storage_path", sa.String(1024), nullable=True),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        _ts("created_at"),
        _ts("updated_at"),
        sa.UniqueConstraint("source_id", "filename", name="uq_documents_source_file"),
    )
    op.create_index("ix_documents_source_id", "documents", ["source_id"])


def _create_fresh() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("token_version", sa.Integer(), nullable=False),
        _ts("created_at"),
        _ts("last_login_at", True),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    _create_api_keys()
    op.create_table(
        "sources",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("connector_type", sa.String(64), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_by_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        _ts("last_sync_at", True),
        _ts("created_at"),
    )
    _create_documents()
    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_id", sa.Integer(), sa.ForeignKey("sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding_json", sa.Text(), nullable=False),
        sa.Column("embedding_model", sa.String(128), nullable=False),
        sa.Column("embedding_dim", sa.Integer(), nullable=False),
        _ts("created_at"),
    )
    op.create_index("ix_chunks_source_id", "chunks", ["source_id"])
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_embedding_model", "chunks", ["embedding_model"])
    op.create_table(
        "queries",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("citations_json", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(32), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        _ts("created_at"),
    )
    op.create_index("ix_queries_user_id", "queries", ["user_id"])
    op.create_table(
        "saved_answers",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("query_id", sa.Integer(), sa.ForeignKey("queries.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("note", sa.String(500), nullable=False),
        _ts("created_at"),
        sa.UniqueConstraint("query_id"),
    )


# ---------------------------------------------------------------- 0.x beta upgrade


def _legacy_present() -> bool:
    insp = sa.inspect(op.get_bind())
    if not insp.has_table("users"):
        return False
    return "token_version" not in {c["name"] for c in insp.get_columns("users")}


def _fk_name(table: str, column: str) -> str | None:
    for fk in sa.inspect(op.get_bind()).get_foreign_keys(table):
        if fk.get("constrained_columns") == [column]:
            name = fk.get("name")
            return str(name) if name else None
    return None


def _replace_fk(batch: Any, table: str, column: str, target: str, ondelete: str, name: str) -> None:
    old = _fk_name(table, column)
    if old:
        batch.drop_constraint(old, type_="foreignkey")
    batch.create_foreign_key(name, target, [column], ["id"], ondelete=ondelete)


def _upgrade_legacy() -> None:
    """Bring a 0.x beta schema (created by create_all) up to 1.0 in place."""
    bind = op.get_bind()
    zero = sa.text("0")
    with op.batch_alter_table("users") as b:
        b.add_column(sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()))
        b.add_column(sa.Column("token_version", sa.Integer(), nullable=False, server_default=zero))
        b.add_column(sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True))
    op.execute("UPDATE users SET email = lower(email)")
    with op.batch_alter_table("sources") as b:
        b.add_column(sa.Column("error", sa.Text(), nullable=True))
        b.add_column(sa.Column("created_by_id", sa.Integer(), nullable=True))
        b.create_foreign_key("fk_sources_created_by_id_users", "users", ["created_by_id"], ["id"], ondelete="SET NULL")
    with op.batch_alter_table("queries") as b:
        b.add_column(sa.Column("user_id", sa.Integer(), nullable=True))
        b.add_column(sa.Column("mode", sa.String(32), nullable=False, server_default="extractive"))
        b.add_column(sa.Column("latency_ms", sa.Integer(), nullable=False, server_default=zero))
        b.create_foreign_key("fk_queries_user_id_users", "users", ["user_id"], ["id"], ondelete="SET NULL")
        b.create_index("ix_queries_user_id", ["user_id"], unique=False)
    with op.batch_alter_table("saved_answers") as b:
        b.add_column(sa.Column("user_id", sa.Integer(), nullable=True))
        b.alter_column("note", type_=sa.String(500), existing_type=sa.String(255), existing_nullable=False)
        b.create_foreign_key("fk_saved_answers_user_id_users", "users", ["user_id"], ["id"], ondelete="SET NULL")
        _replace_fk(b, "saved_answers", "query_id", "queries", "CASCADE", "fk_saved_answers_query_id_queries")
    _create_api_keys()
    _create_documents()
    with op.batch_alter_table("chunks") as b:
        b.add_column(sa.Column("document_id", sa.Integer(), nullable=True))
        b.add_column(sa.Column("position", sa.Integer(), nullable=False, server_default=zero))
        # beta used 256-dim md5-hashed embeddings, identical to the 1.0 "hashed-256" model
        b.add_column(sa.Column("embedding_model", sa.String(128), nullable=False, server_default="hashed-256"))
        b.add_column(sa.Column("embedding_dim", sa.Integer(), nullable=False, server_default="256"))
        b.create_foreign_key(
            "fk_chunks_document_id_documents", "documents", ["document_id"], ["id"], ondelete="CASCADE"
        )
        _replace_fk(b, "chunks", "source_id", "sources", "CASCADE", "fk_chunks_source_id_sources")
        b.create_index("ix_chunks_document_id", ["document_id"], unique=False)
        b.create_index("ix_chunks_embedding_model", ["embedding_model"], unique=False)
    if bind.dialect.name == "postgresql" and sa.inspect(bind).has_table("chunk_vectors"):
        op.drop_table("chunk_vectors")  # beta mirror (fixed 256 dims, no FK); rebuilt below


# ---------------------------------------------------------------- pgvector


def _create_pgvector() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    try:
        with bind.begin_nested():
            bind.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception:  # extension missing or no privilege: the python fallback is used
        return
    op.execute(
        "CREATE TABLE IF NOT EXISTS chunk_vectors ("
        "chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE, "
        "embedding vector NOT NULL)"
    )
    op.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) "
        "SELECT id, CAST(embedding_json AS vector) FROM chunks "
        "WHERE embedding_json LIKE '[%' AND embedding_json <> '[]' ON CONFLICT DO NOTHING"
    )


def upgrade() -> None:
    if _legacy_present():
        _upgrade_legacy()
    else:
        _create_fresh()
    _create_pgvector()


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chunk_vectors")
    for table in ("saved_answers", "queries", "chunks", "documents", "sources", "api_keys", "users"):
        op.drop_table(table)
