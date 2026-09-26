"""Retrieval abstraction.

PythonCosineStore (default): cosine similarity in Python over the embeddings
stored in Chunk.embedding_json. Always works, on any database.

PgVectorStore: used only when DATABASE_URL points at PostgreSQL. Best-effort:
creates the vector extension and a chunk_vectors side table; any failure at
setup or query time falls back to PythonCosineStore. The side table assumes
the default 256-dim hashed embeddings (EMBEDDINGS=hashed); upsert is
best-effort and never blocks ingest.

NOTE: the test suite runs on SQLite, so the Python path is the tested one.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.orm import Session

from app import models
from app.ai.providers import cosine
from app.core.config import settings

log = logging.getLogger(__name__)


def _parse_embedding(raw: str) -> list[float] | None:
    try:
        vec = json.loads(raw)
        if isinstance(vec, list) and vec and all(isinstance(x, (int, float)) for x in vec):
            return [float(x) for x in vec]
    except (ValueError, TypeError):
        pass
    return None


class VectorStore(Protocol):
    name: str

    def upsert(self, db: Session, chunk_id: int, embedding: list[float]) -> None: ...
    def search(
        self,
        db: Session,
        query_embedding: list[float],
        top_k: int = 5,
        source_ids: list[int] | None = None,
    ) -> list[tuple[models.Chunk, float]]: ...


class PythonCosineStore:
    """Pure-Python cosine retrieval. Works everywhere."""

    name = "python-cosine"

    def upsert(self, db: Session, chunk_id: int, embedding: list[float]) -> None:
        # embeddings already live on the Chunk row; nothing extra to do.
        return None

    def search(
        self,
        db: Session,
        query_embedding: list[float],
        top_k: int = 5,
        source_ids: list[int] | None = None,
    ) -> list[tuple[models.Chunk, float]]:
        q = db.query(models.Chunk)
        if source_ids:
            q = q.filter(models.Chunk.source_id.in_(source_ids))
        scored: list[tuple[models.Chunk, float]] = []
        for chunk in q.all():
            vec = _parse_embedding(chunk.embedding_json)
            if not vec or len(vec) != len(query_embedding):
                continue
            score = cosine(query_embedding, vec)
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


class PgVectorStore:
    """pgvector-backed retrieval. Best-effort; falls back on any error."""

    name = "pgvector"
    DIM = 256

    def _setup(self, db: Session) -> None:
        db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        db.execute(
            text(
                "CREATE TABLE IF NOT EXISTS chunk_vectors ("
                "chunk_id INTEGER PRIMARY KEY, embedding vector(256))"
            )
        )
        db.commit()

    def upsert(self, db: Session, chunk_id: int, embedding: list[float]) -> None:
        try:
            if len(embedding) != self.DIM:
                return  # side table only tracks the default hashed embeddings
            self._setup(db)
            literal = "[" + ",".join(repr(float(x)) for x in embedding) + "]"
            db.execute(
                text(
                    "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES "
                    "(:cid, :emb::vector) ON CONFLICT (chunk_id) DO UPDATE "
                    "SET embedding = EXCLUDED.embedding"
                ),
                {"cid": chunk_id, "emb": literal},
            )
            db.commit()
        except Exception as exc:
            log.warning("pgvector upsert failed, continuing without it: %s", exc)

    def search(
        self,
        db: Session,
        query_embedding: list[float],
        top_k: int = 5,
        source_ids: list[int] | None = None,
    ) -> list[tuple[models.Chunk, float]]:
        self._setup(db)
        literal = "[" + ",".join(repr(float(x)) for x in query_embedding) + "]"
        sql = (
            "SELECT c.id, (v.embedding <=> :q::vector) AS distance "
            "FROM chunk_vectors v JOIN chunks c ON c.id = v.chunk_id "
        )
        params: dict = {"q": literal, "k": top_k}
        if source_ids:
            sql += "WHERE c.source_id = ANY(:sids) "
            params["sids"] = list(source_ids)
        sql += "ORDER BY distance ASC LIMIT :k"
        rows = db.execute(text(sql), params).all()
        chunks = {c.id: c for c in db.query(models.Chunk).all()}
        out: list[tuple[models.Chunk, float]] = []
        for chunk_id, distance in rows:
            chunk = chunks.get(chunk_id)
            if chunk is not None:
                out.append((chunk, 1.0 / (1.0 + float(distance))))
        return out


def get_store() -> VectorStore:
    """Pick the store for the configured database."""
    if settings.DATABASE_URL.startswith("postgresql"):
        return PgVectorStore()
    return PythonCosineStore()


def search_chunks(
    db: Session,
    query_embedding: list[float],
    top_k: int = 5,
    source_ids: list[int] | None = None,
) -> list[tuple[models.Chunk, float]]:
    """Top-k retrieval with automatic fallback to the Python store."""
    store = get_store()
    if isinstance(store, PgVectorStore):
        try:
            return store.search(db, query_embedding, top_k, source_ids)
        except Exception as exc:
            log.warning("pgvector search failed (%s); falling back to python-cosine", exc)
    return PythonCosineStore().search(db, query_embedding, top_k, source_ids)
