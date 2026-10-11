"""Vector retrieval.

``PythonCosineStore`` scores the embeddings stored on each chunk row with numpy. It
works on every database and is exact; it is the right choice up to a few hundred
thousand chunks.

``PgVectorStore`` is used on PostgreSQL when the ``vector`` extension is available
(the ``chunk_vectors`` mirror table is created by migration 0001). It mirrors every
embedding into a pgvector column and ranks with the ``<=>`` cosine-distance operator.
Any pgvector failure falls back to the Python store, so retrieval never breaks.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

import numpy as np
from sqlalchemy import bindparam, inspect, text
from sqlalchemy.orm import Session

from enterprise_brain import models
from enterprise_brain.config import Settings

log = logging.getLogger(__name__)

Hit = tuple[models.Chunk, float]


def parse_embedding(raw: str) -> list[float] | None:
    try:
        vec = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(vec, list) or not vec:
        return None
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vec):
        return None
    return [float(x) for x in vec]


def vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class VectorStore(Protocol):
    @property
    def name(self) -> str: ...

    def upsert(self, db: Session, chunk_id: int, embedding: list[float]) -> None: ...

    def search(
        self,
        db: Session,
        query: list[float],
        model_id: str,
        top_k: int,
        source_ids: list[int] | None = None,
    ) -> list[Hit]: ...


class PythonCosineStore:
    name = "python-cosine"

    def upsert(self, db: Session, chunk_id: int, embedding: list[float]) -> None:
        return None  # embeddings already live on the chunk row

    def search(
        self,
        db: Session,
        query: list[float],
        model_id: str,
        top_k: int,
        source_ids: list[int] | None = None,
    ) -> list[Hit]:
        q = db.query(models.Chunk).filter(models.Chunk.embedding_model == model_id)
        if source_ids is not None:
            if not source_ids:
                return []
            q = q.filter(models.Chunk.source_id.in_(source_ids))
        chunks: list[models.Chunk] = []
        rows: list[list[float]] = []
        for chunk in q.order_by(models.Chunk.id).all():
            vec = parse_embedding(chunk.embedding_json)
            if vec is not None and len(vec) == len(query):
                chunks.append(chunk)
                rows.append(vec)
        if not chunks:
            return []
        matrix = np.asarray(rows, dtype=np.float64)
        qv = np.asarray(query, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1) * float(np.linalg.norm(qv))
        safe = np.where(norms > 0, norms, 1.0)
        scores = np.where(norms > 0, (matrix @ qv) / safe, 0.0)
        order = np.argsort(-scores, kind="stable")
        out: list[Hit] = []
        for i in order[:top_k]:
            score = float(scores[i])
            if score <= 0:
                break
            out.append((chunks[int(i)], score))
        return out


class PgVectorStore:
    name = "pgvector"

    def upsert(self, db: Session, chunk_id: int, embedding: list[float]) -> None:
        try:
            with db.begin_nested():
                db.execute(
                    text(
                        "INSERT INTO chunk_vectors (chunk_id, embedding) "
                        "VALUES (:cid, CAST(:emb AS vector)) "
                        "ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding"
                    ),
                    {"cid": chunk_id, "emb": vector_literal(embedding)},
                )
        except Exception as exc:  # never block ingest on the mirror table
            log.warning("pgvector upsert failed", extra={"chunk_id": chunk_id, "error": str(exc)})

    def search(
        self,
        db: Session,
        query: list[float],
        model_id: str,
        top_k: int,
        source_ids: list[int] | None = None,
    ) -> list[Hit]:
        if source_ids is not None and not source_ids:
            return []
        sql = (
            "SELECT c.id AS id, (v.embedding <=> CAST(:q AS vector)) AS distance "
            "FROM chunk_vectors v JOIN chunks c ON c.id = v.chunk_id "
            "WHERE c.embedding_model = :model AND vector_dims(v.embedding) = :dim "
        )
        params: dict[str, object] = {
            "q": vector_literal(query),
            "model": model_id,
            "dim": len(query),
            "k": top_k,
        }
        if source_ids is not None:
            sql += "AND c.source_id IN :sids "
            params["sids"] = list(source_ids)
        sql += "ORDER BY distance ASC, c.id ASC LIMIT :k"
        stmt = text(sql)
        if source_ids is not None:
            stmt = stmt.bindparams(bindparam("sids", expanding=True))
        rows = db.execute(stmt, params).all()
        ids = [int(r.id) for r in rows]
        by_id: dict[int, models.Chunk] = {}
        if ids:
            by_id = {c.id: c for c in db.query(models.Chunk).filter(models.Chunk.id.in_(ids)).all()}
        out: list[Hit] = []
        for r in rows:
            chunk = by_id.get(int(r.id))
            similarity = 1.0 - float(r.distance)
            if chunk is not None and similarity > 0:
                out.append((chunk, similarity))
        return out


_pgvector_available: dict[str, bool] = {}


def pgvector_available(db: Session, settings: Settings) -> bool:
    if not settings.is_postgres:
        return False
    url = settings.DATABASE_URL
    if url not in _pgvector_available:
        try:
            _pgvector_available[url] = inspect(db.get_bind()).has_table("chunk_vectors")
        except Exception:  # pragma: no cover - defensive
            _pgvector_available[url] = False
    return _pgvector_available[url]


class AutoStore:
    """Uses pgvector when available on this database, else Python cosine."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.python = PythonCosineStore()
        self.pg = PgVectorStore()

    @property
    def name(self) -> str:
        return "pgvector (with python fallback)" if self.settings.is_postgres else self.python.name

    def upsert(self, db: Session, chunk_id: int, embedding: list[float]) -> None:
        if pgvector_available(db, self.settings):
            self.pg.upsert(db, chunk_id, embedding)

    def search(
        self,
        db: Session,
        query: list[float],
        model_id: str,
        top_k: int,
        source_ids: list[int] | None = None,
    ) -> list[Hit]:
        if pgvector_available(db, self.settings):
            try:
                with db.begin_nested():
                    return self.pg.search(db, query, model_id, top_k, source_ids)
            except Exception as exc:
                log.warning("pgvector search failed; using python-cosine", extra={"error": str(exc)})
        return self.python.search(db, query, model_id, top_k, source_ids)


def get_store(settings: Settings) -> VectorStore:
    return AutoStore(settings)


def reset_pgvector_cache() -> None:
    _pgvector_available.clear()
