"""Ingest pipeline: extract text -> chunk (overlapping) -> embed -> store."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from sqlalchemy.orm import Session

from app import models
from app.ai.providers import get_provider
from app.core.config import settings

log = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".csv"}

_st_model = None


def extract_text(path: str | Path, suffix: str) -> str:
    """Extract raw text from an uploaded file. Raises ValueError on bad type."""
    suffix = suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix in (".txt", ".md", ".csv"):
        return Path(path).read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, size: int = 600, overlap: int = 120) -> list[str]:
    """Split text into overlapping character chunks.

    Chunk ends prefer a sentence boundary (small lookahead past `size`);
    otherwise they snap to a word boundary. Chunks therefore start mid-text
    (overlap) but end on complete sentences, which keeps retrieved sentences
    citable without fragments.
    """
    text = " ".join(text.split())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            look = text[end : end + 200]
            m = re.search(r"[.!?](\s|$)", look)
            if m:
                end = end + m.end()
            else:
                snap = text.rfind(" ", start, end)
                if snap > start + size // 2:
                    end = snap
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts. Hashed numpy embeddings by default; sentence-transformers
    when EMBEDDINGS=st (imported lazily; falls back to hashed with a warning)."""
    global _st_model
    if settings.EMBEDDINGS == "st":
        try:
            if _st_model is None:
                from sentence_transformers import SentenceTransformer

                _st_model = SentenceTransformer("all-MiniLM-L6-v2")
            vecs = _st_model.encode(texts, normalize_embeddings=True)
            return [v.tolist() for v in vecs]
        except Exception as exc:  # pragma: no cover - optional dependency
            log.warning("sentence-transformers unavailable (%s); using hashed embeddings", exc)
    return get_provider().embed(texts)


def ingest_file(db: Session, source: models.Source, path: str | Path) -> int:
    """Run the full ingest pipeline for one file into an existing Source row.

    Returns the number of chunks stored.
    """
    from app.vectorstore import get_store

    path = Path(path)
    text = extract_text(path, path.suffix)
    chunks = chunk_text(text)
    if not chunks:
        return 0
    embeddings = embed_texts(chunks)
    store = get_store()
    created = 0
    for chunk_text_value, embedding in zip(chunks, embeddings):
        chunk = models.Chunk(
            source_id=source.id,
            text=chunk_text_value,
            embedding_json=json.dumps(embedding),
        )
        db.add(chunk)
        db.flush()  # assign chunk.id for the vector side table
        store.upsert(db, chunk.id, embedding)
        created += 1
    db.commit()
    return created


def ingest_text(db: Session, source: models.Source, text: str) -> int:
    """Ingest raw text directly (used by the seed and tests)."""
    from app.vectorstore import get_store

    chunks = chunk_text(text)
    if not chunks:
        return 0
    embeddings = embed_texts(chunks)
    store = get_store()
    created = 0
    for chunk_text_value, embedding in zip(chunks, embeddings):
        chunk = models.Chunk(
            source_id=source.id,
            text=chunk_text_value,
            embedding_json=json.dumps(embedding),
        )
        db.add(chunk)
        db.flush()
        store.upsert(db, chunk.id, embedding)
        created += 1
    db.commit()
    return created
