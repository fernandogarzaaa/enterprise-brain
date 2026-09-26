"""Ask: retrieve top-k chunks, build a cited answer.

Local path: extractive summarizer that picks the sentences covering the most
query terms and cites the sources they came from as [1], [2], ...
LLM path: when an LLM provider is configured (LLM_BASE_URL + LLM_API_KEY),
the retrieved context is passed through it for a richer answer, keeping the
same citation mapping. Every asked question is persisted as a Query row.
"""

from __future__ import annotations

import json
import re

from sqlalchemy.orm import Session

from app import models
from app.ai.providers import get_provider
from app.core.config import settings
from app.ingest import embed_texts
from app.vectorstore import search_chunks

_STOPWORDS = set(
    "the a an and or of to in on for with is are was were be been by as at from "
    "that this it its into what who when where how do does did can could should "
    "there their them they our your you we i he she him her his hers ours yours"
    .split()
)


def _terms(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def extractive_answer(
    question: str, hits: list[tuple[models.Chunk, float]], max_sentences: int = 4
) -> tuple[str, list[dict]]:
    """Build a cited answer from retrieved chunks. Pure function, fully tested."""
    if not hits:
        return (
            "I couldn't find anything in the knowledge base that answers that question. "
            "Try rephrasing, or add more sources under Knowledge Sources.",
            [],
        )
    terms = set(_terms(question))
    scored: list[tuple[int, float, int, str]] = []
    for idx, (chunk, score) in enumerate(hits):
        for sent in _sentences(chunk.text):
            if len(sent) < 30:
                continue
            if not (sent[0].isupper() or sent[0].isdigit()):
                continue  # skip mid-sentence fragments from chunk overlap
            overlap = len(terms & set(_terms(sent)))
            scored.append((overlap, score, idx, sent))
    if not scored or max(s[0] for s in scored) == 0:
        return (
            "I found related documents but nothing that directly answers that question. "
            "Try rephrasing with terms from your documents.",
            [],
        )
    scored.sort(key=lambda s: (-s[0], -s[1], s[2]))
    picked: list[tuple[int, str]] = []
    for _, _, idx, sent in scored:
        low = sent.lower()
        # skip exact dupes and sentences subsumed by (or subsuming) a picked one
        if any(low in p.lower() or p.lower() in low for _, p in picked):
            continue
        picked.append((idx, sent))
        if len(picked) >= max_sentences:
            break
    # citation numbers: order sources by first appearance in the answer
    source_order: list[int] = []
    for idx, _ in picked:
        sid = hits[idx][0].source_id
        if sid not in source_order:
            source_order.append(sid)
    num = {sid: n + 1 for n, sid in enumerate(source_order)}
    parts: list[str] = []
    for idx, sent in picked:
        sent = sent if sent[-1] in ".!?" else sent + "."
        parts.append(f"{sent} [{num[hits[idx][0].source_id]}]")
    answer = " ".join(parts)
    citations = [
        {
            "n": n + 1,
            "source_id": sid,
            "source_name": _source_name(hits, sid),
            "snippet": _source_snippet(hits, sid),
        }
        for n, sid in enumerate(source_order)
    ]
    return answer, citations


def _source_name(hits: list[tuple[models.Chunk, float]], source_id: int) -> str:
    for chunk, _ in hits:
        if chunk.source_id == source_id:
            src = getattr(chunk, "_source_cache", None)
            return src.name if src else f"Source #{source_id}"
    return f"Source #{source_id}"


def _source_snippet(hits: list[tuple[models.Chunk, float]], source_id: int) -> str:
    for chunk, _ in hits:
        if chunk.source_id == source_id:
            return chunk.text[:160]
    return ""


def _llm_answer(
    question: str, hits: list[tuple[models.Chunk, float]], citations: list[dict]
) -> str:
    numbered = []
    for c in citations:
        numbered.append(f"[{c['n']}] {c['source_name']}: {c['snippet']}")
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the question using ONLY the numbered source snippets below. "
                "Keep every factual claim tied to a citation like [1] or [2]. "
                "If the snippets do not answer it, say so plainly."
            ),
        },
        {"role": "user", "content": "Sources:\n" + "\n".join(numbered) + f"\n\nQuestion: {question}"},
    ]
    try:
        out = get_provider().chat(messages)
    except Exception:
        out = ""
    return out.strip()


def ask(
    db: Session, question: str, source_ids: list[int] | None = None, top_k: int = 5
) -> dict:
    """Ask a question. Returns {query_id, answer, citations} and stores the query."""
    question = question.strip()
    hits = []
    if question:
        qvec = embed_texts([question])[0]
        hits = search_chunks(db, qvec, top_k=top_k, source_ids=source_ids or None)
        # attach source names for citation rendering
        if hits:
            src_ids = {c.source_id for c, _ in hits}
            srcs = {s.id: s for s in db.query(models.Source).filter(models.Source.id.in_(src_ids)).all()}
            for chunk, _ in hits:
                chunk._source_cache = srcs.get(chunk.source_id)  # type: ignore[attr-defined]
    answer, citations = extractive_answer(question, hits)
    if settings.LLM_BASE_URL and settings.LLM_API_KEY and citations:
        enhanced = _llm_answer(question, hits, citations)
        if enhanced:
            answer = enhanced
    row = models.Query(
        question=question, answer=answer, citations_json=json.dumps(citations)
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"query_id": row.id, "answer": answer, "citations": citations}
