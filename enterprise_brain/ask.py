"""Answering: retrieve the top-k chunks, then build a cited answer.

* Extractive (always available): pick the sentences covering the most question terms
  and cite their sources as [1], [2], ...
* LLM (when configured): send the full retrieved chunks, numbered by source, to the
  chat provider with instructions to cite. Citation markers that do not refer to a
  retrieved source are removed. If the provider fails, the extractive answer is used.

Every question is stored as a Query row with the answer, citations, mode and latency.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from enterprise_brain import models
from enterprise_brain.config import Settings
from enterprise_brain.llm import ChatProvider, Embedder, LLMError, Message, sentences
from enterprise_brain.vectorstore import Hit, get_store

log = logging.getLogger(__name__)

STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was were be been by as at from "
    "that this it its into what who whom which when where why how do does did can could "
    "should would will there their them they our your you we i he she him her his hers "
    "ours yours me my us any all about tell show give list".split()
)

NO_HITS_ANSWER = (
    "I couldn't find anything in the knowledge base that answers that question. "
    "Try rephrasing, or add more sources under Knowledge Sources."
)
NO_MATCH_ANSWER = (
    "I found related documents but nothing that directly answers that question. "
    "Try rephrasing with terms from your documents."
)
SNIPPET_CHARS = 200
MIN_SENTENCE_CHARS = 30
_CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. Answer the user's question using ONLY "
    "the numbered sources provided. Cite every factual claim with the source number in "
    "square brackets, like [1] or [2]. Be concise and specific; quote figures, dates and "
    "names exactly. If the sources do not contain the answer, say so plainly and do not "
    "guess."
)


def terms(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def is_heading(sentence: str) -> bool:
    """True for shouty title lines such as ``MASTER SERVICES AGREEMENT - Acme Corp.``"""
    letters = [c for c in sentence if c.isalpha()]
    if not letters:
        return True
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) > 0.5


@dataclass
class Answer:
    answer: str
    citations: list[dict[str, Any]]
    mode: str
    query_id: int | None = None
    latency_ms: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "answer": self.answer,
            "citations": self.citations,
            "mode": self.mode,
            "latency_ms": self.latency_ms,
            "warnings": self.warnings,
        }


def source_names(db: Session, hits: list[Hit]) -> dict[int, str]:
    ids = {c.source_id for c, _ in hits}
    if not ids:
        return {}
    rows = db.query(models.Source.id, models.Source.name).filter(models.Source.id.in_(ids)).all()
    return {int(i): str(n) for i, n in rows}


def build_citations(source_order: list[int], hits: list[Hit], names: dict[int, str]) -> list[dict[str, Any]]:
    out = []
    for n, sid in enumerate(source_order, start=1):
        best = next(c for c, _ in hits if c.source_id == sid)
        out.append(
            {
                "n": n,
                "source_id": sid,
                "source_name": names.get(sid, f"Source #{sid}"),
                "chunk_id": best.id,
                "snippet": best.text[:SNIPPET_CHARS],
            }
        )
    return out


def extractive_answer(
    question: str, hits: list[Hit], names: dict[int, str], max_sentences: int = 4
) -> tuple[str, list[dict[str, Any]]]:
    """Build a cited answer from retrieved chunks without an LLM."""
    if not hits:
        return NO_HITS_ANSWER, []
    q_terms = set(terms(question))
    scored: list[tuple[int, float, int, str]] = []
    for idx, (chunk, score) in enumerate(hits):
        for sent in sentences(chunk.text):
            if len(sent) < MIN_SENTENCE_CHARS:
                continue
            if not (sent[0].isupper() or sent[0].isdigit()):
                continue  # mid-sentence fragment created by chunk overlap
            if is_heading(sent):
                continue
            overlap = len(q_terms & set(terms(sent)))
            if overlap:
                scored.append((overlap, score, idx, sent))
    if not scored:
        return NO_MATCH_ANSWER, []
    scored.sort(key=lambda s: (-s[0], -s[1], s[2]))
    picked: list[tuple[int, str]] = []
    for _, _, idx, sent in scored:
        low = sent.lower()
        if any(low in p.lower() or p.lower() in low for _, p in picked):
            continue  # duplicate or subsumed sentence (chunk overlap)
        picked.append((idx, sent))
        if len(picked) >= max_sentences:
            break
    source_order: list[int] = []
    for idx, _ in picked:
        sid = hits[idx][0].source_id
        if sid not in source_order:
            source_order.append(sid)
    number = {sid: n for n, sid in enumerate(source_order, start=1)}
    parts = []
    for idx, sent in picked:
        sent = sent if sent[-1] in ".!?" else sent + "."
        parts.append(f"{sent} [{number[hits[idx][0].source_id]}]")
    return " ".join(parts), build_citations(source_order, hits, names)


def build_llm_messages(
    question: str,
    hits: list[Hit],
    source_order: list[int],
    names: dict[int, str],
    max_context_chars: int = 12000,
) -> list[Message]:
    blocks: list[str] = []
    used = 0
    for n, sid in enumerate(source_order, start=1):
        remaining = max_context_chars - used
        if remaining <= 0:
            break
        body = " ".join(c.text for c, _ in hits if c.source_id == sid)[:remaining]
        used += len(body)
        blocks.append(f"[{n}] {names.get(sid, f'Source #{sid}')}: {body}")
    user = "Sources:\n" + "\n".join(blocks) + f"\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def clean_citations(answer: str, valid: int) -> tuple[str, set[int]]:
    """Drop citation markers outside 1..valid; return the cleaned text and used numbers."""
    used: set[int] = set()

    def repl(m: re.Match[str]) -> str:
        n = int(m.group(1))
        if 1 <= n <= valid:
            used.add(n)
            return m.group(0)
        return ""

    cleaned = _CITATION_RE.sub(repl, answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, used


def ask(
    db: Session,
    settings: Settings,
    embedder: Embedder,
    chat: ChatProvider | None,
    question: str,
    *,
    user_id: int | None = None,
    source_ids: list[int] | None = None,
    top_k: int | None = None,
) -> Answer:
    started = time.perf_counter()
    question = " ".join(question.split())
    if not question:
        raise ValueError("Question is required")
    if len(question) > settings.MAX_QUESTION_CHARS:
        raise ValueError(f"Question must be at most {settings.MAX_QUESTION_CHARS} characters")
    k = top_k or settings.TOP_K
    query_vec = embedder.embed([question])[0]
    hits = get_store(settings).search(db, query_vec, embedder.model_id, k, source_ids)
    names = source_names(db, hits)
    answer_text, citations = extractive_answer(question, hits, names)
    mode = "extractive" if citations else "none"
    warnings: list[str] = []

    if chat is not None and hits:
        # Rank sources by their best chunk so [1] is the most relevant one.
        order: list[int] = []
        for c, _ in hits:
            if c.source_id not in order:
                order.append(c.source_id)
        try:
            raw = chat.complete(build_llm_messages(question, hits, order, names))
            cleaned, used = clean_citations(raw, len(order))
            if cleaned:
                answer_text = cleaned
                citations = [c for c in build_citations(order, hits, names) if c["n"] in used]
                mode = "llm"
        except LLMError as exc:
            log.warning("LLM answer failed; using extractive answer", extra={"error": str(exc)})
            warnings.append("The language model was unavailable; showing an extractive answer.")

    latency = int((time.perf_counter() - started) * 1000)
    row = models.Query(
        user_id=user_id,
        question=question,
        answer=answer_text,
        citations_json=json.dumps(citations),
        mode=mode,
        latency_ms=latency,
    )
    db.add(row)
    db.commit()
    log.info("question answered", extra={"query_id": row.id, "mode": mode, "latency_ms": latency})
    return Answer(answer_text, citations, mode, row.id, latency, warnings)
