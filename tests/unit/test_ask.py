from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest
from sqlalchemy.orm import Session

from enterprise_brain import models
from enterprise_brain.ask import (
    NO_HITS_ANSWER,
    NO_MATCH_ANSWER,
    SNIPPET_CHARS,
    SYSTEM_PROMPT,
    Answer,
    ask,
    build_citations,
    build_llm_messages,
    clean_citations,
    extractive_answer,
    is_heading,
    source_names,
    terms,
)
from enterprise_brain.config import Settings
from enterprise_brain.deps import AppContext
from enterprise_brain.ingest import store_document
from enterprise_brain.llm import HashedEmbedder, LLMError, MockChatProvider


def chunk(cid: int, sid: int, text: str) -> models.Chunk:
    return models.Chunk(id=cid, source_id=sid, text=text)


NAMES = {1: "Acme MSA", 2: "Vendor list"}


def test_terms_drops_stopwords() -> None:
    assert terms("What is the status of the Acme contract?") == ["status", "acme", "contract"]
    assert terms("Tell me about ALL invoices") == ["invoices"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("MASTER SERVICES AGREEMENT - Acme Corp.", True),
        ("The Acme contract is ACTIVE until 2027.", False),
        ("12345 67890.", True),
        ("Ab", False),
        ("AB c", True),
        ("ABcd", False),
    ],
)
def test_is_heading(text: str, expected: bool) -> None:
    assert is_heading(text) is expected


def test_extractive_no_hits() -> None:
    assert extractive_answer("anything", [], {}) == (NO_HITS_ANSWER, [])


def test_extractive_no_overlap() -> None:
    hits = [(chunk(1, 1, "The quarterly report shows strong growth overall."), 0.9)]
    assert extractive_answer("zebra giraffe", hits, NAMES) == (NO_MATCH_ANSWER, [])


def test_extractive_cites_and_orders_sources() -> None:
    hits = [
        (chunk(10, 1, "The Acme contract is active until December 2027. Payment terms are Net 30 days."), 0.8),
        (chunk(20, 2, "CloudServe Hosting is the approved cloud vendor for Acme workloads."), 0.5),
    ]
    answer, cites = extractive_answer("Is the Acme contract active?", hits, NAMES)
    assert answer.startswith("The Acme contract is active until December 2027. [1]")
    assert "CloudServe Hosting is the approved cloud vendor for Acme workloads. [2]" in answer
    assert "Payment terms" not in answer  # zero overlap sentences are dropped
    assert [c["n"] for c in cites] == [1, 2]
    assert cites[0] == {
        "n": 1,
        "source_id": 1,
        "source_name": "Acme MSA",
        "chunk_id": 10,
        "snippet": hits[0][0].text[:SNIPPET_CHARS],
    }
    assert cites[1]["source_name"] == "Vendor list"


def test_extractive_filters_short_fragment_heading_and_dupes() -> None:
    text = (
        "ACME CONTRACT SUMMARY FOR REVIEW TEAM. Acme short. "
        "and the acme contract fragment continues here for a while. "
        "The Acme contract renews every January for twelve months."
    )
    hits = [(chunk(1, 1, text), 0.9), (chunk(2, 1, "The Acme contract renews every January for twelve months."), 0.8)]
    answer, cites = extractive_answer("acme contract renews", hits, NAMES)
    assert answer == "The Acme contract renews every January for twelve months. [1]"
    assert len(cites) == 1


def test_extractive_adds_period_and_respects_max() -> None:
    hits = [(chunk(1, 1, "Acme invoice one is overdue now. Acme invoice two is overdue now! Acme invoice three remains unpaid today"), 0.9)]
    answer, _ = extractive_answer("acme invoice overdue", hits, NAMES, max_sentences=1)
    assert answer == "Acme invoice one is overdue now. [1]"
    answer3, _ = extractive_answer("acme invoice three", hits, {1: "X"}, max_sentences=5)
    assert answer3.startswith("Acme invoice three remains unpaid today. [1]")
    assert "Acme invoice two is overdue now! [1]" in answer3


def test_extractive_ranks_by_overlap_then_score() -> None:
    hits = [
        (chunk(1, 1, "Acme pays the invoice within thirty days."), 0.2),
        (chunk(2, 2, "Acme invoice approval needs the CFO sign-off."), 0.9),
    ]
    answer, cites = extractive_answer("acme invoice approval", hits, NAMES, max_sentences=1)
    assert answer == "Acme invoice approval needs the CFO sign-off. [1]"
    assert cites[0]["source_id"] == 2
    tie = [
        (chunk(1, 1, "Acme invoice handled by finance team."), 0.2),
        (chunk(2, 2, "Acme invoice handled by sales team."), 0.9),
    ]
    a2, c2 = extractive_answer("acme invoice", tie, NAMES, max_sentences=1)
    assert c2[0]["source_id"] == 2 and "sales" in a2


def test_build_citations_unknown_name() -> None:
    hits = [(chunk(5, 9, "text"), 1.0)]
    assert build_citations([9], hits, {})[0]["source_name"] == "Source #9"


def test_build_llm_messages_numbers_sources_and_truncates() -> None:
    hits = [(chunk(1, 1, "A" * 50), 0.9), (chunk(2, 2, "B" * 50), 0.8), (chunk(3, 1, "C" * 10), 0.7)]
    msgs = build_llm_messages("Why?", hits, [1, 2], NAMES)
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = msgs[1]["content"]
    assert user.startswith("Sources:\n[1] Acme MSA: " + "A" * 50 + " " + "C" * 10 + "\n[2] Vendor list: ")
    assert user.endswith("\n\nQuestion: Why?")
    short = build_llm_messages("q", hits, [1, 2], NAMES, max_context_chars=30)[1]["content"]
    assert "[1] Acme MSA: " + "A" * 30 + "\n" in short + "\n"
    assert "[2]" not in short


def test_clean_citations() -> None:
    assert clean_citations("Fact [1] and [3] and [0].", 2) == ("Fact [1] and and .", {1})
    assert clean_citations("A [2] B [2]", 2) == ("A [2] B [2]", {2})
    assert clean_citations("  none  ", 3) == ("none", set())


def test_answer_as_dict() -> None:
    a = Answer("x", [], "none", 3, 12, ["w"])
    assert a.as_dict() == {
        "query_id": 3,
        "answer": "x",
        "citations": [],
        "mode": "none",
        "latency_ms": 12,
        "warnings": ["w"],
    }
    assert Answer("y", [], "llm").warnings == []


# ---------------------------------------------------------------- full ask() against a DB


def _seed(db: Session, settings: Settings) -> tuple[int, int]:
    e = HashedEmbedder(settings.HASHED_EMBED_DIM)
    a = models.Source(name="Acme MSA", kind="document")
    b = models.Source(name="Leave policy", kind="document")
    db.add_all([a, b])
    db.flush()
    store_document(
        db, settings, e, a, "acme.txt", b"The Acme contract is active until December 2027. It is billed quarterly."
    )
    store_document(db, settings, e, b, "leave.txt", b"Employees receive twenty days of paid leave per year.")
    db.commit()
    return a.id, b.id


def test_ask_extractive_persists_query(ctx: AppContext, db: Session) -> None:
    a_id, _ = _seed(db, ctx.settings)
    result = ask(db, ctx.settings, ctx.embedder, None, "  Is the Acme   contract active? ", user_id=None)
    assert result.mode == "extractive"
    assert "[1]" in result.answer and "Acme" in result.answer
    assert result.citations[0]["source_id"] == a_id
    row = db.get(models.Query, result.query_id)
    assert row is not None
    assert row.question == "Is the Acme contract active?"
    assert row.mode == "extractive"
    assert json.loads(row.citations_json) == result.citations
    assert row.latency_ms == result.latency_ms >= 0


def test_ask_source_filter(ctx: AppContext, db: Session) -> None:
    _, b_id = _seed(db, ctx.settings)
    result = ask(db, ctx.settings, ctx.embedder, None, "Acme contract", source_ids=[b_id])
    assert all(c["source_id"] == b_id for c in result.citations)
    empty = ask(db, ctx.settings, ctx.embedder, None, "Acme contract", source_ids=[])
    assert empty.mode == "none" and empty.answer == NO_HITS_ANSWER


def test_ask_validation(ctx: AppContext, db: Session) -> None:
    with pytest.raises(ValueError, match="required"):
        ask(db, ctx.settings, ctx.embedder, None, "   ")
    with pytest.raises(ValueError, match="at most 2000"):
        ask(db, ctx.settings, ctx.embedder, None, "x" * 2001)
    assert db.query(models.Query).count() == 0


def test_ask_no_data(ctx: AppContext, db: Session) -> None:
    r = ask(db, ctx.settings, ctx.embedder, None, "anything at all")
    assert (r.mode, r.answer, r.citations) == ("none", NO_HITS_ANSWER, [])


def test_ask_with_mock_llm(ctx: AppContext, db: Session) -> None:
    a_id, _ = _seed(db, ctx.settings)
    r = ask(db, ctx.settings, ctx.embedder, MockChatProvider(), "Is the Acme contract active?")
    assert r.mode == "llm"
    assert r.answer == "The Acme contract is active until December 2027. [1]"
    assert [c["source_id"] for c in r.citations] == [a_id]
    assert r.warnings == []


class StubChat:
    name = "stub"

    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.seen: list[Sequence[dict[str, str]]] = []

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        self.seen.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_ask_llm_failure_falls_back(ctx: AppContext, db: Session) -> None:
    _seed(db, ctx.settings)
    r = ask(db, ctx.settings, ctx.embedder, StubChat(LLMError("down")), "Is the Acme contract active?")
    assert r.mode == "extractive"
    assert r.warnings == ["The language model was unavailable; showing an extractive answer."]
    assert "[1]" in r.answer


def test_ask_llm_invalid_citations_removed(ctx: AppContext, db: Session) -> None:
    _seed(db, ctx.settings)
    stub = StubChat("It is active [1] [7].")
    r = ask(db, ctx.settings, ctx.embedder, stub, "Is the Acme contract active?", top_k=1)
    assert r.answer == "It is active [1] ."
    assert [c["n"] for c in r.citations] == [1]
    assert "Question: Is the Acme contract active?" in stub.seen[0][1]["content"]


def test_ask_llm_no_citations_or_empty(ctx: AppContext, db: Session) -> None:
    _seed(db, ctx.settings)
    r = ask(db, ctx.settings, ctx.embedder, StubChat("The sources do not say."), "Is the Acme contract active?")
    assert (r.mode, r.citations) == ("llm", [])
    r2 = ask(db, ctx.settings, ctx.embedder, StubChat("[9]"), "Is the Acme contract active?")
    assert r2.mode == "extractive"  # nothing left after cleaning -> keep extractive


def test_ask_llm_not_called_without_hits(ctx: AppContext, db: Session) -> None:
    stub = StubChat("should not be used")
    r = ask(db, ctx.settings, ctx.embedder, stub, "anything")
    assert stub.seen == [] and r.mode == "none"


def test_source_names(db: Session, ctx: AppContext) -> None:
    a_id, b_id = _seed(db, ctx.settings)
    hits: list[Any] = [(chunk(1, a_id, "x"), 1.0), (chunk(2, b_id, "y"), 1.0)]
    assert source_names(db, hits) == {a_id: "Acme MSA", b_id: "Leave policy"}
    assert source_names(db, []) == {}
