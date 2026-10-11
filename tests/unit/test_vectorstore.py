from __future__ import annotations

import json

import pytest
from sqlalchemy.orm import Session

from enterprise_brain import models
from enterprise_brain.deps import AppContext
from enterprise_brain.vectorstore import (
    AutoStore,
    PythonCosineStore,
    get_store,
    parse_embedding,
    pgvector_available,
    vector_literal,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("[1, 2.5]", [1.0, 2.5]),
        ("[]", None),
        ("{}", None),
        ("not json", None),
        ('["a"]', None),
        ("[true, 1]", None),
        ("[1, null]", None),
    ],
)
def test_parse_embedding(raw: str, expected: list[float] | None) -> None:
    assert parse_embedding(raw) == expected


def test_parse_embedding_type_error() -> None:
    assert parse_embedding(None) is None  # type: ignore[arg-type]


def test_vector_literal() -> None:
    assert vector_literal([1, 0.5, -2]) == "[1.0,0.5,-2.0]"


def _add(db: Session, sid: int, text: str, vec: list[float], model: str = "m") -> models.Chunk:
    c = models.Chunk(
        source_id=sid, text=text, embedding_json=json.dumps(vec), embedding_model=model, embedding_dim=len(vec)
    )
    db.add(c)
    db.flush()
    return c


def test_python_store_ranks_filters_and_limits(db: Session) -> None:
    s1 = models.Source(name="one")
    s2 = models.Source(name="two")
    db.add_all([s1, s2])
    db.flush()
    best = _add(db, s1.id, "best", [1.0, 0.0])
    mid = _add(db, s2.id, "mid", [1.0, 1.0])
    _add(db, s1.id, "orthogonal", [0.0, 1.0])
    _add(db, s1.id, "negative", [-1.0, 0.0])
    _add(db, s1.id, "other model", [1.0, 0.0], model="other")
    _add(db, s1.id, "wrong dim", [1.0, 0.0, 0.0])
    _add(db, s1.id, "zero", [0.0, 0.0])
    bad = models.Chunk(source_id=s1.id, text="bad", embedding_json="oops", embedding_model="m")
    db.add(bad)
    db.commit()
    store = PythonCosineStore()
    assert store.name == "python-cosine"
    hits = store.search(db, [1.0, 0.0], "m", top_k=10)
    assert [c.text for c, _ in hits] == ["best", "mid"]
    assert hits[0][1] == pytest.approx(1.0)
    assert hits[1][1] == pytest.approx(2**-0.5)
    assert [c.id for c, _ in store.search(db, [1.0, 0.0], "m", top_k=1)] == [best.id]
    assert [c.id for c, _ in store.search(db, [1.0, 0.0], "m", 10, source_ids=[s2.id])] == [mid.id]
    assert store.search(db, [1.0, 0.0], "m", 10, source_ids=[]) == []
    assert store.search(db, [1.0, 0.0], "nope", 10) == []
    assert store.search(db, [0.0, 0.0], "m", 10) == []
    store.upsert(db, best.id, [1.0])  # no-op


def test_python_store_tie_is_stable(db: Session) -> None:
    s = models.Source(name="s")
    db.add(s)
    db.flush()
    first = _add(db, s.id, "a", [1.0, 0.0])
    _add(db, s.id, "b", [2.0, 0.0])
    db.commit()
    hits = PythonCosineStore().search(db, [1.0, 0.0], "m", 2)
    assert hits[0][0].id == first.id


def test_auto_store_uses_python_on_sqlite(ctx: AppContext, db: Session) -> None:
    store = get_store(ctx.settings)
    assert isinstance(store, AutoStore)
    if ctx.settings.is_sqlite:
        assert store.name == "python-cosine"
        assert pgvector_available(db, ctx.settings) is False
    s = models.Source(name="s")
    db.add(s)
    db.flush()
    c = _add(db, s.id, "x", [1.0, 0.0])
    store.upsert(db, c.id, [1.0, 0.0])
    db.commit()
    assert [h.id for h, _ in store.search(db, [1.0, 0.0], "m", 3)] == [c.id]
