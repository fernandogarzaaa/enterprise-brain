import io
import os

from fastapi.testclient import TestClient

from app import models
from app.core.db import SessionLocal
from app.ingest import chunk_text, extract_text, ingest_text
from app.main import app

client = TestClient(app)


def _login():
    r = client.post(
        "/api/auth/login",
        json={"email": "admin@clusterx.local", "password": "ChangeMe123!"},
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


AUTH = None


def _auth():
    global AUTH
    if AUTH is None:
        AUTH = _login()
    return AUTH


# ---------------------------------------------------------------- ingest


def test_chunk_text_overlaps():
    text = "The quick brown fox jumps. " * 120
    chunks = chunk_text(text, size=600, overlap=120)
    assert len(chunks) > 1
    # overlap: the tail words of chunk 0 reappear in chunk 1
    tail_words = chunks[0].split()[-4:]
    assert all(w in chunks[1].split() for w in tail_words)
    # every chunk ends on a sentence boundary (no mid-sentence fragments)
    assert all(c.rstrip().endswith((".", "!", "?")) for c in chunks)


def test_extract_text_txt_and_csv(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello enterprise brain")
    assert "enterprise" in extract_text(p, ".txt")
    c = tmp_path / "data.csv"
    c.write_text("client,revenue\nAcme,60000\n")
    assert "Acme" in extract_text(c, ".csv")


def test_ingest_text_stores_chunks():
    db = SessionLocal()
    try:
        src = db.query(models.Source).filter(models.Source.kind == "document").first()
        n_before = db.query(models.Chunk).filter(models.Chunk.source_id == src.id).count()
        n = ingest_text(db, src, " ".join(["testing one two three"] * 80))
        assert n > 0
        n_after = db.query(models.Chunk).filter(models.Chunk.source_id == src.id).count()
        assert n_after == n_before + n
        # rollback the extra chunks so other tests see stable counts
        for ch in (
            db.query(models.Chunk)
            .filter(models.Chunk.source_id == src.id)
            .order_by(models.Chunk.id.desc())
            .limit(n)
            .all()
        ):
            db.delete(ch)
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------- seed sanity


def test_seed_created_sources_and_chunks():
    db = SessionLocal()
    try:
        assert db.query(models.Source).count() >= 13  # 10 docs + 3 connectors
        assert db.query(models.Chunk).count() >= 15  # sentence-snapped chunks
    finally:
        db.close()


# ---------------------------------------------------------------- ask


def test_ask_returns_cited_answer():
    r = client.post(
        "/api/ask",
        headers=_auth(),
        json={"question": "What is the status of the Acme Corp contract?"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "[1]" in data["answer"], data["answer"]
    assert "Acme" in data["answer"]
    assert data["citations"], "expected at least one citation"
    assert "Acme" in data["citations"][0]["source_name"]


def test_ask_records_query_and_recent_lists_it():
    q = "Who are our approved cloud vendors?"
    r = client.post("/api/ask", headers=_auth(), json={"question": q})
    assert r.status_code == 200
    r2 = client.get("/api/queries?limit=5", headers=_auth())
    assert r2.status_code == 200
    assert any(row["question"] == q for row in r2.json())


def test_ask_no_match_returns_honest_fallback():
    r = client.post(
        "/api/ask",
        headers=_auth(),
        json={"question": "zxqj wobble fnord quantum banana"},
    )
    assert r.status_code == 200
    assert r.json()["citations"] == []
    assert "couldn't find" in r.json()["answer"] or "nothing" in r.json()["answer"].lower()


def test_save_and_unsave_query():
    r = client.post("/api/ask", headers=_auth(), json={"question": "What is the invoice approval policy?"})
    qid = r.json()["query_id"]
    r2 = client.post(f"/api/queries/{qid}/save", headers=_auth())
    assert r2.status_code == 200
    saved_id = r2.json()["saved_id"]
    page = client.get("/saved", headers=_auth())
    assert page.status_code == 200
    assert "invoice approval" in page.text.lower()
    r3 = client.delete(f"/api/saved/{saved_id}", headers=_auth())
    assert r3.status_code == 200


# ---------------------------------------------------------------- sources: upload + connectors


def test_upload_txt_ingests():
    r = client.post(
        "/api/sources/upload",
        headers=_auth(),
        files={"file": ("policy.txt", io.BytesIO(b"Remote work stipend is $150 per month for home office costs. " * 40), "text/plain")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["chunks"] > 0
    r2 = client.post(
        "/api/ask", headers=_auth(), json={"question": "What is the remote work stipend?"}
    )
    assert "$150" in r2.json()["answer"] or "stipend" in r2.json()["answer"].lower()


def test_upload_rejects_bad_type():
    r = client.post(
        "/api/sources/upload",
        headers=_auth(),
        files={"file": ("evil.exe", io.BytesIO(b"nope"), "application/octet-stream")},
    )
    assert r.status_code == 400


def test_connector_create_and_sync(tmp_path=None):
    staging = os.environ["EBRAIN_STAGING"]
    r = client.post(
        "/api/connectors",
        headers=_auth(),
        json={"name": "Test CRM Sync", "type": "crm"},
    )
    assert r.status_code == 200, r.text
    cid = r.json()["id"]
    assert os.path.isdir(r.json()["staging_dir"])
    # drop a staged file, then sync
    with open(os.path.join(r.json()["staging_dir"], "accounts.txt"), "w") as f:
        f.write("Key account: Wonka Industries, ARR $99,000, health score 88. " * 30)
    r2 = client.post(f"/api/connectors/{cid}/sync", headers=_auth())
    assert r2.status_code == 200, r2.text
    assert r2.json()["chunks_added"] > 0
    r3 = client.post(
        "/api/ask",
        headers=_auth(),
        json={"question": "What is the ARR of Wonka Industries?", "source_ids": [cid]},
    )
    assert "Wonka" in r3.json()["answer"]


def test_connector_rejects_bad_type():
    r = client.post(
        "/api/connectors", headers=_auth(), json={"name": "X", "type": "telepathy"}
    )
    assert r.status_code == 400


# ---------------------------------------------------------------- insights / team / pages


def test_insights_json():
    r = client.get("/api/insights", headers=_auth())
    assert r.status_code == 200
    d = r.json()
    assert d["total_queries"] >= 3
    assert d["total_sources"] >= 13
    assert d["total_chunks"] >= 15
    assert d["chunks_per_source"]
    assert any("acme" in t["term"] for t in d["top_terms"])


def test_team_invite_creates_member():
    r = client.post(
        "/api/team/invite",
        headers=_auth(),
        json={"email": "teammate@example.com", "name": "Teammate"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["temporary_password"]
    # duplicate invite is rejected
    r2 = client.post(
        "/api/team/invite",
        headers=_auth(),
        json={"email": "teammate@example.com"},
    )
    assert r2.status_code == 409


def test_pages_render():
    for path in ["/", "/sources", "/insights", "/saved", "/team", "/settings"]:
        r = client.get(path, headers=_auth())
        assert r.status_code == 200, path
        assert "Enterprise Brain" in r.text or "Ask" in r.text


def test_ask_requires_auth():
    # fresh client: no login cookie carried over from earlier tests
    anon = TestClient(app)
    r = anon.post("/api/ask", json={"question": "hi"})
    assert r.status_code == 401
