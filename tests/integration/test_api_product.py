"""End-to-end API flows: sources, connectors, asking, saved answers, team, insights, pages."""

from __future__ import annotations

import io
from typing import Any

import httpx
import respx
from fastapi.testclient import TestClient

from enterprise_brain.main import create_app
from tests.conftest import SettingsFactory, login

POLICY = b"Remote work stipend is $150 per month for home office costs. " * 30


# ---------------------------------------------------------------- demo data + ask


def test_demo_data_seeded_and_answers_cite(demo_client: TestClient, demo_headers: dict[str, str]) -> None:
    sources = demo_client.get("/api/sources", headers=demo_headers).json()
    assert len(sources) == 13
    assert sum(1 for s in sources if s["kind"] == "connector") == 3
    assert all(s["chunks"] > 0 for s in sources)
    r = demo_client.post("/api/ask", headers=demo_headers, json={"question": "What is the status of the Acme Corp contract?"})
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "extractive"
    assert "[1]" in data["answer"] and "Acme" in data["answer"]
    assert "Acme" in data["citations"][0]["source_name"]
    overdue = demo_client.post("/api/ask", headers=demo_headers, json={"question": "Which invoices are overdue?"}).json()
    assert "OVERDUE" in overdue["answer"] or "overdue" in overdue["answer"].lower()


def test_demo_seed_is_idempotent(settings_factory: SettingsFactory) -> None:
    s = settings_factory(DEMO_DATA=True)
    create_app(s)
    app = create_app(s)
    with TestClient(app) as c:
        h = login(c)
        assert len(c.get("/api/sources", headers=h).json()) == 13


def test_ask_validation_and_honest_fallback(demo_client: TestClient, demo_headers: dict[str, str]) -> None:
    assert demo_client.post("/api/ask", headers=demo_headers, json={"question": ""}).status_code == 422
    blank = demo_client.post("/api/ask", headers=demo_headers, json={"question": "   "})
    assert blank.status_code == 400 and blank.json()["detail"] == "Question is required"
    long = demo_client.post("/api/ask", headers=demo_headers, json={"question": "x" * 2001})
    assert long.status_code == 400
    assert demo_client.post("/api/ask", headers=demo_headers, json={"question": "q", "top_k": 0}).status_code == 422
    nothing = demo_client.post("/api/ask", headers=demo_headers, json={"question": "zxqj wobble fnord"}).json()
    assert nothing["citations"] == [] and nothing["mode"] == "none"


def test_queries_are_private_per_user(
    client: TestClient, admin_headers: dict[str, str], member: dict[str, Any]
) -> None:
    client.post("/api/ask", headers=admin_headers, json={"question": "admin question"})
    client.post("/api/ask", headers=member["headers"], json={"question": "member question"})
    mine = client.get("/api/queries", headers=member["headers"]).json()
    assert [q["question"] for q in mine] == ["member question"]
    everyone = client.get("/api/queries?everyone=true", headers=admin_headers).json()
    assert {q["question"] for q in everyone} == {"admin question", "member question"}
    member_everyone = client.get("/api/queries?everyone=true", headers=member["headers"]).json()
    assert [q["question"] for q in member_everyone] == ["member question"]
    admin_q = client.get("/api/queries?everyone=true", headers=admin_headers).json()[-1]
    assert client.get(f"/api/queries/{admin_q['id']}", headers=member["headers"]).status_code == 404
    full = client.get(f"/api/queries/{admin_q['id']}", headers=admin_headers).json()
    assert full["citations"] == [] and full["question"] == "admin question"
    assert client.get("/api/queries?limit=0", headers=admin_headers).status_code == 422
    assert len(client.get("/api/queries?limit=1&everyone=true", headers=admin_headers).json()) == 1


def test_save_and_unsave(client: TestClient, admin_headers: dict[str, str], member: dict[str, Any]) -> None:
    client.post("/api/sources/upload", headers=admin_headers, files={"file": ("p.txt", io.BytesIO(POLICY), "text/plain")})
    qid = client.post("/api/ask", headers=admin_headers, json={"question": "What is the remote work stipend?"}).json()["query_id"]
    r = client.post(f"/api/queries/{qid}/save", headers=admin_headers)
    assert r.status_code == 200
    saved_id = r.json()["saved_id"]
    again = client.post(f"/api/queries/{qid}/save", headers=admin_headers, json={"note": "for HR"})
    assert again.json() == {"saved_id": saved_id, "query_id": qid, "note": "for HR"}
    listed = client.get("/api/saved", headers=member["headers"]).json()  # shared with the team
    assert listed[0]["note"] == "for HR" and listed[0]["citations"]
    page = client.get("/saved", headers=admin_headers)
    assert "remote work stipend" in page.text.lower() and "for HR" in page.text
    assert client.delete(f"/api/saved/{saved_id}", headers=member["headers"]).status_code == 403
    assert client.delete(f"/api/saved/{saved_id}", headers=admin_headers).json() == {"deleted": saved_id}
    assert client.delete(f"/api/saved/{saved_id}", headers=admin_headers).status_code == 404
    assert client.post("/api/queries/9999/save", headers=admin_headers).status_code == 404


# ---------------------------------------------------------------- uploads


def test_upload_then_ask(client: TestClient, admin_headers: dict[str, str]) -> None:
    r = client.post(
        "/api/sources/upload",
        headers=admin_headers,
        files={"file": ("policy.txt", io.BytesIO(POLICY), "text/plain")},
        data={"name": "Stipend policy"},
    )
    assert r.status_code == 201, r.text
    assert r.json()["name"] == "Stipend policy" and r.json()["chunks"] > 0
    ans = client.post("/api/ask", headers=admin_headers, json={"question": "What is the remote work stipend?"}).json()
    assert "$150" in ans["answer"]
    detail = client.get(f"/api/sources/{r.json()['source_id']}", headers=admin_headers).json()
    assert detail["files"][0]["filename"] == "policy.txt"
    assert detail["files"][0]["size_bytes"] == len(POLICY)
    assert detail["documents"] == 1 and "staging_dir" not in detail


def test_upload_rejections(settings_factory: SettingsFactory) -> None:
    app = create_app(settings_factory(MAX_UPLOAD_MB=1))
    with TestClient(app) as c:
        h = login(c)
        bad = c.post("/api/sources/upload", headers=h, files={"file": ("evil.exe", io.BytesIO(b"MZ"), "x")})
        assert bad.status_code == 400 and "Unsupported" in bad.json()["detail"]
        empty = c.post("/api/sources/upload", headers=h, files={"file": ("e.txt", io.BytesIO(b""), "text/plain")})
        assert empty.status_code == 400
        big = c.post(
            "/api/sources/upload", headers=h, files={"file": ("b.txt", io.BytesIO(b"a" * (1024 * 1024 + 5)), "text/plain")}
        )
        assert big.status_code == 400 and "1 MB" in big.json()["detail"]
        blank = c.post("/api/sources/upload", headers=h, files={"file": ("w.txt", io.BytesIO(b"   "), "text/plain")})
        assert blank.status_code == 422 and "no extractable text" in blank.json()["detail"]
        corrupt = c.post("/api/sources/upload", headers=h, files={"file": ("c.pdf", io.BytesIO(b"%PDF-x"), "application/pdf")})
        assert corrupt.status_code == 422
        assert c.get("/api/sources", headers=h).json() == []


# ---------------------------------------------------------------- connectors


def test_connector_lifecycle(client: TestClient, admin_headers: dict[str, str]) -> None:
    assert client.get("/api/connectors/types", headers=admin_headers).json()["gdrive"] == "Google Drive"
    r = client.post("/api/connectors", headers=admin_headers, json={"name": "Test CRM", "type": "crm"})
    assert r.status_code == 201
    cid = r.json()["id"]
    assert r.json()["staging_dir"].endswith(f"{cid}-test-crm")
    empty = client.post(f"/api/connectors/{cid}/sync", headers=admin_headers).json()
    assert empty["note"].startswith("No supported files staged")
    staged = client.post(
        f"/api/connectors/{cid}/files",
        headers=admin_headers,
        files={"file": ("accounts.txt", io.BytesIO(b"Key account: Wonka Industries, ARR $99,000, health 88. " * 20), "text/plain")},
    )
    assert staged.status_code == 201 and staged.json()["staged"] == "accounts.txt"
    s1 = client.post(f"/api/connectors/{cid}/sync", headers=admin_headers).json()
    assert (s1["added"], s1["status"]) == (1, "ready")
    assert s1["chunks_added"] == s1["chunks_indexed"] > 0
    assert s1["note"].startswith(f"Indexed {s1['chunks_indexed']} chunks: 1 new")
    s2 = client.post(f"/api/connectors/{cid}/sync", headers=admin_headers).json()
    assert s2["note"] == "Up to date: 1 unchanged file(s)."
    ans = client.post(
        "/api/ask", headers=admin_headers, json={"question": "What is the ARR of Wonka Industries?", "source_ids": [cid]}
    ).json()
    assert "Wonka" in ans["answer"]
    bad = client.post(
        f"/api/connectors/{cid}/files", headers=admin_headers, files={"file": ("x.json", io.BytesIO(b"{bad"), "x")}
    )
    assert bad.status_code == 201
    s3 = client.post(f"/api/connectors/{cid}/sync", headers=admin_headers).json()
    assert s3["status"] == "error" and s3["note"].startswith("Synced with 1 error(s): x.json")
    listed = {s["id"]: s for s in client.get("/api/sources", headers=admin_headers).json()}
    assert listed[cid]["error"].startswith("x.json")


def test_connector_validation(client: TestClient, admin_headers: dict[str, str]) -> None:
    assert client.post("/api/connectors", headers=admin_headers, json={"name": "X", "type": "telepathy"}).status_code == 400
    assert client.post("/api/connectors", headers=admin_headers, json={"name": "", "type": "crm"}).status_code == 422
    assert client.post("/api/connectors", headers=admin_headers, json={"name": "   ", "type": "crm"}).status_code == 400
    assert client.post("/api/connectors/999/sync", headers=admin_headers).status_code == 404
    doc = client.post("/api/sources/upload", headers=admin_headers, files={"file": ("d.txt", io.BytesIO(POLICY), "text/plain")})
    did = doc.json()["source_id"]
    assert client.post(f"/api/connectors/{did}/sync", headers=admin_headers).status_code == 404
    staged = client.post(f"/api/connectors/{did}/files", headers=admin_headers, files={"file": ("a.txt", io.BytesIO(b"x"), "t")})
    assert staged.status_code == 400
    assert client.get("/api/sources/999", headers=admin_headers).status_code == 404


def test_delete_source_permissions(client: TestClient, admin_headers: dict[str, str], member: dict[str, Any]) -> None:
    mine = client.post(
        "/api/sources/upload", headers=member["headers"], files={"file": ("m.txt", io.BytesIO(POLICY), "text/plain")}
    ).json()["source_id"]
    theirs = client.post(
        "/api/sources/upload", headers=admin_headers, files={"file": ("a.txt", io.BytesIO(POLICY), "text/plain")}
    ).json()["source_id"]
    assert client.delete(f"/api/sources/{theirs}", headers=member["headers"]).status_code == 403
    assert client.delete(f"/api/sources/{mine}", headers=member["headers"]).json() == {"deleted": mine}
    assert client.delete(f"/api/sources/{theirs}", headers=admin_headers).status_code == 200
    assert client.delete(f"/api/sources/{theirs}", headers=admin_headers).status_code == 404
    assert client.get("/api/insights", headers=admin_headers).json()["total_chunks"] == 0


# ---------------------------------------------------------------- team


def test_team_management(client: TestClient, admin_headers: dict[str, str], member: dict[str, Any]) -> None:
    mid = member["id"]
    assert member["role"] == "member" and member["temporary_password"]
    team = client.get("/api/team", headers=member["headers"]).json()
    assert [u["email"] for u in team] == ["admin@example.com", "member@example.com"]
    assert client.post("/api/team/invite", headers=member["headers"], json={"email": "z@example.com"}).status_code == 403
    dup = client.post("/api/team/invite", headers=admin_headers, json={"email": "MEMBER@example.com"})
    assert dup.status_code == 409
    assert client.post("/api/team/invite", headers=admin_headers, json={"email": "nope"}).status_code == 400
    assert client.post("/api/team/invite", headers=admin_headers, json={"email": "r@example.com", "role": "god"}).status_code == 400
    promoted = client.patch(f"/api/team/{mid}", headers=admin_headers, json={"role": "admin", "name": "Mia"}).json()
    assert (promoted["role"], promoted["name"]) == ("admin", "Mia")
    me = client.get("/api/auth/me", headers=admin_headers).json()
    self_demote = client.patch(f"/api/team/{me['id']}", headers=admin_headers, json={"role": "member"})
    assert self_demote.status_code == 400
    reset = client.post(f"/api/team/{mid}/reset-password", headers=admin_headers).json()
    assert reset["email"] == "member@example.com"
    assert client.get("/api/auth/me", headers=member["headers"]).status_code == 401  # signed out
    assert login(client, "member@example.com", reset["temporary_password"])
    assert client.delete(f"/api/team/{me['id']}", headers=admin_headers).status_code == 400
    assert client.delete(f"/api/team/{mid}", headers=admin_headers).json() == {"deleted": mid}
    assert client.patch("/api/team/999", headers=admin_headers, json={"role": "member"}).status_code == 404
    assert client.post("/api/team/999/reset-password", headers=admin_headers).status_code == 404


# ---------------------------------------------------------------- insights + pages


def test_insights(demo_client: TestClient, demo_headers: dict[str, str]) -> None:
    for q in ["What is the Acme contract status?", "Acme invoice?", "zzqx nonsense"]:
        demo_client.post("/api/ask", headers=demo_headers, json={"question": q})
    d = demo_client.get("/api/insights", headers=demo_headers).json()
    assert d["total_queries"] == 3
    assert d["total_sources"] == 13
    assert d["total_members"] == 1
    assert d["unanswered_queries"] == 1
    assert d["answer_modes"] == {"extractive": 2, "none": 1}
    assert d["avg_latency_ms"] is not None
    assert d["top_terms"][0] == {"term": "acme", "count": 2}
    counts = [c["chunks"] for c in d["chunks_per_source"]]
    assert counts == sorted(counts, reverse=True)


def test_empty_insights(client: TestClient, admin_headers: dict[str, str]) -> None:
    d = client.get("/api/insights", headers=admin_headers).json()
    assert d["total_queries"] == 0 and d["avg_latency_ms"] is None and d["top_terms"] == []


def test_pages_render(demo_client: TestClient, demo_headers: dict[str, str]) -> None:
    demo_client.post("/api/ask", headers=demo_headers, json={"question": "Acme contract?"})
    for path, marker in [
        ("/", "Ask the Enterprise Brain"),
        ("/sources", "Knowledge Sources"),
        ("/insights", "Questions asked"),
        ("/saved", "Saved Answers"),
        ("/team", "Invite a member"),
        ("/settings", "Server configuration"),
    ]:
        r = demo_client.get(path, headers=demo_headers)
        assert r.status_code == 200, path
        assert marker in r.text, path
        assert "v1.0.0" in r.text
        assert "cdn." not in r.text  # no third-party CDNs in production pages
    assert "Acme contract?" in demo_client.get("/", headers=demo_headers).text
    assert demo_client.get("/static/app.css").status_code == 200
    assert demo_client.get("/static/vendor/chart.umd.min.js").status_code == 200
    login_page = demo_client.get("/login")
    assert "Sign in" in login_page.text
    assert "Too many attempts" in demo_client.get("/login?error=locked").text
    assert "Invalid email or password" in demo_client.get("/login?error=1").text


def test_member_pages_hide_admin_bits(client: TestClient, member: dict[str, Any]) -> None:
    team = client.get("/team", headers=member["headers"])
    assert "Invite a member" not in team.text
    settings = client.get("/settings", headers=member["headers"])
    assert "Server configuration" not in settings.text and "API keys" in settings.text
    assert "Try asking" not in client.get("/", headers=member["headers"]).text  # no sources yet


# ---------------------------------------------------------------- LLM modes end-to-end


def test_mock_llm_mode_end_to_end(settings_factory: SettingsFactory) -> None:
    app = create_app(settings_factory(DEMO_DATA=True, LLM_PROVIDER="mock"))
    with TestClient(app) as c:
        h = login(c)
        r = c.post("/api/ask", headers=h, json={"question": "Which invoices are overdue?"}).json()
        assert r["mode"] == "llm"
        assert r["answer"].endswith("]")
        assert r["citations"] and all(1 <= cit["n"] <= 5 for cit in r["citations"])
        page = c.get("/settings", headers=h).text
        assert "mock LLM" in page


@respx.mock(assert_all_called=False)
def test_openai_compatible_mode_end_to_end(settings_factory: SettingsFactory, respx_mock: respx.MockRouter) -> None:
    respx_mock.route(host="testserver").pass_through()
    route = respx_mock.post("https://api.groq.example/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Initech owes $31,200 [1]."}}]})
    )
    app = create_app(
        settings_factory(
            DEMO_DATA=True, LLM_BASE_URL="https://api.groq.example/openai/v1", LLM_API_KEY="gsk-test", LLM_MAX_RETRIES=0
        )
    )
    with TestClient(app) as c:
        h = login(c)
        r = c.post("/api/ask", headers=h, json={"question": "Which invoices are overdue?"}).json()
        assert r["mode"] == "llm"
        assert r["answer"] == "Initech owes $31,200 [1]."
        assert [cit["n"] for cit in r["citations"]] == [1]
        sent = route.calls.last.request
        assert sent.headers["authorization"] == "Bearer gsk-test"
        route.mock(return_value=httpx.Response(503))
        fallback = c.post("/api/ask", headers=h, json={"question": "Which invoices are overdue?"}).json()
        assert fallback["mode"] == "extractive"
        assert fallback["warnings"]


@respx.mock(assert_all_called=False)
def test_openai_embeddings_provider_error_is_502(settings_factory: SettingsFactory, respx_mock: respx.MockRouter) -> None:
    respx_mock.route(host="testserver").pass_through()
    respx_mock.post("http://emb.example/v1/embeddings").mock(return_value=httpx.Response(401, text="no"))
    app = create_app(settings_factory(EMBEDDINGS="openai", EMBED_BASE_URL="http://emb.example/v1", LLM_MAX_RETRIES=0))
    with TestClient(app) as c:
        h = login(c)
        r = c.post("/api/ask", headers=h, json={"question": "anything"})
        assert r.status_code == 502
        assert r.json()["detail"].startswith("Embedding/LLM provider error")
