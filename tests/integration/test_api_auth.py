from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from enterprise_brain.deps import COOKIE_NAME
from enterprise_brain.main import create_app
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, SettingsFactory, login


def test_health_ready_version(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "app": "Enterprise Brain", "version": "1.0.0"}
    assert client.get("/api/health/ready").json() == {"status": "ok", "database": "ok"}
    assert client.get("/api/version").json() == {"version": "1.0.0"}


def test_security_headers_and_request_id(client: TestClient) -> None:
    r = client.get("/api/health", headers={"X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert "strict-transport-security" not in r.headers
    bad = client.get("/api/health", headers={"X-Request-ID": "bad id with spaces!"})
    assert len(bad.headers["x-request-id"]) == 32


def test_login_json_and_me(client: TestClient) -> None:
    r = client.post("/api/auth/login", json={"email": " Admin@Example.com ", "password": ADMIN_PASSWORD})
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer" and body["expires_in"] == 720 * 60
    cookie = r.headers["set-cookie"]
    assert f"{COOKIE_NAME}=" in cookie and "HttpOnly" in cookie and "samesite=lax" in cookie.lower()
    assert "Secure" not in cookie
    me = client.get("/api/auth/me")  # cookie auth
    assert me.status_code == 200
    assert me.json()["email"] == ADMIN_EMAIL and me.json()["role"] == "admin"
    assert me.json()["last_login_at"] is not None


def test_login_failures_and_rate_limit(settings_factory: SettingsFactory) -> None:
    app = create_app(settings_factory(LOGIN_RATE_LIMIT=2))
    with TestClient(app) as c:
        bad = {"email": ADMIN_EMAIL, "password": "wrong"}
        assert c.post("/api/auth/login", json=bad).status_code == 401
        assert c.post("/api/auth/login", json=bad).json() == {"detail": "Invalid credentials"}
        limited = c.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
        assert limited.status_code == 429
        assert int(limited.headers["retry-after"]) >= 1
        form = c.post("/login", data={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}, follow_redirects=False)
        assert form.headers["location"] == "/login?error=locked"
        # other accounts are not locked out
        assert c.post("/api/auth/login", json={"email": "x@example.com", "password": "y"}).status_code == 401


def test_successful_login_resets_limit(settings_factory: SettingsFactory) -> None:
    app = create_app(settings_factory(LOGIN_RATE_LIMIT=2))
    with TestClient(app) as c:
        c.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": "wrong"})
        assert c.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}).status_code == 200
        c.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": "wrong"})
        assert c.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}).status_code == 200


def test_login_validation(client: TestClient) -> None:
    r = client.post("/api/auth/login", json={"email": "a", "password": ""})
    assert r.status_code == 422
    assert r.json()["detail"] == "Invalid request"
    assert {tuple(e["loc"]) for e in r.json()["errors"]} == {("body", "email"), ("body", "password")}


def test_form_login_and_logout(client: TestClient) -> None:
    bad = client.post("/login", data={"email": ADMIN_EMAIL, "password": "nope"}, follow_redirects=False)
    assert bad.status_code == 303 and bad.headers["location"] == "/login?error=1"
    ok = client.post("/login", data={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}, follow_redirects=False)
    assert ok.status_code == 303 and ok.headers["location"] == "/"
    assert client.get("/", follow_redirects=False).status_code == 200
    assert client.get("/login", follow_redirects=False).headers["location"] == "/"
    out = client.get("/logout", follow_redirects=False)
    assert out.headers["location"] == "/login"
    client.cookies.clear()
    assert client.get("/", follow_redirects=False).headers["location"] == "/login"
    api_out = client.post("/api/auth/logout")
    assert api_out.status_code == 200 and f'{COOKIE_NAME}=""' in api_out.headers["set-cookie"]


def test_auth_required_everywhere(client: TestClient) -> None:
    for method, path in [
        ("get", "/api/auth/me"),
        ("post", "/api/ask"),
        ("get", "/api/queries"),
        ("get", "/api/sources"),
        ("get", "/api/team"),
        ("get", "/api/insights"),
        ("get", "/api/saved"),
        ("get", "/api/auth/api-keys"),
    ]:
        r = getattr(client, method)(path)
        assert r.status_code == 401, path
        assert r.headers["www-authenticate"] == "Bearer"
    for page in ["/", "/sources", "/insights", "/saved", "/team", "/settings"]:
        assert client.get(page, follow_redirects=False).headers["location"] == "/login"


def test_invalid_tokens_rejected(client: TestClient) -> None:
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": "Basic abc"}).status_code == 401
    client.cookies.set(COOKIE_NAME, "garbage")
    assert client.get("/api/auth/me").status_code == 401


def test_cookie_csrf_origin_check(client: TestClient) -> None:
    client.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    evil = client.post("/api/ask", json={"question": "hi"}, headers={"Origin": "https://evil.example"})
    assert evil.status_code == 403
    assert evil.json()["detail"] == "Cross-origin request rejected"
    same = client.post("/api/ask", json={"question": "hi"}, headers={"Origin": "http://testserver"})
    assert same.status_code == 200
    referer = client.post("/api/ask", json={"question": "hi"}, headers={"Referer": "https://evil.example/x"})
    assert referer.status_code == 403
    assert client.get("/api/auth/me", headers={"Origin": "https://evil.example"}).status_code == 200  # safe method


def test_bearer_ignores_origin(client: TestClient, admin_headers: dict[str, str]) -> None:
    r = client.post("/api/ask", json={"question": "hi"}, headers={**admin_headers, "Origin": "https://other.app"})
    assert r.status_code == 200


def test_change_password_revokes_other_sessions(client: TestClient, admin_headers: dict[str, str]) -> None:
    other_session = login(client)
    r = client.post(
        "/api/auth/password",
        headers=admin_headers,
        json={"current_password": ADMIN_PASSWORD, "new_password": "brand-new-pass-9"},
    )
    assert r.status_code == 200
    new_token = r.json()["access_token"]
    client.cookies.clear()
    assert client.get("/api/auth/me", headers=other_session).status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {new_token}"}).status_code == 200
    assert login(client, ADMIN_EMAIL, "brand-new-pass-9")
    wrong = client.post(
        "/api/auth/password",
        headers={"Authorization": f"Bearer {new_token}"},
        json={"current_password": "nope", "new_password": "another-pass-9"},
    )
    assert wrong.status_code == 403
    weak = client.post(
        "/api/auth/password",
        headers={"Authorization": f"Bearer {new_token}"},
        json={"current_password": "brand-new-pass-9", "new_password": "short"},
    )
    assert weak.status_code == 400 and "at least 10" in weak.json()["detail"]


def test_api_keys_endpoints(client: TestClient, admin_headers: dict[str, str]) -> None:
    created = client.post("/api/auth/api-keys", headers=admin_headers, json={"name": "Slack bot"})
    assert created.status_code == 201
    data: dict[str, Any] = created.json()
    key = data["key"]
    assert key.startswith("eb_") and data["prefix"].endswith("_...") and data["revoked"] is False
    listed = client.get("/api/auth/api-keys", headers=admin_headers).json()
    assert [k["name"] for k in listed] == ["Slack bot"] and "key" not in listed[0]
    assert client.get("/api/auth/me", headers={"X-API-Key": key}).json()["email"] == ADMIN_EMAIL
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {key}"}).status_code == 200
    # API keys are not cookies: no CSRF check
    assert client.post("/api/ask", headers={"X-API-Key": key, "Origin": "https://x"}, json={"question": "q"}).status_code == 200
    assert client.get("/api/auth/me", headers={"X-API-Key": key + "x"}).status_code == 401
    assert client.delete(f"/api/auth/api-keys/{data['id']}", headers=admin_headers).json() == {"revoked": data["id"]}
    assert client.get("/api/auth/me", headers={"X-API-Key": key}).status_code == 401
    assert client.get("/api/auth/api-keys", headers=admin_headers).json()[0]["revoked"] is True
    assert client.delete("/api/auth/api-keys/999", headers=admin_headers).status_code == 404
    assert client.post("/api/auth/api-keys", headers=admin_headers, json={"name": ""}).status_code == 422


def test_production_mode(settings_factory: SettingsFactory) -> None:
    app = create_app(settings_factory(ENVIRONMENT="production", SECRET_KEY="p" * 40))
    with TestClient(app) as c:
        assert c.get("/docs").status_code == 404
        assert c.get("/openapi.json").status_code == 404
        r = c.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
        assert "Secure" in r.headers["set-cookie"]
        assert r.headers["strict-transport-security"].startswith("max-age=31536000")


def test_docs_available_in_dev(client: TestClient) -> None:
    assert client.get("/openapi.json").status_code == 200


def test_cors(settings_factory: SettingsFactory) -> None:
    app = create_app(settings_factory(CORS_ORIGINS="https://portal.example.com"))
    with TestClient(app) as c:
        r = c.options(
            "/api/ask",
            headers={"Origin": "https://portal.example.com", "Access-Control-Request-Method": "POST"},
        )
        assert r.headers["access-control-allow-origin"] == "https://portal.example.com"
        r2 = c.options(
            "/api/ask", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
        )
        assert "access-control-allow-origin" not in r2.headers


def test_unhandled_error_returns_500_json(client: TestClient, admin_headers: dict[str, str], monkeypatch: Any) -> None:
    import enterprise_brain.api.answers as answers

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(answers, "ask", boom)
    r = client.post("/api/ask", headers=admin_headers, json={"question": "q"})
    assert r.status_code == 500
    assert r.json()["detail"] == "Internal server error"
    assert r.json()["request_id"] == r.headers["x-request-id"]


def test_deactivated_user_token_rejected(client: TestClient, admin_headers: dict[str, str], member: dict[str, Any]) -> None:
    assert client.get("/api/auth/me", headers=member["headers"]).status_code == 200
    client.patch(f"/api/team/{member['id']}", headers=admin_headers, json={"is_active": False})
    assert client.get("/api/auth/me", headers=member["headers"]).status_code == 401
