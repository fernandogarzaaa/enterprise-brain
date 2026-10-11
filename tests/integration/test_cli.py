"""The ``enterprise-brain`` command-line interface, run in-process against a temp database."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from enterprise_brain import cli
from enterprise_brain.config import get_settings
from enterprise_brain.deps import AppContext
from enterprise_brain.main import create_app
from enterprise_brain.scheduler import run_sync_job, start_scheduler
from enterprise_brain.sources import create_connector, staging_dir_for
from tests.conftest import PG_URL, SettingsFactory


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", PG_URL or f"sqlite:///{tmp_path / 'cli.db'}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("SECRET_KEY", "c" * 40)
    monkeypatch.setenv("BCRYPT_ROUNDS", "4")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.delenv("ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def run(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = cli.main(list(args))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert "enterprise-brain 1.0.0" in capsys.readouterr().out


def test_requires_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert e.value.code == 2


def test_migrate_seed_ask_ingest_sync_reindex(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(capsys, "migrate")[0:2] == (0, "database migrated to head\n")
    code, out, _ = run(capsys, "seed-demo")
    assert code == 0 and json.loads(out) == {"sources": 13, "chunks": json.loads(out)["chunks"]}
    assert json.loads(out)["chunks"] > 0
    code, out, _ = run(capsys, "seed-demo")
    assert json.loads(out) == {"sources": 0, "chunks": 0}

    code, out, _ = run(capsys, "ask", "Which invoices are overdue?")
    assert code == 0 and "[1]" in out and "  [1] " in out
    code, out, _ = run(capsys, "ask", "--json", "--top-k", "2", "Acme contract status?")
    data = json.loads(out)
    assert data["mode"] == "extractive" and data["citations"]

    doc = env / "handbook.md"
    doc.write_text("# Handbook\nThe office closes at 6pm on Fridays for all staff members.")
    code, out, _ = run(capsys, "ingest", str(doc), "--name", "Handbook")
    assert code == 0 and out.startswith(f"ingested {doc} -> source ") and "(1 chunks)" in out
    code, _, err = run(capsys, "ingest", str(env / "missing.txt"), str(env / "bad.exe"))
    assert code == 1 and err.count("error:") == 2

    code, out, _ = run(capsys, "sync", "--all")
    lines = [json.loads(line) for line in out.splitlines()]
    assert code == 0 and len(lines) == 3 and all(line["unchanged"] == 1 for line in lines)
    code, out, _ = run(capsys, "sync", str(lines[0]["id"]))
    assert code == 0 and len(out.splitlines()) == 1
    assert run(capsys, "sync")[0] == 2

    code, out, _ = run(capsys, "reindex")
    assert code == 0 and out.startswith("re-embedded ") and out.endswith("with hashed-256\n")


def test_sync_reports_errors_exit_code(env: Path, capsys: pytest.CaptureFixture[str], settings_factory: Any) -> None:
    ctx = cli.load_context()
    db = ctx.sessions()
    try:
        src = create_connector(db, ctx.settings, "Broken", "custom")
        (staging_dir_for(ctx.settings, src) / "x.json").write_bytes(b"{bad")
    finally:
        db.close()
    code, out, _ = run(capsys, "sync", "--all")
    assert code == 1 and json.loads(out)["errors"]


def test_create_admin(env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    code, out, _ = run(
        capsys, "create-admin", "--email", "Ops@Example.com", "--name", "Ops", "--password", "ops-pass-123"
    )
    assert (code, out) == (0, "admin created: ops@example.com\n")
    code, _, err = run(capsys, "create-admin", "--email", "ops@example.com", "--password", "ops-pass-123")
    assert code == 2 and "already a member" in err
    code, _, err = run(capsys, "create-admin", "--email", "weak@example.com", "--password", "weak")
    assert code == 2 and "at least 10" in err
    monkeypatch.setenv("EB_NEW_ADMIN_PASSWORD", "env-pass-1234")
    assert run(capsys, "create-admin", "--email", "env@example.com")[0] == 0
    monkeypatch.delenv("EB_NEW_ADMIN_PASSWORD")
    answers = iter(["typed-pass-123", "different-123"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt="": next(answers))
    code, _, err = run(capsys, "create-admin", "--email", "typed@example.com")
    assert code == 2 and "do not match" in err
    same = iter(["typed-pass-123", "typed-pass-123"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt="": next(same))
    assert run(capsys, "create-admin", "--email", "typed@example.com")[0] == 0


def test_check_config_and_validation_error(
    env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    code, out, _ = run(capsys, "check-config")
    assert code == 0 and "Answer mode: extractive (local, no LLM)" in out
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("SECRET_KEY", "weak")
    code, _, err = run(capsys, "check-config")
    assert code == 2 and "configuration error" in err and "SECRET_KEY" in err


def test_serve_invokes_uvicorn(env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    import uvicorn

    seen: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda target, **kw: seen.update(target=target, **kw))
    monkeypatch.setenv("AUTO_MIGRATE", "true")
    assert run(capsys, "serve", "--port", "9000", "--workers", "2")[0] == 0
    assert seen["target"] == "enterprise_brain.main:app_factory"
    assert (seen["factory"], seen["port"], seen["workers"], seen["host"]) == (True, 9000, 2, "0.0.0.0")
    assert seen["proxy_headers"] is True and seen["reload"] is False
    assert os.environ["AUTO_MIGRATE"] == "false"  # workers don't race migrations
    assert (env / "cli.db").exists() or PG_URL


def test_app_factory(env: Path) -> None:
    from enterprise_brain.main import app_factory

    app = app_factory()
    assert app.title == "Enterprise Brain"


def test_scheduler_job(ctx: AppContext, settings_factory: SettingsFactory) -> None:
    db = ctx.sessions()
    try:
        src = create_connector(db, ctx.settings, "Sched", "crm")
        (staging_dir_for(ctx.settings, src) / "a.txt").write_text("Scheduled sync indexes this account note today.")
    finally:
        db.close()
    assert run_sync_job(ctx) == 1
    assert run_sync_job(ctx) == 0
    sched = start_scheduler(ctx.__class__(**{**ctx.__dict__, "settings": settings_factory(SYNC_INTERVAL_MINUTES=5)}))
    try:
        job = sched.get_job("sync-connectors")
        assert job is not None and job.trigger.interval.total_seconds() == 300
    finally:
        sched.shutdown(wait=False)


def test_lifespan_starts_scheduler(settings_factory: SettingsFactory) -> None:
    from fastapi.testclient import TestClient

    app = create_app(settings_factory(SYNC_INTERVAL_MINUTES=60))
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200
