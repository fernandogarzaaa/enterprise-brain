"""Command-line interface: ``enterprise-brain <command>``."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from enterprise_brain import __version__, models
from enterprise_brain.config import Settings, get_settings
from enterprise_brain.deps import AppContext


def load_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


def load_context(migrate: bool = True) -> AppContext:
    from enterprise_brain.db import run_migrations
    from enterprise_brain.logging_config import configure_logging
    from enterprise_brain.main import build_context

    settings = load_settings()
    configure_logging(settings.LOG_LEVEL, settings.log_format, stream=sys.stderr)  # keep stdout for output
    if migrate:
        run_migrations(settings)
    return build_context(settings)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from enterprise_brain.db import run_migrations

    settings = load_settings()
    if settings.AUTO_MIGRATE:
        run_migrations(settings)  # once, in the parent, before workers start
    os.environ["AUTO_MIGRATE"] = "false"
    uvicorn.run(
        "enterprise_brain.main:app_factory",
        factory=True,
        host=args.host or settings.HOST,
        port=args.port or settings.PORT,
        workers=args.workers or settings.WORKERS,
        reload=args.reload,
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"),
        log_config=None,
    )
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    from enterprise_brain.db import run_migrations

    run_migrations(load_settings(), args.revision)
    print(f"database migrated to {args.revision}")
    return 0


def cmd_create_admin(args: argparse.Namespace) -> int:
    from enterprise_brain.users import UserError, create_user

    ctx = load_context()
    password = args.password or os.environ.get("EB_NEW_ADMIN_PASSWORD") or ""
    if not password:
        password = getpass.getpass("Password: ")
        if password != getpass.getpass("Repeat password: "):
            print("error: passwords do not match", file=sys.stderr)
            return 2
    db = ctx.sessions()
    try:
        user = create_user(db, ctx.settings, args.email, password, name=args.name or "", role="admin")
    except UserError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        db.close()
    print(f"admin created: {user.email}")
    return 0


def cmd_seed_demo(args: argparse.Namespace) -> int:
    from enterprise_brain.demo import seed_demo

    ctx = load_context()
    db = ctx.sessions()
    try:
        counts = seed_demo(db, ctx.settings, ctx.embedder)
    finally:
        db.close()
    print(json.dumps(counts))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from enterprise_brain.ingest import IngestError
    from enterprise_brain.sources import SourceError, create_document_source

    ctx = load_context()
    db = ctx.sessions()
    failures = 0
    try:
        for raw in args.paths:
            path = Path(raw)
            try:
                src = create_document_source(
                    db,
                    ctx.settings,
                    ctx.embedder,
                    path.name,
                    path.read_bytes(),
                    name=args.name if len(args.paths) == 1 else None,
                )
                print(f"ingested {path} -> source {src.id} ({len(src.chunks)} chunks)")
            except (OSError, SourceError, IngestError) as exc:
                failures += 1
                print(f"error: {path}: {exc}", file=sys.stderr)
    finally:
        db.close()
    return 1 if failures else 0


def cmd_sync(args: argparse.Namespace) -> int:
    from enterprise_brain.sources import sync_connector

    if not args.all and not args.ids:
        print("error: give connector ids or --all", file=sys.stderr)
        return 2
    ctx = load_context()
    db = ctx.sessions()
    errors = 0
    try:
        q = db.query(models.Source).filter(models.Source.kind == "connector")
        if not args.all:
            q = q.filter(models.Source.id.in_(args.ids))
        for src in q.order_by(models.Source.id).all():
            result = sync_connector(db, ctx.settings, ctx.embedder, src)
            errors += len(result.errors)
            print(json.dumps({"id": src.id, "name": src.name, **result.as_dict()}))
    finally:
        db.close()
    return 1 if errors else 0


def cmd_ask(args: argparse.Namespace) -> int:
    from enterprise_brain.ask import ask

    ctx = load_context()
    db = ctx.sessions()
    try:
        result = ask(
            db,
            ctx.settings,
            ctx.embedder,
            ctx.chat,
            args.question,
            source_ids=args.source_id or None,
            top_k=args.top_k,
        )
    finally:
        db.close()
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(result.answer)
        for c in result.citations:
            print(f"  [{c['n']}] {c['source_name']}")
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    from enterprise_brain.ingest import reindex_all

    ctx = load_context()
    db = ctx.sessions()
    try:
        n = reindex_all(db, ctx.settings, ctx.embedder)
    finally:
        db.close()
    print(f"re-embedded {n} chunks with {ctx.embedder.model_id}")
    return 0


def cmd_check_config(args: argparse.Namespace) -> int:
    for k, v in load_settings().public_summary().items():
        print(f"{k}: {v}")
    return 0


def cmd_healthcheck(args: argparse.Namespace) -> int:
    """Exit 0 when the local server's readiness endpoint answers (used by Docker)."""
    import httpx

    port = args.port or load_settings().PORT
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/api/health/ready", timeout=args.timeout)
    except httpx.HTTPError as exc:
        print(f"unhealthy: {exc}", file=sys.stderr)
        return 1
    if resp.status_code != 200:
        print(f"unhealthy: HTTP {resp.status_code}", file=sys.stderr)
        return 1
    print("ok")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="enterprise-brain", description="Enterprise Brain server and admin tools.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the web server")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--workers", type=int)
    s.add_argument("--reload", action="store_true", help="auto-reload (development)")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("migrate", help="apply database migrations")
    s.add_argument("--revision", default="head")
    s.set_defaults(func=cmd_migrate)

    s = sub.add_parser("create-admin", help="create an admin user")
    s.add_argument("--email", required=True)
    s.add_argument("--name")
    s.add_argument("--password", help="omit to be prompted (or set EB_NEW_ADMIN_PASSWORD)")
    s.set_defaults(func=cmd_create_admin)

    s = sub.add_parser("seed-demo", help="load the sample documents and connectors")
    s.set_defaults(func=cmd_seed_demo)

    s = sub.add_parser("ingest", help="ingest files as document sources")
    s.add_argument("paths", nargs="+")
    s.add_argument("--name", help="source name (single file only)")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("sync", help="sync connectors from their staging folders")
    s.add_argument("ids", nargs="*", type=int)
    s.add_argument("--all", action="store_true")
    s.set_defaults(func=cmd_sync)

    s = sub.add_parser("ask", help="ask a question from the terminal")
    s.add_argument("question")
    s.add_argument("--source-id", type=int, action="append")
    s.add_argument("--top-k", type=int)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("reindex", help="re-embed all chunks after changing EMBEDDINGS")
    s.set_defaults(func=cmd_reindex)

    s = sub.add_parser("healthcheck", help="probe the local server's readiness endpoint")
    s.add_argument("--port", type=int)
    s.add_argument("--timeout", type=float, default=4.0)
    s.set_defaults(func=cmd_healthcheck)

    s = sub.add_parser("check-config", help="validate configuration and print a summary")
    s.set_defaults(func=cmd_check_config)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ValidationError as exc:
        print(f"configuration error:\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
