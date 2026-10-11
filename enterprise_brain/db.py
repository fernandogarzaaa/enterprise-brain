"""Database engine, sessions and Alembic migrations."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from enterprise_brain.config import Settings


class Base(DeclarativeBase):
    pass


def ensure_sqlite_dir(url: str) -> None:
    """Create the parent directory of a file-based SQLite database."""
    if url.startswith("sqlite:///") and ":memory:" not in url:
        Path(url[len("sqlite:///") :]).parent.mkdir(parents=True, exist_ok=True)


def make_engine(settings: Settings) -> Engine:
    url = settings.DATABASE_URL
    if url.startswith("sqlite"):
        ensure_sqlite_dir(url)
        engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

        return engine
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)


def make_sessionmaker(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def alembic_config(settings: Settings) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(resources.files("enterprise_brain") / "migrations"))
    cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL.replace("%", "%%"))
    return cfg


def run_migrations(settings: Settings, revision: str = "head") -> None:
    """Upgrade the database schema to ``revision`` (default: latest)."""
    ensure_sqlite_dir(settings.DATABASE_URL)
    command.upgrade(alembic_config(settings), revision)


def check_database(session: Session) -> None:
    """Raise if the database is unreachable."""
    session.execute(text("SELECT 1"))
