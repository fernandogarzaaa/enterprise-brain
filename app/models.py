from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text

from app.core.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), default="", nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(32), default="member", nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)


# --- Product models ---


class Source(Base):
    """A knowledge source: an uploaded document or a connector record."""

    __tablename__ = "sources"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    kind = Column(String(32), default="document", nullable=False)  # document | connector
    connector_type = Column(String(64), nullable=True)  # crm, erp, gdrive, ...
    status = Column(String(32), default="ready", nullable=False)  # ready | syncing | error
    last_sync_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)


class Chunk(Base):
    """One text chunk of a source, with its embedding stored as JSON."""

    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    embedding_json = Column(Text, nullable=False, default="[]")
    created_at = Column(DateTime, default=utcnow, nullable=False)


class Query(Base):
    """A question asked through Ask, with the answer that was returned."""

    __tablename__ = "queries"

    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False, default="")
    citations_json = Column(Text, nullable=False, default="[]")
    created_at = Column(DateTime, default=utcnow, nullable=False)


class SavedAnswer(Base):
    """A starred query answer."""

    __tablename__ = "saved_answers"

    id = Column(Integer, primary_key=True)
    query_id = Column(Integer, ForeignKey("queries.id"), nullable=False, unique=True)
    note = Column(String(255), default="", nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)
