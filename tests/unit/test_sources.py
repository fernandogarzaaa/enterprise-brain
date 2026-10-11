from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from enterprise_brain import models
from enterprise_brain.deps import AppContext
from enterprise_brain.ingest import IngestError, reindex_all, store_document
from enterprise_brain.llm import HashedEmbedder, LLMError
from enterprise_brain.sources import (
    CONNECTOR_TYPES,
    SourceError,
    SyncResult,
    create_connector,
    create_document_source,
    delete_source,
    slugify,
    stage_connector_file,
    staging_dir_for,
    sync_all_connectors,
    sync_connector,
    upload_dir_for,
    validate_name,
    validate_upload,
)

TEXT = b"The Wonka account has ARR of $99,000 and a health score of 88. " * 20


def test_slugify() -> None:
    assert slugify("Salesforce CRM (EU)") == "salesforce-crm-eu"
    assert slugify("!!!") == "source"


def test_validate_name() -> None:
    assert validate_name("  My   CRM ") == "My CRM"
    with pytest.raises(SourceError, match="required"):
        validate_name("   ")
    with pytest.raises(SourceError, match="at most 255"):
        validate_name("x" * 256)
    assert validate_name("x" * 255) == "x" * 255


def test_validate_upload(ctx: AppContext) -> None:
    s = ctx.settings
    assert validate_upload("../a/b/report.TXT", b"x", s) == "report.TXT"
    with pytest.raises(SourceError, match="Unsupported file type"):
        validate_upload("evil.exe", b"x", s)
    with pytest.raises(SourceError, match="empty"):
        validate_upload("a.txt", b"", s)
    small = s.model_copy(update={"MAX_UPLOAD_MB": 1})
    with pytest.raises(SourceError, match="1 MB"):
        validate_upload("a.txt", b"x" * (1024 * 1024 + 1), small)
    assert validate_upload("a.txt", b"x" * (1024 * 1024), small) == "a.txt"


def test_connector_types() -> None:
    assert set(CONNECTOR_TYPES) == {"crm", "erp", "gdrive", "sharepoint", "email", "contracts", "vendors", "custom"}


def test_create_document_source(ctx: AppContext, db: Session) -> None:
    src = create_document_source(db, ctx.settings, ctx.embedder, "../x/notes.txt", TEXT, user_id=None)
    assert (src.name, src.kind, src.status) == ("notes.txt", "document", "ready")
    assert src.last_sync_at is not None
    stored = upload_dir_for(ctx.settings, src) / "notes.txt"
    assert stored.read_bytes() == TEXT
    (doc,) = src.documents
    assert doc.chunk_count == len(src.chunks) > 0
    assert doc.storage_path == str(stored)
    assert [c.position for c in sorted(src.chunks, key=lambda c: c.position)] == list(range(doc.chunk_count))
    assert all(c.embedding_model == "hashed-256" and c.embedding_dim == 256 for c in src.chunks)
    named = create_document_source(db, ctx.settings, ctx.embedder, "b.txt", TEXT, name="  Board notes ")
    assert named.name == "Board notes"


def test_create_document_source_failure_cleans_up(ctx: AppContext, db: Session) -> None:
    with pytest.raises(IngestError, match="no extractable text"):
        create_document_source(db, ctx.settings, ctx.embedder, "blank.txt", b"   \n  ")
    assert db.query(models.Source).count() == 0
    assert not any(ctx.settings.upload_dir.glob("*/blank.txt"))


def test_store_document_replaces_chunks(ctx: AppContext, db: Session) -> None:
    src = models.Source(name="s")
    db.add(src)
    db.flush()
    d1 = store_document(db, ctx.settings, ctx.embedder, src, "a.txt", TEXT)
    db.commit()
    first = d1.chunk_count
    d2 = store_document(db, ctx.settings, ctx.embedder, src, "a.txt", b"Short replacement text for the account.")
    db.commit()
    assert d2.id == d1.id
    assert d2.chunk_count == 1 < first
    assert db.query(models.Chunk).filter(models.Chunk.document_id == d1.id).count() == 1
    assert d2.size_bytes == len(b"Short replacement text for the account.")


class BadEmbedder:
    model_id = "bad"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0]]  # always one vector


def test_store_document_vector_count_mismatch(ctx: AppContext, db: Session) -> None:
    src = models.Source(name="s")
    db.add(src)
    db.flush()
    with pytest.raises(IngestError, match="wrong number of vectors"):
        store_document(db, ctx.settings, BadEmbedder(), src, "a.txt", TEXT)


def test_create_connector_and_stage(ctx: AppContext, db: Session) -> None:
    with pytest.raises(SourceError, match="Unknown connector type"):
        create_connector(db, ctx.settings, "X", "telepathy")
    src = create_connector(db, ctx.settings, " Sales  CRM ", " CRM ")
    assert (src.name, src.kind, src.connector_type, src.status) == ("Sales CRM", "connector", "crm", "ready")
    sdir = staging_dir_for(ctx.settings, src)
    assert sdir == ctx.settings.staging_dir / f"{src.id}-sales-crm"
    assert sdir.is_dir()
    dest = stage_connector_file(ctx.settings, src, "../accounts.txt", TEXT)
    assert dest == sdir / "accounts.txt" and dest.read_bytes() == TEXT
    doc_src = create_document_source(db, ctx.settings, ctx.embedder, "d.txt", TEXT)
    with pytest.raises(SourceError, match="only be staged on connectors"):
        stage_connector_file(ctx.settings, doc_src, "a.txt", TEXT)
    with pytest.raises(SourceError, match="Only connectors can be synced"):
        sync_connector(db, ctx.settings, ctx.embedder, doc_src)


def test_sync_is_incremental(ctx: AppContext, db: Session) -> None:
    src = create_connector(db, ctx.settings, "CRM", "crm")
    sdir = staging_dir_for(ctx.settings, src)
    empty = sync_connector(db, ctx.settings, ctx.embedder, src)
    assert empty == SyncResult()
    assert src.status == "ready" and src.last_sync_at is not None

    (sdir / "a.txt").write_bytes(TEXT)
    (sdir / "b.md").write_bytes(b"# Accounts\nGlobex is a customer with ARR of $170,000 this year.")
    (sdir / "ignored.exe").write_bytes(b"MZ")
    (sdir / "subdir").mkdir()
    r1 = sync_connector(db, ctx.settings, ctx.embedder, src)
    assert (r1.added, r1.updated, r1.unchanged, r1.removed, r1.errors) == (2, 0, 0, 0, [])
    assert r1.chunks_indexed == sum(d.chunk_count for d in src.documents) > 0

    r2 = sync_connector(db, ctx.settings, ctx.embedder, src)
    assert (r2.added, r2.updated, r2.unchanged, r2.removed, r2.chunks_indexed) == (0, 0, 2, 0, 0)

    (sdir / "a.txt").write_bytes(b"Wonka churned in October and is no longer a customer of ours.")
    (sdir / "b.md").unlink()
    r3 = sync_connector(db, ctx.settings, ctx.embedder, src)
    assert (r3.added, r3.updated, r3.unchanged, r3.removed) == (0, 1, 0, 1)
    assert r3.chunks_indexed == 1
    db.refresh(src)
    assert [d.filename for d in src.documents] == ["a.txt"]
    assert db.query(models.Chunk).filter(models.Chunk.source_id == src.id).count() == 1


def test_sync_reports_bad_files_and_continues(ctx: AppContext, db: Session) -> None:
    src = create_connector(db, ctx.settings, "Drive", "gdrive")
    sdir = staging_dir_for(ctx.settings, src)
    (sdir / "bad.json").write_bytes(b"{nope")
    (sdir / "empty.txt").write_bytes(b"   ")
    (sdir / "good.txt").write_bytes(TEXT)
    r = sync_connector(db, ctx.settings, ctx.embedder, src)
    assert r.added == 1
    assert len(r.errors) == 2
    assert r.errors[0].startswith("bad.json: Could not read .json file")
    assert r.errors[1] == "empty.txt: empty.txt: no extractable text"
    assert src.status == "error"
    assert src.error == "; ".join(r.errors)
    (sdir / "bad.json").unlink()
    (sdir / "empty.txt").unlink()
    r2 = sync_connector(db, ctx.settings, ctx.embedder, src)
    assert r2.errors == [] and src.status == "ready" and src.error is None


def test_sync_size_limit(ctx: AppContext, db: Session) -> None:
    small = ctx.settings.model_copy(update={"MAX_UPLOAD_MB": 1})
    src = create_connector(db, small, "Big", "custom")
    (staging_dir_for(small, src) / "big.txt").write_bytes(b"a " * (600 * 1024))
    r = sync_connector(db, small, ctx.embedder, src)
    assert r.errors == ["big.txt: exceeds the 1 MB limit"]


class FlakyEmbedder(HashedEmbedder):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if any("FAIL" in t for t in texts):
            raise LLMError("provider down")
        return super().embed(texts)


def test_sync_provider_error_is_reported(ctx: AppContext, db: Session) -> None:
    src = create_connector(db, ctx.settings, "E", "email")
    (staging_dir_for(ctx.settings, src) / "x.txt").write_bytes(b"FAIL this document please now.")
    r = sync_connector(db, ctx.settings, FlakyEmbedder(), src)
    assert r.errors == ["x.txt: provider down"]


def test_sync_all_connectors(ctx: AppContext, db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    a = create_connector(db, ctx.settings, "A", "crm")
    b = create_connector(db, ctx.settings, "B", "erp")
    create_document_source(db, ctx.settings, ctx.embedder, "doc.txt", TEXT)
    (staging_dir_for(ctx.settings, a) / "a.txt").write_bytes(TEXT)
    results = sync_all_connectors(db, ctx.settings, ctx.embedder)
    assert set(results) == {a.id, b.id}
    assert results[a.id].added == 1 and results[b.id].added == 0

    import enterprise_brain.sources as mod

    def boom(*args: object, **kwargs: object) -> SyncResult:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(mod, "sync_connector", boom)
    crashed = sync_all_connectors(db, ctx.settings, ctx.embedder)
    assert crashed[a.id].errors == ["disk on fire"]


def test_delete_source_removes_rows_and_files(ctx: AppContext, db: Session) -> None:
    doc_src = create_document_source(db, ctx.settings, ctx.embedder, "d.txt", TEXT)
    conn = create_connector(db, ctx.settings, "C", "crm")
    stage_connector_file(ctx.settings, conn, "c.txt", TEXT)
    sync_connector(db, ctx.settings, ctx.embedder, conn)
    up_dir, st_dir = upload_dir_for(ctx.settings, doc_src), staging_dir_for(ctx.settings, conn)
    assert up_dir.exists() and st_dir.exists()
    delete_source(db, ctx.settings, doc_src)
    delete_source(db, ctx.settings, conn)
    assert not up_dir.exists() and not st_dir.exists()
    assert db.query(models.Chunk).count() == 0
    assert db.query(models.Document).count() == 0
    assert db.query(models.Source).count() == 0


def test_reindex_all(ctx: AppContext, db: Session) -> None:
    create_document_source(db, ctx.settings, ctx.embedder, "d.txt", TEXT)
    create_document_source(db, ctx.settings, ctx.embedder, "e.txt", TEXT)
    total = db.query(models.Chunk).count()
    n = reindex_all(db, ctx.settings, HashedEmbedder(64), batch=1)
    assert n == total
    assert {c.embedding_model for c in db.query(models.Chunk).all()} == {"hashed-64"}
    assert {c.embedding_dim for c in db.query(models.Chunk).all()} == {64}


def test_staging_dir_override(settings_factory: object, tmp_path: Path) -> None:
    s = settings_factory(STAGING_DIR=tmp_path / "exports")  # type: ignore[operator]
    src = models.Source(id=7, name="My CRM")
    assert staging_dir_for(s, src) == tmp_path / "exports" / "7-my-crm"
