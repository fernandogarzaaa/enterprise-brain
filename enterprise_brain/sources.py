"""Knowledge-source operations: document uploads, connectors and incremental sync."""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from enterprise_brain import models
from enterprise_brain.config import Settings
from enterprise_brain.ingest import ALLOWED_SUFFIXES, IngestError, safe_filename, sha256_bytes, store_document
from enterprise_brain.llm import Embedder, LLMError
from enterprise_brain.models import utcnow

log = logging.getLogger(__name__)

CONNECTOR_TYPES: dict[str, str] = {
    "crm": "CRM",
    "erp": "ERP",
    "gdrive": "Google Drive",
    "sharepoint": "SharePoint",
    "email": "Email",
    "contracts": "Contracts",
    "vendors": "Vendors",
    "custom": "Custom",
}
MAX_NAME_LENGTH = 255


class SourceError(ValueError):
    """Invalid input for a source operation (maps to HTTP 400)."""


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "source"


def staging_dir_for(settings: Settings, source: models.Source) -> Path:
    """Each connector owns ``<staging>/<id>-<slug>/``; ids keep same-named connectors apart."""
    return settings.staging_dir / f"{source.id}-{slugify(source.name)}"


def upload_dir_for(settings: Settings, source: models.Source) -> Path:
    return settings.upload_dir / str(source.id)


def validate_name(name: str) -> str:
    name = " ".join(name.split())
    if not name:
        raise SourceError("Name is required")
    if len(name) > MAX_NAME_LENGTH:
        raise SourceError(f"Name must be at most {MAX_NAME_LENGTH} characters")
    return name


def validate_upload(filename: str, data: bytes, settings: Settings) -> str:
    clean = safe_filename(filename)
    suffix = Path(clean).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise SourceError(f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}")
    if not data:
        raise SourceError("File is empty")
    if len(data) > settings.max_upload_bytes:
        raise SourceError(f"File exceeds the {settings.MAX_UPLOAD_MB} MB upload limit")
    return clean


def create_document_source(
    db: Session,
    settings: Settings,
    embedder: Embedder,
    filename: str,
    data: bytes,
    user_id: int | None = None,
    name: str | None = None,
) -> models.Source:
    """Store an uploaded file as a new document source and index it (commits)."""
    clean = validate_upload(filename, data, settings)
    source = models.Source(name=validate_name(name or clean), kind="document", status="ready", created_by_id=user_id)
    db.add(source)
    db.flush()
    dest_dir = upload_dir_for(settings, source)
    dest = dest_dir / clean
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        store_document(db, settings, embedder, source, clean, data, storage_path=str(dest))
        source.last_sync_at = utcnow()
        db.commit()
    except Exception:
        db.rollback()
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise
    return source


def create_connector(
    db: Session, settings: Settings, name: str, connector_type: str, user_id: int | None = None
) -> models.Source:
    ctype = connector_type.strip().lower()
    if ctype not in CONNECTOR_TYPES:
        raise SourceError(f"Unknown connector type. Choose from: {', '.join(CONNECTOR_TYPES)}")
    source = models.Source(
        name=validate_name(name), kind="connector", connector_type=ctype, status="ready", created_by_id=user_id
    )
    db.add(source)
    db.commit()
    staging_dir_for(settings, source).mkdir(parents=True, exist_ok=True)
    return source


def stage_connector_file(settings: Settings, source: models.Source, filename: str, data: bytes) -> Path:
    """Write an uploaded file into the connector's staging folder (picked up by sync)."""
    if source.kind != "connector":
        raise SourceError("Files can only be staged on connectors")
    clean = validate_upload(filename, data, settings)
    sdir = staging_dir_for(settings, source)
    sdir.mkdir(parents=True, exist_ok=True)
    dest = sdir / clean
    dest.write_bytes(data)
    return dest


@dataclass
class SyncResult:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks_indexed: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sync_connector(db: Session, settings: Settings, embedder: Embedder, source: models.Source) -> SyncResult:
    """Incrementally import the connector's staging folder.

    New files are indexed, changed files (by SHA-256) re-indexed, unchanged files
    skipped, and documents whose file disappeared are removed with their chunks.
    A failing file is reported in ``errors`` without aborting the others.
    """
    if source.kind != "connector":
        raise SourceError("Only connectors can be synced")
    sdir = staging_dir_for(settings, source)
    sdir.mkdir(parents=True, exist_ok=True)
    source.status = "syncing"
    db.commit()

    result = SyncResult()
    existing = {d.filename: d for d in db.query(models.Document).filter(models.Document.source_id == source.id).all()}
    seen: set[str] = set()
    for path in sorted(sdir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        name = path.name
        seen.add(name)
        try:
            data = path.read_bytes()
            if len(data) > settings.max_upload_bytes:
                raise IngestError(f"exceeds the {settings.MAX_UPLOAD_MB} MB limit")
            prior = existing.get(name)
            if prior is not None and prior.sha256 == sha256_bytes(data):
                result.unchanged += 1
                continue
            doc = store_document(db, settings, embedder, source, name, data, storage_path=str(path))
            db.commit()
            result.chunks_indexed += doc.chunk_count
            if prior is None:
                result.added += 1
            else:
                result.updated += 1
        except (IngestError, LLMError, OSError) as exc:
            db.rollback()
            result.errors.append(f"{name}: {exc}")
            log.warning("sync file failed", extra={"source_id": source.id, "file": name, "error": str(exc)})

    for name, doc in existing.items():
        if name not in seen:
            db.delete(doc)
            result.removed += 1
    source.status = "error" if result.errors else "ready"
    source.error = "; ".join(result.errors)[:2000] if result.errors else None
    source.last_sync_at = utcnow()
    db.commit()
    log.info("connector synced", extra={"source_id": source.id, **result.as_dict()})
    return result


def sync_all_connectors(db: Session, settings: Settings, embedder: Embedder) -> dict[int, SyncResult]:
    out: dict[int, SyncResult] = {}
    for source in db.query(models.Source).filter(models.Source.kind == "connector").all():
        try:
            out[source.id] = sync_connector(db, settings, embedder, source)
        except Exception as exc:  # keep syncing the rest
            db.rollback()
            log.exception("connector sync crashed", extra={"source_id": source.id})
            out[source.id] = SyncResult(errors=[str(exc)])
    return out


def delete_source(db: Session, settings: Settings, source: models.Source) -> None:
    """Delete a source, its documents and chunks, and its files on disk."""
    dirs = [upload_dir_for(settings, source)]
    if source.kind == "connector":
        dirs.append(staging_dir_for(settings, source))
    db.delete(source)
    db.commit()
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
