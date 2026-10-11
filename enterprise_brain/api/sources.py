"""Knowledge sources: uploads, connectors, staging and sync."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from enterprise_brain import models
from enterprise_brain.config import Settings
from enterprise_brain.deps import AppContext, get_ctx, get_current_user, get_db
from enterprise_brain.sources import (
    CONNECTOR_TYPES,
    SourceError,
    create_connector,
    create_document_source,
    delete_source,
    stage_connector_file,
    staging_dir_for,
    sync_connector,
)

router = APIRouter(tags=["sources"])


class ConnectorIn(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    type: str = Field(min_length=1, max_length=64)


def chunk_counts(db: Session) -> dict[int, int]:
    rows = db.query(models.Chunk.source_id, func.count(models.Chunk.id)).group_by(models.Chunk.source_id).all()
    return {int(sid): int(n) for sid, n in rows}


def source_dict(s: models.Source, chunks: int, settings: Settings) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": s.id,
        "name": s.name,
        "kind": s.kind,
        "connector_type": s.connector_type,
        "status": s.status,
        "error": s.error,
        "chunks": chunks,
        "documents": len(s.documents),
        "created_by_id": s.created_by_id,
        "last_sync_at": s.last_sync_at.isoformat() if s.last_sync_at else None,
        "created_at": s.created_at.isoformat() if s.created_at else None,
    }
    if s.kind == "connector":
        out["staging_dir"] = str(staging_dir_for(settings, s))
    return out


def get_source(db: Session, source_id: int) -> models.Source:
    s = db.get(models.Source, source_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return s


async def read_limited(file: UploadFile, limit: int) -> bytes:
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise SourceError(f"File exceeds the {limit // (1024 * 1024)} MB upload limit")
    return data


@router.get("/api/sources")
def api_sources(
    ctx: AppContext = Depends(get_ctx),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    counts = chunk_counts(db)
    rows = db.query(models.Source).order_by(models.Source.created_at.desc(), models.Source.id.desc()).all()
    return [source_dict(s, counts.get(s.id, 0), ctx.settings) for s in rows]


@router.get("/api/sources/{source_id}")
def api_source(
    source_id: int,
    ctx: AppContext = Depends(get_ctx),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    s = get_source(db, source_id)
    out = source_dict(s, chunk_counts(db).get(s.id, 0), ctx.settings)
    out["files"] = [
        {
            "id": d.id,
            "filename": d.filename,
            "size_bytes": d.size_bytes,
            "chunks": d.chunk_count,
            "sha256": d.sha256,
            "updated_at": d.updated_at.isoformat() if d.updated_at else None,
        }
        for d in s.documents
    ]
    return out


@router.post("/api/sources/upload", status_code=201)
async def api_upload(
    file: UploadFile,
    name: str | None = Form(default=None, max_length=255),
    ctx: AppContext = Depends(get_ctx),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    data = await read_limited(file, ctx.settings.max_upload_bytes)
    src = create_document_source(
        db, ctx.settings, ctx.embedder, file.filename or "upload", data, user_id=user.id, name=name
    )
    return {"source_id": src.id, "name": src.name, "chunks": chunk_counts(db).get(src.id, 0)}


@router.get("/api/connectors/types")
def api_connector_types(user: models.User = Depends(get_current_user)) -> dict[str, str]:
    return CONNECTOR_TYPES


@router.post("/api/connectors", status_code=201)
def api_create_connector(
    payload: ConnectorIn,
    ctx: AppContext = Depends(get_ctx),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    src = create_connector(db, ctx.settings, payload.name, payload.type, user_id=user.id)
    sdir = staging_dir_for(ctx.settings, src)
    return {
        "id": src.id,
        "name": src.name,
        "type": src.connector_type,
        "staging_dir": str(sdir),
        "hint": f"Upload files to this connector (or drop exports into {sdir}), then press Sync.",
    }


@router.post("/api/connectors/{source_id}/files", status_code=201)
async def api_stage_file(
    source_id: int,
    file: UploadFile,
    ctx: AppContext = Depends(get_ctx),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    src = get_source(db, source_id)
    data = await read_limited(file, ctx.settings.max_upload_bytes)
    dest = stage_connector_file(ctx.settings, src, file.filename or "upload", data)
    return {"source_id": src.id, "staged": dest.name, "hint": "Press Sync to index staged files."}


@router.post("/api/connectors/{source_id}/sync")
def api_sync(
    source_id: int,
    ctx: AppContext = Depends(get_ctx),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    src = get_source(db, source_id)
    if src.kind != "connector":
        raise HTTPException(status_code=404, detail="Connector not found")
    result = sync_connector(db, ctx.settings, ctx.embedder, src)
    changed = result.added + result.updated + result.removed
    if result.errors:
        note = f"Synced with {len(result.errors)} error(s): " + "; ".join(result.errors)
    elif changed:
        note = (
            f"Indexed {result.chunks_indexed} chunks: {result.added} new, {result.updated} updated, "
            f"{result.removed} removed, {result.unchanged} unchanged file(s)."
        )
    elif result.unchanged:
        note = f"Up to date: {result.unchanged} unchanged file(s)."
    else:
        note = f"No supported files staged in {staging_dir_for(ctx.settings, src)} yet."
    return {
        "id": src.id,
        "status": src.status,
        **result.as_dict(),
        "chunks_added": result.chunks_indexed,
        "last_sync_at": src.last_sync_at.isoformat() if src.last_sync_at else None,
        "note": note,
    }


@router.delete("/api/sources/{source_id}")
def api_delete_source(
    source_id: int,
    ctx: AppContext = Depends(get_ctx),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    src = get_source(db, source_id)
    if src.created_by_id not in (user.id, None) and not user.is_admin:
        raise HTTPException(status_code=403, detail="Only the creator or an admin can delete this source")
    delete_source(db, ctx.settings, src)
    return {"deleted": source_id}
