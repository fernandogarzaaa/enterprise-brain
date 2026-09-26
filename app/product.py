"""Enterprise Brain product router: Ask UI, knowledge sources, insights, team."""

from __future__ import annotations

import json
import os
import re
import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import models
from app.ask import ask as ask_question
from app.core.config import settings
from app.core.db import get_db
from app.core.deps import get_current_user, page_or_login, require_admin
from app.core.security import get_password_hash
from app.ingest import ALLOWED_SUFFIXES, ingest_file
from app.models import utcnow
from app.vectorstore import get_store

router = APIRouter()
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("data/uploads")
STAGING_ROOT = Path(os.environ.get("EBRAIN_STAGING", "data/staging"))

CONNECTOR_TYPES = ["crm", "erp", "gdrive", "sharepoint", "email", "contracts", "vendors", "custom"]
CONNECTOR_LABELS = {
    "crm": "CRM", "erp": "ERP", "gdrive": "Google Drive", "sharepoint": "SharePoint",
    "email": "Email", "contracts": "Contracts", "vendors": "Vendors", "custom": "Custom",
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _base_ctx(request: Request, user) -> dict:
    return {
        "request": request,
        "app_name": settings.APP_NAME,
        "user": user,
    }


def _page_or_redirect(request: Request, db: Session):
    u = page_or_login(request, db)
    return u


# ---------------------------------------------------------------- ask


@router.get("/", response_class=HTMLResponse)
def ask_page(request: Request, db: Session = Depends(get_db)):
    u = _page_or_redirect(request, db)
    if isinstance(u, RedirectResponse):
        return u
    sources = db.query(models.Source).order_by(models.Source.name).all()
    chunk_counts = _chunk_counts(db)
    recent = db.query(models.Query).order_by(models.Query.id.desc()).limit(10).all()
    ctx = _base_ctx(request, u)
    ctx.update(
        sources=sources,
        chunk_counts=chunk_counts,
        recent_queries=recent,
        try_prompts=[
            "What is the status of the Acme Corp contract?",
            "Who are our approved cloud vendors?",
            "What is the invoice approval policy?",
            "What was Q3 2026 revenue by client?",
            "Which invoices are overdue?",
        ],
    )
    return templates.TemplateResponse(request, "ask.html", ctx)


class AskIn(BaseModel):
    question: str
    source_ids: list[int] | None = None


@router.post("/api/ask")
def api_ask(payload: AskIn, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not payload.question.strip():
        return JSONResponse({"detail": "Question is required"}, status_code=400)
    return ask_question(db, payload.question, source_ids=payload.source_ids)


@router.get("/api/queries")
def api_recent_queries(
    limit: int = 10, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
):
    rows = db.query(models.Query).order_by(models.Query.id.desc()).limit(min(limit, 50)).all()
    return [
        {
            "id": r.id,
            "question": r.question,
            "answer": r.answer[:220],
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@router.post("/api/queries/{query_id}/save")
def api_save_query(
    query_id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
):
    q = db.query(models.Query).filter(models.Query.id == query_id).first()
    if not q:
        return JSONResponse({"detail": "Query not found"}, status_code=404)
    saved = db.query(models.SavedAnswer).filter(models.SavedAnswer.query_id == query_id).first()
    if not saved:
        saved = models.SavedAnswer(query_id=query_id)
        db.add(saved)
        db.commit()
        db.refresh(saved)
    return {"saved_id": saved.id, "query_id": query_id}


# ---------------------------------------------------------------- sources


def _chunk_counts(db: Session) -> dict[int, int]:
    from sqlalchemy import func

    rows = (
        db.query(models.Chunk.source_id, func.count(models.Chunk.id))
        .group_by(models.Chunk.source_id)
        .all()
    )
    return {sid: n for sid, n in rows}


@router.get("/sources", response_class=HTMLResponse)
def sources_page(request: Request, db: Session = Depends(get_db)):
    u = _page_or_redirect(request, db)
    if isinstance(u, RedirectResponse):
        return u
    sources = db.query(models.Source).order_by(models.Source.created_at.desc()).all()
    ctx = _base_ctx(request, u)
    ctx.update(
        sources=sources,
        chunk_counts=_chunk_counts(db),
        connector_types=CONNECTOR_TYPES,
        connector_labels=CONNECTOR_LABELS,
        staging_root=str(STAGING_ROOT),
    )
    return templates.TemplateResponse(request, "sources.html", ctx)


@router.post("/api/sources/upload")
def api_upload(
    file: UploadFile,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return JSONResponse(
            {"detail": f"Unsupported file type. Allowed: {sorted(ALLOWED_SUFFIXES)}"},
            status_code=400,
        )
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{secrets.token_hex(8)}{suffix}"
    with dest.open("wb") as f:
        f.write(file.file.read())
    src = models.Source(name=file.filename or dest.name, kind="document", status="ready")
    db.add(src)
    db.flush()
    try:
        n = ingest_file(db, src, dest)
    except Exception as exc:
        db.rollback()
        dest.unlink(missing_ok=True)
        return JSONResponse({"detail": f"Ingest failed: {exc}"}, status_code=422)
    src.last_sync_at = utcnow()
    db.commit()
    return {"source_id": src.id, "name": src.name, "chunks": n}


class ConnectorIn(BaseModel):
    name: str
    type: str


@router.post("/api/connectors")
def api_create_connector(
    payload: ConnectorIn,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ctype = payload.type.strip().lower()
    if ctype not in CONNECTOR_TYPES:
        return JSONResponse(
            {"detail": f"Unknown connector type. Choose from {CONNECTOR_TYPES}"}, status_code=400
        )
    name = payload.name.strip()
    if not name:
        return JSONResponse({"detail": "Name is required"}, status_code=400)
    src = models.Source(name=name, kind="connector", connector_type=ctype, status="ready")
    db.add(src)
    db.commit()
    db.refresh(src)
    sdir = STAGING_ROOT / _slug(name)
    sdir.mkdir(parents=True, exist_ok=True)
    return {
        "id": src.id,
        "name": src.name,
        "type": ctype,
        "staging_dir": str(sdir),
        "hint": f"Drop files into {sdir} then press Sync.",
    }


def sync_connector(db: Session, source: models.Source) -> int:
    """Import every supported file staged under data/staging/<slug>/ into the
    connector's Source row. Returns chunks added. Honest: only staged files."""
    sdir = STAGING_ROOT / _slug(source.name)
    sdir.mkdir(parents=True, exist_ok=True)
    source.status = "syncing"
    db.commit()
    total = 0
    try:
        for path in sorted(sdir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            total += ingest_file(db, source, path)
        source.status = "ready"
        source.last_sync_at = utcnow()
        db.commit()
    except Exception:
        source.status = "error"
        db.commit()
        raise
    return total


@router.post("/api/connectors/{source_id}/sync")
def api_sync_connector(
    source_id: int,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    src = (
        db.query(models.Source)
        .filter(models.Source.id == source_id, models.Source.kind == "connector")
        .first()
    )
    if not src:
        return JSONResponse({"detail": "Connector not found"}, status_code=404)
    try:
        n = sync_connector(db, src)
    except Exception as exc:
        return JSONResponse({"detail": f"Sync failed: {exc}"}, status_code=500)
    sdir = STAGING_ROOT / _slug(src.name)
    return {
        "id": src.id,
        "chunks_added": n,
        "last_sync_at": src.last_sync_at.isoformat() if src.last_sync_at else None,
        "staging_dir": str(sdir),
        "note": (
            f"Imported {n} chunks from staged files."
            if n
            else f"No supported files found in {sdir} yet."
        ),
    }


@router.delete("/api/sources/{source_id}")
def api_delete_source(
    source_id: int,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    src = db.query(models.Source).filter(models.Source.id == source_id).first()
    if not src:
        return JSONResponse({"detail": "Source not found"}, status_code=404)
    db.query(models.Chunk).filter(models.Chunk.source_id == source_id).delete()
    db.delete(src)
    db.commit()
    return {"deleted": source_id}


# ---------------------------------------------------------------- insights


def _top_terms(db: Session, limit: int = 10) -> list[dict]:
    from collections import Counter

    from app.ask import _terms

    counter: Counter[str] = Counter()
    for (q,) in db.query(models.Query.question).all():
        counter.update(_terms(q))
    return [{"term": t, "count": n} for t, n in counter.most_common(limit)]


@router.get("/api/insights")
def api_insights(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    from sqlalchemy import func

    total_queries = db.query(func.count(models.Query.id)).scalar() or 0
    total_sources = db.query(func.count(models.Source.id)).scalar() or 0
    total_chunks = db.query(func.count(models.Chunk.id)).scalar() or 0
    per_source = (
        db.query(models.Source.name, func.count(models.Chunk.id))
        .outerjoin(models.Chunk, models.Chunk.source_id == models.Source.id)
        .group_by(models.Source.id, models.Source.name)
        .order_by(func.count(models.Chunk.id).desc())
        .all()
    )
    saved = db.query(func.count(models.SavedAnswer.id)).scalar() or 0
    return {
        "total_queries": total_queries,
        "total_sources": total_sources,
        "total_chunks": total_chunks,
        "saved_answers": saved,
        "chunks_per_source": [{"source": name, "chunks": n} for name, n in per_source],
        "top_terms": _top_terms(db),
    }


@router.get("/insights", response_class=HTMLResponse)
def insights_page(request: Request, db: Session = Depends(get_db)):
    u = _page_or_redirect(request, db)
    if isinstance(u, RedirectResponse):
        return u
    ctx = _base_ctx(request, u)
    ctx.update(insights=api_insights(user=u, db=db))
    return templates.TemplateResponse(request, "insights.html", ctx)


# ---------------------------------------------------------------- saved answers


@router.get("/saved", response_class=HTMLResponse)
def saved_page(request: Request, db: Session = Depends(get_db)):
    u = _page_or_redirect(request, db)
    if isinstance(u, RedirectResponse):
        return u
    rows = (
        db.query(models.SavedAnswer, models.Query)
        .join(models.Query, models.SavedAnswer.query_id == models.Query.id)
        .order_by(models.SavedAnswer.id.desc())
        .all()
    )
    saved = []
    for s, q in rows:
        saved.append(
            {
                "id": s.id,
                "note": s.note,
                "created_at": s.created_at,
                "question": q.question,
                "answer": q.answer,
                "citations": json.loads(q.citations_json or "[]"),
            }
        )
    ctx = _base_ctx(request, u)
    ctx.update(saved=saved)
    return templates.TemplateResponse(request, "saved.html", ctx)


@router.delete("/api/saved/{saved_id}")
def api_unsave(
    saved_id: int,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = db.query(models.SavedAnswer).filter(models.SavedAnswer.id == saved_id).first()
    if not s:
        return JSONResponse({"detail": "Not found"}, status_code=404)
    db.delete(s)
    db.commit()
    return {"deleted": saved_id}


# ---------------------------------------------------------------- team


@router.get("/team", response_class=HTMLResponse)
def team_page(request: Request, db: Session = Depends(get_db)):
    u = _page_or_redirect(request, db)
    if isinstance(u, RedirectResponse):
        return u
    members = db.query(models.User).order_by(models.User.created_at).all()
    ctx = _base_ctx(request, u)
    ctx.update(members=members, is_admin=(u.role == "admin"))
    return templates.TemplateResponse(request, "team.html", ctx)


class InviteIn(BaseModel):
    email: str
    name: str = ""


@router.post("/api/team/invite")
def api_invite(
    payload: InviteIn,
    admin: models.User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    email = payload.email.strip().lower()
    if not email or "@" not in email:
        return JSONResponse({"detail": "Valid email is required"}, status_code=400)
    if db.query(models.User).filter(models.User.email == email).first():
        return JSONResponse({"detail": "That email is already a member"}, status_code=409)
    temp_password = secrets.token_urlsafe(12)
    member = models.User(
        email=email,
        name=payload.name.strip() or email.split("@")[0],
        hashed_password=get_password_hash(temp_password),
        role="member",
    )
    db.add(member)
    db.commit()
    # The temporary password is shown once to the inviter; they share it with the member.
    return {"email": email, "role": "member", "temporary_password": temp_password}


# ---------------------------------------------------------------- settings


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, db: Session = Depends(get_db)):
    u = _page_or_redirect(request, db)
    if isinstance(u, RedirectResponse):
        return u
    db_url = settings.DATABASE_URL
    masked = db_url.split("@")[-1] if "@" in db_url else db_url
    ctx = _base_ctx(request, u)
    ctx.update(
        cfg={
            "app_name": settings.APP_NAME,
            "embeddings": settings.EMBEDDINGS,
            "llm_configured": bool(settings.LLM_BASE_URL and settings.LLM_API_KEY),
            "llm_model": settings.LLM_CHAT_MODEL,
            "vector_store": get_store().name,
            "database": masked,
            "uploads": str(UPLOAD_DIR),
            "staging": str(STAGING_ROOT),
        }
    )
    return templates.TemplateResponse(request, "settings.html", ctx)
