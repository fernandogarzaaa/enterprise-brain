"""Health, readiness, version and usage insights."""

from __future__ import annotations

from collections import Counter
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from enterprise_brain import __version__, models
from enterprise_brain.ask import terms
from enterprise_brain.db import check_database
from enterprise_brain.deps import AppContext, get_ctx, get_current_user, get_db

router = APIRouter(tags=["system"])


@router.get("/api/health")
def health(ctx: AppContext = Depends(get_ctx)) -> dict[str, str]:
    """Liveness: the process is up."""
    return {"status": "ok", "app": ctx.settings.APP_NAME, "version": __version__}


@router.get("/api/health/ready")
def ready(db: Session = Depends(get_db)) -> Response:
    """Readiness: the database answers."""
    try:
        check_database(db)
    except Exception as exc:
        return JSONResponse({"status": "unavailable", "database": str(exc)[:200]}, status_code=503)
    return JSONResponse({"status": "ok", "database": "ok"})


@router.get("/api/version")
def version() -> dict[str, str]:
    return {"version": __version__}


def compute_insights(db: Session, top: int = 10) -> dict[str, Any]:
    per_source = (
        db.query(models.Source.name, func.count(models.Chunk.id))
        .outerjoin(models.Chunk, models.Chunk.source_id == models.Source.id)
        .group_by(models.Source.id, models.Source.name)
        .order_by(func.count(models.Chunk.id).desc(), models.Source.name)
        .all()
    )
    counter: Counter[str] = Counter()
    for (q,) in db.query(models.Query.question).order_by(models.Query.id.desc()).limit(5000).all():
        counter.update(set(terms(q)))
    modes = {
        str(k): int(v)
        for k, v in db.query(models.Query.mode, func.count(models.Query.id)).group_by(models.Query.mode).all()
    }
    avg_latency = db.query(func.avg(models.Query.latency_ms)).scalar()
    return {
        "total_queries": int(db.query(func.count(models.Query.id)).scalar() or 0),
        "total_sources": int(db.query(func.count(models.Source.id)).scalar() or 0),
        "total_chunks": int(db.query(func.count(models.Chunk.id)).scalar() or 0),
        "total_members": int(db.query(func.count(models.User.id)).scalar() or 0),
        "saved_answers": int(db.query(func.count(models.SavedAnswer.id)).scalar() or 0),
        "unanswered_queries": modes.get("none", 0),
        "answer_modes": modes,
        "avg_latency_ms": round(float(avg_latency), 1) if avg_latency is not None else None,
        "chunks_per_source": [{"source": n, "chunks": int(c)} for n, c in per_source],
        "top_terms": [{"term": t, "count": n} for t, n in counter.most_common(top)],
    }


@router.get("/api/insights")
def api_insights(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    return compute_insights(db)
