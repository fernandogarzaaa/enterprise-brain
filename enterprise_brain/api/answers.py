"""Ask questions, list past queries, and manage saved (starred) answers."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from enterprise_brain import models
from enterprise_brain.ask import ask
from enterprise_brain.deps import AppContext, get_ctx, get_current_user, get_db

router = APIRouter(tags=["ask"])


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    source_ids: list[int] | None = Field(default=None, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=50)


class SaveIn(BaseModel):
    note: str = Field(default="", max_length=500)


@router.post("/api/ask")
def api_ask(
    payload: AskIn,
    ctx: AppContext = Depends(get_ctx),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    result = ask(
        db,
        ctx.settings,
        ctx.embedder,
        ctx.chat,
        payload.question,
        user_id=user.id,
        source_ids=payload.source_ids,
        top_k=payload.top_k,
    )
    return result.as_dict()


def query_dict(q: models.Query, full: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": q.id,
        "question": q.question,
        "answer": q.answer if full else q.answer[:220],
        "mode": q.mode,
        "latency_ms": q.latency_ms,
        "user_id": q.user_id,
        "created_at": q.created_at.isoformat() if q.created_at else None,
    }
    if full:
        out["citations"] = json.loads(q.citations_json or "[]")
    return out


@router.get("/api/queries")
def api_queries(
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    everyone: bool = Query(default=False, description="Admins: include every member's queries"),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    q = db.query(models.Query)
    if not (everyone and user.is_admin):
        q = q.filter(models.Query.user_id == user.id)
    rows = q.order_by(models.Query.id.desc()).offset(offset).limit(limit).all()
    return [query_dict(r) for r in rows]


def visible_query(db: Session, user: models.User, query_id: int) -> models.Query:
    q = db.get(models.Query, query_id)
    if q is None or (q.user_id != user.id and not user.is_admin):
        raise HTTPException(status_code=404, detail="Query not found")
    return q


@router.get("/api/queries/{query_id}")
def api_query(
    query_id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict[str, Any]:
    return query_dict(visible_query(db, user, query_id), full=True)


@router.post("/api/queries/{query_id}/save")
def api_save(
    query_id: int,
    payload: SaveIn | None = None,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    q = visible_query(db, user, query_id)
    saved = db.query(models.SavedAnswer).filter(models.SavedAnswer.query_id == q.id).one_or_none()
    if saved is None:
        saved = models.SavedAnswer(query_id=q.id, user_id=user.id, note=payload.note if payload else "")
        db.add(saved)
        db.commit()
    elif payload is not None:
        saved.note = payload.note
        db.commit()
    return {"saved_id": saved.id, "query_id": q.id, "note": saved.note}


def saved_rows(db: Session) -> list[dict[str, Any]]:
    rows = (
        db.query(models.SavedAnswer, models.Query)
        .join(models.Query, models.SavedAnswer.query_id == models.Query.id)
        .order_by(models.SavedAnswer.id.desc())
        .all()
    )
    return [
        {
            "id": s.id,
            "query_id": q.id,
            "note": s.note,
            "question": q.question,
            "answer": q.answer,
            "citations": json.loads(q.citations_json or "[]"),
            "saved_by": s.user_id,
            "created_at": s.created_at,
        }
        for s, q in rows
    ]


@router.get("/api/saved")
def api_saved(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    out = saved_rows(db)
    for row in out:
        row["created_at"] = row["created_at"].isoformat() if row["created_at"] else None
    return out


@router.delete("/api/saved/{saved_id}")
def api_unsave(
    saved_id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict[str, Any]:
    s = db.get(models.SavedAnswer, saved_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Saved answer not found")
    if s.user_id not in (user.id, None) and not user.is_admin:
        raise HTTPException(status_code=403, detail="Only the member who saved it or an admin can remove it")
    db.delete(s)
    db.commit()
    return {"deleted": saved_id}
