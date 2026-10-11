"""Team management (any member can list; only admins can change)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from enterprise_brain import models, users
from enterprise_brain.api.auth import user_dict
from enterprise_brain.deps import AppContext, get_ctx, get_current_user, get_db, require_admin
from enterprise_brain.security import generate_temporary_password

router = APIRouter(tags=["team"])


class InviteIn(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    name: str = Field(default="", max_length=255)
    role: str = Field(default="member")


class MemberUpdateIn(BaseModel):
    role: str | None = None
    is_active: bool | None = None
    name: str | None = Field(default=None, max_length=255)


def get_member(db: Session, user_id: int) -> models.User:
    m = db.get(models.User, user_id)
    if m is None:
        raise HTTPException(status_code=404, detail="Member not found")
    return m


@router.get("/api/team")
def api_team(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return [user_dict(u) for u in db.query(models.User).order_by(models.User.created_at, models.User.id).all()]


@router.post("/api/team/invite", status_code=201)
def api_invite(
    payload: InviteIn,
    ctx: AppContext = Depends(get_ctx),
    admin: models.User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    temp = generate_temporary_password()
    member = users.create_user(
        db, ctx.settings, payload.email, temp, name=payload.name, role=payload.role, enforce_policy=False
    )
    # Shown once to the inviter, who shares it with the member over a trusted channel.
    return {**user_dict(member), "temporary_password": temp}


@router.patch("/api/team/{user_id}")
def api_update_member(
    user_id: int,
    payload: MemberUpdateIn,
    admin: models.User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    member = users.update_member(
        db, admin, get_member(db, user_id), role=payload.role, is_active=payload.is_active, name=payload.name
    )
    return user_dict(member)


@router.post("/api/team/{user_id}/reset-password")
def api_reset_password(
    user_id: int,
    ctx: AppContext = Depends(get_ctx),
    admin: models.User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    member = get_member(db, user_id)
    temp = users.reset_password(db, ctx.settings, member)
    return {"id": member.id, "email": member.email, "temporary_password": temp}


@router.delete("/api/team/{user_id}")
def api_delete_member(
    user_id: int, admin: models.User = Depends(require_admin), db: Session = Depends(get_db)
) -> dict[str, Any]:
    users.delete_member(db, admin, get_member(db, user_id))
    return {"deleted": user_id}
