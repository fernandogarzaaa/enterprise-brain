"""Sign-in, sign-out, the current user, password changes and personal API keys."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from enterprise_brain import models, users
from enterprise_brain.deps import COOKIE_NAME, AppContext, get_ctx, get_current_user, get_db
from enterprise_brain.security import create_access_token

router = APIRouter(tags=["auth"])


class LoginIn(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=256)


class PasswordChangeIn(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


class ApiKeyIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)


def client_key(request: Request, email: str) -> str:
    host = request.client.host if request.client else "unknown"
    return f"{host}|{email.strip().lower()}"


def make_token(ctx: AppContext, user: models.User) -> str:
    s = ctx.settings
    return create_access_token(
        user.id, user.token_version, s.SECRET_KEY.get_secret_value(), s.ACCESS_TOKEN_EXPIRE_MINUTES
    )


def set_session_cookie(ctx: AppContext, response: Response, token: str) -> None:
    s = ctx.settings
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        secure=s.cookie_secure,
        samesite="lax",
        max_age=s.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )


def attempt_login(ctx: AppContext, db: Session, request: Request, email: str, password: str) -> models.User | int:
    """Return the user, or the retry-after seconds when rate limited (0 = bad credentials)."""
    key = client_key(request, email)
    limiter = ctx.login_limiter
    if limiter.is_limited(key):
        return limiter.retry_after(key)
    user = users.authenticate(db, email, password)
    if user is None:
        limiter.hit(key)
        return 0
    limiter.reset(key)
    return user


@router.post("/api/auth/login")
def login_api(
    payload: LoginIn, request: Request, ctx: AppContext = Depends(get_ctx), db: Session = Depends(get_db)
) -> Response:
    result = attempt_login(ctx, db, request, payload.email, payload.password)
    if isinstance(result, int):
        if result:
            return JSONResponse(
                {"detail": "Too many login attempts. Try again later."},
                status_code=429,
                headers={"Retry-After": str(result)},
            )
        return JSONResponse({"detail": "Invalid credentials"}, status_code=401)
    token = make_token(ctx, result)
    resp = JSONResponse(
        {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": ctx.settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        }
    )
    set_session_cookie(ctx, resp, token)
    return resp


@router.post("/login", include_in_schema=False)
def login_form(
    request: Request,
    email: str = Form(..., max_length=255),
    password: str = Form(..., max_length=256),
    ctx: AppContext = Depends(get_ctx),
    db: Session = Depends(get_db),
) -> Response:
    result = attempt_login(ctx, db, request, email, password)
    if isinstance(result, int):
        return RedirectResponse("/login?error=" + ("locked" if result else "1"), status_code=303)
    resp = RedirectResponse("/", status_code=303)
    set_session_cookie(ctx, resp, make_token(ctx, result))
    return resp


@router.post("/api/auth/logout")
def logout_api() -> Response:
    resp = JSONResponse({"detail": "Signed out"})
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


@router.get("/logout", include_in_schema=False)
def logout_page() -> Response:
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


def user_dict(u: models.User) -> dict[str, Any]:
    return {
        "id": u.id,
        "email": u.email,
        "name": u.name,
        "role": u.role,
        "is_active": u.is_active,
        "created_at": u.created_at.isoformat() if u.created_at else None,
        "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
    }


@router.get("/api/auth/me")
def me(user: models.User = Depends(get_current_user)) -> dict[str, Any]:
    return user_dict(user)


@router.post("/api/auth/password")
def change_password(
    payload: PasswordChangeIn,
    ctx: AppContext = Depends(get_ctx),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    users.change_password(db, ctx.settings, user, payload.current_password, payload.new_password)
    token = make_token(ctx, user)
    resp = JSONResponse({"detail": "Password changed. Other sessions were signed out.", "access_token": token})
    set_session_cookie(ctx, resp, token)  # keep this session alive
    return resp


def key_dict(k: models.ApiKey) -> dict[str, Any]:
    return {
        "id": k.id,
        "name": k.name,
        "prefix": f"eb_{k.prefix}_...",
        "created_at": k.created_at.isoformat() if k.created_at else None,
        "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
        "revoked": k.revoked_at is not None,
    }


@router.get("/api/auth/api-keys")
def list_keys(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    keys = db.query(models.ApiKey).filter(models.ApiKey.user_id == user.id).order_by(models.ApiKey.id.desc()).all()
    return [key_dict(k) for k in keys]


@router.post("/api/auth/api-keys", status_code=201)
def create_key(
    payload: ApiKeyIn, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict[str, Any]:
    created = users.create_api_key(db, user, payload.name)
    return {**key_dict(created.key), "key": created.plaintext}


@router.delete("/api/auth/api-keys/{key_id}")
def revoke_key(
    key_id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict[str, Any]:
    users.revoke_api_key(db, user, key_id)
    return {"revoked": key_id}
