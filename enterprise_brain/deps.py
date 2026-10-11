"""Request-scoped dependencies: application context, DB sessions and authentication."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request, status
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session, sessionmaker

from enterprise_brain import models
from enterprise_brain.config import Settings
from enterprise_brain.llm import ChatProvider, Embedder
from enterprise_brain.security import RateLimiter, decode_access_token
from enterprise_brain.users import user_for_api_key

COOKIE_NAME = "eb_session"
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


@dataclass
class AppContext:
    settings: Settings
    sessions: sessionmaker[Session]
    embedder: Embedder
    chat: ChatProvider | None
    login_limiter: RateLimiter
    templates: Jinja2Templates


def get_ctx(request: Request) -> AppContext:
    ctx: AppContext = request.app.state.ctx
    return ctx


def get_db(ctx: AppContext = Depends(get_ctx)) -> Iterator[Session]:
    db = ctx.sessions()
    try:
        yield db
    finally:
        db.close()


def same_origin(request: Request) -> bool:
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        return True  # non-browser clients; browsers send Origin on fetch/form POSTs
    return urlsplit(origin).netloc == request.headers.get("host", "")


def resolve_user(request: Request, db: Session, settings: Settings) -> models.User | None:
    """Authenticate via API key, bearer token or session cookie (in that order)."""
    api_key = request.headers.get("x-api-key", "")
    auth = request.headers.get("authorization", "")
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if api_key or bearer.startswith("eb_"):
        request.state.auth_method = "api_key"
        return user_for_api_key(db, api_key or bearer)
    token, method = bearer, "bearer"
    if not token:
        token, method = request.cookies.get(COOKIE_NAME, ""), "cookie"
    if not token:
        return None
    claims = decode_access_token(token, settings.SECRET_KEY.get_secret_value())
    if claims is None:
        return None
    user = db.get(models.User, claims.user_id)
    if user is None or not user.is_active or user.token_version != claims.token_version:
        return None
    request.state.auth_method = method
    return user


def optional_user(
    request: Request, db: Session = Depends(get_db), ctx: AppContext = Depends(get_ctx)
) -> models.User | None:
    return resolve_user(request, db, ctx.settings)


def get_current_user(request: Request, user: models.User | None = Depends(optional_user)) -> models.User:
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if (
        getattr(request.state, "auth_method", "") == "cookie"
        and request.method in UNSAFE_METHODS
        and not same_origin(request)
    ):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-origin request rejected")
    return user


def require_admin(user: models.User = Depends(get_current_user)) -> models.User:
    if not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return user
