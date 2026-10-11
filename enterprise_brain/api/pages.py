"""Server-rendered HTML pages (the web UI). All data mutations go through the JSON API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy.orm import Session

from enterprise_brain import __version__, models
from enterprise_brain.api.answers import saved_rows
from enterprise_brain.api.sources import chunk_counts
from enterprise_brain.api.system import compute_insights
from enterprise_brain.deps import AppContext, get_ctx, get_db, optional_user
from enterprise_brain.sources import CONNECTOR_TYPES, staging_dir_for

router = APIRouter(include_in_schema=False)

TRY_PROMPTS = [
    "What is the status of the Acme Corp contract?",
    "Who are our approved cloud vendors?",
    "What is the invoice approval policy?",
    "What was Q3 2026 revenue by client?",
    "Which invoices are overdue?",
]

NAV = [
    ("/", "Ask"),
    ("/sources", "Knowledge Sources"),
    ("/insights", "Insights"),
    ("/saved", "Saved Answers"),
    ("/team", "Team"),
    ("/settings", "Settings"),
]


def render(ctx: AppContext, request: Request, template: str, user: models.User, active: str, **extra: Any) -> Response:
    data = {
        "app_name": ctx.settings.APP_NAME,
        "version": __version__,
        "user": user,
        "nav": NAV,
        "active": active,
        **extra,
    }
    return ctx.templates.TemplateResponse(request, template, data)


def login_redirect() -> Response:
    return RedirectResponse("/login", status_code=303)


@router.get("/login", response_class=HTMLResponse)
def login_page(
    request: Request, ctx: AppContext = Depends(get_ctx), user: models.User | None = Depends(optional_user)
) -> Response:
    if user is not None:
        return RedirectResponse("/", status_code=303)
    return ctx.templates.TemplateResponse(
        request, "auth/login.html", {"app_name": ctx.settings.APP_NAME, "error": request.query_params.get("error")}
    )


@router.get("/", response_class=HTMLResponse)
def ask_page(
    request: Request,
    ctx: AppContext = Depends(get_ctx),
    user: models.User | None = Depends(optional_user),
    db: Session = Depends(get_db),
) -> Response:
    if user is None:
        return login_redirect()
    sources = db.query(models.Source).order_by(models.Source.name).all()
    recent = (
        db.query(models.Query).filter(models.Query.user_id == user.id).order_by(models.Query.id.desc()).limit(10).all()
    )
    return render(
        ctx,
        request,
        "ask.html",
        user,
        "/",
        sources=sources,
        chunk_counts=chunk_counts(db),
        recent_queries=recent,
        try_prompts=TRY_PROMPTS if sources else [],
    )


@router.get("/sources", response_class=HTMLResponse)
def sources_page(
    request: Request,
    ctx: AppContext = Depends(get_ctx),
    user: models.User | None = Depends(optional_user),
    db: Session = Depends(get_db),
) -> Response:
    if user is None:
        return login_redirect()
    sources = db.query(models.Source).order_by(models.Source.created_at.desc(), models.Source.id.desc()).all()
    return render(
        ctx,
        request,
        "sources.html",
        user,
        "/sources",
        sources=sources,
        chunk_counts=chunk_counts(db),
        connector_labels=CONNECTOR_TYPES,
        staging_dirs={s.id: str(staging_dir_for(ctx.settings, s)) for s in sources if s.kind == "connector"},
        max_upload_mb=ctx.settings.MAX_UPLOAD_MB,
    )


@router.get("/insights", response_class=HTMLResponse)
def insights_page(
    request: Request,
    ctx: AppContext = Depends(get_ctx),
    user: models.User | None = Depends(optional_user),
    db: Session = Depends(get_db),
) -> Response:
    if user is None:
        return login_redirect()
    return render(ctx, request, "insights.html", user, "/insights", insights=compute_insights(db))


@router.get("/saved", response_class=HTMLResponse)
def saved_page(
    request: Request,
    ctx: AppContext = Depends(get_ctx),
    user: models.User | None = Depends(optional_user),
    db: Session = Depends(get_db),
) -> Response:
    if user is None:
        return login_redirect()
    return render(ctx, request, "saved.html", user, "/saved", saved=saved_rows(db))


@router.get("/team", response_class=HTMLResponse)
def team_page(
    request: Request,
    ctx: AppContext = Depends(get_ctx),
    user: models.User | None = Depends(optional_user),
    db: Session = Depends(get_db),
) -> Response:
    if user is None:
        return login_redirect()
    members = db.query(models.User).order_by(models.User.created_at, models.User.id).all()
    return render(ctx, request, "team.html", user, "/team", members=members, is_admin=user.is_admin)


@router.get("/settings", response_class=HTMLResponse)
def settings_page(
    request: Request,
    ctx: AppContext = Depends(get_ctx),
    user: models.User | None = Depends(optional_user),
    db: Session = Depends(get_db),
) -> Response:
    if user is None:
        return login_redirect()
    keys = (
        db.query(models.ApiKey)
        .filter(models.ApiKey.user_id == user.id, models.ApiKey.revoked_at.is_(None))
        .order_by(models.ApiKey.id.desc())
        .all()
    )
    cfg = ctx.settings.public_summary() if user.is_admin else {}
    return render(ctx, request, "settings.html", user, "/settings", cfg=cfg, api_keys=keys)
