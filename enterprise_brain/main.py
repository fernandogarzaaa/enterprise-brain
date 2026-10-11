"""FastAPI application factory."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib import resources

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from enterprise_brain import __version__
from enterprise_brain.api import answers, auth, pages, sources, system, team
from enterprise_brain.config import Settings, get_settings
from enterprise_brain.db import make_engine, make_sessionmaker, run_migrations
from enterprise_brain.deps import AppContext
from enterprise_brain.ingest import IngestError
from enterprise_brain.llm import LLMError, get_chat_provider, get_embedder
from enterprise_brain.logging_config import configure_logging, request_id_var
from enterprise_brain.security import RateLimiter
from enterprise_brain.sources import SourceError
from enterprise_brain.users import UserError, ensure_admin

log = logging.getLogger(__name__)

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; "
        "form-action 'self'"
    ),
}


def build_context(settings: Settings) -> AppContext:
    """Create the shared services (DB sessions, providers) for the app or the CLI."""
    pkg = resources.files("enterprise_brain")
    return AppContext(
        settings=settings,
        sessions=make_sessionmaker(make_engine(settings)),
        embedder=get_embedder(settings),
        chat=get_chat_provider(settings),
        login_limiter=RateLimiter(settings.LOGIN_RATE_LIMIT, settings.LOGIN_RATE_WINDOW_SECONDS),
        templates=Jinja2Templates(directory=str(pkg / "templates")),
    )


def bootstrap(ctx: AppContext) -> None:
    """Create the first admin and (optionally) the demo data."""
    db = ctx.sessions()
    try:
        ensure_admin(db, ctx.settings)
        if ctx.settings.DEMO_DATA:
            from enterprise_brain.demo import seed_demo

            seed_demo(db, ctx.settings, ctx.embedder)
    finally:
        db.close()


def _error_handler(default_status: int) -> Callable[[Request, Exception], Awaitable[Response]]:
    async def handler(_request: Request, exc: Exception) -> Response:
        return JSONResponse({"detail": str(exc)}, status_code=getattr(exc, "status", default_status))

    return handler


async def _llm_error(_request: Request, exc: Exception) -> Response:
    return JSONResponse({"detail": f"Embedding/LLM provider error: {exc}"}, status_code=502)


async def _validation_error(_request: Request, exc: Exception) -> Response:
    assert isinstance(exc, RequestValidationError)
    errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", "")} for e in exc.errors()]
    return JSONResponse({"detail": "Invalid request", "errors": errors}, status_code=422)


async def _http_error(_request: Request, exc: Exception) -> Response:
    assert isinstance(exc, HTTPException)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.LOG_LEVEL, settings.log_format)
    if settings.AUTO_MIGRATE:
        run_migrations(settings)
    ctx = build_context(settings)
    bootstrap(ctx)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        scheduler = None
        if settings.SYNC_INTERVAL_MINUTES:
            from enterprise_brain.scheduler import start_scheduler

            scheduler = start_scheduler(ctx)
        log.info("startup complete", extra={"version": __version__, "environment": settings.ENVIRONMENT})
        yield
        if scheduler is not None:
            scheduler.shutdown(wait=False)

    is_prod = settings.ENVIRONMENT == "production"
    app = FastAPI(
        title=settings.APP_NAME,
        version=__version__,
        lifespan=lifespan,
        docs_url=None if is_prod else "/docs",
        redoc_url=None,
        openapi_url=None if is_prod else "/openapi.json",
    )
    app.state.ctx = ctx

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            try:
                response = await call_next(request)
            except Exception:
                log.exception("unhandled error", extra={"path": request.url.path})
                response = JSONResponse({"detail": "Internal server error", "request_id": rid}, status_code=500)
            response.headers["X-Request-ID"] = rid
            for k, v in SECURITY_HEADERS.items():
                response.headers.setdefault(k, v)
            if settings.cookie_secure:
                response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
            if not request.url.path.startswith(("/static", "/api/health")):
                log.info(
                    "request",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                    },
                )
            return response
        finally:
            request_id_var.reset(token)

    app.add_exception_handler(SourceError, _error_handler(400))
    app.add_exception_handler(IngestError, _error_handler(422))
    app.add_exception_handler(UserError, _error_handler(400))
    app.add_exception_handler(ValueError, _error_handler(400))
    app.add_exception_handler(LLMError, _llm_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(HTTPException, _http_error)

    static_dir = resources.files("enterprise_brain") / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    for module in (system, auth, answers, sources, team, pages):
        app.include_router(module.router)
    return app


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn --factory enterprise_brain.main:app_factory``."""
    return create_app()
