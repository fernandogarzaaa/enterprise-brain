"""Structured logging: JSON lines in production, readable text in development.

A per-request id (``X-Request-ID``) is stored in a context variable and attached to
every log record emitted while the request is handled.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Attributes every LogRecord has; anything else was passed via ``extra=`` and is emitted.
_RESERVED = frozenset(
    vars(logging.LogRecord("x", logging.INFO, "x", 0, "x", None, None)).keys()
) | {"message", "asctime", "request_id"}

TEXT_FORMAT = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"


def extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in vars(record).items() if k not in _RESERVED and not k.startswith("_")}


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with any ``extra=`` fields merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        payload.update(extra_fields(record))
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable lines with ``key=value`` extras appended."""

    def __init__(self) -> None:
        super().__init__(TEXT_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        line = super().format(record)
        extras = extra_fields(record)
        if extras:
            line += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        return line


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Install a single stdout handler on the root logger (idempotent)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, "_enterprise_brain", False):
            root.removeHandler(existing)
    handler._enterprise_brain = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)
    # uvicorn's access log duplicates our request log line.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
