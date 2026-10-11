from __future__ import annotations

import json
import logging
import sys

import pytest

from enterprise_brain.logging_config import (
    JsonFormatter,
    RequestIdFilter,
    TextFormatter,
    configure_logging,
    extra_fields,
    request_id_var,
)


def _record(msg: str = "hello %s", args: tuple[object, ...] = ("world",), **extra: object) -> logging.LogRecord:
    rec = logging.LogRecord("eb.test", logging.INFO, __file__, 1, msg, args, None)
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def test_extra_fields_only_custom() -> None:
    assert extra_fields(_record(source_id=3, _private=1)) == {"source_id": 3}


def test_json_formatter() -> None:
    rec = _record(source_id=3)
    RequestIdFilter().filter(rec)
    data = json.loads(JsonFormatter().format(rec))
    assert data["message"] == "hello world"
    assert data["level"] == "INFO"
    assert data["logger"] == "eb.test"
    assert data["request_id"] == "-"
    assert data["source_id"] == 3
    assert data["ts"].endswith("+00:00")
    assert "exc_info" not in data


def test_json_formatter_exception_and_request_id() -> None:
    token = request_id_var.set("rid-1")
    try:
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            rec = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())
        RequestIdFilter().filter(rec)
        data = json.loads(JsonFormatter().format(rec))
    finally:
        request_id_var.reset(token)
    assert data["request_id"] == "rid-1"
    assert "RuntimeError: boom" in data["exc_info"]


def test_json_formatter_without_filter_and_unserialisable() -> None:
    data = json.loads(JsonFormatter().format(_record(obj=object())))
    assert data["request_id"] == "-"
    assert data["obj"].startswith("<object")


def test_text_formatter_appends_extras() -> None:
    token = request_id_var.set("abc")
    try:
        line = TextFormatter().format(_record(status=200, path="/x"))
    finally:
        request_id_var.reset(token)
    assert "[abc] eb.test: hello world" in line
    assert line.endswith("status=200 path=/x")
    plain = TextFormatter().format(_record())
    assert plain.endswith("hello world")


def test_configure_logging_is_idempotent(capsys: pytest.CaptureFixture[str]) -> None:
    root = logging.getLogger()
    before = len(root.handlers)
    configure_logging("DEBUG", "json")
    configure_logging("INFO", "json")
    ours = [h for h in root.handlers if getattr(h, "_enterprise_brain", False)]
    assert len(ours) == 1
    assert len(root.handlers) <= before + 1
    assert isinstance(ours[0].formatter, JsonFormatter)
    assert root.level == logging.INFO
    assert logging.getLogger("uvicorn.access").level == logging.WARNING
    configure_logging("WARNING", "text")
    ours = [h for h in root.handlers if getattr(h, "_enterprise_brain", False)]
    assert isinstance(ours[0].formatter, TextFormatter)
