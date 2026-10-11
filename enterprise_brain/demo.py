"""Demo data: 10 sample documents and 3 connectors, indexed through the real pipeline.

Enabled with ``DEMO_DATA=true`` or ``enterprise-brain seed-demo``. Idempotent: sources
that already exist by name are skipped.
"""

from __future__ import annotations

import json
import logging
from importlib import resources

from sqlalchemy.orm import Session

from enterprise_brain import models
from enterprise_brain.config import Settings
from enterprise_brain.ingest import store_document
from enterprise_brain.llm import Embedder
from enterprise_brain.models import utcnow
from enterprise_brain.sources import create_connector, stage_connector_file, sync_connector

log = logging.getLogger(__name__)


def seed_demo(db: Session, settings: Settings, embedder: Embedder) -> dict[str, int]:
    root = resources.files("enterprise_brain") / "demo_data"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    existing = {name for (name,) in db.query(models.Source.name).all()}
    counts = {"sources": 0, "chunks": 0}

    for item in manifest["documents"]:
        if item["name"] in existing:
            continue
        data = (root / "documents" / item["file"]).read_bytes()
        src = models.Source(name=item["name"], kind="document", status="ready", last_sync_at=utcnow())
        db.add(src)
        db.flush()
        doc = store_document(db, settings, embedder, src, item["file"], data)
        db.commit()
        counts["sources"] += 1
        counts["chunks"] += doc.chunk_count

    for item in manifest["connectors"]:
        if item["name"] in existing:
            continue
        src = create_connector(db, settings, item["name"], item["type"])
        data = (root / "connectors" / item["file"]).read_bytes()
        stage_connector_file(settings, src, item["file"], data)
        result = sync_connector(db, settings, embedder, src)
        counts["sources"] += 1
        counts["chunks"] += result.chunks_indexed

    if counts["sources"]:
        log.info("demo data seeded", extra=counts)
    return counts
