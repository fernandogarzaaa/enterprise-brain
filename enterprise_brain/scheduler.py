"""Optional background sync of every connector (``SYNC_INTERVAL_MINUTES`` > 0).

Run it in exactly one process: with several replicas, enable it on one of them only
(or run ``enterprise-brain sync --all`` from cron instead).
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from enterprise_brain.deps import AppContext
from enterprise_brain.sources import sync_all_connectors

log = logging.getLogger(__name__)


def run_sync_job(ctx: AppContext) -> int:
    db = ctx.sessions()
    try:
        results = sync_all_connectors(db, ctx.settings, ctx.embedder)
        return sum(r.chunks_indexed for r in results.values())
    finally:
        db.close()


def start_scheduler(ctx: AppContext) -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        run_sync_job,
        "interval",
        minutes=ctx.settings.SYNC_INTERVAL_MINUTES,
        args=[ctx],
        id="sync-connectors",
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    log.info("connector sync scheduled", extra={"minutes": ctx.settings.SYNC_INTERVAL_MINUTES})
    return scheduler
