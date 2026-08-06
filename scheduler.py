"""
scheduler.py
Background job: every 12 hours, re-check all pending medicine-tracking
records via the same LangGraph flow used by the MCP tool (is_recheck=True).

Run this as a separate long-running process alongside the MCP server:
    python scheduler.py
"""

import sys
import time
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler

from tracker_db import init_db, get_pending_records  # <-- confirm this function name in tracker_db.py
from tracker_graph import run_recheck


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def run_all_pending_checks() -> None:
    """Fetch every pending (not-yet-found) record and push it through the graph."""
    log("Scheduler tick: fetching pending records...")
    try:
        pending_records = get_pending_records()
    except Exception as e:
        log(f"Failed to fetch pending records: {e}")
        return

    if not pending_records:
        log("No pending records to check.")
        return

    log(f"Found {len(pending_records)} pending record(s). Rechecking each...")

    for record in pending_records:
        try:
            result = run_recheck(record)
            log(f"Record id={record.get('id')} ({record.get('medicine_name')}): {result}")
        except Exception as e:
            log(f"Error while rechecking record id={record.get('id')}: {e}")


def main():
    init_db()
    log("Scheduler starting - rechecking pending medicine trackers every 12 hours.")

    scheduler = BlockingScheduler()
    scheduler.add_job(
        run_all_pending_checks,
        trigger="interval",
        hours=12,
        id="medicine_recheck_job",
        next_run_time=datetime.now(),  # run once immediately on startup, then every 12h
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log("Scheduler stopped.")


if __name__ == "__main__":
    main()