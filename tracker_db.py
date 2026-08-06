"""
tracker_db.py
SQLite storage for medicine-availability tracking requests.
Separate from conversation memory (InMemorySaver) - this is persistent, independent state.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "medicine_tracker.db"


def init_db():
    """Create the tracking table if it doesn't exist. Call this once at startup."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS medicine_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            medicine_name TEXT NOT NULL,
            url TEXT NOT NULL,
            recipient_email TEXT NOT NULL,
            status TEXT DEFAULT 'pending',       -- 'pending' or 'found'
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_checked_at TEXT,
            last_reminder_at TEXT,
            found_at TEXT,
            check_attempts INTEGER DEFAULT 0,
            last_error TEXT
        )
    """)
    conn.commit()
    conn.close()


def add_pending_record(medicine_name: str, url: str, recipient_email: str) -> int:
    """Insert a new pending tracking request. Returns the new record's id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "INSERT INTO medicine_tracking (medicine_name, url, recipient_email) VALUES (?, ?, ?)",
        (medicine_name, url, recipient_email),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_pending_records() -> list[dict]:
    """Return all records still marked 'pending'."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM medicine_tracking WHERE status = 'pending'"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def mark_found(record_id: int):
    """Mark a record as found - stops further checks/reminders for it."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE medicine_tracking SET status = 'found', found_at = ? WHERE id = ?",
        (datetime.now().isoformat(), record_id),
    )
    conn.commit()
    conn.close()


def record_check_attempt(record_id: int, error: str | None = None):
    """Record that a check attempt occurred and optionally save an error.

    This function increments the `check_attempts` counter and updates
    `last_checked_at`. If the check failed, the textual error is stored in
    `last_error` so operators can inspect failures. Records remain in the
    'pending' state to allow scheduled rechecks.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE medicine_tracking SET last_checked_at = ?, check_attempts = check_attempts + 1, last_error = ? WHERE id = ?",
        (datetime.now().isoformat(), error, record_id),
    )
    conn.commit()
    conn.close()


def should_send_reminder(record: dict) -> bool:
    """Return True when a reminder should be sent for the given record.

    A reminder is due if at least 12 hours have passed since the last
    reminder was sent, or since record creation if no reminder has been sent
    yet.
    """
    reference_time_str = record["last_reminder_at"] or record["created_at"]
    reference_time = datetime.fromisoformat(reference_time_str)
    return datetime.now() - reference_time >= timedelta(hours=12)


def mark_reminder_sent(record_id: int):
    """Update the record to indicate that a reminder email was sent now."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE medicine_tracking SET last_reminder_at = ? WHERE id = ?",
        (datetime.now().isoformat(), record_id),
    )
    conn.commit()
    conn.close()