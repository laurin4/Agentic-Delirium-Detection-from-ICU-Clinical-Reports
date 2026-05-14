"""
Optional SQLite append-log for prediction rows (CSV remains canonical).

Enable with: ENABLE_SQLITE_LOGGING=true
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Mapping


def init_prediction_db(path: Path | str) -> None:
    """Create parent dirs and ensure the predictions table exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                PatientenID TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prediction_rows_patient ON prediction_rows (PatientenID)"
        )
        conn.commit()
    finally:
        conn.close()


def log_prediction_row(path: Path | str, row_dict: Mapping[str, Any]) -> None:
    """Insert one prediction row as JSON payload (UTF-8, default=str for non-JSON types)."""
    p = Path(path)
    pid = str(row_dict.get("PatientenID", "") or "")
    payload = json.dumps(dict(row_dict), ensure_ascii=False, default=str)
    conn = sqlite3.connect(str(p))
    try:
        conn.execute(
            "INSERT INTO prediction_rows (PatientenID, payload) VALUES (?, ?)",
            (pid, payload),
        )
        conn.commit()
    finally:
        conn.close()
