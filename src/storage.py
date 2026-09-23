"""SQLite persistence; labels remain in evaluation files, never inspection history."""
from __future__ import annotations

import csv
import io
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


HISTORY_COLUMNS = (
    "run_id", "item_index", "inspected_at", "source_name", "source_sha256", "model_id",
    "machine_type", "machine_id", "sample_rate", "duration_seconds", "channel_index",
    "score", "threshold", "decision", "processing_ms", "status", "error_message",
)


def init_db(db_path: str | Path) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=10)) as connection, connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                item_index INTEGER NOT NULL,
                inspected_at TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_sha256 TEXT,
                model_id TEXT NOT NULL,
                machine_type TEXT NOT NULL,
                machine_id TEXT NOT NULL,
                sample_rate INTEGER,
                duration_seconds REAL,
                channel_index INTEGER NOT NULL,
                score REAL,
                threshold REAL,
                decision TEXT,
                processing_ms REAL NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('success','invalid_input','error')),
                error_message TEXT,
                UNIQUE(run_id, item_index)
            )
        """)


def save_result(db_path: str | Path, result: dict[str, Any]) -> bool:
    """Insert once per run/item; a deliberate rerun must have a new run_id."""
    init_db(db_path)
    values = [result.get(column) for column in HISTORY_COLUMNS]
    placeholders = ",".join("?" for _ in HISTORY_COLUMNS)
    columns = ",".join(HISTORY_COLUMNS)
    with closing(sqlite3.connect(db_path, timeout=10)) as connection, connection:
        cursor = connection.execute(
            f"INSERT INTO inspections ({columns}) VALUES ({placeholders}) "
            "ON CONFLICT(run_id,item_index) DO NOTHING", values,
        )
        return cursor.rowcount == 1


def read_history(db_path: str | Path, limit: int = 1000) -> list[dict[str, Any]]:
    if not Path(db_path).exists():
        return []
    if not 1 <= limit <= 100_000:
        raise ValueError("履歴件数は1～100000件にしてください。")
    with closing(sqlite3.connect(db_path, timeout=10)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"SELECT {','.join(HISTORY_COLUMNS)} FROM inspections ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def history_csv(rows: list[dict[str, Any]], include_save_status: bool = False, include_reference: bool = False) -> bytes:
    """UTF-8 BOM for Excel; untrusted names cannot become spreadsheet formulas."""
    output = io.StringIO(newline="")
    columns = HISTORY_COLUMNS + (("save_status", "save_error") if include_save_status else ())
    if include_reference:
        columns += ("reference_label", "comparison")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        safe = {}
        for column in columns:
            value = row.get(column)
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            safe[column] = value
        writer.writerow(safe)
    return output.getvalue().encode("utf-8-sig")
