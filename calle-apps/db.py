"""SQLite persistence for meeting-room bookings."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(os.getenv("BOOKINGS_DB", Path(__file__).resolve().parent / "bookings.db"))

ROOMS = ("A", "B", "C")
OPEN_TIME = "09:00"
CLOSE_TIME = "21:00"

SCHEMA = """
CREATE TABLE IF NOT EXISTS bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_name TEXT NOT NULL,
    email TEXT NOT NULL,
    purpose TEXT NOT NULL,
    booking_date TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    room_number TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def find_conflicts(
    room_number: str,
    booking_date: str,
    start_time: str,
    end_time: str,
    db_path: Path | None = None,
) -> list[sqlite3.Row]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM bookings
            WHERE room_number = ?
              AND booking_date = ?
              AND start_time < ?
              AND end_time > ?
            ORDER BY start_time
            """,
            (room_number, booking_date, end_time, start_time),
        ).fetchall()
    return list(rows)


def insert_booking(
    person_name: str,
    email: str,
    purpose: str,
    booking_date: str,
    start_time: str,
    end_time: str,
    room_number: str,
    db_path: Path | None = None,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO bookings (
                person_name, email, purpose, booking_date,
                start_time, end_time, room_number, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                person_name,
                email,
                purpose,
                booking_date,
                start_time,
                end_time,
                room_number,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_all_bookings(db_path: Path | None = None) -> list[sqlite3.Row]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM bookings
            ORDER BY booking_date, start_time, room_number
            """
        ).fetchall()
    return list(rows)
