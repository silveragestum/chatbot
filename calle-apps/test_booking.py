"""Tests for SQLite booking tools and conflict rejection."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path


class BookingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)
        os.environ["BOOKINGS_DB"] = str(self.db_path)
        import db as db_mod
        import app as app_mod

        db_mod.DB_PATH = self.db_path
        self.db = db_mod
        self.app = app_mod

    def tearDown(self) -> None:
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_list_rooms(self) -> None:
        text = self.app.list_rooms.invoke({})
        self.assertIn("A, B, C", text)
        self.assertIn("09:00", text)

    def test_save_then_conflict_rejected(self) -> None:
        first = self.app.save_booking.invoke(
            {
                "person_name": "Ada Lovelace",
                "email": "ada@example.com",
                "purpose": "product review",
                "date": "2026-09-10",
                "start_time": "10:00",
                "end_time": "11:00",
                "room_number": "B",
            }
        )
        self.assertTrue(first.startswith("SAVED"), first)
        rows = self.db.list_all_bookings(self.db_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["email"], "ada@example.com")
        self.assertEqual(rows[0]["room_number"], "B")

        second = self.app.save_booking.invoke(
            {
                "person_name": "Alan Turing",
                "email": "alan@example.com",
                "purpose": "planning",
                "date": "2026-09-10",
                "start_time": "10:30",
                "end_time": "11:30",
                "room_number": "B",
            }
        )
        self.assertTrue(second.startswith("REJECTED"), second)
        self.assertIn("conflict", second.lower())
        self.assertEqual(len(self.db.list_all_bookings(self.db_path)), 1)

    def test_same_slot_other_room_ok(self) -> None:
        self.app.save_booking.invoke(
            {
                "person_name": "Ada Lovelace",
                "email": "ada@example.com",
                "purpose": "review",
                "date": "2026-09-10",
                "start_time": "10:00",
                "end_time": "11:00",
                "room_number": "A",
            }
        )
        other = self.app.save_booking.invoke(
            {
                "person_name": "Grace Hopper",
                "email": "grace@example.com",
                "purpose": "standup",
                "date": "2026-09-10",
                "start_time": "10:00",
                "end_time": "11:00",
                "room_number": "C",
            }
        )
        self.assertTrue(other.startswith("SAVED"), other)
        self.assertEqual(len(self.db.list_all_bookings(self.db_path)), 2)

    def test_outside_hours_rejected(self) -> None:
        result = self.app.save_booking.invoke(
            {
                "person_name": "Ada Lovelace",
                "email": "ada@example.com",
                "purpose": "late call",
                "date": "2026-09-10",
                "start_time": "20:00",
                "end_time": "22:00",
                "room_number": "A",
            }
        )
        self.assertTrue(result.startswith("REJECTED"), result)
        self.assertEqual(len(self.db.list_all_bookings(self.db_path)), 0)

    def test_invalid_email_rejected(self) -> None:
        result = self.app.save_booking.invoke(
            {
                "person_name": "Ada",
                "email": "not-an-email",
                "purpose": "chat",
                "date": "2026-09-10",
                "start_time": "10:00",
                "end_time": "11:00",
                "room_number": "A",
            }
        )
        self.assertIn("email", result.lower())
        self.assertEqual(len(self.db.list_all_bookings(self.db_path)), 0)


if __name__ == "__main__":
    unittest.main()
