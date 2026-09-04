"""Tests for in-memory meeting-room booking tools."""

import unittest

from app import BOOKINGS, book_room, cancel_booking, check_availability, list_rooms


class BookingTests(unittest.TestCase):
    def setUp(self) -> None:
        BOOKINGS.clear()

    def test_list_rooms_includes_harbor(self) -> None:
        text = list_rooms.invoke({})
        self.assertIn("Harbor", text)
        self.assertIn("seats 8", text)

    def test_book_and_conflict(self) -> None:
        first = book_room.invoke(
            {
                "room": "Harbor",
                "date": "2026-09-10",
                "start_time": "10:00",
                "end_time": "11:00",
                "client_name": "Acme",
                "attendees": 6,
            }
        )
        self.assertTrue(first.startswith("Booked Harbor"))
        conflict = check_availability.invoke(
            {
                "room": "Harbor",
                "date": "2026-09-10",
                "start_time": "10:30",
                "end_time": "11:30",
            }
        )
        self.assertIn("NOT available", conflict)

    def test_cancel_booking(self) -> None:
        book_room.invoke(
            {
                "room": "Nook",
                "date": "2026-09-11",
                "start_time": "09:00",
                "end_time": "09:30",
                "client_name": "Lee",
                "attendees": 1,
            }
        )
        cancelled = cancel_booking.invoke(
            {
                "room": "Nook",
                "date": "2026-09-11",
                "start_time": "09:00",
                "client_name": "Lee",
            }
        )
        self.assertTrue(cancelled.startswith("Cancelled"))
        free = check_availability.invoke(
            {
                "room": "Nook",
                "date": "2026-09-11",
                "start_time": "09:00",
                "end_time": "09:30",
            }
        )
        self.assertIn("is available", free)


if __name__ == "__main__":
    unittest.main()
