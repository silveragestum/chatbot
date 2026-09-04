"""Meeting-room booking chatbot: LangChain agent + Mistral + Gradio + SQLite."""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI

from db import (
    CLOSE_TIME,
    OPEN_TIME,
    ROOMS,
    find_conflicts,
    insert_booking,
    list_all_bookings,
)

load_dotenv()

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

SYSTEM_PROMPT = """You are Calle, a meeting-room booking assistant.

Rooms: A, B, and C. Hours: 09:00–21:00 (9am–9pm). Times are local 24-hour clock.

You MUST extract these fields from the user before saving:
1) person name
2) email
3) meeting purpose
4) booking date
5) start time
6) end time
7) meeting room number (A, B, or C)

Ask for anything that is missing. Do not invent an email, purpose, or name.

When all seven fields are present, call save_booking. That tool writes to SQLite only if
the slot does not overlap an existing booking in the same room on the same date.
If the tool reports a conflict or rejection, tell the user clearly that the booking was
NOT saved and suggest another room or time. Never claim a booking succeeded unless
save_booking returned a success message.

You may also call check_availability or list_bookings to help the user.
"""


def _parse_date(value: str) -> str:
    value = value.strip()
    today = date.today()
    lowered = value.lower()
    if lowered in {"today"}:
        return today.isoformat()
    if lowered in {"tomorrow"}:
        return (today + timedelta(days=1)).isoformat()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Could not parse date '{value}'. Use YYYY-MM-DD.")


def _parse_time(value: str) -> str:
    value = value.strip().upper().replace(" ", "")
    for fmt in ("%H:%M", "%H%M", "%I:%M%p", "%I%p"):
        try:
            return datetime.strptime(value, fmt).strftime("%H:%M")
        except ValueError:
            continue
    raise ValueError(f"Could not parse time '{value}'. Use HH:MM.")


def _normalize_room(room: str) -> str | None:
    key = room.strip().upper()
    if key.startswith("ROOM"):
        key = key.replace("ROOM", "", 1).strip()
    if key in ROOMS:
        return key
    return None


def _hours_ok(start: str, end: str) -> str | None:
    if start >= end:
        return "Start time must be before end time."
    if start < OPEN_TIME or end > CLOSE_TIME:
        return f"Bookings are only allowed between {OPEN_TIME} and {CLOSE_TIME}."
    return None


@tool
def list_rooms() -> str:
    """List the three meeting rooms and opening hours."""
    return (
        f"Meeting rooms: A, B, C. Allowed times: {OPEN_TIME}–{CLOSE_TIME} (9am–9pm)."
    )


@tool
def check_availability(room: str, date: str, start_time: str, end_time: str) -> str:
    """Check whether room A, B, or C is free for a date and time range."""
    try:
        day = _parse_date(date)
        start = _parse_time(start_time)
        end = _parse_time(end_time)
    except ValueError as exc:
        return str(exc)
    room_number = _normalize_room(room)
    if room_number is None:
        return f"Unknown room '{room}'. Use A, B, or C."
    hours_error = _hours_ok(start, end)
    if hours_error:
        return hours_error
    conflicts = find_conflicts(room_number, day, start, end)
    if conflicts:
        details = ", ".join(
            f"{row['start_time']}-{row['end_time']} ({row['person_name']})"
            for row in conflicts
        )
        return (
            f"CONFLICT: Room {room_number} is NOT available on {day} {start}-{end}. "
            f"Existing: {details}."
        )
    return f"Room {room_number} is available on {day} from {start} to {end}."


@tool
def save_booking(
    person_name: str,
    email: str,
    purpose: str,
    date: str,
    start_time: str,
    end_time: str,
    room_number: str,
) -> str:
    """Save a booking to SQLite after conflict checks.

    Required: person name, email, meeting purpose, date, start time, end time, room (A/B/C).
    Rejects and does not write if the room/time overlaps an existing booking or is outside hours.
    """
    try:
        day = _parse_date(date)
        start = _parse_time(start_time)
        end = _parse_time(end_time)
    except ValueError as exc:
        return f"REJECTED: {exc}"
    room = _normalize_room(room_number)
    if room is None:
        return "REJECTED: Room must be A, B, or C."
    name = person_name.strip()
    mail = email.strip()
    why = purpose.strip()
    if not name:
        return "REJECTED: Name is required."
    if not EMAIL_RE.match(mail):
        return "REJECTED: A valid email is required."
    if not why:
        return "REJECTED: Meeting purpose is required."
    hours_error = _hours_ok(start, end)
    if hours_error:
        return f"REJECTED: {hours_error}"
    conflicts = find_conflicts(room, day, start, end)
    if conflicts:
        details = ", ".join(
            f"{row['start_time']}-{row['end_time']} booked by {row['person_name']} "
            f"({row['purpose']})"
            for row in conflicts
        )
        return (
            f"REJECTED: conflict on room {room} for {day} {start}-{end}. "
            f"Not saved. Overlaps: {details}."
        )
    booking_id = insert_booking(name, mail, why, day, start, end, room)
    return (
        f"SAVED booking #{booking_id}: {name} <{mail}> room {room} on {day} "
        f"{start}-{end} for '{why}'."
    )


@tool
def list_bookings() -> str:
    """List bookings stored in the SQLite database."""
    rows = list_all_bookings()
    if not rows:
        return "No bookings in the database yet."
    lines = []
    for row in rows:
        lines.append(
            f"- #{row['id']} room {row['room_number']} {row['booking_date']} "
            f"{row['start_time']}-{row['end_time']}: {row['person_name']} "
            f"<{row['email']}> — {row['purpose']}"
        )
    return "\n".join(lines)


TOOLS = [list_rooms, check_availability, save_booking, list_bookings]


def _build_agent():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is not set. Export it or put it in a .env file."
        )
    model = ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key,
        temperature=0.2,
    )
    return create_agent(model=model, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


def _history_to_messages(history: list[dict[str, str]]) -> list[Any]:
    messages: list[Any] = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content") or ""
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _run_agent(user_message: str, history: list[dict[str, str]]) -> str:
    agent = _build_agent()
    messages = _history_to_messages(history)
    messages.append(HumanMessage(content=user_message))
    result = agent.invoke({"messages": messages})
    final = result["messages"][-1]
    content = getattr(final, "content", final)
    return content if isinstance(content, str) else str(content)


def chat(message: str, history: list[dict[str, str]]) -> str:
    try:
        return _run_agent(message, history)
    except Exception as exc:  # noqa: BLE001 — surface API/setup errors in the UI
        return f"Sorry, I could not complete that request: {exc}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Calle — Meeting Room Booking") as demo:
        gr.Markdown(
            """
            # Calle — Meeting Room Booking
            Rooms **A, B, C** · 09:00–21:00 · Bookings saved to SQLite when the slot is free.
            """
        )
        gr.ChatInterface(
            fn=chat,
            examples=[
                "What rooms can I book?",
                "Book room B tomorrow 10:00-11:30. I'm Ada Lovelace, ada@example.com, purpose: product review.",
                "Show current bookings.",
            ],
        )
    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)
