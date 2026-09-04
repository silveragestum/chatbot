"""Meeting-room booking chatbot: LangChain + Mistral + Gradio."""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI

load_dotenv()

ROOMS = [
    {
        "id": "aurora",
        "name": "Aurora",
        "capacity": 4,
        "features": ["TV", "whiteboard", "video conferencing"],
    },
    {
        "id": "harbor",
        "name": "Harbor",
        "capacity": 8,
        "features": ["projector", "whiteboard", "phone"],
    },
    {
        "id": "summit",
        "name": "Summit",
        "capacity": 16,
        "features": ["video conferencing", "projector", "catering table"],
    },
    {
        "id": "nook",
        "name": "Nook",
        "capacity": 2,
        "features": ["quiet booth", "webcam"],
    },
]

# In-memory bookings: (room_id, date_iso, start_hhmm, end_hhmm) -> client name
BOOKINGS: dict[tuple[str, str, str, str], str] = {}

SYSTEM_PROMPT = """You are Calle, a professional meeting-room booking assistant for clients.

Your job is to:
- Help clients find a suitable meeting room
- Check availability
- Book, update, or cancel reservations
- Confirm details clearly (room, date, time, attendee count, client name)

Available rooms: Aurora (4), Harbor (8), Summit (16), Nook (2).
Hours: 08:00–18:00, weekdays. Times are 24-hour local time.

Always collect: client name, date, start time, end time, number of attendees, and any equipment needs.
Use the tools to list rooms, check availability, book, and cancel. Do not invent bookings.
If something is unavailable, suggest alternatives.
Be concise, courteous, and confirm the final reservation in a short summary.
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


def _overlaps(start_a: str, end_a: str, start_b: str, end_b: str) -> bool:
    return start_a < end_b and start_b < end_a


def _find_room(name_or_id: str) -> dict[str, Any] | None:
    key = name_or_id.strip().lower()
    for room in ROOMS:
        if room["id"] == key or room["name"].lower() == key:
            return room
    return None


@tool
def list_rooms() -> str:
    """List meeting rooms with capacity and features."""
    lines = []
    for room in ROOMS:
        features = ", ".join(room["features"])
        lines.append(
            f"- {room['name']} (id: {room['id']}): seats {room['capacity']}; {features}"
        )
    return "\n".join(lines)


@tool
def check_availability(room: str, date: str, start_time: str, end_time: str) -> str:
    """Check if a named room is free for a date and time range."""
    try:
        day = _parse_date(date)
        start = _parse_time(start_time)
        end = _parse_time(end_time)
    except ValueError as exc:
        return str(exc)
    found = _find_room(room)
    if not found:
        return f"Unknown room '{room}'. Use list_rooms to see options."
    if start >= end:
        return "Start time must be before end time."
    conflicts = [
        f"{s}-{e} ({client})"
        for (rid, d, s, e), client in BOOKINGS.items()
        if rid == found["id"] and d == day and _overlaps(start, end, s, e)
    ]
    if conflicts:
        return f"{found['name']} is NOT available on {day} {start}-{end}. Conflicts: {', '.join(conflicts)}."
    return f"{found['name']} is available on {day} from {start} to {end}."


@tool
def book_room(
    room: str,
    date: str,
    start_time: str,
    end_time: str,
    client_name: str,
    attendees: int = 1,
) -> str:
    """Book a meeting room for a client. Fails if the slot is taken or the room is too small."""
    try:
        day = _parse_date(date)
        start = _parse_time(start_time)
        end = _parse_time(end_time)
    except ValueError as exc:
        return str(exc)
    found = _find_room(room)
    if not found:
        return f"Unknown room '{room}'."
    if attendees > found["capacity"]:
        return (
            f"{found['name']} only seats {found['capacity']}. "
            "Choose a larger room such as Harbor or Summit."
        )
    if start >= end:
        return "Start time must be before end time."
    for (rid, d, s, e), existing in BOOKINGS.items():
        if rid == found["id"] and d == day and _overlaps(start, end, s, e):
            return f"Cannot book: {found['name']} is held by {existing} ({s}-{e})."
    BOOKINGS[(found["id"], day, start, end)] = client_name.strip()
    return (
        f"Booked {found['name']} on {day} {start}-{end} for {client_name} "
        f"({attendees} attendee(s))."
    )


@tool
def cancel_booking(room: str, date: str, start_time: str, client_name: str) -> str:
    """Cancel an existing booking for a client."""
    try:
        day = _parse_date(date)
        start = _parse_time(start_time)
    except ValueError as exc:
        return str(exc)
    found = _find_room(room)
    if not found:
        return f"Unknown room '{room}'."
    for key, existing in list(BOOKINGS.items()):
        rid, d, s, _e = key
        if rid == found["id"] and d == day and s == start:
            if existing.lower() != client_name.strip().lower():
                return f"Booking exists but is under '{existing}', not '{client_name}'."
            del BOOKINGS[key]
            return f"Cancelled {found['name']} on {day} at {start} for {existing}."
    return "No matching booking found."


@tool
def list_bookings() -> str:
    """List all current in-memory bookings."""
    if not BOOKINGS:
        return "No bookings yet."
    lines = []
    for (rid, d, s, e), client in sorted(BOOKINGS.items(), key=lambda item: item[0][1:]):
        name = next(r["name"] for r in ROOMS if r["id"] == rid)
        lines.append(f"- {name} on {d} {s}-{e}: {client}")
    return "\n".join(lines)


TOOLS = [list_rooms, check_availability, book_room, cancel_booking, list_bookings]


def _build_llm() -> Any:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is not set. Export it or put it in a .env file."
        )
    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key,
        temperature=0.3,
    ).bind_tools(TOOLS)


def _run_agent(user_message: str, history: list[dict[str, str]]) -> str:
    llm = _build_llm()
    messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history:
        role = turn.get("role")
        content = turn.get("content") or ""
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_message))

    tool_map = {t.name: t for t in TOOLS}
    for _ in range(8):
        response = None
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = llm.invoke(messages)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                text = str(exc).lower()
                if "429" in text or "rate limit" in text:
                    time.sleep(2 ** attempt)
                    continue
                raise
        if response is None:
            raise last_error or RuntimeError("Mistral request failed")
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return response.content if isinstance(response.content, str) else str(
                response.content
            )
        for call in tool_calls:
            name = call["name"]
            args = call.get("args") or {}
            tool_fn = tool_map.get(name)
            if tool_fn is None:
                result = f"Unknown tool: {name}"
            else:
                result = tool_fn.invoke(args)
            messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )
    return "I needed too many tool steps. Please restate your booking request."


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
            Chat to check availability and book Aurora, Harbor, Summit, or Nook.
            """
        )
        gr.ChatInterface(
            fn=chat,
            type="messages",
            examples=[
                "What rooms do you have?",
                "Book Harbor tomorrow 10:00-11:30 for Acme, 6 people.",
                "Is Summit free today from 14:00 to 15:00?",
            ],
        )
    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)
