"""HR assistant chatbot: LangChain agent + Mistral + Gradio."""

from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
import gradio as gr

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = """You are an HR assistant that processes holiday / leave applications.

Your job:
1. Extract these key fields from the conversation (ask for any that are missing):
   - User name
   - Email
   - Purpose (reason for leave)
   - Start Date
   - End Date
2. When you have both Start Date and End Date, you MUST call the count_working_days tool.
   Never count working days yourself.
3. When you have all five fields, you MUST call record_holiday_application with the extracted values.
4. Then show the user a clear confirmation of:
   User name, Email, Purpose, Start Date, End Date, and the working-day count
   (Monday to Friday, inclusive). Ask them to confirm whether the information is correct.
   If they say it is wrong, collect the corrections and call the tools again.

Working days are Monday through Friday only. Do not invent holidays.

For other HR questions that are not a holiday application, answer normally.
This is general guidance, not legal advice.
"""

DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
)


def load_api_key() -> str:
    env_key = os.getenv("MISTRAL_API_KEY")
    if env_key:
        return env_key
    key_path = ROOT / "mistral_key.json"
    data = json.loads(key_path.read_text(encoding="utf-8"))
    return f"{data['a']}{data['b']}"


def parse_date(value: str) -> date:
    text = str(value).strip()
    if "T" in text:
        text = text.split("T", 1)[0]
    text = text[:10]
    last_error = None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError as exc:
            last_error = exc
    raise ValueError(
        f"Could not parse date '{value}'. Use YYYY-MM-DD (for example 2026-09-01)."
    ) from last_error


def working_days_between(start: date, end: date) -> int:
    if start > end:
        start, end = end, start
    total = 0
    current = start
    while current <= end:
        if current.weekday() < 5:
            total += 1
        current += timedelta(days=1)
    return total


@tool
def count_working_days(start_date: str, end_date: str) -> str:
    """Count Monday-Friday working days between two dates, inclusive.

    Args:
        start_date: Leave start date (YYYY-MM-DD preferred).
        end_date: Leave end date (YYYY-MM-DD preferred).
    """
    start = parse_date(start_date)
    end = parse_date(end_date)
    ordered_start, ordered_end = (start, end) if start <= end else (end, start)
    count = working_days_between(start, end)
    note = ""
    if start > end:
        note = " Dates were swapped so the range runs from earlier to later."
    return (
        f"{count} working day(s) (Monday to Friday) from {ordered_start.isoformat()} "
        f"to {ordered_end.isoformat()}, inclusive. Weekends are excluded.{note}"
    )


@tool
def record_holiday_application(
    user_name: str,
    email: str,
    purpose: str,
    start_date: str,
    end_date: str,
) -> str:
    """Save holiday application fields extracted from the conversation and attach the working-day count.

    Call this only after User name, Email, Purpose, Start Date, and End Date are known.

    Args:
        user_name: Applicant's full name.
        email: Applicant's email address.
        purpose: Reason for the holiday or leave.
        start_date: First day of leave (YYYY-MM-DD preferred).
        end_date: Last day of leave (YYYY-MM-DD preferred).
    """
    name = (user_name or "").strip()
    mail = (email or "").strip()
    reason = (purpose or "").strip()
    missing = [
        label
        for label, value in (
            ("User name", name),
            ("Email", mail),
            ("Purpose", reason),
            ("Start Date", start_date),
            ("End Date", end_date),
        )
        if not value
    ]
    if missing:
        return f"Missing fields: {', '.join(missing)}. Ask the user for these before confirming."
    if not re.search(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", mail):
        return f"Email '{mail}' does not look valid. Ask the user for a correct email."
    working = count_working_days.invoke(
        {"start_date": start_date, "end_date": end_date}
    )
    start = parse_date(start_date)
    end = parse_date(end_date)
    ordered_start, ordered_end = (start, end) if start <= end else (end, start)
    return (
        "Holiday application draft\n"
        f"- User name: {name}\n"
        f"- Email: {mail}\n"
        f"- Purpose: {reason}\n"
        f"- Start Date: {ordered_start.isoformat()}\n"
        f"- End Date: {ordered_end.isoformat()}\n"
        f"- Working days: {working}\n"
        "Ask the user to confirm whether this information is correct."
    )


TOOLS = [count_working_days, record_holiday_application]

api_key = load_api_key()

MODEL_CANDIDATES = (
    "open-mistral-nemo",
    "mistral-tiny",
    "ministral-8b-latest",
)


def _llm(model: str) -> ChatMistralAI:
    return ChatMistralAI(
        model=model,
        api_key=api_key,
        temperature=0.1,
        max_retries=2,
    )


def _agent(model: str):
    return create_agent(_llm(model), tools=TOOLS, system_prompt=SYSTEM_PROMPT)


def _message_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content)


def _history_messages(history: list, user_text: str) -> list:
    messages = []
    for turn in history:
        if isinstance(turn, dict):
            role, content = turn.get("role"), turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            user_turn, assistant_text = turn
            if user_turn:
                messages.append(HumanMessage(content=user_turn))
            if assistant_text:
                messages.append(AIMessage(content=assistant_text))
    messages.append(HumanMessage(content=user_text))
    return messages


def _normalize_date_input(value) -> str:
    if value is None or value == "":
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value).strip()


def chat(message: str, history: list | None, start_date=None, end_date=None) -> str:
    if isinstance(message, dict):
        message = str(message.get("text") or message.get("content") or "")
    history = history or []
    start_date = _normalize_date_input(start_date)
    end_date = _normalize_date_input(end_date)
    user_text = (message or "").strip()
    extras = []
    if start_date:
        extras.append(f"Start Date: {start_date}")
    if end_date:
        extras.append(f"End Date: {end_date}")
    if extras:
        user_text = f"{user_text}\n\n" + "\n".join(extras) if user_text else "\n".join(extras)

    messages = _history_messages(history, user_text)
    last_error = None
    for model in MODEL_CANDIDATES:
        try:
            result = _agent(model).invoke({"messages": messages})
            final = result["messages"][-1]
            text = _message_text(getattr(final, "content", final)).strip()
            return text or "I could not produce a reply. Please try again."
        except Exception as exc:
            last_error = exc
            text = str(exc)
            if "429" in text or "rate_limited" in text or "capacity" in text.lower():
                continue
            return f"Sorry, I could not complete that request: {exc}"
    return (
        "The Mistral API is busy right now. Please send the question again in a moment."
        f" (last error: {last_error})"
    )


with gr.Blocks(title="HR Assistant") as demo:
    gr.Markdown(
        "## HR Assistant\n"
        "Apply for holiday leave in chat. The agent extracts **User name**, **Email**, "
        "**Purpose**, **Start Date**, and **End Date**, counts **working days** "
        "(Monday to Friday), then asks you to confirm."
    )
    chatbot = gr.Chatbot(height=480)
    with gr.Row():
        start_in = gr.Textbox(label="Start date", placeholder="YYYY-MM-DD")
        end_in = gr.Textbox(label="End date", placeholder="YYYY-MM-DD")
    msg_in = gr.Textbox(
        label="Message",
        placeholder="e.g. I am Jane Doe, jane@acme.com. Annual leave 1–15 Sep 2026 for a family trip.",
    )
    send = gr.Button("Send", variant="primary")

    def respond(message, history, start_date, end_date):
        history = history or []
        if not (message or "").strip() and not (start_date and end_date):
            return history, message
        display = (message or "").strip()
        if start_date and end_date:
            display = display or f"Holiday dates from {start_date} to {end_date}"
        reply = chat(message, history, start_date, end_date)
        history = history + [
            {"role": "user", "content": display},
            {"role": "assistant", "content": reply},
        ]
        return history, ""

    inputs = [msg_in, chatbot, start_in, end_in]
    send.click(respond, inputs, [chatbot, msg_in])
    msg_in.submit(respond, inputs, [chatbot, msg_in])

    gr.Examples(
        examples=[
            [
                "I am Jane Doe, jane@acme.com. I want annual leave from 2026-09-01 to 2026-09-15 for a family trip.",
                "2026-09-01",
                "2026-09-15",
            ],
            [
                "Please process my holiday: Alex Chen, alex.chen@company.com, medical appointment, 2026-10-05 to 2026-10-07.",
                "2026-10-05",
                "2026-10-07",
            ],
        ],
        inputs=[msg_in, start_in, end_in],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
