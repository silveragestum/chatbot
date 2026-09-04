"""HR assistant chatbot: LangChain + Mistral + Gradio."""

from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
import gradio as gr

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = (
    "You are an HR assistant. Help employees and managers with human-resources "
    "questions: policies, leave, benefits, hiring, onboarding, performance, "
    "workplace conduct, and related processes. Be clear, professional, and "
    "practical. If a question is outside HR, say so and steer back to HR topics. "
    "This is general guidance, not legal advice. "
    "When the user asks how many working days are in a date range, or provides "
    "a start date and end date for leave or attendance, you MUST call the "
    "count_working_days tool. Do not count the days yourself. Working days are "
    "Monday through Friday, inclusive of both dates, excluding Saturday and Sunday."
)

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
        start_date: Start of the range (YYYY-MM-DD preferred).
        end_date: End of the range (YYYY-MM-DD preferred).
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


TOOLS = [count_working_days]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

api_key = load_api_key()

# mistral-small-latest hits rate limits on this account; these models work.
MODEL_CANDIDATES = (
    "open-mistral-nemo",
    "mistral-tiny",
    "ministral-8b-latest",
)


def _llm(model: str) -> ChatMistralAI:
    return ChatMistralAI(
        model=model,
        api_key=api_key,
        temperature=0.2,
        max_retries=2,
    ).bind_tools(TOOLS)


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


def _run_with_tools(llm, messages: list) -> str:
    response = llm.invoke(messages)
    for _ in range(6):
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            text = _message_text(response.content).strip()
            return text or "I could not produce a reply. Please try again."
        messages.append(response)
        for call in tool_calls:
            tool_fn = TOOLS_BY_NAME.get(call["name"])
            if tool_fn is None:
                result = f"Unknown tool: {call['name']}"
            else:
                try:
                    result = tool_fn.invoke(call["args"])
                except Exception as exc:
                    result = f"Tool error: {exc}"
            messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )
        response = llm.invoke(messages)
    return _message_text(response.content).strip() or "Please try that question again."


def _normalize_date_input(value) -> str:
    if value is None or value == "":
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value).strip()


def chat(message: str, history: list, start_date=None, end_date=None) -> str:
    start_date = _normalize_date_input(start_date)
    end_date = _normalize_date_input(end_date)
    user_text = message or ""
    if start_date and end_date:
        user_text = (
            f"{user_text}\n\nStart date: {start_date}\nEnd date: {end_date}\n"
            "Use the count_working_days tool with these dates."
        ).strip()
    elif start_date or end_date:
        user_text = (
            f"{user_text}\n\nPartial dates provided — start: {start_date or '(missing)'}, "
            f"end: {end_date or '(missing)'}. Ask for the missing date if needed."
        ).strip()

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
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

    last_error = None
    for model in MODEL_CANDIDATES:
        try:
            return _run_with_tools(_llm(model), list(messages))
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


demo = gr.ChatInterface(
    fn=chat,
    title="HR Assistant",
    description=(
        "Ask HR questions, or enter a start date and end date to count "
        "working days (Monday to Friday)."
    ),
    additional_inputs=[
        gr.Textbox(label="Start date", placeholder="YYYY-MM-DD", value=""),
        gr.Textbox(label="End date", placeholder="YYYY-MM-DD", value=""),
    ],
    examples=[
        ["How many working days between 2026-09-01 and 2026-09-15?", "2026-09-01", "2026-09-15"],
        ["What should I include in an onboarding checklist?", "", ""],
        ["How do I handle a conflict between two teammates?", "", ""],
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
