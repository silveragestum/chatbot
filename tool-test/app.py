"""HR assistant chatbot: LangChain agent + Mistral + Gradio."""

from datetime import date, datetime, timedelta
from email.message import EmailMessage
import json
import os
from pathlib import Path
import re
import smtplib

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gradio as gr

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
POLICY_FILENAME = "Unisoft Human Resources Policy.pdf"
CHROMA_DIR = ROOT / "chroma_unisoft_hr"
_vectorstore = None

SYSTEM_PROMPT = """You are an HR assistant for Unisoft.

Holiday / leave applications:
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
5. ONLY after the user clearly confirms the details (for example: yes, correct, confirmed,
   looks good), you MUST call send_confirmation_email with the confirmed fields so a
   Gmail confirmation is sent to the applicant. Do not send email before they confirm.
   After the tool runs, tell the user whether the email was sent.

Unisoft policy questions:
For any question about Unisoft HR rules, leave entitlements, working hours, sick leave,
remote work, probation, conduct, or this company policy, you MUST call
search_unisoft_hr_policy with the user's question. Answer using only the retrieved
policy text. If the retrieved text does not contain the answer, say the Unisoft Human
Resources Policy does not specify it. Do not invent policy.

Working days are Monday through Friday only. Do not invent holidays.
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


def find_policy_pdf() -> Path:
    candidates = [
        ROOT / POLICY_FILENAME,
        ROOT.parent / POLICY_FILENAME,
        Path.cwd() / POLICY_FILENAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not find {POLICY_FILENAME} in {ROOT}, {ROOT.parent}, or {Path.cwd()}."
    )


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    pdf_path = find_policy_pdf()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name="unisoft_hr_policy",
        )
        return _vectorstore
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="unisoft_hr_policy",
    )
    return _vectorstore


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
        "Ask the user to confirm whether this information is correct. "
        "Do not send email until they confirm."
    )


_runtime_gmail = {"gmail_address": "", "app_password": ""}


def load_gmail_credentials() -> tuple[str, str]:
    address = (
        (_runtime_gmail.get("gmail_address") or "").strip()
        or os.getenv("GMAIL_ADDRESS", "").strip()
    )
    password = (
        (_runtime_gmail.get("app_password") or "").strip()
        or os.getenv("GMAIL_APP_PASSWORD", "").strip()
    )
    config_path = ROOT / "gmail_config.json"
    if config_path.exists() and (not address or not password):
        data = json.loads(config_path.read_text(encoding="utf-8"))
        address = address or str(data.get("gmail_address") or "").strip()
        password = password or str(data.get("app_password") or "").strip()
    return address, password.replace(" ", "")


def build_confirmation_email_body(
    user_name: str,
    email: str,
    purpose: str,
    start_date: str,
    end_date: str,
    working_days: str,
) -> str:
    return (
        f"Hello {user_name},\n\n"
        "This is confirmation of your holiday / leave application.\n\n"
        f"User name: {user_name}\n"
        f"Email: {email}\n"
        f"Purpose: {purpose}\n"
        f"Start Date: {start_date}\n"
        f"End Date: {end_date}\n"
        f"Working days (Monday to Friday): {working_days}\n\n"
        "If anything is incorrect, reply to HR.\n\n"
        "HR Assistant\n"
    )


def send_gmail_message(to_email: str, subject: str, body: str) -> str:
    address, password = load_gmail_credentials()
    if not address or not password:
        return (
            "Gmail is not configured. Create a Google App Password at "
            "https://myaccount.google.com/apppasswords then set GMAIL_ADDRESS and "
            "GMAIL_APP_PASSWORD, fill tool-test/gmail_config.json, or enter them in the UI."
        )
    message = EmailMessage()
    message["From"] = address
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(body)
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(address, password)
        smtp.send_message(message)
    return f"Confirmation email sent to {to_email} from {address}."


@tool
def send_confirmation_email(
    user_name: str,
    email: str,
    purpose: str,
    start_date: str,
    end_date: str,
    working_days: str,
) -> str:
    """Send a holiday confirmation email via Gmail after the user confirms the details.

    Call this only when the user has confirmed that User name, Email, Purpose,
    Start Date, End Date, and working days are correct.

    Args:
        user_name: Confirmed applicant name.
        email: Applicant email address (recipient).
        purpose: Confirmed leave purpose.
        start_date: Confirmed start date.
        end_date: Confirmed end date.
        working_days: Working-day count already calculated by count_working_days.
    """
    mail = (email or "").strip()
    if not re.search(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", mail):
        return f"Cannot send email: '{email}' is not a valid address."
    body = build_confirmation_email_body(
        user_name=user_name.strip(),
        email=mail,
        purpose=(purpose or "").strip(),
        start_date=(start_date or "").strip(),
        end_date=(end_date or "").strip(),
        working_days=(working_days or "").strip(),
    )
    try:
        return send_gmail_message(
            to_email=mail,
            subject="Holiday application confirmation",
            body=body,
        )
    except smtplib.SMTPAuthenticationError:
        return (
            "Gmail login failed. Use an App Password (not your normal Gmail password) from "
            "https://myaccount.google.com/apppasswords and make sure 2-Step Verification is on."
        )
    except Exception as exc:
        return f"Could not send confirmation email: {exc}"


@tool
def search_unisoft_hr_policy(question: str) -> str:
    """Search the Unisoft Human Resources Policy PDF (Chroma vector store) for answers.

    Use this for any question about Unisoft leave, hours, conduct, remote work, or HR rules.

    Args:
        question: The employee's question about the Unisoft HR policy.
    """
    store = get_vectorstore()
    docs = store.similarity_search((question or "").strip(), k=4)
    if not docs:
        return "No matching passages were found in Unisoft Human Resources Policy.pdf."
    parts = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page")
        page_note = f" (page {page + 1})" if isinstance(page, int) else ""
        parts.append(f"Passage {i}{page_note}:\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


TOOLS = [count_working_days, record_holiday_application, send_confirmation_email, search_unisoft_hr_policy]

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


def chat(
    message: str,
    history: list | None,
    start_date=None,
    end_date=None,
    gmail_address=None,
    gmail_app_password=None,
) -> str:
    if isinstance(message, dict):
        message = str(message.get("text") or message.get("content") or "")
    history = history or []
    _runtime_gmail["gmail_address"] = (gmail_address or "").strip()
    _runtime_gmail["app_password"] = (gmail_app_password or "").strip()
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
        "Apply for holiday leave in chat, or ask any question about the "
        "**Unisoft Human Resources Policy** (RAG over Unisoft Human Resources Policy.pdf). "
        "The agent extracts **User name**, **Email**, **Purpose**, **Start Date**, and **End Date**, "
        "counts **working days** (Monday to Friday), asks you to confirm, then can send a "
        "**Gmail confirmation**.\n\n"
        "Create a Gmail App Password at [myaccount.google.com/apppasswords]"
        "(https://myaccount.google.com/apppasswords) (2-Step Verification required)."
    )
    chatbot = gr.Chatbot(height=480)
    with gr.Row():
        start_in = gr.Textbox(label="Start date", placeholder="YYYY-MM-DD")
        end_in = gr.Textbox(label="End date", placeholder="YYYY-MM-DD")
    with gr.Accordion("Gmail (confirmation email)", open=True):
        gmail_in = gr.Textbox(label="Gmail address (sender)", placeholder="you@gmail.com")
        gmail_pw = gr.Textbox(
            label="Gmail App Password",
            placeholder="16-character app password",
            type="password",
        )
    msg_in = gr.Textbox(
        label="Message",
        placeholder="e.g. I am Jane Doe, jane@acme.com. Annual leave 1–15 Sep 2026 for a family trip.",
    )
    send = gr.Button("Send", variant="primary")

    def respond(message, history, start_date, end_date, gmail_address, gmail_app_password):
        history = history or []
        if not (message or "").strip() and not (start_date and end_date):
            return history, message
        display = (message or "").strip()
        if start_date and end_date:
            display = display or f"Holiday dates from {start_date} to {end_date}"
        reply = chat(
            message,
            history,
            start_date,
            end_date,
            gmail_address,
            gmail_app_password,
        )
        history = history + [
            {"role": "user", "content": display},
            {"role": "assistant", "content": reply},
        ]
        return history, ""

    inputs = [msg_in, chatbot, start_in, end_in, gmail_in, gmail_pw]
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
                "How many annual leave days do Unisoft employees get in their first year?",
                "",
                "",
            ],
        ],
        inputs=[msg_in, start_in, end_in],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
