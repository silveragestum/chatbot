# Calle Apps — Meeting Room Chatbot

A Gradio chatbot that books meeting rooms for clients. A LangChain agent extracts booking details and a `save_booking` tool writes them to SQLite only when there is no conflict.

## Setup

```bash
cd calle-apps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MISTRAL_API_KEY="your-key"
python app.py
```

Open the Gradio URL (default `http://127.0.0.1:7860`).

## Rooms and hours

Rooms **A**, **B**, and **C**. Bookings allowed **09:00–21:00**.

Each saved row includes: person name, email, meeting purpose, date, start time, end time, and room number.

Overlapping times on the same room and date are rejected and not stored. The database file is `calle-apps/bookings.db`.

## Tests

```bash
python -m unittest test_booking.py
```
