# Calle Apps — Meeting Room Chatbot

A Gradio chatbot that books meeting rooms for clients. It uses LangChain with Mistral and an in-memory room calendar.

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

## Rooms

| Room   | Capacity | Features |
|--------|----------|----------|
| Aurora | 4        | TV, whiteboard, video conferencing |
| Harbor | 8        | projector, whiteboard, phone |
| Summit | 16       | video conferencing, projector, catering table |
| Nook   | 2        | quiet booth, webcam |

Bookings live in process memory and reset when the app restarts.
