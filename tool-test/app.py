"""HR assistant chatbot: LangChain + Mistral + Gradio."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
import gradio as gr

load_dotenv(Path(__file__).resolve().parent / ".env")

SYSTEM_PROMPT = (
    "You are an HR assistant. Help employees and managers with human-resources "
    "questions: policies, leave, benefits, hiring, onboarding, performance, "
    "workplace conduct, and related processes. Be clear, professional, and "
    "practical. If a question is outside HR, say so and steer back to HR topics. "
    "This is general guidance, not legal advice."
)

api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise RuntimeError(
        "Set MISTRAL_API_KEY in the environment or in tool-test/.env"
    )

llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=api_key,
    temperature=0.4,
    max_retries=6,
)


def chat(message: str, history: list) -> str:
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history:
        if isinstance(turn, dict):
            role, content = turn.get("role"), turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            user_text, assistant_text = turn
            if user_text:
                messages.append(HumanMessage(content=user_text))
            if assistant_text:
                messages.append(AIMessage(content=assistant_text))
    messages.append(HumanMessage(content=message))
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as exc:
        text = str(exc)
        if "429" in text or "rate_limited" in text:
            return (
                "The Mistral API rate limit was reached. Wait a moment and try again."
            )
        return f"Sorry, I could not complete that request: {exc}"


demo = gr.ChatInterface(
    fn=chat,
    title="HR Assistant",
    description="Ask questions about HR policies, leave, hiring, benefits, and workplace issues.",
    examples=[
        "How many vacation days do employees typically get in their first year?",
        "What should I include in an onboarding checklist?",
        "How do I handle a conflict between two teammates?",
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
