"""
gradio_app.py — interactive demo for the support ticket RAG assistant.

Run with: python3 app/gradio_app.py
(run from the project root so relative paths resolve)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import pandas as pd

from src.rag_assistant import SupportRAGAssistant

kb_df = pd.read_csv("data/knowledge_base.csv")
assistant = SupportRAGAssistant(kb_df)


def process_ticket(ticket_text):
    if not ticket_text.strip():
        return "Please enter a ticket.", "", "", ""

    result = assistant.handle_ticket(ticket_text)

    category_str = f"**{result['category']}**"
    sentiment_str = f"Sentiment: **{result['sentiment']}** | Urgency: **{result['urgency']}**"

    retrieved_str = "\n\n".join(
        f"**{row['kb_id']}** ({row['category']}, similarity={row.get('similarity', 0):.2f})\n"
        f"Issue: {row['issue_text']}\nResolution: {row['resolution_text']}"
        for _, row in result["retrieved"].iterrows()
    )

    response_str = result["suggested_response"]
    backend_note = "🔗 Generated via LangChain + OpenAI" if result["used_llm"] else "📋 Extractive (offline fallback — set OPENAI_API_KEY for LLM-generated responses)"

    return category_str, sentiment_str, retrieved_str, f"{response_str}\n\n*{backend_note}*"


with gr.Blocks(title="Support Ticket RAG Assistant") as demo:
    gr.Markdown("# 🎫 AI Support Ticket Triage & Resolution Assistant")
    gr.Markdown(
        "Paste an incoming support ticket below. The assistant classifies its category, "
        "scores sentiment/urgency, retrieves the most relevant past resolution via semantic "
        "search, and drafts a suggested response."
    )

    with gr.Row():
        with gr.Column():
            ticket_input = gr.Textbox(
                label="Incoming Ticket",
                placeholder="e.g. I was charged twice for my subscription this month, please help ASAP!",
                lines=4,
            )
            submit_btn = gr.Button("Analyze Ticket", variant="primary")

        with gr.Column():
            category_output = gr.Markdown(label="Predicted Category")
            sentiment_output = gr.Markdown(label="Sentiment & Urgency")

    retrieved_output = gr.Markdown(label="Retrieved Similar Cases")
    response_output = gr.Markdown(label="Suggested Response")

    submit_btn.click(
        process_ticket,
        inputs=[ticket_input],
        outputs=[category_output, sentiment_output, retrieved_output, response_output],
    )

    gr.Examples(
        examples=[
            "I've been charged twice for my March subscription and I'm really frustrated, please fix this immediately.",
            "The app keeps crashing whenever I open the reports screen.",
            "My order was supposed to arrive 10 days ago and tracking hasn't updated.",
        ],
        inputs=ticket_input,
    )

if __name__ == "__main__":
    demo.launch()
