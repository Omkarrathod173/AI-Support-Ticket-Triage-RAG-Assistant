"""
rag_assistant.py — orchestrates the full pipeline for one incoming ticket:
classify category -> score sentiment/urgency -> retrieve similar past
resolutions -> draft a suggested response.

Two paths for the final response-drafting step:
- LangChain + OpenAI: the real production path — takes the retrieved KB
  context and asks an LLM to draft a response grounded in it. Requires
  `langchain`, `openai`, and a valid OPENAI_API_KEY (and internet).
- Extractive fallback: builds a response directly from the top-retrieved
  resolution text, no generation involved. No API key or internet needed
  — this is what powers the results in reports/findings.md.
"""

import os
import pandas as pd

from src.vector_store import get_vector_store
from src.classification import get_classifier
from src.sentiment import get_sentiment_analyzer

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    HAS_LANGCHAIN_OPENAI = True
except ImportError:
    HAS_LANGCHAIN_OPENAI = False


RESPONSE_PROMPT = """You are a support agent. A customer wrote:
"{ticket_text}"

Here is the most relevant past resolution for a similar issue:
"{retrieved_resolution}"

Write a short, empathetic response to the customer using this resolution as guidance."""


class SupportRAGAssistant:
    def __init__(self, kb_df: pd.DataFrame):
        self.kb_df = kb_df
        self.vector_store = get_vector_store(kb_df, text_col="issue_text", id_col="kb_id")
        self.classifier = get_classifier()
        self.sentiment_analyzer = get_sentiment_analyzer()
        self.use_llm = HAS_LANGCHAIN_OPENAI and bool(os.environ.get("OPENAI_API_KEY"))
        if self.use_llm:
            self.llm = ChatOpenAI(model="gpt-4o-mini")

    def handle_ticket(self, text: str, top_k=3):
        category, category_scores = self.classifier.classify(text)
        sentiment, urgency = self.sentiment_analyzer.analyze(text)
        retrieved = self.vector_store.query(text, top_k=top_k)
        top_resolution = retrieved.iloc[0]["resolution_text"]

        if self.use_llm:
            prompt = ChatPromptTemplate.from_template(RESPONSE_PROMPT)
            chain = prompt | self.llm
            response = chain.invoke(
                {"ticket_text": text, "retrieved_resolution": top_resolution}
            ).content
        else:
            response = (
                f"Thanks for reaching out. Based on similar past cases, here's how "
                f"we can resolve this: {top_resolution} "
                f"(Ticket categorized as {category}, priority: {urgency}.)"
            )

        return {
            "category": category,
            "category_scores": category_scores,
            "sentiment": sentiment,
            "urgency": urgency,
            "retrieved": retrieved,
            "suggested_response": response,
            "used_llm": self.use_llm,
        }


if __name__ == "__main__":
    kb_df = pd.read_csv("data/knowledge_base.csv")
    assistant = SupportRAGAssistant(kb_df)

    print(f"Response generation: {'LangChain + OpenAI' if assistant.use_llm else 'Extractive (offline fallback, no API key set)'}")

    test_ticket = "Hi, I was charged twice for my subscription this month and I'm pretty frustrated, can someone fix this ASAP?"
    result = assistant.handle_ticket(test_ticket)

    print(f"\nTicket: {test_ticket}")
    print(f"Category: {result['category']}")
    print(f"Sentiment: {result['sentiment']}  Urgency: {result['urgency']}")
    print(f"\nTop retrieved KB match: {result['retrieved'].iloc[0]['kb_id']} — {result['retrieved'].iloc[0]['issue_text']}")
    print(f"\nSuggested response:\n{result['suggested_response']}")
