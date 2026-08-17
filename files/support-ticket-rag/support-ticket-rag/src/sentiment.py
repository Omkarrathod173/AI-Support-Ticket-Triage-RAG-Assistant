"""
sentiment.py — sentiment and urgency scoring for incoming tickets.

Two paths behind one interface:
- HFSentimentAnalyzer: real production path, Hugging Face's
  sentiment-analysis pipeline (distilbert-base-uncased-finetuned-sst-2 by
  default). Requires `transformers` + `torch` + internet on first use.
- LexiconSentimentAnalyzer: offline fallback using a small curated
  word/phrase lexicon for sentiment and urgency. Simpler than a trained
  transformer, but transparent and always runs — this is what powers the
  results in reports/findings.md.
"""

try:
    from transformers import pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

NEGATIVE_WORDS = {
    "frustrating", "frustrated", "upset", "unacceptable", "disappointed", "angry",
    "furious", "terrible", "worst", "annoyed", "ridiculous",
}
POSITIVE_WORDS = {
    "thanks", "appreciate", "great", "helpful", "no rush",
}
URGENT_WORDS = {
    "immediately", "urgent", "asap", "today", "now", "emergency", "critical",
}


class HFSentimentAnalyzer:
    def __init__(self):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers not installed — pip install transformers torch")
        self.pipe = hf_pipeline("sentiment-analysis")

    def analyze(self, text: str):
        result = self.pipe(text)[0]
        sentiment = "Positive" if result["label"] == "POSITIVE" else "Negative"
        urgency = "High" if any(w in text.lower() for w in URGENT_WORDS) else "Medium"
        return sentiment, urgency


class LexiconSentimentAnalyzer:
    """Offline fallback: keyword lexicon for sentiment and urgency."""

    def analyze(self, text: str):
        text_lower = text.lower()
        neg_hits = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
        pos_hits = sum(1 for w in POSITIVE_WORDS if w in text_lower)

        if neg_hits > pos_hits:
            sentiment = "Negative"
        elif pos_hits > neg_hits:
            sentiment = "Positive"
        else:
            sentiment = "Neutral"

        urgency = "High" if any(w in text_lower for w in URGENT_WORDS) else \
                  ("Medium" if neg_hits > 0 else "Low")

        return sentiment, urgency


def get_sentiment_analyzer():
    if HAS_TRANSFORMERS:
        return HFSentimentAnalyzer()
    return LexiconSentimentAnalyzer()


if __name__ == "__main__":
    import pandas as pd

    print(f"Backend: {'HuggingFace DistilBERT-SST2' if HAS_TRANSFORMERS else 'Lexicon-based (offline fallback)'}")
    analyzer = get_sentiment_analyzer()

    tickets = pd.read_csv("data/tickets.csv").sample(5, random_state=2)
    for _, row in tickets.iterrows():
        sentiment, urgency = analyzer.analyze(row["text"])
        s_match = "✓" if sentiment == row["true_sentiment"] else "✗"
        u_match = "✓" if urgency == row["true_urgency"] else "✗"
        print(f"\n{s_match} Sentiment: true={row['true_sentiment']:10s} pred={sentiment}")
        print(f"{u_match} Urgency:   true={row['true_urgency']:10s} pred={urgency}")
        print(f"   Text: {row['text'][:80]}...")
