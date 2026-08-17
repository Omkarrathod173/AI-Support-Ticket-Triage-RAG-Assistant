"""
classification.py — zero-shot category classification for incoming tickets.

Two paths behind one interface:
- HFZeroShotClassifier: the real production path, Hugging Face's
  zero-shot-classification pipeline (facebook/bart-large-mnli by default).
  Requires `transformers` + `torch` and internet access to download the
  model on first use.
- SimilarityZeroShotClassifier: an offline fallback that scores each
  category by cosine similarity between the ticket text and a short
  natural-language description of that category — the same "compare text
  to label descriptions, no task-specific training" idea zero-shot
  classification is built on, just via TF-IDF instead of a trained NLI
  model. Same interface, always runs.

This is what powers the results in reports/findings.md, since this sandbox
has no internet access to download the real model.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from transformers import pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

CATEGORY_DESCRIPTIONS = {
    "Billing": "a problem with a charge, invoice, payment, or subscription cost",
    "Technical Issue": "a bug, crash, error, or feature not working correctly in the app",
    "Account Access": "trouble logging in, resetting a password, or account security",
    "Shipping/Delivery": "a problem with an order's delivery, tracking, or shipment",
    "Cancellation/Refund": "a request to cancel a subscription or receive a refund",
}


class HFZeroShotClassifier:
    def __init__(self, labels):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers not installed — pip install transformers torch")
        self.labels = labels
        self.pipe = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def classify(self, text: str):
        result = self.pipe(text, self.labels)
        return result["labels"][0], dict(zip(result["labels"], result["scores"]))


class SimilarityZeroShotClassifier:
    """Offline fallback: cosine similarity between ticket text and each
    category's natural-language description."""

    def __init__(self, category_descriptions=None):
        self.descriptions = category_descriptions or CATEGORY_DESCRIPTIONS
        self.labels = list(self.descriptions.keys())
        self.vectorizer = TfidfVectorizer(stop_words="english")
        # Fit on both the descriptions themselves so the vocabulary covers
        # category-relevant terms
        self.vectorizer.fit(list(self.descriptions.values()))
        self.desc_vectors = self.vectorizer.transform(list(self.descriptions.values()))

    def classify(self, text: str):
        vec = self.vectorizer.transform([text])
        sims = cosine_similarity(vec, self.desc_vectors).flatten()
        scores = dict(zip(self.labels, sims))
        best_label = self.labels[int(np.argmax(sims))]
        return best_label, scores


def get_classifier(labels=None):
    labels = labels or list(CATEGORY_DESCRIPTIONS.keys())
    if HAS_TRANSFORMERS:
        return HFZeroShotClassifier(labels)
    return SimilarityZeroShotClassifier()


if __name__ == "__main__":
    print(f"Backend: {'HuggingFace zero-shot (BART-MNLI)' if HAS_TRANSFORMERS else 'TF-IDF label-similarity (offline fallback)'}")
    clf = get_classifier()

    tickets = pd.read_csv("data/tickets.csv").sample(5, random_state=1)
    for _, row in tickets.iterrows():
        pred, scores = clf.classify(row["text"])
        correct = "✓" if pred == row["true_category"] else "✗"
        print(f"\n{correct} True: {row['true_category']:20s} Pred: {pred}")
        print(f"   Text: {row['text'][:80]}...")
