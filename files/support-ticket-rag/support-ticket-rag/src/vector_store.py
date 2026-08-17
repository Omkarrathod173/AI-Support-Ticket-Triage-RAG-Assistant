"""
vector_store.py — semantic retrieval over the knowledge base.

Two backends behind one interface:
- ChromaVectorStore: the real production path (Chroma + OpenAI or
  sentence-transformers embeddings). Requires `chromadb` and either an
  OpenAI API key or `sentence-transformers` with internet access to
  download the embedding model.
- TFIDFVectorStore: an offline fallback using scikit-learn's TF-IDF +
  cosine similarity. Not true semantic embedding (no synonym understanding),
  but same interface, always runs, and is what powers the results in
  reports/findings.md since this sandbox has neither internet nor an API key.

Swap VECTOR_BACKEND below once you have API access.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

VECTOR_BACKEND = "tfidf"  # "tfidf" (offline, default) or "chroma" (production)

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


class TFIDFVectorStore:
    """Offline fallback: TF-IDF vectors + cosine similarity search."""

    def __init__(self, documents: pd.DataFrame, text_col: str, id_col: str):
        self.documents = documents.reset_index(drop=True)
        self.text_col = text_col
        self.id_col = id_col
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.doc_vectors = self.vectorizer.fit_transform(self.documents[text_col])

    def query(self, text: str, top_k=3):
        q_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(q_vec, self.doc_vectors).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = self.documents.iloc[top_idx].copy()
        results["similarity"] = sims[top_idx]
        return results


class ChromaVectorStore:
    """Production path: real vector embeddings via Chroma. Requires
    `chromadb` and an embedding function (OpenAI API or
    sentence-transformers) — not runnable in this offline sandbox, included
    for when you deploy with real credentials."""

    def __init__(self, documents: pd.DataFrame, text_col: str, id_col: str):
        if not HAS_CHROMA:
            raise ImportError("chromadb not installed — pip install chromadb")
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("support_kb")
        self.collection.add(
            documents=documents[text_col].tolist(),
            ids=documents[id_col].tolist(),
        )
        self.documents = documents

    def query(self, text: str, top_k=3):
        results = self.collection.query(query_texts=[text], n_results=top_k)
        ids = results["ids"][0]
        matched = self.documents[self.documents[self.documents.columns[0]].isin(ids)]
        return matched


def get_vector_store(documents, text_col, id_col):
    if VECTOR_BACKEND == "chroma" and HAS_CHROMA:
        return ChromaVectorStore(documents, text_col, id_col)
    return TFIDFVectorStore(documents, text_col, id_col)


if __name__ == "__main__":
    kb = pd.read_csv("data/knowledge_base.csv")
    store = get_vector_store(kb, text_col="issue_text", id_col="kb_id")
    print(f"Backend: {'Chroma' if HAS_CHROMA and VECTOR_BACKEND == 'chroma' else 'TF-IDF (offline fallback)'}")

    test_query = "My card was billed twice this month, can you fix it?"
    results = store.query(test_query, top_k=3)
    print(f"\nQuery: {test_query}")
    print(results[["kb_id", "category", "issue_text", "similarity"]] if "similarity" in results else results)
