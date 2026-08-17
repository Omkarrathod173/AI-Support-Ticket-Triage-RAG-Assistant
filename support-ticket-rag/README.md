# AI Support Ticket Triage & RAG Resolution Assistant

An end-to-end RAG (Retrieval-Augmented Generation) pipeline for customer support: an
incoming ticket is classified by category (zero-shot), scored for sentiment and urgency,
matched against a knowledge base of past resolutions via semantic search, and used to
draft a grounded suggested response — deployed as an interactive Gradio app.

## TL;DR

Across 150 evaluated tickets: **81.3%** category classification accuracy, **93.3%**
sentiment accuracy, and the correct past resolution retrieved in the top-3 results
**40.0%** of the time. All of this runs on **offline fallbacks** (TF-IDF instead of real
embeddings, a keyword lexicon instead of a trained sentiment model) because this was built
without internet access or an API key — full numbers, per-category breakdown, and an honest
read on where the offline pipeline struggles: [`reports/findings.md`](reports/findings.md).

## ⚠️ About the Data and the "Offline Mode"

This project was built in a sandbox with **no internet access and no API key**, which
matters twice over for an LLM/RAG project:

1. **No real dataset** — uses a synthetic ticket + knowledge-base generator
   (`data/generate_synthetic_tickets.py`) instead of real support logs.
2. **No real models** — `transformers`, `langchain-openai`, and `chromadb` can't download
   models or call an API without internet/a key. Every component has a clean, documented
   offline fallback so the full pipeline still runs and produces real, honest numbers:

| Component | Production path | Offline fallback (used in this run) |
|---|---|---|
| Category classification | HuggingFace zero-shot (BART-MNLI) | TF-IDF similarity to category descriptions |
| Sentiment/urgency | HuggingFace DistilBERT-SST2 | Keyword lexicon |
| Retrieval | Chroma + real embeddings | TF-IDF + cosine similarity |
| Response drafting | LangChain + OpenAI (GPT-4o-mini) | Extractive from top retrieval |

**To run the real versions:** `pip install -r requirements.txt`, `export
OPENAI_API_KEY=your_key`, and in `src/vector_store.py` set `VECTOR_BACKEND = "chroma"`.
Every module auto-detects what's installed — no other code changes needed. Re-run
`main.py` afterward; expect retrieval and urgency accuracy especially to improve, since
those are where the offline fallbacks are weakest (see findings.md for why).

## Project Structure

```
support-ticket-rag/
├── main.py                          # evaluates all tickets, writes reports/findings.md
├── data/
│   └── generate_synthetic_tickets.py
├── src/
│   ├── vector_store.py              # Chroma (production) / TF-IDF (fallback) retrieval
│   ├── classification.py            # HF zero-shot (production) / TF-IDF similarity (fallback)
│   ├── sentiment.py                 # HF sentiment (production) / lexicon (fallback)
│   └── rag_assistant.py             # orchestrates all of the above end to end
├── app/
│   └── gradio_app.py                # interactive demo
├── reports/
│   ├── findings.md                  # auto-generated, real numbers
│   └── evaluation_results.csv       # per-ticket predictions, for your own analysis
└── requirements.txt
```

## Setup & Run

```bash
pip install -r requirements.txt   # offline fallbacks work even without the optional packages

python3 data/generate_synthetic_tickets.py   # generate the ticket + KB data
python3 main.py                               # evaluate the full pipeline, write findings.md
python3 app/gradio_app.py                     # launch the interactive demo
```

Every module also runs standalone for debugging, e.g. `python3 -m src.classification`.

## Why This Isn't Just Another "Chatbot Wrapper" Project

Most student LLM projects call an API and report that it "works." This one:
- **Measures retrieval quality separately from generation quality** (hit-rate@k), because a
  fluent response grounded in the wrong retrieved case is a worse failure mode than a rough
  response grounded in the right one — and it's invisible if you only eyeball outputs.
- **Reports per-category accuracy breakdowns and the specific confusion pairs**, not one
  aggregate number.
- **Explains its own weakest metric** (urgency accuracy) with a concrete mechanism — the
  heuristic conflates urgency with negative sentiment, while the ground truth treats them
  as independent — rather than reporting a lower number without comment.
- **Built with graceful fallbacks by design**, so the architecture separates "what the
  method does" from "which specific model/API executes it" — the kind of separation that
  matters in a real production system where you might swap embedding providers or LLMs.

## Interview Prep

- Why measure retrieval hit-rate separately from response quality?
- What's the actual difference between TF-IDF similarity and a real embedding model —
  can you give a concrete example where they'd disagree? (See the findings.md discussion.)
- Why is urgency accuracy the weakest metric, and how would you fix it?
- How would zero-shot classification (BART-MNLI) actually work under the hood, versus the
  TF-IDF fallback used here?
- What would change about this system's reliability requirements moving from a demo to
  production (rate limits, cost per ticket, latency, hallucination risk in generated
  responses)?

## Limitations

- Synthetic data, not real support tickets — see above.
- Offline fallbacks are meaningfully weaker than the production stack, especially for
  retrieval and urgency — this is documented and quantified in `findings.md`, not hidden.
- The knowledge base is small (25 entries) — a real deployment needs a much larger,
  continuously updated KB, plus a strategy for keeping embeddings in sync as it grows.
