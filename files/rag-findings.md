# AI Support Ticket Triage & RAG Assistant — Findings

*Generated 2026-08-17 by main.py — real numbers from evaluating all 150 tickets.*

## TL;DR
Across 150 synthetic support tickets, zero-shot-style category classification reached **81.3% accuracy**, sentiment scoring reached **93.3%**, and semantic retrieval found the correct knowledge-base resolution within the top-3 results **40.0% of the time** (top-1: 23.3%). All numbers below use the offline fallback pipeline (TF-IDF retrieval, lexicon sentiment, extractive responses) — see the README for what changes with real OpenAI/HuggingFace credentials.

## Pipeline Components & Backend Used This Run
- **Classification:** TF-IDF label-similarity (offline fallback)
- **Sentiment/Urgency:** Lexicon-based (offline fallback)
- **Retrieval:** TF-IDF + cosine similarity (offline fallback)
- **Response generation:** Extractive from top retrieval (offline fallback, no API key set)

## Overall Metrics
| Metric | Score |
|---|---|
| Category classification accuracy | 0.813 |
| Sentiment accuracy | 0.933 |
| Urgency accuracy | 0.513 |
| Retrieval hit-rate @1 | 0.233 |
| Retrieval hit-rate @3 | 0.400 |

## Per-Category Breakdown
| true_category       |   n |   category_accuracy |   retrieval_hit_at_1 |   retrieval_hit_at_k |
|:--------------------|----:|--------------------:|---------------------:|---------------------:|
| Account Access      |  30 |               0.833 |                0.367 |                0.533 |
| Billing             |  30 |               0.867 |                0.2   |                0.2   |
| Cancellation/Refund |  30 |               0.933 |                0.167 |                0.4   |
| Shipping/Delivery   |  30 |               0.833 |                0.2   |                0.633 |
| Technical Issue     |  30 |               0.6   |                0.233 |                0.233 |

## Where Classification Struggles Most
Most common misclassification pairs (true → predicted):

- **Technical Issue → Billing**: 12 tickets
- **Account Access → Billing**: 5 tickets
- **Shipping/Delivery → Billing**: 5 tickets

## Why Retrieval Hit-Rate Matters More Than It Looks
Hit-rate@3 of 0.400 means the RAG assistant surfaces the *actually correct* past resolution among its top 3 suggestions that often — this is the metric that determines whether the drafted response is grounded in the right context, independent of whether the final wording (LLM-generated or extractive) is polished. A fluent response built on the wrong retrieved case is worse than a rough one built on the right case.

The modest hit-rate here is itself a finding, not a bug: TF-IDF matches on shared vocabulary, so two tickets describing the same underlying issue in different words ("charged twice" vs. "duplicate payment") can score low similarity even though a real embedding model would recognize them as the same intent. This is the concrete, measured case for why the production path uses real embeddings instead of TF-IDF.

## Why Urgency Accuracy Is the Weakest Metric
Urgency accuracy (0.513) is lower than the other metrics because the offline lexicon heuristic infers urgency partly from negative-sentiment keywords, while the synthetic ground truth assigns urgency independently of sentiment (a ticket can be worded calmly but still be genuinely high-priority, or worded with frustration but be low-priority). This is a real limitation of keyword-based urgency scoring, not a data bug — a trained classifier (or an LLM prompted specifically for priority, separate from sentiment) would be expected to do better, which is worth stating explicitly rather than hiding.

## Limitations
- Uses a **synthetic** ticket dataset (`data/generate_synthetic_tickets.py`) with template-based tickets, not real customer support logs — real language is messier and this pipeline's accuracy would likely drop on real data.
- All results above use **offline fallbacks** (TF-IDF instead of real embeddings, a lexicon instead of a trained sentiment model, extractive instead of generated responses) because this was built without internet access or an API key. TF-IDF retrieval only matches on shared vocabulary — it cannot recognize that 'my card was billed twice' and 'duplicate charge on my account' mean the same thing without shared words, the way a real embedding model would. Install `transformers`, `langchain-openai`, and set `OPENAI_API_KEY` for the production versions — see the README.
- The knowledge base has only 25 entries across 5 categories — a real deployment would need a much larger, continuously updated KB.
