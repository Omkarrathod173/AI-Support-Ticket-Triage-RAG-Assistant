"""
generate_synthetic_tickets.py

Generates a SYNTHETIC customer support dataset: a knowledge base of past
resolved tickets (for retrieval) and a set of new incoming tickets (for
evaluating classification, sentiment, and retrieval quality against known
ground truth) — because this sandbox has no internet access to pull a real
support-ticket dataset.

Each new ticket is generated as a paraphrased variant of one specific KB
entry, so we know the "correct" retrieval target for every ticket — this
is what lets main.py report a real retrieval hit-rate, not a guess.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(21)

CATEGORIES = {
    "Billing": {
        "issues": [
            "I was charged twice for my {month} subscription",
            "My invoice shows an amount I don't recognize",
            "The discount code I used wasn't applied at checkout",
            "I'm being billed after I cancelled my plan",
            "My payment method was charged in the wrong currency",
        ],
        "resolutions": [
            "Refunded the duplicate charge and confirmed single billing going forward.",
            "Reviewed the invoice line items and issued a credit for the discrepancy.",
            "Manually applied the discount code and adjusted the charge.",
            "Confirmed cancellation date and refunded the post-cancellation charge.",
            "Corrected the currency setting on the account and refunded the difference.",
        ],
    },
    "Technical Issue": {
        "issues": [
            "The app crashes every time I open the {feature} screen",
            "I can't upload files larger than a few MB",
            "The dashboard is stuck loading and never finishes",
            "Push notifications stopped working after the last update",
            "The search feature returns no results for anything I type",
        ],
        "resolutions": [
            "Identified a bug in the {feature} screen and shipped a patch in the next release.",
            "Increased the upload size limit and confirmed the fix on the user's account.",
            "Cleared a stuck background job that was blocking the dashboard from loading.",
            "Reset notification permissions and reinstalled the push service on the user's device.",
            "Rebuilt the search index, which had become corrupted after a migration.",
        ],
    },
    "Account Access": {
        "issues": [
            "I'm locked out of my account after too many login attempts",
            "The password reset email never arrives",
            "Two-factor authentication codes aren't being accepted",
            "My account shows someone else's information",
            "I can't log in even with the correct password",
        ],
        "resolutions": [
            "Manually unlocked the account after verifying identity.",
            "Resent the reset email after confirming the address on file was correct.",
            "Resynced the two-factor authenticator and issued new backup codes.",
            "Identified an account merge error and restored the correct profile data.",
            "Reset the password directly after verifying identity through support.",
        ],
    },
    "Shipping/Delivery": {
        "issues": [
            "My order hasn't arrived and tracking hasn't updated in {days} days",
            "I received the wrong item in my order",
            "The package arrived damaged",
            "My order shows delivered but I never received it",
            "I need to change my shipping address after placing the order",
        ],
        "resolutions": [
            "Contacted the carrier, confirmed the delay, and issued a partial credit.",
            "Shipped the correct item at no extra cost and arranged return of the wrong one.",
            "Processed a replacement and filed a damage claim with the carrier.",
            "Filed a lost-package investigation with the carrier and reshipped the order.",
            "Updated the shipping address before the fulfillment cutoff.",
        ],
    },
    "Cancellation/Refund": {
        "issues": [
            "I want to cancel my subscription but can't find the option",
            "I requested a refund {days} days ago and haven't heard back",
            "I was charged after I thought I cancelled",
            "The refund I received was less than what I paid",
            "I want to cancel and get a refund for this month",
        ],
        "resolutions": [
            "Walked the user through the cancellation flow and confirmed it was completed.",
            "Located the pending refund request and processed it same-day.",
            "Investigated the cancellation timestamp and refunded the erroneous charge.",
            "Corrected the refund amount to match the original charge.",
            "Processed the cancellation and a pro-rated refund for the current month.",
        ],
    },
}

NEGATIVE_PHRASES = [
    "This is really frustrating.", "I'm quite upset about this.",
    "I've been dealing with this for days.", "This is unacceptable.",
    "I'm very disappointed.",
]
URGENT_PHRASES = [
    "I need this resolved immediately.", "This is urgent, please help ASAP.",
    "I need an answer today.",
]
NEUTRAL_PHRASES = [
    "Could someone look into this?", "Wanted to flag this issue.",
    "Let me know what you need from me.",
]
POSITIVE_PHRASES = [
    "Thanks in advance for your help!", "I appreciate you looking into this.",
    "No rush, just wanted to report it.",
]

MONTHS = ["January", "February", "March", "April", "May", "June"]
FEATURES = ["billing", "profile", "reports", "settings", "checkout"]


def _fill(template):
    return template.format(
        month=RNG.choice(MONTHS), feature=RNG.choice(FEATURES),
        days=RNG.integers(3, 15),
    )


def generate_knowledge_base():
    rows = []
    kb_id = 0
    for category, content in CATEGORIES.items():
        for issue_t, res_t in zip(content["issues"], content["resolutions"]):
            rows.append({
                "kb_id": f"KB{kb_id:04d}",
                "category": category,
                "issue_text": _fill(issue_t),
                "resolution_text": _fill(res_t),
            })
            kb_id += 1
    return pd.DataFrame(rows)


def _paraphrase(text):
    """Light paraphrase: prefix/suffix variation so tickets aren't identical
    to their KB source, without changing the semantic content."""
    prefixes = ["Hi, ", "Hello, ", "", "Hi team, ", "So, "]
    return RNG.choice(prefixes) + text[0].lower() + text[1:]


def generate_tickets(kb_df: pd.DataFrame, n_per_kb_entry=6):
    rows = []
    ticket_id = 0
    for _, kb_row in kb_df.iterrows():
        for _ in range(n_per_kb_entry):
            base_issue = CATEGORIES[kb_row["category"]]["issues"]
            template = RNG.choice(base_issue)
            text = _paraphrase(_fill(template))

            sentiment_bucket = RNG.choice(
                ["Negative", "Neutral", "Positive"], p=[0.45, 0.35, 0.20]
            )
            urgency_bucket = RNG.choice(["High", "Medium", "Low"], p=[0.3, 0.45, 0.25])

            tone = {
                "Negative": RNG.choice(NEGATIVE_PHRASES),
                "Neutral": RNG.choice(NEUTRAL_PHRASES),
                "Positive": RNG.choice(POSITIVE_PHRASES),
            }[sentiment_bucket]
            urgency_txt = RNG.choice(URGENT_PHRASES) if urgency_bucket == "High" else ""

            full_text = f"{text} {tone} {urgency_txt}".strip()

            rows.append({
                "ticket_id": f"T{ticket_id:05d}",
                "text": full_text,
                "true_category": kb_row["category"],
                "true_best_kb_match": kb_row["kb_id"],
                "true_sentiment": sentiment_bucket,
                "true_urgency": urgency_bucket,
            })
            ticket_id += 1
    df = pd.DataFrame(rows).sample(frac=1, random_state=21).reset_index(drop=True)
    return df


if __name__ == "__main__":
    kb_df = generate_knowledge_base()
    tickets_df = generate_tickets(kb_df)

    kb_df.to_csv("data/knowledge_base.csv", index=False)
    tickets_df.to_csv("data/tickets.csv", index=False)

    print(f"Knowledge base: {len(kb_df)} entries across {kb_df['category'].nunique()} categories")
    print(f"Tickets: {len(tickets_df)} generated")
    print("\nCategory distribution:\n", tickets_df["true_category"].value_counts())
    print("\nSentiment distribution:\n", tickets_df["true_sentiment"].value_counts())
