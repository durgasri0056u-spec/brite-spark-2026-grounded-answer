# The Grounded Answer

A question-answering assistant for a county benefits policy manual.
It answers in plain language, **cites the exact clause it relied on**,
and says **"I don't know, here is who to ask"** when the manual doesn't
cover the question.

Problem 1 · AI/RAG — Brite Spark 2026

## How it works

- `data/policy_manual.md` — a synthetic 12-clause policy manual (all
  data is synthetic, per handbook rules).
- `app.py` — loads the manual, splits it into clauses, builds a
  TF-IDF index over them (scikit-learn), and retrieves the
  best-matching clause for each question.
- If the best match scores below a confidence threshold, the
  assistant refuses to guess and points to a human contact instead.

No API keys, no internet access, and no paid services are required —
retrieval is entirely local.

## Setup

```bash
pip install -r requirements.txt