"""
The Grounded Answer — a policy-manual Q&A assistant.

Answers resident/caseworker questions from a county policy manual,
always citing the exact clause it relied on, and says
"I don't know, here is who to ask" when the manual doesn't cover it.

No API key, no internet, no paid services required — retrieval runs
locally with TF-IDF (scikit-learn).
"""

import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MANUAL_PATH = Path(__file__).parent / "data" / "policy_manual.md"

# Below this similarity score, we refuse to answer rather than guess.
CONFIDENCE_THRESHOLD = 0.20

FALLBACK_CONTACT = "the County Benefits Office front desk (M–F, 9am–4pm) or benefits-help@county.example.gov"


def load_clauses(path: Path):
    """Split the manual into (clause_id, title, text) chunks on '## Clause X.Y — Title' headers."""
    raw = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^## (Clause [\d.]+) — (.+)$", re.MULTILINE)
    matches = list(pattern.finditer(raw))

    clauses = []
    for i, m in enumerate(matches):
        clause_id, title = m.group(1), m.group(2)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        clauses.append({"id": clause_id, "title": title, "text": body})
    return clauses


class GroundedAnswerAssistant:
    def __init__(self, clauses):
        self.clauses = clauses
        corpus = [f"{c['title']}. {c['text']}" for c in clauses]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    def answer(self, question: str):
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        best_idx = scores.argmax()
        best_score = scores[best_idx]

        if best_score < CONFIDENCE_THRESHOLD:
            return {
                "answered": False,
                "message": (
                    "I don't know — the policy manual doesn't clearly cover this. "
                    f"Please check with {FALLBACK_CONTACT}."
                ),
                "confidence": round(float(best_score), 3),
            }

        clause = self.clauses[best_idx]
        return {
            "answered": True,
            "message": f"{clause['text']}",
            "cited_clause": f"{clause['id']} — {clause['title']}",
            "confidence": round(float(best_score), 3),
        }


def main():
    clauses = load_clauses(MANUAL_PATH)
    assistant = GroundedAnswerAssistant(clauses)

    print("The Grounded Answer — policy manual assistant")
    print(f"Loaded {len(clauses)} clauses from {MANUAL_PATH.name}")
    print("Ask a question, or type 'quit' to exit.\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break

        result = assistant.answer(question)
        print(f"A: {result['message']}")
        if result["answered"]:
            print(f"   [Source: {result['cited_clause']} | confidence: {result['confidence']}]")
        else:
            print(f"   [confidence: {result['confidence']} — below threshold {CONFIDENCE_THRESHOLD}]")
        print()


if __name__ == "__main__":
    sys.exit(main())