
import argparse
import re
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval import retrieve


TEST_CASES = [
    {
        "id": "T1",
        "question": "What actions are required after a storage temperature excursion?",
        "expected": "temperature_excursion.txt",
        "evidence_terms": ["quarantine", "disposition"],
    },
    {
    "id": "T2",
    "question": "Who should be informed about a temperature problem?",
    "expected": "temperature_excursion.txt",
    "evidence_terms": [
        "QA Temperature Review Team",
        "phone",
        "email",
    ],
},
    {
        "id": "T3",
        "question": "How should a process deviation be investigated and closed?",
        "expected": "deviation_management.txt",
        "evidence_terms": ["root cause", "closure"],
    },
    {
        "id": "T4",
        "question": "Which roles must approve a document before it becomes effective?",
        "expected": "document_control.txt",
        "evidence_terms": [
            "Author",
            "Department Head",
            "QA Reviewer",
            "QA Head",
        ],
    },
    {
        "id": "T5",
        "question": "How quickly must a previous document revision be withdrawn from all points of use?",
        "expected": "document_control.txt",
        "evidence_terms": [
            "previous revision",
            "within 1 business day",
        ],
    },
    {
        "id": "T6",
        "question": "How long are excursion investigation reports retained?",
        "expected": "temperature_excursion.txt",
        "evidence_terms": ["investigation reports", "retained for 7 years"],
    },
]


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace for case-insensitive matching."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[\s\u00a0]+", " ", text)
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    return text.strip()


def evaluate_retrieval(results: list[dict], test: dict) -> dict:
    """
    Evaluate retrieval using only the expected source.

    Evidence is considered complete at the first rank where all required
    terms have appeared in expected-source passages retrieved so far.
    """
    expected_source = test["expected"]
    required_terms = test["evidence_terms"]
    normalized_terms = {
        term: normalize_text(term) for term in required_terms
    }

    found_terms = set()
    source_hit_rank = None
    complete_evidence_rank = None

    for rank, result in enumerate(results, start=1):
        if result["source"] != expected_source:
            continue

        if source_hit_rank is None:
            source_hit_rank = rank

        chunk_text = normalize_text(result["text"])

        for term, normalized_term in normalized_terms.items():
            if normalized_term in chunk_text:
                found_terms.add(term)

        if (
            complete_evidence_rank is None
            and len(found_terms) == len(required_terms)
        ):
            complete_evidence_rank = rank

    missing_terms = [
        term for term in required_terms if term not in found_terms
    ]

    return {
        "source_hit_rank": source_hit_rank,
        "complete_evidence_rank": complete_evidence_rank,
        "found_terms": sorted(found_terms),
        "missing_terms": missing_terms,
        "source_hit": source_hit_rank is not None,
        "evidence_complete": (
            complete_evidence_rank is not None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ATLAS retrieval quality."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of passages to retrieve (default: 3).",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    source_hits = 0
    evidence_hits = 0
    reciprocal_ranks = []

    print("=" * 72)
    print("ATLAS — RETRIEVAL EVALUATION")
    print("=" * 72)
    print(f"Questions: {len(TEST_CASES)}")
    print(f"Retrieval depth: top_k={args.top_k}")

    for test in TEST_CASES:
        results = retrieve(test["question"], top_k=args.top_k)
        evaluation = evaluate_retrieval(results, test)

        source_hit_rank = evaluation["source_hit_rank"]
        evidence_rank = evaluation["complete_evidence_rank"]

        source_hits += int(evaluation["source_hit"])
        evidence_hits += int(evaluation["evidence_complete"])
        reciprocal_ranks.append(
            1 / evidence_rank if evidence_rank is not None else 0
        )

        print(f"\n{'-' * 72}")
        print(f"{test['id']}: {test['question']}")
        print(f"Expected source: {test['expected']}")
        print(f"Source Hit: {source_hit_rank or 'NOT FOUND'}")
        print(
            "Complete evidence rank: "
            f"{evidence_rank or 'INCOMPLETE'}"
        )
        print(f"Found terms: {evaluation['found_terms']}")
        print(f"Missing terms: {evaluation['missing_terms']}")

        for rank, result in enumerate(results, start=1):
            print(
                f"  {rank}. {result['source']} "
                f"| chunk={result['chunk_index']} "
                f"| distance={result['distance']:.4f}"
            )

    total = len(TEST_CASES)
    source_hit_rate = source_hits / total
    evidence_recall = evidence_hits / total
    mrr = sum(reciprocal_ranks) / total

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print("=" * 72)
    print(f"Questions: {total}")
    print(f"Source Hit@{args.top_k}: {source_hit_rate:.2%}")
    print(f"Evidence Recall@{args.top_k}: {evidence_recall:.2%}")
    print(f"Evidence MRR@{args.top_k}: {mrr:.3f}")
    print(
        "\nNote: These metrics use manually labeled evidence terms. "
        "They are stricter than any-term matching, but they do not "
        "establish semantic relevance or answer correctness."
    )


if __name__ == "__main__":
    main()
