
import json
import re
import sys
import unicodedata
from pathlib import Path
from unittest import case, result

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.generation import generate_answer
from app.retrieval import retrieve


EVALUATION_FILE = (
    PROJECT_ROOT / "data" / "evaluation" / "grounding_eval.json"
)

ABSTENTION_PHRASES = (
    "could not find supporting evidence",
    "insufficient evidence",
    "not provided in the available documents",
    "not available in the available documents",
    "cannot determine from the available documents",
    "do not contain",
    "does not contain",
    "do not include",
    "does not include",
    "no information about",
    "cannot be determined",
    "no evidence available to answer",
    "no evidence to answer",
    "there is no evidence",
    "not enough information",
    "unable to answer",
    "i don't have enough information",
    "i do not have enough information",
    "insufficient information to answer",
)


def normalize_text(text: str) -> str:
    """Normalize Unicode, whitespace, and dash characters."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[\s\u00a0]+", " ", text)
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    return text.strip()

ANSWER_TERM_ALIASES = {
    "7": ("7", "seven"),
    "years": ("year", "years"),
}


def contains_all_answer_terms(text: str, terms: list[str]) -> bool:
    normalized_text = normalize_text(text)

    for term in terms:
        normalized_term = normalize_text(term)
        aliases = ANSWER_TERM_ALIASES.get(normalized_term)

        if aliases is None:
            aliases = (normalized_term,)

        if not any(
            re.search(
                rf"(?<!\w){re.escape(alias)}(?!\w)",
                normalized_text,
            )
            for alias in aliases
        ):
            return False

    return True
def load_cases() -> list[dict]:
    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {EVALUATION_FILE}"
        )

    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError("Evaluation dataset must be a JSON list.")

    return cases


def contains_all_terms(text: str, terms: list[str]) -> bool:
    normalized_text = normalize_text(text)
    return all(
        normalize_text(term) in normalized_text
        for term in terms
    )


def detects_explicit_abstention(answer: str) -> bool:
    normalized_answer = normalize_text(answer)
    return any(
        normalize_text(phrase) in normalized_answer
        for phrase in ABSTENTION_PHRASES
    )


def get_source_documents(result: dict) -> set[str]:
    return {
        source["document"]
        for source in result.get("sources", [])
        if source.get("document")
    }


def get_cited_passages(result: dict, passages: list[dict]) -> list[dict]:
    """Resolve generated citation IDs to the exact retrieved passages."""
    cited_passages = []
    seen_ids = set()

    for source in result.get("sources", []):
        citation = source.get("citation", "")

        if not re.fullmatch(r"S\d+", citation):
            continue

        if citation in seen_ids:
            continue

        seen_ids.add(citation)
        passage_index = int(citation[1:]) - 1

        if not 0 <= passage_index < len(passages):
            continue

        passage = passages[passage_index]

        # Confirm the returned source metadata matches the passage
        # represented by this citation ID.
        if (
            source.get("document") == passage.get("source")
            and source.get("chunk_index") == passage.get("chunk_index")
        ):
            cited_passages.append(passage)

    return cited_passages


def evaluate_case(case: dict) -> dict:
    question = case["question"]
    passages = retrieve(question, top_k=5)
    result = generate_answer(question, passages)

    answer = result.get("answer", "")
    source_documents = get_source_documents(result)
    cited_passages = get_cited_passages(result, passages)
    print("\nDEBUG: CITATION-TO-PASSAGE MAPPING")

    for source in result.get("sources", []):
        citation = source.get("citation", "")
        print(f"\nCitation: [{citation}]")

        if not re.fullmatch(r"S\d+", citation):
            print("Invalid citation format")
            continue

        index = int(citation[1:]) - 1

        if 0 <= index < len(passages):
            passage = passages[index]
            print(f"Document: {passage['source']}")
            print(f"Chunk: {passage['chunk_index']}")
            print(f"Passage: {passage['text']}")

    print("\nDEBUG: REQUIRED EVIDENCE TERM MATCHES")

    for term in case.get("required_evidence_terms", []):
        found = any(
        normalize_text(term) in normalize_text(passage["text"])
        for passage in cited_passages
        )
        print(f"{term!r}: {found}")

    required_terms = case.get("required_answer_terms", [])
    expected_sources = set(case.get("expected_sources", []))
    evidence_terms = case.get("required_evidence_terms", [])
    should_abstain = case.get("should_abstain", False)

    answer_terms_pass = (
    contains_all_answer_terms(answer, required_terms)
    if required_terms
    else None
)

    source_presence_pass = (
        expected_sources.issubset(source_documents)
        if expected_sources
        else None
    )

    cited_evidence = "\n".join(
        passage["text"] for passage in cited_passages
    )

    citation_evidence_pass = (
        bool(cited_passages)
        and contains_all_terms(cited_evidence, evidence_terms)
        if evidence_terms
        else None
    )

    abstention_signal = detects_explicit_abstention(answer)

    if should_abstain:
        # An abstention must explicitly acknowledge missing evidence.
        # A generic citation error is not considered a successful abstention.
        abstention_pass = abstention_signal and not source_documents
        checks = [abstention_pass]
    else:
        abstention_pass = None
        checks = [
            answer_terms_pass is True,
            source_presence_pass is True,
            citation_evidence_pass is True,
        ]

    return {
        "id": case["id"],
        "question": question,
        "answer": answer,
        "source_documents": sorted(source_documents),
        "cited_passage_count": len(cited_passages),
        "answer_terms_pass": answer_terms_pass,
        "source_presence_pass": source_presence_pass,
        "citation_evidence_pass": citation_evidence_pass,
        "abstention_signal": abstention_signal,
        "abstention_pass": abstention_pass,
        "status": "PASS" if all(checks) else "FAIL",
    }


def main() -> None:
    cases = load_cases()
    results = []

    print("=" * 72)
    print("ATLAS — CITATION-TO-EVIDENCE EVALUATION")
    print("=" * 72)
    print(f"Dataset: {EVALUATION_FILE}")
    print(f"Test cases: {len(cases)}")
    print("Retrieval depth: top_k=5")

    for case in cases:
        print(f"\n{'-' * 72}")
        print(f"Case: {case['id']}")
        print(f"Question: {case['question']}")

        try:
            result = evaluate_case(case)
        except Exception as exc:
            result = {
                "id": case.get("id", "UNKNOWN"),
                "question": case.get("question", ""),
                "error": str(exc),
                "status": "ERROR",
            }

        results.append(result)

        print(f"Answer: {result.get('answer', result.get('error', ''))}")
        print(f"Source documents: {result.get('source_documents', [])}")
        print(f"Cited passages: {result.get('cited_passage_count', 'N/A')}")
        print(f"Answer terms pass: {result.get('answer_terms_pass', 'N/A')}")
        print(f"Expected source pass: {result.get('source_presence_pass', 'N/A')}")
        print(f"Citation evidence pass: {result.get('citation_evidence_pass', 'N/A')}")
        print(f"Explicit abstention: {result.get('abstention_signal', 'N/A')}")
        print(f"Abstention pass: {result.get('abstention_pass', 'N/A')}")
        print(f"Result: {result['status']}")

    passed = sum(r["status"] == "PASS" for r in results)
    failed = sum(r["status"] == "FAIL" for r in results)
    errors = sum(r["status"] == "ERROR" for r in results)

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print("=" * 72)
    print(f"Total cases: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")

    if results:
        print(f"Pass rate: {passed / len(results) * 100:.1f}%")

    print(
        "\nLimitations: evidence-term matching is heuristic. It checks "
        "whether expected terms occur in the cited passages; it does not "
        "prove that every claim is semantically supported by its citation."
    )


if __name__ == "__main__":
    main()
