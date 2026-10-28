
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval import retrieve


DATASET_PATH = ROOT / "data" / "evaluation" / "rag_eval.json"
TOP_K = 3


def main():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    hits = 0
    reciprocal_ranks = []

    for case in cases:
        results = retrieve(case["question"], top_k=TOP_K)

        expected_source = case["expected_source"].lower()
        expected_evidence = case["expected_evidence"].casefold()

        source_found = any(
            result["source"].lower() == expected_source
            for result in results
        )

        evidence_rank = next(
            (
                rank
                for rank, result in enumerate(results, start=1)
                if expected_evidence in result["text"].casefold()
            ),
            None,
        )

        evidence_found = evidence_rank is not None
        hits += int(evidence_found)
        reciprocal_ranks.append(
            1 / evidence_rank if evidence_rank is not None else 0
        )

        print(f"\n[{case['id']}] {case['question']}")
        print(f"Expected source found: {source_found}")
        print(f"Evidence found in top {TOP_K}: {evidence_found}")
        print(f"Evidence rank: {evidence_rank or 'Not found'}")

        for rank, result in enumerate(results, start=1):
            print(
                f"  {rank}. {result['source']} "
                f"(chunk {result['chunk_index']})"
            )

    total = len(cases)
    recall = hits / total if total else 0
    mrr = sum(reciprocal_ranks) / total if total else 0

    print("\n--- RETRIEVAL EVALUATION SUMMARY ---")
    print(f"Questions evaluated: {total}")
    print(f"Evidence Recall@{TOP_K}: {recall:.1%}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.3f}")


if __name__ == "__main__":
    main()
