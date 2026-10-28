
import sys
from pathlib import Path

# Make the project root importable when this script runs from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval import retrieve


def main():
    question = "Which team receives all temperature excursion notices?"

    passages = retrieve(question, top_k=3)

    if not passages:
        print("No passages were retrieved.")
        return

    for rank, passage in enumerate(passages, start=1):
        print(f"\n{'=' * 72}")
        print(f"RANK: {rank}")
        print(f"SOURCE: {passage['source']}")
        print(f"CHUNK: {passage['chunk_index']}")
        print(f"DISTANCE: {passage.get('distance')}")
        print("\nPASSAGE TEXT:")
        print(passage["text"])


if __name__ == "__main__":
    main()
