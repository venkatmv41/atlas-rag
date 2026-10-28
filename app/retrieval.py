
from sentence_transformers import SentenceTransformer

from app.ingestion import MODEL_NAME, get_collection

_model = None


def get_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def retrieve(question: str, top_k: int = 3) -> list[dict]:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    collection = get_collection()

    if collection.count() == 0:
        raise RuntimeError("Knowledge base is empty. Run ingestion first.")

    query_embedding = get_model().encode(
        question.strip(),
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    for i, chunk in enumerate(results["documents"][0]):
        matches.append({
            "text": chunk,
            "source": results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": results["distances"][0][i],
        })

    return matches


if __name__ == "__main__":
    question = "How do I configure a Kubernetes cluster?"

    for result in retrieve(question):
        print(f"\nSource: {result['source']}")
        print(f"Distance: {result['distance']:.4f}")
        print(result["text"])
