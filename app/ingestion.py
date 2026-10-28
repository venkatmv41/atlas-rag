
from pathlib import Path
import hashlib
import re

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
SOP_DIR = ROOT / "data" / "sops"
DB_DIR = ROOT / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "atlas_sops"


# - SentenceTransformer converts text into numeric vectors that capture semantic similarity.
# - Chunking makes longer documents searchable in smaller units.
# - Metadata connects retrieved text back to its source document.
# - PersistentClient stores the vector database locally.
# - Replacing existing chunks for a document avoids stale chunks when that file changes.
# One design trade-off: character-based chunking is simple for this first sprint, but it can split a procedure mid-sentence. We'll investigate section-aware chunking later.

def split_into_chunks(text: str, size: int = 800,
                      overlap: int = 120) -> list[str]:
    """Split text into overlapping character windows."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > overlap >= 0")

    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_documents():
    if not SOP_DIR.exists():
        raise FileNotFoundError(f"SOP folder not found: {SOP_DIR}")

    files = sorted(SOP_DIR.glob("*.txt"))
    if not files:
        raise ValueError(f"No .txt documents found in {SOP_DIR}")

    model = SentenceTransformer(MODEL_NAME)
    collection = get_collection()
    total_chunks = 0

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        chunks = split_into_chunks(text)

        if not chunks:
            print(f"Skipping empty document: {file_path.name}")
            continue

        # Replace existing chunks for this source so reruns
        # don't leave duplicate or stale chunks behind.
        existing = collection.get(
            where={"source": file_path.name}
        )
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        document_id = hashlib.sha256(
            file_path.name.encode("utf-8")
        ).hexdigest()[:12]

        ids = [
            f"{document_id}-{i:04d}"
            for i in range(len(chunks))
        ]

        metadatas = [{
            "source": file_path.name,
            "document_id": document_id,
            "chunk_index": i,
        } for i in range(len(chunks))]

        embeddings = model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        total_chunks += len(chunks)
        print(f"Indexed {file_path.name}: {len(chunks)} chunks")

    print(f"Total chunks indexed: {total_chunks}")
    print(f"Collection size: {collection.count()}")


if __name__ == "__main__":
    ingest_documents()
