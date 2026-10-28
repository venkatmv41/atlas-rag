
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.retrieval import retrieve
from app.generation import generate_answer

app = FastAPI(title="ATLAS Knowledge Intelligence API")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest):
    try:
        passages = retrieve(request.question, top_k=5)
        result = generate_answer(request.question, passages)
        return result

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        # Log the exception in production rather than exposing
        # internal provider or configuration details to callers.
        print(f"ATLAS request failed: {type(exc).__name__}")
        raise HTTPException(
            status_code=502,
            detail="The knowledge service could not complete the request.",
        ) from exc
