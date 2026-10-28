
from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_returns_answer_and_sources(monkeypatch):
    # Replace retrieval and generation so this test makes no external calls.
    monkeypatch.setattr(
        main,
        "retrieve",
        lambda question, top_k=3: [
            {
                "text": "Temperature excursions must be reported to QA.",
                "source": "temperature_excursion.txt",
                "chunk_index": 2,
                "distance": 0.15,
            }
        ],
    )

    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda question, passages: {
            "answer": "Report the excursion to QA. [S1]",
            "sources": [
                {
                    "citation": "S1",
                    "document": "temperature_excursion.txt",
                    "chunk_index": 2,
                }
            ],
            "grounded": None,
        },
    )

    response = client.post(
        "/ask",
        json={"question": "Who owns a temperature-excursion investigation?"},
    )

    assert response.status_code == 200

    body = response.json()
    assert "[S1]" in body["answer"]
    assert body["sources"][0]["citation"] == "S1"
    assert body["sources"][0]["document"] == "temperature_excursion.txt"


def test_ask_rejects_too_short_question():
    response = client.post("/ask", json={"question": "hi"})

    assert response.status_code == 422

def test_ask_returns_503_for_configuration_failure(monkeypatch):
    def fail_generation(question, passages):
        raise RuntimeError("GROQ_API_KEY is not configured")

    monkeypatch.setattr(main, "retrieve", lambda question, top_k=3: [])
    monkeypatch.setattr(main, "generate_answer", fail_generation)

    response = client.post(
        "/ask",
        json={"question": "What is the excursion reporting process?"},
    )

    assert response.status_code == 503
    assert "GROQ_API_KEY is not configured" in response.json()["detail"]


def test_ask_hides_unexpected_exception_details(monkeypatch):
    def fail_retrieval(question, top_k=3):
        raise Exception("internal-provider-secret")

    monkeypatch.setattr(main, "retrieve", fail_retrieval)

    response = client.post(
        "/ask",
        json={"question": "What is the excursion reporting process?"},
    )

    assert response.status_code == 502
    assert "internal-provider-secret" not in response.text
    assert response.json()["detail"] == (
        "The knowledge service could not complete the request."
    )


def test_ask_rejects_question_over_maximum_length():
    response = client.post(
        "/ask",
        json={"question": "x" * 2001},
    )

    assert response.status_code == 422