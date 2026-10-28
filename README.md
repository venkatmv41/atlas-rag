# ATLAS --- Enterprise Knowledge Intelligence API

ATLAS is a retrieval-augmented generation (RAG) API that answers
questions using a collection of documents, retrieves supporting
passages, and returns answers with source citations. It is built with
FastAPI, ChromaDB, Sentence Transformers, and Groq.

> **Important:** The included SOPs are synthetic demonstration
> documents, not official pharmaceutical, regulatory, or operational
> guidance. Do not use this project as a source of real-world compliance
> or safety decisions without replacing and validating the documents.

## Features

-   **Document ingestion:** reads sample SOP text files, splits them
    into overlapping chunks, creates embeddings, and stores them in
    ChromaDB.
-   **Semantic retrieval:** retrieves relevant passages from the vector
    store.
-   **Grounded answers:** uses a Groq-hosted language model to generate
    answers from retrieved evidence.
-   **Source citations:** answers can reference retrieved passages using
    labels such as `[S1]`.
-   **Abstention behavior:** the generation prompt instructs the model
    to say when supplied evidence is insufficient.
-   **FastAPI endpoints:** health check and question-answering endpoint,
    with interactive API documentation.
-   **Evaluation and tests:** includes API, retrieval, and generation
    tests, plus retrieval and grounding evaluation scripts.
-   **Docker support:** run the service with Docker Compose.
-   **GitHub Actions CI:** runs the test suite and verifies that the
    Docker image builds.

## Architecture

``` text
Question
   |
   v
FastAPI /ask
   |
   v
Embedding model  ---> ChromaDB vector search
                           |
                           v
                    Retrieved passages
                           |
                           v
                    Groq chat model
                           |
                           v
             Answer with source citations
```

## Technology stack

-   Python
-   FastAPI and Uvicorn
-   ChromaDB
-   Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
-   Groq API (configured through environment variables)
-   Pytest
-   Docker and Docker Compose
-   GitHub Actions

## Repository layout

``` text
atlas-rag/
├── app/
│   ├── main.py
│   ├── ingestion.py
│   ├── retrieval.py
│   └── generation.py
├── data/
│   ├── sops/
│   └── evaluation/
├── scripts/
│   ├── ingest.py
│   ├── evaluate_rag.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_grounding.py
│   └── inspect_retrieval.py
├── tests/
├── .github/workflows/ci.yml
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── .env.example
└── README.md
```

The exact contents may evolve as the project develops.

## Prerequisites

Choose either the local Python setup or Docker setup.

-   Git
-   Python compatible with the pinned packages in `requirements.txt`
    (the CI workflow currently uses Python 3.14)
-   Docker Desktop with Docker Compose, if using Docker
-   A Groq API key for answer generation

The first run may download the embedding model and install sizeable
machine-learning dependencies. Allow time and disk space for these
downloads.

## 1. Clone the repository

``` bash
git clone https://github.com/<YOUR-GITHUB-USERNAME>/atlas-rag.git
cd atlas-rag
```

Replace `<YOUR-GITHUB-USERNAME>` with the repository owner's GitHub
username, or use the clone URL displayed on the repository page.

## 2. Configure environment variables

Create a local `.env` file from the example.

**Windows PowerShell**

``` powershell
Copy-Item .env.example .env
notepad .env
```

**macOS/Linux**

``` bash
cp .env.example .env
```

Set values using the variable names already present in `.env.example`.
Provide your own Groq API key and the model configured for this project
(currently `openai/gpt-oss-20b`).

Never commit `.env` or publish your API key. Keep `.env.example` limited
to placeholder values.

## Option A --- Run locally with Python

**Windows PowerShell**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS/Linux**

``` bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Ingest the sample documents

From the repository root:

``` bash
python scripts/ingest.py
```

Ingestion creates or updates the local ChromaDB index. Run ingestion
again when you add or change source documents.

### Start the API

``` bash
python -m uvicorn app.main:app --reload
```

The API should be available at `http://127.0.0.1:8000`.

## Option B --- Run with Docker Compose

Make sure Docker Desktop is running, then run from the repository root:

``` bash
docker compose up -d --build
docker compose ps
docker compose logs --tail=100 atlas
```

The Compose configuration uses named volumes for ChromaDB data and the
Hugging Face model cache. The API container runs as a non-root user.
First startup may take time while the embedding model is downloaded and
loaded.

### Ingest documents in the container

If the vector index is empty, run ingestion from the application
package:

``` bash
docker compose exec atlas python -c "from app.ingestion import ingest_documents; ingest_documents()"
```

Run this from the repository root. Check the container logs if ingestion
fails.

### Stop the service

``` bash
docker compose down
```

Named volumes are retained by default when using `docker compose down`.
To remove volumes and persisted data, use `docker compose down -v` only
when you intentionally want to delete that data.

## API usage

### Health check

``` http
GET /health
```

Example:

``` bash
curl http://127.0.0.1:8000/health
```

Expected response:

``` json
{
  "status": "ok"
}
```

### Ask a question

``` http
POST /ask
Content-Type: application/json
```

Example request:

``` json
{
  "question": "Who is responsible for investigating a temperature excursion?"
}
```

Example using `curl`:

``` bash
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d "{\"question\":\"Who is responsible for investigating a temperature excursion?\"}"
```

The response includes a generated answer and source information when
available. Exact response fields and citations depend on the current
implementation and retrieved evidence.

### Interactive API documentation

Open `http://127.0.0.1:8000/docs` in a browser to explore and call the
endpoints through Swagger UI.

## Tests and evaluation

Run the automated test suite from the repository root:

``` bash
python -m pytest -v
```

Evaluation scripts are available under `scripts/`. Depending on the
script, they assess retrieval quality, grounding checks, or inspect
retrieved passages.

``` bash
python scripts/evaluate_retrieval.py
python scripts/evaluate_grounding.py
python scripts/evaluate_rag.py
```

Evaluation scores depend on the documents, questions, retrieval
settings, model behavior, and code version. Treat small evaluation sets
as development signals, not proof of production-level reliability. Some
answer-generation paths may require valid Groq credentials.

## GitHub Actions

The CI workflow runs on pushes and pull requests targeting the
configured `main` or `master` branches. It runs the Python test suite
and builds the Docker image.

A passing CI run verifies those workflow steps; it does not guarantee
every environment, credential, or runtime configuration will work for
every user.

## Configuration and data

-   **Groq credentials:** supply your own API key through `.env`.
-   **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` is
    downloaded by the model library when required.
-   **Vector database:** ChromaDB stores embeddings and metadata in its
    configured persistence directory.
-   **Sample documents:** text files under `data/sops/` are
    demonstration content.
-   **Evaluation data:** files under `data/evaluation/` provide small
    development evaluation sets.

Do not put confidential, regulated, or personal information into the
sample setup without reviewing data handling, access controls,
retention, and the external model provider's policies.

## Troubleshooting

### Missing Groq API key or generation errors

-   Check that `.env` exists and contains the expected variable names.
-   Restart the service after changing environment variables.
-   Do not paste secrets into issue reports or commit them to GitHub.

### No useful answers or no retrieved sources

-   Confirm that ingestion completed successfully.
-   Check that expected documents exist under `data/sops/`.
-   Review retrieval evaluation and inspect-retrieval scripts.
-   Verify that the question is answerable from the indexed documents.

### Docker container is unhealthy

``` bash
docker compose ps
docker compose logs --tail=100 atlas
```

Confirm that the configured health check uses `/health` and that the
service starts without errors.

### First run is slow

The first run may download Python dependencies and the embedding model.
Later runs can reuse local or named-volume caches, depending on how the
application is launched.

### Port 8000 is already in use

Stop the other service using port 8000, or update the port mapping in
`compose.yaml` and use the corresponding local URL.

## Security notes

-   Never commit `.env`, API keys, tokens, or other secrets.
-   Review `.gitignore` before publishing.
-   Use synthetic or non-sensitive documents for demonstrations.
-   Treat retrieved document content as untrusted input.
-   Review provider terms and data policies before using external model
    APIs with sensitive information.

## Current limitations

-   The included SOPs and evaluation set are small synthetic examples.
-   Retrieval and grounding metrics on a small evaluation set do not
    establish general accuracy.
-   Language-model answers may still be incomplete or incorrect; check
    citations against source documents.
-   Local ChromaDB persistence and model caching require appropriate
    storage and permissions.
-   A production deployment would need additional operational, security,
    monitoring, scaling, and data-governance work.

## Contributing

Contributions are welcome. A useful contribution workflow is:

1.  Fork the repository.
2.  Create a feature branch.
3.  Make a focused change and add or update tests.
4.  Run `python -m pytest -v`.
5.  Open a pull request describing the change and its validation.

Please do not include secrets, private documents, or generated database
files in commits.

## License

A license file must be added to the repository to define the terms under
which others may use, modify, and redistribute this project. Until a
license is added, do not assume the repository is MIT-licensed or
otherwise freely reusable.

------------------------------------------------------------------------

Built as a practical learning project for exploring retrieval-augmented
generation, source-grounded answers, evaluation, API development,
containerization, and continuous integration.
