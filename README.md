# Simple RAG Service

A learning-oriented Retrieval-Augmented Generation (RAG) service for analyzing SMS spam messages.

## Version 1

This is an initial RAG prototype built with Python, FastAPI, and Groq.

### What it does

1. Loads an SMS spam CSV dataset into memory at service startup.
2. Exposes `GET /retrieve`, which finds relevant messages using naive word-overlap scoring.
3. Exposes `POST /rag/query`, which:
   - Retrieves the top-k matching SMS messages.
   - Adds them as context to an LLM prompt.
   - Calls Groq to generate a grounded response.
   - Returns the answer and the retrieved messages.

## Retrieval limitation

The current retriever loops through every SMS message and scores it by the number of query words found in its text. It depends on exact word overlap and does not capture semantic similarity.

This is intentional: it establishes a simple, inspectable baseline before adding embeddings and FAISS-based vector search.

## API

### Health check

```bash
GET /health
```

### Retrieve messages

```bash
GET /retrieve?query=free%20entry%20win&top_k=3
```

### RAG chat

```bash
POST /rag/query
Content-Type: application/json
```

Example body:

```json
{
  "message": "Is this SMS likely spam? Free entry to win a prize.",
  "top_k": 3
}
```

## Local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="your_api_key"
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to use the interactive API documentation.


## Architecture and roadmap

For the broader architecture (including `secure-llm-gateway` and the planned GraphRAG service), see [docs/architecture-and-roadmap.md](docs/architecture-and-roadmap.md).