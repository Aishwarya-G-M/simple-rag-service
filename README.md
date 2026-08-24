## How it works (current version)

This service implements a minimal end‑to‑end RAG pipeline over an SMS spam corpus:

- **Data**: Loads a CSV of labeled SMS messages (`ham`/`spam`) into memory at startup.
- **Retrieval (naive)**: For a given query, it performs simple word‑overlap retrieval:
  - Splits the query into words.
  - For each SMS, counts how many query words appear in the message text.
  - Returns the top‑k messages with the highest overlap score.
- **Generation**: Sends the retrieved messages plus the user’s question to a Groq LLM with a fraud‑analysis prompt, and returns the model’s grounded answer.

Endpoints:
- `GET /retrieve?query=<text>&top_k=<int>` – returns top‑k relevant SMS messages.
- `POST /chat` – RAG endpoint: retrieves context, calls Groq, returns `{answer, retrieved}`.

This is a prototype RAG system; the next step is to replace the naive retrieval with embeddings + FAISS for semantic search.