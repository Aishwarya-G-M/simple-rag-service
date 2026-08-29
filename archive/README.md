# Archived experiment scripts

These scripts were used during the early development of the RAG service to explore embeddings and FAISS. They are kept here for learning and reference.

The production logic now lives in:

- `app/faiss_retriever.py` – FAISS-based semantic retriever.
- `app/main.py` – FastAPI endpoints (`/rag/query`, `/retrieve-semantic`, etc.).

## Scripts

- `test_embedding.py` – Verified that `sentence-transformers` can embed a single sentence.
- `embed_all_messages.py` – Embedded all SMS messages from the CSV and inspected the embedding matrix shape.
- `build_faiss_index.py` – Built a FAISS index over all message embeddings and tested nearest-neighbor search.

These are not required to run the service. They can be executed directly with Python for experimentation, e.g.:

```bash
python scripts/archive/test_embedding.py
```