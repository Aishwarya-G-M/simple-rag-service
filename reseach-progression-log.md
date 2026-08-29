## 2026-08-24
- Implemented: naive word-overlap retrieval and Groq-backed /rag/query.
- Observed: exact keyword matches retrieve obvious spam examples well.
- Limitation: retrieval scans every document and cannot capture semantic similarity.
- Next experiment: sentence-transformer embeddings + FAISS; compare retrieved results.