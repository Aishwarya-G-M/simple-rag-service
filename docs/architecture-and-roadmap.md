# Architecture and Roadmap

This document describes how `simple-rag-service` fits into the larger system, the planned `graphrag-service`, and how we will compare simple RAG vs GraphRAG while using a shared `secure-llm-gateway`.

## High-level architecture

```text
                     ┌──────────────────────┐
                     │   secure-llm-gateway │
                     │  (auth, policy, LLM) │
                     └──────────▲───────────┘
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                    │
   ┌───────┴────────┐   ┌───────┴────────┐   ┌───────┴────────┐
   │ simple-rag-    │   │ graphrag-      │   │ other LLM-     │
   │ service        │   │ service        │   │ using services │
   │ (FAISS RAG)    │   │ (GraphRAG)     │   │                │
   └────────────────┘   └────────────────┘   └────────────────┘
```

- **`secure-llm-gateway`**
  - Central gateway for all LLM calls.
  - Responsibilities:
    - Authentication and authorization.
    - Rate limiting, quotas, cost tracking.
    - Input/output filtering (PII, prompt injection, content policies).
    - Unified API to multiple LLM providers (Groq, OpenAI, etc.).
    - Logging and auditing.

- **`simple-rag-service`**
  - Domain-specific RAG service over a text corpus (e.g., SMS spam / fraud messages).
  - Implements:
    - Document ingestion and embedding.
    - Vector store (FAISS).
    - Retrieval strategies:
      - Naive word-overlap (deprecated).
      - Semantic retrieval (embeddings + FAISS).
    - RAG chat endpoint that constructs prompts and calls `secure-llm-gateway`.

- **`graphrag-service` (planned)**
  - Another domain-specific RAG service, but using GraphRAG.
  - Implements:
    - Graph-based indexing of the corpus (entities, relationships).
    - Graph-aware retrieval (multi-hop, community-level summaries).
    - RAG chat endpoint that also calls `secure-llm-gateway`.

Both RAG services call the **same** `secure-llm-gateway`, so the LLM and security layer are constant. The variable is the retrieval strategy.

## Current state: `simple-rag-service`

`simple-rag-service` currently provides:

- Embedding-based semantic retrieval:
  - Model: `sentence-transformers/all-MiniLM-L6-v2`.
  - Vector store: FAISS.
- Endpoints:
  - `GET /health`
  - `GET /`
  - `GET /retrieve` (deprecated, naive word-overlap).
  - `POST /retrieve-semantic`
  - `POST /chat` (RAG with semantic retrieval + LLM).
- Tests for the retriever and `/retrieve-semantic` endpoint.

For details on usage, see the main [`README.md`](../README.md).

## Planned: `graphrag-service`

We will build a new service, `graphrag-service`, that:

- Uses the same (or an augmented) corpus as `simple-rag-service`.
- Implements GraphRAG:
  - Constructs a graph over documents and entities.
  - Supports multi-hop and corpus-level queries.
- Exposes similar endpoints:
  - `POST /chat-graph` (GraphRAG-based RAG).
  - Possibly `POST /retrieve-graph` for raw retrieval.

Like `simple-rag-service`, it will call `secure-llm-gateway` for all LLM interactions.

## Comparison plan

The goal is to compare:

> (simple-RAG + secure-llm-gateway) vs (GraphRAG + secure-llm-gateway)

with the gateway and LLM provider held constant.

### Evaluation approach

1. **Corpus**
   - Start with the existing SMS spam / fraud dataset.
   - Optionally augment with:
     - Fraud reports, news articles, or a small knowledge base to make multi-hop questions more meaningful.

2. **Question sets**
   - Define a small set of questions where:
     - Simple RAG should perform reasonably:
       - “Give me examples of spam messages about prizes.”
     - GraphRAG is expected to shine:
       - “What are the common tactics used in prize-related scams across the corpus?”
       - “How do scammers ask victims to contact them, and what channels are mentioned?”

3. **Metrics (manual / semi-automatic)**
   - For each question, record:
     - Retrieved context (top-k documents / graph nodes).
     - Generated answer from each service.
   - Manually judge:
     - Relevance of retrieved context.
     - Correctness and completeness of answers.
     - Whether multi-hop reasoning is needed and whether the answer reflects it.

4. **Operational comparison**
   - Complexity:
     - Indexing pipeline (vector vs graph).
     - Codebase size and maintainability.
   - Latency:
     - Index build time.
     - Query time (retrieval + generation).
   - Extensibility:
     - How easy it is to add new data sources or retrieval strategies.

### Expected outcomes

- A clear understanding of:
  - When GraphRAG provides meaningful benefits over simple RAG.
  - The additional complexity and cost it introduces.
- Documentation of:
  - Example questions and answers from both systems.
  - Trade-offs and recommendations for future use.

## Roadmap

### Near term

- [ ] Keep `simple-rag-service` as the v0.2.0 baseline (vanilla RAG with FAISS).
- [ ] Update `simple-rag-service` to call `secure-llm-gateway` instead of calling Groq directly.
- [ ] Decide on the final corpus for the GraphRAG experiment (SMS-only vs augmented).

### Medium term

- [ ] Implement `graphrag-service`:
  - Choose a GraphRAG implementation (e.g., Microsoft GraphRAG or a custom approach).
  - Implement graph construction and retrieval.
  - Expose `/chat-graph` (and optionally `/retrieve-graph`).
- [ ] Define the evaluation question set and run side-by-side comparisons.
- [ ] Document results and trade-offs.

### Longer term

- [ ] Generalize patterns into a reusable “RAG service template” that can be configured for:
  - Different corpora.
  - Different retrieval strategies (vector, graph, hybrid).
- [ ] Integrate both RAG services with a front-end or higher-level API for end-to-end demos.

## Relationship to other docs

- For usage, setup, and current endpoints of `simple-rag-service`, see [`README.md`](../README.md).
- This document focuses on architecture, the role of `secure-llm-gateway`, and the planned GraphRAG comparison.