from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Dict, Any

from fastapi import FastAPI, Query
from .documents import load_documents
from .faiss_retriever import FaissRetriever
from .llm import generate_answer
from .retriever import naive_retriever
from pydantic import BaseModel
import time

from metrics.metrics import RagRequestMetrics
from metrics.logger import logger as metrics_logger
from metrics.helper import is_safe, did_abstain, is_correct
from .schemas import ChatResponse, ChatRequest

# In-memory document store (for now)
DOCUMENTS: List[Dict[str, Any]] = []

# Global semantic retriever
retriever = FaissRetriever()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("Starting up: loading documents...")
    global DOCUMENTS
    DOCUMENTS = load_documents("data")
    print(f"Loaded {len(DOCUMENTS)} documents into memory.")
    retriever.load()
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(
    title="Simple RAG Service",
    description="Generic retrieval-augmented generation over a configured document corpus.",
    lifespan=lifespan,
)

class RetrieveSemanticRequest(BaseModel):
    message: str
    k: int = 5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Simple RAG Service is running (generic RAG over the configured corpus)."}

@app.get("/retrieve")
def retrieve(
    query: str = Query(..., description="Query string to search for (naive word-overlap, deprecated)."),
    top_k: int = Query(5, ge=1, le=50, description="Number of results to return"),
):
    results = naive_retriever(query, DOCUMENTS, top_k=top_k)
    return {
        "query": query,
        "top_k": top_k,
        "results": results,
        "note": "This endpoint uses the deprecated naive retriever. Use /retrieve-semantic for semantic search.",
    }

@app.post("/rag/query", response_model=ChatResponse)
def rag_query(req: ChatRequest):
    started = time.perf_counter()

    # 1) Retrieve relevant docs
    retrieved = retriever.retrieve_similar(req.message, k=req.top_k)

    # 2) Generate answer
    answer = generate_answer(req.message, retrieved)
    model_name = "groq-llm"  # or your actual model ID

    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    # 3) Build minimal metrics object
    metrics = RagRequestMetrics(
        scenario_id=req.scenario_id or "",
        backend="baseline_rag",
        input_type=req.input_type or "benign",
        attack_type=req.attack_type,
        top_k=req.top_k,
        model_name=model_name,
        safe=is_safe(answer, req),
        contradicts_kg=None,
        abstained=did_abstain(answer),
        correct=is_correct(answer, req),
        latency_ms=latency_ms,
    )

    # 4) Emit structured JSON log
    metrics_logger.info(
        "rag_request_completed",
        extra={"metrics": metrics.dict()},
    )

    # 5) Return normal response
    return ChatResponse(answer=answer, retrieved=retrieved)

@app.post("/retrieve-semantic")
def retrieve_semantic(req: RetrieveSemanticRequest):
    results = retriever.retrieve_similar(req.message, k=req.k)
    return {"query": req.message, "results": results}