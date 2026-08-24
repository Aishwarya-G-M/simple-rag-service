from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Dict, Any

from fastapi import FastAPI, Query
from .documents import load_documents
from .faiss_retriever import FaissRetriever
from .llm import generate_answer
from .retriever import naive_retriever
from pydantic import BaseModel

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

app = FastAPI(title = "Simple RAG Service", lifespan=lifespan)

class ChatRequest(BaseModel):
    message: str
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    retrieved: List[Dict[str, Any]]

class RetrieveSemanticRequest(BaseModel):
    message: str
    k: int = 5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Simple RAG Service is running."}

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

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # native-retriever is deprecated - it was used for learning purpose
    # retrieved = naive_retriever(req.message, DOCUMENTS, top_k=req.top_k)

    # Retrieve relevant docs
    retrieved = retriever.retrieve_similar(req.message, k = req.top_k)
    # Generate answer using Groq
    answer = generate_answer(req.message, retrieved)
    return ChatResponse(answer=answer, retrieved=retrieved)

@app.post("/retrieve-semantic")
def retrieve_semantic(req: RetrieveSemanticRequest):
    results = retriever.retrieve_similar(req.message, k=req.k)
    return {"query": req.message, "results": results}