from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Dict, Any

from fastapi import FastAPI, Query
from .documents import load_documents
from .llm import generate_answer
from .retriever import naive_retriever
from pydantic import BaseModel

# In-memory document store (for now)
DOCUMENTS: List[Dict[str, Any]] = []

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("Starting up: loading documents...")
    global DOCUMENTS
    DOCUMENTS = load_documents("data")
    print(f"Loaded {len(DOCUMENTS)} documents into memory.")
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


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Simple RAG Service is running."}

@app.get("/retrieve")
def retrieve(
    query: str = Query(..., description="Query string to search for"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results to return"),
):
    results = naive_retriever(query, DOCUMENTS, top_k=top_k)
    return {
        "query": query,
        "top_k": top_k,
        "results": results,
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Retrieve relevant docs
    retrieved = naive_retriever(req.message, DOCUMENTS, top_k=req.top_k)
    # Generate answer using Groq
    answer = generate_answer(req.message, retrieved)
    return ChatResponse(answer=answer, retrieved=retrieved)