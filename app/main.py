from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Dict, Any

from fastapi import FastAPI
from .documents import load_documents

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


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Simple RAG Service is running."}


