from fastapi import FastAPI

app = FastAPI(title = "Simple RAG Service")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Simple RAG Service is running."}
