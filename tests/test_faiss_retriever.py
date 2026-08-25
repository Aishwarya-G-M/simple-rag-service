from pathlib import Path

from dotenv import load_dotenv

from app.faiss_retriever import FaissRetriever

load_dotenv()

def test_faiss_retriever_retrieve_similar():
    retriever = FaissRetriever()  # uses CSV_PATH and MODEL_NAME from .env
    retriever.load()

    query = "Free entry to win a cash prize. Text WIN now."
    results = retriever.retrieve_similar(query, k=3)

    assert len(results) == 3
    for r in results:
        assert "index" in r
        assert "distance" in r
        assert "text" in r
        assert isinstance(r["index"], int)
        assert isinstance(r["distance"], float)
        assert isinstance(r["text"], str)

    # Sanity check: distances should be non-negative
    assert all(r["distance"] >= 0 for r in results)