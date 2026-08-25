from dotenv import load_dotenv
from fastapi.testclient import TestClient
from app.main import app, retriever  # import the global retriever

# Ensure the retriever is loaded once at import time for tests
retriever.load()

client = TestClient(app)

def test_retrieve_semantic_endpoint():
    resp = client.post(
        "/retrieve-semantic",
        json={"message": "Free entry to win a cash prize. Text WIN now.", "k": 3},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "query" in data
    assert "results" in data
    assert data["query"] == "Free entry to win a cash prize. Text WIN now."
    assert isinstance(data["results"], list)
    assert len(data["results"]) == 3

    for r in data["results"]:
        assert "index" in r
        assert "distance" in r
        assert "text" in r
        assert isinstance(r["index"], int)
        assert isinstance(r["distance"], float)
        assert isinstance(r["text"], str)