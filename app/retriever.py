import warnings
from typing import List, Dict, Any

def naive_retriever(query: str, documents: List[Dict[str, Any]], top_k: int=5) -> List[Dict[str, Any]]:
    warnings.warn(
        "naive_retriever is deprecated and kept for reference/experiments only. "
        "Use the FAISS-based retriever for semantic retrieval.",
        DeprecationWarning,
        stacklevel=2,
    )
    """
        Very simple retrieval:
          - Lowercase query and document text.
          - Count how many query words appear in each document.
          - Return top_k documents sorted by score.
    """
    query_words = set(query.lower().split())

    scored = []
    for doc in documents:
        text = doc["text"].lower()
        # Very crude: count matching words
        score = sum(1 for w in query_words if w in text)
        if score > 0:
            scored.append((score, doc))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:top_k]]