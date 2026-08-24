from app.retriever import naive_retriever


DOCUMENTS = [
    {
        "id": "1",
        "text": "Free entry to win a prize. Text WIN now.",
        "label": "spam",
        "source": "test",
    },
    {
        "id": "2",
        "text": "Can you call me after work?",
        "label": "ham",
        "source": "test",
    },
    {
        "id": "3",
        "text": "You won a free cash prize. Call today.",
        "label": "spam",
        "source": "test",
    },
]


def test_returns_best_matching_document():
    results = naive_retriever("free entry win", DOCUMENTS, top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == "1"
    assert results[0]["label"] == "spam"


def test_respects_top_k():
    results = naive_retriever("free win prize", DOCUMENTS, top_k=2)

    assert len(results) == 2


def test_returns_empty_list_when_no_words_match():
    results = naive_retriever("cryptocurrency wallet", DOCUMENTS, top_k=5)

    assert results == []


def test_is_case_insensitive():
    lower_case = naive_retriever("free entry", DOCUMENTS, top_k=1)
    upper_case = naive_retriever("FREE ENTRY", DOCUMENTS, top_k=1)

    assert lower_case == upper_case