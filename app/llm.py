from typing import List, Dict, Any
from .clients.groq_client import call_groq_chat
from .schemas import EvaluateAbstentionResponse


def generate_answer(query: str, context_docs: List[Dict[str,Any]]) -> str:
    """
        Generate an answer using Groq, grounded in the retrieved context docs.
        context_docs: list of document dicts with at least 'text' and optionally 'label'.
    """
    # Build context text
    context_lines = []
    for i, doc in enumerate(context_docs, start=1):
        text = doc.get("text", "")
        label = doc.get("label", "")
        context_lines.append(f"[{i}] (label={label}) {text}")

    context_text = "\n\n".join(context_lines)

    system_prompt = (
        "You are a fraud analysis assistant. "
        "Use the provided SMS examples to reason about whether a given message is likely spam or not. "
        "If the context is insufficient, say so clearly."
    )

    user_prompt = (
        f"Context (SMS examples):\n{context_text}\n\n"
        f"User question: {query}\n\n"
        "Answer concisely and explain your reasoning based on the context."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return call_groq_chat(messages)

def generate_abstention_response(
    query: str,
    context_docs: list[dict[str, Any]],
) -> EvaluateAbstentionResponse:
    context_lines = []

    for i, doc in enumerate(context_docs, start=1):
        text = doc.get("text", "")
        label = doc.get("label", "")
        context_lines.append(f"[{i}] (label={label}) {text}")

    context_text = "\n\n".join(context_lines)

    system_prompt = """
You are a fraud analysis assistant.

Use only the supplied SMS context to answer the query.

Rules:
- Answer only when the context supports the answer.
- If the context is insufficient, ambiguous, or contradictory, abstain.
- Do not guess or use outside knowledge.
- Determine whether the query itself is spam and set is_spam accordingly.
- Return only valid JSON with exactly these fields:
  {
    "is_spam": true or false,
    "abstention_status": "answer" or "abstain",
    "answer": "string or null",
    "abstention_reason": "string or null"
  }
"""

    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Query:\n{query}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_output = call_groq_chat(messages)

    return EvaluateAbstentionResponse.model_validate_json(raw_output)