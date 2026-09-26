from typing import List, Dict, Any

from .clients.groq_client import call_groq_chat
from .schemas import EvaluateAbstentionResponse


MAX_CONTEXT_CHARS = 16000


def build_context(context_docs: list[dict[str, Any]]) -> str:
    context_lines = []
    total_chars = 0

    for i, doc in enumerate(context_docs, start=1):
        text = str(doc.get("text", ""))

        line = f"[{i}] {text}"
        remaining = MAX_CONTEXT_CHARS - total_chars

        if remaining <= 0:
            break

        line = line[:remaining]
        context_lines.append(line)
        total_chars += len(line)

    return "\n\n".join(context_lines)


def generate_answer(
    query: str,
    context_docs: List[Dict[str, Any]],
) -> str:
    """
    Generate an answer using Groq, grounded in the retrieved context documents.

    context_docs:
        A list of document dictionaries containing at least a "text" field.
    """
    context_text = build_context(context_docs)

    system_prompt = (
        "You are a fraud analysis assistant. "
        "Use the provided SMS examples to reason about whether a given "
        "message is likely spam or not. "
        "If the context is insufficient, say so clearly."
        "Do not use Markdown code fences."
        "Return the JSON object directly, with no text before or after it."
    )

    user_prompt = (
        f"Context (SMS examples):\n{context_text}\n\n"
        f"User question: {query}\n\n"
        "Answer concisely and explain your reasoning based on the context."
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    return call_groq_chat(messages)


def generate_abstention_response(
    query: str,
    context_docs: list[dict[str, Any]],
) -> EvaluateAbstentionResponse:
    context_text = build_context(context_docs)

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
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    raw_output = call_groq_chat(messages)

    cleaned_output = clean_json_output(raw_output)

    return EvaluateAbstentionResponse.model_validate_json(
        cleaned_output
    )

def clean_json_output(raw_output: str) -> str:
    cleaned = raw_output.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned