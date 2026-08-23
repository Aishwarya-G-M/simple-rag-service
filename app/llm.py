from typing import List, Dict, Any
from .clients.groq_client import call_groq_chat

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
