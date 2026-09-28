from __future__ import annotations
from pydantic import json

from .clients.groq_client import call_groq_chat
from .schemas import EvaluateAbstentionResponse

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 16000


def build_context(
    context_docs: list[dict[str, Any]],
) -> str:
    context_lines: list[str] = []
    total_chars = 0

    for index, doc in enumerate(
        context_docs,
        start=1,
    ):
        text = str(doc.get("text", ""))
        line = f"[{index}] {text}"

        remaining = MAX_CONTEXT_CHARS - total_chars

        if remaining <= 0:
            break

        line = line[:remaining]
        context_lines.append(line)
        total_chars += len(line)

    return "\n\n".join(context_lines)


def generate_answer(
    query: str,
    context_docs: list[dict[str, Any]],
) -> str:
    """
    Generate a regular grounded answer.
    """
    context_text = build_context(context_docs)

    system_prompt = (
        "You are a fraud analysis assistant. "
        "Use only the supplied SMS context. "
        "If the context is insufficient, say so clearly. "
        "Do not use outside knowledge."
    )

    user_prompt = (
        f"Context (SMS examples):\n{context_text}\n\n"
        f"User question:\n{query}\n\n"
        "Answer concisely and ground every claim in the context."
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
You are a fraud-analysis assistant.

Use only the supplied SMS context.

Rules:
- Answer only when the supplied context supports the answer.
- If the context is insufficient, ambiguous, or contradictory, abstain.
- Do not guess or use outside knowledge.
- Do not provide instructions that facilitate fraud, evasion, theft,
  money laundering, or other wrongdoing.
- Set is_spam to null when the context does not support a reliable
  spam classification.
- If abstention_status is "abstain", answer must be null.
- If abstention_status is "answer", abstention_reason must be null.
- Return only one valid JSON object.
- Do not return Markdown.
- Do not return a refusal sentence outside the JSON object.

Return exactly this shape:

{
  "is_spam": true,
  "abstention_status": "answer",
  "answer": "string",
  "abstention_reason": null
}

For abstention, use:

{
  "is_spam": null,
  "abstention_status": "abstain",
  "answer": null,
  "abstention_reason": "string"
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

    try:
        json_text = extract_json_object(raw_output)

        response = (
            EvaluateAbstentionResponse.model_validate_json(
                json_text
            )
        )

        validate_abstention_invariants(response)

        return response

    except (
        ValueError,
        ValidationError,
    ) as exc:
        logger.warning(
            "Invalid structured abstention output: "
            "error=%s raw_output=%r",
            exc,
            raw_output[:1000],
        )

        return fallback_abstention_response(
            query=query,
            reason=(
                "The model did not return a valid structured "
                "abstention response."
            ),
        )


def extract_json_object(raw_output: str) -> str:
    cleaned = raw_output.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "No JSON object found in model output"
        )

    candidate = cleaned[start:end + 1]

    json.loads(candidate)

    return candidate


def validate_abstention_invariants(
    response: EvaluateAbstentionResponse,
) -> None:
    if response.abstention_status == "abstain":
        if response.answer is not None:
            raise ValueError(
                "Abstention response must have answer=null"
            )

        if not response.abstention_reason:
            raise ValueError(
                "Abstention response must include a reason"
            )

    if response.abstention_status == "answer":
        if not response.answer:
            raise ValueError(
                "Answer response must include answer text"
            )

        if response.abstention_reason is not None:
            raise ValueError(
                "Answer response must have "
                "abstention_reason=null"
            )


def fallback_abstention_response(
    query: str,
    reason: str,
) -> EvaluateAbstentionResponse:
    fields = {
        "query": query,
        "is_spam": None,
        "abstention_status": "abstain",
        "answer": None,
        "abstention_reason": reason,
    }

    try:
        return EvaluateAbstentionResponse.model_validate(
            fields
        )
    except ValidationError:
        fields.pop("query", None)

        return EvaluateAbstentionResponse.model_validate(
            fields
        )