from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal


class ChatRequest(BaseModel):
    message: str
    top_k: int = 5
    # optional: used only by evaluation / attack suite
    scenario_id: str | None = None  # which test case
    input_type: str | None = None  # "benign" | "attack"
    attack_type: str | None = None  # e.g. "smishing", "prompt_injection"


class ChatResponse(BaseModel):
    answer: str
    retrieved: List[Dict[str, Any]]

class EvaluateAbstentionRequest(BaseModel):
    query: str
    top_k: int = 150

class EvaluateAbstentionResponse(BaseModel):
    is_spam: bool
    abstention_status: Literal["answer", "abstain"]
    answer: str | None = None
    abstention_reason: str | None = None