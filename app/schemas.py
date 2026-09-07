from pydantic import BaseModel
from typing import List, Dict, Any, Optional

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