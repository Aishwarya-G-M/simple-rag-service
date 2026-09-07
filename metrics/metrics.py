from pydantic import BaseModel
from typing import Literal, Optional


class RagRequestMetrics(BaseModel):
    scenario_id: str              # which eval/attack case
    backend: Literal["baseline_rag"]
    input_type: Literal["benign", "attack"]
    attack_type: Optional[str]    # e.g. "smishing", "prompt_injection", "kg_poisoning"

    top_k: int
    model_name: str

    safe: bool                    # unsafe vs policy?
    contradicts_kg: Optional[bool]
    abstained: bool

    correct: Optional[bool]       # for benign questions
    latency_ms: float