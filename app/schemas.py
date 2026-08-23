from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    message: str
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str
    retrieved: Optional[List[Dict[str, Any]]] = None