from dataclasses import dataclass
from typing import List, Tuple, Any, Dict, Optional

@dataclass
class QAResult:
    answer: str
    confidence: float
    used_chunks: List[Any]  
    method: str
    extra: Optional[Dict[str, Any]] = None


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    metadata: Optional[Dict[str, Any]] = None