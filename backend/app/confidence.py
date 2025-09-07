class ConfidenceScorer:
    def __init__(self, w_sim: float = 0.6, w_qa: float = 0.4, min_context_chars: int = 200):
        self.w_sim = w_sim
        self.w_qa = w_qa
        self.min_context_chars = min_context_chars

    def __call__(self, max_similarity: float, qa_score: float, context: str) -> float:
        base = (self.w_sim * float(max_similarity)) + (self.w_qa * float(qa_score))
        if len(context) < self.min_context_chars:
            base *= 0.85
        return float(max(0.0, min(1.0, base)))
