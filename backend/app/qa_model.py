from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import torch
from typing import Dict, Optional
from config import settings
import numpy as np

class QAModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            settings.QA_MODEL or "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.QA_MODEL or "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"
        )
        self.min_confidence = 0.2
        
    def answer(self, context: str, question: str, conversation_history: str = "") -> Dict:
        inputs = self.tokenizer(
            f"{conversation_history} [SEP] {question}",
            context,
            truncation="only_second",
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        confidence = torch.mean(torch.stack([
            torch.max(torch.softmax(outputs.start_logits, dim=1)),
            torch.max(torch.softmax(outputs.end_logits, dim=1))
        ])).item()
        
        answer = self.tokenizer.decode(
            inputs.input_ids[0][answer_start:answer_end],
            skip_special_tokens=True
        )
        
        return {
            "answer": answer if confidence >= self.min_confidence else None,
            "confidence": confidence,
            "context_used": context
        }