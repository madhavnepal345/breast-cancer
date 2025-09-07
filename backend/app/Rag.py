from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from retrieval.retriever import RetrievedChunk


class RAG:
    def __init__(self, model_name: str = "google/flan-t5-base", device: Optional[int] = None, max_new_tokens: int = 256):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
# Auto-detect model type (seq2seq vs causal)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.kind = "seq2seq"
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.kind = "causal"
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM '{model_name}': {e}")


    def _build_prompt(self, question: str, context_blocks: List[RetrievedChunk]) -> str:
        context_text = "\n\n".join([f"[Chunk {c.id} | sim={c.score:.2f}]\n{c.text}" for c in context_blocks])
        instructions = (
            "You are a careful cancer-awareness assistant. Answer the question using ONLY the provided context. "
            "If the answer is not present, say 'I don't have enough information in the provided materials.' "
            "Be concise, factual, and non-diagnostic."
            )
        return (
            f"{instructions}\n\nContext:\n{context_text}\n\nQuestion: {question}\nAnswer:"
            )


    def generate(self, question: str, context_blocks: List[RetrievedChunk]) -> str:
        prompt = self._build_prompt(question, context_blocks)
        if self.kind == "seq2seq":
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip()