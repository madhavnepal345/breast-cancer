from typing import List, Tuple
from knowledgebase import KnowledgeBase
from qa_types import QAResult, RetrievedChunk
from biobert_qa import BioBertQA
from confidence import ConfidenceScorer
from vector_db import VectorDB
from retriever import Retriever
import re
import traceback

# For LLaMA integration
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class CancerQAEngine:
    def __init__(
        self,
        vectordb: VectorDB,
        retriever_k: int = 5,
        biobert_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        llama_model: str = "meta-llama/Llama-2-7b-chat-hf",
    ):
        try:
            # Knowledge base
            self.kb = KnowledgeBase()

            # Retriever
            self.retriever = Retriever(vectordb, top_k=retriever_k)

            # BioBERT QA model
            self.biobert = BioBertQA(model_name=biobert_model)

            # Confidence scorer
            self.conf = ConfidenceScorer()

            # LLaMA model for rewriting/fluidifying answers
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model)
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model,
                device_map="auto",   # Automatically assign GPU/CPU
                torch_dtype="auto"   # Use mixed precision if possible
            )
            self.llama_pipeline = pipeline(
                "text-generation",
                model=self.llama_model,
                tokenizer=self.llama_tokenizer,
                max_length=512,
                do_sample=True,
                temperature=0.7
            )

            print(f"[INFO] CancerQAEngine initialized with {retriever_k} top-k retrieval.")
        except Exception as e:
            print("[ERROR] Failed to initialize CancerQAEngine:", e)
            traceback.print_exc()
            raise e

    # ------------------- Helper Methods ------------------- #
    def _extract_keywords_from_question(self, question: str) -> List[str]:
        """Extract keywords from user question (cancer types, organs)."""
        try:
            return re.findall(r'\b\w+\b', question.lower())
        except Exception as e:
            print("[ERROR] Keyword extraction failed:", e)
            return []

    def _filter_chunks_by_metadata(self, chunks: List[RetrievedChunk], question: str) -> List[RetrievedChunk]:
        """Filter chunks based on metadata relevance to question keywords."""
        try:
            keywords = self._extract_keywords_from_question(question)
            filtered = []
            for c in chunks:
                meta = c.metadata or {}
                text_match = any(k in c.text.lower() for k in keywords)
                cancer_match = any(k in [ct.lower() for ct in meta.get("cancer_types", [])] for k in keywords)
                organ_match = any(k in [o.lower() for o in meta.get("organs_affected", [])] for k in keywords)
                if text_match or cancer_match or organ_match:
                    filtered.append(c)
            return filtered if filtered else chunks
        except Exception as e:
            print("[ERROR] Chunk filtering failed:", e)
            traceback.print_exc()
            return chunks

    def _concat_context(self, chunks: List[RetrievedChunk], limit_chars: int = 3500) -> str:
        """Concatenate retrieved chunks into a context string with metadata."""
        try:
            out = []
            total = 0
            for c in chunks:
                meta_text = []
                meta = c.metadata or {}
                if meta.get("cancer_types"):
                    meta_text.append("Cancer Types: " + ", ".join(meta["cancer_types"]))
                if meta.get("organs_affected"):
                    meta_text.append("Organs: " + ", ".join(meta["organs_affected"]))
                if meta.get("tumor_characteristics"):
                    meta_text.append("Tumor Characteristics: " + ", ".join(meta["tumor_characteristics"]))
                if meta.get("treatments"):
                    meta_text.append("Treatments: " + ", ".join(meta["treatments"]))

                full_chunk = "\n".join(meta_text + [c.text])
                if total + len(full_chunk) > limit_chars:
                    remaining = max(0, limit_chars - total)
                    if remaining > 0:
                        out.append(full_chunk[:remaining])
                    break
                out.append(full_chunk)
                total += len(full_chunk)
            return "\n\n".join(out)
        except Exception as e:
            print("[ERROR] Context concatenation failed:", e)
            traceback.print_exc()
            return ""

    def _rewrite_with_llama(self, answer: str, context: str, question: str) -> str:
        """Rewrite or fluidify BioBERT's answer using LLaMA."""
        try:
            prompt = f"Rewrite the following answer to make it more clear and comprehensive based on the context.\n\nContext: {context}\n\nQuestion: {question}\nAnswer: {answer}\n\nRewritten Answer:"
            output = self.llama_pipeline(prompt)
            rewritten_answer = output[0]["generated_text"]
            # Remove prompt from generated text if included
            rewritten_answer = rewritten_answer.replace(prompt, "").strip()
            return rewritten_answer
        except Exception as e:
            print("[ERROR] LLaMA rewriting failed:", e)
            traceback.print_exc()
            return answer

    # ------------------- Main Method ------------------- #
    def ask(self, question: str, method_order: Tuple[str, ...] = ("kb", "biobert", "llama")) -> QAResult:
        """Answer a question using Knowledge Base, BioBERT, and LLaMA."""
        try:
            # 1. Knowledge Base
            if "kb" in method_order:
                kb_ans = self.kb.maybe_answer(question)
                if kb_ans:
                    return QAResult(
                        answer=kb_ans,
                        confidence=0.95,
                        used_chunks=[],
                        method="kb"
                    )

            # 2. Retrieve relevant chunks
            chunks = self.retriever.fetch(question)
            chunks = self._filter_chunks_by_metadata(chunks, question)
            context_text = self._concat_context(chunks)
            max_sim = max([c.score for c in chunks], default=0.0)

            # 3. BioBERT extract answer
            if "biobert" in method_order:
                if not context_text.strip() or not chunks:
                    return QAResult(
                        answer="I don't have enough information to answer that.",
                        confidence=0.2,
                        used_chunks=chunks,
                        method="fallback"
                    )

                qa_result = self.biobert.answer(question, context_text)
                biobert_answer = qa_result.get("answer", "")
                biobert_score = qa_result.get("score", 0.0)

                # Boost confidence based on metadata
                metadata_boost = 0.05 if any(
                    k in [ct.lower() for c in chunks[0].metadata.get("cancer_types", [])]
                    for k in self._extract_keywords_from_question(question)
                ) else 0.0

                conf = self.conf(max_similarity=max_sim, qa_score=biobert_score, context=context_text) + metadata_boost

                if biobert_answer and conf >= 0.55:
                    # 4. LLaMA rewrite/fluidify
                    if "llama" in method_order:
                        rewritten_answer = self._rewrite_with_llama(biobert_answer, context_text, question)
                    else:
                        rewritten_answer = biobert_answer

                    return QAResult(
                        answer=rewritten_answer,
                        confidence=conf,
                        used_chunks=chunks,
                        method="biobert+llama" if "llama" in method_order else "biobert",
                        extra={"qa_score": biobert_score}
                    )

            # 5. Fallback
            return QAResult(
                answer="I don't have enough information to answer that.",
                confidence=0.2,
                used_chunks=chunks,
                method="fallback"
            )

        except Exception as e:
            print("[ERROR] QA engine failed:", e)
            traceback.print_exc()
            return QAResult(
                answer="An error occurred while processing your question.",
                confidence=0.0,
                used_chunks=[],
                method="error"
            )
