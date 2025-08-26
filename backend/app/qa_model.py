from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import torch
from typing import Dict, Optional, Tuple, List
import logging
from config import settings
import re
import numpy as np

class QAModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = settings.QA_MODEL or "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        try:
            # Load model with error handling
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Initialize a separate generator for medical explanations
            self.generator = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",  # Better for instruction following
                device=0 if self.device == "cuda" else -1,
            )
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise RuntimeError("Could not initialize QA model")

        # Configuration
        self.min_confidence = settings.MIN_CONFIDENCE or 0.3
        self.min_answer_length = settings.MIN_ANSWER_LENGTH or 10
        self.max_input_length = 512

    def _clean_answer(self, answer: str) -> str:
        """Post-process answer text"""
        if not answer:
            return ""
            
        answer = re.sub(r'\s+', ' ', answer).strip()
        answer = re.sub(r'\[.*?\]', '', answer)
        answer = re.sub(r'\(.*?\)', '', answer)
        
        # Remove common QA artifacts
        artifacts = ["question:", "answer:", "context:", "comprehensive answer:"]
        for artifact in artifacts:
            answer = answer.replace(artifact, "")
            
        return answer

    def _calculate_confidence(self, start_logits, end_logits) -> float:
        """Calculate a more robust confidence score"""
        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs = torch.softmax(end_logits, dim=-1)
        
        start_score = torch.max(start_probs).item()
        end_score = torch.max(end_probs).item()
        
        # Geometric mean for combined confidence
        confidence = (start_score * end_score) ** 0.5
        
        # Adjust for medical queries (more conservative)
        return min(confidence * 0.9, confidence)  # Slightly reduce confidence for safety

    def _extract_answer(self, inputs, outputs) -> Tuple[str, float]:
        """Extract answer span with improved confidence scoring"""
        try:
            # Get the most likely start and end positions
            start_idx = torch.argmax(outputs.start_logits)
            end_idx = torch.argmax(outputs.end_logits)
            
            # Ensure end position is after start position
            if end_idx < start_idx:
                end_idx = start_idx + 1
                
            confidence = self._calculate_confidence(outputs.start_logits, outputs.end_logits)
            
            # Extract the answer tokens
            answer_tokens = inputs.input_ids[0][start_idx:end_idx+1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            return self._clean_answer(answer), confidence
            
        except Exception as e:
            self.logger.error(f"Answer extraction failed: {e}")
            return "", 0.0

    def _is_medical_query(self, question: str) -> bool:
        """Check if question is medical-related"""
        medical_keywords = [
            'cancer', 'symptom', 'disease', 'illness', 'treatment', 'diagnosis',
            'medicine', 'drug', 'pain', 'health', 'medical', 'doctor', 'hospital',
            'condition', 'virus', 'infection', 'prescription', 'pharmacy', 'tumor',
            'therapy', 'clinical', 'patient', 'sick', 'ache', 'fever', 'cough'
        ]
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in medical_keywords)

    def _generate_medical_explanation(self, question: str, short_answer: str, context: str) -> str:
        """Generate a safe medical explanation"""
        try:
            prompt = f"""
            Based on the following medical information, provide a careful and accurate response.
            If you're not certain, say so. Always recommend consulting a healthcare professional.
            
            Question: {question}
            Key Information: {short_answer}
            Context: {context[:300]}
            
            Answer:
            """
            
            generated = self.generator(
                prompt,
                max_length=200,
                num_return_sequences=1,
                temperature=0.3,  # Lower temperature for more conservative answers
                do_sample=True,
                truncation=True
            )
            
            answer = generated[0]['generated_text'].split("Answer:")[-1].strip()
            return self._clean_answer(answer)
            
        except Exception as e:
            self.logger.warning(f"Medical explanation generation failed: {e}")
            return short_answer  # Fallback to short answer

    def estimate_confidence(self, question: str, context: str) -> float:
        """Quick confidence estimation without full answer generation"""
        try:
            inputs = self.tokenizer(
                question,
                context,
                truncation="only_second",
                max_length=256,  # Shorter for faster estimation
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            confidence = self._calculate_confidence(outputs.start_logits, outputs.end_logits)
            return confidence
            
        except Exception as e:
            self.logger.error(f"Confidence estimation failed: {e}")
            return 0.0

    def answer(self, 
              context: str, 
              question: str, 
              conversation_history: str = "",
              generate_full_answer: bool = True) -> Dict:
        """
        Generate answer with improved confidence scoring
        """
        try:
            # Prepare input
            history_prefix = f"{conversation_history} " if conversation_history else ""
            full_question = f"{history_prefix}{question}"
            
            inputs = self.tokenizer(
                full_question,
                context,
                truncation="only_second",
                max_length=self.max_input_length,
                return_tensors="pt",
                padding="max_length",
                stride=128,  # Allow some overlap for better context handling
                return_overflowing_tokens=True
            ).to(self.device)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract base answer from the first sequence (most relevant)
            extracted_answer, confidence = self._extract_answer(inputs, outputs)
            
            # Handle medical queries with extra caution
            is_medical = self._is_medical_query(question)
            if is_medical:
                # Be more conservative with medical answers
                confidence = confidence * 0.8  # Reduce confidence for medical queries
                
            final_answer = extracted_answer
            was_expanded = False
            
            # Generate comprehensive answer if requested and confident enough
            if (generate_full_answer and 
                confidence >= self.min_confidence and 
                len(extracted_answer) > 0):
                
                if is_medical:
                    final_answer = self._generate_medical_explanation(question, extracted_answer, context)
                else:
                    final_answer = extracted_answer  # For non-medical, keep original
                
                was_expanded = final_answer != extracted_answer
            
            # Validate answer
            if (confidence < self.min_confidence or 
                len(final_answer.strip()) < self.min_answer_length):
                return {
                    "answer": None,
                    "confidence": confidence,
                    "context_used": context[:200] + "..." if len(context) > 200 else context,
                    "was_expanded": was_expanded,
                    "is_medical": is_medical,
                    "error": "Low confidence or insufficient answer"
                }
            
            # Add medical disclaimer if needed
            if is_medical and "consult a healthcare professional" not in final_answer.lower():
                final_answer += " Please consult a healthcare professional for medical advice."
            
            return {
                "answer": final_answer,
                "confidence": confidence,
                "context_used": context[:200] + "..." if len(context) > 200 else context,
                "was_expanded": was_expanded,
                "is_medical": is_medical
            }
            
        except Exception as e:
            self.logger.error(f"QA failed for question '{question}': {e}")
            return {
                "answer": None,
                "confidence": 0.0,
                "context_used": "",
                "error": str(e)
            }