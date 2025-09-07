from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import os

class BioBertQA:
    def __init__(self, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", device=-1):
        """
        Load either a fine-tuned model or the base PubMedBERT QA model.
        """
        # If a local fine-tuned checkpoint exists, use it
        if os.path.isdir(model_name):
            model_path = model_name
        else:
            model_path = model_name  # Hugging Face model

        self.qa = pipeline(
            "question-answering",
            model=AutoModelForQuestionAnswering.from_pretrained(model_path),
            tokenizer=AutoTokenizer.from_pretrained(model_path),
            device=device
        )

    def answer(self, question: str, context: str):
        """
        Returns a dict with 'answer' and 'score'. Handles list outputs safely.
        """
        if not context.strip():
            return {"answer": "", "score": 0.0}

        qa_out = self.qa(question=question, context=context)

        # Handle list outputs from some pipelines
        if isinstance(qa_out, list):
            if len(qa_out) == 0:
                return {"answer": "", "score": 0.0}
            qa_out = qa_out[0]

        # Ensure dictionary
        if not isinstance(qa_out, dict):
            return {"answer": "", "score": 0.0}

        return {"answer": qa_out.get("answer", ""), "score": float(qa_out.get("score", 0.0))}
