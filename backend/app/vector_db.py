import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
import fitz  # PyMuPDF for PDF parsing
import os
import uuid
from config import settings  # ✅ Importing settings

class VectorDatabase:
    def __init__(self, persist_dir: str = "chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="cancer_encyclopedia",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )
        self._load_synonyms()
        
        # ✅ Load document from path defined in config
        self._load_document_from_config()

    def _load_synonyms(self):
        try:
            with open("data/medical_synonyms.json") as f:
                self.synonyms = json.load(f)
        except:
            self.synonyms = {}

    def _expand_query(self, query: str) -> str:
        words = query.split()
        expanded = []
        for word in words:
            expanded.append(word)
            if word.lower() in self.synonyms:
                expanded.extend(self.synonyms[word.lower()])
        return " ".join(expanded)

    def _load_document_from_config(self):
        """
        Load PDF document from the configured path if not already loaded.
        """
        if self.collection.count() > 0:
            return  # Skip loading if already indexed

        pdf_path = settings.PDF_PATH
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Configured document {pdf_path} not found")

        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if not text:
                continue
            self.collection.add(
                documents=[text],
                metadatas=[{"source":str(pdf_path), "page": page_num}],
                ids=[str(uuid.uuid4())]
            )

    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        expanded_query = self._expand_query(query)
        vector_results = self.collection.query(
            query_texts=[expanded_query],
            n_results=k * 2,
            include=["documents", "metadatas", "distances"]
        )
        
        ranked_results = []
        for doc, meta, dist in zip(
            vector_results["documents"][0],
            vector_results["metadatas"][0],
            vector_results["distances"][0]
        ):
            score = (0.7 * (1 - dist)) + (0.3 * fuzz.token_sort_ratio(query, doc) / 100)
            ranked_results.append({
                "text": doc,
                "score": score,
                "metadata": meta
            })
        
        return sorted(ranked_results, key=lambda x: x["score"], reverse=True)[:k]

    def document_count(self) -> int:
        return self.collection.count()
