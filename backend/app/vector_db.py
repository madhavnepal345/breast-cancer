import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
import json
from pathlib import Path
import fitz  # PyMuPDF for PDF parsing
import os
import uuid
import re
from config import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz

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
        self.initialized = False
        
        # Load document from config path if specified
        if hasattr(settings, 'PDF_PATH') and settings.PDF_PATH:
            self.load_pdf_document(settings.PDF_PATH)

    def _load_synonyms(self):
        """Load medical synonyms for query expansion"""
        try:
            synonym_path = Path("data/medical_synonyms.json")
            if synonym_path.exists():
                with open(synonym_path) as f:
                    self.synonyms = json.load(f)
            else:
                # Default medical synonyms
                self.synonyms = {
                    "cancer": ["tumor", "malignancy", "carcinoma", "neoplasm"],
                    "symptom": ["sign", "indication", "manifestation", "complaint"],
                    "treatment": ["therapy", "management", "intervention", "care"],
                    "diagnosis": ["identification", "detection", "assessment"],
                    "pain": ["discomfort", "ache", "soreness", "hurt"],
                    "drug": ["medication", "medicine", "pharmaceutical", "prescription"]
                }
        except:
            self.synonyms = {}

    def _expand_query(self, query: str) -> str:
        """Expand query with medical synonyms"""
        words = query.lower().split()
        expanded = []
        for word in words:
            expanded.append(word)
            if word in self.synonyms:
                expanded.extend(self.synonyms[word])
        return " ".join(expanded)

    def load_pdf_document(self, pdf_path: str, chunk_size: int = 500) -> Dict:
        """
        Load and process a PDF document into the vector database
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Maximum character length for each chunk
            
        Returns:
            Dictionary with loading statistics
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        documents = []
        metadatas = []
        ids = []
        
        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                for page_num in range(total_pages):
                    page = doc.load_page(page_num)
                    text = page.get_text().strip()
                    
                    if not text:
                        continue
                    
                    # Clean text
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    # Split into chunks if too long
                    if len(text) > chunk_size:
                        chunks = self._chunk_text(text, chunk_size)
                        for chunk_idx, chunk in enumerate(chunks):
                            documents.append(chunk)
                            metadatas.append({
                                "source": os.path.basename(pdf_path),
                                "page": page_num + 1,
                                "chunk": chunk_idx + 1,
                                "type": "pdf",
                                "total_chunks": len(chunks)
                            })
                            ids.append(f"{os.path.basename(pdf_path)}_p{page_num+1}_c{chunk_idx+1}")
                    else:
                        documents.append(text)
                        metadatas.append({
                            "source": os.path.basename(pdf_path),
                            "page": page_num + 1,
                            "chunk": 1,
                            "type": "pdf",
                            "total_chunks": 1
                        })
                        ids.append(f"{os.path.basename(pdf_path)}_p{page_num+1}")
            
            # Add to collection in batches to avoid memory issues
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
            
            self.initialized = True
            
            return {
                "success": True,
                "pages_processed": total_pages,
                "chunks_created": len(documents),
                "filename": os.path.basename(pdf_path)
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF: {str(e)}")

    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks while preserving sentence boundaries"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Perform hybrid search using both vector similarity and text matching
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with text, score, and metadata
        """
        if self.collection.count() == 0:
            return []
        
        expanded_query = self._expand_query(query)
        
        # Vector search
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
            # Calculate hybrid score (70% vector similarity, 30% text match)
            vector_score = 1 - dist  # Convert distance to similarity
            text_score = fuzz.token_sort_ratio(query, doc) / 100.0
            hybrid_score = (0.7 * vector_score) + (0.3 * text_score)
            
            ranked_results.append({
                "text": doc,
                "score": hybrid_score,
                "metadata": meta,
                "vector_score": vector_score,
                "text_score": text_score
            })
        
        # Return top k results
        return sorted(ranked_results, key=lambda x: x["score"], reverse=True)[:k]

    def semantic_search(self, query: str, k: int = 5) -> List[Dict]:
        """Pure semantic search using vector similarity"""
        if self.collection.count() == 0:
            return []
        
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            {
                "text": doc,
                "score": 1 - dist,  # Convert distance to similarity
                "metadata": meta
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def keyword_search(self, query: str, k: int = 5) -> List[Dict]:
        """Keyword-based search using text matching"""
        if self.collection.count() == 0:
            return []
        
        # Get all documents for keyword matching
        all_docs = self.collection.get(include=["documents", "metadatas"])
        
        ranked_results = []
        for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
            score = fuzz.token_sort_ratio(query, doc) / 100.0
            if score > 0.3:  # Minimum threshold
                ranked_results.append({
                    "text": doc,
                    "score": score,
                    "metadata": meta
                })
        
        return sorted(ranked_results, key=lambda x: x["score"], reverse=True)[:k]

    def document_count(self) -> int:
        """Get total number of document chunks in the database"""
        return self.collection.count()

    def get_document_info(self) -> Dict:
        """Get information about loaded documents"""
        if self.collection.count() == 0:
            return {"total_documents": 0, "sources": []}
        
        all_metas = self.collection.get(include=["metadatas"])["metadatas"]
        sources = {}
        
        for meta in all_metas:
            source = meta.get("source", "unknown")
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        
        return {
            "total_documents": self.collection.count(),
            "sources": sources
        }

    def clear_database(self) -> bool:
        """Clear all documents from the database"""
        try:
            self.collection.delete(where={})
            self.initialized = False
            return True
        except Exception as e:
            print(f"Error clearing database: {e}")
            return False

    def delete_document(self, source_name: str) -> bool:
        """Delete all chunks from a specific document"""
        try:
            self.collection.delete(where={"source": source_name})
            return True
        except Exception as e:
            print(f"Error deleting document {source_name}: {e}")
            return False