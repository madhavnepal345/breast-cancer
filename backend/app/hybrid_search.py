from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from scipy.sparse import csr_matrix
import pickle
from pathlib import Path

class HybridSearch:
    def __init__(self, alpha: float = 0.5, top_k: int = 10):
        """
        Enhanced hybrid search combining semantic and keyword search
        
        Args:
            alpha: Weight for semantic vs keyword search (0-1)
                  0.5 = equal weight, higher values favor semantic search
            top_k: Number of results to return
        """
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=50000,  # Prevent memory issues
            ngram_range=(1, 2)  # Include bigrams
        )
        self.tfidf_matrix = None
        self.documents = []
        self.alpha = alpha
        self.top_k = top_k
        self.logger = logging.getLogger(__name__)
        
    def build_index(self, documents: List[str]) -> None:
        """
        Build TF-IDF index with error handling and memory optimization
        
        Args:
            documents: List of text documents to index
        """
        try:
            self.documents = documents
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)
            self.logger.info(f"Built TF-IDF index for {len(documents)} documents")
        except Exception as e:
            self.logger.error(f"Index building failed: {str(e)}")
            raise RuntimeError("Could not build search index")

    def save_index(self, file_path: Path) -> None:
        """Save the search index to disk"""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'matrix': self.tfidf_matrix,
                    'documents': self.documents
                }, f)
            self.logger.info(f"Saved index to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save index: {str(e)}")
            raise

    def load_index(self, file_path: Path) -> None:
        """Load a pre-built search index"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.tfidf_matrix = data['matrix']
                self.documents = data['documents']
            self.logger.info(f"Loaded index from {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to load index: {str(e)}")
            raise

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to 0-1 range"""
        if scores.max() == scores.min():
            return np.ones_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())

    def search(self, 
              query: str, 
              vector_results: List[Dict],
              alpha: Optional[float] = None) -> List[Dict]:
        """
        Enhanced hybrid search with:
        - Score normalization
        - Flexible weighting
        - Better result formatting
        
        Args:
            query: Search query text
            vector_results: List of dicts with 'text' and 'score' from vector search
            alpha: Optional override for class alpha parameter
            
        Returns:
            List of results with combined scores, sorted by relevance
        """
        try:
            alpha = alpha if alpha is not None else self.alpha
            
            # Convert vector results to uniform format
            vec_scores = np.array([r['score'] for r in vector_results])
            texts = [r['text'] for r in vector_results]
            
            # Get TF-IDF scores
            query_vec = self.vectorizer.transform([query])
            kw_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Normalize scores
            norm_vec = self._normalize_scores(vec_scores)
            norm_kw = self._normalize_scores(kw_scores)
            
            # Combine scores
            combined_scores = (alpha * norm_vec) + ((1 - alpha) * norm_kw)
            
            # Prepare results with metadata
            results = []
            for i, score in enumerate(combined_scores):
                results.append({
                    'text': texts[i],
                    'score': float(score),
                    'vector_score': float(vec_scores[i]),
                    'keyword_score': float(kw_scores[i]),
                    'combined_score': float(score)
                })
            
            # Sort and return top_k results
            return sorted(results, key=lambda x: x['combined_score'], reverse=True)[:self.top_k]
            
        except Exception as e:
            self.logger.error(f"Search failed for query '{query}': {str(e)}")
            return []

    def batch_search(self, 
                    queries: List[str], 
                    vector_results: List[List[Dict]]) -> List[List[Dict]]:
        """
        Process multiple queries efficiently
        """
        return [self.search(q, res) for q, res in zip(queries, vector_results)]