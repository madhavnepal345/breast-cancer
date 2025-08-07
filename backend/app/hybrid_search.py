from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple

class HybridSearch:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        
    def build_index(self, documents: List[str]):
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        
    def search(self, 
               query: str, 
               vector_results: List[Tuple[str, float]], 
               alpha: float = 0.7) -> List[Tuple[str, float]]:
        
        query_vec = self.vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        combined_results = []
        for (text, vec_score), kw_score in zip(vector_results, keyword_scores):
            combined_score = alpha * (1 - vec_score) + (1 - alpha) * (1 - kw_score)
            combined_results.append((text, 1 - combined_score))  # Convert back to similarity
            
        return sorted(combined_results, key=lambda x: x[1], reverse=True)