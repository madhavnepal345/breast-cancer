import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_db import VectorDB
from typing import List
from retrieval.qa_types import RetrievedChunk


class Retriever:
    def __init__(self, vectordb: VectorDB, top_k: int = 5):
        self.db = vectordb
        self.top_k = top_k

    def fetch(self, query_text: str) -> List[RetrievedChunk]:
        try:
            res = self.db.query(query_text, n_results=self.top_k)
            
            # Debug print to see what we're getting
            print(f"VectorDB query result type: {type(res)}")
            print(f"VectorDB query result structure: {res}")
            
            # Handle different response formats from vector databases
            if isinstance(res, list):
                # Some vector DBs return a list directly
                if not res:
                    return []
                
                # If it's a list of results, take the first one
                if isinstance(res[0], dict):
                    res = res[0]
                else:
                    # If it's a flat list, we need to handle differently
                    print(f"Unexpected list format in VectorDB response")
                    return []
            
            # Now res should be a dictionary
            if not isinstance(res, dict):
                print(f"VectorDB returned unexpected type: {type(res)}")
                return []
            
            # Safely extract data with proper error handling
            ids = res.get("ids", [])
            docs = res.get("documents", [])
            dists = res.get("distances", [])
            
            # Handle nested lists (common in ChromaDB and similar)
            if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
                ids = ids[0]
            if isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], list):
                docs = docs[0]
            if isinstance(dists, list) and len(dists) > 0 and isinstance(dists[0], list):
                dists = dists[0]
            
            # Ensure all are lists and same length
            if not all(isinstance(x, list) for x in [ids, docs, dists]):
                print("VectorDB response format issue: not all fields are lists")
                return []
            
            if not (len(ids) == len(docs) == len(dists)):
                print(f"VectorDB response length mismatch: ids={len(ids)}, docs={len(docs)}, dists={len(dists)}")
                # Take minimum length to avoid index errors
                min_len = min(len(ids), len(docs), len(dists))
                ids = ids[:min_len]
                docs = docs[:min_len]
                dists = dists[:min_len]
            
            chunks = []
            for i, _id in enumerate(ids):
                if i < len(docs) and i < len(dists):
                    sim = max(0.0, min(1.0, 1 - dists[i]))  # Clamp similarity to [0,1]
                    chunks.append(RetrievedChunk(id=_id, text=docs[i], score=sim))
            
            return chunks
            
        except Exception as e:
            print(f"Error in Retriever.fetch: {e}")
            print(f"Query: {query_text}")
            return []