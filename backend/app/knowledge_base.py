import numpy as np
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
from pathlib import Path
from config import settings
from vector_db import VectorDatabase  # Import the ChromaDB implementation

class KnowledgeBase:
    def __init__(self):
        """Initialize with ChromaDB vector database and embedding model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(settings.EMBEDDING_MODEL).to(self.device)
        self.vector_db = VectorDatabase()
        
    def extract_text(self) -> str:
        """Extract text from PDF with improved error handling"""
        try:
            with open(settings.PDF_PATH, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = []
                for i, page in enumerate(reader.pages):
                    try:
                        if page_text := page.extract_text():
                            text.append(page_text)
                    except Exception as e:
                        print(f"Error reading page {i}: {str(e)}")
                return ' '.join(text)
        except Exception as e:
            raise RuntimeError(f"Failed to extract text: {str(e)}")
            
    def chunk_text(self, text: str) -> List[Dict]:
        """
        Improved text chunking with section tracking
        Returns list of dicts with text and metadata
        """
        sections = text.split('\n\n')  # Split by double newlines (common in PDFs)
        chunks = []
        current_chunk = []
        current_length = 0
        current_section = "Introduction"  # Default section
        
        for section in sections:
            # Detect section headers (all caps or followed by colon)
            if (section.isupper() or ':' in section.split()[0]) and len(section.split()) < 10:
                current_section = section.strip()
                continue
                
            sentences = section.split('. ')
            for sentence in sentences:
                sentence_length = len(sentence.split())
                if current_length + sentence_length <= settings.CHUNK_SIZE:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                else:
                    chunk_text = '. '.join(current_chunk) + '.'
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "section": current_section,
                            "length": len(chunk_text.split()),
                            "source": settings.PDF_PATH.name
                        }
                    })
                    current_chunk = [sentence]
                    current_length = sentence_length
        
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "section": current_section,
                    "length": len(chunk_text.split()),
                    "source": settings.PDF_PATH.name
                }
            })
            
        return chunks
        
    def embed(self, text: str) -> List[float]:
        """Generate embeddings and return as list for ChromaDB"""
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding='max_length',
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert to list for ChromaDB compatibility
        return np.mean(outputs.last_hidden_state.cpu().numpy(), axis=1).flatten().tolist()
    
    def process_document(self):
        """Full document processing pipeline"""
        print("Extracting text from PDF...")
        text = self.extract_text()
        
        print("Chunking text...")
        chunks = self.chunk_text(text)
        
        print("Generating embeddings...")
        embeddings = [self.embed(chunk["text"]) for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        print("Storing in vector database...")
        self.vector_db.add_vectors(
            embeddings=embeddings,
            texts=texts,
            metadata=metadatas
        )
        
        print(f"Processed {len(chunks)} chunks successfully")

