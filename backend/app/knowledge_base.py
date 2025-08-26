import numpy as np
import PyPDF2
import torch
import logging
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional
from pathlib import Path
from config import settings
from vector_db import VectorDatabase
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class KnowledgeBase:
    def __init__(self):
        """Initialize with ChromaDB vector database and embedding model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(__name__)
        self._initialize_models()
        self.vector_db = VectorDatabase()
        self.chunk_overlap = settings.CHUNK_OVERLAP or 50  # Word overlap between chunks

    def _initialize_models(self):
        """Initialize models with error handling"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL)
            self.model = AutoModel.from_pretrained(settings.EMBEDDING_MODEL).to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise RuntimeError("Could not initialize embedding models")

    def extract_text(self, pdf_path: Path) -> str:
        """Robust text extraction from PDF with page-level error handling"""
        text = []
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                for i in tqdm(range(total_pages), desc="Extracting pages"):
                    try:
                        page = reader.pages[i]
                        if page_text := page.extract_text():
                            # Add page metadata
                            text.append(f"<PAGE {i+1}>\n{page_text.strip()}")
                    except Exception as e:
                        self.logger.warning(f"Error reading page {i+1}: {str(e)}")
                        continue
                        
                if not text:
                    raise ValueError("No text could be extracted from the PDF")
                    
                return '\n\n'.join(text)
                
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {str(e)}")
            raise RuntimeError(f"Text extraction failed: {str(e)}")

    def _smart_chunking(self, text: str) -> List[Dict]:
        """
        Advanced chunking that:
        - Preserves document structure
        - Maintains context with overlap
        - Handles sections and subsections
        """
        chunks = []
        sections = re.split(r'\n\s*\n', text)  # Split by multiple newlines
        current_section = "Document"
        
        for section in sections:
            # Detect section headers
            if self._is_section_header(section):
                current_section = section.strip()
                continue
                
            words = section.split()
            for i in range(0, len(words), settings.CHUNK_SIZE - self.chunk_overlap):
                chunk_words = words[i:i + settings.CHUNK_SIZE]
                chunk_text = ' '.join(chunk_words)
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "section": current_section,
                        "word_count": len(chunk_words),
                        "source": str(settings.PDF_PATH),
                        "chunk_id": f"{len(chunks):04d}"
                    }
                })
                
        return chunks

    def _is_section_header(self, text: str) -> bool:
        """Identify section headers in document text"""
        lines = text.split('\n')
        if len(lines) != 1:
            return False
            
        line = lines[0].strip()
        return (
            line.isupper() or 
            line.endswith(':') or 
            (len(line.split()) <= 5 and any(c.isdigit() for c in line))
        )

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch process embeddings for efficiency"""
        try:
            inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Pooling strategy (mean pooling)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
            return [emb.flatten().tolist() for emb in embeddings]
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            raise RuntimeError("Could not generate embeddings")

    def process_document(self, pdf_path: Optional[Path] = None):
        """
        Full document processing pipeline with:
        - Progress tracking
        - Batch processing
        - Error recovery
        """
        try:
            pdf_path = pdf_path or settings.PDF_PATH
            self.logger.info(f"Processing document: {pdf_path.name}")
            
            # Text extraction
            text = self.extract_text(pdf_path)
            
            # Chunking
            chunks = self._smart_chunking(text)
            self.logger.info(f"Created {len(chunks)} text chunks")
            
            # Batch processing for embeddings
            batch_size = settings.EMBEDDING_BATCH_SIZE or 8
            embeddings = []
            
            with ThreadPoolExecutor() as executor:
                batches = [
                    [chunk["text"] for chunk in chunks[i:i + batch_size]]
                    for i in range(0, len(chunks), batch_size)
                ]
                
                for batch in tqdm(batches, desc="Generating embeddings"):
                    try:
                        embeddings.extend(self.embed_batch(batch))
                    except Exception as e:
                        self.logger.error(f"Batch failed: {str(e)}")
                        continue
            
            # Prepare data for vector DB
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Store in vector database
            self.vector_db.add_vectors(
                embeddings=embeddings,
                texts=texts,
                metadata=metadatas
            )
            
            self.logger.info(f"Successfully processed {len(chunks)} chunks")
            return len(chunks)
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {str(e)}")
            raise RuntimeError(f"Processing failed: {str(e)}")

    def query(self, text: str, k: int = 3) -> List[Dict]:
        """Query the knowledge base with error handling"""
        try:
            embedding = self.embed(text)
            return self.vector_db.query(embedding, k=k)
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            return []