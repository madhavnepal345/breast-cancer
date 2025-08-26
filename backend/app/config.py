from pathlib import Path

class Settings:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    QA_MODEL = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"
    CHUNK_SIZE = 256
    CHUNK_OVERLAP = 50
    EMBEDDING_BATCH_SIZE = 8  # For GPU optimization
    MIN_CONFIDENCE = 0.4  # Only accept answers with >=40% confidence
    MIN_ANSWER_LENGTH = 50

    PDF_PATH = Path("data/cancer_encyclopedia.pdf")
    HYBRID_SEARCH_ALPHA = 0.7  # Weight for vector vs keyword search


    
settings = Settings()
