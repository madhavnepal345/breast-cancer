from pathlib import Path

class Settings:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    QA_MODEL = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"
    CHUNK_SIZE = 500
    PDF_PATH = Path("data/cancer_encyclopedia.pdf")
    HYBRID_SEARCH_ALPHA = 0.7  # Weight for vector vs keyword search
    
settings = Settings()
print(f"Using embedding model: {settings.EMBEDDING_MODEL}")