import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load env variables before importing modules that depend on them
load_dotenv()

from configs.settings import settings
from src.parsers.pdf_parser import PDFParser
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.bm25_store import BM25Manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting ingestion pipeline...")
    
    # 1. Parse PDF and extract chunks
    pdf_path = str(settings.base_dir / "ifc-annual-report-2024-financials.pdf")
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found at {pdf_path}")
        return
        
    parser = PDFParser(chunk_size=1000, chunk_overlap=200)
    chunks = parser.get_document_chunks(pdf_path)
    
    if not chunks:
        logger.error("No text chunks extracted.")
        return
        
    # 2. Create Vector Stores
    vector_manager = VectorStoreManager()
    
    # Create FAISS
    vector_manager.create_faiss_store(chunks)
    
    # Create Qdrant
    vector_manager.create_qdrant_store(chunks)
    
    # Create BM25
    bm25_manager = BM25Manager()
    bm25_manager.create_bm25_store(chunks)
    
    logger.info("Ingestion complete.")

if __name__ == "__main__":
    main()
