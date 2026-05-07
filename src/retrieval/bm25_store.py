import os
import json
import logging
from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from configs.settings import settings

logger = logging.getLogger(__name__)

class BM25Manager:
    def __init__(self):
        self.bm25_index_path = str(settings.indices_dir / "bm25_docs.json")

    def create_bm25_store(self, chunks: List[Document]) -> BM25Retriever:
        """Create a new BM25 index from chunks and save documents to disk."""
        logger.info("Creating BM25 sparse index...")
        
        retriever = BM25Retriever.from_documents(chunks)
        
        # Persist documents to disk as JSON to avoid Pydantic pickle issues
        os.makedirs(settings.indices_dir, exist_ok=True)
        docs_dict = [{"page_content": d.page_content, "metadata": d.metadata} for d in chunks]
        
        with open(self.bm25_index_path, 'w', encoding='utf-8') as f:
            json.dump(docs_dict, f)
            
        logger.info(f"BM25 index documents saved to {self.bm25_index_path}")
        return retriever

    def load_bm25_store(self) -> BM25Retriever:
        """Load documents from disk and rebuild BM25 index."""
        if not os.path.exists(self.bm25_index_path):
            raise FileNotFoundError(
                f"BM25 index not found at {self.bm25_index_path}. Run ingest.py first."
            )
            
        logger.info("Loading BM25 sparse index...")
        with open(self.bm25_index_path, 'r', encoding='utf-8') as f:
            docs_dict = json.load(f)
            
        chunks = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in docs_dict]
        retriever = BM25Retriever.from_documents(chunks)
            
        return retriever
