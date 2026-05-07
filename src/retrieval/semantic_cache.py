import os
import logging
import shutil
from pathlib import Path
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from configs.settings import settings
from src.embeddings.vertex_embeddings import get_embeddings

logger = logging.getLogger(__name__)

class SemanticCacheManager:
    """Manages an in-memory semantic cache using FAISS to speed up query response."""
    
    def __init__(self, threshold: float = 0.95, cache_namespace: str = "default"):
        self.cache_path = str(settings.indices_dir / "semantic_cache" / cache_namespace)
        self.embeddings = get_embeddings()
        self.threshold = threshold
        self.vector_store = self._load_or_create_cache()

    @staticmethod
    def cleanup_old_namespaces(cache_root: Path, active_namespace: str) -> None:
        """Remove old session-scoped cache directories, preserving the active one."""
        if not cache_root.exists():
            return

        for entry in cache_root.iterdir():
            if not entry.is_dir():
                continue
            if entry.name == active_namespace:
                continue
            shutil.rmtree(entry, ignore_errors=True)
        
    def _load_or_create_cache(self) -> FAISS:
        """Load existing cache or create a fresh empty one."""
        if os.path.exists(self.cache_path) and os.path.isdir(self.cache_path):
            try:
                return FAISS.load_local(self.cache_path, self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                logger.warning(f"Could not load semantic cache: {e}. Creating new one.")
                
        # Create an empty FAISS index by embedding a dummy document
        dummy_doc = Document(page_content="INITIAL_CACHE_DOCUMENT", metadata={"answer": ""})
        store = FAISS.from_documents([dummy_doc], self.embeddings)
        return store

    def check_cache(self, query: str) -> Optional[str]:
        """Check if a semantically similar query exists in the cache."""
        try:
            # We use relevance scores which normalize to [0, 1] for typical embeddings
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=1)
            
            if not results:
                return None
                
            doc, score = results[0]
            
            # Skip the dummy document
            if doc.page_content == "INITIAL_CACHE_DOCUMENT":
                return None
                
            logger.info(f"Cache check for '{query[:30]}...': closest match score {score:.4f}")
            
            if score >= self.threshold:
                logger.info(f"⚡ Semantic cache HIT! (Score: {score:.4f})")
                return doc.metadata.get("answer")
            
            return None
        except Exception as e:
            logger.error(f"Error checking semantic cache: {e}")
            return None
            
    def update_cache(self, query: str, answer: str) -> None:
        """Add a new query-answer pair to the semantic cache and save to disk."""
        try:
            doc = Document(
                page_content=query,
                metadata={"answer": answer}
            )
            self.vector_store.add_documents([doc])
            self.vector_store.save_local(self.cache_path)
            logger.info(f"Added query to semantic cache and saved.")
        except Exception as e:
            logger.error(f"Error updating semantic cache: {e}")
