import os
import logging
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from configs.settings import settings
from src.embeddings.vertex_embeddings import get_embeddings

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """Return a QdrantClient configured for the current QDRANT_MODE.

    - server: connects to Qdrant running in Docker (or any remote server)
              via HTTP. Multiple clients can connect simultaneously — no
              file-lock issues.
    - local:  opens the on-disk file store. Only one process may hold the
              lock at a time; prefer 'server' for the Streamlit app.
    """
    if settings.qdrant_mode == "server":
        logger.info(f"Connecting to Qdrant server at {settings.qdrant_url}")
        return QdrantClient(url=settings.qdrant_url)
    else:
        qdrant_path = str(settings.indices_dir / "qdrant_db")
        logger.info(f"Opening local Qdrant storage at {qdrant_path}")
        return QdrantClient(path=qdrant_path)


class VectorStoreManager:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.faiss_index_path = str(settings.indices_dir / "faiss_index")

    # ── FAISS (no Docker needed — pure local file) ─────────────────────────────

    def create_faiss_store(self, chunks: List[Document]) -> FAISS:
        """Create a new FAISS vector store from chunks and save it."""
        logger.info("Creating FAISS index...")
        batch_size = 10
        vector_store = None
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, self.embeddings)
            else:
                vector_store.add_documents(batch)
        vector_store.save_local(self.faiss_index_path)
        logger.info(f"FAISS index saved to {self.faiss_index_path}")
        return vector_store

    def load_faiss_store(self) -> FAISS:
        """Load an existing FAISS vector store from disk."""
        if not os.path.exists(self.faiss_index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {self.faiss_index_path}. Run ingest.py first."
            )
        logger.info("Loading FAISS index...")
        return FAISS.load_local(
            self.faiss_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    # ── Qdrant (Docker server recommended; local fallback available) ───────────

    def create_qdrant_store(self, chunks: List[Document]) -> Qdrant:
        """Embed chunks and upload to Qdrant (used by ingest.py)."""
        logger.info(f"Pushing chunks to Qdrant [{settings.qdrant_mode} mode]...")
        client = get_qdrant_client()
        batch_size = 10
        vector_store = None
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            if vector_store is None:
                vector_store = Qdrant.from_documents(
                    batch,
                    self.embeddings,
                    url=settings.qdrant_url if settings.qdrant_mode == "server" else None,
                    path=str(settings.indices_dir / "qdrant_db") if settings.qdrant_mode == "local" else None,
                    collection_name=settings.text_collection_name,
                    force_recreate=True,
                )
            else:
                vector_store.add_documents(batch)
                
        logger.info(
            f"Qdrant collection '{settings.text_collection_name}' ready."
        )
        return vector_store

    def load_qdrant_store(self, client: QdrantClient | None = None) -> Qdrant:
        """Load the Qdrant vector store.

        Args:
            client: Optional pre-built QdrantClient (e.g. from st.cache_resource).
                    If None, a new client is created via get_qdrant_client().
        """
        if client is None:
            client = get_qdrant_client()
        return Qdrant(
            client=client,
            collection_name=settings.text_collection_name,
            embeddings=self.embeddings,
        )
