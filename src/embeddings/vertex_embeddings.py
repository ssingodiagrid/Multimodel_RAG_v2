from langchain_google_vertexai import VertexAIEmbeddings
from configs.settings import settings
import os

def get_embeddings() -> VertexAIEmbeddings:
    """Initialize and return the Vertex AI embeddings client."""
    # Ensure credentials are set. If not, it will fall back to ADC.
    # In production, we assume GOOGLE_APPLICATION_CREDENTIALS is set in .env
    
    return VertexAIEmbeddings(
        model_name=settings.embedding_model,
        project=settings.gcp_project,
        location=settings.gcp_location,
        max_retries=3
    )
