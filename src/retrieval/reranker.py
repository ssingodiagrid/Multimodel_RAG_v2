import logging
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

logger = logging.getLogger(__name__)

def get_reranker(base_retriever, top_k: int = 5) -> ContextualCompressionRetriever:
    """
    Wrap a base retriever with a HuggingFace CrossEncoder re-ranker.
    
    Args:
        base_retriever: The retriever pulling initial Top-N results
        top_k: The final number of documents to return after re-ranking
    """
    logger.info("Initializing CrossEncoder re-ranker model: cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=top_k)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever
