import logging
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

logger = logging.getLogger(__name__)

def get_multi_query_retriever(base_retriever, llm):
    """
    Wraps a base retriever with LangChain's MultiQueryRetriever.
    It uses the LLM to generate multiple variations of the user's query,
    executes retrieval for each, and returns the unique union of all retrieved documents.
    """
    logger.info("Initializing MultiQueryRetriever for Query Decomposition...")
    
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
    return multi_query_retriever
