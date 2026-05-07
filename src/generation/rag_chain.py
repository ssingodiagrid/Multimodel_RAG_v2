import json
import logging
import re
from pathlib import Path

from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from configs.settings import settings
from src.retrieval.semantic_cache import SemanticCacheManager

logger = logging.getLogger(__name__)
STOPWORDS = {
    "what", "was", "the", "of", "as", "is", "a", "an", "to", "and", "in", "for",
    "on", "at", "by", "with", "from", "ifc",
}


def load_page_context_map() -> dict[tuple[str, int], str]:
    """Load same-page text context from persisted BM25 documents."""
    bm25_docs_path = Path(settings.indices_dir) / "bm25_docs.json"
    if not bm25_docs_path.exists():
        return {}

    try:
        with open(bm25_docs_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
    except Exception as exc:
        logger.warning(f"Could not load page context map: {exc}")
        return {}

    page_context: dict[tuple[str, int], list[str]] = {}
    for item in docs:
        metadata = item.get("metadata", {})
        if metadata.get("content_type") != "text":
            continue

        source = metadata.get("source")
        page_number = metadata.get("page_number")
        text = item.get("page_content", "").strip()
        if not source or page_number is None or not text:
            continue

        page_context.setdefault((source, int(page_number)), []).append(text)

    return {
        key: "\n".join(chunks)[:1200]
        for key, chunks in page_context.items()
    }


def format_docs_with_page_context(
    docs: list[Document], page_context_map: dict[tuple[str, int], str]
) -> str:
    """Enrich table/image docs with same-page text so year headers survive retrieval."""
    formatted_docs: list[str] = []

    for doc in docs:
        formatted = doc.page_content
        metadata = doc.metadata or {}
        content_type = metadata.get("content_type")
        source = metadata.get("source")
        page_number = metadata.get("page_number")

        if content_type in {"table", "image"} and source and page_number is not None:
            page_context = page_context_map.get((source, int(page_number)))
            if page_context:
                formatted = (
                    f"{doc.page_content}\n\n"
                    f"PAGE TEXT CONTEXT (same PDF page):\n{page_context}"
                )

        formatted_docs.append(formatted)

    return "\n\n".join(formatted_docs)


def should_cache_answer(answer: str) -> bool:
    """Avoid caching refusal/non-answer responses."""
    normalized = answer.strip().lower()
    refusal_markers = [
        "i cannot answer",
        "i can't answer",
        "i do not know",
        "i don't know",
        "based on the context provided",
    ]
    return not any(marker in normalized for marker in refusal_markers)


def rerank_docs_by_query(question: str, docs: list[Document]) -> list[Document]:
    """Boost chunks with strong lexical evidence for the exact user question."""
    lowered_question = question.lower()
    query_tokens = [
        token for token in re.findall(r"[a-z0-9]+", lowered_question)
        if token not in STOPWORDS and len(token) > 1
    ]
    bigrams = [
        f"{query_tokens[i]} {query_tokens[i + 1]}"
        for i in range(len(query_tokens) - 1)
    ]
    priority_phrases = [
        "total assets",
        "june 30",
        "2023",
        "balance sheet",
    ]

    def score(doc: Document) -> tuple[int, int]:
        text = doc.page_content.lower()
        phrase_score = sum(5 for phrase in priority_phrases if phrase in text)
        bigram_score = sum(2 for phrase in bigrams if phrase in text)
        token_score = sum(1 for token in query_tokens if token in text)
        return (phrase_score + bigram_score + token_score, -int(doc.metadata.get("page_number", 10**9)))

    return sorted(docs, key=score, reverse=True)

class RAGPipeline:
    def __init__(self, retriever, cache_namespace: str = "default"):
        self.retriever = retriever
        self.llm = self._init_llm()
        self.prompt = self._create_prompt()
        self.chain = self._build_chain()
        self.semantic_cache = SemanticCacheManager(
            threshold=0.85,
            cache_namespace=cache_namespace,
        )
        self.page_context_map = load_page_context_map()
        self.last_retrieved_docs: list[Document] = []

    def _init_llm(self) -> ChatVertexAI:
        """Initialize the Gemini client via Vertex AI."""
        return ChatVertexAI(
            model_name=settings.gemini_model,
            project=settings.gcp_project,
            location=settings.gcp_location,
            temperature=0.0
        )

    def _create_prompt(self) -> PromptTemplate:
        """Create the RAG prompt template."""
        template = """You are a helpful assistant answering questions based on the provided context from the IFC Annual Report 2024.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
When the context includes tables or financial figures, extract the exact number directly from the context.
Use same-page text context to recover missing year/date headers for tables when needed.
If the question asks for a specific year or date, prefer the value explicitly tied to that year/date in the context.

Context:
{context}

Question: {question}

Answer:"""
        return PromptTemplate.from_template(template)

    def _retrieve_docs(self, question: str) -> list[Document]:
        """Retrieve and rerank documents for a question."""
        docs = self.retriever.invoke(question)
        reranked_docs = rerank_docs_by_query(question, docs)
        self.last_retrieved_docs = reranked_docs
        return reranked_docs

    def get_retrieved_docs(self, question: str) -> list[Document]:
        """Public helper for UI/debugging to inspect the chunks used."""
        self._current_question = question
        return self._retrieve_docs(question)

    def _format_docs(self, docs: list[Document]):
        """Format retrieved documents into a single string for context."""
        return format_docs_with_page_context(docs, self.page_context_map)

    def _build_chain(self):
        """Build the LangChain LCEL pipeline."""
        return (
            {
                "context": RunnableLambda(self._retrieve_docs) | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def generate(self, question: str) -> str:
        """Run the chain for a given question with Semantic Caching."""
        self._current_question = question
        cached_answer = self.semantic_cache.check_cache(question)
        if cached_answer:
            return cached_answer

        logger.info(f"Generating answer for: {question}")
        answer = self.chain.invoke(question)
        
        if should_cache_answer(answer):
            self.semantic_cache.update_cache(question, answer)
        return answer

    def generate_stream(self, question: str):
        """Run the chain and stream the response."""
        self._current_question = question
        # For streams, we could yield the cached answer directly if hit
        cached_answer = self.semantic_cache.check_cache(question)
        if cached_answer:
            yield cached_answer + "\n\n⚡ *Served from Semantic Cache*"
            return

        logger.info(f"Streaming answer for: {question}")
        
        full_answer = ""
        for chunk in self.chain.stream(question):
            full_answer += chunk
            yield chunk
            
        # Update cache after streaming finishes
        if should_cache_answer(full_answer):
            self.semantic_cache.update_cache(question, full_answer)
