 Comparative Analysis: Two RAG System Implementations

**Analysis Date:** April 25, 2026  
**Projects Analyzed:**
1. `rag_system/` - Production-grade multimodal RAG with Docker deployment

## 1. Project Structure Comparison

### rag_system/ Structure
```
rag_system/
├── app/                    # Streamlit UI
├── configs/                # Pydantic settings
├── data/
│   ├── raw/               # Source PDFs
│   ├── processed/         # Extracted chunks, ColPali patches
│   └── indices/           # FAISS indices
├── src/
│   ├── parsers/           # Text, table, image extraction
│   ├── embeddings/        # Google text-embedding-004
│   ├── retrieval/         # FAISS, Qdrant, BM25, hybrid, reranker
│   ├── generation/        # Gemini client, RAG chain
│   ├── evaluation/        # RAGAS, LLM judge
│   └── utils/             # Langfuse, semantic cache
├── docker-compose.yml     # Qdrant + app services
├── Dockerfile
└── ingest.py              # Master ingestion script
```

**Key Structural Differences:**
- `rag_system/` uses a **service-oriented** structure with Docker deployment

- `rag_system/` has a master `ingest.py` script; `RAG/` has `main.py` with separate scripts
---

## 2. Configuration Management

### rag_system/ Configuration
- **Framework:** Pydantic `BaseSettings` with automatic `.env` loading
- **File:** [`configs/settings.py`](rag_system/configs/settings.py)
- **Features:**
  - Type-safe configuration with Field validators
  - Automatic environment variable mapping
  - Path objects for file references
  - Separate collections for text/tables/images in Qdrant

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    gcp_project: str = Field("your-project", env="GCP_PROJECT")
    gemini_model: str = Field("gemini-2.0-flash-001", env="GEMINI_MODEL")
    embedding_model: str = Field("text-embedding-004", env="EMBEDDING_MODEL")
```
**Comparison:**
- `rag_system/` has **cleaner, type-safe** configuration with Pydantic


## 3. Technology Stack Comparison

| Component | rag_system/ | RAG/ |
|-----------|-------------|------|
| **PDF Processing** | PyMuPDF, pdfplumber | PyMuPDF, pdfplumber |
| **Text Embeddings** | Google text-embedding-004 | sentence-transformers (BGE-small) |
| **Vector Store** | FAISS + Qdrant | FAISS (primary) + Qdrant (optional) |
| **Sparse Retrieval** | BM25 (rank-bm25) | BM25 (rank-bm25) |
| **Reranking** | sentence-transformers CrossEncoder | sentence-transformers CrossEncoder |
| **LLM** | Gemini 2.0 Flash (Vertex AI) | Gemini 2.0 Flash (Vertex AI) |
| **Framework** | LangChain LCEL | Custom pipeline orchestration |
| **UI** | Streamlit | Streamlit |
| **Observability** | Langfuse | Langfuse |
| **Deployment** | Docker Compose | Local/manual |
| **Testing** | Basic tests | Extensive pytest suite (30+ files) |

**Key Differences:**
- `rag_system/` uses **Google's text-embedding-004** (cloud-based)
- `RAG/` uses **BGE-small** (local sentence-transformers)
- `rag_system/` leverages **LangChain LCEL** for pipeline composition
---

## 4. Retrieval Pipeline Architecture

### rag_system/ Retrieval
**File:** [`src/retrieval/hybrid.py`](rag_system/src/retrieval/hybrid.py)

**Approach:**
- Uses **Reciprocal Rank Fusion (RRF)** for merging dense + sparse results
- LangChain-based retriever composition
- Separate vector stores for text/tables/images in Qdrant
- Standard RRF constant k=60

```python
def reciprocal_rank_fusion(ranked_lists: List[List[Document]], k: int = 60):
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            doc_id = str(hash(doc.page_content[:200]))
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
```
## 5. Generation Pipeline

### rag_system/ Generation
**File:** [`src/generation/rag_chain.py`](rag_system/src/generation/rag_chain.py)

**Approach:**
- **LangChain LCEL** (LangChain Expression Language) pipeline
- Declarative chain composition
- Built-in streaming support
- Automatic Langfuse tracing via callbacks

```python
class RAGPipeline:
    def __init__(self, text_store, table_store, image_store, bm25, use_reranker, mode):
        # LCEL chain: retrieve → rerank → generate → trace
        self.chain = (
            RunnablePassthrough()
            | RunnableLambda(self._retrieve)
            | RunnableLambda(self._rerank)
            | RunnableLambda(self._generate)
        )
```
### Unique Features in rag_system/
1. **Docker Deployment** - Production-ready containerization
2. **LangChain LCEL** - Declarative pipeline composition
3. **Separate Qdrant Collections** - Better modality isolation
4. **Google text-embedding-004** - Cloud-based embeddings

