# Phase 3: Advanced Retrieval Techniques — Implementation Plan

## Objective
Elevate the baseline RAG system built in Phase 1 by introducing advanced retrieval strategies. The focus of Phase 3 is to improve retrieval precision through **Re-ranking**, enhance context awareness using **Metadata Integration**, and optionally implement robust **Sparse & Hybrid Retrieval** to handle exact keyword matches.

---

## Task Breakdown

### 1. Re-ranking Implementation
Currently, our retriever pulls the top-K chunks based purely on embedding similarity (dense retrieval). Re-ranking introduces a secondary model that evaluates the semantic relevance of the retrieved chunks against the exact query, re-ordering them to ensure the most relevant context is at the top.

**Core Tasks:**
- **Cross-Encoder Integration:** Integrate a HuggingFace Cross-Encoder (e.g., `BAAI/bge-reranker-base` or `cross-encoder/ms-marco-MiniLM-L-6-v2`) via LangChain's `ContextualCompressionRetriever`.
- **Pipeline Update:** Update `src/generation/rag_chain.py` to use a two-stage retrieval process:
  1. Base retriever pulls `Top-N` (e.g., N=15) chunks using Vector/Hybrid search.
  2. Cross-Encoder scores and re-ranks them, returning the `Top-K` (e.g., K=5) to the LLM.
- **Custom Re-ranking (Optional extension):**
  - Explore using Gemini as an LLM-based re-ranker (prompting the LLM to score relevance).
  - Investigate Graph-based re-ranking (e.g., using conceptual linkages between chunks).

### 2. Metadata Integration & Filtering
Basic dense retrieval searches over all text equally. Metadata integration allows us to slice and filter the vector space, making searches more precise (e.g., "What was the Net Income shown *in a table* on *page 15*?").

**Core Tasks:**
- **Parser Update (`src/parsers/pdf_parser.py`):** Modify the extraction logic to consistently tag chunks with metadata:
  - `page_number`
  - `content_type` (e.g., 'text', 'table', 'header')
  - `section_title` (if discernible from TOC or headers)
- **Vector Store Update (`src/retrieval/vector_store.py`):** Ensure Qdrant is configured to index these metadata fields as payload arrays.
- **Filtering Logic:** Implement retrieval functions that accept LangChain metadata filters (e.g., filtering Qdrant payloads).
- **UI Update (`app/streamlit_app.py`):** Add sidebar controls allowing the user to explicitly filter by page ranges or content types during their search.

### 3. Sparse & Hybrid Retrieval (Optional)
Dense vectors (Embeddings) are great for semantic meaning but often fail at exact keyword matches (e.g., specific acronyms, serial numbers). Hybrid retrieval combines Dense (Semantic) and Sparse (Keyword) methods.

**Core Tasks:**
- **BM25 Implementation:** Introduce a BM25 sparse retriever (`langchain_community.retrievers.BM25Retriever`) built over the extracted document chunks.
- **Ensemble Retrieval:** Combine the Qdrant dense retriever and the BM25 sparse retriever using LangChain's `EnsembleRetriever`.
- **Score Fusion:** Configure the `EnsembleRetriever` to use Reciprocal Rank Fusion (RRF) to smoothly blend the semantic and keyword scores.
- **UI Update:** Add a "BM25 (Sparse)" and "Hybrid (Dense + Sparse)" option to the retrieval backend radio buttons in the Streamlit app.

---

## Technical Stack Adjustments
- **Sentence-Transformers:** For running local Cross-Encoder re-rankers.
- **Rank_BM25:** For implementing sparse retrieval.

New dependencies to add to `requirements.txt`:
```bash
sentence-transformers>=2.6.0
rank_bm25>=0.2.2
```

---

## Success Metrics for Phase 3
To prove that Phase 3 is successful, we will run the `evaluate.py` script created in Phase 2 comparing the Phase 1 Baseline against the Phase 3 Advanced Retriever.

**Expected Improvements:**
1. **Context Precision** (RAGAS) should significantly increase due to the Cross-Encoder.
2. **Context Recall** (RAGAS) should improve due to pulling a larger `Top-N` before re-ranking.
3. **LLM Judge Completeness** should increase when testing hybrid retrieval with acronyms or specific financial terminology.

---

## File Structure Additions (Proposed)

```text
Rag_project/
├── src/
│   ├── retrieval/
│   │   ├── reranker.py        # New: CrossEncoder logic
│   │   ├── bm25_store.py      # New: Sparse index generation and loading
│   │   └── vector_store.py    # Update: Add metadata indexing
│   └── parsers/
│       └── pdf_parser.py      # Update: Extract page & section metadata
└── app/
    └── streamlit_app.py       # Update: UI filters for metadata & re-ranking toggle
```
