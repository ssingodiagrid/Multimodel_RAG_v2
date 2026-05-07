# Phase 4: Advanced Architectural Optimizations

This phase introduces cutting-edge optimizations to the RAG architecture, focusing on system efficiency, reduced API costs, and the ability to handle highly complex, multi-faceted user queries. 

## 1. Semantic Caching (Efficiency & Cost Reduction)

Traditional exact-match caching fails in NLP because users rarely ask the exact same question word-for-word. **Semantic Caching** embeds the incoming user query and checks the cache for previously answered questions that are *semantically similar*. If a match is found, the system immediately returns the cached answer, bypassing the retriever and LLM entirely.

### Architecture Design:
*   **Vector Backend:** We will leverage our existing **FAISS** or **Qdrant** infrastructure to act as the cache layer.
*   **Thresholding:** Establish a high similarity threshold (e.g., Cosine Similarity > 0.95). 
    *   *If similarity > 0.95:* Return cached answer.
    *   *If similarity < 0.95:* Execute full RAG pipeline and store the new Question-Answer pair in the cache.
*   **Implementation Strategy:** Use `langchain.cache.QdrantSemanticCache` or build a custom cache wrapper in `src/generation/rag_chain.py` that intercepts the query before it hits the LLM.

### Steps to Implement:
- [ ] Create `src/retrieval/semantic_cache.py` to initialize the cache vector store.
- [ ] Wrap the `RAGPipeline.invoke()` method to first query the cache.
- [ ] Add a UI indicator in Streamlit (e.g., "⚡ *Served from Semantic Cache*") to visually verify cache hits.
- [ ] (Optional) Implement Cache Eviction (LRU or TTL) to prevent the cache from growing indefinitely.

---

## 2. Multi-hop Retrieval (Complex Query Handling)

Standard RAG struggles with questions that require reasoning across disparate parts of a document. For example: *"How does the net income of FY24 compare to the projected climate investments mentioned in the CEO's letter?"* This requires retrieving from the financial tables *and* the executive summary.

**Multi-hop Retrieval** (or Query Decomposition) allows the system to perform iterative retrieval passes or break down a complex question into sub-questions.

### Architecture Design:
*   **Query Decomposition (Basic Implementation):** 
    *   Intercept the user's complex query.
    *   Use the LLM (Gemini) to rewrite the query into 2-3 distinct, simpler sub-queries.
    *   Execute retrieval (Hybrid + Reranking) for *each* sub-query independently.
    *   Pool, deduplicate, and pass all retrieved chunks to the LLM for the final answer synthesis.
*   **Iterative Retrieval (Conceptual/Advanced):**
    *   Use an Agentic framework (like **LangGraph**).
    *   The LLM determines if the current retrieved context is sufficient to answer the question. If not, it generates a *new* search query based on what it just learned, looping until it has all the necessary facts.

### Steps to Implement:
- [ ] Create `src/generation/query_transformer.py` using LangChain's `MultiQueryRetriever` or a custom LLM prompt to decompose queries.
- [ ] Update `rag_chain.py` to support a `multi_hop=True` toggle.
- [ ] Integrate the Multi-hop logic with the existing Re-ranker (to ensure the pooled results from multiple queries don't exceed token limits).
- [ ] Add a visual breakdown in the Streamlit UI showing the user the exact sub-queries the LLM generated.
