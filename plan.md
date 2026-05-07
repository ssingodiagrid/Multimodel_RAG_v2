# Architecture and Reverse Engineering Report: Multimodal RAG System

Based on the reverse engineering of the `rag_system` project, here is a detailed breakdown of its architecture, development phases, and the technology stack used.

## 1. Detailed System Architecture

The project is a production-grade, multimodal Retrieval-Augmented Generation (RAG) system specifically built to ingest, process, and query the **IFC Annual Report 2024**. The architecture is heavily modularized, integrating modern LLM techniques and containerized for deployment.

### Core Architecture Components:
- **Presentation / UI Layer**: A responsive web application built with **Streamlit** (`app/streamlit_app.py`). It provides a phase-selector sidebar, streaming response generation, source citation cards, and visual tabs for performance benchmarking (FAISS vs Qdrant) and RAGAS evaluation.
- **Orchestration Layer**: **LangChain** is used to chain together prompts, retrievers, and the LLM via LCEL (LangChain Expression Language). The main pipeline is in `src/generation/rag_chain.py`.
- **LLM & Embedding Engine**: Powered exclusively by **Google Vertex AI**. It utilizes `gemini-2.0-flash` for multimodal text/image generation and `text-embedding-004` for creating dense vectors.
- **Storage & Retrieval Layer**:
  - **FAISS**: In-memory flat index vector store used for fast, naive text retrieval and semantic caching.
  - **Qdrant**: Disk-persisted HNSW vector store, deployed via Docker. Used to handle high-dimensional vector storage for tables and images which demand scalable metadata filtering and persistence.
- **Data Ingestion Pipeline**: Custom parsers (`src/parsers/`) split the source PDF into logical multimodal chunks (text blocks, table structures, and images).
- **Observability Layer**: **Langfuse** is integrated for comprehensive request tracing, pipeline latency tracking, and monitoring generation metrics.

---

## 2. Breakdown of Building Phases & Tech Stack

The system was constructed sequentially across 6 distinct phases, increasing in complexity from basic text retrieval to advanced multimodal integration.

### Phase 1: Text RAG (Naive)
- **Goal**: Establish the foundational text-based pipeline to extract and query standard paragraphs.
- **Tech Stack**: 
  - **PyMuPDF (`fitz`)**: For fast, accurate text extraction from the PDF.
  - **FAISS & Qdrant**: For vector indexing of the text chunks.
  - **Google Gemini & text-embedding-004**: For generating text embeddings and base text generation.
  - **LangChain**: Connecting the retriever to the LLM prompt.

### Phase 2: Evaluation
- **Goal**: Implement metrics to objectively measure the quality of RAG answers to prevent hallucination.
- **Tech Stack**: 
  - **RAGAS**: Computes context precision, recall, faithfulness, and answer relevancy.
  - **LLM-as-a-Judge**: Custom evaluation scripts (`llm_judge.py`) using Gemini to score the outputs based on correctness, completeness, and clarity.
  - **Datasets, Pandas, Numpy**: Used for dataset manipulation and evaluation scoring operations.

### Phase 3: Hybrid Search + Re-ranking
- **Goal**: Improve retrieval accuracy by combining keyword search (sparse) with semantic search (dense) and re-ranking the combined results.
- **Tech Stack**: 
  - **BM25 (`rank-bm25`)**: Used for the sparse retrieval/keyword matching.
  - **Cross-Encoder Re-ranker (`sentence-transformers`, `torch`)**: Evaluates query-document pairs to produce highly accurate final ranking scores.

### Phase 4: Advanced RAG (Semantic Caching)
- **Goal**: Reduce latency and API costs by caching answers to semantically identical or highly similar queries.
- **Tech Stack**: 
  - **FAISS (In-memory)**: Used as a lightweight vector cache. Incoming queries are embedded and compared to previous queries; if the similarity crosses a threshold, the cached answer is served instantly.

### Phase 5.1: Multimodal – Tables
- **Goal**: Accurately parse tabular financial data which typical text extraction tools mangle.
- **Tech Stack**: 
  - **pdfplumber**: Specialized library to identify table bounding boxes and extract row/column structures cleanly.
  - **Gemini Structured Output**: Uses the LLM to summarize and format the tabular data into searchable text representations before vectorization. Stored in Qdrant.

### Phase 5.2: Multimodal – Images
- **Goal**: Allow the system to query visual elements, such as charts and graphs present in the annual report.
- **Tech Stack**: 
  - **PyMuPDF**: For image extraction from PDF pages.
  - **Pillow (PIL)**: For image processing.
  - **Gemini 2.0 Flash**: Acts as a Vision LLM to generate descriptive captions of the charts/images. These text captions are then embedded and stored in Qdrant.

### Phase 6: ColPali-Inspired Late Interaction Retrieval
- **Goal**: Implement a cutting-edge visual retrieval approach that treats documents entirely as visual patches rather than parsed text.
- **Tech Stack**: 
  - **PaliGemma Concepts / Multi-modal Embedding**: Dividing PDF pages into 4x4 image patches, embedding these patches, and performing retrieval via **MaxSim** (Maximum Similarity).
  - **Qdrant**: To store the high density of image patch embeddings.
  - **Gemini**: Fed the highest-scoring raw images directly for visual question answering.

---

## 3. Infrastructure & Deployment
- **Docker & Docker Compose**: The entire project is containerized for reproducibility. `docker-compose.yml` spins up the `qdrant` vector database alongside the `rag-app` Streamlit container.
- **Environment Management**: `python-dotenv`, `pydantic-settings` are used for robust typed configuration, reading variables such as `GCP_PROJECT`, `QDRANT_URL`, and Langfuse keys from `.env`.
- **GCP Service Account Auth**: A `gcp-key.json` is mounted into the Docker container to securely authenticate the Vertex AI API calls.
