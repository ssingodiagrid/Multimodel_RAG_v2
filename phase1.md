# Phase 1: Text RAG (Naive) - Detailed Breakdown

## Objective
Establish the foundational text-based pipeline to extract, index, and query standard paragraphs from the IFC Annual Report 2024 using a basic RAG architecture.

## Tech Stack
- **PDF Processing:** PyMuPDF (`fitz`)
- **Embeddings:** Google Vertex AI (`text-embedding-004`)
- **Vector Stores:** FAISS (in-memory) & Qdrant
- **LLM:** Google Gemini 2.0 Flash
- **Orchestration:** LangChain (LCEL)
- **Configuration:** Pydantic `BaseSettings`
- **UI:** Streamlit

---

## Task Breakdown

### 1. Project Initialization & Configuration
- **1.1 Directory Structure Setup:** 
  Create the core directories: `app/`, `configs/`, `data/raw/`, `data/processed/`, `src/parsers/`, `src/embeddings/`, `src/retrieval/`, and `src/generation/`.
- **1.2 Environment Management:**
  Create a `.env` file and set up GCP authentication using `gcp-key.json`.
- **1.3 Settings Configuration:**
  Implement `configs/settings.py` using Pydantic `BaseSettings` to manage environment variables like `GCP_PROJECT`, `GEMINI_MODEL`, `EMBEDDING_MODEL`, and vector store paths type-safely.

### 2. Document Parsing & Text Extraction
- **2.1 PDF Text Extractor:**
  Implement a parser using PyMuPDF (`fitz`) in `src/parsers/` to read the raw PDF and cleanly extract standard text paragraphs.
- **2.2 Text Chunking:**
  Implement a chunking strategy (e.g., using LangChain's `RecursiveCharacterTextSplitter`) to split the extracted text into manageable chunks suitable for embedding.

### 3. Embeddings & Vector Indexing
- **3.1 Embedding Client:**
  Configure the Google Vertex AI embedding client in `src/embeddings/` to use `text-embedding-004`.
- **3.2 Vector Store Initialization:**
  Set up FAISS for fast in-memory indexing and Qdrant for persistent vector storage in `src/retrieval/`.
- **3.3 Master Ingestion Script:**
  Create `ingest.py` to orchestrate the pipeline: read PDF -> extract text -> chunk -> generate embeddings -> index into FAISS and Qdrant.

### 4. Generation Pipeline & Orchestration
- **4.1 LLM Client:**
  Initialize the Google Gemini client in `src/generation/` for text generation.
- **4.2 LangChain LCEL Setup:**
  Create the `RAGPipeline` class in `src/generation/rag_chain.py`. Use LangChain Expression Language (LCEL) to build the retrieval chain: `Retrieve (FAISS/Qdrant) -> Prompt Template -> Generate (Gemini)`.

### 5. Basic User Interface
- **5.1 Streamlit App:**
  Create a foundational UI in `app/streamlit_app.py` allowing users to input queries.
- **5.2 Pipeline Integration:**
  Connect the Streamlit UI to the LangChain LCEL pipeline to display the generated answers and the text chunks retrieved as context, with streaming output (token-by-token via LangChain `.stream()`).

---

### 6. UI Enhancement — Vector Store Switcher Sidebar *(Added)*
- **6.1 Sidebar Panel:**  
  Added a persistent left sidebar (`st.sidebar`) to the Streamlit app with the following controls:
  - **Vector Store Radio Button:** Three options — `FAISS`, `Qdrant`, `Both (Merged)`.
  - **Top-K Slider:** Lets the user control how many chunks (1–15) are retrieved per query.
  - **Clear Chat Button:** Wipes the chat history and reruns the app.

- **6.2 Dynamic Pipeline Reloading:**  
  The pipeline is stored in `st.session_state`. A `pipeline_key` (combination of store choice and top-k value) is used to detect changes. When the user switches stores or adjusts top-k, the retriever and pipeline are rebuilt automatically without restarting the app.

- **6.3 Both (Merged) Mode — LangChain `MergerRetriever`:**  
  When `Both (Merged)` is selected, both the FAISS and Qdrant retrievers are initialized and combined using LangChain's `MergerRetriever`. This merges the ranked result lists from both stores before passing context to Gemini.

- **6.4 Active Store Badge:**  
  The main UI shows a live badge (🟦 FAISS / 🟩 Qdrant / 🟪 Both) and the current top-k value so the user always knows which retriever is active.

**Files modified:** `app/streamlit_app.py`
