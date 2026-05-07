# IFC Annual Report Multimodal RAG

A modular Retrieval-Augmented Generation (RAG) project for querying the **IFC Annual Report 2024** with text, tables, images, and an experimental Phase 6 visual-document pipeline.

## Overview

This project evolves step by step from a basic text RAG system into a richer multimodal pipeline:

- **Phase 1**: Text RAG with PDF parsing, chunking, embeddings, and answer generation
- **Phase 2**: Evaluation with RAGAS and LLM-as-a-Judge
- **Phase 3**: Hybrid retrieval with BM25, dense search, metadata filters, and reranking
- **Phase 4**: Semantic caching and multi-hop retrieval
- **Phase 5**: Multimodal support for tables and images
- **Phase 6**: Early ColPali-like visual-document preprocessing with page rendering and patch extraction

The current app uses **Streamlit** for the UI, **Google Vertex AI / Gemini** for generation and embeddings, and **FAISS + Qdrant + BM25** for retrieval.

## Project Structure

```text
Rag_project/
├── app/
│   └── streamlit_app.py
├── configs/
│   └── settings.py
├── data/
│   ├── evaluation/
│   ├── indices/
│   ├── processed/
│   └── qdrant_storage/
├── src/
│   ├── embeddings/
│   ├── evaluation/
│   ├── generation/
│   ├── parsers/
│   └── retrieval/
├── tests/
├── ingest.py
├── ingest_phase6.py
├── evaluate.py
├── phase1.md ... phase6.md
└── requirements.txt
```

## Core Features

- PDF parsing with **PyMuPDF** and **pdfplumber**
- Dense vector search with **FAISS** and **Qdrant**
- Sparse retrieval with **BM25**
- Optional reranking with a cross-encoder
- Streamlit chat UI with retrieval controls and chunk inspection
- Table summarization and image captioning for multimodal retrieval
- Session-scoped semantic cache
- Evaluation dashboard with RAGAS and judge-based scoring
- Phase 6 visual preprocessing:
  - render pages to images
  - split pages into fixed-grid patches
  - persist page/patch manifests for future multimodal retrieval

## Requirements

- Python 3.10+
- A Google Cloud project with Vertex AI enabled
- A service account key for Vertex AI access
- Optional: Docker, if you want to run Qdrant as a server

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GCP_PROJECT=your-gcp-project
GCP_LOCATION=us-central1
GEMINI_MODEL=gemini-2.0-flash-001
EMBEDDING_MODEL=text-embedding-004
GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json
QDRANT_URL=http://localhost:6333
QDRANT_MODE=server
TEXT_COLLECTION_NAME=ifc_text
```

Place your Google service account key at `gcp-key.json`, or update `GOOGLE_APPLICATION_CREDENTIALS` to point to your key file.

## Running Qdrant

If you want to use Qdrant in server mode:

```bash
docker compose up -d
```

If you prefer local-file Qdrant mode, set:

```env
QDRANT_MODE=local
```

## Ingestion

### Standard RAG ingestion

Build the text/table/image retrieval assets:

```bash
python ingest.py
```

This creates:

- FAISS index
- Qdrant collection
- BM25 document store

### Phase 6 visual preprocessing

Build the page-image and patch manifest scaffold:

```bash
python ingest_phase6.py
```

This writes rendered pages, patch crops, and a manifest under:

```text
data/processed/phase6_visual/
```

## Running the App

Start the Streamlit interface:

```bash
streamlit run app/streamlit_app.py
```

### Current UI capabilities

- Choose retrieval backend:
  - FAISS
  - Qdrant
  - BM25
  - Hybrid (Qdrant + BM25)
- Enable:
  - cross-encoder reranking
  - multi-hop retrieval
- Apply metadata filters:
  - page range
  - content type (`text`, `table`, `image`)
- Inspect retrieved chunks used for the final answer

## Evaluation

Run the evaluation pipeline:

```bash
python evaluate.py
```

Optional flags:

```bash
python evaluate.py --skip-ragas
python evaluate.py --skip-judge
python evaluate.py --top-k 8
```

Generated files are written to:

```text
data/evaluation/
```

## Phase Documents

The implementation roadmap for each project stage is documented in:

- [phase1.md](./phase1.md)
- [phase2.md](./phase2.md)
- [phase3.md](./phase3.md)
- [phase4.md](./phase4.md)
- [phase5.md](./phase5.md)
- [phase6.md](./phase6.md)

## Important Notes

- `.env` and `gcp-key.json` should never be committed.
- `data/indices/`, `data/qdrant_storage/`, `data/processed/`, and local caches are machine-specific artifacts and are ignored by default.
- The project currently includes an early Phase 6 scaffold for visual document patches, but not yet a full ColPali-style multimodal retrieval stack.

## Next Steps

Recommended next implementation targets for Phase 6:

1. Add multimodal patch embeddings
2. Store patch vectors and metadata in Qdrant
3. Implement late interaction / MaxSim retrieval
4. Pass retrieved visual evidence to a multimodal generator
5. Add patch-level source attribution in the UI
# Multimodel_RAG_v2
