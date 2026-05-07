# Phase 5: Multimodal RAG (Tables & Images)

This phase upgrades the RAG system from a pure text processor to a fully Multimodal understanding engine, capable of accurately parsing, indexing, and reasoning over complex financial tables and visual charts present in the IFC Annual Report.

## Phase 5.1: Incorporating Tables

Traditional PDF parsers often destroy the row/column relationship of tabular data. We will implement a structured pipeline to retain this critical financial data.

### 1. Table Extraction & Representation
- **Tooling:** Use `pdfplumber` to accurately identify table bounding boxes and extract clean row/column structures.
- **LLM Summarization:** Pass the raw extracted table to Gemini using a Structured Output prompt to generate a highly searchable, textual summary of the table's contents (e.g., "Table showing Net Income for FY24: $1.5B...").
- **Indexing:** Embed this textual summary and store it in Qdrant with `content_type="table"` metadata.

### 2. Querying Tabular Data
- **Prompt Engineering:** Develop specific instructions for the RAG LLM to properly interpret tabular data retrieved from the vector store, allowing it to perform simple comparative calculations or direct numerical extraction.

### 3. Integrated Retrieval
- Modify the existing retrieval pipeline to search simultaneously across standard text chunks and table summary representations.

### (Optional) Plotting Functionality
- Implement a capability allowing the LLM to generate Python/matplotlib code to dynamically plot the retrieved table data.

---

## Phase 5.2: Incorporating Images (Charts & Graphs)

Financial reports contain critical visual data that is completely lost in standard text extraction. We will utilize Vision LLMs to unlock this information.

### 1. Image Extraction & Understanding
- **Tooling:** Use PyMuPDF (`fitz`) and `Pillow (PIL)` to scan PDF pages, identify image objects (charts, graphs, infographics), and extract them as raw image files.
- **Vision LLM Captioning:** Feed these extracted images directly to `gemini-2.0-flash` (acting as a Vision model) to generate rich, highly descriptive textual captions of the visual data.

### 2. Querying Visual Data
- **Indexing:** Embed the Gemini-generated image captions and store them in Qdrant alongside standard text chunks, categorized with `content_type="image"`.
- **Retrieval:** Adapt the retriever to pull these visual descriptions when users ask questions about trends or charts.
- **Synthesis:** Develop prompts that allow the main generation LLM to seamlessly weave image-derived facts into its final answers.
