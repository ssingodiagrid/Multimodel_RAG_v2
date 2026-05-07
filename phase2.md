# Phase 2: Evaluation — Detailed Implementation Plan

## Objective
Implement a robust, multi-layered evaluation framework to objectively measure the quality of the Phase 1 RAG pipeline answers. This phase prevents hallucination and establishes a measurable quality baseline using two distinct approaches: **RAGAS** (automated framework metrics) and **LLM-as-a-Judge** (Gemini-powered scoring).

## Available Evaluation Dataset
The project already contains a pre-built evaluation dataset:
- **File:** `RAG_evaluation_dataset - convertcsv (2).pdf`
- **Fields per row:**
  | Column | Description |
  |--------|-------------|
  | `Question` | The user query |
  | `Ground_Truth_Context` | The expected source passage from the IFC report |
  | `Ground_Truth_Answer` | The expected correct answer |
  | `Page_Number` | Source page in the PDF |
  | `Context_Content_Type` | `text`, `table`, `image`, or `combination` |

This dataset will serve as ground truth for all evaluations.

## Tech Stack
- **RAGAS** — Automated RAG metrics: context precision, recall, faithfulness, answer relevancy
- **Gemini 2.0 Flash (Vertex AI)** — LLM-as-a-Judge for scoring outputs on correctness, completeness, and clarity
- **Pandas** — Dataset loading, manipulation, and scoring output tabulation
- **Datasets (HuggingFace)** — RAGAS requires datasets in HuggingFace `Dataset` format

---

## Task Breakdown

### 1. Install New Dependencies
Add the following packages to `requirements.txt` and install them:
- `ragas>=0.1.7`
- `datasets>=2.18.0`
- `pandas>=2.0.0`

```bash
pip install ragas datasets pandas
```

---

### 2. Dataset Loader — `src/evaluation/dataset_loader.py`
Parse the evaluation PDF into a structured Pandas DataFrame.

**Steps:**
- Use `PyMuPDF` (`fitz`) to extract the raw text from each page.
- Parse the structured rows into a `pandas.DataFrame` with columns: `question`, `ground_truth_context`, `ground_truth_answer`, `page_number`, `context_content_type`.
- Export a helper function `load_eval_dataset(pdf_path: str) -> pd.DataFrame` to be used by both evaluation scripts.

**Output schema:**
```python
{
  "question": str,
  "ground_truth_context": str,
  "ground_truth_answer": str,    # aka "ground_truth" for RAGAS
  "page_number": int,
  "context_content_type": str    # "text", "table", "image", "combination"
}
```

---

### 3. Pipeline Answer Generator — `src/evaluation/answer_generator.py`
For each question in the dataset, run it through the Phase 1 RAG pipeline and collect the results.

**Steps:**
- Load the FAISS vector store via `VectorStoreManager`.
- Initialize the `RAGPipeline`.
- For each row in the evaluation dataset, call `pipeline.generate(question)` to get the answer **and** retrieve the top-k context chunks.
- Store results in a DataFrame with columns: `question`, `answer`, `contexts` (list of retrieved chunks), `ground_truth`.

**Output schema** (RAGAS-compatible):
```python
{
  "question": str,
  "answer": str,           # generated answer
  "contexts": List[str],   # retrieved chunks used as context
  "ground_truth": str      # from Ground_Truth_Answer column
}
```

---

### 4. RAGAS Evaluation — `src/evaluation/ragas_eval.py`
Run the standard RAGAS evaluation suite on the generated answers.

**Metrics to compute:**
| Metric | What it measures |
|--------|-----------------|
| `faithfulness` | Is the answer grounded in the retrieved context? (hallucination check) |
| `answer_relevancy` | Is the answer relevant to the question asked? |
| `context_precision` | Are the retrieved chunks relevant to the question? |
| `context_recall` | Does the context cover the ground truth? |

**Steps:**
1. Convert the generated answers DataFrame into a HuggingFace `Dataset` object.
2. Configure RAGAS to use `ChatVertexAI` (Gemini) as the evaluation LLM and `VertexAIEmbeddings` as the embedding model (so no OpenAI key is needed).
3. Call `evaluate(dataset, metrics=[...])` to run the suite.
4. Save the resulting scores to `data/evaluation/ragas_results.csv`.

**Code structure:**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

def run_ragas_evaluation(eval_df: pd.DataFrame) -> pd.DataFrame:
    dataset = Dataset.from_pandas(eval_df)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=vertex_llm,
        embeddings=vertex_embeddings,
    )
    return result.to_pandas()
```

---

### 5. LLM-as-a-Judge — `src/evaluation/llm_judge.py`
Use Gemini directly as an evaluator to score generated answers against the ground truth.

**Scoring dimensions (each scored 1–5):**
| Dimension | Description |
|-----------|-------------|
| `correctness` | Does the answer match the ground truth answer factually? |
| `completeness` | Does the answer cover all important aspects of the ground truth? |
| `clarity` | Is the answer well-written and easy to understand? |

**Steps:**
1. For each row in the evaluation dataset, construct a structured prompt:
   ```
   You are an expert evaluator for a RAG system...
   Question: {question}
   Ground Truth: {ground_truth_answer}
   Generated Answer: {generated_answer}
   Score correctness, completeness, and clarity each from 1 to 5.
   Return a JSON like: {"correctness": X, "completeness": Y, "clarity": Z}
   ```
2. Call Gemini via `ChatVertexAI` and parse the JSON response.
3. Aggregate scores and save to `data/evaluation/llm_judge_results.csv`.

**Output schema:**
```python
{
  "question": str,
  "correctness": int,      # 1-5
  "completeness": int,     # 1-5
  "clarity": int,          # 1-5
  "average_score": float   # mean of the 3 dimensions
}
```

---

### 6. Master Evaluation Runner — `evaluate.py`
Create a single CLI script at the project root to run the entire evaluation pipeline in sequence.

**Steps:**
1. Load dataset → `dataset_loader.py`
2. Generate answers → `answer_generator.py`
3. Run RAGAS → `ragas_eval.py`
4. Run LLM-as-a-Judge → `llm_judge.py`
5. Print a combined summary table to console.
6. Save all results to `data/evaluation/`.

**Usage:**
```bash
python evaluate.py
```

---

### 7. Streamlit Dashboard Integration — `app/streamlit_app.py`
Add a new **"📊 Evaluation"** tab to the existing Streamlit UI to visualize results.

**Tab contents:**
- **RAGAS Metrics**: A bar chart showing average `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` scores.
- **LLM Judge Scores**: A grouped bar chart showing `correctness`, `completeness`, and `clarity` distributions.
- **Per-question Breakdown**: A searchable data table showing scores for every question in the eval dataset.
- **Content Type Filter**: Filter results by `context_content_type` (text/table/image/combination) to identify which content types the pipeline struggles with.

---

## File Structure After Phase 2

```
Rag_project/
├── src/
│   └── evaluation/
│       ├── __init__.py
│       ├── dataset_loader.py      # Parses eval PDF into DataFrame
│       ├── answer_generator.py    # Runs questions through the RAG pipeline
│       ├── ragas_eval.py          # RAGAS metric computation
│       └── llm_judge.py           # Gemini-based scoring
├── data/
│   └── evaluation/
│       ├── ragas_results.csv      # RAGAS output scores
│       └── llm_judge_results.csv  # LLM judge output scores
└── evaluate.py                    # Master runner script
```

---

## Verification Plan

| Step | Command | Expected Output |
|------|---------|----------------|
| Install deps | `pip install ragas datasets pandas` | No errors |
| Run evaluations | `python evaluate.py` | Prints summary scores table |
| Check RAGAS results | `cat data/evaluation/ragas_results.csv` | CSV with 4 metric columns |
| Check judge results | `cat data/evaluation/llm_judge_results.csv` | CSV with correctness/completeness/clarity |
| View in UI | `streamlit run app/streamlit_app.py` | Evaluation tab visible with charts |
