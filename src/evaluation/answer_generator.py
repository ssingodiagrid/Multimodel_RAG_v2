"""
answer_generator.py
Runs each eval question through the Phase 1 RAG pipeline and collects
answers + retrieved context chunks. Output is RAGAS-compatible.
"""

import sys
import logging
from pathlib import Path
from typing import Any
import pandas as pd

# Allow imports from project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.retrieval.vector_store import VectorStoreManager, get_qdrant_client
from src.generation.rag_chain import RAGPipeline

logger = logging.getLogger(__name__)


def _build_pipeline(use_faiss: bool = True, top_k: int = 5) -> tuple[Any, Any]:
    """Load vector store and build the RAG pipeline. Returns (pipeline, retriever)."""
    vm = VectorStoreManager()
    if use_faiss:
        store = vm.load_faiss_store()
    else:
        client = get_qdrant_client()
        store = vm.load_qdrant_store(client=client)
    retriever = store.as_retriever(search_kwargs={"k": top_k})
    pipeline = RAGPipeline(retriever)
    return pipeline, retriever


def generate_answers(eval_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    For each question in eval_df, run the RAG pipeline and collect:
      - answer: the generated answer string
      - contexts: list of retrieved chunk texts (for RAGAS)
      - ground_truth: the expected answer (for RAGAS)

    Returns a RAGAS-compatible DataFrame.
    """
    logger.info("Building RAG pipeline for answer generation...")
    pipeline, retriever = _build_pipeline(use_faiss=True, top_k=top_k)

    questions = eval_df["question"].tolist()
    ground_truths = eval_df["ground_truth_answer"].tolist()

    answers: list[str] = []
    contexts: list[list[str]] = []

    for idx, question in enumerate(questions):
        logger.info(f"[{idx+1}/{len(questions)}] Answering: {question[:80]}...")
        try:
            # Retrieve context chunks
            docs = retriever.invoke(question)
            chunk_texts = [doc.page_content for doc in docs]

            # Generate answer
            answer = pipeline.generate(question)

            answers.append(answer)
            contexts.append(chunk_texts)
        except Exception as e:
            logger.error(f"Failed on question {idx}: {e}")
            answers.append("")
            contexts.append([])

    result_df = pd.DataFrame({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
        # Preserve metadata for filtering in Streamlit
        "context_content_type": eval_df["context_content_type"].tolist(),
        "page_number": eval_df["page_number"].tolist(),
    })

    logger.info("Answer generation complete.")
    return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.evaluation.dataset_loader import load_eval_dataset
    eval_pdf = project_root / "RAG_evaluation_dataset - convertcsv (2).pdf"
    df = load_eval_dataset(str(eval_pdf))
    results = generate_answers(df, top_k=5)
    print(results[["question", "answer"]].head(3).to_string())
