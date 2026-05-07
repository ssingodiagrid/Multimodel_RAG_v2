"""
ragas_eval.py
Runs the RAGAS evaluation suite (faithfulness, answer_relevancy,
context_precision, context_recall) using Gemini + VertexAI embeddings.
"""

import sys
import logging
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

logger = logging.getLogger(__name__)

OUTPUT_PATH = project_root / "data" / "evaluation" / "ragas_results.csv"


def run_ragas_evaluation(answers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run RAGAS metrics on the generated answers DataFrame.

    Required columns in answers_df:
        question, answer, contexts (List[str]), ground_truth

    Returns a DataFrame with per-question metric scores plus aggregate means.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
    from configs.settings import settings

    logger.info("Initializing Vertex AI LLM and embeddings for RAGAS...")
    llm = ChatVertexAI(
        model_name=settings.gemini_model,
        project=settings.gcp_project,
        location=settings.gcp_location,
        temperature=0.0,
    )
    embeddings = VertexAIEmbeddings(
        model_name=settings.embedding_model,
        project=settings.gcp_project,
        location=settings.gcp_location,
    )

    # RAGAS needs exactly these 4 columns
    ragas_df = answers_df[["question", "answer", "contexts", "ground_truth"]].copy()

    logger.info(f"Running RAGAS on {len(ragas_df)} samples...")
    dataset = Dataset.from_pandas(ragas_df)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    scores_df = result.to_pandas()

    # Merge metadata back in
    for col in ["context_content_type", "page_number"]:
        if col in answers_df.columns:
            scores_df[col] = answers_df[col].values

    # Save to CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"RAGAS results saved to {OUTPUT_PATH}")

    return scores_df


def summarize_ragas(scores_df: pd.DataFrame) -> pd.Series:
    """Return mean scores across all metrics."""
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    available = [m for m in metrics if m in scores_df.columns]
    return scores_df[available].mean()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    existing = project_root / "data" / "evaluation" / "ragas_results.csv"
    if existing.exists():
        df = pd.read_csv(existing)
        print(summarize_ragas(df))
    else:
        print("No results yet. Run evaluate.py first.")
