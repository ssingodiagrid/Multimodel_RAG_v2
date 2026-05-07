"""
llm_judge.py
Uses Gemini as an LLM-as-a-Judge to score generated answers on:
  - correctness  (1-5): factual match with ground truth
  - completeness (1-5): covers all important aspects of ground truth
  - clarity      (1-5): well-written, easy to understand
"""

import sys
import json
import logging
import re
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

logger = logging.getLogger(__name__)

OUTPUT_PATH = project_root / "data" / "evaluation" / "llm_judge_results.csv"

JUDGE_PROMPT = """You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system.

Your task is to score the Generated Answer against the Ground Truth Answer for the given Question.

Score each of the following dimensions on a scale of 1 to 5:
- correctness: Does the generated answer match the ground truth factually? (1=completely wrong, 5=perfectly correct)
- completeness: Does the generated answer cover all important aspects of the ground truth? (1=missing everything, 5=fully covers it)
- clarity: Is the generated answer well-written and easy to understand? (1=confusing, 5=very clear)

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {generated_answer}

Respond ONLY with a valid JSON object with exactly these keys: correctness, completeness, clarity.
Example: {{"correctness": 4, "completeness": 3, "clarity": 5}}
"""


def _parse_scores(response_text: str) -> dict:
    """Extract JSON scores from LLM response, with fallback."""
    try:
        # Try direct JSON parse
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    # Fallback: find JSON block in response
    match = re.search(r'\{[^{}]+\}', response_text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    logger.warning(f"Could not parse scores from response: {response_text[:200]}")
    return {"correctness": None, "completeness": None, "clarity": None}


def run_llm_judge(answers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score each generated answer with Gemini as judge.

    Required columns: question, answer, ground_truth
    Optional: context_content_type, page_number

    Returns a DataFrame with scores per question.
    """
    from langchain_google_vertexai import ChatVertexAI
    from langchain_core.messages import HumanMessage
    from configs.settings import settings

    llm = ChatVertexAI(
        model_name=settings.gemini_model,
        project=settings.gcp_project,
        location=settings.gcp_location,
        temperature=0.0,
    )

    records = []
    total = len(answers_df)

    for idx, row in answers_df.iterrows():
        logger.info(f"[{idx+1}/{total}] Judging: {str(row['question'])[:70]}...")
        prompt = JUDGE_PROMPT.format(
            question=row["question"],
            ground_truth=row["ground_truth"],
            generated_answer=row["answer"],
        )
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            scores = _parse_scores(response.content)
        except Exception as e:
            logger.error(f"LLM judge failed on row {idx}: {e}")
            scores = {"correctness": None, "completeness": None, "clarity": None}

        record = {
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "generated_answer": row["answer"],
            **scores,
        }
        # Preserve metadata
        for col in ["context_content_type", "page_number"]:
            if col in row.index:
                record[col] = row[col]

        records.append(record)

    result_df = pd.DataFrame(records)

    # Compute average score per row (only for numeric columns)
    score_cols = ["correctness", "completeness", "clarity"]
    result_df["average_score"] = result_df[score_cols].mean(axis=1)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"LLM Judge results saved to {OUTPUT_PATH}")

    return result_df


def summarize_judge(judge_df: pd.DataFrame) -> pd.Series:
    """Return mean scores across correctness, completeness, clarity."""
    cols = ["correctness", "completeness", "clarity", "average_score"]
    available = [c for c in cols if c in judge_df.columns]
    return judge_df[available].mean()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    existing = project_root / "data" / "evaluation" / "llm_judge_results.csv"
    if existing.exists():
        df = pd.read_csv(existing)
        print(summarize_judge(df))
    else:
        print("No results yet. Run evaluate.py first.")
