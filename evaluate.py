"""
evaluate.py — Master evaluation runner for Phase 2.

Usage:
    python evaluate.py [--skip-ragas] [--skip-judge] [--top-k 5]

Options:
    --skip-ragas    Skip the RAGAS metric suite (faster)
    --skip-judge    Skip the LLM-as-a-Judge scoring
    --top-k N       Number of chunks to retrieve per question (default: 5)
"""

import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 RAG Evaluation Runner")
    parser.add_argument("--skip-ragas", action="store_true", help="Skip RAGAS evaluation")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-a-Judge")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve per question")
    args = parser.parse_args()

    eval_pdf = project_root / "RAG_evaluation_dataset - convertcsv (2).pdf"
    if not eval_pdf.exists():
        logger.error(f"Evaluation PDF not found at {eval_pdf}")
        sys.exit(1)

    # ── Step 1: Load dataset ───────────────────────────────────────────────────
    print_section("Step 1 / 4 — Loading evaluation dataset")
    from src.evaluation.dataset_loader import load_eval_dataset
    eval_df = load_eval_dataset(str(eval_pdf))
    print(f"  ✔ {len(eval_df)} questions loaded")
    print(f"  Content types: {eval_df['context_content_type'].value_counts().to_dict()}")

    # ── Step 2: Generate answers ───────────────────────────────────────────────
    print_section("Step 2 / 4 — Generating answers via RAG pipeline")
    from src.evaluation.answer_generator import generate_answers
    answers_df = generate_answers(eval_df, top_k=args.top_k)

    answers_path = project_root / "data" / "evaluation" / "generated_answers.csv"
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    # Save a version without list columns for easy inspection
    save_df = answers_df.copy()
    save_df["contexts"] = save_df["contexts"].apply(lambda x: " ||| ".join(x) if isinstance(x, list) else x)
    save_df.to_csv(answers_path, index=False)
    print(f"  ✔ Answers saved to {answers_path}")

    # ── Step 3: RAGAS ─────────────────────────────────────────────────────────
    if not args.skip_ragas:
        print_section("Step 3 / 4 — Running RAGAS evaluation")
        from src.evaluation.ragas_eval import run_ragas_evaluation, summarize_ragas
        ragas_df = run_ragas_evaluation(answers_df)
        summary = summarize_ragas(ragas_df)
        print("\n  RAGAS Summary (mean scores):")
        for metric, score in summary.items():
            print(f"    {metric:<25} {score:.4f}")
    else:
        print_section("Step 3 / 4 — RAGAS (skipped)")
        ragas_df = None

    # ── Step 4: LLM-as-a-Judge ────────────────────────────────────────────────
    if not args.skip_judge:
        print_section("Step 4 / 4 — LLM-as-a-Judge scoring")
        from src.evaluation.llm_judge import run_llm_judge, summarize_judge
        judge_df = run_llm_judge(answers_df)
        summary = summarize_judge(judge_df)
        print("\n  LLM Judge Summary (mean scores out of 5):")
        for metric, score in summary.items():
            print(f"    {metric:<25} {score:.2f}")
    else:
        print_section("Step 4 / 4 — LLM-as-a-Judge (skipped)")
        judge_df = None

    # ── Final Summary ─────────────────────────────────────────────────────────
    print_section("Evaluation Complete")
    print(f"  Output directory: {project_root / 'data' / 'evaluation'}")
    print("  Files:")
    for f in (project_root / "data" / "evaluation").glob("*.csv"):
        print(f"    ✔ {f.name}")
    print("\nTo view results in the UI, run:")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
