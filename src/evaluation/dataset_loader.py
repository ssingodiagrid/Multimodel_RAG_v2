"""
dataset_loader.py
Parses the RAG evaluation PDF into a structured pandas DataFrame.

PDF structure (single page, each record is 5 consecutive lines):
  Line 1: Question text
  Line 2: Ground_Truth_Context
  Line 3: Ground_Truth_Answer
  Line 4: "<page_number> <content_type>"   e.g. "4 table"
  (no explicit separator — next record starts immediately)
"""

import fitz  # PyMuPDF
import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)

_HEADERS = {"Question", "Ground_Truth_Context", "Ground_Truth_Answer", "Page_Number", "Context_Content_Type"}
_CONTENT_TYPES = {"text", "table", "image", "combination"}


def load_eval_dataset(pdf_path: str) -> pd.DataFrame:
    """Parse the evaluation PDF and return a clean DataFrame.

    Returns a DataFrame with columns:
        question, ground_truth_context, ground_truth_answer,
        page_number, context_content_type
    """
    logger.info(f"Loading evaluation dataset from: {pdf_path}")
    doc = fitz.open(pdf_path)
    raw_text = "\n".join(page.get_text() for page in doc)

    # Split into non-empty lines, skip the 5 header lines
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    # Drop the column header lines at the top
    start = 0
    for i, line in enumerate(lines):
        if line == "Question":
            start = i + 1
            break
    # Skip remaining header names right after "Question"
    while start < len(lines) and lines[start] in _HEADERS:
        start += 1
    lines = lines[start:]

    records = []
    i = 0
    while i < len(lines):
        # The last field of each record looks like "<number> <content_type>"
        # We use this as a delimiter: collect lines until we hit one matching that pattern,
        # then group everything up to that point into a 3-field record.
        
        # Accumulate lines for this record
        block: list[str] = []
        while i < len(lines):
            line = lines[i]
            # Check if this line is a "page_num content_type" terminator
            m = re.match(r'^(\d+)\s+(text|table|image|combination)$', line)
            if m:
                page_num = int(m.group(1))
                content_type = m.group(2)
                i += 1
                break
            block.append(line)
            i += 1
        else:
            # No terminator found — remaining lines are incomplete
            break

        if not block:
            continue

        # block has at least: [question..., context..., answer...]
        # The boundary between fields is not explicitly marked; we rely on the
        # fact that the PDF dumps each cell as one line (they may wrap but rarely do).
        # Best split: first line = question, last line = answer, middle = context.
        if len(block) == 1:
            # degenerate: only question present
            question = block[0]
            context = ""
            answer = ""
        elif len(block) == 2:
            question = block[0]
            context = ""
            answer = block[1]
        else:
            # question = first line
            # answer = last line
            # context = everything in between
            question = block[0]
            answer = block[-1]
            context = " ".join(block[1:-1])

        records.append({
            "question": question,
            "ground_truth_context": context,
            "ground_truth_answer": answer,
            "page_number": page_num,
            "context_content_type": content_type,
        })

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} evaluation rows.")
    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else "RAG_evaluation_dataset - convertcsv (2).pdf"
    df = load_eval_dataset(path)
    print(df[["question", "page_number", "context_content_type"]].to_string())
    print(f"\nTotal rows: {len(df)}")
    print(f"Content types:\n{df['context_content_type'].value_counts()}")
