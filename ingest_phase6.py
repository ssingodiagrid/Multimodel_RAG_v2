import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from configs.settings import settings
from src.parsers.visual_document_parser import VisualDocumentParser


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting Phase 6 visual document preprocessing...")
    pdf_path = settings.base_dir / "ifc-annual-report-2024-financials.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    output_root = settings.processed_data_dir / "phase6_visual"
    parser = VisualDocumentParser(
        output_root=output_root,
        patch_rows=4,
        patch_cols=4,
        render_dpi=144,
    )
    manifest = parser.process_document(pdf_path)
    logger.info("Phase 6 manifest written to %s", manifest["manifest_path"])


if __name__ == "__main__":
    main()
