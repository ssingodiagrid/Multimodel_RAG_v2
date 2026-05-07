import json
import logging
from pathlib import Path
from typing import Any

import fitz
from PIL import Image


logger = logging.getLogger(__name__)


class VisualDocumentParser:
    """Render PDF pages to images and split each page into fixed-grid patches."""

    def __init__(
        self,
        output_root: Path,
        patch_rows: int = 4,
        patch_cols: int = 4,
        render_dpi: int = 144,
    ):
        self.output_root = Path(output_root)
        self.patch_rows = patch_rows
        self.patch_cols = patch_cols
        self.render_dpi = render_dpi

    def process_document(self, pdf_path: str | Path) -> dict[str, Any]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_name = pdf_path.stem
        doc_output_root = self.output_root / doc_name
        pages_dir = doc_output_root / "pages"
        patches_dir = doc_output_root / "patches"
        pages_dir.mkdir(parents=True, exist_ok=True)
        patches_dir.mkdir(parents=True, exist_ok=True)

        manifest_pages: list[dict[str, Any]] = []
        total_patches = 0

        with fitz.open(pdf_path) as doc:
            for page_index in range(len(doc)):
                page = doc[page_index]
                page_number = page_index + 1
                page_image_path, width, height = self._render_page_image(
                    page,
                    page_number,
                    pages_dir,
                )
                patches = self._extract_grid_patches(
                    page_image_path=page_image_path,
                    page_number=page_number,
                    patches_dir=patches_dir,
                    page_width=width,
                    page_height=height,
                )
                total_patches += len(patches)
                manifest_pages.append(
                    {
                        "page_number": page_number,
                        "page_image_path": str(page_image_path),
                        "width": width,
                        "height": height,
                        "patches": patches,
                    }
                )

        manifest = {
            "source_pdf": str(pdf_path),
            "document_id": doc_name,
            "render_dpi": self.render_dpi,
            "patch_grid": {"rows": self.patch_rows, "cols": self.patch_cols},
            "total_pages": len(manifest_pages),
            "total_patches": total_patches,
            "pages": manifest_pages,
        }

        manifest_path = doc_output_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
        logger.info(
            "Phase 6 visual preprocessing complete for %s: %s pages, %s patches",
            pdf_path.name,
            manifest["total_pages"],
            total_patches,
        )
        return manifest

    def _render_page_image(
        self,
        page: fitz.Page,
        page_number: int,
        pages_dir: Path,
    ) -> tuple[Path, int, int]:
        scale = self.render_dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        page_image_path = pages_dir / f"page_{page_number:03d}.png"
        pixmap.save(page_image_path)
        return page_image_path, pixmap.width, pixmap.height

    def _extract_grid_patches(
        self,
        page_image_path: Path,
        page_number: int,
        patches_dir: Path,
        page_width: int,
        page_height: int,
    ) -> list[dict[str, Any]]:
        patches: list[dict[str, Any]] = []
        with Image.open(page_image_path) as image:
            patch_width = page_width // self.patch_cols
            patch_height = page_height // self.patch_rows

            for row in range(self.patch_rows):
                for col in range(self.patch_cols):
                    x1 = col * patch_width
                    y1 = row * patch_height
                    x2 = page_width if col == self.patch_cols - 1 else (col + 1) * patch_width
                    y2 = page_height if row == self.patch_rows - 1 else (row + 1) * patch_height

                    patch = image.crop((x1, y1, x2, y2))
                    patch_id = f"page_{page_number:03d}_r{row}_c{col}"
                    patch_image_path = patches_dir / f"{patch_id}.png"
                    patch.save(patch_image_path)

                    patches.append(
                        {
                            "patch_id": patch_id,
                            "page_number": page_number,
                            "row": row,
                            "col": col,
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            "patch_image_path": str(patch_image_path),
                        }
                    )
        return patches
