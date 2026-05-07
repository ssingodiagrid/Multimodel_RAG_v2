import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz

from src.parsers.visual_document_parser import VisualDocumentParser


class VisualDocumentParserTests(unittest.TestCase):
    def _create_sample_pdf(self, pdf_path: Path) -> None:
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        page.insert_text((40, 60), "Phase 6 test page")
        page.draw_rect(fitz.Rect(200, 80, 360, 220), color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
        doc.save(pdf_path)
        doc.close()

    def test_process_document_renders_pages_and_grid_patches(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pdf_path = tmp_path / "sample.pdf"
            self._create_sample_pdf(pdf_path)

            parser = VisualDocumentParser(
                output_root=tmp_path / "processed",
                patch_rows=2,
                patch_cols=2,
                render_dpi=72,
            )
            manifest = parser.process_document(pdf_path)

            self.assertEqual(manifest["source_pdf"], str(pdf_path))
            self.assertEqual(manifest["total_pages"], 1)
            self.assertEqual(manifest["total_patches"], 4)
            self.assertEqual(len(manifest["pages"]), 1)

            page_info = manifest["pages"][0]
            self.assertTrue(Path(page_info["page_image_path"]).exists())
            self.assertEqual(len(page_info["patches"]), 4)

            first_patch = page_info["patches"][0]
            self.assertTrue(Path(first_patch["patch_image_path"]).exists())
            self.assertEqual(first_patch["bbox"]["x1"], 0)
            self.assertEqual(first_patch["bbox"]["y1"], 0)
            self.assertEqual(first_patch["row"], 0)
            self.assertEqual(first_patch["col"], 0)

            manifest_path = Path(manifest["manifest_path"])
            self.assertTrue(manifest_path.exists())
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["total_patches"], 4)


if __name__ == "__main__":
    unittest.main()
