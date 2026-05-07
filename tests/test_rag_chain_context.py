import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.documents import Document

from src.generation import rag_chain
from src.retrieval.semantic_cache import SemanticCacheManager


class TableContextEnrichmentTests(unittest.TestCase):
    def test_table_docs_include_same_page_text_context(self):
        docs = [
            Document(
                page_content="TABLE SUMMARY:\nTotal assets decreased from $110,547 to $108,187.",
                metadata={
                    "source": "ifc.pdf",
                    "page_number": 4,
                    "content_type": "table",
                },
            )
        ]
        page_context_map = {
            ("ifc.pdf", 4): (
                "AS OF THE YEAR ENDED JUNE 30 (US$ in millions) 2024 2023 "
                "Balance Sheet Total assets $108,187 $110,547"
            )
        }

        formatted = rag_chain.format_docs_with_page_context(docs, page_context_map)

        self.assertIn("TABLE SUMMARY", formatted)
        self.assertIn("PAGE TEXT CONTEXT", formatted)
        self.assertIn("JUNE 30", formatted)
        self.assertIn("2023", formatted)

    def test_text_docs_are_not_duplicated_with_page_context(self):
        docs = [
            Document(
                page_content="Balance Sheet Total assets $108,187 $110,547",
                metadata={
                    "source": "ifc.pdf",
                    "page_number": 4,
                    "content_type": "text",
                },
            )
        ]

        formatted = rag_chain.format_docs_with_page_context(docs, {("ifc.pdf", 4): "ignored"})

        self.assertEqual(formatted, "Balance Sheet Total assets $108,187 $110,547")

    def test_non_answers_are_not_cached(self):
        self.assertFalse(rag_chain.should_cache_answer("I cannot answer this question based on the context provided."))
        self.assertFalse(rag_chain.should_cache_answer("I don't know based on the provided context."))
        self.assertTrue(rag_chain.should_cache_answer("Total assets as of June 30, 2023, were $110,547 million."))

    def test_keyword_rerank_promotes_exact_total_assets_match(self):
        docs = [
            Document(
                page_content="The carrying value of IFC's outstanding investment portfolio was $58.7 billion at June 30, 2024 and $51.5 billion at June 30, 2023.",
                metadata={"page_number": 6, "content_type": "text"},
            ),
            Document(
                page_content="AS OF THE YEAR ENDED JUNE 30 (US$ in millions) 2024 2023 Balance Sheet Total assets $108,187 $110,547",
                metadata={"page_number": 4, "content_type": "text"},
            ),
        ]

        reranked = rag_chain.rerank_docs_by_query(
            "What was the total value of IFC's assets as of June 30, 2023?",
            docs,
        )

        self.assertIn("Total assets", reranked[0].page_content)


class SessionScopedCacheTests(unittest.TestCase):
    def test_cache_path_uses_session_namespace(self):
        with patch.object(SemanticCacheManager, "_load_or_create_cache", return_value=None):
            cache = SemanticCacheManager(cache_namespace="session-abc")
        expected_suffix = Path("semantic_cache") / "session-abc"
        self.assertTrue(cache.cache_path.endswith(str(expected_suffix)))

    def test_cleanup_old_cache_namespaces_preserves_active_one(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "session-a").mkdir()
            (root / "session-b").mkdir()
            (root / "active-session").mkdir()
            (root / "notes.txt").write_text("keep me", encoding="utf-8")

            SemanticCacheManager.cleanup_old_namespaces(root, active_namespace="active-session")

            self.assertFalse((root / "session-a").exists())
            self.assertFalse((root / "session-b").exists())
            self.assertTrue((root / "active-session").exists())
            self.assertTrue((root / "notes.txt").exists())


if __name__ == "__main__":
    unittest.main()
