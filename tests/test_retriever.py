import io
import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.rag import retriever


class QueryEmbeddingTests(unittest.TestCase):
    def response(self, vectors):
        return io.BytesIO(json.dumps({"embeddings": vectors}).encode())

    def test_query_uses_instruction_without_silent_truncation(self):
        with patch.object(retriever, "urlopen") as request:
            request.return_value = self.response([[0.1] * 1024])
            vector = retriever.embed_query("  Who must file?  ")

        payload = json.loads(request.call_args.args[0].data)
        self.assertTrue(payload["input"][0].startswith("Instruct: "))
        self.assertTrue(payload["input"][0].endswith("\nQuery: Who must file?"))
        self.assertFalse(payload["truncate"])
        self.assertEqual(vector.shape, (1024,))
        self.assertEqual(vector.dtype, np.float32)

    def test_unusable_vectors_fail_before_database_search(self):
        for vectors in ([], [[0.1] * 768], [[0.0] * 1024], [[float("nan")] * 1024]):
            with self.subTest(vectors_length=len(vectors)):
                with patch.object(retriever, "urlopen", return_value=self.response(vectors)):
                    with self.assertRaises(RuntimeError):
                        retriever.embed_query("Who must file?")

    def test_empty_query_does_not_call_ollama(self):
        with patch.object(retriever, "urlopen") as request:
            with self.assertRaises(ValueError):
                retriever.embed_query("  ")
            request.assert_not_called()

    def test_document_batch_is_embedded_verbatim(self):
        texts = ["Chapter 1 > Filing\nFirst passage", "Second passage"]
        with patch.object(retriever, "urlopen") as request:
            request.return_value = self.response([[0.1] * 1024, [0.2] * 1024])
            vectors = retriever.embed_texts(texts)
        self.assertEqual(json.loads(request.call_args.args[0].data)["input"], texts)
        self.assertEqual(vectors.shape, (2, 1024))


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.conn = MagicMock()
        self.connection = patch.object(retriever, "get_conn")
        self.connection.start().return_value.__enter__.return_value = self.conn
        self.addCleanup(self.connection.stop)
        self.registration = patch.object(retriever, "register_vector")
        self.registration.start()
        self.addCleanup(self.registration.stop)
        self.embedding = patch.object(retriever, "embed_query")
        self.embed_query = self.embedding.start()
        self.embed_query.return_value = np.ones(1024, dtype=np.float32)
        self.addCleanup(self.embedding.stop)
        self.search = patch.object(retriever, "vector_search")
        self.vector_search = self.search.start()
        self.addCleanup(self.search.stop)
        self.chunks = patch.object(retriever, "fetch_chunks", return_value=[])
        self.fetch_chunks = self.chunks.start()
        self.addCleanup(self.chunks.stop)

    def configure_database(self, count, index_exists=True):
        self.conn.execute.side_effect = [
            MagicMock(),
            MagicMock(fetchone=lambda: (count,)),
            MagicMock(fetchone=lambda: (index_exists,)),
            MagicMock(),
        ]

    def test_missing_embeddings_fail_instead_of_returning_unembedded_chunks(self):
        self.configure_database(0)
        with self.assertRaisesRegex(RuntimeError, "No stored embeddings"):
            retriever.retrieve("Who must file?")
        self.embed_query.assert_not_called()
        self.vector_search.assert_not_called()

    def test_missing_hnsw_does_not_fall_back_to_another_search(self):
        self.configure_database(820, index_exists=False)
        with self.assertRaisesRegex(RuntimeError, "No valid cosine HNSW index"):
            retriever.retrieve("Who must file?")
        self.vector_search.assert_not_called()

    def test_top_five_and_stored_citations_reach_the_caller(self):
        self.configure_database(820)
        self.vector_search.return_value = [
            ("chunk-1", "Stored IRS citation", "Source passage", 0.75)
        ]
        result = retriever.retrieve("Who must file?")
        self.assertEqual(self.vector_search.call_args.kwargs["k"], 5)
        self.assertEqual(result["matches"], [{
            "chunk_id": "chunk-1", "citation": "Stored IRS citation",
            "text": "Source passage", "score": 1 / 61,
            "vector_score": 0.75, "bm25_score": None,
            "vector_rank": 1, "bm25_rank": None,
        }])
        statements = [call.args[0] for call in self.conn.execute.call_args_list]
        self.assertEqual(statements[0], "SET TRANSACTION READ ONLY")
        self.assertFalse(any(s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE")) for s in statements))

    def test_keyword_only_candidate_can_reach_agent_results(self):
        self.configure_database(820)
        self.vector_search.return_value = [
            (name, f"Citation {name}", "Other source passage", 0.9 - i / 10)
            for i, name in enumerate(["a", "b", "c", "d", "e"])
        ]
        self.fetch_chunks.return_value = [
            {"id": "b", "citation": "Citation b", "text": "Shared passage",
             "embed_text": "Form 13614-C interview preparation"},
            {"id": "keyword-only", "citation": "Keyword source", "text": "Keyword passage",
             "embed_text": "13614-C 13614-C 13614-C"},
            {"id": "other", "citation": "Other", "text": "Unrelated",
             "embed_text": "wages income deductions"},
        ]
        result = retriever.retrieve("13614-C")
        self.assertEqual(result["retrieval_method"], "hnsw_bm25_rrf")
        self.assertEqual(len(result["matches"]), 5)
        self.assertEqual(result["matches"][0]["chunk_id"], "b")
        keyword_hit = next(row for row in result["matches"] if row["chunk_id"] == "keyword-only")
        self.assertIsNone(keyword_hit["vector_score"])
        self.assertIsNotNone(keyword_hit["bm25_score"])
        self.assertEqual(keyword_hit["citation"], "Keyword source")
        self.fetch_chunks.assert_called_once_with(self.conn, 2025)


if __name__ == "__main__":
    unittest.main()
