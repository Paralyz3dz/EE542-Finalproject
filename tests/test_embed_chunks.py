import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.rag.embed_chunks import generate


class LocalExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.input = Path(self.directory.name) / "chunks.json"
        self.output = Path(self.directory.name) / "vectors.jsonl"
        self.chunks = [
            {"id": "chunk-1", "embed_text": "Heading > Section\nFirst passage"},
            {"id": "chunk-2", "embed_text": "Second passage"},
        ]
        self.input.write_text(json.dumps(self.chunks), encoding="utf-8")

    def test_export_is_local_and_preserves_text_identity(self):
        with patch("src.rag.embed_chunks.embed_texts", return_value=np.ones((2, 1024))) as embed:
            with patch("src.rag.retriever.get_conn", side_effect=AssertionError("Database accessed")):
                generate(self.input, self.output)
        self.assertEqual(embed.call_args.args[0], [c["embed_text"] for c in self.chunks])
        rows = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual([row["id"] for row in rows], [c["id"] for c in self.chunks])
        for chunk, row in zip(self.chunks, rows):
            self.assertEqual(len(row["embedding"]), 1024)
            self.assertEqual(row["content_hash"], hashlib.sha256(chunk["embed_text"].encode()).hexdigest()[:16])
        self.assertFalse(self.output.with_suffix(".jsonl.partial").exists())

    def test_failed_batch_can_resume_without_reembedding_completed_chunks(self):
        with patch("src.rag.embed_chunks.embed_texts", side_effect=[np.ones((1, 1024)), RuntimeError("Ollama stopped")]):
            with self.assertRaisesRegex(RuntimeError, "Ollama stopped"):
                generate(self.input, self.output, batch_size=1)
        self.assertFalse(self.output.exists())
        with patch("src.rag.embed_chunks.embed_texts", return_value=np.ones((1, 1024))) as embed:
            generate(self.input, self.output, batch_size=1)
        embed.assert_called_once_with([self.chunks[1]["embed_text"]])
        self.assertEqual(len(self.output.read_text().splitlines()), 2)

    def test_changed_text_cannot_reuse_stale_partial_vectors(self):
        with patch("src.rag.embed_chunks.embed_texts", side_effect=[np.ones((1, 1024)), RuntimeError("Stopped")]):
            with self.assertRaises(RuntimeError):
                generate(self.input, self.output, batch_size=1)
        self.chunks[0]["embed_text"] = "Changed passage"
        self.input.write_text(json.dumps(self.chunks), encoding="utf-8")
        with patch("src.rag.embed_chunks.embed_texts") as embed:
            with self.assertRaisesRegex(ValueError, "does not match"):
                generate(self.input, self.output)
            embed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
