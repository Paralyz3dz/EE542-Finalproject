import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.rag.corpus_hygiene import filter_chunks, load_policy
from src.db.queries import vector_search
from src.rag import retriever
from tests import test_retriever as fixtures


def chunk(identifier, text):
    return {"id": identifier, "text": text, "embed_text": text,
            "content_hash": hashlib.sha256(text.encode()).hexdigest()[:16]}


def entry(c):
    return {"id": c["id"], "content_hash": c["content_hash"],
            "text_sha256": hashlib.sha256(c["text"].encode()).hexdigest(), "reason": "index"}


class HygienePolicyTests(unittest.TestCase):
    def test_only_reviewed_fingerprinted_chunks_are_excluded(self):
        noise = chunk("noise", "Capital gains 54 Losses 55")
        rule = chunk("rule", "A genuine rule that happens to mention an index")
        kept, excluded, stale = filter_chunks([noise, rule], {"exclusions": [entry(noise)]})
        self.assertEqual(kept, [rule])
        self.assertEqual(excluded, ["noise"])
        self.assertEqual(stale, [])

    def test_changed_text_or_embedding_is_retained(self):
        original = chunk("noise", "Reference list")
        for changed in (dict(original, text="A new substantive rule"), dict(original, content_hash="0" * 16)):
            kept, excluded, stale = filter_chunks([changed], {"exclusions": [entry(original)]})
            self.assertEqual(kept, [changed])
            self.assertEqual(excluded, [])
            self.assertEqual(stale, ["noise"])

    def test_exported_chunks_without_content_hash_use_same_text_fingerprint(self):
        original = chunk("noise", "Reference list")
        exported = {k:v for k,v in original.items() if k != "content_hash"}
        self.assertEqual(filter_chunks([exported], {"exclusions": [entry(original)]})[1], ["noise"])

    def test_policy_rejects_duplicates_and_protected_blocks(self):
        record = entry(chunk("noise", "Reference list"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.json"
            for policy in ({"version":1,"exclusions":[record,record]},
                           {"version":1,"exclusions":[record],"protected_ids":["noise"]}):
                path.write_text(json.dumps(policy))
                with self.assertRaises(ValueError):
                    load_policy(path)

    def test_reviewed_policy_keeps_worksheets_and_mixed_rules(self):
        policy = load_policy()
        ids = {r["id"] for r in policy["exclusions"]}
        self.assertEqual(len(ids), 440)
        self.assertIn("irs_pub_544_2025_ch00_280", ids)
        self.assertIn("irs_pub_575_2025_ch00_012", ids)
        for protected in ("irs_pub_590_b_2025_ch00_226", "irs_pub_544_2025_ch00_289", "irs_pub_334_2025_ch00_067"):
            self.assertNotIn(protected, ids)

    def test_vector_exclusion_is_bound_and_applied_before_limit(self):
        conn = MagicMock()
        vector_search(conn, [1], "model", 2025, k=10, excluded_ids=["x'); DROP TABLE chunks; --"])
        sql, parameters = conn.execute.call_args.args
        self.assertLess(sql.index("id <> ALL"), sql.index("LIMIT"))
        self.assertNotIn("DROP TABLE", sql)
        self.assertEqual(parameters["k"], 10)
        self.assertEqual(parameters["excluded_ids"], ["x'); DROP TABLE chunks; --"])


class HygieneRetrievalTests(unittest.TestCase):
    setUp = fixtures.RetrievalTests.setUp
    configure_database = fixtures.RetrievalTests.configure_database

    def test_both_routes_filter_before_candidate_selection(self):
        self.configure_database(6004)
        noise = chunk("noise", "Reference list")
        valid = chunk("valid", "Substantive pension rule")
        self.fetch_chunks.return_value = [noise, valid]
        self.vector_search.return_value = [("valid", "IRS Pub 575 (2025)", valid["text"], .8)]
        with patch('src.rag.corpus_hygiene.load_policy', return_value={"exclusions":[entry(noise)]}), patch.object(retriever, 'bm25_search', return_value=[]) as bm25:
            result = retriever.retrieve('pension', candidate_k=10, exclude_reference_chunks=True)
        self.assertEqual(self.vector_search.call_args.kwargs['excluded_ids'], ['noise'])
        self.assertEqual(self.vector_search.call_args.kwargs['k'], 10)
        self.assertEqual(bm25.call_args.args[1], [valid])
        self.assertEqual(bm25.call_args.kwargs['k'], 10)
        self.assertEqual(result['matches'][0]['chunk_id'], 'valid')

    def test_disabled_cleanup_does_not_load_policy(self):
        self.configure_database(6004)
        self.vector_search.return_value = []
        with patch.object(retriever, 'filter_chunks') as filtering:
            retriever.retrieve('pension', exclude_reference_chunks=False)
        filtering.assert_not_called()
        self.assertNotIn('excluded_ids', self.vector_search.call_args.kwargs)


if __name__ == '__main__':
    unittest.main()
