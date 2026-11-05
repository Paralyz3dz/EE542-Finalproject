import unittest

from src.rag.hybrid import bm25_search, fuse_rankings


def chunk(identifier, text):
    return {"id": identifier, "embed_text": text, "text": text,
            "citation": f"Source {identifier}"}


def match(identifier, score):
    return {"chunk_id": identifier, "text": f"Passage {identifier}",
            "citation": f"Source {identifier}", "score": score}


class BM25Tests(unittest.TestCase):
    def test_form_numbers_and_case_match(self):
        chunks = [chunk("form", "File Form 1040-SR"), chunk("other", "Student loans")]
        hits = bm25_search("FORM 1040-sr", chunks)
        self.assertEqual([hit["chunk_id"] for hit in hits], ["form"])
        self.assertEqual(hits[0]["citation"], "Source form")

    def test_no_overlap_does_not_invent_keyword_hits(self):
        self.assertEqual(bm25_search("mortgage", [chunk("income", "Taxable wages")]), [])

    def test_empty_corpus_or_tokens_return_no_keyword_hits(self):
        self.assertEqual(bm25_search("income", []), [])
        self.assertEqual(bm25_search("???", [chunk("a", "Income")]), [])
        self.assertEqual(bm25_search("income", [chunk("a", " ")]), [])

    def test_common_terms_with_nonpositive_scores_still_match(self):
        hits = bm25_search("income", [chunk("a", "income income")])
        self.assertEqual([hit["chunk_id"] for hit in hits], ["a"])
        self.assertLessEqual(hits[0]["score"], 0)


class FusionTests(unittest.TestCase):
    def test_both_routes_beat_one_route_despite_incompatible_raw_scores(self):
        hits = fuse_rankings(
            [match("vector-only", 9999), match("shared", 0.01)],
            [match("shared", 0.001), match("keyword-only", 500)],
        )
        self.assertEqual(hits[0]["chunk_id"], "shared")
        self.assertAlmostEqual(hits[0]["score"], 1 / 62 + 1 / 61)
        self.assertEqual(hits[0]["vector_score"], 0.01)
        self.assertEqual(hits[0]["bm25_score"], 0.001)
        self.assertEqual(hits[0]["citation"], "Source shared")

    def test_duplicates_do_not_inflate_fusion_scores(self):
        hits = fuse_rankings([match("a", 0.9), match("a", 0.8), match("b", 0.7)], [])
        self.assertEqual(len(hits), 2)
        self.assertAlmostEqual(hits[0]["score"], 1 / 61)
        self.assertEqual(hits[1]["vector_rank"], 2)
        self.assertIsNone(hits[0]["bm25_score"])

    def test_fused_output_is_limited_to_five_unique_chunks(self):
        hits = fuse_rankings([match(str(i), 0.9) for i in range(5)],
                             [match(str(i), 99) for i in range(4, 9)])
        self.assertEqual(len(hits), 5)
        self.assertEqual(len({hit["chunk_id"] for hit in hits}), 5)
        self.assertEqual(hits[0]["chunk_id"], "4")


if __name__ == "__main__":
    unittest.main()
