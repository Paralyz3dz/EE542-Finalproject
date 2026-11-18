import json
import unittest
from unittest.mock import patch

import numpy as np
import pytest

from src.rag import retriever
from src.rag.reranker import rerank_matches, options
from tests import test_retriever as fixtures


def candidates(count=10):
    return [{"chunk_id": f"c{i}", "citation": f"Official source {i}", "text": f"Body {i}", "score": 1 / (60 + i), "vector_rank": i} for i in range(1, count + 1)]


def test_reranker_can_promote_a_passage_outside_rrf_top_five():
    original = candidates()
    passages = {m["chunk_id"]: "Heading\n" + m["text"] for m in original}
    pairs_seen = []
    def score(pairs):
        pairs_seen.extend(pairs)
        return [-5, -4, -3, -2, -1, 10, 9, 8, 7, 6]
    result = rerank_matches("Question", original, passages, scorer=score)
    assert [m["chunk_id"] for m in result] == ["c6", "c7", "c8", "c9", "c10"]
    assert result[0]["citation"] == original[5]["citation"]
    assert result[0]["text"] == original[5]["text"]
    assert result[0]["rrf_score"] == original[5]["score"]
    assert result[0]["score"] == result[0]["rerank_score"] == 10
    assert pairs_seen[0] == ("Question", "Heading\nBody 1")
    assert "rerank_score" not in original[0]


def test_reranker_ties_preserve_the_rrf_order():
    original = candidates()
    result = rerank_matches("Question", original, {m["chunk_id"]: m["text"] for m in original}, scorer=lambda _: [1] * 10)
    assert [m["chunk_id"] for m in result] == [m["chunk_id"] for m in original[:5]]


@pytest.mark.parametrize("scores", [[1] * 9, [float("nan")] * 10, [float("inf")] * 10, [[1]] * 10])
def test_invalid_scores_cannot_silently_replace_ranking(scores):
    original = candidates()
    with pytest.raises(RuntimeError):
        rerank_matches("Question", original, {m["chunk_id"]: m["text"] for m in original}, scorer=lambda _: scores)


def test_reranker_failure_propagates_without_a_silent_fallback():
    original = candidates()
    def failed(_):
        raise RuntimeError("Model unavailable")
    with pytest.raises(RuntimeError, match="Model unavailable"):
        rerank_matches("Question", original, {m["chunk_id"]: m["text"] for m in original}, scorer=failed)


def test_missing_passages_are_rejected_before_model_scoring():
    with pytest.raises(ValueError, match="complete passages"):
        rerank_matches("Question", candidates(), {}, scorer=lambda _: pytest.fail("Scoring must not run"))


class RerankRetrievalTests(unittest.TestCase):
    setUp = fixtures.RetrievalTests.setUp
    configure_database = fixtures.RetrievalTests.configure_database

    def test_production_pipeline_preserves_ten_candidates_before_reranking_to_five(self):
        self.configure_database(8752)
        original = candidates()
        self.vector_search.return_value = [(m["chunk_id"], m["citation"], m["text"], .9) for m in original]
        self.fetch_chunks.return_value = [{"id":m["chunk_id"], "text":m["text"], "embed_text":"Heading\n"+m["text"]} for m in original]
        settings = {"tax_year":2025, "retrieve_top_k":5, "retrieve_candidate_k":10, "retrieve_fusion_k":10, "rrf_constant":60, "rerank_enabled":True, "rerank_top_k":5}
        with patch.object(retriever.json, "loads", return_value=settings), patch.object(retriever, "bm25_search", return_value=[]) as sparse, patch.object(retriever, "rerank_matches", side_effect=lambda q, pool, texts, **kw: list(reversed(pool))[:5]) as rank:
            result = retriever.retrieve("Question")
        assert self.vector_search.call_args.kwargs["k"] == sparse.call_args.kwargs["k"] == 10
        assert len(rank.call_args.args[1]) == 10
        assert rank.call_args.args[2]["c10"] == "Heading\nBody 10"
        assert rank.call_args.kwargs["k"] == 5
        assert [m["chunk_id"] for m in result["matches"]] == ["c10","c9","c8","c7","c6"]
        assert result["retrieval_method"] == "hnsw_bm25_rrf_rerank"

    def test_disabled_reranking_returns_the_same_rrf_prefix_without_loading_a_model(self):
        self.configure_database(8752)
        self.vector_search.return_value = [(f"c{i}", "Source", "Text", .9) for i in range(10)]
        settings = {"tax_year":2025, "retrieve_top_k":5, "retrieve_candidate_k":10, "retrieve_fusion_k":10, "rrf_constant":60, "rerank_enabled":False}
        with patch.object(retriever.json,"loads",return_value=settings), patch.object(retriever,"bm25_search",return_value=[]), patch.object(retriever,"rerank_matches") as rank:
            result= retriever.retrieve("Question")
        rank.assert_not_called()
        assert [m["chunk_id"] for m in result["matches"]] == [f"c{i}" for i in range(5)]

    def test_invalid_rerank_flag_is_rejected_before_database_access(self):
        with self.assertRaises(ValueError):
            retriever.retrieve("Question", rerank_enabled="yes")
        self.conn.execute.assert_not_called()
