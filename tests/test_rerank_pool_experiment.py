import unittest
from unittest.mock import patch

import pytest

from eval.run_rerank_pool_experiment import main, summarize, validate_pool
from src.rag import retriever
from src.rag.reranker import rerank_matches
from tests import test_retriever as fixtures


def test_overlapping_routes_can_have_fewer_than_fifteen_unique_candidates():
    dense = [{"chunk_id": f"c{i}"} for i in range(10)]
    sparse = [{"chunk_id": f"c{i}"} for i in range(2, 12)]
    validate_pool(dense, sparse, [{"chunk_id": f"c{i}"} for i in range(12)])
    with pytest.raises(ValueError):
        validate_pool(dense, sparse, dense)


def test_union_larger_than_fifteen_is_capped_at_fifteen():
    dense = [{"chunk_id": f"c{i}"} for i in range(10)]
    sparse = [{"chunk_id": f"c{i}"} for i in range(10, 20)]
    validate_pool(dense, sparse, (dense + sparse)[:15])
    with pytest.raises(ValueError):
        validate_pool(dense, sparse, (dense + sparse)[:10])


def records():
    return [{"id": f"q{i}", "round": 1, "baseline": {"hit_at_3": i < 75, "hit_at_5": i < 85, "mrr_at_5": 1.0 if i < 75 else .2 if i < 85 else 0},
             "reranked": {"hit_at_3": i < 78, "hit_at_5": i < 87, "mrr_at_5": 1.0 if i < 78 else .2 if i < 87 else 0},
             "actual_fusion_count": 12 if i < 4 else 15, "rrf_source_coverage_at_10": i < 96, "rrf_source_coverage_at_15": i < 98,
             "retrieval_seconds": 3.0, "rerank_seconds": 6.0, "elapsed_seconds": 9.0} for i in range(104)]


def baseline_summary():
    return {"reranked_top5": {"hits_at_3": 75, "hit_at_3": 75 / 104, "hits_at_5": 85, "hit_at_5": 85 / 104, "mrr_at_5": sum(r["baseline"]["mrr_at_5"] for r in records()) / 104},
            "mean_total_seconds": 7.0}


def test_summary_compares_against_the_reranked_baseline():
    import numpy as np
    baseline = baseline_summary()
    baseline["reranked_top5"]["mrr_at_5"] = float(np.mean([r["baseline"]["mrr_at_5"] for r in records()]))
    result = summarize(records(), baseline)
    assert result["baseline_rrf10_reranked_top5"]["hits_at_5"] == 85
    assert result["rrf15_reranked_top5"]["hits_at_5"] == 87
    assert result["rrf_source_coverage_at_15"] == 98
    assert result["actual_fusion_count_histogram"] == {12: 4, 15: 100}
    assert result["gained_hit_at_5"] == ["q85", "q86"]
    assert result["primary_improved"]


def test_summary_rejects_a_changed_saved_baseline():
    baseline = baseline_summary()
    baseline["reranked_top5"]["hits_at_5"] = 84
    with pytest.raises(ValueError, match="Baseline metrics"):
        summarize(records(), baseline)


@pytest.mark.parametrize("kind", ["incomplete", "duplicate", "extra_round"])
def test_summary_rejects_invalid_rounds(kind):
    rows = records()
    if kind == "incomplete":
        rows.pop()
    elif kind == "duplicate":
        rows[-1] = dict(rows[0])
    else:
        rows[-1]["round"] = 2
    with pytest.raises(ValueError, match="one complete"):
        summarize(rows, baseline_summary())


def test_completed_pool_round_cannot_run_again(tmp_path, monkeypatch):
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["run_rerank_pool_experiment", "--env-file", str(tmp_path / "missing.env"), "--out", str(tmp_path)])
    with patch("eval.run_rerank_pool_experiment.observe") as observe, patch("eval.run_rerank_pool_experiment.snapshot") as snapshot:
        with pytest.raises(ValueError, match="cannot be rerun"):
            main()
    observe.assert_not_called()
    snapshot.assert_not_called()


class PoolRetrievalTests(unittest.TestCase):
    setUp = fixtures.RetrievalTests.setUp
    configure_database = fixtures.RetrievalTests.configure_database

    def test_production_can_promote_a_passage_from_rrf_rank_fifteen(self):
        self.configure_database(8752)
        chunks = [{"id": f"c{i:02}", "citation": "Source", "text": f"Body {i}", "embed_text": f"Heading {i} Body {i}"} for i in range(1, 16)]
        self.fetch_chunks.return_value = chunks
        self.vector_search.return_value = [(c["id"], c["citation"], c["text"], .9) for c in chunks[:10]]
        sparse = [{"chunk_id": c["id"], "citation": c["citation"], "text": c["text"], "score": .8} for c in chunks[5:]]
        settings = {"tax_year": 2025, "retrieve_top_k": 5, "retrieve_candidate_k": 10, "retrieve_fusion_k": 15, "rrf_constant": 60, "rerank_enabled": True, "rerank_top_k": 5}
        def rank(query, pool, texts, **kwargs):
            assert len(pool) == 15
            assert kwargs["k"] == 5
            return rerank_matches(query, pool, texts, k=5, scorer=lambda pairs: list(range(len(pairs))))
        with patch.object(retriever.json, "loads", return_value=settings), patch.object(retriever, "bm25_search", return_value=sparse) as bm25, patch.object(retriever, "rerank_matches", side_effect=rank) as rerank:
            result = retriever.retrieve("Question")
        assert self.vector_search.call_args.kwargs["k"] == bm25.call_args.kwargs["k"] == 10
        assert len(result["matches"]) == 5
        assert result["matches"][0]["rrf_rank"] == 15
        assert len(rerank.call_args.args[1]) == 15
