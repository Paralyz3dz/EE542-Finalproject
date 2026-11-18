from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import numpy as np
import pytest

from eval.run_rerank_blend_experiment import main, observe, summarize
from src.rag import reranker, retriever
from tests import test_retriever as fixtures


def matches():
    return [{"chunk_id": "c1", "citation": "Source A", "text": "Body A", "score": 1.0},
            {"chunk_id": "c2", "citation": "Source B", "text": "Body B", "score": 2.0},
            {"chunk_id": "c3", "citation": "Source C", "text": "Body C", "score": 3.0}]


def test_half_weight_normalizes_before_combining_and_preserves_raw_scores():
    pool = matches()
    result = reranker.rank_scored_matches(pool, [6, 4, 0], k=3, weight=.5)
    assert [m["chunk_id"] for m in result] == ["c2", "c1", "c3"]
    assert result[0]["normalized_rrf_score"] == .5
    assert result[0]["normalized_rerank_score"] == pytest.approx(2 / 3)
    assert result[0]["score"] == result[0]["blend_score"] == pytest.approx(7 / 12)
    assert result[0]["rrf_score"] == 2
    assert result[0]["rerank_score"] == 4
    assert result[0]["text"] == "Body B"
    assert result[0]["rerank_rank"] == 2
    assert result[0]["blend_rank"] == 1
    assert "blend_score" not in pool[0]


def test_constant_score_lists_produce_finite_stable_rankings():
    pool = [{**m, "score": 2.0} for m in matches()]
    result = reranker.rank_scored_matches(pool, [4, 4, 4], k=3, weight=.5)
    assert [m["chunk_id"] for m in result] == ["c1", "c2", "c3"]
    assert all(m["score"] == 0 and np.isfinite(m["score"]) for m in result)


def test_zero_weight_preserves_the_original_raw_rerank_contract():
    result = reranker.rank_scored_matches(matches(), [-6, 4, 0], k=3, weight=0)
    assert [m["chunk_id"] for m in result] == ["c2", "c3", "c1"]
    assert [m["score"] for m in result] == [4, 0, -6]
    assert not any("blend_score" in m for m in result)


def test_full_rrf_weight_uses_rrf_scores_only():
    result = reranker.rank_scored_matches(matches(), [100, 0, -100], k=3, weight=1)
    assert [m["chunk_id"] for m in result] == ["c3", "c2", "c1"]


@pytest.mark.parametrize("weight", [-.1, 1.1, float("nan"), float("inf"), True, "0.5", None])
def test_invalid_weights_are_rejected_before_model_scoring(weight):
    scorer = Mock()
    with pytest.raises(ValueError, match="rerank_rrf_weight"):
        reranker.rerank_matches("Question", matches(), {m["chunk_id"]: m["text"] for m in matches()}, settings={"rerank_rrf_weight": weight}, scorer=scorer)
    scorer.assert_not_called()


def test_summary_uses_paired_baseline_and_reports_saved_baseline_drift():
    rows = [{"id": f"q{i}", "round": 1,
             "baseline": {"hit_at_3": i < 75, "hit_at_5": i < 85, "mrr_at_5": 1.0 if i < 75 else .2 if i < 85 else 0},
             "blended": {"hit_at_3": i < 78, "hit_at_5": i < 86, "mrr_at_5": 1.0 if i < 78 else .2 if i < 86 else 0},
             "rrf_source_coverage_at_10": i < 96, "saved_baseline_rankings_reproduced": i != 1,
             "saved_baseline_metrics_reproduced": True, "retrieval_seconds": 3., "rerank_seconds": 4., "elapsed_seconds": 7.} for i in range(104)]
    result = summarize(rows, {"reranked_top5": {}, "mean_total_seconds": 7.})
    assert result["paired_pure_rerank_top5"]["hits_at_5"] == 85
    assert result["blended_top5"]["hits_at_5"] == 86
    assert result["gained_hit_at_5"] == ["q85"]
    assert result["saved_baseline_ranking_differences"] == ["q1"]
    assert result["primary_improved"]
    with pytest.raises(ValueError, match="one complete"):
        summarize(rows[:-1], {})


def test_completed_blend_round_cannot_run_again(tmp_path, monkeypatch):
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["run_rerank_blend_experiment", "--env-file", str(tmp_path / "missing.env"), "--out", str(tmp_path)])
    with patch("eval.run_rerank_blend_experiment.observe") as observation, patch("eval.run_rerank_blend_experiment.snapshot") as snapshot:
        with pytest.raises(ValueError, match="cannot be rerun"):
            main()
    observation.assert_not_called()
    snapshot.assert_not_called()


class BlendRetrievalTests(unittest.TestCase):
    setUp = fixtures.RetrievalTests.setUp
    configure_database = fixtures.RetrievalTests.configure_database

    def test_production_blend_scores_ten_candidates_once_and_keeps_five(self):
        self.configure_database(8752)
        chunks = [{"id": f"c{i:02}", "citation": "Source", "text": f"Body {i}", "embed_text": f"Heading {i} Body {i}"} for i in range(10)]
        self.fetch_chunks.return_value = chunks
        self.vector_search.return_value = [(c["id"], c["citation"], c["text"], .9) for c in chunks]
        settings = {"tax_year": 2025, "retrieve_top_k": 5, "retrieve_candidate_k": 10, "retrieve_fusion_k": 10, "rrf_constant": 60,
                    "rerank_enabled": True, "rerank_top_k": 5, "rerank_rrf_weight": .5}
        scorer = Mock(return_value=list(range(10)))
        with patch.object(retriever.json, "loads", return_value=settings), patch.object(retriever, "bm25_search", return_value=[]) as sparse, patch.object(reranker, "prepare", return_value=SimpleNamespace(score=scorer)):
            result = retriever.retrieve("Question")
        assert self.vector_search.call_args.kwargs["k"] == sparse.call_args.kwargs["k"] == 10
        scorer.assert_called_once()
        assert len(scorer.call_args.args[0]) == 10
        assert scorer.call_args.args[0][0] == ("Question", "Heading 0 Body 0")
        assert len(result["matches"]) == 5
        assert all(m["score"] == m["blend_score"] for m in result["matches"])
