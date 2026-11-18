import json
from unittest.mock import patch

import pytest

from eval.run_rerank_experiment import main, source_found, summarize


def records():
    return [{"id":f"q{i}", "round":1, "baseline":{"hit_at_3":i<76,"hit_at_5":i<82,"mrr_at_5":1.0 if i<76 else .2 if i<82 else 0},
             "reranked":{"hit_at_3":i<80,"hit_at_5":i<86,"mrr_at_5":1.0 if i<80 else .2 if i<86 else 0},
             "rrf_source_coverage_at_10":i<90,"retrieval_seconds":2.0,"rerank_seconds":5.0,"elapsed_seconds":7.0} for i in range(104)]


def test_paired_summary_separates_candidate_coverage_and_final_hits():
    result=summarize(records())
    assert result["baseline_rrf_top5"]["hits_at_5"]==82
    assert result["reranked_top5"]["hits_at_5"]==86
    assert result["rrf_source_coverage_at_10"]==90
    assert result["gained_hit_at_5"]==["q82","q83","q84","q85"]
    assert result["lost_hit_at_5"]==[]
    assert result["mean_total_seconds"]==7
    assert result["primary_improved"] is True


@pytest.mark.parametrize("kind", ["incomplete","duplicate","extra_round"])
def test_incomplete_or_repeated_rounds_are_rejected(kind):
    rows=records()
    if kind=="incomplete":rows.pop()
    elif kind=="duplicate":rows[-1]=dict(rows[0])
    else:rows[-1]["round"]=2
    with pytest.raises(ValueError):summarize(rows)


def test_candidate_ceiling_checks_beyond_the_fifth_position():
    matches=[{"citation":"IRS Pub 527 (2025), p. 1"} for _ in range(9)]+[{"citation":"IRS Pub 17 (2025), p. 3"}]
    question={"source":"IRS Pub 17"}
    assert source_found(question,matches)
    assert not source_found(question,matches[:5])


def test_completed_rerank_round_is_not_executed_again(tmp_path,monkeypatch):
    (tmp_path/"summary.json").write_text("{}",encoding="utf-8")
    monkeypatch.setattr("sys.argv",["run_rerank_experiment","--env-file",str(tmp_path/"missing.env"),"--out",str(tmp_path)])
    with patch("eval.run_rerank_experiment.observe") as observe,patch("eval.run_rerank_experiment.snapshot") as snapshot:
        with pytest.raises(ValueError,match="cannot be rerun"):main()
    observe.assert_not_called()
    snapshot.assert_not_called()
