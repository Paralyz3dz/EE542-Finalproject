import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from eval.run_passage_relevance import build_case, judge_case, main, quote_status, score_case, summarize, validate_judgment


def example():
    question = {"id": "q_test", "module": "base", "question": "Did you have a job?", "why_ask": "Identify wage income", "retrieval_query": "wage income", "source": "Expected Source"}
    matches = [{"chunk_id": f"c{i}", "citation": f"Other Source (2025), p. {i}", "score": .1} for i in range(5)]
    passages = {f"c{i}": {"text": f"Wages are income. Details {i}.", "embed_text": f"Income\nWages are income. Details {i}."} for i in range(5)}
    return build_case(question, {"matches": matches}, passages)


def judgment(case, score=2):
    return {"assessments": [{"passage_id": p["passage_id"], "score": score, "reason": "Discusses wage income", "evidence_quote": "Wages are income." if score else ""} for p in case["mapping"]]}


def test_judge_payload_hides_gold_source_rank_scores_and_citations():
    case = example()
    assert set(case["payload"]) == {"tax_year", "question", "retrieval_query", "interview_purpose", "passages"}
    assert all(set(p) == {"passage_id", "heading", "body"} for p in case["payload"]["passages"])
    assert "Expected Source" not in json.dumps(case["payload"])
    assert "Other Source" not in json.dumps(case["payload"])
    assert case["payload"] == example()["payload"]
    assert sorted(p["original_rank"] for p in case["mapping"]) == [1, 2, 3, 4, 5]
    assert all(p["heading"] == "Income" for p in case["mapping"])


def test_cross_source_passages_can_be_directly_relevant():
    case = example()
    grades = validate_judgment(case, judgment(case))
    row = score_case(case, grades)
    assert not row["source_hit_at_5"]
    assert row["direct_hit_at_5"]
    assert row["direct_precision_at_5"] == 1
    assert row["direct_relevant_other_source"]
    assert [g["rank"] for g in grades] == [1, 2, 3, 4, 5]


def test_topical_passages_do_not_count_as_direct_hits():
    case = example()
    row = score_case(case, validate_judgment(case, judgment(case, 1)))
    assert not row["direct_hit_at_5"]
    assert row["direct_precision_at_5"] == 0
    assert row["mean_grade"] == 1


def test_mrr_uses_original_retrieval_rank_after_shuffled_judging():
    case = example()
    graded = judgment(case, 0)
    target = next(p["passage_id"] for p in case["mapping"] if p["original_rank"] == 4)
    grade = next(g for g in graded["assessments"] if g["passage_id"] == target)
    grade.update(score=2, evidence_quote="Wages are income.")
    row = score_case(case, validate_judgment(case, graded))
    assert row["direct_hit_at_5"] and not row["direct_hit_at_3"]
    assert row["direct_mrr_at_5"] == .25
    assert row["direct_precision_at_5"] == .2


@pytest.mark.parametrize("kind", ["missing", "duplicate", "invalid_score", "bool_score"])
def test_invalid_or_ungrounded_judgments_are_rejected(kind):
    case = example()
    value = judgment(case)
    grade = value["assessments"][0]
    if kind == "missing":
        value["assessments"].pop()
    elif kind == "duplicate":
        value["assessments"][-1] = dict(grade)
    elif kind == "invalid_score":
        grade["score"] = 3
    elif kind == "bool_score":
        grade["score"] = True
    elif kind == "fabricated_quote":
        grade["evidence_quote"] = "A rule not present in the body"
    elif kind == "missing_quote":
        grade["evidence_quote"] = ""
    else:
        grade["score"] = 0
    with pytest.raises(ValueError):
        validate_judgment(case, value)


def test_source_confusion_and_passage_precision_are_reported_separately():
    case = example()
    base = score_case(case, validate_judgment(case, judgment(case)))
    rows = [{**base, "id": f"q{i}", "source_hit_at_5": i < 85} for i in range(104)]
    summary = summarize(rows)
    assert summary["direct_hits_at_5"] == 104
    assert summary["passage_grade_counts"] == {"0": 0, "1": 0, "2": 520}
    assert summary["source_vs_relevance"]["source_hit_false_direct_hit_true"]["count"] == 19
    assert summary["direct_precision_at_5"] == 1
    with pytest.raises(ValueError, match="one complete"):
        summarize(rows[:-1])


def test_judge_uses_one_structured_api_call_with_only_blinded_payload():
    case = example()
    client = Mock()
    response = SimpleNamespace(id="test-response", model="test-model", system_fingerprint="test", usage=None,
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(judgment(case)), refusal=None))])
    client.chat.completions.create.return_value = response
    outcome = judge_case(client, "test-model", case)
    assert outcome["status"] == "ok"
    client.chat.completions.create.assert_called_once()
    assert json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"]) == case["payload"]
    assert client.chat.completions.create.call_args.kwargs["response_format"]["json_schema"]["strict"]


def test_failed_calls_are_not_silently_scored_zero_or_expose_exception_secrets():
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("sensitive detail")
    outcome = judge_case(client, "test-model", example())
    assert outcome["status"] == "error" and "result" not in outcome
    assert "sensitive detail" not in json.dumps(outcome)


def test_completed_relevance_round_refuses_more_model_calls(tmp_path, monkeypatch):
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["run_passage_relevance", "--env-file", str(tmp_path / "missing.env"), "--out", str(tmp_path)])
    with patch("eval.run_passage_relevance.OpenAI") as client:
        with pytest.raises(ValueError, match="cannot be rerun"):
            main()
    client.assert_not_called()


def test_unverified_quotes_flag_review_without_overwriting_model_relevance():
    case = example()
    value = judgment(case)
    value["assessments"][0]["evidence_quote"] = "A sentence not present in the body"
    grades = validate_judgment(case, value)
    flagged = next(g for g in grades if g["passage_id"] == value["assessments"][0]["passage_id"])
    assert flagged["score"] == 2
    assert flagged["quote_status"] == "unverified" and not flagged["quote_verified"]
    assert score_case(case, grades)["unverified_positive_quotes"] == 1


def test_quote_audit_checks_case_and_ordered_ellipsis_without_accepting_fabrications():
    assert quote_status("wages are income.", "Wages are income.") == "normalized_verbatim"
    assert quote_status("Wages...income.", "Wages are taxable income.") == "ellipsized_verbatim"
    assert quote_status("income...Wages", "Wages are taxable income.") == "unverified"
    assert quote_status("Wages are not income.", "Wages are taxable income.") == "unverified"
