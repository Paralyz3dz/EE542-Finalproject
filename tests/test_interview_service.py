"""Interview service + model normalizer, without FastAPI, DB, Ollama or OpenAI."""
import pytest

from src.app.interviews import InterviewClosed, InterviewNotFound, InterviewService, list_playbooks
from src.llm.normalize import make_normalizer


def no_search(query):
    """Stub search: one fake Pub 17 passage for any query."""
    return {"matches": [{"citation": "Pub 17 (stub)", "text": "passage for " + query}]}


def test_playbooks_are_listed():
    """All playbook folders are offered as scenarios."""
    ids = [p["id"] for p in list_playbooks()]
    assert {"ca_w2_side_gig", "ca_w2_family", "ca_retiree", "ca_rental"} <= set(ids)


def test_full_interview_then_preparer_approves():
    """Service: start, invalid answer, answers to the summary; closed interviews reject answers; the preparer can decide once."""
    svc = InterviewService(search=no_search)
    r = svc.start("ca_w2_side_gig")
    iid = r["state"]["interview_id"]
    q = r["output"]
    assert q["type"] == "question" and q["input_type"] == "enum" and "single" in q["options"]
    assert q["step_number"] == 1 and q["citations"][0]["citation"] == "Pub 17 (stub)"

    assert svc.answer(iid, "divorced")["output"]["type"] == "error"
    assert svc.answer(iid, "single")["output"]["input_type"] == "bool"
    svc.answer(iid, "yes")
    r = svc.answer(iid, "12,000")
    assert r["output"]["type"] == "summary"
    assert r["state"]["status"] == "waiting_for_preparer"

    with pytest.raises(InterviewClosed):
        svc.answer(iid, "yes")
    assert svc.decide(iid, approve=True, note="ok")["state"]["status"] == "approved"
    with pytest.raises(InterviewClosed):
        svc.decide(iid, approve=False)
    assert svc.get(iid)["history"][-1] == {"decision": "approved", "note": "ok"}


def test_unknown_ids_and_scenarios():
    """Unknown interview ids and unknown scenarios raise errors."""
    svc = InterviewService(search=no_search)
    with pytest.raises(InterviewNotFound):
        svc.answer("nope", "yes")
    with pytest.raises(ValueError):
        svc.start("not_a_playbook")


def test_search_results_are_cached():
    """Showing the same question again does not run retrieval a second time."""
    calls = []
    def counting(q):
        """Stub search that records each query it receives."""
        calls.append(q)
        return {"matches": []}
    svc = InterviewService(search=counting)
    iid = svc.start("ca_w2_side_gig")["state"]["interview_id"]
    svc.answer(iid, "divorced")
    assert len(calls) == 1


def fake_chat(value, reason="ok"):
    """Build a stub model that always answers with this value and reason."""
    return lambda system, user: {"value": value, "reason": reason}


def test_free_text_is_read_by_model_and_validated():
    """A free-text note is turned into a valid value by the model."""
    svc = InterviewService(search=no_search, normalizer=make_normalizer(chat=fake_chat("married_joint")))
    iid = svc.start("ca_w2_side_gig")["state"]["interview_id"]
    r = svc.answer(iid, "she's married and they file together")
    assert r["state"]["facts"]["filing_status"] == "married_joint"
    assert r["state"]["last_parsed_by"] == "model"


def test_exact_answers_never_call_the_model():
    """Exact answers are parsed by rules; the model is never called."""
    def must_not_run(system, user):
        """Stub model that fails the test if it is ever called."""
        raise AssertionError("model called for an exact answer")
    svc = InterviewService(search=no_search, normalizer=make_normalizer(chat=must_not_run))
    iid = svc.start("ca_w2_side_gig")["state"]["interview_id"]
    assert svc.answer(iid, "single")["state"]["last_parsed_by"] == "rules"


def test_invalid_model_output_is_rejected():
    """A model proposal outside the allowed options is rejected; no fact is written."""
    svc = InterviewService(search=no_search, normalizer=make_normalizer(chat=fake_chat("divorced")))
    iid = svc.start("ca_w2_side_gig")["state"]["interview_id"]
    r = svc.answer(iid, "it's complicated")
    assert r["output"]["type"] == "error"
    assert r["state"]["facts"] == {}


def test_model_says_unclear():
    """When the model says the note is unclear, the assistant sees its reason as an error."""
    svc = InterviewService(search=no_search, normalizer=make_normalizer(chat=fake_chat(None, "note is about rent")))
    iid = svc.start("ca_w2_side_gig")["state"]["interview_id"]
    r = svc.answer(iid, "they pay rent")
    assert r["output"]["type"] == "error" and "note is about rent" in r["output"]["message"]


def test_no_api_key_means_no_normalizer(monkeypatch):
    """Without an API key there is no normalizer (rules only)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("src.llm.normalize.is_configured", lambda: False)
    assert make_normalizer() is None
