"""Interview service + model normalizer, without FastAPI, DB, Ollama or OpenAI."""
import sqlite3

import pytest

from src.app.interviews import InterviewClosed, InterviewNotFound, InterviewService, list_playbooks
from src.db.interview_store import InterviewStore
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


def test_real_search_receives_question_context_and_retries_fallback(monkeypatch):
    calls = []

    def search(query, *, question):
        calls.append((query, question))
        return {"matches": [], "rerank_fallback": len(calls) == 1}

    monkeypatch.setattr("src.workflow.engine._default_search", search)
    service = InterviewService()
    started = service.start("ca_w2_side_gig")
    interview_id = started["state"]["interview_id"]
    service.answer(interview_id, "divorced")
    service.answer(interview_id, "divorced")
    assert len(calls) == 2
    query, question = calls[0]
    assert query == question["retrieval_query"]
    assert question["question"] == started["output"]["question"]
    assert question["why_ask"] == started["output"]["why_ask"]


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


def test_intake_with_priority_from_the_service():
    """The service lists the intake with its sections and starts it with a priority."""
    intake = [p for p in list_playbooks() if p["id"] == "intake"][0]
    assert any(s["id"] == "side_gig" for s in intake["sections"])
    svc = InterviewService(search=no_search)
    r = svc.start("intake", ["side_gig"])
    assert r["output"]["fact"] == "has_self_employment"
    assert r["state"]["priority"] == ["side_gig"]
    with pytest.raises(ValueError):
        svc.start("intake", ["bogus"])



def test_saved_interview_resumes_after_service_restart(tmp_path):
    database = tmp_path / "interviews.sqlite3"
    owner = "original-browser"
    first = InterviewService(search=no_search, store=InterviewStore(path=database))
    initial = first.start("ca_w2_side_gig", owner=owner)
    iid = initial["state"]["interview_id"]
    first.answer(iid, "single", initial["output"]["question_id"], owner=owner)
    first.start("ca_retiree", owner=owner)

    restarted = InterviewService(search=no_search, store=InterviewStore(path=database))
    rows = restarted.list(owner)
    assert len(rows) == 2
    assert rows[-1]["interview_id"] == iid
    assert set(rows[-1]) == {"interview_id", "status", "created_at", "updated_at", "title"}
    assert restarted.list("another-browser") == []
    with pytest.raises(InterviewNotFound):
        restarted.get(iid, owner="another-browser")

    saved = restarted.get(iid, owner=owner)
    assert saved["turns"] == [{"question": initial["output"]["question"], "answer": "single"}]
    assert saved["output"]["fact"] == "has_side_income"
    restarted.answer(iid, "yes", saved["output"]["question_id"], owner=owner)
    restarted.answer(iid, "12000", owner=owner)
    restarted.decide(iid, approve=True, note="Reviewed", owner=owner)

    final = first.get(iid, owner=owner)
    assert first.list(owner)[0]["interview_id"] == iid
    assert final["output"]["type"] == "summary"
    assert final["output"]["status"] == "approved"
    assert final["history"][-1] == {"decision": "approved", "note": "Reviewed"}


def test_saved_attachments_can_be_kept_or_removed_after_restart(tmp_path, monkeypatch):
    database = tmp_path / "interviews.sqlite3"
    first = InterviewService(search=no_search, document_chat=lambda: None, store=InterviewStore(path=database))
    monkeypatch.setattr(
        "src.app.interviews.extract_document_facts",
        lambda *args: [{"name": "intake.txt", "facts": {"filing_status": "single"}}],
    )
    staged = first.stage_documents("ca_w2_side_gig", [("intake.txt", b"")], owner="browser")
    doc_id = staged["files"][0]["id"]
    result = first.start("ca_w2_side_gig", document_ids=[doc_id], owner="browser")
    iid = result["state"]["interview_id"]

    restarted = InterviewService(search=no_search, store=InterviewStore(path=database))
    assert restarted.get(iid, owner="browser")["documents"] == [{"id": doc_id, "name": "intake.txt", "count": 1}]
    assert restarted.sync_documents(iid, [doc_id], owner="browser")["output"]["fact"] == "has_side_income"
    assert restarted.sync_documents(iid, [], owner="browser")["output"]["fact"] == "filing_status"
    assert restarted.get(iid, owner="browser")["documents"] == []


def test_store_rejects_a_stale_write(tmp_path):
    database = tmp_path / "interviews.sqlite3"
    store = InterviewStore(path=database)
    svc = InterviewService(search=no_search, store=store)
    result = svc.start("ca_w2_side_gig")
    iid = result["state"]["interview_id"]
    stale = InterviewStore(path=database).get("", iid)
    svc.answer(iid, "single")
    assert store.save("", stale) is False
    assert svc.get(iid)["state"]["facts"] == {"filing_status": "single"}


def test_existing_database_migrates_and_saves_interview_names(tmp_path):
    database = tmp_path / "interviews.sqlite3"
    original = InterviewService(search=no_search, store=InterviewStore(path=database))
    initial = original.start("ca_w2_side_gig", owner="browser")
    iid = initial["state"]["interview_id"]
    connection = sqlite3.connect(database)
    try:
        with connection:
            connection.execute("ALTER TABLE guided_interviews DROP COLUMN title")
    finally:
        connection.close()

    migrated = InterviewService(search=no_search, store=InterviewStore(path=database))
    assert migrated.list("browser")[0]["title"] is None
    assert migrated.get(iid, owner="browser")["output"] == initial["output"]
    migrated.rename(iid, "Follow-up", owner="browser")
    migrated.answer(iid, "single", owner="browser")

    reopened = InterviewService(search=no_search, store=InterviewStore(path=database))
    assert reopened.list("browser")[0]["title"] == "Follow-up"
    assert reopened.get(iid, owner="browser")["output"]["fact"] == "has_side_income"


def test_deleted_interview_stays_deleted_after_restart_and_stale_write(tmp_path):
    database = tmp_path / "interviews.sqlite3"
    store = InterviewStore(path=database)
    owner = "browser"
    svc = InterviewService(search=no_search, store=store)
    result = svc.start("ca_w2_side_gig", owner=owner)
    iid = result["state"]["interview_id"]
    stale = store.get(owner, iid)

    svc.delete(iid, owner=owner)
    restarted_store = InterviewStore(path=database)
    restarted = InterviewService(search=no_search, store=restarted_store)
    assert restarted_store.save(owner, stale) is False
    assert restarted.list(owner) == []
    with pytest.raises(InterviewNotFound):
        restarted.get(iid, owner=owner)
