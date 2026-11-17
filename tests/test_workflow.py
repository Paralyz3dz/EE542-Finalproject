"""Workflow engine and playbooks: conditions, question order, review, parsing, step, references."""
import pytest

from src.workflow.engine import load_scenario


def test_scenario_files_load():
    """Both playbook files load, and every question refers to a defined fact."""
    definition, questions = load_scenario("ca_w2_side_gig")
    fact_names = {f["name"] for f in definition["facts"]}
    assert "has_side_income" in fact_names
    assert all(q["fact"] in fact_names for q in questions)


def test_check_conditions():
    """check(): no conditions -> True; matching value -> True; unknown fact -> False."""
    from src.workflow.conditions import check
    assert check(None, {}) is True
    assert check([{"fact": "has_side_income", "equals": True}], {"has_side_income": True}) is True
    assert check([{"fact": "has_side_income", "equals": True}], {"has_side_income": None}) is False


def test_first_question_is_filing_status():
    """A new interview starts with the first fact in scenario.json."""
    from src.workflow.engine import next_question, start_interview
    state = start_interview("ca_w2_side_gig")
    assert next_question(state)["fact"] == "filing_status"


def test_side_income_asked_only_after_yes():
    """A follow-up question appears only after its ask_when condition holds."""
    from src.workflow.engine import next_question, start_interview
    state = start_interview("ca_w2_side_gig")
    state.facts.update({"filing_status": "single", "age_65_or_older": False,
                        "ca_resident_full_year": True, "w2_wages": 60000,
                        "has_side_income": True})
    assert next_question(state)["fact"] == "side_income"


def test_review_adds_escalation_flag_once():
    """A matching escalation rule adds one flag with a reason, even if review runs twice."""
    from src.workflow.engine import review, start_interview
    state = start_interview("ca_w2_side_gig")
    state.facts.update({"has_side_income": True, "has_employees": True})
    review(state)
    review(state)
    assert [f["rule"] for f in state.flags] == ["has_employees"]
    assert state.flags[0]["reason"]


def test_review_no_flags_for_simple_client():
    """A simple client triggers no flags, and a 'no' answer needs no documents."""
    from src.workflow.engine import review, start_interview
    state = start_interview("ca_w2_side_gig")
    state.facts.update({"filing_status": "single", "ca_resident_full_year": True,
                        "has_side_income": False})
    review(state)
    assert state.flags == []
    assert state.missing_docs == []


def test_review_collects_docs_without_duplicates():
    """Required documents of answered questions are collected once."""
    from src.workflow.engine import review, start_interview
    state = start_interview("ca_w2_side_gig")
    state.facts.update({"has_side_income": True, "side_income": 12000})
    review(state)
    review(state)
    assert state.missing_docs == ["1099-NEC", "1099-K", "income records"]


def test_parse_value_by_type():
    """parse_value converts bool / money / enum / list answers and rejects invalid ones."""
    from src.workflow.engine import parse_value
    assert parse_value({"name": "x", "type": "bool"}, " Yes ") is True
    assert parse_value({"name": "x", "type": "bool"}, "no") is False
    assert parse_value({"name": "x", "type": "money"}, "$12,000") == 12000
    assert parse_value({"name": "x", "type": "money"}, "99.5") == 99.5
    assert parse_value({"name": "x", "type": "enum", "options": ["single", "head_of_household"]},
                       "Head of household") == "head_of_household"
    assert parse_value({"name": "x", "type": "list"}, "1099-NEC, 1099-K") == ["1099-NEC", "1099-K"]
    for fact, bad in [({"name": "x", "type": "bool"}, "maybe"),
                      ({"name": "x", "type": "money"}, "a lot"),
                      ({"name": "x", "type": "money"}, "-5"),
                      ({"name": "x", "type": "enum", "options": ["single"]}, "divorced")]:
        try:
            parse_value(fact, bad)
            assert False, f"{bad!r} should be rejected"
        except ValueError:
            pass


def test_step_runs_a_full_interview():
    """step(): first question, an invalid answer re-shows the question, then on to the summary."""
    from src.workflow.engine import start_interview, step
    stub = lambda q: {"matches": [{"citation": "Pub 17 (stub)", "text": "passage"}]}
    state = start_interview("ca_w2_side_gig")
    state, out = step(state, search=stub)
    assert out["type"] == "question" and out["fact"] == "filing_status"
    assert out["citations"][0]["citation"] == "Pub 17 (stub)"

    state, out = step(state, "divorced", search=stub)
    assert out["type"] == "error" and out["guide"]["fact"] == "filing_status"
    assert state.facts["filing_status"] is None

    state, out = step(state, "single", search=stub)
    assert out["fact"] == "has_side_income"
    state, out = step(state, "yes", search=stub)
    assert out["fact"] == "side_income"
    state, out = step(state, "12,000", search=stub)
    assert out["type"] == "summary"
    assert out["facts"] == {"filing_status": "single", "has_side_income": True, "side_income": 12000}
    assert out["missing_docs"] == ["1099-NEC", "1099-K", "income records"]
    assert state.status == "waiting_for_preparer"
    assert state.asked == ["q_filing_status", "q_has_side_income", "q_side_income"]


def test_guide_survives_retrieval_failure():
    """If retrieval fails, the guide is still returned without citations."""
    from src.workflow.engine import build_guide, next_question, start_interview
    def broken(q):
        """Stub search that fails like a stopped database."""
        raise ConnectionError("database asleep")
    state = start_interview("ca_w2_side_gig")
    guide = build_guide(state, next_question(state), search=broken)
    assert guide["citations"] == [] and "ConnectionError" in guide["retrieval_error"]


import json as _json
from pathlib import Path as _Path

PLAYBOOK_DIR = _Path(__file__).resolve().parents[1] / "playbooks"
ALL_PLAYBOOKS = sorted(p.name for p in PLAYBOOK_DIR.iterdir() if (p / "scenario.json").exists())
if (PLAYBOOK_DIR / "modules").exists():
    ALL_PLAYBOOKS.append("intake")
SAMPLE_ANSWER = {"bool": "yes", "money": "1000", "list": "1099-NEC"}


def _validate(name):
    """Structure check for one playbook: fact types, enum options, references to facts, unique ids."""
    from src.workflow.engine import load_scenario
    definition, questions = load_scenario(name)
    names = [f["name"] for f in definition["facts"]]
    assert definition["id"] == name
    assert len(names) == len(set(names)), "duplicate fact names"
    for f in definition["facts"]:
        assert f["type"] in ("bool", "money", "enum", "list"), f
        if f["type"] == "enum":
            assert f.get("options"), f
        for c in f.get("ask_when") or []:
            assert c["fact"] in names, f"{f['name']}: ask_when uses unknown fact {c['fact']}"
    for r in definition["escalation_rules"]:
        assert r["reason"]
        for c in r["when"]:
            assert c["fact"] in names, f"rule {r['id']} uses unknown fact {c['fact']}"
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    for q in questions:
        assert q["fact"] in names, f"question {q['id']} asks unknown fact"
        for key in ("question", "why_ask", "retrieval_query", "source"):
            assert q[key], f"{q['id']} missing {key}"


def _run_all_yes(name):
    """Run a whole interview with a sample answer per fact type; return (state, summary)."""
    from src.workflow.engine import load_scenario, start_interview, step
    definition, _ = load_scenario(name)
    kinds = {f["name"]: f for f in definition["facts"]}
    stub = lambda q: {"matches": []}
    state = start_interview(name)
    state, out = step(state, search=stub)
    for _ in range(300):
        if out["type"] == "summary":
            return state, out
        f = kinds[out["fact"]]
        answer = f["options"][0] if f["type"] == "enum" else SAMPLE_ANSWER[f["type"]]
        state, out = step(state, answer, search=stub)
    raise AssertionError(f"{name}: interview did not finish")


def test_every_playbook_is_valid():
    """Every playbook folder passes the structure check."""
    assert len(ALL_PLAYBOOKS) >= 4
    for name in ALL_PLAYBOOKS:
        _validate(name)


def test_every_playbook_runs_to_summary():
    """Every playbook can be run to a summary; the new playbooks cover all required facts (ca_w2_side_gig still has gaps)."""
    for name in ALL_PLAYBOOKS:
        state, out = _run_all_yes(name)
        assert out["status"] == "waiting_for_preparer"
        if name != "ca_w2_side_gig":
            assert out["not_yet_covered"] == [], f"{name}: {out['not_yet_covered']}"


def test_retiree_early_distribution_is_escalated():
    """Retiree: a distribution before 59 1/2 raises only the early_distribution flag."""
    from src.workflow.engine import start_interview, step
    stub = lambda q: {"matches": []}
    state = start_interview("ca_retiree")
    state, out = step(state, search=stub)
    answers = {"filing_status": "single", "age_65_or_older": "no", "ca_resident_full_year": "yes",
               "receives_social_security": "no", "has_retirement_distribution": "yes",
               "retirement_distributions": "20000", "under_59_half_at_distribution": "yes",
               "did_rollover": "no", "still_working": "no", "made_estimated_payments": "no"}
    while out["type"] != "summary":
        state, out = step(state, answers[out["fact"]], search=stub)
    assert [f["rule"] for f in out["flags"]] == ["early_distribution"]
    assert "1099-R" in out["missing_docs"]


PUB17 = lambda q: {"matches": [{"citation": "IRS Pub 17 (2025), 'Filing Status', p. 23", "text": "passage"}]}


def test_references_marked_by_what_retrieval_found():
    """Playbook references are marked retrieved=True when that document appears in the passages, else False."""
    from src.workflow.engine import _doc_key, build_guide, next_question, start_interview
    assert _doc_key("IRS Pub 17, Chapter 2") == _doc_key("IRS Pub 17 (2025), 'Filing Status', p. 23")
    state = start_interview("ca_w2_side_gig")
    guide = build_guide(state, next_question(state), search=PUB17)
    assert guide["references"] == [{"reference": "IRS Pub 17, Chapter 2", "retrieved": True}]

    state.facts["filing_status"] = "single"
    guide = build_guide(state, next_question(state), search=PUB17)
    assert [r["retrieved"] for r in guide["references"]] == [False, False]


def test_summary_only_claims_retrieved_citations():
    """The summary lists only citations retrieval returned; references never returned are reported as knowledge-base gaps."""
    from src.workflow.engine import start_interview, step
    state = start_interview("ca_w2_side_gig")
    state, out = step(state, search=PUB17)
    for answer in ["single", "yes", "12000"]:
        state, out = step(state, answer, search=PUB17)
    assert out["type"] == "summary"
    assert out["citations_retrieved"] == ["IRS Pub 17 (2025), 'Filing Status', p. 23"]
    assert "IRS Pub 17, Chapter 2" in out["references_to_verify"]
    assert "IRS Pub 17, Chapter 2" not in out["references_not_in_kb"]
    assert "Schedule SE instructions" in out["references_not_in_kb"]
    assert "sources" not in out


def test_references_unchecked_when_retrieval_is_down():
    """With retrieval down, references are 'unchecked', not 'missing'."""
    from src.workflow.engine import start_interview, step
    def down(q):
        """Stub search that fails like a stopped database."""
        raise ConnectionError("db asleep")
    state = start_interview("ca_w2_side_gig")
    state, out = step(state, search=down)
    assert out["references"][0]["retrieved"] is None
    for answer in ["single", "no"]:
        state, out = step(state, answer, search=down)
    assert out["citations_retrieved"] == [] and out["references_not_in_kb"] == []
    assert "IRS Pub 17, Chapter 2" in out["references_unchecked"]


def test_intake_composes_modules_with_base_first():
    """The modular intake starts with base_intake and lists base parts and topic modules as sections."""
    from src.workflow.engine import load_scenario, sections
    definition, questions = load_scenario("intake")
    assert definition["facts"][0]["module"] == "base_intake"
    ids = [s["id"] for s in sections(definition)]
    assert ids[:5] == ["about_you", "household", "income", "expenses", "events"]
    assert {"side_gig", "family", "retiree", "rental"} <= set(ids)
    assert len(questions) >= len([f for f in definition["facts"] if f["module"] == "base_intake"])


def test_no_priority_follows_playbook_order():
    """Without a priority the first question is the first base_intake fact."""
    from src.workflow.engine import load_scenario, next_question, start_interview
    definition, _ = load_scenario("intake")
    assert next_question(start_interview("intake"))["fact"] == definition["facts"][0]["name"]


def test_priority_section_is_asked_first():
    """A prioritized base part comes first; the rest keeps its order afterwards."""
    from src.workflow.engine import next_question, start_interview, step
    state = start_interview("intake", ["expenses"])
    q = next_question(state)
    assert q["fact"] == "paid_mortgage_interest"
    stub = lambda q: {"matches": []}
    state, out = step(state, search=stub)
    assert out["section"] == "Expenses"


def test_priority_module_starts_with_its_gate_question():
    """Choosing a topic module asks the base question that switches it on first."""
    from src.workflow.engine import next_question, start_interview
    for module, gate in [("side_gig", "has_self_employment"), ("family", "has_dependents"),
                         ("retiree", "has_retirement_income"), ("rental", "owns_rental")]:
        assert next_question(start_interview("intake", [module]))["fact"] == gate


def test_module_facts_wait_for_their_gate():
    """A topic module's facts are not askable until its activate_when holds."""
    from src.workflow.conditions import check
    from src.workflow.engine import load_scenario
    definition, _ = load_scenario("intake")
    for f in definition["facts"]:
        if f["module"] != "base_intake":
            assert not check(f.get("ask_when"), {}), f["name"]


def test_unknown_priority_is_rejected():
    """Priority ids must be real sections."""
    from src.workflow.engine import start_interview
    try:
        start_interview("intake", ["not_a_section"])
        assert False
    except ValueError:
        pass


def test_reminders_appear_in_summary():
    """A reminder whose condition holds is listed in the summary."""
    from src.workflow.engine import start_interview, summarize
    state = start_interview("intake")
    state.facts["made_donations"] = True
    assert any("donation" in r for r in summarize(state)["reminders"])



def test_modules_are_consistent():
    """Every module fact has a question; every condition, trigger and reminder names a real fact."""
    from src.workflow.engine import load_scenario
    definition, questions = load_scenario("intake")
    names = [f["name"] for f in definition["facts"]]
    assert len(names) == len(set(names)), "a fact name is used twice across modules"
    asked = {q["fact"] for q in questions}
    for f in definition["facts"]:
        assert f["name"] in asked, f"{f['module']}.{f['name']} has no question"
        if f["type"] == "enum":
            assert f.get("options"), f["name"]
        for c in f.get("ask_when") or []:
            assert c["fact"] in names, f"{f['name']}: unknown fact {c['fact']}"
    for group in ("escalation_rules", "reminders"):
        for r in definition[group]:
            for c in r["when"]:
                assert c["fact"] in names, f"{group} {r['id']}: unknown fact {c['fact']}"
    for t in definition["triggers"]:
        assert t["keywords"]
        for fact in t["ask_first"]:
            assert fact in names, f"trigger {t['id']}: unknown fact {fact}"
    for q in questions:
        for key in ("question", "why_ask", "source", "origin", "status"):
            assert q.get(key), f"{q['id']} missing {key}"


def test_side_gig_flow_with_priority():
    """Assistant unsure about self-employment: gate first, then the side-gig questions."""
    from src.workflow.engine import start_interview, step
    stub = lambda q: {"matches": []}
    state = start_interview("intake", ["side_gig"])
    state, out = step(state, search=stub)
    assert out["fact"] == "has_self_employment"
    state, out = step(state, "yes", search=stub)
    assert out["section"] == "Self-employment and gig income"
    assert out["fact"] == "does_rideshare_delivery"


def test_side_gig_escalations_and_reminders():
    """Hobby selling, actual car costs and no records produce flags and reminders."""
    from src.workflow.engine import review, start_interview, summarize
    state = start_interview("intake")
    state.facts.update({"has_self_employment": True, "sells_online": True, "selling_for_profit": False,
                        "uses_car_for_business": True, "car_expense_method": "actual_costs",
                        "record_keeping": "no_records", "made_estimated_payments": False})
    review(state)
    rules = {f["rule"] for f in state.flags}
    assert {"esc_selling_for_profit", "esc_car_expense_method_actual_costs"} <= rules
    reminders = " ".join(summarize(state)["reminders"])
    assert "mileage log" in reminders and "estimated tax" in reminders and "keep from now on" in reminders


def test_background_triggers_reorder_in_mention_order():
    """Trigger words in the background note move their follow-ups up, in the order mentioned."""
    from src.workflow.engine import start_interview, step
    stub = lambda q: {"matches": []}
    state = start_interview("intake", [], "two kids at home, and she travels for her Etsy shop")
    assert "side_gig" in state.priority
    state, out = step(state, search=stub)
    assert out["fact"] == "has_dependents" and out["triggered_by"] == "kids"
    state, out = step(state, "yes", search=stub)
    assert out["fact"] == "has_child_under_17"


def test_trigger_word_in_an_answer():
    """A trigger word typed in an answer note asks its follow-up next."""
    from src.workflow.engine import start_interview, step
    stub = lambda q: {"matches": []}
    from src.llm.normalize import make_normalizer
    reader = make_normalizer(chat=lambda system, user: {"value": "yes", "reason": "said yes"})
    state = start_interview("intake")
    state, out = step(state, search=stub)
    state, out = step(state, "yes, and he gave a lot to charity this year", search=stub, normalizer=reader)
    assert out["fact"] == "made_donations" and out["triggered_by"] == "charity"


def test_module_gates_open_their_modules():
    """Yes to a base gate question opens that module; no keeps it closed."""
    from src.workflow.engine import next_question, start_interview
    for gate, module in [("has_dependents", "family"), ("has_retirement_income", "retiree"), ("owns_rental", "rental")]:
        state = start_interview("intake", [module])
        state.facts[gate] = True
        assert next_question(state)["module"] == module
        state.facts[gate] = False
        assert next_question(state)["module"] == "base_intake"


def test_button_answers_do_not_trigger():
    """An exact answer such as the option 'business_miles' never fires the 'miles' trigger."""
    from src.workflow.engine import start_interview, step
    stub = lambda q: {"matches": []}
    state = start_interview("intake", ["side_gig"])
    state, out = step(state, search=stub)
    for answer in ["yes", "no", "no", "no"]:
        state, out = step(state, answer, search=stub)
    while out.get("fact") != "car_expense_method":
        state, out = step(state, "yes" if out.get("input_type") == "bool" else (out.get("options") or ["1000"])[0], search=stub)
    state, out = step(state, "business_miles", search=stub)
    assert state.boosted == []
