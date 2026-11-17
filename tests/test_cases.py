"""Case replay: fictionalized IRS test cases and demo clients run through the intake, start to summary.

tests/cases/*.json are based on IRS MeF ATS test scenarios and Form 6744 (VITA test) interview notes,
moved to California with names and IDs dropped, plus two fictional clients. Each case fixes the
client's answers and what the interview must produce: which modules open and which situations go to
the preparer (the exact set of escalation flags). The official outcome is quoted in each file, so the
expected flags can be checked against what the IRS material says, without a preparer's sign-off.
"""
import json
from pathlib import Path

import pytest

from src.workflow.engine import load_scenario, start_interview, step

CASES = sorted((Path(__file__).parent / "cases").glob("*.json"))
TYPICAL = {"ca_resident_full_year": "yes", "is_us_citizen": "yes"}


def no_search(query):
    """No knowledge base in these tests."""
    return {"matches": []}


def default_answer(question: dict) -> str:
    """Typical answer for a question the case does not mention: yes to the two residency questions, no, first option, 0."""
    if question["fact"] in TYPICAL:
        return TYPICAL[question["fact"]]
    if question["input_type"] == "bool":
        return "no"
    if question["input_type"] == "enum":
        return question["options"][0]
    return "0"


def run_case(case: dict) -> tuple[dict, list[str]]:
    """Play one case. Returns (summary, facts in the order they were asked)."""
    state = start_interview("intake", case.get("priority", []), case.get("background", ""))
    state, out = step(state, search=no_search)
    asked = []
    for _ in range(300):
        if out["type"] == "summary":
            return out, asked
        assert out["type"] == "question", out
        asked.append(out["fact"])
        state, out = step(state, case["answers"].get(out["fact"], default_answer(out)), search=no_search)
    raise AssertionError("interview did not finish in 300 turns")


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_case(path):
    """The case's client opens the expected modules and raises exactly the expected escalations."""
    case = json.loads(path.read_text(encoding="utf-8"))
    expect = case["expect"]
    summary, asked = run_case(case)
    definition, _ = load_scenario("intake")
    module_of = {f["name"]: f["module"] for f in definition["facts"]}
    modules = {module_of[f] for f in asked}

    assert sorted(f["rule"] for f in summary["flags"]) == sorted(expect["flags"])
    assert asked[:len(expect.get("first_questions", []))] == expect.get("first_questions", [])
    assert set(expect.get("modules_asked", [])) <= modules
    assert not set(expect.get("modules_not_asked", [])) & modules
    for doc in expect.get("documents_contain", []):
        assert doc in summary["missing_docs"], doc
    assert summary["not_yet_covered"] == []
    assert len(asked) == len(set(asked)), "a question was asked twice"


def test_cases_only_answer_real_facts():
    """A typo in a fact name would silently fall back to the default answer."""
    definition, _ = load_scenario("intake")
    names = {f["name"] for f in definition["facts"]}
    for path in CASES:
        case = json.loads(path.read_text(encoding="utf-8"))
        assert set(case["answers"]) <= names, path.name
        assert case["source"] and case["official_outcome"], path.name
