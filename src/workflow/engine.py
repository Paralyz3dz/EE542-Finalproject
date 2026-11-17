"""
engine.py: the interview workflow (the "digital clone" loop).

One turn = step():
  1. extract_facts  - the model turns the assistant's notes into structured facts; code validates
  2. review         - code: contradictions, escalation rules, missing documents
  3. next_question  - code: first required fact that is still unknown and may be asked now
  4. build_guide    - retrieval: why to ask, how to ask, and the source text
  5. finished?      - no: show the guide to the assistant; yes: summarize -> wait for the preparer

Rules of thumb
  - Which question comes next is decided by code from the scenario, never by the model.
  - Numbers come from code or official documents, never from the model.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.workflow.conditions import check
from src.workflow.state import State

ROOT = Path(__file__).resolve().parents[2]
PLAYBOOKS = ROOT / "playbooks"
logger = logging.getLogger(__name__)

def load_scenario(scenario: str) -> tuple[dict, list[dict]]:
    """Return (scenario definition, list of questions) from playbooks/<scenario>/."""
    folder = PLAYBOOKS / scenario
    definition = json.loads((folder / "scenario.json").read_text(encoding="utf-8"))
    questions = json.loads((folder / "questions.json").read_text(encoding="utf-8"))
    return definition, questions



def start_interview(scenario: str) -> State:
    """New interview: every fact of the scenario starts as None (unknown)."""
    definition, _ = load_scenario(scenario)
    facts = {f["name"]: None for f in definition["facts"]}
    return State(scenario=scenario, facts=facts)


def next_question(state: State) -> dict | None:
    """Pure code. Return the question for the first fact that is
         required, still None, and whose ask_when conditions hold (conditions.check);
       None when nothing is left to ask.
    """
    definition, questions = load_scenario(state.scenario)
    by_fact = {q["fact"]: q for q in questions}
    for f in definition["facts"]:
        name = f["name"]
        if not f.get("required"):
            continue
        if state.facts.get(name) is not None:
            continue
        if not check(f.get("ask_when"), state.facts):
            continue
        q = by_fact.get(name)
        if q is None:
            continue
        return q
    return None


def review(state: State) -> State:
    """Pure code, run after every answer:
         - escalation_rules from scenario.json -> add to state.flags (no duplicates)
         - required_docs of answered questions -> state.missing_docs (no duplicates)
         - (later) simple contradictions, e.g. single but a spouse mentioned
    Returns the same state object, updated.
    """
    definition, questions = load_scenario(state.scenario)

    flagged = [f["rule"] for f in state.flags]
    for rule in definition["escalation_rules"]:
        hit = check(rule["when"], state.facts)
        if hit and rule["id"] not in flagged:
            state.flags.append({"rule": rule["id"], "reason": rule["reason"]})

    for q in questions:
        value = state.facts.get(q["fact"])
        if not value:
            continue
        for doc in q["required_docs"]:
            if doc not in state.missing_docs:
                state.missing_docs.append(doc)
    return state


YES = {"yes", "y", "true", "1", "是", "有", "对"}
NO = {"no", "n", "false", "0", "否", "没有", "无", "不是"}
NONE_WORDS = {"none", "no", "n/a", "无", "没有"}


def fact_definition(definition: dict, name: str) -> dict:
    """The scenario.json entry for one fact (name, type, options, ...)."""
    for f in definition["facts"]:
        if f["name"] == name:
            return f
    raise KeyError(f"Fact {name!r} is not defined in scenario {definition['id']!r}")


def parse_value(fact: dict, text: str):
    """Convert the assistant's typed answer into the fact's type, or raise ValueError
    with a message the assistant can act on. No model involved."""
    raw = text.strip()
    low = raw.lower()
    kind = fact["type"]

    if kind == "bool":
        if low in YES:
            return True
        if low in NO:
            return False
        raise ValueError(f"{fact['name']}: please answer yes or no (got {raw!r})")

    if kind == "money":
        cleaned = low.replace("$", "").replace(",", "").replace(" ", "")
        try:
            amount = float(cleaned)
        except ValueError:
            raise ValueError(f"{fact['name']}: please enter an amount such as 12000 (got {raw!r})") from None
        if amount < 0:
            raise ValueError(f"{fact['name']}: amount cannot be negative")
        return int(amount) if amount.is_integer() else round(amount, 2)

    if kind == "enum":
        key = low.replace("-", "_").replace(" ", "_")
        if key in fact["options"]:
            return key
        raise ValueError(f"{fact['name']}: choose one of {', '.join(fact['options'])} (got {raw!r})")

    if kind == "list":
        if low in NONE_WORDS:
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]

    raise ValueError(f"{fact['name']}: unknown fact type {kind!r} in scenario.json")


def extract_facts(state: State, question: dict, answer_text: str, normalizer=None) -> dict:
    """Turn the assistant's answer into {fact_name: value}, validated by the fact's type.

    1. Try the exact parser (no model): "yes", "12,000", "single" ...
    2. If that fails and a normalizer is given (an LLM, see src/llm/normalize.py),
       ask it to turn the free text ("she took about 20k out of her IRA") into a
       plain value ("20000"), then validate THAT with the same parser.
    The model only proposes; parse_value() decides. Bad model output never enters state.
    Returns {"facts": {...}, "parsed_by": "rules" | "model"}.
    """
    definition, _ = load_scenario(state.scenario)
    fact = fact_definition(definition, question["fact"])
    try:
        return {"facts": {fact["name"]: parse_value(fact, answer_text)}, "parsed_by": "rules"}
    except ValueError as rule_error:
        if normalizer is None:
            raise
        proposal = normalizer(fact, question, answer_text)    
        try:
            value = parse_value(fact, proposal)
        except ValueError:
            raise rule_error from None                        
        return {"facts": {fact["name"]: value}, "parsed_by": "model"}



def _default_search(query: str) -> dict:
    from src.rag.retriever import retrieve   
    return retrieve(query)


def _doc_key(text: str) -> str:
    """'IRS Pub 17, Chapter 2' / "IRS Pub 17 (2025), 'Filing status', p. 23" -> 'irs pub 17'."""
    head = re.sub(r"\s*\(\d{4}\)", "", text).split(",")[0]
    return " ".join(head.lower().split())


def check_references(source: str, citations: list[dict], retrieval_ok: bool) -> list[dict]:
    """Playbook references (hand-written "where this rule lives") vs what retrieval actually
    returned. retrieved: True = this document appeared in the passages, False = it did not
    (likely not in the knowledge base yet), None = retrieval was down, so unknown."""
    found = {_doc_key(c["citation"]) for c in citations if c.get("citation")}
    refs = []
    for part in source.split(";"):
        ref = part.strip()
        if ref:
            refs.append({"reference": ref,
                         "retrieved": (_doc_key(ref) in found) if retrieval_ok else None})
    return refs


def build_guide(state: State, question: dict, search=None, top_n: int = 3) -> dict:
    """What the assistant sees for the next question: the playbook text plus the
    official source passages. If retrieval is down, the guide still works without citations.

    Two kinds of "source", kept apart on purpose:
      citations  - passages retrieval really returned (verified text from the knowledge base)
      references - the playbook's hand-written pointer to where the rule lives (to verify)
    Both are recorded in state.evidence so the summary only claims what was retrieved."""
    search = search or _default_search
    definition, _ = load_scenario(state.scenario)
    fact = fact_definition(definition, question["fact"])
    citations, retrieval_error = [], None
    try:
        result = search(question["retrieval_query"])
        for m in result.get("matches", [])[:top_n]:
            citations.append({"citation": m.get("citation"), "text": (m.get("text") or "")[:600]})
    except Exception as exc:                  
        retrieval_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Retrieval failed for %r: %s", question["retrieval_query"], retrieval_error)

    references = check_references(question["source"], citations, retrieval_error is None)
    state.evidence[question["id"]] = {"citations": [c["citation"] for c in citations],
                                      "references": references}

    return {
        "type": "question",
        "question_id": question["id"],
        "fact": question["fact"],
        "input_type": fact["type"],            
        "options": fact.get("options", []),   
        "step_number": len(state.asked),       
        "question": question["question"],
        "why_ask": question["why_ask"],
        "required_docs": question["required_docs"],
        "references": references,             
        "citations": citations,               
        "retrieval_error": retrieval_error,
        "flags": list(state.flags),           
    }


def summarize(state: State) -> dict:
    """Summary for the preparer to approve. Plain template now; a model-written
    narrative can be added later on top of these same fields."""
    definition, questions = load_scenario(state.scenario)
    by_fact = {q["fact"]: q for q in questions}
    answered = {k: v for k, v in state.facts.items() if v is not None}
    unanswered = [f["name"] for f in definition["facts"]
                  if f.get("required") and state.facts.get(f["name"]) is None
                  and check(f.get("ask_when"), state.facts)]

   
    asked_answered = [by_fact[k]["id"] for k in answered if k in by_fact]
    retrieved, verify, missing, unchecked = set(), set(), set(), set()
    for qid in asked_answered:
        ev = state.evidence.get(qid, {})
        retrieved.update(c for c in ev.get("citations", []) if c)
        for r in ev.get("references", []):
            verify.add(r["reference"])
            if r["retrieved"] is False:
                missing.add(r["reference"])
            elif r["retrieved"] is None:
                unchecked.add(r["reference"])

    state.status = "waiting_for_preparer"
    return {
        "type": "summary",
        "interview_id": state.interview_id,
        "scenario": definition["title"],
        "facts": answered,
        "not_yet_covered": unanswered,       
        "flags": list(state.flags),
        "needs_preparer_attention": bool(state.flags),
        "missing_docs": list(state.missing_docs),
        "citations_retrieved": sorted(retrieved),      
        "references_to_verify": sorted(verify),        
        "references_not_in_kb": sorted(missing),       
        "references_unchecked": sorted(unchecked),     
        "turns": len(state.history),
        "status": state.status,
    }


def _question_by_id(state: State, question_id: str) -> dict:
    _, questions = load_scenario(state.scenario)
    for q in questions:
        if q["id"] == question_id:
            return q
    raise KeyError(question_id)


def _show(state: State, question: dict, search) -> dict:
    if not state.asked or state.asked[-1] != question["id"]:
        state.asked.append(question["id"])
    return build_guide(state, question, search=search)


def step(state: State, answer_text: str | None = None, search=None,
         normalizer=None) -> tuple[State, dict]:
    """One turn. Returns (state, output). output["type"] is
         "question" - guide for the next question,
         "error"    - the answer could not be parsed; the same question is shown again,
         "summary"  - interview finished, waiting for the preparer.
    Call step(state) with no answer to get the first question.
    search / normalizer are injected (real RAG / LLM in the app, stubs in tests).
    """
    if state.status != "in_progress":
        raise RuntimeError(f"Interview is {state.status}; nothing to step.")

    if answer_text is None or not state.asked:
        question = next_question(state)
        return (state, summarize(state)) if question is None else (state, _show(state, question, search))

    current = _question_by_id(state, state.asked[-1])
    try:
        extracted = extract_facts(state, current, answer_text, normalizer=normalizer)
    except ValueError as exc:
        guide = build_guide(state, current, search=search)
        return state, {"type": "error", "message": str(exc), "guide": guide}

    new_facts = extracted["facts"]
    state.facts.update(new_facts)
    state.history.append({"question_id": current["id"], "answer_text": answer_text,
                          "facts": new_facts, "parsed_by": extracted["parsed_by"]})
    review(state)

    question = next_question(state)
    if question is None:
        return state, summarize(state)
    return state, _show(state, question, search)
