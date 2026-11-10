"""
engine.py: the guided-interview workflow (the preparer's "digital clone").

Data in:
    playbooks/<scenario>/scenario.json   facts to collect (name, type, required, ask_when)
                                         and escalation rules (id, when, reason)
    playbooks/<scenario>/questions.json  one question per fact (question, why_ask,
                                         required_docs, retrieval_query, source)
    State                                what is known so far (src/workflow/state.py)

One turn, step(state, answer):
    1. extract_facts   answer text -> typed fact value (rules first, model only as a fallback)
    2. review          escalation rules -> flags, required documents -> missing_docs
    3. next_question   first required, unknown, askable fact that has a question
    4. build_guide     question + why + documents + retrieved official text
    5. summarize       when nothing is left: summary for the preparer, status waiting_for_preparer

Principles:
    - Which question comes next is decided by code from the playbook, never by the model.
    - A model may propose a value, but parse_value() validates it before it enters the state.
    - The summary only claims passages that retrieval actually returned.
    - Retrieval (search) and the model (normalizer) are passed in, so tests need neither.
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

YES = {"yes", "y", "true", "1", "是", "有", "对"}
NO = {"no", "n", "false", "0", "否", "没有", "无", "不是"}
NONE_WORDS = {"none", "no", "n/a", "无", "没有"}


def load_scenario(scenario: str) -> tuple[dict, list[dict]]:
    """Read playbooks/<scenario>/scenario.json and questions.json.

    Returns (definition, questions): definition is the scenario dict
    (id, title, facts, escalation_rules); questions is the list of question dicts.
    """
    folder = PLAYBOOKS / scenario
    definition = json.loads((folder / "scenario.json").read_text(encoding="utf-8"))
    questions = json.loads((folder / "questions.json").read_text(encoding="utf-8"))
    return definition, questions


def start_interview(scenario: str) -> State:
    """Create a new State for this scenario with every fact set to None (unknown).

    Reads: the "facts" list of scenario.json (only each fact's "name").
    """
    definition, _ = load_scenario(scenario)
    facts = {f["name"]: None for f in definition["facts"]}
    return State(scenario=scenario, facts=facts)


def next_question(state: State) -> dict | None:
    """Decide the next question. Pure code, no model.

    Walks scenario.json facts in order and returns the question (from questions.json)
    for the first fact that is
        1. required,
        2. still unknown (None) in state.facts,
        3. allowed now: its ask_when conditions hold (conditions.check),
        4. covered by a question in questions.json (facts without one are skipped).
    Returns None when nothing is left to ask, which ends the interview.
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
    """Run the playbook's checks after every accepted answer. Pure code.

    Reads: scenario.json "escalation_rules" and each question's "required_docs".
    Updates the same state object and returns it:
        - every rule whose "when" conditions hold is added to state.flags once
          ({"rule": id, "reason": text});
        - for every question whose fact is answered with a truthy value (True, an amount,
          a non-empty list), its required_docs are added to state.missing_docs once.
          A "no" (False) answer needs no documents.
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


def fact_definition(definition: dict, name: str) -> dict:
    """Return the scenario.json entry (name, type, options, required, ask_when) for one fact.

    Raises KeyError if the fact is not defined, which means questions.json and
    scenario.json disagree.
    """
    for f in definition["facts"]:
        if f["name"] == name:
            return f
    raise KeyError(f"Fact {name!r} is not defined in scenario {definition['id']!r}")


def parse_value(fact: dict, text: str):
    """Convert answer text into the fact's type, or raise ValueError. No model involved.

    Reads: fact["type"] and, for enums, fact["options"].
        bool   "yes" / "no" (also y, true, 1, 是, 有, 没有 ...)         -> True / False
        money  "$12,000", "99.5"; negative amounts rejected           -> int or float
        enum   must be one of options; "Head of household" accepted   -> "head_of_household"
        list   comma-separated, or "none"                             -> list of str
    The ValueError message tells the assistant what to enter instead.
    """
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
    """Turn the assistant's answer into a validated fact value.

    Reads: the fact definition of question["fact"] from scenario.json.
    Steps:
        1. parse_value(answer_text): exact answers such as "yes", "12,000", "single".
        2. Only if that fails and a normalizer is given (src/llm/normalize.py): the model
           turns free text ("she took about 20k out of her IRA") into a plain value
           ("20000"), and that proposal goes through parse_value() again.
    If the model is unavailable, unsure, or proposes an invalid value, the original
    ValueError is raised, so nothing unvalidated is ever written into the state.
    Returns {"facts": {fact_name: value}, "parsed_by": "rules" | "model"}.
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
    """Real retrieval (src/rag/retriever.retrieve: Ollama embedding + RDS hybrid search).

    Imported inside the function so the engine and its tests load without DB / Ollama.
    """
    from src.rag.retriever import retrieve
    return retrieve(query)


def _doc_key(text: str) -> str:
    """Reduce a reference or citation to its document name for comparison.

    'IRS Pub 17, Chapter 2' and "IRS Pub 17 (2025), 'Filing status', p. 23" both become
    'irs pub 17': drop a "(year)", keep the part before the first comma, lower-case.
    """
    head = re.sub(r"\s*\(\d{4}\)", "", text).split(",")[0]
    return " ".join(head.lower().split())


def check_references(source: str, citations: list[dict], retrieval_ok: bool) -> list[dict]:
    """Compare the playbook's hand-written references with what retrieval returned.

    Reads:
        source        question["source"], e.g. "IRS Pub 590-B; FTB 3805P"
        citations     passages returned by retrieval ({"citation", "text"})
        retrieval_ok  False when retrieval raised an error
    Returns one {"reference", "retrieved"} per reference, where retrieved is
        True   the document appeared among the retrieved passages,
        False  it did not (probably not in the knowledge base yet),
        None   retrieval was down, so it could not be checked.
    """
    found = {_doc_key(c["citation"]) for c in citations if c.get("citation")}
    refs = []
    for part in source.split(";"):
        ref = part.strip()
        if ref:
            refs.append({"reference": ref,
                         "retrieved": (_doc_key(ref) in found) if retrieval_ok else None})
    return refs


def build_guide(state: State, question: dict, search=None, top_n: int = 3) -> dict:
    """Build what the assistant sees for one question.

    Reads: the question dict, its fact definition (type, options), state.flags,
    and search(question["retrieval_query"]) (real retrieval unless a stub is passed).
    Retrieval errors (DB stopped, Ollama off, network) are logged and the guide is
    still returned without passages, so the interview never stops because of RAG.

    Two kinds of "source" are kept apart:
        citations   passages retrieval really returned (top_n, text cut to 600 chars)
        references  the playbook's pointer to where the rule lives, marked by
                    check_references() as found / not found / unchecked
    Both are stored in state.evidence[question id] for summarize().

    Returns {"type": "question", question_id, fact, input_type, options, step_number,
             question, why_ask, required_docs, references, citations, retrieval_error, flags}.
    input_type / options tell the page which buttons to show (Yes/No, option chips, text).
    """
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
    """Build the summary for the preparer and set state.status = "waiting_for_preparer".

    Reads: state.facts, flags, missing_docs, history, evidence; scenario.json facts;
    questions.json (to map facts to question ids).
    Returns {"type": "summary", ...} with:
        facts                  every answered fact (value not None)
        not_yet_covered        required, askable facts the playbook has no question for yet
        flags / needs_preparer_attention / missing_docs
        citations_retrieved    citations really returned for the answered questions
        references_to_verify   all playbook references of the answered questions
        references_not_in_kb   references retrieval never returned (knowledge-base gaps)
        references_unchecked   references not checked because retrieval was down
        turns, status
    """
    definition, questions = load_scenario(state.scenario)
    by_fact = {q["fact"]: q for q in questions}
    answered = {k: v for k, v in state.facts.items() if v is not None}
    unanswered = [f["name"] for f in definition["facts"]
                  if f.get("required") and state.facts.get(f["name"]) is None
                  and check(f.get("ask_when"), state.facts)]

    answered_ids = [by_fact[k]["id"] for k in answered if k in by_fact]
    retrieved, verify, missing, unchecked = set(), set(), set(), set()
    for qid in answered_ids:
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
    """Return the question dict with this id from questions.json (KeyError if absent)."""
    _, questions = load_scenario(state.scenario)
    for q in questions:
        if q["id"] == question_id:
            return q
    raise KeyError(question_id)


def _show(state: State, question: dict, search) -> dict:
    """Mark the question as the current one (append to state.asked once) and build its guide."""
    if not state.asked or state.asked[-1] != question["id"]:
        state.asked.append(question["id"])
    return build_guide(state, question, search=search)


def step(state: State, answer_text: str | None = None, search=None,
         normalizer=None) -> tuple[State, dict]:
    """Run one turn of the interview. Returns (state, output).

    step(state)          -> the first question (no answer yet)
    step(state, answer)  -> the answer is for the last question in state.asked:
                            extract_facts -> update facts and history -> review ->
                            next_question -> build_guide, or summarize when done

    output["type"] is one of
        "question"  guide for the next question (see build_guide)
        "error"     the answer could not be understood; state is unchanged and
                    output["guide"] shows the same question again
        "summary"   interview finished (see summarize)
    search / normalizer are injected: real retrieval and the LLM in the app, stubs in tests.
    Raises RuntimeError if the interview is no longer in progress.
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
