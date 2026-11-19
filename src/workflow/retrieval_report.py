"""
retrieval_report.py: check the knowledge base against the playbook quickly, without a model or a chat.

    python -m src.workflow.retrieval_report                      every question's retrieval_query, once
    python -m src.workflow.retrieval_report --case baseline_w2_only
                                                                 replay one fictional case (tests/cases/)
    python -m src.workflow.retrieval_report --all-cases          replay every case
    add  --out report.md                                         also write the report to a file

For each question it runs the same retrieval as the interview (top 5 passages) and compares the
documents returned with the question's `source` (engine.check_references). The report lists, per
source document, how many of its questions found it (hit@5), the questions that missed and what came
back instead, and the documents that were never returned: the knowledge base gaps to fix first.

Needs the database and Ollama (src/rag/retriever.py), like the web page with TAX_AGENT_RETRIEVAL=on.
Uses fictional cases only; nothing is saved to the interview store.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from src.workflow.engine import INTAKE, ROOT, _doc_key, check_references, load_scenario, start_interview, step

CASES = ROOT / "tests" / "cases"
TYPICAL = {"ca_resident_full_year": "yes", "is_us_citizen": "yes"}
TOP_N = 5


def cached(search):
    """Wrap a search function so each query runs once."""
    memo: dict[str, dict] = {}

    def run(query: str) -> dict:
        if query not in memo:
            memo[query] = search(query)
        return memo[query]
    return run


def check_question(question: dict, search) -> dict:
    """Retrieve for one question and compare with its source. Returns one report row."""
    try:
        matches = (search(question["retrieval_query"]).get("matches") or [])[:TOP_N]
        ok = True
    except Exception as exc:
        return {"id": question["id"], "fact": question["fact"], "source": question.get("source", ""),
                "error": f"{type(exc).__name__}: {exc}"[:200], "refs": [], "retrieved": []}
    citations = [{"citation": m.get("citation"), "text": m.get("text", "")} for m in matches]
    return {"id": question["id"], "fact": question["fact"], "source": question.get("source", ""),
            "refs": check_references(question.get("source", ""), citations, ok),
            "retrieved": [c["citation"] for c in citations if c["citation"]]}


def all_questions(search) -> list[dict]:
    """One row per playbook question, from its fixed retrieval_query."""
    _, questions = load_scenario(INTAKE)
    return [check_question(q, search) for q in questions]


def default_answer(question: dict) -> str:
    """Answer for a question the case does not mention (same rule as tests/test_cases.py)."""
    if question["fact"] in TYPICAL:
        return TYPICAL[question["fact"]]
    if question["input_type"] == "bool":
        return "no"
    return question["options"][0] if question["input_type"] == "enum" else "0"


def replay_case(path: Path, search) -> list[dict]:
    """Play a fictional case start to summary with real retrieval; one row per question asked."""
    case = json.loads(path.read_text(encoding="utf-8"))
    _, questions = load_scenario(INTAKE)
    by_id = {q["id"]: q for q in questions}
    state = start_interview(INTAKE, case.get("priority", []), case.get("background", ""))
    state, out = step(state, search=search)
    rows = []
    for _ in range(300):
        if out["type"] == "summary":
            return rows
        rows.append(check_question(by_id[out["question_id"]], search))
        state, out = step(state, case["answers"].get(out["fact"], default_answer(out)), search=search)
    raise RuntimeError(f"{path.stem}: interview did not finish in 300 turns")


def _doc(citation: str) -> str:
    """Document name of a citation, as written: "IRS Pub 525 (2025), 'x', p. 2" -> "IRS Pub 525"."""
    return citation.split(" (")[0].split(",")[0].strip()


def render(title: str, rows: list[dict]) -> str:
    """Markdown report: coverage per source document, misses, documents never returned."""
    errors = [r for r in rows if r.get("error")]
    per_doc = defaultdict(lambda: [0, 0])
    misses, seen = [], set()
    for r in rows:
        seen.update(_doc_key(c) for c in r["retrieved"])
        for ref in r["refs"]:
            per_doc[ref["reference"]][1] += 1
            if ref["retrieved"]:
                per_doc[ref["reference"]][0] += 1
            else:
                misses.append((r, ref["reference"]))
    checked = sum(total for _, total in per_doc.values())
    hits = sum(hit for hit, _ in per_doc.values())
    never = sorted(ref for ref in per_doc if _doc_key(ref) not in seen)

    lines = [f"# Retrieval report: {title}", "",
             f"{len(rows)} questions, {checked} references, {hits} found in the top {TOP_N} "
             f"({(100 * hits // checked) if checked else 0}%)."]
    if errors:
        lines += ["", f"**Retrieval failed for {len(errors)} questions**, e.g. `{errors[0]['error']}`. "
                      "Check the database and Ollama."]
    lines += ["", "## Coverage per source document", "", "| Source | Found / questions |", "|---|---|"]
    for ref, (hit, total) in sorted(per_doc.items(), key=lambda kv: (kv[1][0] / kv[1][1], kv[0])):
        lines.append(f"| {ref} | {hit} / {total} |")
    lines += ["", "## Never returned for any question (likely missing from the knowledge base)", ""]
    lines += [f"- {ref}" for ref in never] or ["- none"]
    lines += ["", "## Questions whose source was not in the top results", "",
              "| Question | Source | Retrieved instead |", "|---|---|---|"]
    for r, ref in misses:
        got = ", ".join(dict.fromkeys(_doc(c) for c in r["retrieved"])) or "nothing"
        lines.append(f"| `{r['id']}` | {ref} | {got} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, search=None) -> str:
    """Run the chosen check and print the report; returns it (tests pass a stub search)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", help="case file name in tests/cases/, without .json")
    parser.add_argument("--all-cases", action="store_true", help="replay every case in tests/cases/")
    parser.add_argument("--out", help="also write the report to this file")
    args = parser.parse_args(argv)
    if search is None:
        from src.rag.retriever import retrieve
        search = retrieve
    search = cached(search)

    if args.case or args.all_cases:
        paths = [CASES / f"{args.case}.json"] if args.case else sorted(CASES.glob("*.json"))
        report = "\n".join(render(f"case {p.stem}", replay_case(p, search)) for p in paths)
    else:
        report = render("every playbook question", all_questions(search))
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    return report


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
