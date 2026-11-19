"""
interviews.py: the interview service behind the web API.

Framework-free on purpose: api.py only translates HTTP <-> these methods, so this logic
is tested without FastAPI and could sit behind any other UI later.
Interviews and their current outputs are stored by owner so they can be resumed.

Reads from .env:
    TAX_AGENT_RETRIEVAL  "off" = no retrieval (no DB / Ollama needed)
    TAX_AGENT_LLM        "off" = exact answers only
    OPENAI_API_KEY       used for Listwise ranking and the model note reader
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import monotonic
from uuid import uuid4

from dotenv import load_dotenv

from src.app.documents import TAX_YEAR, extract_document_facts
from src.db.interview_store import Interview, InterviewStore
from src.llm.interview_qa import respond_to_question
from src.workflow.engine import (INTAKE, MODULES, PLAYBOOKS, apply_triggers, load_scenario, record_answer, review, sections,
                                 start_interview, step, summarize)
from src.workflow.state import State

logger = logging.getLogger(__name__)

MCP_OWNER = "mcp-assistant"


class InterviewNotFound(KeyError):
    """No accessible interview with this id."""


class InterviewClosed(RuntimeError):
    """The interview is not in a status that allows this action."""


class InvalidInterviewAnswer(ValueError):
    pass


class DocumentExtractionUnavailable(RuntimeError):
    pass


class StagedDocumentUnavailable(RuntimeError):
    pass


@dataclass
class StagedDocument:
    owner: str
    scenario: str
    name: str
    facts: dict
    expires_at: float


def list_playbooks() -> list[dict]:
    """Return {"id", "title", "status", "sections"} for each playbook.

    The modular intake (playbooks/modules/) comes first and lists the sections an assistant can
    put first; then every single-scenario playbook (playbooks/<folder>/scenario.json).
    """
    items = []
    if MODULES.exists():
        definition, _ = load_scenario(INTAKE)
        items.append({"id": INTAKE, "title": definition["title"], "status": "draft",
                      "sections": sections(definition)})
    for folder in sorted(PLAYBOOKS.iterdir()):
        path = folder / "scenario.json"
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            items.append({"id": d["id"], "title": d["title"], "status": d.get("status", ""), "sections": []})
    return items


def snapshot(state: State) -> dict:
    """The part of the state the page's sidebar shows.

    Returns interview_id, scenario, status, answered facts (not None), flags,
    missing_docs, number of answers, and whether the last answer was read by
    "rules" or the "model".
    """
    history = [entry for entry in state.history if "tax_qa" not in entry]
    return {
        "interview_id": state.interview_id,
        "scenario": state.scenario,
        "status": state.status,
        "facts": {k: v for k, v in state.facts.items() if v is not None},
        "flags": state.flags,
        "missing_docs": state.missing_docs,
        "answered": sum("attachments" not in entry or "answer_text" in entry for entry in history),
        "priority": list(state.priority),
        "triggers": sorted({b["keyword"] for b in state.boosted}),
        "last_parsed_by": history[-1].get("parsed_by") if history else None,
    }


def _no_search(query: str) -> dict:
    """Retrieval switched off: always return no passages."""
    return {"matches": []}


class InterviewService:
    """Owns all running interviews and calls the engine for each turn.

    search      retrieval function (None = real retrieval via engine._default_search)
    normalizer  model note reader from src/llm/normalize.py (None = rules only)
    """

    def __init__(self, search=None, normalizer=None, document_chat=None, store=None, *, tax_chat=None):
        self.search = search
        self.normalizer = normalizer
        self.document_chat = document_chat
        self.tax_chat = tax_chat
        self._store = store if store is not None else InterviewStore()
        self._lock = Lock()
        self._cache: dict[tuple, dict] = {}
        self._documents: dict[str, StagedDocument] = {}

    @classmethod
    def from_env(cls) -> "InterviewService":
        """Build the service from .env switches: real retrieval unless TAX_AGENT_RETRIEVAL=off,
        model note reader if OPENAI_API_KEY is set and TAX_AGENT_LLM is not off."""
        root = Path(__file__).resolve().parents[2]
        load_dotenv(root / ".env")
        search = None
        if os.getenv("TAX_AGENT_RETRIEVAL", "on").lower() == "off":
            search = _no_search
        normalizer = None
        document_chat = None
        if os.getenv("TAX_AGENT_LLM", "on").lower() != "off":
            from src.llm.client import chat_json
            from src.llm.normalize import make_normalizer
            normalizer = make_normalizer()
            if normalizer is not None:
                document_chat = chat_json
        logger.info("Interview service: retrieval=%s, llm=%s",
                    "off" if search else "on", "on" if normalizer else "off")
        store = InterviewStore(
            path=os.getenv("TAX_AGENT_INTERVIEW_DB_PATH") or root / "logs" / "interviews.sqlite3",
            postgres_url=os.getenv("TAX_AGENT_INTERVIEW_DATABASE_URL"),
        )
        return cls(search=search, normalizer=normalizer, document_chat=document_chat, store=store,
                   tax_chat=document_chat)

    def _cached_search(self, query: str, *, question: dict | None = None) -> dict:
        """Retrieve once per query and reuse the result.

        A question shown again after an unreadable answer has the same retrieval_query,
        so it does not embed and query the database a second time. Failed searches raise
        and are not cached, so they are retried next time.
        """
        context = question if self.search is None and question is not None else {}
        key = (query, context.get("id"), context.get("question"), context.get("why_ask"))
        if key in self._cache:
            return self._cache[key]
        if self.search is not None:
            result = self.search(query)
        else:
            from src.workflow.engine import _default_search
            result = _default_search(query, question=question)
        if not result.get("rerank_fallback"):
            self._cache[key] = result
        return result

    def _search_question(self, question: dict) -> dict:
        return self._cached_search(question["retrieval_query"], question=question)

    def _get(self, interview_id: str, owner: str) -> Interview:
        interview = self._store.get(owner, interview_id)
        if interview is None:
            raise InterviewNotFound(interview_id)
        return interview

    def _save(self, owner: str, interview: Interview) -> None:
        if not self._store.save(owner, interview):
            raise InterviewClosed("The interview changed. Reopen it from history and try again.")

    def list(self, owner: str = "") -> list[dict]:
        with self._lock:
            return self._store.list(owner)

    def rename(self, interview_id: str, title: str, *, owner: str = "") -> dict:
        with self._lock:
            if not self._store.rename(owner, interview_id, title):
                raise InterviewNotFound(interview_id)
        return {"title": title}

    def delete(self, interview_id: str, *, owner: str = "") -> None:
        with self._lock:
            if not self._store.delete(owner, interview_id):
                raise InterviewNotFound(interview_id)

    def start(self, scenario: str, priority: list[str] | None = None, background: str = "",
              document_ids: list[str] | None = None, *, owner: str = "") -> dict:
        """Start an interview for a playbook id and return its first question.

        priority: sections the assistant is unsure about and wants asked first (intake only).
        background: optional short client description; its trigger words reorder the interview.
        Raises ValueError for an unknown scenario or priority section.
        Returns {"output": first question guide, "state": snapshot}.
        """
        if scenario not in {p["id"] for p in list_playbooks()}:
            raise ValueError(f"Unknown scenario {scenario!r}")
        state = start_interview(scenario, priority, background)
        with self._lock:
            documents = self._selected_documents(state, document_ids or [], owner, {})
            _, conflicts = self._update_document_facts(state, documents)
            state, out = step(state, question_search=self._search_question, normalizer=self.normalizer)
            self._store.create(owner, Interview(state, out, documents, conflicts))
            return {"output": out, "state": snapshot(state), "conflicts": conflicts}

    def answer(self, interview_id: str, answer_text: str | None, question_id: str | None = None,
               *, owner: str = "", document_ids: list[str] | None = None, tax_questions: bool = False) -> dict:
        """Submit the assistant's answer to the current question (one engine step).

        Raises InterviewNotFound, or InterviewClosed if the interview already ended.
        Returns {"output": question | error | summary, "state": snapshot}.
        """
        with self._lock:
            interview = self._get(interview_id, owner)
            state = interview.state
            attachments_only = bool(document_ids) and answer_text is None
            if state.status != "in_progress" and not (attachments_only and state.status == "waiting_for_preparer"):
                raise InterviewClosed(f"Interview is {state.status}")
            if question_id is not None and (not state.asked or state.asked[-1] != question_id):
                raise InterviewClosed("The current question changed. Refresh the interview before answering.")

        guide = interview.output
        tax_qa = None
        if tax_questions and answer_text is not None and guide["type"] == "question":
            from src.workflow.engine import _default_search

            tax_qa = respond_to_question(self.tax_chat, self.search or _default_search, guide, answer_text, TAX_YEAR)

        with self._lock:
            if tax_qa is not None:
                state.history.append({"question": guide["question"], "message": answer_text, "tax_qa": tax_qa})
                self._save(owner, interview)
                return {"output": {"type": "tax_qa", "tax_qa": tax_qa, "guide": guide}, "state": snapshot(state)}
            if document_ids:
                return self._answer_with_documents(interview, answer_text, document_ids, owner)
            interview.state, out = step(state, answer_text, question_search=self._search_question,
                                        normalizer=self.normalizer)
            interview.output = out.get("guide", out)
            self._save(owner, interview)
            return {"output": out, "state": snapshot(interview.state)}

    def _answer_with_documents(self, interview: Interview, answer_text: str | None,
                               document_ids: list[str], owner: str) -> dict:
        state = interview.state
        selected = self._selected_documents(state, document_ids, owner, interview.documents)
        uploaded = [{"id": key, "name": doc["name"]} for key, doc in selected.items()
                    if key not in interview.documents]
        if answer_text is None and not uploaded:
            return {"output": interview.output, "state": snapshot(state), "conflicts": interview.conflicts}

        if answer_text is not None:
            error = record_answer(state, answer_text, self.normalizer)
            if error is not None:
                raise InvalidInterviewAnswer(error)
            if uploaded:
                state.history[-1]["attachments"] = uploaded
        else:
            state.history.append({"attachments": uploaded})

        interview.documents = interview.documents | selected
        changed, interview.conflicts = self._update_document_facts(state, interview.documents)
        if changed:
            state.status = "in_progress"
        if state.status == "in_progress":
            interview.state, interview.output = step(state, question_search=self._search_question, normalizer=self.normalizer)
        else:
            interview.output = summarize(state)
        self._save(owner, interview)
        return {"output": interview.output, "state": snapshot(state), "conflicts": interview.conflicts}

    def stage_documents(self, scenario: str, files: list[tuple[str, bytes]], *, owner: str = "") -> dict:
        if scenario not in {p["id"] for p in list_playbooks()}:
            raise ValueError(f"Unknown scenario {scenario!r}")
        if self.document_chat is None:
            raise DocumentExtractionUnavailable("Document extraction requires an enabled model and OPENAI_API_KEY.")
        definition, questions = load_scenario(scenario)
        extracted = extract_document_facts(files, definition, questions, self.document_chat)
        reports = []
        with self._lock:
            now = monotonic()
            self._documents = {key: doc for key, doc in self._documents.items() if doc.expires_at > now}
            for item in extracted:
                name, facts = item["name"], item["facts"]
                report = {"id": None, "name": name, "count": len(facts)}
                if item.get("error"):
                    report["error"] = item["error"]
                if facts:
                    document_id = str(uuid4())
                    self._documents[document_id] = StagedDocument(owner, scenario, name, facts, now + 24 * 60 * 60)
                    report["id"] = document_id
                reports.append(report)
        return {"tax_year": TAX_YEAR, "files": reports}

    def remove_document(self, document_id: str, *, owner: str = "") -> None:
        with self._lock:
            document = self._documents.get(document_id)
            if document is not None and document.owner == owner:
                del self._documents[document_id]

    def _selected_documents(self, state: State, document_ids: list[str], owner: str, saved: dict) -> dict:
        documents = {}
        now = monotonic()
        for document_id in dict.fromkeys(document_ids):
            if document_id in saved:
                documents[document_id] = saved[document_id]
                continue
            document = self._documents.get(document_id)
            if (document is None or document.owner != owner or document.expires_at <= now
                    or document.scenario != state.scenario):
                raise StagedDocumentUnavailable("A staged file is unavailable. Remove it or upload it again.")
            documents[document_id] = {"name": document.name, "facts": document.facts}
        return documents

    def _update_document_facts(self, state: State, documents: dict) -> tuple[bool, list[str]]:
        candidates, sources, conflicts = {}, {}, set()
        for document in documents.values():
            for key, value in document["facts"].items():
                if candidates.setdefault(key, value) != value:
                    conflicts.add(key)
                sources.setdefault(key, []).append(document["name"])

        previous = {}
        manual_history = []
        for entry in state.history:
            if entry.get("parsed_by") == "document":
                previous.update(entry["facts"])
            else:
                manual_history.append(entry)
        for key in previous:
            state.facts[key] = None

        accepted = {}
        for key, value in candidates.items():
            manual_value = state.facts.get(key)
            if manual_value is not None and manual_value != value:
                conflicts.add(key)
            if manual_value is None and key not in conflicts:
                accepted[key] = value
        state.facts.update(accepted)

        if previous or accepted:
            _, questions = load_scenario(state.scenario)
            question_ids = {q["fact"]: q["id"] for q in questions}
            state.history = manual_history + [
                {"question_id": question_ids[key], "facts": {key: value}, "parsed_by": "document",
                 "files": list(dict.fromkeys(sources[key]))}
                for key, value in accepted.items()
            ]
        changed = previous != accepted
        if changed:
            state.flags.clear()
            state.missing_docs.clear()
            review(state)
        return changed, sorted(conflicts)

    def sync_documents(self, interview_id: str, document_ids: list[str], *, owner: str = "") -> dict:
        with self._lock:
            interview = self._get(interview_id, owner)
            state = interview.state
            if state.status not in {"in_progress", "waiting_for_preparer"}:
                raise InterviewClosed(f"Interview is {state.status}")
            documents = self._selected_documents(state, document_ids, owner, interview.documents)
            changed, conflicts = self._update_document_facts(state, documents)
            if changed:
                state.status = "in_progress"
            if state.status == "in_progress":
                interview.state, interview.output = step(state, question_search=self._search_question, normalizer=self.normalizer)
            else:
                interview.output = summarize(state)
            interview.documents = documents
            interview.conflicts = conflicts
            self._save(owner, interview)
            return {"output": interview.output, "state": snapshot(interview.state), "conflicts": conflicts}

    def get(self, interview_id: str, *, owner: str = "") -> dict:
        with self._lock:
            interview = self._get(interview_id, owner)
        state = interview.state
        _, questions = load_scenario(state.scenario)
        prompts = {question["id"]: question["question"] for question in questions}
        turns = []
        for entry in state.history:
            if "tax_qa" in entry:
                turns.append({"question": entry["question"], "answer": entry["message"], "tax_qa": entry["tax_qa"]})
                continue
            turn = {}
            if "answer_text" in entry:
                turn.update(question=prompts.get(entry["question_id"], "Previous question"), answer=entry["answer_text"])
            if "attachments" in entry:
                turn["attachments"] = [item for item in entry["attachments"] if item["id"] in interview.documents]
                if not turn["attachments"]:
                    del turn["attachments"]
            if turn:
                turns.append(turn)
        return {
            "state": snapshot(state),
            "history": state.history,
            "turns": turns,
            "output": interview.output,
            "documents": [
                {"id": key, "name": document["name"], "count": len(document["facts"])}
                for key, document in interview.documents.items()
            ],
            "conflicts": interview.conflicts,
        }

    def decide(self, interview_id: str, approve: bool, note: str = "", *, owner: str = "") -> dict:
        """Record the preparer's decision on the summary.

        Allowed only while status is "waiting_for_preparer" (else InterviewClosed).
        Sets status to "approved" or "returned" and appends {"decision", "note"} to history.
        Returns {"state": snapshot}.
        """
        with self._lock:
            interview = self._get(interview_id, owner)
            state = interview.state
            if state.status != "waiting_for_preparer":
                raise InterviewClosed(f"Interview is {state.status}, not waiting for the preparer")
            state.status = "approved" if approve else "returned"
            state.history.append({"decision": state.status, "note": note})
            interview.output = interview.output | {"status": state.status}
            self._save(owner, interview)
            return {"state": snapshot(state)}

    def add_note(self, interview_id: str, note: str, *, owner: str = "") -> list[str]:
        """Record something else the client said; its trigger words move matching questions forward."""
        with self._lock:
            interview = self._get(interview_id, owner)
            state = interview.state
            if state.status != "in_progress":
                raise InterviewClosed(f"Interview is {state.status}")
            definition, _ = load_scenario(state.scenario)
            triggers = apply_triggers(state, definition, note)
            state.history.append({"note": note, "triggers": triggers})
            self._save(owner, interview)
            return triggers

    def preparer_list(self, status: str | None = None, limit: int = 100) -> list[dict]:
        """Every interview (web page and MCP), newest first, for the preparer review page."""
        with self._lock:
            rows = self._store.list_all(status, limit)
        items = []
        for row in rows:
            state = row["state"]
            history = state.get("history", [])
            items.append({
                "interview_id": row["interview_id"],
                "channel": "mcp" if row["owner"] == MCP_OWNER else "web",
                "status": row["status"], "title": row["title"],
                "created_at": row["created_at"], "updated_at": row["updated_at"],
                "answered": sum(1 for entry in history if entry.get("question_id")),
                "escalations": len(state.get("flags", [])),
                "priority": state.get("priority", []),
                "background": next((entry["background"] for entry in history if entry.get("background")), ""),
            })
        return items

    def preparer_get(self, interview_id: str) -> dict:
        """One interview of any owner, with question labels and the summary once finished."""
        with self._lock:
            found = self._store.find(interview_id)
        if found is None:
            raise InterviewNotFound(interview_id)
        result = self.get(interview_id, owner=found["owner"])
        _, questions = load_scenario(result["state"]["scenario"])
        output = result["output"]
        return result | {
            "channel": "mcp" if found["owner"] == MCP_OWNER else "web",
            "created_at": found["created_at"], "updated_at": found["updated_at"], "title": found["title"],
            "summary": output if output.get("type") == "summary" else None,
            "labels": {q["fact"]: q["question"] for q in questions},
        }

    def preparer_decide(self, interview_id: str, approve: bool, note: str = "") -> dict:
        """The preparer's decision on an interview of any owner (see decide)."""
        with self._lock:
            found = self._store.find(interview_id)
        if found is None:
            raise InterviewNotFound(interview_id)
        return self.decide(interview_id, approve, note, owner=found["owner"])
