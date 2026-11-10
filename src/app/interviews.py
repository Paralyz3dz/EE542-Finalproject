"""
interviews.py: the interview service behind the web API.

Framework-free on purpose: api.py only translates HTTP <-> these methods, so this logic
is tested without FastAPI and could sit behind any other UI later.
Interviews live in memory (one dict, lost on restart); an interviews table can replace
_store later without changing the public methods.

Reads from .env:
    TAX_AGENT_RETRIEVAL  "off" = no retrieval (no DB / Ollama needed)
    TAX_AGENT_LLM        "off" = never call the model, exact answers only
    OPENAI_API_KEY       needed for the model note reader
"""
from __future__ import annotations

import json
import logging
import os
from threading import Lock

from src.workflow.engine import PLAYBOOKS, start_interview, step
from src.workflow.state import State

logger = logging.getLogger(__name__)


class InterviewNotFound(KeyError):
    """No interview with this id (never started, or the server restarted)."""


class InterviewClosed(RuntimeError):
    """The interview is not in a status that allows this action."""


def list_playbooks() -> list[dict]:
    """Return {"id", "title", "status"} for every playbooks/<folder>/scenario.json."""
    items = []
    for folder in sorted(PLAYBOOKS.iterdir()):
        path = folder / "scenario.json"
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            items.append({"id": d["id"], "title": d["title"], "status": d.get("status", "")})
    return items


def snapshot(state: State) -> dict:
    """The part of the state the page's sidebar shows.

    Returns interview_id, scenario, status, answered facts (not None), flags,
    missing_docs, number of answers, and whether the last answer was read by
    "rules" or the "model".
    """
    return {
        "interview_id": state.interview_id,
        "scenario": state.scenario,
        "status": state.status,
        "facts": {k: v for k, v in state.facts.items() if v is not None},
        "flags": state.flags,
        "missing_docs": state.missing_docs,
        "answered": len(state.history),
        "last_parsed_by": state.history[-1].get("parsed_by") if state.history else None,
    }


def _no_search(query: str) -> dict:
    """Retrieval switched off: always return no passages."""
    return {"matches": []}


class InterviewService:
    """Owns all running interviews and calls the engine for each turn.

    search      retrieval function (None = real retrieval via engine._default_search)
    normalizer  model note reader from src/llm/normalize.py (None = rules only)
    """

    def __init__(self, search=None, normalizer=None):
        """Store the injected search / normalizer; start with no interviews and an empty cache."""
        self.search = search
        self.normalizer = normalizer
        self._store: dict[str, State] = {}
        self._lock = Lock()
        self._cache: dict[str, dict] = {}

    @classmethod
    def from_env(cls) -> "InterviewService":
        """Build the service from .env switches: real retrieval unless TAX_AGENT_RETRIEVAL=off,
        model note reader if OPENAI_API_KEY is set and TAX_AGENT_LLM is not off."""
        search = None
        if os.getenv("TAX_AGENT_RETRIEVAL", "on").lower() == "off":
            search = _no_search
        normalizer = None
        if os.getenv("TAX_AGENT_LLM", "on").lower() != "off":
            from src.llm.normalize import make_normalizer
            normalizer = make_normalizer()
        logger.info("Interview service: retrieval=%s, llm=%s",
                    "off" if search else "on", "on" if normalizer else "off")
        return cls(search=search, normalizer=normalizer)

    def _cached_search(self, query: str) -> dict:
        """Retrieve once per query and reuse the result.

        A question shown again after an unreadable answer has the same retrieval_query,
        so it does not embed and query the database a second time. Failed searches raise
        and are not cached, so they are retried next time.
        """
        if query not in self._cache:
            if self.search is not None:
                result = self.search(query)
            else:
                from src.workflow.engine import _default_search
                result = _default_search(query)
            self._cache[query] = result
        return self._cache[query]

    def _get(self, interview_id: str) -> State:
        """Return the stored State or raise InterviewNotFound."""
        state = self._store.get(interview_id)
        if state is None:
            raise InterviewNotFound(interview_id)
        return state

    def start(self, scenario: str) -> dict:
        """Start an interview for a playbook id and return its first question.

        Raises ValueError for an unknown scenario.
        Returns {"output": first question guide, "state": snapshot}.
        """
        if scenario not in {p["id"] for p in list_playbooks()}:
            raise ValueError(f"Unknown scenario {scenario!r}")
        state = start_interview(scenario)
        state, out = step(state, search=self._cached_search, normalizer=self.normalizer)
        with self._lock:
            self._store[state.interview_id] = state
        return {"output": out, "state": snapshot(state)}

    def answer(self, interview_id: str, answer_text: str) -> dict:
        """Submit the assistant's answer to the current question (one engine step).

        Raises InterviewNotFound, or InterviewClosed if the interview already ended.
        Returns {"output": question | error | summary, "state": snapshot}.
        """
        with self._lock:
            state = self._get(interview_id)
            if state.status != "in_progress":
                raise InterviewClosed(f"Interview is {state.status}")
            state, out = step(state, answer_text, search=self._cached_search,
                              normalizer=self.normalizer)
        return {"output": out, "state": snapshot(state)}

    def get(self, interview_id: str) -> dict:
        """Return {"state": snapshot, "history": full history} for the preparer view."""
        state = self._get(interview_id)
        return {"state": snapshot(state), "history": state.history}

    def decide(self, interview_id: str, approve: bool, note: str = "") -> dict:
        """Record the preparer's decision on the summary.

        Allowed only while status is "waiting_for_preparer" (else InterviewClosed).
        Sets status to "approved" or "returned" and appends {"decision", "note"} to history.
        Returns {"state": snapshot}.
        """
        with self._lock:
            state = self._get(interview_id)
            if state.status != "waiting_for_preparer":
                raise InterviewClosed(f"Interview is {state.status}, not waiting for the preparer")
            state.status = "approved" if approve else "returned"
            state.history.append({"decision": state.status, "note": note})
        return {"state": snapshot(state)}
