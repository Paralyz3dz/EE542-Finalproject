"""
state.py: everything the engine remembers about one client interview.

The state is plain data (str / dict / list / bool / number), so the whole object can be
saved as one JSONB value (a future interviews table) and loaded back to pause / resume.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from uuid import uuid4


@dataclass
class State:
    """One interview.

    Fields:
        scenario      playbook folder name, e.g. "ca_w2_side_gig"
        interview_id  random UUID string, used by the API to find this interview
        facts         fact name -> value; None means "not known yet"
                      (False / 0 / [] are real answers, not "unknown")
        asked         question ids shown to the assistant, in order; the last one is
                      the question currently waiting for an answer
        flags         escalations that fired: {"rule": rule id, "reason": text}
        missing_docs  documents the client still has to bring, no duplicates
        history       one record per accepted answer:
                      {"question_id", "answer_text", "facts", "parsed_by": "rules" | "model"},
                      plus preparer decisions {"decision", "note"}
        evidence      question id -> {"citations": [retrieved citation strings],
                                      "references": [{"reference", "retrieved"}]}
        status        "in_progress" -> "waiting_for_preparer" -> "approved" | "returned"
    """
    scenario: str
    interview_id: str = field(default_factory=lambda: str(uuid4()))
    facts: dict = field(default_factory=dict)
    asked: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    missing_docs: list = field(default_factory=list)
    history: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    status: str = "in_progress"

    def to_dict(self) -> dict:
        """Plain dict of all fields (for JSON / database storage)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "State":
        """Rebuild a State from to_dict() output."""
        return cls(**data)
