"""
state.py: what the engine remembers about one client interview.

The whole state is plain data (dict / list / str / number), so it can be saved as one
JSONB value in the database (interviews table) and loaded back to pause / resume.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from uuid import uuid4


@dataclass
class State:
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
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "State":
        return cls(**data)
