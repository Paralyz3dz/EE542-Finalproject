"""
conditions.py: evaluate the small condition format used in playbooks.

A condition list looks like
    [{"fact": "has_side_income", "equals": true}]
and holds only when EVERY item holds. It is used by scenario.json for
"ask_when" (may this question be asked yet?) and "when" (does this escalation fire?).
Conditions are data, not Python strings, so loading a playbook never runs code (no eval).
"""
from __future__ import annotations


def check(conditions: list[dict] | None, facts: dict) -> bool:
    """Return True if all conditions hold for these facts.

    Reads:
        conditions  list of {"fact": name, "equals": value}, or None / [] for "no conditions"
        facts       State.facts (fact name -> value, None = unknown)
    Rules:
        - no conditions -> True (the question can always be asked)
        - a fact that is unknown (None) or missing -> False, so a follow-up is never asked
          before its parent question has been answered
        - otherwise the fact's value must equal "equals"
    """
    if not conditions:
        return True
    for cond in conditions:
        actual = facts.get(cond["fact"])
        if actual is None or actual != cond["equals"]:
            return False
    return True
