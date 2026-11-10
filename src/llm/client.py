"""
client.py: the only place in the project that talks to an LLM provider.

Everything else calls chat_json() and never imports a provider SDK, so switching
OpenAI -> Claude (or a local model) means changing only this file.

Reads from .env: OPENAI_API_KEY, OPENAI_EXTRACT_MODEL or OPENAI_MODEL (default gpt-4o-mini).
"""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv


class LLMUnavailable(RuntimeError):
    """Raised for every model failure: no key, SDK missing, network, quota, bad JSON.

    Callers handle one error type and fall back to rules-only behaviour.
    """


def is_configured() -> bool:
    """True if OPENAI_API_KEY is set (in the environment or .env)."""
    load_dotenv()
    return bool(os.getenv("OPENAI_API_KEY"))


def chat_json(system: str, user: str, model: str | None = None, timeout: float = 30) -> dict:
    """Send one system + one user message and return the model's reply as a dict.

    The request uses temperature 0 and JSON mode, so the reply must be a JSON object.
    The openai package is imported only here, at call time, so tests and rules-only
    runs do not need it installed. Any failure is raised as LLMUnavailable.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMUnavailable("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMUnavailable("pip install openai") from exc

    client = OpenAI(api_key=api_key, timeout=timeout)
    try:
        response = client.chat.completions.create(
            model=model or os.getenv("OPENAI_EXTRACT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return json.loads(response.choices[0].message.content or "{}")
    except Exception as exc:
        raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
