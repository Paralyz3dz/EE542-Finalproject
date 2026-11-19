"""
client.py: the only place in the project that talks to an LLM provider.

Everything else calls chat_json() and never imports a provider SDK, so switching
OpenAI -> Claude (or a local model) means changing only this file.

Reads from .env: OPENAI_API_KEY, OPENAI_EXTRACT_MODEL or OPENAI_MODEL (default gpt-4o-mini).
"""
from __future__ import annotations

from base64 import b64encode
from collections.abc import Sequence
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


def chat_json(system: str, user: str, model: str | None = None, timeout: float = 30, *,
              images: Sequence[tuple[str, bytes]] = (), response_format: dict | None = None,
              max_completion_tokens: int | None = None) -> dict:
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

    content = user
    if images:
        content = [{"type": "text", "text": user}]
        for mime_type, data in images:
            encoded = b64encode(data).decode("ascii")
            content.append({"type": "image_url", "image_url": {
                "url": f"data:{mime_type};base64,{encoded}",
                "detail": "high",
            }})
    try:
        client = OpenAI(api_key=api_key, timeout=timeout)
        response = client.chat.completions.create(
            model=model or os.getenv("OPENAI_EXTRACT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
            temperature=0,
            response_format=response_format or {"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
            **({"max_completion_tokens": max_completion_tokens} if max_completion_tokens is not None else {}),
        )
        return json.loads(response.choices[0].message.content or "{}")
    except Exception as exc:
        raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
