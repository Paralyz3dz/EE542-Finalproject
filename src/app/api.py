"""
api.py: FastAPI app.

    /                 free-form tax chat (LangChain agent, src/agent.py)
    /interview        guided client interview page
    /api/chat         chat endpoint
    /api/playbooks, /api/interviews/...   guided interview endpoints (src/app/interviews.py)
"""
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
from threading import Lock
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from src.agent import create_tax_agent
from src.app.interviews import (InterviewClosed, InterviewNotFound, InterviewService,
                                list_playbooks)


logger = logging.getLogger(__name__)


@dataclass
class Session:
    """One chat conversation: its message history and a lock so one session handles one request at a time."""
    messages: list = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the chat agent and the interview service at startup.

    If the chat agent cannot be created (e.g. no OPENAI_API_KEY), only /api/chat is
    disabled (503); the guided interview still works.
    """
    try:
        app.state.agent = create_tax_agent()
    except Exception as exc:
        logger.warning("Chat agent unavailable (%s); /api/chat will return 503", exc)
        app.state.agent = None
    app.state.interviews = InterviewService.from_env()
    app.state.sessions = {}
    app.state.sessions_lock = Lock()
    yield


app = FastAPI(title="Tax Agent Chat", lifespan=lifespan)


class ChatRequest(BaseModel):
    """Body of POST /api/chat: {"message": text, "session_id": optional UUID of an existing chat}."""
    message: str = Field(min_length=1, max_length=10000)
    session_id: UUID | None = None

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        """Trim spaces; reject a message that is only whitespace (422)."""
        value = value.strip()
        if not value:
            raise ValueError("Message must not be blank.")
        return value


class ChatResponse(BaseModel):
    """Reply of POST /api/chat: the session id to send next time and the agent's answer."""
    session_id: UUID
    answer: str


@app.get("/", response_class=FileResponse)
def index():
    """Serve the free-form chat page."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request):
    """Free-form tax Q&A: run the LangChain agent (retrieval tool + OpenAI) on the session's history.

    503 if the agent is not configured, 404 for an unknown session, 502 if the agent fails;
    a failed turn does not change the saved history."""
    state = request.app.state
    if state.agent is None:
        raise HTTPException(503, "Chat is not configured. Set OPENAI_API_KEY and restart.")
    session_id = payload.session_id or uuid4()
    with state.sessions_lock:
        if payload.session_id is not None:
            session = state.sessions.get(session_id)
            if session is None:
                raise HTTPException(404, "Session expired. Start a new conversation.")
        else:
            session = Session()

    with session.lock:
        messages = deepcopy(session.messages)
        messages.append({"role": "user", "content": payload.message})
        try:
            result = state.agent.invoke(
                {"messages": messages}, config={"recursion_limit": 20}
            )
            final = result["messages"][-1]
            if final.type != "ai" or getattr(final, "tool_calls", []):
                raise ValueError("Agent did not produce a final answer.")
            content = final.content
            answer = content if isinstance(content, str) else "\n".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
            if not answer.strip():
                raise ValueError("Agent returned an empty answer.")
        except Exception:
            logger.exception("Chat agent invocation failed")
            raise HTTPException(502, "Unable to get an answer. Please try again.") from None

        session.messages = result["messages"]
        with state.sessions_lock:
            state.sessions[session_id] = session
    return ChatResponse(session_id=session_id, answer=answer)


class StartRequest(BaseModel):
    """Body of POST /api/interviews: {"scenario": playbook id}."""
    scenario: str = Field(min_length=1, max_length=100)


class AnswerRequest(BaseModel):
    """Body of POST /api/interviews/{id}/answer: {"answer": text, 1-2000 chars}."""
    answer: str = Field(min_length=1, max_length=2000)

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, value: str) -> str:
        """Trim spaces; reject an answer that is only whitespace (422)."""
        value = value.strip()
        if not value:
            raise ValueError("Answer must not be blank.")
        return value


class DecisionRequest(BaseModel):
    """Body of POST /api/interviews/{id}/decision: {"approve": bool, "note": optional text}."""
    approve: bool
    note: str = Field(default="", max_length=2000)


@app.get("/interview", response_class=FileResponse)
def interview_page():
    """Serve the assistant's guided-interview page."""
    return FileResponse(Path(__file__).parent / "static" / "interview.html")


@app.get("/api/playbooks")
def playbooks():
    """List available scenarios for the page's dropdown."""
    return {"playbooks": list_playbooks()}


@app.post("/api/interviews")
def start(payload: StartRequest, request: Request):
    """Start an interview; returns the first question and a sidebar snapshot (404: unknown scenario)."""
    try:
        return request.app.state.interviews.start(payload.scenario)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from None


@app.post("/api/interviews/{interview_id}/answer")
def answer(interview_id: str, payload: AnswerRequest, request: Request):
    """Submit an answer; returns the next question, an error, or the summary (404 / 409)."""
    try:
        return request.app.state.interviews.answer(interview_id, payload.answer)
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found. Start a new one.") from None
    except InterviewClosed as exc:
        raise HTTPException(409, str(exc)) from None


@app.get("/api/interviews/{interview_id}")
def get_interview(interview_id: str, request: Request):
    """Return the interview's snapshot and full history (404 if unknown)."""
    try:
        return request.app.state.interviews.get(interview_id)
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None


@app.post("/api/interviews/{interview_id}/decision")
def decide(interview_id: str, payload: DecisionRequest, request: Request):
    """Preparer approves or returns the summary (409 unless waiting for the preparer)."""
    try:
        return request.app.state.interviews.decide(interview_id, payload.approve, payload.note)
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None
    except InterviewClosed as exc:
        raise HTTPException(409, str(exc)) from None

