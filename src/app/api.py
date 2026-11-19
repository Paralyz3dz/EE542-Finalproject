"""
api.py: FastAPI app.

    /                 free-form tax chat (LangChain agent, src/agent.py)
    /interview        guided client interview page
    /api/chat         chat endpoint
    /api/playbooks, /api/interviews/...   guided interview endpoints (src/app/interviews.py)
    /preparer, /api/preparer/...         preparer review of every interview (web page and MCP)
"""
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import logging
import re
from secrets import token_urlsafe
from pathlib import Path
from threading import Lock
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from src.agent import create_tax_agent
from src.app.documents import MAX_FILE_BYTES, MAX_FILES
from src.app.interviews import (DocumentExtractionUnavailable, InvalidInterviewAnswer, InterviewClosed,
                                InterviewNotFound, InterviewService, StagedDocumentUnavailable,
                                list_playbooks)
from src.app.forms import FormRequest, fill_form, form_catalog, load_forms, prepare_form
from src.workflow.engine import INTAKE


logger = logging.getLogger(__name__)

INTERVIEW_COOKIE = "guided_interview_browser"


def interview_owner(request: Request, response: Response | None = None) -> str:
    token = request.cookies.get(INTERVIEW_COOKIE, "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{43}", token):
        if response is None:
            return ""
        token = token_urlsafe(32)
        response.set_cookie(INTERVIEW_COOKIE, token, max_age=365 * 24 * 60 * 60,
                            httponly=True, samesite="lax", secure=request.url.scheme == "https")
    return sha256(token.encode()).hexdigest()


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
    """Body of POST /api/interviews: {"scenario", "priority": [section ids asked first], "background": client note}."""
    scenario: str = Field(min_length=1, max_length=100)
    priority: list[str] = Field(default_factory=list, max_length=20)
    background: str = Field(default="", max_length=2000)
    document_ids: list[UUID] = Field(default_factory=list, max_length=MAX_FILES)


class AnswerRequest(BaseModel):
    answer: str | None = Field(default=None, min_length=1, max_length=2000)
    question_id: str | None = Field(default=None, max_length=100)
    document_ids: list[UUID] = Field(default_factory=list, max_length=MAX_FILES)

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, value: str | None) -> str | None:
        """Trim spaces; reject an answer that is only whitespace (422)."""
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Answer must not be blank.")
        return value

    @model_validator(mode="after")
    def require_content(self):
        if self.answer is None and not self.document_ids:
            raise ValueError("Send an answer or attach a file.")
        return self


class RenameInterviewRequest(BaseModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100,
                                           pattern=r"^[^\x00-\x1f\x7f]+$")]


class DecisionRequest(BaseModel):
    """Body of POST /api/interviews/{id}/decision: {"approve": bool, "note": optional text}."""
    approve: bool
    note: str = Field(default="", max_length=2000)


class SyncDocumentsRequest(BaseModel):
    document_ids: list[UUID] = Field(max_length=MAX_FILES)


@app.get("/interview", response_class=FileResponse)
def interview_page(request: Request):
    """Serve the assistant's guided-interview page."""
    response = FileResponse(Path(__file__).parent / "static" / "interview.html",
                            headers={"Cache-Control": "no-store"})
    interview_owner(request, response)
    return response


@app.get("/interview/inter-variable.woff2", response_class=FileResponse, include_in_schema=False)
def interview_font():
    font = Path(__file__).parent / "static" / "fonts" / "InterVariable.woff2"
    return FileResponse(font, media_type="font/woff2")


@app.get("/interview/instrument-serif.ttf", response_class=FileResponse, include_in_schema=False)
def interview_title_font():
    font = Path(__file__).parent / "static" / "fonts" / "InstrumentSerif-Regular.ttf"
    return FileResponse(font, media_type="font/ttf")


@app.get("/api/playbooks")
def playbooks():
    """List available scenarios for the page's dropdown."""
    return {"playbooks": list_playbooks()}


@app.get("/api/interviews")
def interview_list(request: Request):
    owner = interview_owner(request)
    interviews = request.app.state.interviews.list(owner) if owner else []
    return JSONResponse({"interviews": interviews}, headers={"Cache-Control": "no-store"})


@app.post("/api/interviews")
def start(payload: StartRequest, request: Request, response: Response):
    """Start an interview; returns the first question and a sidebar snapshot (404 unknown scenario, 422 bad priority)."""
    try:
        owner = interview_owner(request, response)
        result = request.app.state.interviews.start(
            payload.scenario, payload.priority, payload.background,
            [str(document_id) for document_id in payload.document_ids], owner=owner,
        )
        response.headers["Cache-Control"] = "no-store"
        return result
    except StagedDocumentUnavailable as exc:
        raise HTTPException(409, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(404 if "scenario" in str(exc) else 422, str(exc)) from None


@app.post("/api/interviews/{interview_id}/answer")
def answer(interview_id: str, payload: AnswerRequest, request: Request):
    """Submit an answer; returns the next question, an error, or the summary (404 / 409)."""
    try:
        result = request.app.state.interviews.answer(
            interview_id, payload.answer, payload.question_id, owner=interview_owner(request),
            document_ids=[str(document_id) for document_id in payload.document_ids],
            tax_questions=True,
        )
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found. Start a new one.") from None
    except (InterviewClosed, StagedDocumentUnavailable) as exc:
        raise HTTPException(409, str(exc)) from None
    except InvalidInterviewAnswer as exc:
        raise HTTPException(422, str(exc)) from None


@app.post("/api/interview-documents")
async def stage_documents(request: Request, response: Response, scenario: str = INTAKE):
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "multipart/form-data":
        raise HTTPException(415, "Send the files as multipart form data.")

    async def limited_stream():
        remaining = MAX_FILES * MAX_FILE_BYTES + 64 * 1024
        async for chunk in request.stream():
            remaining -= len(chunk)
            if remaining < 0:
                raise HTTPException(413, "This upload is too large. Choose smaller files.")
            yield chunk

    try:
        form = await MultiPartParser(request.headers, limited_stream(), max_files=MAX_FILES,
                                     max_fields=0).parse()
    except MultiPartException:
        raise HTTPException(400, f"Attach up to {MAX_FILES} files in the files field.") from None
    try:
        files = form.getlist("files")
        if not files or len(files) != len(form.multi_items()) or not all(
            isinstance(file, UploadFile) and file.filename for file in files
        ):
            raise HTTPException(422, "Attach one or more files in the files field.")
        uploads = []
        for file in files:
            filename = Path(file.filename.replace("\\", "/")).name
            uploads.append((filename, await file.read(MAX_FILE_BYTES + 1)))
    finally:
        await form.close()

    try:
        owner = interview_owner(request, response)
        result = await run_in_threadpool(request.app.state.interviews.stage_documents, scenario, uploads, owner=owner)
        response.headers["Cache-Control"] = "no-store"
        return result
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from None
    except DocumentExtractionUnavailable as exc:
        raise HTTPException(503, str(exc)) from None


@app.delete("/api/interview-documents/{document_id}", status_code=204)
def remove_document(document_id: UUID, request: Request):
    request.app.state.interviews.remove_document(str(document_id), owner=interview_owner(request))
    return Response(status_code=204)


@app.put("/api/interviews/{interview_id}/documents")
def sync_documents(interview_id: str, payload: SyncDocumentsRequest, request: Request):
    try:
        result = request.app.state.interviews.sync_documents(
            interview_id, [str(document_id) for document_id in payload.document_ids], owner=interview_owner(request),
        )
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None
    except (InterviewClosed, StagedDocumentUnavailable) as exc:
        raise HTTPException(409, str(exc)) from None


@app.get("/api/interviews/{interview_id}")
def get_interview(interview_id: str, request: Request):
    """Return the interview state, history, and current question or summary."""
    try:
        result = request.app.state.interviews.get(interview_id, owner=interview_owner(request))
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None


@app.patch("/api/interviews/{interview_id}")
def rename_interview(interview_id: str, payload: RenameInterviewRequest, request: Request):
    try:
        result = request.app.state.interviews.rename(interview_id, payload.title, owner=interview_owner(request))
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None


@app.delete("/api/interviews/{interview_id}", status_code=204)
def delete_interview(interview_id: str, request: Request):
    try:
        request.app.state.interviews.delete(interview_id, owner=interview_owner(request))
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None
    return Response(status_code=204)


@app.post("/api/interviews/{interview_id}/decision")
def decide(interview_id: str, payload: DecisionRequest, request: Request):
    """Preparer approves or returns the summary (409 unless waiting for the preparer)."""
    try:
        result = request.app.state.interviews.decide(
            interview_id, payload.approve, payload.note, owner=interview_owner(request),
        )
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None
    except InterviewClosed as exc:
        raise HTTPException(409, str(exc)) from None


PREPARER_STATUSES = {"in_progress", "waiting_for_preparer", "approved", "returned"}


@app.get("/preparer", response_class=FileResponse)
def preparer_page():
    return FileResponse(Path(__file__).parent / "static" / "preparer.html", headers={"Cache-Control": "no-store"})


@app.get("/api/preparer/interviews")
def preparer_list(request: Request, status: str | None = None, limit: int = 100):
    """Every interview (web page and MCP), newest first; status filters by state."""
    if status is not None and status not in PREPARER_STATUSES:
        raise HTTPException(422, f"status must be one of {sorted(PREPARER_STATUSES)}")
    items = request.app.state.interviews.preparer_list(status, max(1, min(limit, 500)))
    return JSONResponse({"interviews": items}, headers={"Cache-Control": "no-store"})


@app.get("/api/preparer/interviews/{interview_id}")
def preparer_get(interview_id: str, request: Request):
    try:
        result = request.app.state.interviews.preparer_get(interview_id)
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@app.post("/api/preparer/interviews/{interview_id}/decision")
def preparer_decide(interview_id: str, payload: DecisionRequest, request: Request):
    """Approve or return any interview waiting for the preparer (409 otherwise)."""
    try:
        result = request.app.state.interviews.preparer_decide(interview_id, payload.approve, payload.note)
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None
    except InterviewClosed as exc:
        raise HTTPException(409, str(exc)) from None
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


def completed_interview(interview_id: str, request: Request):
    try:
        state = request.app.state.interviews.get(interview_id, owner=interview_owner(request))["state"]
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None
    if state["status"] not in {"waiting_for_preparer", "approved", "returned"}:
        raise HTTPException(409, "Finish the interview before generating a draft.")
    return state


def form_data(interview_id: str, form_id: str, payload: FormRequest, request: Request):
    state = completed_interview(interview_id, request)
    form = load_forms().get(form_id)
    if form is None:
        raise HTTPException(404, "Tax form template not found.")
    try:
        return form, prepare_form(form, state["facts"], payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


@app.get("/api/interviews/{interview_id}/forms")
def interview_forms(interview_id: str, request: Request):
    state = completed_interview(interview_id, request)
    return JSONResponse({"forms": form_catalog(state["facts"])}, headers={"Cache-Control": "no-store"})


@app.post("/api/interviews/{interview_id}/forms/{form_id}/preview")
def preview_form(interview_id: str, form_id: str, payload: FormRequest, request: Request):
    _, data = form_data(interview_id, form_id, payload, request)
    return JSONResponse(data, headers={"Cache-Control": "no-store"})


@app.post("/api/interviews/{interview_id}/forms/{form_id}")
def download_form(interview_id: str, form_id: str, payload: FormRequest, request: Request):
    form, data = form_data(interview_id, form_id, payload, request)
    try:
        content = fill_form(form, data["fields"])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    return Response(content, media_type="application/pdf", headers={
        "Cache-Control": "no-store",
        "Content-Disposition": f'inline; filename="{form_id}-draft.pdf"',
    })

