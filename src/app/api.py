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


logger = logging.getLogger(__name__)


@dataclass
class Session:
    messages: list = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = create_tax_agent()
    app.state.sessions = {}
    app.state.sessions_lock = Lock()
    yield


app = FastAPI(title="Tax Agent Chat", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    session_id: UUID | None = None

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message must not be blank.")
        return value


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str


@app.get("/", response_class=FileResponse)
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request):
    state = request.app.state
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
