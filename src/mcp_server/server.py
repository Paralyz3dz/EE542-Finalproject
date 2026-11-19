"""
server.py: MCP server for the guided interview, driven by the user's own AI assistant.

Run (normally the MCP host starts it):
    python src/mcp_server/server.py                      stdio: Claude Desktop, Cursor, Claude Code, Gemini CLI,
                                                         Cherry Studio, OpenAI Secure MCP Tunnel (tunnel-client)
    python src/mcp_server/server.py --http 8765          streamable HTTP on 127.0.0.1:8765/mcp (this computer only)
    python src/mcp_server/server.py --http 8765 --public streamable HTTP for a public HTTPS tunnel (ChatGPT):
                                                         needs TAX_AGENT_MCP_TOKEN; served at /mcp/<token>
    python src/mcp_server/server.py --new-token          print a random token for TAX_AGENT_MCP_TOKEN

Why --public needs a token: ChatGPT developer mode connects to a public HTTPS URL with "No authentication"
(OAuth is the other option). Without a secret in the URL anyone who learns the tunnel address could use the
server. The token is a long random path segment, so the URL itself is the key. Use fictional clients only.

Needs the `mcp` package. Works with SDK 2.x (MCPServer) and 1.x (FastMCP).
Setup per host: docs/mcp.md. Nothing is printed to stdout in stdio mode: there stdout is the protocol.
"""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from mcp.types import ToolAnnotations  # noqa: E402

from src.mcp_server import tools  # noqa: E402

try:
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:
    from mcp.server.fastmcp import FastMCP as _Server

INSTRUCTIONS = """\
You are helping a tax preparer's assistant interview a client for a California / federal tax year 2025 \
intake. This server decides which question comes next; you talk to the client.

Rules:
1. Start with start_interview. Ask the client exactly the question in `next.question`, one question at a \
time. You may rephrase it in plain words, but do not change its meaning, skip it or add tax questions of \
your own.
2. Turn the client's reply into the answer format in `next.answer_format` (yes/no, one of `next.options`, \
a number, or a list) and call submit_answer. If the reply is unclear, ask the client again; never guess.
3. Put anything else the client mentions (other jobs, children, travel, a visa, a scholarship) in \
`client_note`; the server uses it to choose follow-up questions. Never put it in `answer`.
4. If `answer_not_accepted` is returned, explain it briefly and ask the same question again.
5. Do not give tax advice: do not say whether something is taxable or deductible, which forms to file or \
how much is owed. If asked, say the preparer will decide. You may use `why_ask` and \
`official_text` to explain why a question is asked, citing the source.
6. `escalations` are topics the preparer will handle; tell the client they will be reviewed, without \
judging them.
7. Ask the client for the documents in `documents_to_request`. Never ask for or repeat full Social \
Security, ITIN, bank or passport numbers or the client's name in the chat, and never pass them to a tool.
8. When the summary arrives, tell the client what `tell_the_client` says and give the assistant the \
`summary_file` path for the preparer.
"""

READ_ONLY = ToolAnnotations.model_validate({"readOnlyHint": True, "destructiveHint": False,
                                            "openWorldHint": False})
WRITES_INTERVIEW = ToolAnnotations.model_validate({"readOnlyHint": False, "destructiveHint": False,
                                                   "idempotentHint": False, "openWorldHint": False})

mcp = _Server("tax-intake", instructions=INSTRUCTIONS)


@mcp.tool(title="List interview topics", annotations=READ_ONLY)
def list_topics() -> dict:
    """List the interview topics that can be asked first (priority) and how the background note works."""
    return tools.list_topics()


@mcp.tool(title="Start a client intake interview", annotations=WRITES_INTERVIEW)
def start_interview(priority: list[str] | None = None, background: str = "") -> dict:
    """Start a client intake interview (California / federal, tax year 2025) and get the first question.

    Ask the client exactly the returned next.question, one at a time, and follow the returned rules.
    priority: topic ids from list_topics that the assistant is unsure about (asked first).
    background: optional short description of the client without name or ID numbers,
        e.g. "F-1 student, works as a TA".
    """
    return tools.start_interview(priority, background)


@mcp.tool(title="Submit the client's answer", annotations=WRITES_INTERVIEW)
def submit_answer(interview_id: str, answer: str, client_note: str = "") -> dict:
    """Submit the client's answer to the current question and get the next question (or the summary).

    answer: in the format given by next.answer_format (yes / no, one option, a number, or a list).
    client_note: anything else the client said that may need follow-up questions (no names or ID numbers).
    Give no tax advice; the preparer reviews the summary.
    """
    return tools.submit_answer(interview_id, answer, client_note)


@mcp.tool(title="Get an interview", annotations=READ_ONLY)
def get_interview(interview_id: str) -> dict:
    """Get the current state of an interview: answers so far, escalations and documents to collect."""
    return tools.get_interview(interview_id)


@mcp.tool(title="Look up official IRS / FTB text", annotations=READ_ONLY)
def lookup_official_text(query: str) -> dict:
    """Search official IRS / FTB passages (tax year 2025) to explain why a question is asked. Not client advice."""
    return tools.lookup_official_text(query)


def http_options(public: bool, token: str | None) -> tuple[str, bool]:
    """Path and whether to check the Host header, for local or public (tunnel) HTTP.

    Local: /mcp, Host must be localhost (the SDK's DNS rebinding protection).
    Public: /mcp/<token>; the Host header is the tunnel's name, so it is not checked, and the secret
    path is what keeps others out. Refuses to start in public mode without a long enough token.
    """
    if not public:
        return "/mcp", True
    if not token or len(token) < 24:
        raise SystemExit("--public needs TAX_AGENT_MCP_TOKEN (at least 24 characters). "
                         "Create one with: python src/mcp_server/server.py --new-token")
    return f"/mcp/{token}", False


def build_http_app(public: bool = False, token: str | None = None):
    """The streamable HTTP app (SDK 2.x), with the path and Host check from http_options."""
    from mcp.server.transport_security import TransportSecuritySettings
    path, check_host = http_options(public, token)
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=check_host,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"])
    return mcp.streamable_http_app(streamable_http_path=path, transport_security=security), path


def serve_http(port: int, public: bool) -> None:
    """Serve streamable HTTP on 127.0.0.1 only; a tunnel (cloudflared, ngrok) forwards public HTTPS to it."""
    token = os.getenv("TAX_AGENT_MCP_TOKEN")
    try:
        app, path = build_http_app(public, token)
    except TypeError:
        path, check_host = http_options(public, token)
        mcp.settings.host, mcp.settings.port, mcp.settings.streamable_http_path = "127.0.0.1", port, path
        if not check_host and hasattr(mcp.settings, "transport_security"):
            from mcp.server.transport_security import TransportSecuritySettings
            mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
        _announce(port, path, public)
        mcp.run(transport="streamable-http")
        return
    import uvicorn
    _announce(port, path, public)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def _announce(port: int, path: str, public: bool) -> None:
    """Say where the server listens, without printing the token itself."""
    shown = path if not public else "/mcp/<TAX_AGENT_MCP_TOKEN>"
    print(f"tax-intake MCP server: http://127.0.0.1:{port}{shown}", file=sys.stderr)
    if public:
        print("Connector URL for ChatGPT: https://<your tunnel host>" + shown, file=sys.stderr)


def main(argv: list[str]) -> None:
    """stdio by default; --http PORT [--public] serves streamable HTTP; --new-token prints a token."""
    if "--new-token" in argv:
        print(secrets.token_urlsafe(32))
    elif "--http" in argv:
        i = argv.index("--http")
        port = int(argv[i + 1]) if len(argv) > i + 1 and argv[i + 1].isdigit() else 8765
        serve_http(port, public="--public" in argv)
    else:
        mcp.run()


if __name__ == "__main__":
    main(sys.argv[1:])
