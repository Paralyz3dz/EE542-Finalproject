# Using your own AI assistant to run the interview (MCP server)

The web page uses the server's OpenAI key to read free-text answers. With the MCP server, the interview runs
inside the user's own AI assistant instead (ChatGPT, Claude Desktop, Gemini CLI, Cherry Studio with
DeepSeek or Qwen, or any other MCP host), on the user's own subscription. Both paths use the server's
OpenAI configuration for Listwise retrieval ranking when available. For guided questions the reranker
receives the playbook query, question and purpose, along with the official passages. For direct lookup,
it receives the supplied query and official passages. Use general tax topics for direct lookup.

| Who | Does |
|---|---|
| The user's assistant (their model) | Talks to the client, rephrases each question, turns the reply into an answer (yes / no, an option, a number), and passes anything else the client said as `client_note` |
| This server (engine + playbook) | Picks the next question, checks every answer, raises escalations for the preparer, writes the summary |

The model never decides the flow: an answer the engine cannot parse is returned as `answer_not_accepted`
and the same question is asked again. The rules for the model (one question at a time, no tax advice, no
full SSN / ITIN / bank / passport numbers, no client names) are sent twice: in the server instructions, and
in the tool results (`rules` with the first question, `reminder` with every later one), because some hosts,
possibly ChatGPT, do not pass the server instructions to the model.

**Use fictional clients only** in this project (IRC 7216). Client answers pass through the AI provider the
tester chose.

## Tools

| Tool | Read-only | What it does |
|---|---|---|
| `list_topics` | yes | Topic ids that can be asked first (`priority`), e.g. `student`, `side_gig` |
| `start_interview(priority, background)` | no | Starts an intake; returns `rules` and the first question with `answer_format`, `why_ask` and any official text |
| `submit_answer(interview_id, answer, client_note)` | no | Sends one answer; returns the next question, or the summary when the interview is complete |
| `get_interview(interview_id)` | yes | Answers so far, escalations, documents to collect |
| `lookup_official_text(query)` | yes | Searches the IRS / FTB knowledge base (needs the database and Ollama running) |

"Read-only" is the MCP `readOnlyHint` annotation. ChatGPT asks the user to confirm every call to a tool
without it, so it asks for `start_interview` and `submit_answer` only.

Interviews are saved in the web server's interview store (`logs/interviews.sqlite3` by default) under the owner
`mcp-assistant`, and the preparer page marks them "AI assistant". If the web server runs on the same computer, they appear on `http://127.0.0.1:8000/preparer`
for review. A copy of each finished summary is also written to `interview_summaries/<interview id>.json`
(in `.gitignore`).

## Student guidance

The student module (Form 8843, Pub 970, Pub 519) is part of the same interview. It opens when the client
says they were a full-time student, and its questions come first when the background or a `client_note`
mentions a student or a visa, or when `priority` includes `student`. For a second-year F-1 student who
works as a TA, the summary includes, for example:

- escalations: resident or nonresident alien status; likely nonresident (exempt student, Form 8843), so
  Form 1040-NR needs a preparer certified in Foreign Student and Scholar tax law;
- reminders: file Form 8843 for 2025 even with no U.S. income; pay for required teaching is taxable, bring
  the W-2;
- documents: passport, visa, I-20 or DS-2019, I-94 travel history, 1098-T, award letters, W-2.

## ChatGPT (developer mode)

ChatGPT connects only to a remote MCP server over HTTPS; it cannot start `server.py` on your computer.
Two ways to reach it:

| | A. Public tunnel (Cloudflare) | B. OpenAI Secure MCP Tunnel |
|---|---|---|
| What it needs | `cloudflared` (free, no account for a quick tunnel) | An OpenAI Platform organization with Tunnels permission, a tunnel id and a runtime API key, `tunnel-client` |
| Server exposure | A public `https://….trycloudflare.com` address; the secret token in the path keeps others out | None: `tunnel-client` makes outbound HTTPS only |
| Our server runs as | `--http 8765 --public` | stdio, started by `tunnel-client` |
| Best for | Quick tests by teammates | A firm's real setup |

Developer mode is a beta for paid plans. Reports differ on whether every plan (for example Plus) may use
write tools; if ChatGPT lists the tools but refuses to call `start_interview`, check your plan.

### A. Public tunnel (quick test)

1. In the project folder, install the requirements and create a token:
   ```powershell
   cd D:\EE542\tax-agent
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python src\mcp_server\server.py --new-token
   ```
   Put the printed value in `.env` as `TAX_AGENT_MCP_TOKEN=...` (never commit `.env`, never paste the
   token in chat or in a PR).
2. Terminal 1, start the server for a tunnel:
   ```powershell
   $env:TAX_AGENT_RETRIEVAL = "off"
   python src\mcp_server\server.py --http 8765 --public
   ```
   It refuses to start without a token of at least 24 characters. It listens on 127.0.0.1 only.
3. Terminal 2, open the tunnel:
   ```powershell
   winget install --id Cloudflare.cloudflared
   cloudflared tunnel --url http://localhost:8765
   ```
   It prints an address like `https://random-words.trycloudflare.com` (new each time).
4. In ChatGPT: **Settings > Apps & Connectors > Advanced settings > Developer mode** on. Then create a
   connector (**Create** or **Add custom MCP server**; menu names change between versions):
   - URL: `https://random-words.trycloudflare.com/mcp/<your token>`
   - Authentication: **No authentication**
5. In a new chat, turn on the connector (the **+** menu, Developer mode) and write:
   *"Use the tax-intake connector's start_interview tool to start a tax intake interview. The client is a
   second-year F-1 student who works as a TA. Ask me one question at a time."*
6. ChatGPT asks to confirm `start_interview` and each `submit_answer`; check the JSON it will send, then
   confirm (choose to allow it for this chat if ChatGPT offers that).
7. When done, stop the tunnel (Ctrl+C in terminal 2). Create a new token if the URL was shared anywhere.

If ChatGPT cannot connect: the URL must end in `/mcp/<token>` exactly; both terminals must still be
running; `http://127.0.0.1:8765/mcp/<token>` should answer locally.

### B. OpenAI Secure MCP Tunnel

Create a tunnel and a runtime API key in the OpenAI Platform (Settings > Tunnels; the role needs the
Tunnels permissions), install `tunnel-client`, then let it start our server over stdio:

```powershell
tunnel-client init --sample sample_mcp_stdio_local --profile tax-intake --tunnel-id <tunnel id> `
  --mcp-command "D:\EE542\tax-agent\.venv\Scripts\python.exe D:\EE542\tax-agent\src\mcp_server\server.py"
tunnel-client doctor --profile tax-intake --explain
tunnel-client run --profile tax-intake
```

Then add the connector in ChatGPT with the tunnel connection, as in OpenAI's Secure MCP Tunnel guide.
No token is needed: the server is never public.

## Claude Desktop (Windows)

1. Install the requirements in the project's virtual environment (`pip install -r requirements.txt`).
2. Quit Claude Desktop completely (tray icon > Quit). It rewrites its config file when it exits, so edit
   the file only while the app is closed.
3. Open **Settings > Extensions > Local MCP servers > Edit config** (or
   `%APPDATA%\Claude\claude_desktop_config.json`) and add, at the top level next to `"preferences"`:
   ```json
   "mcpServers": {
     "tax-intake": {
       "command": "D:\\EE542\\tax-agent\\.venv\\Scripts\\python.exe",
       "args": ["D:\\EE542\\tax-agent\\src\\mcp_server\\server.py"],
       "env": { "TAX_AGENT_RETRIEVAL": "off" }
     }
   },
   ```
   Remove `"TAX_AGENT_RETRIEVAL": "off"` once the database and Ollama are running, to get official
   passages with each question.
4. Start Claude Desktop. `tax-intake` should show **Running** under Local MCP servers.
5. In a new chat: *"Use the tax-intake tools to start an interview. The client is a second-year F-1
   student who works as a TA."*

## Other hosts

| Host | Models | Setup |
|---|---|---|
| Gemini CLI | Gemini | Same `mcpServers` block as Claude Desktop, in `%USERPROFILE%\.gemini\settings.json` |
| Cherry Studio | DeepSeek, Qwen, GPT, Claude, local Ollama models | Settings > MCP Servers > Add: type STDIO, command = the venv `python.exe`, argument = the path to `server.py` |
| Any host with streamable HTTP | any | `python src\mcp_server\server.py --http 8765`, connect to `http://127.0.0.1:8765/mcp` (this computer only) |

Weaker tool-calling models may ask several questions at once or skip the tools; the engine still checks
every answer that reaches it.

## Test without a host

```powershell
npx @modelcontextprotocol/inspector .venv\Scripts\python.exe src\mcp_server\server.py
```

This opens the MCP Inspector in the browser to call the tools by hand (needs Node.js).

## Limits

- Client answers pass through the AI provider the user chose. A real deployment needs the client's
  written consent for that disclosure, an approved provider (no training on inputs) or a local model,
  and a proper login (OAuth) instead of the token in the URL.
- The quality of rephrasing depends on the user's model; the engine still checks every answer.
- Works with MCP Python SDK 2.x (`MCPServer`) and 1.x (`FastMCP`); the tunnel test runs on 2.x.
