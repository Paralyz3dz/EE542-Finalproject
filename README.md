# Tax Agent MVP

A digital clone of a tax preparer, used by an assistant: it guides the client interview
(what to ask next and why), explains options in plain language with IRS / California FTB
citations, flags cases that need the preparer, and writes a summary for the preparer to approve.
It does not file returns or give final tax advice. Only public data and fictional cases.

Scope: California residents (federal base layer + California differences). The preparer's
know-how is written as **playbooks** (JSON); a hand-written **workflow engine** runs them,
decides the next question in code, and calls retrieval and the LLM only at fixed steps.

## Folders
| Folder | What goes here |
|---|---|
| `config/` | Settings (models, thresholds, paths) |
| `data/raw/irs/` | IRS PDFs: Pub 17 (2025), Pub 5101, Form 13614-C |
| `data/raw/ats/` | IRS MeF ATS test scenarios (fictional taxpayers) |
| `data/processed/` | Parsed text chunks, embeddings exports |
| `data/synthetic/` | Fictional client profiles we write |
| `playbooks/<scenario>/` | Interview playbooks: `scenario.json` (facts, ask_when, escalation rules) + `questions.json` |
| `src/ingest/` | PDF parsing and chunking |
| `src/rag/` | Embedding, vector search, reranking |
| `src/workflow/` | Interview workflow engine: `engine.py`, `state.py`, `conditions.py` |
| `src/llm/` | LLM adapter (`client.py`, the only provider call) and note reader (`normalize.py`) |
| `src/app/` | FastAPI backend (`api.py`), interview service (`interviews.py`), chat page and interview page (`static/`) |
| `eval/` | Test cases, TurboTax comparison, metrics |
| `models/` | Fine-tuned embedding model (not committed) |
| `logs/` | Run logs (not committed) |

## Workflow
1. The assistant picks a client scenario (playbook) and starts the interview
2. The engine chooses the next question in code: the first required, unknown fact whose
   `ask_when` conditions hold (follow-ups appear only after the answer that unlocks them)
3. The guide shows the question, why to ask it, documents to request, and passages
   retrieved from IRS / FTB documents (RAG)
4. The assistant records the answer: quick buttons, an exact value, or a free-text note.
   Rules parse it first; only if that fails does the LLM propose a value, which the same
   rules validate before it is saved
5. After every answer: escalation rules raise flags for the preparer; required documents
   are collected
6. When nothing is left to ask: a summary for the preparer (facts, flags, documents,
   citations actually retrieved, references still to verify)
7. On the web, the assistant selects **Submit to preparer** to send the summary for review
8. The preparer approves it or returns it with a note

## Quick start

### Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                 # then fill in your own keys (never commit .env)

python src/rag/chunk_pub17.py          # Pub 17 -> data/processed/pub17_2025_chunks.json + _tables.json
python tests/test_tables.py            # table tests, should print 11/11 passed
python src/rag/tables.py --pages 9 101 # inspect the tables on single pages
python -m src.agent                  # start the basic LangChain conversation agent
```

The agent uses OpenAI through LangChain and keeps the conversation history in
memory for the current run. Its `search_tax_documents` tool uses local Ollama
query embeddings and read-only HNSW + BM25 hybrid retrieval. HNSW uses
PostgreSQL/pgvector cosine search, filtered by tax year and embedding model.
BM25 searches the configured tax year's `embed_text` corpus in memory.
Each route returns up to 5 candidates; equal-weight reciprocal rank fusion
(RRF) deduplicates by chunk ID and returns the final top 5. The RRF constant
is 60 (`rrf_constant` in `config/settings.json`), following the
[original RRF method](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).
BM25 uses lowercase word/number tokens, including IRS form numbers, with no
stemming or translation. Chunks with no matching query tokens are excluded.
Set `OPENAI_API_KEY` in `.env`; `OPENAI_MODEL` and
`OPENAI_TEMPERATURE` are optional.

Retrieval also needs `DATABASE_URL`, a running Ollama instance, and these settings
(defaults shown):

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
```

`EMBEDDING_MODEL` is the label stored in `chunks.embedding_model`;
`OLLAMA_EMBEDDING_MODEL` is the local Ollama model tag. The document and query
vectors must come from the same embedding model and use 1024 dimensions.
Queries use Qwen's `Instruct: ...\nQuery: ...` format. Document vectors should
be generated from the existing `embed_text` field without a query instruction.
See the [Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
and [Ollama embedding API](https://docs.ollama.com/api/embed).

Retrieval fuses the vector and BM25 rankings, then uses the configured OpenAI
model to rank the ten fused candidates together and return the top five.
The model receives the retrieval query and official passages. Guided interviews
also send the question and purpose from the playbook. Set `OPENAI_API_KEY` to
enable Listwise ranking. If the model is unavailable or returns an invalid
ranking, retrieval returns the first five RRF candidates. Set
`rerank_enabled: false` in `config/settings.json` to always use RRF order.

Generate document vectors locally from a JSON snapshot containing `id` and
`embed_text` for each chunk:

```powershell
.\.venv\Scripts\python.exe -m src.rag.embed_chunks data/processed/pub17_2025_embedding_input.json
```

This command calls only local Ollama and saves
`data/processed/embeddings_qwen3-0.6b.jsonl`; it does not connect to the database.
The export includes text hashes and model labels, and is compatible with the
existing vector loader. Interrupted runs resume from a `.jsonl.partial` file
after checking the snapshot and model labels still match. An existing completed
export is never overwritten.

The database must already contain populated embeddings with the matching label
and a valid HNSW index on `chunks.embedding` using `vector_cosine_ops`. The
existing `src.db.load_embeddings` command imports precomputed vectors and
creates that index. Retrieval checks these prerequisites and never writes to
the database. It disables sequential scans for its transaction to favor the
vector index; [pgvector documents this planner setting](https://github.com/pgvector/pgvector#why-isnt-a-query-using-an-index).

Inspect retrieval directly before using the agent:

```powershell
.\.venv\Scripts\python.exe -m src.rag.retriever "Who must file a federal income tax return?"
.\.venv\Scripts\python.exe -m src.agent
```

Results include each chunk's ID, stored citation, text, and RRF fusion `score`.
`vector_score` is cosine similarity, `bm25_score` is the keyword score;
`vector_rank` and `bm25_rank` show each route's rank (null if absent).
These scores are not confidence probabilities. The agent is instructed to cite
the returned sources and say when the passages do not support an answer.

## Web chat

With the same `.env`, PostgreSQL embeddings, and running Ollama instance used by
the CLI agent, start the backend from the repository root:

```powershell
python -m uvicorn src.app.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 to chat. Enter sends a message; Shift+Enter adds a
newline. New chat starts a separate conversation. Answers are displayed as plain
text, including the agent's citations. The page and API share the same origin.

`POST /api/chat` accepts `{"message": "Your question", "session_id": null}` and
returns `{"session_id": "...", "answer": "..."}`. Send the returned session ID
with subsequent messages to preserve conversation context, including tool results.

This local demo stores conversations in server memory. Use one worker; restarting
the server clears all sessions. Refreshing the page starts a new conversation.
An expired session returns HTTP 404; select New chat to continue.

Run the API tests without calling OpenAI, Ollama, or PostgreSQL:

```powershell
python -m unittest discover -s tests -p test_chat_api.py
```

## Guided interview

After starting an interview, select **+** beside the answer box to attach up to ten
text-based PDFs, PNG images, JPG/JPEG images, or UTF-8 text files (10 MB each).
Extraction begins when you select the files. While the files are processed, you can
type a reply or remove attachments from the draft. Select **Send** with or without
a reply to apply the remaining attachments. Attachments sent with a tax question
are also applied in that turn. The interview skips questions answered by
the submitted files and gives manually entered answers priority. Sent files appear in
the conversation and reopen with the saved interview.
Documents must show the tax year configured in `config/settings.json`. Upload
scanned pages as PNG or JPG images, or add a selectable text layer to the PDF.

This prototype is intended for fictional or redacted documents. Extraction uses the
existing OpenAI configuration (`OPENAI_API_KEY` and `TAX_AGENT_LLM`) and sends document
text or prepared images to the model. Uploaded originals and full document content are not kept in interview
state. Staged answers are available for 24 hours while the server is running. Answers
used in an interview and their source filenames appear in the interview history.

Start the server as above (`python -m uvicorn src.app.api:app --reload`) and open
http://127.0.0.1:8000/interview. Choose any priority topics and select **Start interview**.
Use the **Interview history** icon on the setup or conversation page to reopen a saved
interview. Use the pencil to rename an entry or the trash icon to delete it after confirmation.
Completed web interviews appear under **In progress** until the assistant selects **Submit to preparer**.
An optional note is saved with the submission and shown in the preparer's review detail.
On the assistant page, **Download summary PDF** exports the interview ID, its start time in UTC,
its title if set, and the collected data as field names and values.
On the preparer page, **Download review PDF** exports the saved review as it stands, including
interview details, questions and answers, follow-up items, sources, uploaded filenames, and the interview record.

The application uses an HTTP-only browser cookie to scope interview history to the
same browser profile and server hostname. Users sharing a browser profile also share
its interview history. Local development saves interviews to `logs/interviews.sqlite3`
by default. Set `TAX_AGENT_INTERVIEW_DB_PATH` for a different SQLite location or
`TAX_AGENT_INTERVIEW_DATABASE_URL` to use PostgreSQL. The configured role needs permission to create and manage the
interview table and index.

| Page | Who | What |
|---|---|---|
| http://127.0.0.1:8000/interview | assistant | One intake for every client: optional topics to ask first and a background note, then one question at a time with why it is asked, documents to request and official passages; summary at the end |
| http://127.0.0.1:8000/preparer | preparer | Interviews from the web page and from MCP, by status; summary, interview record, approve / return |

The intake is Form 13614-C plus topic modules opened by the answers (`playbooks/modules/`):
`side_gig`, `family`, `retiree`, `rental`, `student` (Pub 970, Pub 4012 Tabs J and L, Form 8843).
6 modules, 122 questions, 56 escalation rules, all **draft** (no preparer sign-off yet; see `docs/validation.md`).

Interviews started from your own AI assistant (MCP server) are saved in the same interview store under
the owner `mcp-assistant`, so the preparer reviews both on `/preparer`. Fictional cases only.

Use your own AI assistant instead of our API key: `docs/mcp.md` (ChatGPT desktop app, Claude Desktop,
Gemini CLI, Cherry Studio).

Settings in `.env` (see `.env.example` and `docs/architecture.md`):

```dotenv
OPENAI_API_KEY=            # Listwise ranking, model note reader and chat agent
TAX_AGENT_RETRIEVAL=on     # off = no DB / Ollama needed; questions show no official text
TAX_AGENT_LLM=on           # off = exact answers only
TAX_AGENT_INTERVIEW_DB_PATH=        # optional: SQLite file (default logs/interviews.sqlite3)
TAX_AGENT_INTERVIEW_DATABASE_URL=   # optional: PostgreSQL for interviews
```

Interview API:

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/playbooks` | | available scenarios |
| POST | `/api/interviews` | `{"scenario": "intake", "document_ids": []}` | first unanswered question or summary + sidebar snapshot |
| POST | `/api/interviews/{id}/answer` | `{"answer": "yes", "document_ids": []}` | sends text, staged files, or both, then returns the next question or summary |
| POST | `/api/interview-documents` | multipart form with one or more `files` parts | per-file extraction status and staged IDs |
| DELETE | `/api/interview-documents/{document_id}` | | removes a staged file's extracted answers |
| PUT | `/api/interviews/{id}/documents` | `{"document_ids": ["..."]}` | syncs the active attachments and returns the current question or summary |
| GET | `/api/interviews/{id}` | | snapshot + full history |
| GET | `/api/interviews/{id}/summary.pdf` | | PDF summary of a completed interview in this browser |
| GET | `/api/interviews` | | this browser's interview history |
| POST | `/api/interviews/{id}/submit` | `{"note": ""}` | move a completed web interview to the review queue |
| GET | `/api/preparer/interviews?status=waiting_for_preparer` | | interviews waiting for review (web and MCP), newest first |
| GET | `/api/preparer/interviews/{id}` | | snapshot, history, source, summary, question labels |
| GET | `/api/preparer/interviews/{id}/summary.pdf` | | full PDF review of the selected web or MCP interview, including in-progress interviews |
| POST | `/api/preparer/interviews/{id}/decision` | `{"approve": true, "note": ""}` | preparer decision on any interview |

Tests (no OpenAI, Ollama or PostgreSQL needed):

```powershell
python -m pytest tests -q
```

More: `docs/architecture.md` (structure and roadmap), `docs/validation.md` (how the playbook is checked),
`docs/knowledge-base.md` (documents and chunk format for RAG), `docs/demo.md` (final demo script).

## Status
| Part | State |
|---|---|
| Pub 17 parsing + chunking (`src/rag/chunk_pub17.py`) | done: 820 chunks, 14 chapters, TIP/CAUTION, citations |
| Table extraction (`src/rag/tables.py`) | done: 20 tables, IF/AND/THEN rows as sentences, raw rows saved for code |
| HNSW + BM25 hybrid retrieval | implemented: RRF fusion, final top-k 5; 200-case baseline completed, RRF Recall@5 87.00% (annotated chunk IDs) |
| Retrieval reranking | Listwise is the default; the supplied 122-question comparison reports pooled Recall@5 of 83.10% and Context Precision@5 of 94.51%. See the [evaluation comparison](docs/knowledge-base.md#retrieval-evaluation-comparison) for baselines and limitations. |
| Web chat UI + FastAPI | implemented: local, in-memory conversations |
| Workflow engine (`src/workflow/`) | implemented: modular intake, priority topics, trigger words, escalations, reminders, summary |
| Interview store (`src/db/interview_store.py`) | implemented: per-browser history; MCP interviews under owner `mcp-assistant`; the preparer page reads every owner |
| Preparer review (`/preparer`) | implemented: list by status from both channels, summary, record, approve / return |
| MCP server (`src/mcp_server/`) | implemented: user's own AI drives the interview, engine validates every answer |
| Playbooks | 6 modules, 122 questions; checked against 13614-C, Direct File knockouts, Pub 4012 scope, 9 cases |
| Interview PDF exports (`src/app/interview_pdf.py`) | concise data summary for assistants and full review report for preparers |
| Knowledge base | Pub 17 loaded; 22 documents to add (`docs/knowledge-base.md`) |
| Practitioner input | written feedback + interview with a Big Four tax advisor (business tax) done; USC VITA interview planned |

## Next
- 11/20: knowledge base batch 1 (7 documents); document extraction contract (stage 1 → interview facts)
- 11/23: knowledge base batch 2
- 11/30 – 12/1: final demo (`docs/demo.md`)
- Before any real use: login and roles, encryption and retention, MCP authentication, deployment, CI (`docs/architecture.md`)

## Rules
- Never use real client data (IRC 7216). Fictional cases only.
- All numbers come from code or IRS / FTB documents, never from the model.
- The next question is decided by code from the playbook; the model may only propose values
  that code validates.
- Only cite passages that retrieval actually returned; playbook references are "to verify".
- Keys live in `.env`, which is never committed. Interview recordings and transcripts are
  never committed.
