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
7. The preparer approves it or returns it with a note

## Quick start
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

Start the server as above (`python -m uvicorn src.app.api:app --reload`) and open
http://127.0.0.1:8000/interview. Choose a scenario and select Start interview.

- Left: the conversation. Each question shows why to ask it, documents to request,
  playbook references (✓ found in the knowledge base / ⚠ not in the knowledge base yet),
  and the retrieved official text. Quick-answer buttons match the fact type
  (Yes/No, options); free-text notes are read by the LLM and validated by code.
- Right: status, escalation flags, documents to collect, facts collected.
- End: summary for the preparer with Approve / Return to assistant.

Scenarios (`playbooks/`, all **draft**, not yet confirmed by a preparer):

| Scenario | Questions | Notes |
|---|---|---|
| `ca_w2_side_gig` | 3 | W-2 + side income; being expanded from the practitioner interview |
| `ca_w2_family` | 14 | dependents, childcare, student loans, HSA (CA add-back), renter's credit |
| `ca_retiree` | 12 | Social Security (not taxed by CA), pension / IRA, early distribution |
| `ca_rental` | 11 | rental income, personal use, short-term rental, out-of-state property |

Add a scenario by adding a folder with `scenario.json` and `questions.json`; the engine
does not change. `tests/test_workflow.py` checks every playbook's structure and runs a
full interview with each one.

Settings in `.env` (see `.env.example`):

```dotenv
OPENAI_API_KEY=            # LLM note reader and the chat agent
TAX_AGENT_RETRIEVAL=on     # off = no DB / Ollama needed; guides show no official text
TAX_AGENT_LLM=on           # off = exact answers only, never call the model
```

If the chat agent cannot start (e.g. no `OPENAI_API_KEY`), only `/api/chat` returns 503;
the interview still works. If retrieval fails, the interview continues without passages.

Interview API (state is kept in server memory; restarting clears interviews):

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/playbooks` | | available scenarios |
| POST | `/api/interviews` | `{"scenario": "ca_retiree"}` | first question + sidebar snapshot |
| POST | `/api/interviews/{id}/answer` | `{"answer": "yes"}` | next question, an error (same question again), or the summary |
| GET | `/api/interviews/{id}` | | snapshot + full history |
| POST | `/api/interviews/{id}/decision` | `{"approve": true, "note": ""}` | preparer decision |

Run the workflow tests (no OpenAI, Ollama, or PostgreSQL needed; retrieval and the
model are stubbed):

```powershell
python -m pytest tests/test_workflow.py tests/test_interview_service.py tests/test_interview_api.py -q
```

## Status
| Part | State |
|---|---|
| Pub 17 parsing + chunking (`src/rag/chunk_pub17.py`) | done: 820 chunks, 14 chapters, TIP/CAUTION, citations |
| Table extraction (`src/rag/tables.py`) | done: 20 tables, IF/AND/THEN rows as sentences, raw rows saved for code |
| HNSW + BM25 hybrid retrieval | implemented: RRF fusion, final top-k 5; 200-case baseline completed, RRF Recall@5 87.00% (annotated chunk IDs) |
| Web chat UI + FastAPI | implemented: local, in-memory conversations |
| Workflow engine (`src/workflow/`) | implemented: code-chosen questions, ask_when follow-ups, escalation flags, required documents, summary, preparer approval |
| LLM note reader (`src/llm/`) | implemented: OpenAI via one adapter; model proposes, rules validate |
| Citations vs references | implemented: summary lists only retrieved citations; playbook references marked found / not in KB / unchecked |
| Guided interview page (`/interview`) | implemented: 4 draft scenarios, end-to-end with RDS + Ollama + OpenAI |
| Practitioner input | written feedback + interview with a Big Four tax advisor (business tax) done; USC VITA interview planned |
| California FTB documents in the knowledge base | next |

## Next (target: system ready Nov 23)
- Knowledge base: Schedule C / SE instructions, Pub 334, Pub 463, Pub 505, FTB 540 Booklet,
  Schedule CA (540) instructions, FTB Pub 1001; citation names match playbook `source` names
- Playbooks: expand `ca_w2_side_gig` from the interview (income split, expenses, records,
  hobby vs business); update all references; add record-keeping reminders
- Interview page: collapsible explanations, "send to preparer" button
- Evaluation: questions for the new documents; fictional end-to-end client cases

## Rules
- Never use real client data (IRC 7216). Fictional cases only.
- All numbers come from code or IRS / FTB documents, never from the model.
- The next question is decided by code from the playbook; the model may only propose values
  that code validates.
- Only cite passages that retrieval actually returned; playbook references are "to verify".
- Keys live in `.env`, which is never committed. Interview recordings and transcripts are
  never committed.
