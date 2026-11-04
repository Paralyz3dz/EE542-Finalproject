# Tax Agent MVP

A digital clone of a tax preparer, used by an assistant: it guides the client interview
(what to ask next and why), explains options in plain language with IRS citations,
flags cases that need the preparer, and writes a summary for the preparer to approve.
It does not file returns or give final tax advice. Only public data and fictional cases.

## Folders
| Folder | What goes here |
|---|---|
| `config/` | Settings (models, thresholds, paths) |
| `data/raw/irs/` | IRS PDFs: Pub 17 (2025), Pub 5101, Form 13614-C |
| `data/raw/ats/` | IRS MeF ATS test scenarios (fictional taxpayers) |
| `data/processed/` | Parsed text chunks, embeddings exports |
| `data/synthetic/` | Fictional client profiles we write |
| `playbooks/` | Interview playbooks (JSON): questions, follow-ups, escalation rules |
| `src/ingest/` | PDF parsing and chunking |
| `src/rag/` | Embedding, vector search, reranking |
| `src/workflow/` | Interview workflow engine |
| `src/app/` | FastAPI backend + Streamlit UI |
| `eval/` | Test cases, TurboTax comparison, metrics |
| `models/` | Fine-tuned embedding model (not committed) |
| `logs/` | Run logs (not committed) |

## Workflow
1. Read client documents and extract fields
2. Match scenario playbooks
3. Guide the interview: next question + why
4. Explain options with IRS citations (RAG)
5. Flag cases for the preparer
6. Summarize
7. Preparer reviews and approves

## Quick start
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                 # then fill in your own keys (never commit .env)

python src/rag/chunk_pub17.py          # Pub 17 -> data/processed/pub17_2025_chunks.json + _tables.json
python tests/test_tables.py            # table tests, should print 11/11 passed
python src/rag/tables.py --pages 9 101 # inspect the tables on single pages
```

## Status
| Part | State |
|---|---|
| Pub 17 parsing + chunking (`src/rag/chunk_pub17.py`) | done: 821 chunks, 14 chapters, TIP/CAUTION, citations |
| Table extraction (`src/rag/tables.py`) | done: 20 tables, IF/AND/THEN rows as sentences, raw rows saved for code |
| BM25 retrieval + Recall@5 evaluation | next |
| Playbooks, workflow engine, Claude, UI | planned |

## Rules
- Never use real client data (IRC 7216). Fictional cases only.
- All numbers come from code or IRS documents, never from the model.
- Keys live in `.env`, which is never committed.
