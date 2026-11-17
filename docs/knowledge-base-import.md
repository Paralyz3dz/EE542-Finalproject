# Knowledge-base import

The 21-document inventory and source names are defined in `knowledge-base.md`.
The multi-document pipeline preserves the existing Pub 17 records and adds the other
20 documents. All new records use tax year 2025 and Qwen3-Embedding-0.6B vectors
with 1024 dimensions. The loader performs an additive upsert in one transaction;
it does not remove existing chunks.

Run from the repository root with the project virtual environment:

```powershell
python scripts/prepare_knowledge_base.py
python -m src.rag.build_knowledge_base
python -m src.rag.embed_chunks data/processed/knowledge_base/chunks.json --out data/processed/knowledge_base/embeddings.jsonl --batch-size 8 --num-ctx 8192
python -m src.db.load_knowledge_base
python -m src.rag.check_knowledge_base
```

Downloads use the official URLs in the inventory. Existing local copies of Pub 503,
Pub 970 and the Form 1040 instructions are reused. The two FTB instruction webpages
are retained as HTML and converted to PDFs containing their main content. Citations
for these documents explicitly identify the generated PDF page instead of claiming
an official printed page number.

PDF extraction adapts the body font size to each document and uses the existing
heading, column, run-in, TIP/CAUTION and table handling from the Pub 17 cutter.
Chunks target 150–400 words. Short sections and standalone tables retain their
boundaries even when they contain fewer than 150 words. Longer passages are split
with a hard 400-word maximum. Empty table extractions are excluded.

FTB 3514 instruction pages 3–10 and EITC table pages 22–30 are included. FTB 540
instruction pages 3–23, 25–27, 29–33, 35–36 and 55–68, and tax-table pages 69–73 and 75
are included. Blank form pages are excluded. Schedule CA instructions are indexed
under their own source name as well as within the full 540 booklet. Table rows are also saved in `tax_tables`; numeric
PDF pages retain layout-derived raw cells and require table-specific interpretation
before use as a calculation API.

Local artifacts are stored under `data/processed/knowledge_base/`, which is ignored
by Git. The manifest records source URLs and local paths. JSON chunk and table
snapshots permit extraction review. Vector exports include text hashes and model
names and support resumption after interruption. The loader verifies metadata,
vector dimensions, finite nonzero values and text hashes before connecting to the
remote database configured in `.env`.

Numeric tables can exceed 2048 tokens despite staying below 400 words. Keep the
full 8192-token context and disable truncation. `--num-thread` optionally controls
Ollama CPU threads. If a batch fails, rerun the export command to resume from the
last completed batch; reduce the batch size if CPU requests time out.

`database_before.json` and `database_after.json` record per-document database counts.
`validation.json` records name compliance, stored vector counts, hit@3 for the
104 modular-intake queries and missing references in a baseline W-2 interview.
Presence in the database does not guarantee that a document appears among the
three highest-ranked results; the report measures that separately.

Existing API processes cache retrieved passages per query in `InterviewService`.
A fresh API process reads the expanded knowledge base for queries that were
previously cached.

## Import result (2026-11-17)

The remote import completed and committed successfully. All 21 sources are present,
with 6,004 chunks and 6,004 stored vectors. This run added 5,184 chunks and
800 raw table records; the previous 820 Pub 17 chunks are included in the total.
Chunk IDs, text hashes, vector dimensions and finite nonzero vectors were checked
before writing. Citation names matched the inventory in 100% of the new chunks.
The new chunks contain at most 399 words; 2,367 short chunks retain heading or
standalone-table boundaries instead of being merged into unrelated sections.

| Retrieval check | Result | Target |
|---|---|---|
| All intake questions, source hit@3 | 68/104 (65.4%) | At least 80% |
| Batch 1 questions | 51/79 (64.6%) | At least 80% |
| Batch 2 questions | 17/25 (68.0%) | At least 80% |
| Baseline interview references not returned | IRS Pub 17; IRS Pub 505; IRS Schedule C Instructions | Empty |

Data import is complete; retrieval acceptance remains unmet. The three baseline
references above are present in the database but were absent from the returned
top-three passages for at least one question. The engine's `references_not_in_kb`
field describes returned evidence, so it must not be interpreted as a database
inventory check. Source-aware retrieval or ranking improvements are separate work.

The 11 existing vector-export and retrieval unit tests passed. Python compilation
and `git diff --check` also passed. Detailed per-question results are available in
`data/processed/knowledge_base/validation.json`; the remote per-source counts are
in `data/processed/knowledge_base/database_after.json`.

## Initial retrieval diagnosis

This is a baseline for later optimization, not a claim that retrieval quality has
passed acceptance. The diagnostic queries reused the stored query vectors and
read the remote corpus without changing its records or production ranking.

| Diagnostic | Result |
|---|---|
| HNSW vector search, source hit@3 | 75/104 (72.1%) |
| BM25, source hit@3 | 74/104 (71.2%) |
| Current hybrid ranking, source hit@3 | 68/104 (65.4%) |
| Source found in the union of both top-five candidate lists | 93/104 (89.4%) |
| Exact vector search, source found in top 20 | 102/104 (98.1%) |
| HNSW/exact vector search top-five overlap | 518/520 (99.6%) |
| Top three results all from the same source | 29/104 questions |

Experimental larger candidate pools plus source diversification reached 84/104
(80.8%) on the source-name metric. That experiment has not been applied to the
retriever and does not establish passage relevance or answer accuracy.

The extraction also needs further cleanup: index entries were retrieved as body
passages, some column/worksheet text is out of order, and 528 new non-table chunks
contain fewer than 40 words. A short chunk is not automatically invalid, but these
observations identify extraction and segmentation work before the next corpus
revision. Related official documents can also be relevant while failing the
current single-source-name check.

Follow-up work: remove index/noise passages and repair extraction order, review
short chunks, evaluate candidate-pool sizing and reranking/diversity, and add
passage-level relevance labels alongside source coverage.

Committed baseline artifacts are under
[`eval/results/knowledge-base-2026-11-17`](../eval/results/knowledge-base-2026-11-17):
`manifest.json`, `database_inventory.json`, `validation.json` and
`retrieval_diagnosis.json`. Local vector exports remain under `data/processed/`.
