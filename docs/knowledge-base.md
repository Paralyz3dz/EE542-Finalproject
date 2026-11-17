# Knowledge base for the guided interview

As of 2026-11-16. Tax year 2025, federal (IRS) and California (FTB).

This file defines what goes into the retrieval knowledge base and in what format. The document
list is derived from the interview playbook: every question in `playbooks/modules/*/questions.json`
names the official document its rule comes from (field `source`). The knowledge base holds those
documents, so that each question can show the passage behind it and the summary can tell the
preparer which references were actually found.

## 1. What the knowledge base does, and does not do

| Does | Does not |
|---|---|
| Return official passages for a question's `retrieval_query`, shown as evidence under the question | Decide the next question: that is code reading the playbook |
| Let the engine check each playbook reference (`check_references`): found / not in the knowledge base / unchecked | Hold client documents or answers: nothing from a client ever enters this index |
| Answer free-text rule questions with a citation (later) | Supply amounts and thresholds by similarity search: tables go to `tax_tables` for exact lookup |

If retrieval fails, the interview continues without passages and the error is shown; the knowledge
base is never on the critical path.

## 2. Documents to embed (21)

All tax year 2025. `KB name` is the exact name the playbook uses. `Questions` = playbook questions
citing the document. Files follow the existing naming (`data/raw/irs/p17--2025.pdf`).

### Batch 1: needed first (6 documents, 79 of 104 question references)

| KB name | Questions | Modules | File | Source URL | Status |
|---|---|---|---|---|---|
| IRS Pub 17 | 44 | base_intake, family, retiree | `irs/p17--2025.pdf` | https://www.irs.gov/pub/irs-prior/p17--2025.pdf | in repo, embedded |
| IRS Schedule C Instructions | 10 | base_intake, side_gig | `irs/i1040sc--2025.pdf` | https://www.irs.gov/pub/irs-prior/i1040sc--2025.pdf | to download |
| IRS Pub 527 | 7 | base_intake, rental | `irs/p527--2025.pdf` | https://www.irs.gov/pub/irs-prior/p527--2025.pdf | to download |
| IRS Pub 334 | 7 | side_gig | `irs/p334--2025.pdf` | https://www.irs.gov/pub/irs-prior/p334--2025.pdf | to download |
| IRS Pub 503 | 6 | family | `irs/p503--2025.pdf` | https://www.irs.gov/pub/irs-prior/p503--2025.pdf | to download |
| IRS Pub 463 | 5 | side_gig | `irs/p463--2025.pdf` | https://www.irs.gov/pub/irs-prior/p463--2025.pdf | to download |

### Batch 2: IRS (10 documents)

| KB name | Questions | Modules | File | Source URL |
|---|---|---|---|---|
| IRS Form 1040 Instructions | 3 | base_intake | `irs/i1040gi--2025.pdf` | https://www.irs.gov/pub/irs-prior/i1040gi--2025.pdf |
| IRS Pub 575 | 3 | retiree | `irs/p575--2025.pdf` | https://www.irs.gov/pub/irs-prior/p575--2025.pdf |
| IRS Pub 970 | 2 | base_intake | `irs/p970--2025.pdf` | https://www.irs.gov/pub/irs-prior/p970--2025.pdf |
| IRS Pub 915 | 2 | retiree | `irs/p915--2025.pdf` | https://www.irs.gov/pub/irs-prior/p915--2025.pdf |
| IRS Pub 525 | 2 | side_gig | `irs/p525--2025.pdf` | https://www.irs.gov/pub/irs-prior/p525--2025.pdf |
| IRS Pub 505 | 1 | base_intake | `irs/p505--2025.pdf` | https://www.irs.gov/pub/irs-prior/p505--2025.pdf |
| IRS Pub 925 | 1 | rental | `irs/p925--2025.pdf` | https://www.irs.gov/pub/irs-prior/p925--2025.pdf |
| IRS Pub 544 | 1 | rental | `irs/p544--2025.pdf` | https://www.irs.gov/pub/irs-prior/p544--2025.pdf |
| IRS Pub 590-A | 2 | base_intake, retiree | `irs/p590a--2025.pdf` | https://www.irs.gov/pub/irs-prior/p590a--2025.pdf |
| IRS Pub 590-B | 1 | retiree | `irs/p590b--2025.pdf` | https://www.irs.gov/pub/irs-prior/p590b--2025.pdf |

The Schedule 1-A instructions (tips, overtime, car loan interest, senior deduction) are printed
inside the Form 1040 Instructions; there is no separate file.

### Batch 2: California FTB (5 documents)

| KB name | Questions | Modules | File | Source URL |
|---|---|---|---|---|
| FTB Schedule S Instructions | 2 | base_intake, rental | `ftb/2025-540-s-instructions.pdf` | https://www.ftb.ca.gov/forms/2025/2025-540-s-instructions.html (web page: save as PDF) |
| FTB Pub 1031 | 1 | base_intake | `ftb/2025-1031-publication.pdf` | https://www.ftb.ca.gov/forms/2025/2025-1031-publication.pdf |
| FTB Schedule CA (540) Instructions | 1 | base_intake | `ftb/2025-540-ca-instructions.pdf` | https://www.ftb.ca.gov/forms/2025/2025-540-ca-instructions.html (web page: save as PDF) |
| FTB 3514 Instructions | 1 | family | `ftb/2025-3514-booklet.pdf` | https://www.ftb.ca.gov/forms/2025/2025-3514-booklet.pdf (instruction pages only) |
| FTB 540 Booklet | 2 | base_intake, side_gig | `ftb/2025-540-booklet.pdf` | https://www.ftb.ca.gov/forms/2025/2025-540-booklet.pdf (instruction pages only) |

Before embedding, check that the cover or header says 2025. The 2026 editions appear in early 2027.

### Not embedded

Blank forms (Form 1040, schedules), Form 13614-C, Pub 4012 / Form 6744 / Pub 5101 (2026 editions,
used only for interview structure and wording), IRS ATS test scenarios, the Direct File extracts in
`data/external/directfile/`, and any client document. A document not cited by the playbook is added
only together with a playbook question that cites it, so the list above stays the single source.

## 3. Names and citations

Every chunk's citation starts with the KB name, then the year in brackets, then where in the
document, separated by commas. This is the format `src/rag/chunk_pub17.py` already produces (the first
line is a real Pub 17 citation; the others show the pattern, their page numbers are illustrative):

    IRS Pub 17 (2025), Chapter 1 Filing Information, 'Limit on amount of refund', p. 19
    IRS Pub 463 (2025), Chapter 4 Transportation, 'Standard Mileage Rate', p. 15
    IRS Schedule C Instructions (2025), 'Line 9. Car and truck expenses', p. C-6
    FTB Schedule CA (540) Instructions (2025), Part I Section B, 'Line 3', p. 5

The engine compares a citation with a playbook reference by dropping "(2025)", keeping the text
before the first comma and lower-casing it (`_doc_key` in `src/workflow/engine.py`). So:

| Citation starts with | Matches "IRS Pub 463"? |
|---|---|
| `IRS Pub 463 (2025), ...` | yes |
| `irs pub 463, ...` | yes (case is ignored) |
| `Publication 463 (2025), ...` | no |
| `2025 Publication 463, ...` or `p463--2025.pdf` | no |

## 4. Chunk format

Same record as the existing Pub 17 chunks and the `chunks` table in `src/db/schema.sql`, so
`src/db/load_chunks.py` and `src/rag/embed_chunks.py` load new documents without changes.

```json
{
  "id": "p463_2025_ch04_007",
  "tax_year": 2025,
  "source": "IRS Pub 463 (2025)",
  "part": null,
  "chapter_no": 4,
  "chapter": "Transportation",
  "section": "Car Expenses",
  "subsection": "Standard Mileage Rate",
  "minor": null,
  "runin": null,
  "type": "rule",
  "table_kind": null,
  "printed_pages": [15],
  "pdf_pages": [17],
  "citation": "IRS Pub 463 (2025), Chapter 4 Transportation, 'Standard Mileage Rate', p. 15",
  "n_words": 238,
  "text": "...",
  "embed_text": "Chapter 4 Transportation > Car Expenses > Standard Mileage Rate\n..."
}
```

| Field | Rule |
|---|---|
| `id` | `<file prefix>_2025_<chapter or part>_<nnn>`: `p463_2025_ch04_007`, `i1040sc_2025_ln09_001`, `ftb1031_2025_s02_004` |
| `source` | KB name + " (2025)" |
| heading fields | the heading path the chunk sits under; empty levels are null |
| `type` | `rule` / `tip` / `caution` / `whats_new` / `table` (TIP and CAUTION boxes are their own chunks) |
| `printed_pages` / `pdf_pages` | page numbers printed on the page / 1-based PDF page index |
| `citation` | section 3 |
| `text` | the passage, hyphenation joined, headers and footers removed |
| `embed_text` | heading path + newline + text; the only field that is embedded |

Cutting rules (as in `chunk_pub17.py`):

1. Cut at every heading and bold run-in heading; never across two sections.
2. Merge small pieces and split long ones to 150-400 words.
3. A table is its own chunk (`type: table`); its rows also go to `tax_tables` for exact lookup.
4. Line instructions (Schedule C, Form 1040, Schedule CA): each "Line N." heading is a section.
5. FTB booklets: embed the instruction pages only, not the form pages; the 3514 EITC tables go to
   `tax_tables`.

Embedding: Ollama `qwen3-embedding:0.6b`, stored in PostgreSQL + pgvector (unchanged).

## 5. Retrieval interface used by the engine

```python
search(query: str) -> {"matches": [{"citation": str, "text": str, "chunk_id": str, "score": float}, ...]}
```

- `query` is the question's `retrieval_query` (104 fixed queries, so results can be cached).
- The engine uses the first 3 matches and shows up to 600 characters of each `text`.
- On any failure (database stopped, Ollama off) raise an exception; do not return an empty list,
  or the engine would report the documents as missing from the knowledge base.
- Possible later extension: `search(query, sources=["IRS Pub 463"])` to search only the documents a
  question cites.

## 6. Acceptance checks

| Check | How | Pass |
|---|---|---|
| Names | every chunk's citation, cut by the rule in section 3, is a KB name in section 2 | 100% |
| Coverage | for each of the 104 questions, does one of the top 3 matches for its `retrieval_query` come from its `source` document (hit@3)? | report the number per batch; aim for 80% or more |
| End to end | run one intake interview; the summary field `references_not_in_kb` | empty after batch 2 |

When a playbook question gains a new `source`, add the document to section 2 and embed it.
