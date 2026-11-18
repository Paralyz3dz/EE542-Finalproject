# Corpus hygiene: index and reference-block quarantine

This change removes reviewed index and pure publication/form-reference blocks from
retrieval eligibility. It keeps the K=10 candidate setting, RRF=60, five returned
passages, three displayed passages, the same embeddings and the same 104 queries.
Column order, chunk boundaries and display truncation are separate work.

## Reviewed policy

`config/retrieval_exclusions.json` records 440 exact chunk IDs, their embedding-text
content hashes, SHA-256 text fingerprints and reasons: 370 index blocks, 62 pure
reference-list blocks and eight reference-directory blocks. The source-page and
full-text audit is in `eval/results/corpus-hygiene-2026-11-17/exclusion_audit.json`.
The policy was assembled from source pages and block content, not question-source
labels or retrieval-score optimization.

The PDF index headings were inspected, including headings that are only 10pt.
Index continuations were checked against the actual final pages. Form 1040's index
ends on PDF page 124; taxpayer rights on page 125 and mailing instructions on page
126 are retained. Pub 544 pages 61?62 contain its index; pages 63?64 are publication
and form directories and have a separate reason. Other recorded index spans extend
to their publications' final page. The policy's `page_count` is the source PDF's
length, not an instruction to discard every page after the heading.

First index pages can mix index entries with substantive text, help passages or
worksheets. Cross-boundary and explicitly protected chunks are retained. Merely
having a `Useful Items` heading is insufficient: several chunks with that heading
also contain genuine rules because extraction heading state carried over. Those
mixed chunks are retained. Short chunks and numerical tables are not blanket-filtered.

Original database records, vectors, chunk IDs and raw table rows remain intact for
rollback and comparison. No schema migration or new embedding generation is needed.
Only IDs whose current content and text fingerprints match are quarantined. Changed
content is retained and reported as stale rather than silently suppressed. This
versioned policy must be re-audited after re-chunking; it is not a universal text classifier.

## Retrieval and processing

`exclude_reference_chunks` controls eligibility; older configurations without the
key preserve the old behavior. The retriever accepts an explicit boolean override
for paired tests. Eligible chunks are selected before building the BM25 index.
Vector SQL applies bound ID exclusions before its ORDER BY/LIMIT, allowing eligible
candidates to fill the requested K rather than filtering the final five passages.
Malformed policies fail instead of silently disabling cleanup.

The document builder also writes `retrieval_chunks.json`, `retrieval_tables.json`
and `hygiene_report.json` alongside its existing raw snapshots. The raw outputs and
additive database loader stay compatible with the previous import. The cleaned
exports omit reviewed blocks and their linked reference/index table records; actual
worksheets are protected. The original records remain the audit source, while
production retrieval applies the same eligibility policy to the remote corpus.

## Controlled comparison

```powershell
python -m eval.run_hygiene_experiment --env-file ../EE542-Finalproject/.env --out eval/results/corpus-hygiene-rerun
```

The evaluator reuses the exact previous fingerprinted query vectors and verifies
that the physical remote corpus matches the candidate-pool experiment. Baseline
cleanup=False and treatment cleanup=True each run three times, in alternating
order, with three excluded warmup queries per condition. Both use K=10 and call the
production retriever; full corpus loading and BM25 construction are timed. There
are no answer-generation or judge-model calls and no remote database writes.

Corpora, queries, code, configuration and policy are checked before/after each
condition. Invalidated runs require a new output directory. The evaluator records
source Hit@3/Hit@5/MRR@5, reviewed noise in the top three, actual candidate lengths,
latency, paired changes and full returned passages. Source coverage is still a
provenance metric; eliminating an index hit can lower it without establishing a
loss of substantive evidence. Passage relevance is not automatically judged.

Completed results, source Hit@5/MRR regressions and the adoption rationale are in
[the experiment report](../eval/results/corpus-hygiene-2026-11-17/report.md).
The historical import checker hard-codes unfiltered K=5; use the paired hygiene
runner for this change, not that historical checker.
