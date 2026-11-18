# Materials and chunking repair: one retrieval round

All 21 tax-year 2025 documents were rebuilt and loaded into independent remote tables. The repaired corpus is now the default in the retrieval worktree. The original public corpus remains unchanged.

## Results

Exactly one round of 104 fixed questions was scored. There were no warm-up retrievals, repeated rounds, generated answers or semantic judges. Query vectors are the same validated vectors used by the saved baseline. Candidate K=10, output=5, display=3, and RRF=60 remain fixed.

| Metric | Saved cleaned baseline | Repaired corpus |
|---|---:|---:|
| Primary-source hit@3 | 72/104 (69.23%) | 76/104 (73.08%) |
| Primary-source hit@5 | 80/104 (76.92%) | 82/104 (78.85%) |
| MRR@5 | 0.5696 | 0.5873 |

Hit@3 improved by 4 questions, or 3.85 percentage points. There were 9 gained cases and 5 lost cases. Coverage including preserved duplicate-source citations was also 76/104 at 3 and 82/104 at 5. The 80% primary-source target is still unmet: at least 84/104 would be required.

Mean production retrieval time excluding query embedding was 2.832 seconds; p95 was 3.239 seconds. This round had no warm-up and provides an exploratory timing measurement.

## Material changes

- Original PDF columns, font spans, wrapped headings and chapter/run-in paths replace the old word/font normalization. Native PDF grids and original cell rectangles preserve table rows and columns. Saved HTML instructions use real web sections.
- Front contents, reviewed back indexes, reference catalogs, proof headers and isolated form labels are removed. Same-page help and worksheets are retained and checked against the source PDFs.
- Paragraphs merge within their own section. Sentence/clause splits preserve words, publication/country abbreviations and list numbering. Every final passage is at most 400 words. Semantic table pieces retain whole rows and repeat column headings; complete logical tables remain in exact storage.
- Exact duplicate passages keep all source citations in provenance. Similar wording with different conditions is retained.
- The guide and evidence panel show complete chunks, including evidence beyond the former 600-character limit.

## Inventory and checks

- 21 sources, 8,752 chunks and vectors, 8,390 searchable chunks, 554 complete exact tables.
- 362 numeric/oversized table locators are excluded from both candidate routes. The partial cosine HNSW index was confirmed with EXPLAIN only; this executed no extra retrieval.
- 160 exact duplicate groups were consolidated. Per-document counts in quality.json precede consolidation; corpus totals follow consolidation.
- No reviewed index prompts, Useful Items catalog prompts, blank FTB form pages or split words remain in the checked snapshot. Protected body and worksheet examples passed all 7 checks.
- 151 offline tests and 10 subtests passed. The configured version-routing test verifies the same effective settings used by this round. No additional live smoke queries were run.
- The original 6,004-row corpus retained its content/vector fingerprint through the import and evaluation. All 8,752 new vectors have 1,024 dimensions and the pinned baseline Qwen model identity.

The as-run manifest retains the pre-promotion configuration checksum. The evaluator explicitly selected the repaired corpus; applied_settings.json matches its effective_settings exactly. Promotion followed the completed round and required no additional live query.

The repaired default uses public.rag_materials_v2_chunks and public.rag_materials_v2_tax_tables. Restore corpus_version=original and exclude_reference_chunks=true together to use the prior default.

## Limits and artifacts

Source hit measures whether a named official document appears, not whether its paragraph answers the question or supplies sufficient evidence. Those semantic evaluations remain unimplemented as requested. 7,542 genuine short sections or table locators remain below the 150-word target; heading and run-in boundaries take precedence. The repair does not enforce a 150-word minimum.

The default was selected because the primary-source metric did not regress and both secondary metrics improved. This is a completed material-processing experiment, with further retrieval improvement still needed to reach the 80% source target.

- manifest.json and questions.json: frozen inputs, model/query-vector hashes, corpus/index fingerprint and effective settings.
- results.jsonl: exactly 104 scored retrievals; passages.json: their complete returned passages.
- summary.json, changes.json and misses_at_3.json: aggregate results, all changed metrics and the remaining 28 misses at 3.
- processing_audit.json and quality.json: material checks, source hashes, counts and duplicate provenance.
- remote_load.json, embedding_identity.json and index_validation.json: remote import, pinned embedding setup and actual HNSW plan.
- promotion_policy.json, decision.json and applied_settings.json: predeclared selection rule and applied default.
