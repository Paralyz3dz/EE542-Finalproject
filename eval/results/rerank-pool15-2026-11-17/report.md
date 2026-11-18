# RRF reranking pool: ten versus fifteen

## Result and decision

One fixed 104-question round changed only `retrieve_fusion_k` from 10 to 15. Vector search and BM25 each still return ten candidates; the same pinned BGE cross-encoder returns five passages.

| Metric | RRF 10 -> rerank 5 baseline | RRF up to 15 -> rerank 5 |
|---|---:|---:|
| Primary-source hit@3 | 75/104 (72.12%) | 75/104 (72.12%) |
| Primary-source hit@5 | 85/104 (81.73%) | 85/104 (81.73%) |
| MRR@5 | 0.617788 | 0.620353 |
| Expected-source coverage before reranking | 96/104 (92.31%) | 100/104 (96.15%) |
| Mean complete retrieval seconds | 7.012 | 8.489 |

Two questions gained coverage and two lost it, leaving the primary metric unchanged. MRR@5 increased by 0.002564. The larger pool found the expected source for four additional questions before reranking, but this did not produce a net gain in the final five.

Mean retrieval latency increased by 1.477 seconds (21.1%). The tested mean includes 3.049 seconds before reranking and 5.435 seconds in the reranker; total p95 is 12.265 seconds. Baseline latency comes from the saved previous round, rather than a second live baseline run. Both rounds reuse query vectors and exclude query-embedding generation, model startup and answer generation.

The predeclared policy requires a strict hit@5 improvement to retain the larger pool. After completing and recording the round, `retrieve_fusion_k` was restored to 10; reranking remains enabled. No further live evaluation was run. `applied_settings.json` exactly matches the previous measured RRF-ten effective settings.

## Changed questions

| Change | Question ID |
|---|---|
| Gained | `q_large_medical_expenses` |
| Gained | `q_installed_energy_efficient_items` |
| Lost | `q_receives_social_security` |
| Lost | `q_received_income_without_form` |

`source_change_audit.json` records expected-source positions and the old/new final-five rankings. `changes.json` records all changed source metrics, including rank-only changes.

## Experimental controls

- Exactly one scored round: 104 unique questions, one production retrieval per question, no retrieval or inference warmups, no extra baseline queries.
- The unchanged two candidate routes and RRF top-ten prefix exactly reproduce the saved previous round for every question. The runner checks this before reranking each query.
- The full configuration equals the previous effective configuration except for `retrieve_fusion_k`. RRF constant 60, final output five, BGE model revision, file fingerprints, batch size 4, CPU/float32, 12 threads, 8192-token capacity and dependency versions are fixed. Full `chunk.embed_text` inputs are not truncated, and gold source labels do not enter model scoring.
- There are at most fifteen unique candidates after deduplication. Actual pool sizes were 11 for 4 questions, 12 for 7, 13 for 14, 14 for 15 and 15 for 64. No duplicate candidates or unrelated padding were added.
- Corpus, question, query-vector and model identities match the baseline. The remote corpus fingerprint matches before and after: 8,752 vectors across 21 sources, with 8,390 searchable chunks. No database writes, re-embedding, chunking changes or overlap additions occurred.
- Source coverage is the unchanged strict primary-document metric. It does not establish passage relevance, sufficient evidence or answer correctness; no answer generator or semantic model judge was run.

The as-run `manifest.json` and code fingerprints retain RRF 15. `decision.json` and `applied_settings.json` record the subsequent restoration to RRF 10. The earlier baseline artifacts remain unchanged.

## Reproduction and records

`eval/run_rerank_pool_experiment.py` enforces the single changed setting and refuses completed output directories. Repeating this experiment requires a separately authorized new round/output directory and the measured RRF-15 configuration. The current default intentionally remains RRF 10.

`results.jsonl` stores both route rankings, the complete fused pool, saved baseline matches, new final matches, source metrics and timings. `passages.json` stores complete candidate passages. `summary.json`, `startup.json`, `corpus_after.json` and `tests.json` contain aggregates and validation evidence.

## Validation

The full offline suite passed with RRF 15 before the live round (178 tests and 10 subtests, 17.24 seconds) and again after restoring RRF 10 (178 tests and 10 subtests, 16.19 seconds). Checks cover promotion from RRF rank fifteen, overlap-driven pools smaller than fifteen, duplicate/capped candidate counts, baseline accounting and completed-round refusal. These validations issue no live retrievals.
