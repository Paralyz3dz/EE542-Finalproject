# Corpus hygiene experiment: 2026-11-17

This round only changes index/reference-block eligibility. Both conditions keep K=10, RRF=60, five returned matches, three displayed matches, the same 104 queries and the exact previous cached query vectors. All 21 source documents remain eligible.

The physical remote corpus and vectors remain unchanged at 6,004 chunks. A fingerprinted policy quarantines 440 blocks: 370 index blocks, 62 pure reference lists and eight publication/form-directory blocks. The eligible view contains 5,564 chunks. Full excluded text and source-page locations are in exclusion_audit.json; mixed rules, worksheets and taxpayer-help content are retained.

## Controlled results

Two conditions each ran three times in alternating order: 624 scored retrieval calls for 104 distinct questions, plus excluded warmups. All baseline per-question Hit@3/Hit@5 flags reproduce the preceding K=10 experiment. No embedding generation, answer-generation, judge-model calls or remote database writes were made.

| Metric | K=10 baseline | K=10 with quarantine |
|---|---|---|
| Source Hit@3 | 68.3% | 69.2% |
| Source Hit@5 | 77.9% | 76.9% |
| Source MRR@5 | 0.5760 | 0.5696 |
| Mean retrieval seconds | 2.5997 | 2.6331 |
| P95 retrieval seconds | 3.0460 | 3.0905 |
| Questions with reviewed noise in Top 3, per repetition | 7/104 | 0/104 |
| Reviewed noise passages in Top 3, per repetition | 10 | 0 |
| Vector candidate lists shorter than 10, across all repetitions | 0 | 0 |

Each repetition produced Hit@3=71/104 and Hit@5=81/104 before cleanup, versus Hit@3=72/104 and Hit@5=80/104 after cleanup. Repetitions are not independent new questions. Timing includes production database access, corpus loading, eligibility checks and BM25 construction; excludes query embedding and generation. Host/database load was not controlled.

## Decision and limitations

Enable the reviewed quarantine as a structural evidence-hygiene correction: all identified noise blocks are removed from both candidate paths before selection, while protected substantive blocks remain. This is not an across-metric ranking improvement: source Hit@5 and MRR@5 regress slightly. The 80% source-coverage target remains unmet. The previous candidate-size non-regression criterion must not be reported as passed by this cleanup run.

The source Hit@5 regression is q_is_full_time_student. Its genuine Pub 17 passage pub17_2025_ch03_016 was not excluded and remains the fifth vector candidate. It leaves the final top five after fusion favors Pub 970 passages; the leading replacement discusses support and full-time-student status. This is an observed ranking/source-coverage tradeoff, not proof that passage relevance or sufficiency improved or declined.

Raw records, vectors and IDs are retained. Setting exclude_reference_chunks=false restores the unquarantined path. The default is enabled only after these completed comparisons and the source-page/fingerprint audit; applied_settings.json records the change. Production-schema migration and re-embedding are unnecessary.

## Processing exports and remaining work

The builder writes separate retrieval_chunks.json, retrieval_tables.json and hygiene_report.json. On the existing 20 newly imported documents, the export keeps 4,756 of 5,184 chunks and 605 of 800 table records, with no stale exclusions. Together with 808 eligible Pub 17 chunks, this yields the 5,564-chunk view. Original raw snapshots and the additive loader remain intact.

Eleven mixed reference/introductory blocks are intentionally kept. Column order, heading recognition, chunk boundaries, mixed-block splitting, duplicate content and 600-character display truncation are not repaired in this round. This policy is versioned to the reviewed corpus, not a universal classifier; changed content is retained pending re-audit.

summary.json contains aggregate results; results.jsonl records all runs; passages.json stores returned text. metric_changes.json covers Hit@3, Hit@5 and noise changes, including the student regression. changes.json is the evaluator subset for Hit@3/noise changes. No semantic relevance or answer-quality judge was run.

## Implementation checks

All 46 relevant unit tests passed, including policy fingerprints, protected
worksheets/mixed rules, SQL parameter binding, pre-limit filtering, evaluator
completeness and observer compatibility. No old 200-question dataset was run.
Two read-only default-setting smoke queries requested K=10, excluded all 440
reviewed blocks, indexed 5,564 eligible chunks, returned five matches and reproduced
the cleaned experiment outputs. Details are in default_smoke.json.
