# Intake candidate-pool experiment: 2026-11-17

Only the common per-route candidate count changed. All 104 queries used the complete 21-source, 6,004-chunk corpus, Qwen3-Embedding-0.6B, RRF constant 60, five returned matches and three displayed matches.

Three repetitions used rotated condition order, three excluded warmup queries per condition and shared fingerprinted query vectors. All nine runs completed: 936 scored retrieval calls for 104 distinct questions. No answer-generation or judge API calls were made.

Corpus/vector hashes and index settings matched throughout. Fresh K=5 results reproduced the historical source-hit flags for every question in all three repetitions.

## Source coverage

| Candidate K | Hit@3 | Hit@5 | Source MRR@5 | Candidate source coverage |
|---|---|---|---|---|
| 5 | 68/104 (65.4%) | 81/104 (77.9%) | 0.5655 | 89.4% |
| 10 | 71/104 (68.3%) | 81/104 (77.9%) | 0.5760 | 96.2% |
| 20 | 70/104 (67.3%) | 83/104 (79.8%) | 0.5732 | 99.0% |

All three repetitions produced the same per-question Hit@3/Hit@5 flags. Repetitions are not additional independent test questions.

## Retrieval latency

| Candidate K | Mean seconds | P95 seconds |
|---|---|---|
| 5 | 2.560 | 2.835 |
| 10 | 2.592 | 2.951 |
| 20 | 2.594 | 3.007 |

Times include production database access, full-corpus loading and BM25 construction, with serial requests. Query embedding, generation and judge calls are excluded. Host and database load were not controlled; these are descriptive timings, not a claim of statistical significance.

## Decision

Adopt K=10: it has the highest Hit@3 among settings whose Hit@5 did not regress. K=10 improves source Hit@3 by 3 questions, or 2.9 percentage points, over K=5; Hit@5 remains 81/104. Mean latency rises by about 0.032 seconds and P95 by about 0.116 seconds. K=20 provides better candidate/Top-5 source coverage but lower Hit@3 than K=10.

The 80% acceptance target is still unmet: 71/104 versus the required 84/104. With K=10, the target source enters the candidate pool for 100/104 questions but reaches the top three for only 71/104. Ranking and evidence quality remain follow-up work.

The default setting was applied only after the complete validated experiment. The experiment manifest records the pre-adoption configuration; applied_settings.json records the selected runtime configuration.

## Paired evidence review

The selected setting gains four specified-source hits and loses one, net +3. The cases below are qualitative inspection notes, not graded relevance labels. Full before/after ranks are in changes.json; returned passages are keyed by chunk ID in passages.json.

| Question | Change | Inspection note |
|---|---|---|
| q_has_self_employment | gain | The selected Schedule C passage mixes several update topics. Gig-economy wording appears later in the block and may lie outside the interview guide display of the first 600 characters. |
| q_has_retirement_income | loss | Pub 17 drops out of the first three. The replacement set includes a pension passage in Form 1040 instructions but also a Pub 575 reference list and fragmented material; this remains a quality-review case. |
| q_had_credit_disallowed | gain | The selected Pub 17 passage directly mentions disallowed credits and Form 8862; this is a topically strong source-coverage gain. |
| q_distribution_account_type | gain | The gained Pub 575 passage discusses qualified disaster recovery distributions and a particular box-7 code. Its scope is narrower than the general account-type question. |
| q_retirement_distributions | gain | The gained Pub 575 passage begins with worksheet fragments and then early-distribution exceptions. Matching the expected document does not establish useful general gross/taxable-amount evidence. |

## Limits and next experiment

Source-name hit rate is not passage relevance or answer accuracy. Some source gains retain extraction noise or overly narrow context. No corpus cleaning, embedding change, source filter, diversification or fusion-weight change was made in this experiment.

A next isolated ranking experiment should freeze K=10 and compare against both this selected baseline and historical K=5. The raw results preserve HNSW/BM25 candidates, fused ranks, per-question gains/losses, latency and the frozen inputs. The old 200-question agent dataset was not evaluated.

## Implementation checks

All 29 candidate-evaluation, retriever, hybrid and vector-export unit tests passed.
Two read-only smoke queries verified that the applied default requests ten
candidates from each route, returns five passages and matches the measured K=10
output. Their details are in default_smoke.json.
