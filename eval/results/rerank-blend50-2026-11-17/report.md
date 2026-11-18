# Equal-weight RRF and reranker score blending

## Result and decision

One fixed 104-question round changed only the final scoring policy: independently min-max normalize RRF scores and BGE cross-encoder logits across the same ten candidates, then combine them with weight 0.5 each. Both retrieval routes retain ten candidates, RRF retains ten and the final output retains five.

| Metric | Paired pure reranker baseline | Equal-weight score blend | Change |
|---|---:|---:|---:|
| Primary-source hit@3 | 75/104 (72.12%) | 77/104 (74.04%) | +2 questions |
| Primary-source hit@5 | 85/104 (81.73%) | 81/104 (77.88%) | -4 questions; -3.85 percentage points |
| MRR@5 | 0.617788 | 0.590385 | -0.027404 |
| Expected-source coverage in the RRF pool | 96/104 (92.31%) | 96/104 (92.31%) | Unchanged |

One question gained coverage and five lost it. The interview uses all five passages, so hit@5 is the primary metric. The gain at rank three did not offset losses in the final five. This result applies to the measured normalization and fixed equal weights; no other weights were tested.

The predeclared policy requires a strict hit@5 improvement. After recording the completed round, `rerank_rrf_weight` was restored to 0.0, which preserves pure cross-encoder ranking and raw logit scores. Reranking remains enabled, with the RRF pool still ten. No additional live retrievals were run.

## Formula and implementation

For each query and each of the two score lists over ten candidates:

`normalized(s) = (s - minimum) / (maximum - minimum)`

A constant score list normalizes to all zeros. The final score is:

`blend_score = 0.5 * normalized_rrf_score + 0.5 * normalized_rerank_score`

Ties use original RRF rank, then chunk ID. Scores are not normalized across questions or against gold labels. Raw RRF scores, raw reranker logits, their normalized components, original ranks and final blended ranks are retained. Citations, complete passage bodies and duplicate provenance remain attached to original IDs. A zero RRF weight preserves the previous pure-reranker output contract and does not add blended score fields.

`reranker_before.py` preserves the exact pre-experiment implementation and matches its recorded baseline SHA-256. `implementation.diff` shows the scoring change. The as-run manifest retains weight 0.5; `decision.json` and `applied_settings.json` record the subsequent restoration to zero.

## Changed questions

| Change | Question ID |
|---|---|
| Gained | `q_had_disaster_loss` |
| Lost | `q_has_digital_assets` |
| Lost | `q_has_unemployment` |
| Lost | `q_had_credit_disallowed` |
| Lost | `q_has_farm_income` |
| Lost | `q_main_income_source` |

`source_change_audit.json` retains every expected-source candidate with both raw scores, normalized components and resulting ranks for these questions. It also retains baseline and blended top-five lists. This shows where incorporating RRF displaced source-matching passages; it does not establish that the displaced passages were semantically sufficient.

## Experimental controls

- Exactly one round of 104 unique production retrievals; no extra baseline retrievals, no retrieval/inference warmups and no answer generation or semantic judging.
- Both outputs use the same model scores from one scoring call per question, with the unchanged batch size of four. Pure-reranker baseline computation is additional sorting arithmetic, not another model inference pass.
- All 104 paired pure-reranker top-five ID lists and source metrics exactly reproduce the saved baseline. Vector candidates, BM25 candidates and RRF top-ten rankings match it exactly.
- Tax year, questions, cached query vectors, embedding model digest, corpus fingerprint, BGE revision and file hashes, dependency versions, CPU/float32, 12 threads, 8192-token capacity and complete untruncated passage inputs are fixed. Gold source labels are used only after scoring for metrics.
- Before/after remote corpus fingerprints match: 8,752 vectors across 21 sources, with 8,390 eligible chunks. No database writes, re-embedding, chunking changes or overlap additions occurred.
- Strict primary-document source coverage is unchanged as the metric. It does not measure answer accuracy, paragraph relevance or sufficient evidence.

## Timing

| Timing | Seconds |
|---|---:|
| Saved pure-reranker mean complete retrieval | 7.012 |
| This round mean before reranking | 2.965 |
| This round mean reranker stage, including score sorting | 3.636 |
| This round mean complete retrieval | 6.604 |
| This round complete retrieval p95 | 8.823 |

Model startup is recorded separately in `startup.json`. Timings exclude generating cached query embeddings and interview answers. The saved baseline and measured round ran at different times, so the observed latency difference does not establish a speed gain from score blending. No additional model scoring pass was added.

## Records and reproduction

`eval/run_rerank_blend_experiment.py` enforces the single changed scoring policy, retains a paired baseline from the same scores and refuses completed output directories. `results.jsonl` records all ten scored candidates, component scores, paired/final rankings, source metrics and timings. `passages.json` retains complete candidate passages; `summary.json`, `changes.json`, `corpus_after.json` and `tests.json` retain the result and checks.

Repeating this experiment requires separate authorization for a new round/output directory, the measured weight 0.5 and the preserved baseline implementation snapshot. The current default intentionally uses pure reranking with weight zero.

## Validation

The full offline suite passed before the live round at weight 0.5 (192 tests and 10 subtests, 15.81 seconds) and after restoring weight zero (192 tests and 10 subtests, 16.69 seconds). Checks cover score normalization, constant scores, stable ties, endpoint weights, invalid weights before inference, metadata preservation, the unchanged pure-reranker output contract, production candidate counts, a single scoring call and completed-round refusal. No live retrievals were issued by these tests.
