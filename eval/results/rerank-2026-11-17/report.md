# Cross-encoder reranking experiment

## Result and decision

The requested pipeline is vector top 10 + BM25 top 10 -> RRF top 10 (constant 60) -> BGE reranker top 5. The default in the retrieval worktree now enables reranking because primary-source hit@5 strictly improved on the fixed 104-question evaluation. This follows the selection policy recorded before the round.

| Metric | Same-pool RRF top 5 | Reranked top 5 | Change |
|---|---:|---:|---:|
| Primary-source hit@3 | 76/104 (73.08%) | 75/104 (72.12%) | -1 question |
| Primary-source hit@5 | 82/104 (78.85%) | 85/104 (81.73%) | +3 questions; +2.88 percentage points |
| MRR@5 | 0.587340 | 0.617788 | +0.030449 |

Six questions gained source coverage and three lost it. The interview uses all five passages, so hit@5 is the primary metric. The RRF top-ten pool contained an expected source for 96/104 questions (92.31%); reranking retained it in the final five for 85 questions. This is an observed improvement on the existing fixed set, without a separate held-out set.

| Timing on local CPU | Seconds |
|---|---:|
| Mean retrieval before reranking | 2.896 |
| Mean reranker stage | 4.112 |
| Mean complete retrieval | 7.012 |
| Complete retrieval p95 | 10.092 |
| Model startup in this evaluation process | 10.553 |

Model startup is excluded from per-question timings; production loads the model lazily once per process. Startup depends on filesystem caching. Cached query embeddings were reused, so these timings exclude generating query embeddings and generating interview answers. The baseline timing is the prefix of the same call, including trace instrumentation, rather than a separately timed baseline round.

## Model and inputs

The existing configured model is [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3), a cross-encoder scoring query/passage pairs. Its immutable revision is `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. All six local model/tokenizer files have SHA-256 identities in `manifest.json`. Inference uses CPU, float32, 12 threads, batches of 4 and an 8192-token capacity. Inputs are not truncated; an oversized input raises an error.

Each pair contains the original retrieval query and the complete stored `chunk.embed_text` (heading path plus body). Neither playbook source labels nor gold references enter scoring. Final citations, body text and duplicate provenance remain attached to their original chunk IDs. `score` is the raw cross-encoder logit, with original RRF score/rank retained separately. Failures propagate to the existing retrieval error handling without a silent RRF fallback.

## Controls and audit

- Exactly one scored round: 104 unique questions, one production retrieval per question, zero retrieval warmups and zero answer-generation/model-judge calls.
- Candidate counts, RRF constant, questions, query vectors, embedding model digest, corpus, chunking and eligibility remain fixed. No overlap was added.
- The same RRF top-ten output supplies the paired baseline by taking its first five entries, without additional live baseline queries. All 104 baseline top-five ID lists and metrics exactly reproduce the saved materials-v2 evaluation.
- The remote corpus is unchanged: 8,752 vectors across 21 sources, with 8,390 eligible chunks. Before/after corpus fingerprints match. No database writes or document re-embedding occurred; the original 6,004-chunk corpus is also untouched.
- Primary-document metrics use the unchanged source-name match; additional provenance does not inflate them. They do not measure paragraph relevance, sufficient evidence or answer correctness.
- The as-run manifest preserves `rerank_enabled: false` in raw configuration and `true` in effective settings because the experiment explicitly enables it. Only after completion was the worktree default promoted to `true`; `applied_settings.json` equals the measured effective settings.

`results.jsonl` retains route candidates, paired rankings, timings and per-question metrics. `passages.json` retains the original ten candidate passages. `changes.json` records every question with a changed source metric. `summary.json`, `startup.json`, `corpus_after.json` and `decision.json` record the aggregates and decision.

## Reproduction

Prepare the pinned local model once using the existing environment file:

```powershell
python -m src.rag.download_reranker --env-file ../EE542-Finalproject/.env
```

The model files are ignored under `models/` and are not committed. Normal retrieval reads the configuration and returns the final five reranked passages. Set `rerank_enabled` to `false` to return the same RRF top-five prefix without loading a model.

The completed single-round runner refuses to rerun its output directory. A separately authorized future experiment requires a new output directory:

```powershell
python -m eval.run_rerank_experiment --env-file ../EE542-Finalproject/.env --out eval/results/<new-experiment>
```

## Validation

The full offline suite passed after promoting the measured settings: 169 tests and 10 subtests, in 16.66 seconds. It checks promotion from RRF ranks 6-10, full passage inputs, stable ties, metadata preservation, invalid-score/model errors, the disabled path, pipeline counts and complete single-round accounting. This validation performs no additional live retrievals. See `tests.json`.
