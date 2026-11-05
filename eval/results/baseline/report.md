# Agentic RAG baseline

Completed 200/200 cases with 0 execution errors. Both the answering model and the judge were `gpt-4o-mini`, with concurrency set to 2.

## Retrieval metrics

Metrics use each question's first actual retrieval tool call, with K=5. Relevance is determined by the chunk IDs annotated in the evaluation dataset.

| Retrieval method | Recall@5 | MRR@5 |
|---|---:|---:|
| HNSW | 85.50% | 0.7254 |
| BM25 | 84.75% | 0.6790 |
| RRF | 87.00% | 0.7506 |

## Answer quality and latency

| Metric | Result |
|---|---:|
| Mean automatic answer correctness score (0/0.5/1) | 0.9500 |
| Mean automatic faithfulness score | 0.9848 |
| Mean end-to-end latency | 7.36 seconds |
| Median latency | 6.82 seconds |
| P95 latency | 9.90 seconds |

Correctness and faithfulness were scored for all 200 cases. Latency includes actual retrieval and answer generation, excluding judge calls. Measurements use concurrency 2 and do not represent single-request latency under idle conditions.

Automatic correctness verdicts were correct=188, partial=4, and incorrect=8. These counts have not been verified by human reviewers.

The retrieval tool was called 236 times: 167 cases used 1 call, 30 used 2 calls, and 3 used 3 calls. Each case started an independent conversation. Reference answers and annotated evidence were not passed to the answering model.

## Interpretation and review requirements

- Fusion Recall@5 exceeded HNSW by 1.5 percentage points and BM25 by 2.25 percentage points in this run. These observations do not establish statistical significance.
- Of the 200 questions, 193 have only one annotated relevant chunk. Other valid evidence may be missing from the labels, so ID-based metrics depend on annotation completeness.
- Correctness and faithfulness use custom LLM judges, not the RAGAS library or human grading. Clear judging errors were found: answers 042, 115, and 159 match their references but were marked incorrect.
- Faithfulness is also overestimated in some cases: 064 reverses a boundary condition, 180 deviates from the requested rules, and 189 gives an amount that conflicts with the reference, yet all received a score of 1. Generation scores are provisional and require calibration; they do not establish 95% accuracy or 98.48% actual faithfulness.
- Original automatic scores were preserved. `review_flags.json` lists 10 cases requiring review of judgments or evidence for subsequent judge calibration.
- There are 190 questions about tax year 2025 and 10 about 2026. The latter can be supported by passages in the 2025 publication that explicitly discuss later years. Group results are in `summary.json`.
- The retrieved source corpus hash remained consistent throughout the run. No database writes were performed.

## Files

- `metrics.csv`: Per-question metrics for spreadsheet inspection.
- `results.jsonl`: Per-question answers, references, actual retrieval results, metrics, and judging reasons.
- `summary.json`: Overall metrics and breakdowns by question type and applicable tax year.
- `review_flags.json`: Cases requiring review of automatic judgments.
- `manifest.json`: Dataset and code fingerprints, models, judging rules, and configuration.
