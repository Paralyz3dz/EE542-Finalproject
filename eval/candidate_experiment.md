# Intake candidate-pool experiment

This experiment uses the 104 modular-intake questions and the complete 21-source,
6,004-chunk corpus. The older 200-question agent evaluation is not part of this run.

Only the common candidate count changes: HNSW and BM25 each retrieve 5, 10 or 20
candidates. RRF stays at 60, the caller receives at most five matches, and the
interview uses the first three. Corpus content, embeddings, query instruction,
source-name scoring and full-corpus search remain fixed.

## Run

From the retrieval worktree, use an environment that has the project dependencies:

```powershell
python -m eval.run_candidate_experiment --env-file ../EE542-Finalproject/.env --out eval/results/candidate-pool-rerun
```

The optional environment file is read only. No database writes or answer/judge API
calls are made. The runner creates a fingerprinted query-vector cache with Ollama
under ignored `data/processed/`. The cache includes the question fingerprint,
embedding model name, Ollama model digest, instruction and dimensions. It rejects
mismatched or invalid cached vectors.

Each condition calls the production `retrieve` function. Observation records the
actual HNSW/BM25 candidates and fused output. Query embedding is replaced with the
shared cached vector; database corpus fetching and production BM25 construction
are included in measured latency. Instrumentation and checkpoint output are not
part of the retrieval time. Three warmup queries per condition are excluded.

Three full repetitions run in these orders:

1. K=5, K=10, K=20
2. K=10, K=20, K=5
3. K=20, K=5, K=10

Before and after each condition, the runner verifies corpus hashes, vector hashes,
index definitions and HNSW settings. A changed corpus invalidates the run. Failed
runs leave an `incomplete.json` record. Invalidated runs require a new output
directory. A rerun after a transport failure can resume completed queries only
when the input manifest is identical; changed code or inputs require a new `--out`.
The query cache can be selected separately with `--query-cache`.

## Results and decision

The default output directory is `eval/results/candidate-pool-2026-11-17/`:

- `manifest.json` and `questions.json`: frozen inputs, model and code fingerprints.
- `results.jsonl`: every question in all nine runs, route candidates, output ranks,
  source metrics and retrieval latency.
- `passages.json`: full returned passages, keyed by chunk ID for evidence review.
- `changes.json`: gained/lost source hits paired with K=5 in each repetition.
- `corpus_after.json` and `summary.json`: final corpus check and condition metrics.

Source Hit@3 is primary. Source Hit@5, source MRR@5, candidate source coverage,
mean latency and P95 are secondary. Sources are matched using the existing engine
normalization. These metrics do not score passage relevance or answer correctness.
The historical 68/104 result remains unchanged; any fresh K=5 difference is
reported separately rather than replacing the historical baseline.

A candidate must improve average Hit@3 without reducing average Hit@5. Among
eligible settings, selection uses Hit@3, then Hit@5, then lower P95, then smaller K.
Otherwise K=5 is retained. At least 84/104 hits are required for the 80% target.
Results on these tuning questions alone do not demonstrate generalization or
statistical significance. Source filters, diversification, reranking, chunking and
embedding changes are separate future experiments.

`retrieve(query, candidate_k=20)` supports an explicit experiment override. Normal
calls read `retrieve_candidate_k` from settings, falling back to `retrieve_top_k`
when it is absent. The candidate count must be an integer at least as large as the
output count. The experiment reports a choice; applying it to settings happens
only after the complete validated run.

Use this runner for candidate-size comparisons. The earlier import checker
`src/rag/check_knowledge_base.py` hard-codes five candidates per route and remains
the historical K=5 reference; it does not evaluate a tuned candidate setting.

Completed first-round results and paired evidence review are recorded in
[the experiment report](results/candidate-pool-2026-11-17/report.md). The selected
runtime default is K=10; use a fresh output directory for a new experiment.
