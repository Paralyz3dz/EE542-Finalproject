Run the current Agentic RAG baseline against the evaluation dataset:

```powershell
.\.venv\Scripts\python.exe -m eval.run_baseline --workers 2
```

The runner uses the configured OpenAI model for answers and automatic judging.
`EVAL_MODEL` can select a different judge before starting a new run. The judge
temperature is zero; the agent keeps its existing configuration. Only the
question is passed to the answering agent. Every case starts a fresh conversation.
The database is accessed through the existing read-only retrieval path.

Outputs are saved in `eval/results/baseline/`:

| File | Contents |
|---|---|
| `manifest.json` | Dataset and source hashes, models, prompts, concurrency, and metric definitions |
| `responses.jsonl` | Checkpointed answers, conversation messages, actual retrieval calls, and latency |
| `judgments.jsonl` | Automatic correctness verdicts and statement-level faithfulness judgments |
| `results.jsonl` | Per-question answers, references, retrieval metrics, scores, and diagnostics |
| `summary.json` | Macro metrics, scoring coverage, latency statistics, and breakdowns by type and year |
| `errors.jsonl` | Judge errors, if any |

Metrics:

- Recall@5 and MRR@5 use annotated chunk IDs, separately for HNSW, BM25, and
  RRF. The primary aggregate uses each question's first actual retrieval call.
  No retrieval call scores zero on an answerable question. Per-call metrics and
  cumulative recall are retained for diagnosing repeated retrievals; cumulative
  recall is not called Recall@5 because it can cover more than five chunks.
- Answer correctness uses an automatic reference-based rubric: correct = 1,
  partially correct = 0.5, incorrect = 0. Standard answers and their evidence
  are available only to this judge, never to the answering agent.
- Faithfulness is the proportion of assessable response claims supported by
  the passages actually returned to the agent, across all tool calls. This is
  a custom LLM judge implementation of statement support, not the RAGAS library.
  Answers with no assessable factual claims receive null, not an automatic 1;
  the summary reports the number of scored answers.
- End-to-end latency measures the agent invocation, including tool calls and
  generation, and excludes judging. Mean, median, and p95 are reported under
  the recorded concurrency.

LLM judging is approximate and should be checked against human samples. Chunk
relevance labels may be incomplete, so ID-based retrieval scores can miss other
valid evidence. Publication year and question year are retained separately:
the dataset's 2026 questions are not used to change the configured 2025 corpus.
The supplied passages can explicitly discuss a later year's rules.

Rerun the same command to resume. Completed agent responses and judgments are
reused. A changed dataset, pipeline, prompts, model, or concurrency requires a
different output directory with `--out`. Use `--limit 2` for an initial check
without changing the complete dataset fingerprint.
