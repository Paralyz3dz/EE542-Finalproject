# Passage relevance evaluation

## Result

One automatic round evaluated the saved default pure-reranker top five for all 104 interview questions: 520 passages. No retrieval was rerun and no ranking, embedding or database content changed. The judge was the existing API default `gpt-4o-mini`, returned as `gpt-4o-mini-2024-07-18`, at temperature zero.

| Metric | Automatic result |
|---|---:|
| Questions with a directly relevant passage in top 3 | 101/104 (97.12%) |
| Questions with a directly relevant passage in top 5 | 104/104 (100.00%) |
| Directly relevant passages / all returned passages (Precision@5) | 299/520 (57.50%) |
| Topical passages without a directly useful rule | 155/520 (29.81%) |
| Unrelated passages | 66/520 (12.69%) |
| MRR@5 for a directly relevant passage | 0.880609 |
| Mean graded relevance, out of two | 1.448077 |
| Questions with all five passages directly relevant | 3/104 (2.88%) |

The 100% result means this model assigned at least one grade two to every five-passage set. It does not establish complete evidence, correct applicability, tax-law accuracy or answer accuracy. Precision@5 shows that 42.5% of returned passages were not graded directly relevant. Only three questions had all five passages graded directly relevant, so the saturated hit@5 metric should be accompanied by passage precision and graded distributions.

These are model judgments, not human ground truth. The fixed 104-question set has no independent expert passage labels. The result is a first semantic baseline, not an improvement over the old 81.73% source-coverage score: the metrics measure different things. Recall and nDCG are not reported because all relevant corpus passages are not annotated.

## Rubric and blinded inputs

- Grade 0: unrelated to the specific retrieval need; keyword overlap alone does not count.
- Grade 1: the same tax area, but without a directly useful rule, definition, condition, exception, reporting requirement or explanation for this need.
- Grade 2: explicitly supplies at least one such directly useful element. It need not cover all conditions or every aspect of the question.

The judge receives the interview question, retrieval query, interview purpose and full passage headings/bodies. It does not receive expected source documents, citations, original ranks, retriever scores, chunk IDs or answers. Five candidates are deterministically shuffled for each question and identified only by opaque IDs. Original ranking is restored after grading. A passage can be directly relevant even if it is from a different official document.

Every question receives one judge call containing five separately graded passages. All 104 question IDs and API response IDs are unique. Both initial and final manifests, the exact rubric, response schema, input mappings, file hashes, raw model responses, token usage and quotes are preserved.

## Comparison with source coverage

| Question-level group | Count |
|---|---:|
| Source hit and at least one direct relevance grade | 85 |
| Source miss but at least one direct relevance grade | 19 |
| Source hit with no direct relevance grade | 0 |
| Source miss with no direct relevance grade | 0 |

The judge found directly relevant material for all nineteen source-miss cases. This supports investigating valid cross-document evidence, rather than assuming every source miss is a bad retrieval. It is not independent confirmation that those nineteen cases have sufficient correct evidence. There were also 25 individual passages graded unrelated despite matching the expected source document. Source matching alone cannot establish paragraph relevance.

`source_disagreements.json` lists the nineteen cases. `qualitative_review.json` contains assistant inspection notes for six cases without changing model scores. For example, a notice-response passage from Pub 505 discusses the Document Upload Tool, while a medical-expense passage contains a threshold within a narrow California adjustment. Both illustrate why relevance and complete applicability are distinct.

## Evidence-quote audit and validation recovery

The model was asked to supply a short body quote for positive grades. Quotes are audited independently:

| Positive-grade quote status | Count |
|---|---:|
| Verbatim, after whitespace normalization | 228 |
| Verbatim after case/Unicode punctuation normalization | 8 |
| Ordered verbatim fragments separated by ellipsis | 27 |
| Not mechanically verified | 191 |

These 454 positive grades include both grade one and grade two. The 191 unverified quotes affect 83 questions and require review; paraphrases, truncation and punctuation changes can fail this audit without making the relevance grade wrong. No unverified quote is described as verified evidence. A more restrictive diagnostic, requiring at least one grade-two passage with a mechanically verified quote, covers 92/104 questions (88.46%); 176 grade-two passages have such quotes. Quote matching itself still does not establish correct relevance or sufficient evidence.

The first three calls returned complete rubric scores, but strict quote formatting validation initially rejected two responses. The evaluator was updated to keep quote grounding as a separate audit flag. The rubric prompt, response schema, input data, model and all original grades remained unchanged. Those same responses were revalidated locally; there were zero repeated model calls for recovery. All final grades and quote strings were checked against their original API responses.

`manifest_before_quote_audit.json`, `evaluator_before_quote_audit.py`, `results_before_quote_audit.jsonl`, `initial_quote_validation_errors.json` and `validation_update.json` preserve this history. `quote_review_flags.json` retains every positive grade whose quote is unverified. This validation update is not a new scoring round.

## Execution and artifacts

There were 104 API responses for 104 unique questions, zero calls without an API response, and 180,445 total tokens (149,148 input, 31,297 output). Two calls ran concurrently with SDK retries disabled. There were zero live retrievals, query embeddings, answer generations, or database writes. Saved inputs and retrieval settings remained unchanged throughout the round.

`results.jsonl` contains ranked per-passage grades, reasons, original citations, quotes and audit flags. `judgments.jsonl` contains raw API responses and metadata. `inputs.json` contains the blinded payloads and private mapping back to original ranks. `summary.json`, `diagnostics.json` and `tests.json` contain aggregate checks and results.

The full offline suite passed with the quote-audit protocol: 206 tests and 10 subtests in 16.07 seconds. Earlier strict-quote validation passed 207 tests and 10 subtests before the audit adjustment. No API calls or live retrievals were performed by tests.

The evaluator refuses completed output directories. A separately authorized future evaluation requires a new output directory. This result does not change retrieval defaults or select a new ranking configuration.
