# Document processing and chunking repair

The materials pipeline rebuilds the configured tax-year 2025 sources from the tracked originals. The 23-source corpus includes Form 8843 instructions and Publication 519.

## Processing

- PDF text comes from original font spans and physical column positions. Column grids handle indented lists, full-width introductory text, and wrapped headings. Native table grids take precedence over layout-model guesses; remaining tables use their original PDF cell rectangles. Borderless numeric tables retain aligned cells, coordinates and header context.
- HTML instructions use the saved main content directly. Their citations name real web sections rather than page numbers invented by PDF conversion.
- Front contents, audited back indexes, publication catalogs, proof headers and isolated form labels are removed. Index boundaries are checked against the original bold heading; mixed pages keep the body and worksheets above that boundary.
- Every actual heading and bold run-in starts a separate section. Paragraphs combine only inside that section. Sentence and clause boundaries enforce the 400-word maximum. Short genuine sections remain short; padding them with unrelated topics would change their meaning. Wrapped words are rejoined, with common tax compounds preserved.
- Complete logical tables remain in exact storage. Searchable semantic table pieces retain whole rows and repeat column headers. Numeric locators have `retrieval_eligible=false`; the partial HNSW index and BM25 input enforce this before either candidate limit.
- Exact duplicate passages are consolidated with their original source names, citations, pages and URLs in `provenance`. Similar passages with different wording or conditions remain separate.
- The interview receives the complete retrieved chunk. The evidence panel displays additional citations for shared passages.

The parser uses PyMuPDF 2.1 [structured extraction](https://pymupdf.readthedocs.io/en/latest/document.html#Document.to_json) and native PDF geometry. No generative model rewrites official passages.

## Versioned remote storage

`corpus_version: rag_materials_v3` selects `public.rag_materials_v3_chunks` and `public.rag_materials_v3_tax_tables`. Select `rag_materials_v2` to restore the 21-source corpus. The original import remains available under `corpus_version: original`.

Loading validates all source names, IDs, content hashes, 1024-dimensional vectors, model names, table links and provenance before opening a write transaction. The target tables must not exist. Table creation, inserts and HNSW creation share one transaction. The original corpus inventory and vector fingerprints must remain unchanged. To restore the previous retrieval configuration, select `corpus_version: original` and `exclude_reference_chunks: true` together.

## Reproduction

Run in the retrieval worktree using the project virtual environment. The environment file is read from the existing checkout and is never copied or committed.

```powershell
python -m src.rag.materials --out data/processed/materials-v3
python -m eval.check_materials --input data/processed/materials-v3 --out data/processed/materials-v3-audit
python -m src.rag.embed_materials --env-file .env --input data/processed/materials-v3
python -m src.db.load_materials --env-file .env --input data/processed/materials-v3 --corpus rag_materials_v3
```

Embedding uses the baseline Qwen model digest, 8192 context, 12 threads and batches of 8. The snapshot identity and ordered partial vector file support resuming without combining different inputs. No existing remote corpus is overwritten by this loader.

The `rag_materials_v3` import reused the existing vectors for unchanged passages from `rag_materials_v2`. The new documents added 681 distinct passages and four exact tables. Thirty shared passages preserve Publication 519 as an additional citation.

The recorded material-processing evaluation used the validated 104 question vectors, candidate K=10, output=5, display=3, and RRF=60. The current interview uses all five returned passages, so operational source coverage uses the saved hit@5 metric. It runs one scored round with no warm-up retrievals. A completed output directory is refused. Saved baseline results supply comparison values, so the old corpus is not queried for another scored round. Strict primary-source hit metrics remain unchanged; coverage through preserved duplicate provenance is reported separately. These metrics do not establish semantic relevance or sufficient evidence, and no answer generator or model judge is invoked.

Results and processing checks are recorded in `eval/results/materials-v2-2026-11-17/`.
