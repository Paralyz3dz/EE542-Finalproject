import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np
from dotenv import load_dotenv

from eval.run_candidate_experiment import compact, corpus_state, digest, source_metrics, write_json
from src.db.connection import ROOT, get_conn
from src.db.queries import corpus_table
from src.rag import retriever
from src.workflow.engine import _doc_key, load_scenario


def snapshot(corpus):
    table = corpus_table(corpus)
    with get_conn() as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        rows = conn.execute(f"SELECT id, source, content_hash, embedding_model, md5(text), md5(embed_text), md5(citation), md5(embedding::text), type, printed_pages, pdf_pages, provenance, retrieval_eligible FROM {table} ORDER BY id").fetchall()
        indexes = conn.execute("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname='public' AND tablename=%s ORDER BY indexname", [table.split(".")[1]]).fetchall()
        search = conn.execute("SHOW hnsw.ef_search").fetchone()[0]
    if not rows or len({r[1] for r in rows}) != 21 or any(r[7] is None for r in rows):
        raise ValueError("Incomplete versioned corpus")
    return {"version": corpus, "chunks": len(rows), "sources": 21, "sha256": digest(rows), "indexes": indexes, "hnsw_ef_search": search}


def observe(question, vector, corpus):
    trace = {}
    dense_original, sparse_original = retriever.vector_search, retriever.bm25_search

    def dense(*args, **kwargs):
        if kwargs.get("corpus") != corpus or kwargs.get("k") != 10 or kwargs.get("excluded_ids"):
            raise ValueError("Unexpected retrieval route settings")
        rows = dense_original(*args, **kwargs)
        trace["vector"] = [{"chunk_id": r[0], "citation": r[1], "score": float(r[3])} for r in rows]
        return rows

    def sparse(query, chunks, **kwargs):
        trace["eligible_chunks"] = len(chunks)
        rows = sparse_original(query, chunks, **kwargs)
        trace["bm25"] = compact(rows)
        return rows

    with patch.object(retriever, "embed_query", return_value=vector), patch.object(retriever, "vector_search", side_effect=dense), patch.object(retriever, "bm25_search", side_effect=sparse):
        start = time.perf_counter()
        matches = retriever.retrieve(question["retrieval_query"], candidate_k=10, exclude_reference_chunks=False, corpus_version=corpus)["matches"]
        elapsed = time.perf_counter() - start
    if len(matches) != 5 or len(trace["vector"]) != 10:
        raise ValueError("Unexpected candidate/output count")
    expected = {_doc_key(s) for s in question["source"].split(";")}
    alias_ranks = [i for i, m in enumerate(matches, 1) if expected.intersection({_doc_key(m["citation"])} | {_doc_key(c) for c in m.get("additional_citations", [])})]
    return {"id": question["id"], "query": question["retrieval_query"], "source": question["source"], "round": 1,
            "candidate_k": 10, "elapsed_seconds": elapsed, **source_metrics(question, matches),
            "provenance_hit_at_3": bool(alias_ranks and alias_ranks[0] <= 3),
            "provenance_hit_at_5": bool(alias_ranks), "routes": trace, "matches": compact(matches)}, matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--corpus", default="rag_materials_v2")
    parser.add_argument("--out", type=Path, default=ROOT / "eval/results/materials-v2-2026-11-17")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    if (args.out / "summary.json").exists():
        raise ValueError("This single round is already complete; it must not be rerun")
    prior = json.loads((ROOT / "eval/results/candidate-pool-2026-11-17/manifest.json").read_text(encoding="utf-8"))
    _, questions = load_scenario("intake")
    cache = json.loads((ROOT / "data/processed/candidate_queries.json").read_text(encoding="utf-8"))
    vectors = np.asarray(cache["vectors"], dtype=np.float32)
    identity = {"questions": digest(questions), "model": prior["model"], "instruction": retriever.QUERY_INSTRUCTION, "dimensions": 1024}
    if len(questions) != 104 or cache["identity"] != identity or vectors.shape != (104, 1024) or hashlib.sha256(vectors.tobytes()).hexdigest() != prior["query_vectors_sha256"]:
        raise ValueError("Questions or cached query vectors changed")
    if os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B") != prior["model"]["embedding_model"]:
        raise ValueError("Embedding model changed")
    with urlopen(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/tags", timeout=15) as response:
        models = json.load(response)["models"]
    live = next(m for m in models if m["name"] == prior["model"]["ollama_model"])
    if live["digest"] != prior["model"]["digest"]:
        raise ValueError("Ollama weights changed")
    baseline = corpus_state()
    if digest(baseline) != digest(prior["corpus"]):
        raise ValueError("Original public corpus changed")
    settings = json.loads((ROOT / "config/settings.json").read_text(encoding="utf-8"))
    if (settings["tax_year"], settings["retrieve_top_k"], settings["retrieve_candidate_k"], settings["rrf_constant"]) != (2025, 5, 10, 60):
        raise ValueError("Retrieval tuning changed")
    files = ("src/rag/retriever.py", "src/rag/materials.py", "src/rag/hybrid.py", "src/db/queries.py", "src/db/load_materials.py", "eval/run_materials_experiment.py", "config/settings.json")
    code = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}
    corpus = snapshot(args.corpus)
    manifest = {"corpus": corpus, "baseline": baseline, "questions_sha256": digest(questions), "settings": settings,
                "query_vectors_sha256": prior["query_vectors_sha256"], "model": prior["model"], "code_sha256": code,
                "rounds": 1, "queries": 104, "warmup_queries": 0,
                "effective_settings": {**settings, "corpus_version": args.corpus, "exclude_reference_chunks": False, "retrieve_candidate_k": 10},
                "timing": "Production DB/vector/BM25/RRF pipeline; cached query vectors; no answer or semantic judge",
                "metric": "Same strict primary-document source hit as the saved baseline; additional provenance coverage reported separately"}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists() and digest(json.loads(manifest_path.read_text(encoding="utf-8"))) != digest(manifest):
        raise ValueError("Partial round inputs changed")
    write_json(manifest_path, manifest)
    write_json(args.out / "questions.json", questions)
    result_path = args.out / "results.jsonl"
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()] if result_path.exists() else []
    done = {row["id"] for row in rows}
    if len(done) != len(rows) or not done.issubset({q["id"] for q in questions}):
        raise ValueError("Invalid partial round")
    passage_path = args.out / "passages.json"
    passages = json.loads(passage_path.read_text(encoding="utf-8")) if passage_path.exists() else {}
    with result_path.open("a", encoding="utf-8", newline="\n") as output:
        for i, (question, vector) in enumerate(zip(questions, vectors), 1):
            if question["id"] in done:
                continue
            row, matches = observe(question, vector, args.corpus)
            for match in matches:
                passages[match["chunk_id"]] = {k: match[k] for k in ("citation", "text", "additional_citations") if k in match}
            write_json(passage_path, passages)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            rows.append(row)
            if i % 10 == 0 or i == 104:
                print(f"Single round: {i}/104", flush=True)
    if len(rows) != 104 or digest(snapshot(args.corpus)) != digest(corpus) or corpus_state() != baseline or code != {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}:
        raise ValueError("Incomplete round or changed inputs")
    previous_path = ROOT / "eval/results/corpus-hygiene-2026-11-17/results.jsonl"
    previous = {r["id"]: r for r in [json.loads(line) for line in previous_path.read_text(encoding="utf-8").splitlines()] if r["round"] == 1 and r["cleanup"]}
    changes = [{"id": row["id"], "before": previous[row["id"]], "after": row} for row in rows
               if any(row[key] != previous[row["id"]][key] for key in ("hit_at_3", "hit_at_5", "mrr_at_5"))]
    summary = {"rounds": 1, "queries": 104, "strict_primary_source": {
        "hits_at_3": sum(r["hit_at_3"] for r in rows), "hit_at_3": float(np.mean([r["hit_at_3"] for r in rows])),
        "hits_at_5": sum(r["hit_at_5"] for r in rows), "hit_at_5": float(np.mean([r["hit_at_5"] for r in rows])),
        "mrr_at_5": float(np.mean([r["mrr_at_5"] for r in rows]))},
        "with_preserved_provenance": {"hits_at_3": sum(r["provenance_hit_at_3"] for r in rows), "hits_at_5": sum(r["provenance_hit_at_5"] for r in rows)},
        "saved_cleaned_baseline": {"hits_at_3": sum(r["hit_at_3"] for r in previous.values()), "hits_at_5": sum(r["hit_at_5"] for r in previous.values()), "mrr_at_5": float(np.mean([r["mrr_at_5"] for r in previous.values()]))},
        "mean_seconds": float(np.mean([r["elapsed_seconds"] for r in rows])), "p95_seconds": float(np.percentile([r["elapsed_seconds"] for r in rows], 95)),
        "limitations": "Source provenance metrics do not establish passage relevance or sufficient evidence. No model judge or generated answers."}
    write_json(args.out / "changes.json", changes)
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "corpus_after.json", corpus)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
