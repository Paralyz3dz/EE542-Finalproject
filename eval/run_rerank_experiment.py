import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np
from dotenv import load_dotenv

from eval.run_candidate_experiment import digest, source_metrics, write_json
from eval.run_materials_experiment import snapshot
from src.db.connection import ROOT
from src.rag import retriever
from src.rag.reranker import prepare, options
from src.workflow.engine import _doc_key, load_scenario


KEYS = ("chunk_id", "citation", "score", "vector_score", "bm25_score", "vector_rank", "bm25_rank", "rrf_score", "rrf_rank", "rerank_score", "rerank_rank", "additional_citations")


def compact(matches):
    return [{key: match[key] for key in KEYS if key in match} for match in matches]


def source_found(question, matches):
    expected = {_doc_key(source) for source in question["source"].split(";") if source.strip()}
    return any(_doc_key(match["citation"]) in expected for match in matches)


def observe(question, vector):
    trace, passages = {}, {}
    dense_original, sparse_original = retriever.vector_search, retriever.bm25_search
    fuse_original, rerank_original = retriever.fuse_rankings, retriever.rerank_matches
    baseline = []
    start = time.perf_counter()

    def dense(*args, **kwargs):
        if kwargs.get("k") != 10 or kwargs.get("corpus") != "rag_materials_v2" or kwargs.get("excluded_ids"):
            raise ValueError("Unexpected dense route settings")
        rows = dense_original(*args, **kwargs)
        trace["vector"] = [{"chunk_id": row[0], "citation": row[1], "score": float(row[3])} for row in rows]
        return rows

    def sparse(query, chunks, **kwargs):
        if kwargs.get("k") != 10:
            raise ValueError("Unexpected sparse route settings")
        trace["eligible_chunks"] = len(chunks)
        for chunk in chunks:
            passages[chunk["id"]] = chunk
        result = sparse_original(query, chunks, **kwargs)
        trace["bm25"] = compact(result)
        return result

    def fuse(*args, **kwargs):
        if kwargs.get("k") != 10 or kwargs.get("rrf_constant") != 60:
            raise ValueError("Unexpected fusion settings")
        result = fuse_original(*args, **kwargs)
        if len(result) != 10:
            raise ValueError("RRF must return ten unique candidates")
        trace["rrf"] = compact(result)
        baseline.extend(dict(match) for match in result[:5])
        trace["retrieval_seconds"] = time.perf_counter() - start
        return result

    def rerank(query, matches, texts, **kwargs):
        if len(matches) != 10 or kwargs.get("k") != 5 or [m["chunk_id"] for m in matches] != [m["chunk_id"] for m in trace["rrf"]]:
            raise ValueError("Reranker did not receive the full RRF candidate set")
        if any(texts[m["chunk_id"]] != passages[m["chunk_id"]]["embed_text"] for m in matches):
            raise ValueError("Reranker input differs from the stored passage")
        tick = time.perf_counter()
        result = rerank_original(query, matches, texts, **kwargs)
        trace["rerank_seconds"] = time.perf_counter() - tick
        if len(result) != 5 or not {m["chunk_id"] for m in result}.issubset({m["chunk_id"] for m in matches}):
            raise ValueError("Invalid reranked output set")
        return result

    with patch.object(retriever, "embed_query", return_value=vector), patch.object(retriever, "vector_search", side_effect=dense), patch.object(retriever, "bm25_search", side_effect=sparse), patch.object(retriever, "fuse_rankings", side_effect=fuse), patch.object(retriever, "rerank_matches", side_effect=rerank):
        output = retriever.retrieve(question["retrieval_query"], candidate_k=10, exclude_reference_chunks=False, corpus_version="rag_materials_v2", rerank_enabled=True)
    elapsed = time.perf_counter() - start
    if output["retrieval_method"] != "hnsw_bm25_rrf_rerank":
        raise ValueError("The production reranker was not executed")
    matches = output["matches"]
    selected = {m["chunk_id"] for m in trace["rrf"]}
    kept = {cid: {key: passages[cid][key] for key in ("citation", "text", "embed_text", "provenance") if key in passages[cid]} for cid in selected}
    return {"id": question["id"], "query": question["retrieval_query"], "source": question["source"], "round": 1,
            "candidate_k": 10, "fusion_k": 10, "output_k": 5, "elapsed_seconds": elapsed,
            "retrieval_seconds": trace["retrieval_seconds"], "rerank_seconds": trace["rerank_seconds"],
            "baseline": source_metrics(question, baseline), "reranked": source_metrics(question, matches),
            "rrf_source_coverage_at_10": source_found(question, trace["rrf"]),
            "baseline_matches": compact(baseline), "matches": compact(matches), "routes": trace}, kept


def summarize(rows):
    if len(rows) != 104 or len({row["id"] for row in rows}) != 104 or {row["round"] for row in rows} != {1}:
        raise ValueError("Expected one complete 104-question round")
    def metrics(key):
        return {"hits_at_3": sum(r[key]["hit_at_3"] for r in rows), "hit_at_3": float(np.mean([r[key]["hit_at_3"] for r in rows])),
                "hits_at_5": sum(r[key]["hit_at_5"] for r in rows), "hit_at_5": float(np.mean([r[key]["hit_at_5"] for r in rows])),
                "mrr_at_5": float(np.mean([r[key]["mrr_at_5"] for r in rows]))}
    baseline, reranked = metrics("baseline"), metrics("reranked")
    return {"rounds": 1, "queries": 104, "baseline_rrf_top5": baseline, "reranked_top5": reranked,
            "rrf_source_coverage_at_10": sum(r["rrf_source_coverage_at_10"] for r in rows),
            "gained_hit_at_5": [r["id"] for r in rows if not r["baseline"]["hit_at_5"] and r["reranked"]["hit_at_5"]],
            "lost_hit_at_5": [r["id"] for r in rows if r["baseline"]["hit_at_5"] and not r["reranked"]["hit_at_5"]],
            "mean_retrieval_seconds": float(np.mean([r["retrieval_seconds"] for r in rows])),
            "mean_rerank_seconds": float(np.mean([r["rerank_seconds"] for r in rows])),
            "mean_total_seconds": float(np.mean([r["elapsed_seconds"] for r in rows])),
            "p95_total_seconds": float(np.percentile([r["elapsed_seconds"] for r in rows], 95)),
            "primary_improved": reranked["hits_at_5"] > baseline["hits_at_5"],
            "limitations": "Primary-document source metrics are proxies; no answer generator or semantic sufficiency judge was run."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "eval/results/rerank-2026-11-17")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    if (args.out / "summary.json").exists():
        raise ValueError("This single round is complete and cannot be rerun")
    baseline_dir = ROOT / "eval/results/materials-v2-2026-11-17"
    prior = json.loads((baseline_dir / "manifest.json").read_text(encoding="utf-8"))
    _, questions = load_scenario("intake")
    cache = json.loads((ROOT / "data/processed/candidate_queries.json").read_text(encoding="utf-8"))
    vectors = np.asarray(cache["vectors"], dtype=np.float32)
    if len(questions) != 104 or digest(questions) != prior["questions_sha256"] or cache["identity"] != {"questions": digest(questions), "model": prior["model"], "instruction": retriever.QUERY_INSTRUCTION, "dimensions": 1024}:
        raise ValueError("Questions or query-vector identity changed")
    if vectors.shape != (104, 1024) or hashlib.sha256(vectors.tobytes()).hexdigest() != prior["query_vectors_sha256"]:
        raise ValueError("Cached query vectors changed")
    if os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B") != prior["model"]["embedding_model"]:
        raise ValueError("Embedding model changed")
    with urlopen(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/tags", timeout=15) as response:
        live = next(m for m in json.load(response)["models"] if m["name"] == prior["model"]["ollama_model"])
    if live["digest"] != prior["model"]["digest"]:
        raise ValueError("Embedding weights changed")
    settings = json.loads((ROOT / "config/settings.json").read_text(encoding="utf-8"))
    if tuple(settings[key] for key in ("tax_year", "retrieve_candidate_k", "retrieve_fusion_k", "retrieve_top_k", "rerank_top_k", "rrf_constant")) != (2025, 10, 10, 5, 5, 60):
        raise ValueError("Keep the requested 10 + 10 -> RRF 10 -> rerank 5 configuration")
    corpus = snapshot("rag_materials_v2")
    if digest(corpus) != digest(prior["corpus"]):
        raise ValueError("The document/vector corpus changed")
    tick = time.perf_counter()
    ranker = prepare(settings)
    load_seconds = time.perf_counter() - tick
    files = ("src/rag/retriever.py", "src/rag/hybrid.py", "src/rag/reranker.py", "src/db/queries.py", "eval/run_rerank_experiment.py", "config/settings.json")
    code = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}
    manifest = {"corpus": corpus, "questions_sha256": digest(questions), "query_vectors_sha256": prior["query_vectors_sha256"],
                "embedding_model": prior["model"], "reranker": {**ranker.identity, **options(settings), "device": "cpu", "dtype": "float32", "truncation": False},
                "settings": settings, "effective_settings": {**settings, "rerank_enabled": True}, "code_sha256": code,
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "huggingface-hub", "numpy")},
                "rounds": 1, "queries": 104, "warmup_retrievals": 0, "corpus_writes": 0,
                "baseline": "The first five entries of the same RRF top ten; compared with the saved material baseline",
                "reranker_input": "Original retrieval query and complete chunk.embed_text; no source labels or gold references",
                "selection_policy": "Enable the default only if primary-source hit@5 strictly improves over the paired RRF baseline"}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "manifest.json"
    if path.exists() and digest(json.loads(path.read_text(encoding="utf-8"))) != digest(manifest):
        raise ValueError("Partial round fingerprint changed")
    write_json(path, manifest)
    write_json(args.out / "questions.json", questions)
    if not (args.out / "startup.json").exists():
        write_json(args.out / "startup.json", {"model_load_seconds": load_seconds, "timing": "Excluded from steady-state per-question retrieval timing", "inference_warmups": 0})
    path = args.out / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    done = {r["id"] for r in rows}
    if len(done) != len(rows) or not done.issubset({q["id"] for q in questions}):
        raise ValueError("Invalid partial round")
    passage_path = args.out / "passages.json"
    passages = json.loads(passage_path.read_text(encoding="utf-8")) if passage_path.exists() else {}
    with path.open("a", encoding="utf-8", newline="\n") as output:
        for i, (question, vector) in enumerate(zip(questions, vectors), 1):
            if question["id"] in done:
                continue
            row, candidate_passages = observe(question, vector)
            passages.update(candidate_passages)
            write_json(passage_path, passages)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            rows.append(row)
            if i % 5 == 0 or i == 104:
                print(f"Rerank round: {i}/104; last rerank {row['rerank_seconds']:.2f}s", flush=True)
    if digest(snapshot("rag_materials_v2")) != digest(corpus) or code != {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}:
        raise ValueError("Corpus or code changed during the round")
    previous = {r["id"]: r for r in [json.loads(line) for line in (baseline_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]}
    summary = summarize(rows)
    summary["saved_baseline_top5_differences"] = [r["id"] for r in rows if [m["chunk_id"] for m in r["baseline_matches"]] != [m["chunk_id"] for m in previous[r["id"]]["matches"]]]
    summary["saved_baseline_metric_differences"] = [r["id"] for r in rows if any(r["baseline"][key] != previous[r["id"]][key] for key in ("hit_at_3", "hit_at_5", "mrr_at_5"))]
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "changes.json", [r for r in rows if any(r["baseline"][key] != r["reranked"][key] for key in ("hit_at_3", "hit_at_5", "mrr_at_5"))])
    write_json(args.out / "corpus_after.json", corpus)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
