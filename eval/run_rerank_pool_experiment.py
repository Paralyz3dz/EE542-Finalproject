import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np
from dotenv import load_dotenv

from eval.run_candidate_experiment import digest, source_metrics, write_json
from eval.run_materials_experiment import snapshot
from eval.run_rerank_experiment import compact, source_found
from src.db.connection import ROOT
from src.rag import retriever
from src.rag.reranker import prepare, options
from src.workflow.engine import load_scenario


def validate_pool(vector, sparse, pool):
    expected = min(15, len({m["chunk_id"] for m in vector + sparse}))
    if len(pool) != expected or len({m["chunk_id"] for m in pool}) != expected:
        raise ValueError("RRF must retain up to fifteen unique candidates from the unchanged routes")


def observe(question, vector, previous):
    trace, passages = {}, {}
    dense_original, sparse_original = retriever.vector_search, retriever.bm25_search
    fuse_original, rerank_original = retriever.fuse_rankings, retriever.rerank_matches
    start = time.perf_counter()

    def dense(*args, **kwargs):
        if kwargs.get("k") != 10 or kwargs.get("corpus") != "rag_materials_v2" or kwargs.get("excluded_ids"):
            raise ValueError("Unexpected dense route settings")
        rows = dense_original(*args, **kwargs)
        trace["vector"] = [{"chunk_id": r[0], "citation": r[1], "score": float(r[3])} for r in rows]
        return rows

    def sparse(query, chunks, **kwargs):
        if kwargs.get("k") != 10:
            raise ValueError("Unexpected sparse route settings")
        trace["eligible_chunks"] = len(chunks)
        passages.update({chunk["id"]: chunk for chunk in chunks})
        result = sparse_original(query, chunks, **kwargs)
        trace["bm25"] = compact(result)
        return result

    def fuse(*args, **kwargs):
        if kwargs.get("k") != 15 or kwargs.get("rrf_constant") != 60:
            raise ValueError("Only the RRF pool size may change")
        result = fuse_original(*args, **kwargs)
        validate_pool(trace["vector"], trace["bm25"], result)
        for route in ("vector", "bm25"):
            if trace[route] != previous["routes"][route]:
                raise ValueError("Candidate route differs from the saved baseline: " + route)
        if compact(result[:10]) != previous["routes"]["rrf"]:
            raise ValueError("RRF top-ten prefix differs from the saved baseline")
        trace["rrf"] = compact(result)
        trace["retrieval_seconds"] = time.perf_counter() - start
        return result

    def rerank(query, matches, texts, **kwargs):
        if kwargs.get("k") != 5 or compact(matches) != trace["rrf"]:
            raise ValueError("Reranker did not receive the complete RRF pool")
        if any(texts[m["chunk_id"]] != passages[m["chunk_id"]]["embed_text"] for m in matches):
            raise ValueError("Reranker input differs from the complete stored passage")
        tick = time.perf_counter()
        result = rerank_original(query, matches, texts, **kwargs)
        trace["rerank_seconds"] = time.perf_counter() - tick
        if len(result) != 5 or len({m["chunk_id"] for m in result}) != 5 or not {m["chunk_id"] for m in result}.issubset({m["chunk_id"] for m in matches}):
            raise ValueError("Invalid reranked output set")
        return result

    with patch.object(retriever, "embed_query", return_value=vector), patch.object(retriever, "vector_search", side_effect=dense), patch.object(retriever, "bm25_search", side_effect=sparse), patch.object(retriever, "fuse_rankings", side_effect=fuse), patch.object(retriever, "rerank_matches", side_effect=rerank):
        output = retriever.retrieve(question["retrieval_query"], candidate_k=10, exclude_reference_chunks=False, corpus_version="rag_materials_v2", rerank_enabled=True)
    elapsed = time.perf_counter() - start
    if output["retrieval_method"] != "hnsw_bm25_rrf_rerank":
        raise ValueError("The production reranker was not executed")
    matches = output["matches"]
    kept = {m["chunk_id"]: {key: passages[m["chunk_id"]][key] for key in ("citation", "text", "embed_text", "provenance") if key in passages[m["chunk_id"]]} for m in trace["rrf"]}
    return {"id": question["id"], "query": question["retrieval_query"], "source": question["source"], "round": 1,
            "candidate_k": 10, "fusion_k": 15, "actual_fusion_count": len(trace["rrf"]), "output_k": 5,
            "elapsed_seconds": elapsed, "retrieval_seconds": trace["retrieval_seconds"], "rerank_seconds": trace["rerank_seconds"],
            "baseline": previous["reranked"], "reranked": source_metrics(question, matches),
            "baseline_matches": previous["matches"], "matches": compact(matches), "routes": trace,
            "rrf_source_coverage_at_10": source_found(question, trace["rrf"][:10]),
            "rrf_source_coverage_at_15": source_found(question, trace["rrf"])}, kept


def summarize(rows, baseline_summary):
    if len(rows) != 104 or len({r["id"] for r in rows}) != 104 or {r["round"] for r in rows} != {1}:
        raise ValueError("Expected one complete 104-question round")
    def metrics(key):
        return {"hits_at_3": sum(r[key]["hit_at_3"] for r in rows), "hit_at_3": float(np.mean([r[key]["hit_at_3"] for r in rows])),
                "hits_at_5": sum(r[key]["hit_at_5"] for r in rows), "hit_at_5": float(np.mean([r[key]["hit_at_5"] for r in rows])),
                "mrr_at_5": float(np.mean([r[key]["mrr_at_5"] for r in rows]))}
    baseline, reranked = metrics("baseline"), metrics("reranked")
    if baseline != baseline_summary["reranked_top5"]:
        raise ValueError("Baseline metrics differ from the recorded RRF-ten experiment")
    return {"rounds": 1, "queries": 104, "baseline_rrf10_reranked_top5": baseline, "rrf15_reranked_top5": reranked,
            "rrf_source_coverage_at_10": sum(r["rrf_source_coverage_at_10"] for r in rows),
            "rrf_source_coverage_at_15": sum(r["rrf_source_coverage_at_15"] for r in rows),
            "actual_fusion_count_histogram": dict(sorted(Counter(r["actual_fusion_count"] for r in rows).items())),
            "gained_hit_at_5": [r["id"] for r in rows if not r["baseline"]["hit_at_5"] and r["reranked"]["hit_at_5"]],
            "lost_hit_at_5": [r["id"] for r in rows if r["baseline"]["hit_at_5"] and not r["reranked"]["hit_at_5"]],
            "baseline_mean_total_seconds": baseline_summary["mean_total_seconds"],
            "mean_retrieval_seconds": float(np.mean([r["retrieval_seconds"] for r in rows])),
            "mean_rerank_seconds": float(np.mean([r["rerank_seconds"] for r in rows])),
            "mean_total_seconds": float(np.mean([r["elapsed_seconds"] for r in rows])),
            "p95_total_seconds": float(np.percentile([r["elapsed_seconds"] for r in rows], 95)),
            "primary_improved": reranked["hits_at_5"] > baseline["hits_at_5"],
            "baseline_candidate_routes_and_rrf10_prefix_reproduced": True,
            "limitations": "Primary-document source coverage only; saved baseline timing; no semantic judge or generated answers"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "eval/results/rerank-pool15-2026-11-17")
    args = parser.parse_args()
    if (args.out / "summary.json").exists():
        raise ValueError("This single round is complete and cannot be rerun")
    load_dotenv(args.env_file)
    baseline_dir = ROOT / "eval/results/rerank-2026-11-17"
    prior = json.loads((baseline_dir / "manifest.json").read_text(encoding="utf-8"))
    baseline_summary = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    baseline_rows = [json.loads(line) for line in (baseline_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    previous = {r["id"]: r for r in baseline_rows}
    _, questions = load_scenario("intake")
    cache = json.loads((ROOT / "data/processed/candidate_queries.json").read_text(encoding="utf-8"))
    vectors = np.asarray(cache["vectors"], dtype=np.float32)
    expected_identity = {"questions": digest(questions), "model": prior["embedding_model"], "instruction": retriever.QUERY_INSTRUCTION, "dimensions": 1024}
    if len(questions) != 104 or len(baseline_rows) != len(previous) or set(previous) != {q["id"] for q in questions} or digest(questions) != prior["questions_sha256"] or cache["identity"] != expected_identity:
        raise ValueError("Questions or query-vector identity changed")
    if vectors.shape != (104, 1024) or hashlib.sha256(vectors.tobytes()).hexdigest() != prior["query_vectors_sha256"]:
        raise ValueError("Cached query vectors changed")
    if os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B") != prior["embedding_model"]["embedding_model"]:
        raise ValueError("Embedding model changed")
    with urlopen(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/tags", timeout=15) as response:
        live = next(m for m in json.load(response)["models"] if m["name"] == prior["embedding_model"]["ollama_model"])
    if live["digest"] != prior["embedding_model"]["digest"]:
        raise ValueError("Embedding weights changed")
    settings = json.loads((ROOT / "config/settings.json").read_text(encoding="utf-8"))
    if settings != {**prior["effective_settings"], "retrieve_fusion_k": 15}:
        raise ValueError("Only retrieve_fusion_k may change from ten to fifteen")
    corpus = snapshot("rag_materials_v2")
    if digest(corpus) != digest(prior["corpus"]):
        raise ValueError("The document/vector corpus changed")
    for name, expected in prior["code_sha256"].items():
        if name != "config/settings.json" and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Baseline implementation changed: " + name)
    tick = time.perf_counter()
    ranker = prepare(settings)
    load_seconds = time.perf_counter() - tick
    identity = {**ranker.identity, **options(settings), "device": "cpu", "dtype": "float32", "truncation": False}
    versions = {name: importlib.metadata.version(name) for name in prior["versions"]}
    if identity != prior["reranker"] or versions != prior["versions"]:
        raise ValueError("Reranker identity or dependency versions changed")
    files = (*prior["code_sha256"], "eval/run_rerank_pool_experiment.py")
    code = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}
    manifest = {"corpus": corpus, "questions_sha256": digest(questions), "query_vectors_sha256": prior["query_vectors_sha256"],
                "embedding_model": prior["embedding_model"], "reranker": identity, "settings": settings, "versions": versions, "code_sha256": code,
                "changed_setting": {"retrieve_fusion_k": {"before": 10, "after": 15}}, "rounds": 1, "queries": 104,
                "warmup_retrievals": 0, "corpus_writes": 0, "baseline_directory": "eval/results/rerank-2026-11-17",
                "baseline_files_sha256": {f: hashlib.sha256((baseline_dir / f).read_bytes()).hexdigest() for f in ("manifest.json", "summary.json", "results.jsonl")},
                "baseline": "Saved reranked top five from RRF ten; live routes and RRF-ten prefix must exactly reproduce it",
                "reranker_input": "Original query and complete chunk.embed_text; no source labels or gold references",
                "selection_policy": "Keep RRF fifteen only if primary-source hit@5 strictly improves; otherwise restore RRF ten"}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "manifest.json"
    if path.exists() and digest(json.loads(path.read_text(encoding="utf-8"))) != digest(manifest):
        raise ValueError("Partial round fingerprint changed")
    write_json(path, manifest)
    write_json(args.out / "questions.json", questions)
    if not (args.out / "startup.json").exists():
        write_json(args.out / "startup.json", {"model_load_seconds": load_seconds, "excluded_from_per_question_timing": True, "inference_warmups": 0})
    path = args.out / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    done = {r["id"] for r in rows}
    if len(done) != len(rows) or not done.issubset(previous):
        raise ValueError("Invalid partial round")
    passage_path = args.out / "passages.json"
    passages = json.loads(passage_path.read_text(encoding="utf-8")) if passage_path.exists() else {}
    with path.open("a", encoding="utf-8", newline="\n") as output:
        for i, (question, vector) in enumerate(zip(questions, vectors), 1):
            if question["id"] in done:
                continue
            row, candidate_passages = observe(question, vector, previous[question["id"]])
            passages.update(candidate_passages)
            write_json(passage_path, passages)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            rows.append(row)
            if i % 5 == 0 or i == 104:
                print(f"RRF-15 round: {i}/104; candidates {row['actual_fusion_count']}; last rerank {row['rerank_seconds']:.2f}s", flush=True)
    after = snapshot("rag_materials_v2")
    if digest(after) != digest(corpus) or code != {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}:
        raise ValueError("Corpus or code changed during the round")
    for name, expected in manifest["baseline_files_sha256"].items():
        if hashlib.sha256((baseline_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Saved baseline changed during the round")
    summary = summarize(rows, baseline_summary)
    write_json(args.out / "corpus_after.json", after)
    write_json(args.out / "changes.json", [r for r in rows if any(r["baseline"][key] != r["reranked"][key] for key in ("hit_at_3", "hit_at_5", "mrr_at_5"))])
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
