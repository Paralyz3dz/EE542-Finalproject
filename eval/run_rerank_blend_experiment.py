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
from eval.run_rerank_experiment import KEYS, observe as observe_base
from src.db.connection import ROOT
from src.rag import reranker
from src.workflow.engine import load_scenario


EXTRA_KEYS = ("normalized_rrf_score", "normalized_rerank_score", "blend_score", "blend_rank")


def compact(matches):
    return [{key: match[key] for key in (*KEYS, *EXTRA_KEYS) if key in match} for match in matches]


def observe(question, vector, previous):
    trace = {}
    original = reranker.rank_scored_matches

    def score(matches, scores, *, k, weight):
        if len(matches) != 10 or k != 5 or weight != 0.5:
            raise ValueError("Expected ten candidates, fixed half-weight blend and five outputs")
        pure = original(matches, scores, k=10, weight=0.0)
        blended = original(matches, scores, k=10, weight=0.5)
        trace.update(pure=pure, blended=blended)
        return blended[:k]

    with patch.object(reranker, "rank_scored_matches", side_effect=score) as scoring:
        row, passages = observe_base(question, vector)
    if scoring.call_count != 1:
        raise ValueError("Exactly one scoring pass is required")
    for route in ("vector", "bm25", "rrf"):
        if row["routes"][route] != previous["routes"][route]:
            raise ValueError("Candidate route differs from saved baseline: " + route)
    pure, blended = trace["pure"][:5], trace["blended"][:5]
    if [m["chunk_id"] for m in blended] != [m["chunk_id"] for m in row["matches"]]:
        raise ValueError("Production output differs from blended scores")
    aliases = {m["chunk_id"]: m.get("additional_citations", []) for m in row["matches"]}
    blended = [{**m, **({"additional_citations": aliases[m["chunk_id"]]} if aliases.get(m["chunk_id"]) else {})} for m in blended]
    baseline = source_metrics(question, pure)
    saved_metrics_reproduced = all(baseline[key] == previous["reranked"][key] for key in ("hit_at_3", "hit_at_5", "mrr_at_5"))
    return {"id": row["id"], "query": row["query"], "source": row["source"], "round": 1,
            "candidate_k": 10, "fusion_k": 10, "output_k": 5, "rrf_weight": 0.5,
            "elapsed_seconds": row["elapsed_seconds"], "retrieval_seconds": row["retrieval_seconds"], "rerank_seconds": row["rerank_seconds"],
            "baseline": baseline, "blended": source_metrics(question, blended),
            "baseline_matches": compact(pure), "matches": compact(blended),
            "rrf_source_coverage_at_10": row["rrf_source_coverage_at_10"],
            "saved_baseline_rankings_reproduced": [m["chunk_id"] for m in pure] == [m["chunk_id"] for m in previous["matches"]],
            "saved_baseline_metrics_reproduced": saved_metrics_reproduced,
            "routes": row["routes"], "scored_candidates": compact(trace["blended"])}, passages


def summarize(rows, saved_summary):
    if len(rows) != 104 or len({r["id"] for r in rows}) != 104 or {r["round"] for r in rows} != {1}:
        raise ValueError("Expected one complete 104-question round")
    def metrics(key):
        return {"hits_at_3": sum(r[key]["hit_at_3"] for r in rows), "hit_at_3": float(np.mean([r[key]["hit_at_3"] for r in rows])),
                "hits_at_5": sum(r[key]["hit_at_5"] for r in rows), "hit_at_5": float(np.mean([r[key]["hit_at_5"] for r in rows])),
                "mrr_at_5": float(np.mean([r[key]["mrr_at_5"] for r in rows]))}
    baseline, blended = metrics("baseline"), metrics("blended")
    return {"rounds": 1, "queries": 104, "rrf_weight": 0.5, "paired_pure_rerank_top5": baseline, "blended_top5": blended,
            "saved_pure_rerank_top5": saved_summary["reranked_top5"],
            "rrf_source_coverage_at_10": sum(r["rrf_source_coverage_at_10"] for r in rows),
            "gained_hit_at_5": [r["id"] for r in rows if not r["baseline"]["hit_at_5"] and r["blended"]["hit_at_5"]],
            "lost_hit_at_5": [r["id"] for r in rows if r["baseline"]["hit_at_5"] and not r["blended"]["hit_at_5"]],
            "saved_baseline_ranking_differences": [r["id"] for r in rows if not r["saved_baseline_rankings_reproduced"]],
            "saved_baseline_metric_differences": [r["id"] for r in rows if not r["saved_baseline_metrics_reproduced"]],
            "saved_baseline_mean_seconds": saved_summary["mean_total_seconds"],
            "mean_retrieval_seconds": float(np.mean([r["retrieval_seconds"] for r in rows])),
            "mean_rerank_seconds": float(np.mean([r["rerank_seconds"] for r in rows])),
            "mean_total_seconds": float(np.mean([r["elapsed_seconds"] for r in rows])),
            "p95_total_seconds": float(np.percentile([r["elapsed_seconds"] for r in rows], 95)),
            "primary_improved": blended["hits_at_5"] > baseline["hits_at_5"],
            "limitations": "Strict primary-document source coverage only; no answer generator or semantic sufficiency judge"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "eval/results/rerank-blend50-2026-11-17")
    args = parser.parse_args()
    if (args.out / "summary.json").exists():
        raise ValueError("This single round is complete and cannot be rerun")
    load_dotenv(args.env_file)
    baseline_dir = ROOT / "eval/results/rerank-2026-11-17"
    prior = json.loads((baseline_dir / "manifest.json").read_text(encoding="utf-8"))
    saved_summary = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    previous_rows = [json.loads(line) for line in (baseline_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    previous = {r["id"]: r for r in previous_rows}
    _, questions = load_scenario("intake")
    cache = json.loads((ROOT / "data/processed/candidate_queries.json").read_text(encoding="utf-8"))
    vectors = np.asarray(cache["vectors"], dtype=np.float32)
    expected_identity = {"questions": digest(questions), "model": prior["embedding_model"], "instruction": "Given a tax question, retrieve relevant passages from IRS publications that answer the question.", "dimensions": 1024}
    if len(questions) != 104 or len(previous_rows) != len(previous) or set(previous) != {q["id"] for q in questions} or digest(questions) != prior["questions_sha256"] or cache["identity"] != expected_identity:
        raise ValueError("Questions or cached query identity changed")
    if vectors.shape != (104, 1024) or hashlib.sha256(vectors.tobytes()).hexdigest() != prior["query_vectors_sha256"]:
        raise ValueError("Cached query vectors changed")
    if os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B") != prior["embedding_model"]["embedding_model"]:
        raise ValueError("Embedding model changed")
    with urlopen(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/tags", timeout=15) as response:
        live = next(m for m in json.load(response)["models"] if m["name"] == prior["embedding_model"]["ollama_model"])
    if live["digest"] != prior["embedding_model"]["digest"]:
        raise ValueError("Embedding weights changed")
    settings = json.loads((ROOT / "config/settings.json").read_text(encoding="utf-8"))
    if settings != {**prior["effective_settings"], "rerank_rrf_weight": 0.5}:
        raise ValueError("Only the final scoring policy may change")
    corpus = snapshot("rag_materials_v2")
    if digest(corpus) != digest(prior["corpus"]):
        raise ValueError("The document/vector corpus changed")
    for name, expected in prior["code_sha256"].items():
        if name not in ("config/settings.json", "src/rag/reranker.py") and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Unrelated implementation changed: " + name)
    before = args.out / "reranker_before.py"
    if hashlib.sha256(before.read_bytes()).hexdigest() != prior["code_sha256"]["src/rag/reranker.py"]:
        raise ValueError("Baseline reranker snapshot differs")
    tick = time.perf_counter()
    ranker = reranker.prepare(settings)
    load_seconds = time.perf_counter() - tick
    identity = {**ranker.identity, **reranker.options(settings), "device": "cpu", "dtype": "float32", "truncation": False}
    versions = {name: importlib.metadata.version(name) for name in prior["versions"]}
    if identity != prior["reranker"] or versions != prior["versions"]:
        raise ValueError("Reranker model or dependency versions changed")
    files = (*prior["code_sha256"], "eval/run_rerank_blend_experiment.py")
    code = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}
    manifest = {"corpus": corpus, "questions_sha256": digest(questions), "query_vectors_sha256": prior["query_vectors_sha256"],
                "embedding_model": prior["embedding_model"], "reranker": identity, "settings": settings, "versions": versions, "code_sha256": code,
                "changed_setting": {"rerank_rrf_weight": {"before": 0.0, "after": 0.5}},
                "normalization": "Per-query min-max independently across the same ten candidates; constant score lists normalize to zero",
                "formula": "0.5 * normalized_rrf_score + 0.5 * normalized_rerank_score",
                "tie_break": "Original RRF rank, then chunk ID", "rounds": 1, "queries": 104, "warmup_retrievals": 0, "corpus_writes": 0,
                "baseline_directory": "eval/results/rerank-2026-11-17",
                "baseline_files_sha256": {f: hashlib.sha256((baseline_dir / f).read_bytes()).hexdigest() for f in ("manifest.json", "summary.json", "results.jsonl")},
                "baseline": "Pure rerank top five computed from the same model scores; checked against saved rankings and metrics",
                "selection_policy": "Keep weight 0.5 only if primary-source hit@5 strictly improves; otherwise restore weight zero"}
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
                print(f"Blend-50 round: {i}/104; last rerank {row['rerank_seconds']:.2f}s", flush=True)
    after = snapshot("rag_materials_v2")
    if digest(after) != digest(corpus) or code != {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}:
        raise ValueError("Corpus or code changed during the round")
    for name, expected in manifest["baseline_files_sha256"].items():
        if hashlib.sha256((baseline_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Saved baseline changed during the round")
    summary = summarize(rows, saved_summary)
    write_json(args.out / "corpus_after.json", after)
    write_json(args.out / "changes.json", [r for r in rows if any(r["baseline"][key] != r["blended"][key] for key in ("hit_at_3", "hit_at_5", "mrr_at_5"))])
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
