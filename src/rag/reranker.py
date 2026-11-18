import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.db.connection import ROOT


DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def model_path(model, revision):
    if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", model) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("A public model ID and pinned commit revision are required")
    return ROOT / "models" / model.replace("/", "--") / revision


def options(settings):
    result = {"model": os.getenv("RERANKER_MODEL", DEFAULT_MODEL),
              "revision": settings.get("reranker_revision", DEFAULT_REVISION),
              "batch_size": settings.get("rerank_batch_size", 4),
              "max_length": settings.get("rerank_max_length", 8192),
              "num_threads": settings.get("rerank_num_threads", 12)}
    model_path(result["model"], result["revision"])
    if any(type(result[key]) is not int or result[key] < 1 for key in ("batch_size", "max_length", "num_threads")):
        raise ValueError("Reranker batch size, maximum length and thread count must be positive integers")
    return result


class LocalReranker:
    def __init__(self, model, revision, max_length, num_threads):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = model_path(model, revision)
        identity_path = path / "model_identity.json"
        if not identity_path.exists():
            raise RuntimeError("Reranker is not prepared; run python -m src.rag.download_reranker")
        self.identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if self.identity["model"] != model or self.identity["revision"] != revision:
            raise ValueError("Reranker identity differs from the pinned configuration")
        for name, expected in self.identity["files_sha256"].items():
            with (path / name).open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != expected:
                raise ValueError("Reranker file fingerprint changed: " + name)
        torch.set_num_threads(num_threads)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.model, loading = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=torch.float32, output_loading_info=True)
        if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise ValueError("The pinned reranker weights did not load completely")
        self.model.to("cpu").eval()
        capacity = self.model.config.max_position_embeddings - self.model.config.pad_token_id - 1
        if max_length > capacity or self.model.config.num_labels != 1:
            raise ValueError("Unsupported reranker capacity or label count")
        self.max_length = max_length
        self.num_threads = num_threads

    def score(self, pairs, batch_size):
        import torch
        values = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            inputs = self.tokenizer([p[0] for p in batch], [p[1] for p in batch], padding=True, truncation=False, return_tensors="pt")
            if inputs["input_ids"].shape[1] > self.max_length:
                raise ValueError("Reranker input exceeds the configured capacity; no passage was truncated")
            with torch.inference_mode():
                scores = self.model(**inputs, return_dict=True).logits.reshape(-1).float().cpu().numpy()
            values.extend(float(value) for value in scores)
        return np.asarray(values, dtype=np.float32)


@lru_cache(maxsize=1)
def load_model(model, revision, max_length, num_threads):
    return LocalReranker(model, revision, max_length, num_threads)


def prepare(settings):
    configured = options(settings)
    return load_model(configured["model"], configured["revision"], configured["max_length"], configured["num_threads"])


def rrf_weight(settings):
    value = settings.get("rerank_rrf_weight", 0.0)
    if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("rerank_rrf_weight must be a finite number between zero and one")
    return float(value)


def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    spread = float(np.max(values) - np.min(values))
    return (values - np.min(values)) / spread if spread else np.zeros_like(values)


def rank_scored_matches(matches, scores, *, k=5, weight=0.0):
    weight = rrf_weight({"rerank_rrf_weight": weight})
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (len(matches),) or not np.isfinite(values).all():
        raise RuntimeError("Reranker returned invalid relevance scores")
    if not matches:
        return []
    rrf = np.asarray([m["score"] for m in matches], dtype=np.float64)
    if not np.isfinite(rrf).all():
        raise ValueError("RRF returned invalid relevance scores")
    ranked = [{**match, "rrf_score": match["score"], "rrf_rank": i,
               "score": float(score), "rerank_score": float(score)}
              for i, (match, score) in enumerate(zip(matches, values), 1)]
    pure = sorted(ranked, key=lambda match: (-match["rerank_score"], match["rrf_rank"], match["chunk_id"]))
    for i, match in enumerate(pure, 1):
        match["rerank_rank"] = i
    if weight == 0:
        return pure[:k]
    rrf_normalized, rerank_normalized = minmax(rrf), minmax(values)
    for match, a, b in zip(ranked, rrf_normalized, rerank_normalized):
        match.update(normalized_rrf_score=float(a), normalized_rerank_score=float(b),
                     blend_score=float(weight * a + (1 - weight) * b))
        match["score"] = match["blend_score"]
    ranked.sort(key=lambda match: (-match["blend_score"], match["rrf_rank"], match["chunk_id"]))
    return [{**match, "blend_rank": i} for i, match in enumerate(ranked[:k], 1)]


def rerank_matches(query, matches, passages, *, k=5, settings=None, scorer=None):
    if not query.strip() or type(k) is not int or k < 1:
        raise ValueError("A nonempty query and positive output count are required")
    if not matches:
        return []
    ids = [m["chunk_id"] for m in matches]
    if len(set(ids)) != len(ids) or any(cid not in passages or not passages[cid].strip() for cid in ids):
        raise ValueError("Reranker requires unique candidates with their complete passages")
    weight = rrf_weight(settings or {})
    pairs = [(query, passages[cid]) for cid in ids]
    if scorer is None:
        configured = options(settings or {})
        scorer = prepare(settings or {}).score
        scores = scorer(pairs, batch_size=configured["batch_size"])
    else:
        scores = scorer(pairs)
    return rank_scored_matches(matches, scores, k=k, weight=weight)
