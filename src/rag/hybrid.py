from __future__ import annotations

import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold())


def bm25_search(query: str, chunks: list[dict], k: int = 5) -> list[dict]:
    if k < 1:
        raise ValueError("k must be positive.")
    query_tokens = tokenize(query)
    documents = [(chunk, tokenize(chunk["embed_text"])) for chunk in chunks]
    documents = [(chunk, tokens) for chunk, tokens in documents if tokens]
    if not query_tokens or not documents:
        return []

    scores = BM25Okapi([tokens for _, tokens in documents]).get_scores(query_tokens)
    query_terms = set(query_tokens)
    matched = [
        (chunk, float(score))
        for (chunk, tokens), score in zip(documents, scores)
        if query_terms.intersection(tokens)
    ]
    matched.sort(key=lambda item: (-item[1], item[0]["id"]))
    return [
        {"chunk_id": chunk["id"], "citation": chunk["citation"],
         "text": chunk["text"], "score": score}
        for chunk, score in matched[:k]
    ]


def fuse_rankings(
    vector_matches: list[dict], keyword_matches: list[dict],
    k: int = 5, rrf_constant: int = 60,
) -> list[dict]:
    if k < 1 or rrf_constant < 0:
        raise ValueError("k must be positive and rrf_constant nonnegative.")
    merged = {}
    for source, matches in (("vector", vector_matches), ("bm25", keyword_matches)):
        seen = set()
        for match in matches:
            chunk_id = match["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank = len(seen)
            if chunk_id not in merged:
                merged[chunk_id] = {
                    "chunk_id": chunk_id, "citation": match["citation"],
                    "text": match["text"], "score": 0.0,
                    "vector_score": None, "bm25_score": None,
                    "vector_rank": None, "bm25_rank": None,
                }
            entry = merged[chunk_id]
            entry["score"] += 1.0 / (rrf_constant + rank)
            entry[f"{source}_score"] = match["score"]
            entry[f"{source}_rank"] = rank

    return sorted(merged.values(), key=lambda row: (-row["score"], row["chunk_id"]))[:k]
