from __future__ import annotations

import argparse
import json
import logging
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
from pgvector.psycopg import register_vector

from src.db.connection import ROOT, get_conn
from src.db.queries import fetch_chunks, vector_search, corpus_table
from src.llm.client import LLMUnavailable
from src.rag.hybrid import bm25_search, fuse_rankings
from src.rag.corpus_hygiene import filter_chunks
from src.rag.listwise import DEFAULT_MODEL as LISTWISE_MODEL, rank_listwise
from src.rag.reranker import rerank_matches, options as reranker_options


QUERY_INSTRUCTION = (
    "Given a tax question, retrieve relevant passages from IRS publications "
    "that answer the question."
)
EMBEDDING_DIM = 1024


def embed_query(query: str) -> np.ndarray:
    if not query.strip():
        raise ValueError("The retrieval query must not be empty.")
    return embed_texts([f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query.strip()}"])[0]


def embed_texts(texts: list[str], options: dict | None = None) -> np.ndarray:
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Embedding inputs must not be empty.")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    payload = json.dumps({
        "model": os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
        "input": texts,
        "truncate": False,
        **({"options": options} if options else {}),
    }).encode("utf-8")
    request = Request(
        f"{base_url}/api/embed", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Ollama embedding request failed. Check OLLAMA_BASE_URL, "
            "OLLAMA_EMBEDDING_MODEL, and that Ollama is running."
        ) from exc

    embeddings = result.get("embeddings", [])
    if len(embeddings) != len(texts):
        raise RuntimeError("Ollama returned an unexpected number of embeddings.")
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.shape != (len(texts), EMBEDDING_DIM):
        raise RuntimeError(
            f"Ollama returned shape {vectors.shape}; the database requires "
            f"{EMBEDDING_DIM} dimensions."
        )
    if not np.isfinite(vectors).all() or not np.any(vectors, axis=1).all():
        raise RuntimeError("Ollama returned a non-finite or zero embedding.")
    return vectors


def retrieve(query: str, *, candidate_k: int | None = None, exclude_reference_chunks: bool | None = None,
             corpus_version: str | None = None, rerank_enabled: bool | None = None,
             question: dict | None = None) -> dict:
    if not query.strip():
        raise ValueError("The retrieval query must not be empty.")
    settings = json.loads((ROOT / "config/settings.json").read_text(encoding="utf-8"))
    corpus = corpus_version if corpus_version is not None else settings.get("corpus_version", "original")
    table = corpus_table(corpus)
    corpus_options = {"corpus": corpus} if corpus != "original" else {}
    eligibility = " AND retrieval_eligible" if corpus != "original" else ""
    tax_year = settings["tax_year"]
    top_k = settings["retrieve_top_k"]
    if candidate_k is None:
        candidate_k = settings.get("retrieve_candidate_k", top_k)
    if type(candidate_k) is not int or candidate_k < top_k:
        raise ValueError("candidate_k must be an integer at least retrieve_top_k.")
    if exclude_reference_chunks is None:
        exclude_reference_chunks = settings.get("exclude_reference_chunks", False)
    if type(exclude_reference_chunks) is not bool:
        raise ValueError("exclude_reference_chunks must be a boolean.")
    if rerank_enabled is None:
        rerank_enabled = settings.get("rerank_enabled", False)
    if type(rerank_enabled) is not bool:
        raise ValueError("rerank_enabled must be a boolean.")
    rerank_method = settings.get("rerank_method", "bge") if rerank_enabled else None
    if rerank_enabled and rerank_method not in ("bge", "listwise"):
        raise ValueError("rerank_method must be bge or listwise.")
    fusion_k = settings.get("retrieve_fusion_k", top_k)
    if type(fusion_k) is not int or not top_k <= fusion_k <= 2 * candidate_k:
        raise ValueError("retrieve_fusion_k must cover the output and fit within the candidate union.")
    if rerank_enabled:
        if settings.get("rerank_top_k", top_k) != top_k:
            raise ValueError("rerank_top_k must match retrieve_top_k.")
        if rerank_method == "bge":
            reranker_options(settings)
    rrf_constant = settings["rrf_constant"]
    model = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

    with get_conn() as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        register_vector(conn)
        available = conn.execute(
            f"SELECT count(*) FROM {table} WHERE tax_year = %s "
            f"AND embedding_model = %s AND embedding IS NOT NULL{eligibility}",
            [tax_year, model],
        ).fetchone()[0]
        if not available:
            raise RuntimeError(
                f"No stored embeddings for tax year {tax_year} and model {model!r}. "
                "Populate chunks.embedding and embedding_model before retrieval."
            )
        has_hnsw = conn.execute(f"""
            SELECT EXISTS (
                SELECT 1 FROM pg_index i
                JOIN pg_class idx ON idx.oid = i.indexrelid
                JOIN pg_am am ON am.oid = idx.relam
                JOIN pg_opclass ops ON ops.oid = i.indclass[0]
                WHERE i.indrelid = '{table}'::regclass
                  AND i.indisvalid AND am.amname = 'hnsw'
                  AND ops.opcname = 'vector_cosine_ops'
                  AND pg_get_indexdef(i.indexrelid, 1, true) = 'embedding'
            )
        """).fetchone()[0]
        if not has_hnsw:
            raise RuntimeError(
                f"No valid cosine HNSW index on {table}.embedding. "
                "The existing src.db.load_embeddings command creates this index."
            )

        query_vector = embed_query(query)
        chunks = fetch_chunks(conn, tax_year, **corpus_options)
        excluded_ids = []
        if exclude_reference_chunks:
            chunks, excluded_ids, stale = filter_chunks(chunks)
            if stale:
                logging.getLogger(__name__).warning("Retained %d chunks with stale exclusion fingerprints", len(stale))
        conn.execute("SET LOCAL enable_seqscan = off")
        options = {"excluded_ids": excluded_ids} if excluded_ids else {}
        rows = vector_search(conn, query_vector, model, tax_year, k=candidate_k, **options, **corpus_options)

    vector_matches = [
        {"chunk_id": row[0], "citation": row[1], "text": row[2], "score": row[3]}
        for row in rows
    ]
    keyword_matches = bm25_search(query, chunks, k=candidate_k)
    fused = fuse_rankings(vector_matches, keyword_matches, k=fusion_k, rrf_constant=rrf_constant)
    matches = fused[:top_k]
    method = "hnsw_bm25_rrf"
    fallback = False
    if rerank_method == "bge":
        matches = rerank_matches(query, fused, {c["id"]: c["embed_text"] for c in chunks}, k=top_k, settings=settings)
        method = "hnsw_bm25_rrf_rerank"
    elif rerank_method == "listwise" and len(fused) > 1:
        candidate_ids = {match["chunk_id"] for match in fused}
        passages = {c["id"]: c["embed_text"] for c in chunks if c["id"] in candidate_ids}
        try:
            matches = rank_listwise(query, fused, passages, tax_year=tax_year, k=top_k,
                                    model=settings.get("listwise_model", LISTWISE_MODEL), question=question)
            method = "hnsw_bm25_rrf_listwise"
        except LLMUnavailable:
            logging.getLogger(__name__).warning("Listwise reranking unavailable. Using RRF order.")
            fallback = True
    provenance = {c["id"]: c.get("provenance", []) for c in chunks}
    for match in matches:
        aliases = [p["citation"] for p in provenance.get(match["chunk_id"], []) if p["citation"] != match["citation"]]
        if aliases:
            match["additional_citations"] = aliases
    return {
        "tax_year": tax_year,
        "embedding_model": model,
        "retrieval_method": method,
        **({"rerank_fallback": True} if fallback else {}),
        "matches": matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Question to retrieve IRS passages for")
    args = parser.parse_args()
    print(json.dumps(retrieve(args.query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
