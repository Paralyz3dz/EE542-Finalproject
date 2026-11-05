from __future__ import annotations

import argparse
import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
from pgvector.psycopg import register_vector

from src.db.connection import ROOT, get_conn
from src.db.queries import fetch_chunks, vector_search
from src.rag.hybrid import bm25_search, fuse_rankings


QUERY_INSTRUCTION = (
    "Given a tax question, retrieve relevant passages from IRS publications "
    "that answer the question."
)
EMBEDDING_DIM = 1024


def embed_query(query: str) -> np.ndarray:
    if not query.strip():
        raise ValueError("The retrieval query must not be empty.")
    return embed_texts([f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query.strip()}"])[0]


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Embedding inputs must not be empty.")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    payload = json.dumps({
        "model": os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
        "input": texts,
        "truncate": False,
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


def retrieve(query: str) -> dict:
    if not query.strip():
        raise ValueError("The retrieval query must not be empty.")
    settings = json.loads((ROOT / "config/settings.json").read_text(encoding="utf-8"))
    tax_year = settings["tax_year"]
    top_k = settings["retrieve_top_k"]
    rrf_constant = settings["rrf_constant"]
    model = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

    with get_conn() as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        register_vector(conn)
        available = conn.execute(
            "SELECT count(*) FROM public.chunks WHERE tax_year = %s "
            "AND embedding_model = %s AND embedding IS NOT NULL",
            [tax_year, model],
        ).fetchone()[0]
        if not available:
            raise RuntimeError(
                f"No stored embeddings for tax year {tax_year} and model {model!r}. "
                "Populate chunks.embedding and embedding_model before retrieval."
            )
        has_hnsw = conn.execute("""
            SELECT EXISTS (
                SELECT 1 FROM pg_index i
                JOIN pg_class idx ON idx.oid = i.indexrelid
                JOIN pg_am am ON am.oid = idx.relam
                JOIN pg_opclass ops ON ops.oid = i.indclass[0]
                WHERE i.indrelid = 'public.chunks'::regclass
                  AND i.indisvalid AND am.amname = 'hnsw'
                  AND ops.opcname = 'vector_cosine_ops'
                  AND pg_get_indexdef(i.indexrelid, 1, true) = 'embedding'
            )
        """).fetchone()[0]
        if not has_hnsw:
            raise RuntimeError(
                "No valid cosine HNSW index on public.chunks.embedding. "
                "The existing src.db.load_embeddings command creates this index."
            )

        query_vector = embed_query(query)
        chunks = fetch_chunks(conn, tax_year)
        conn.execute("SET LOCAL enable_seqscan = off")
        rows = vector_search(conn, query_vector, model, tax_year, k=top_k)

    vector_matches = [
        {"chunk_id": row[0], "citation": row[1], "text": row[2], "score": row[3]}
        for row in rows
    ]
    keyword_matches = bm25_search(query, chunks, k=top_k)
    return {
        "tax_year": tax_year,
        "embedding_model": model,
        "retrieval_method": "hnsw_bm25_rrf",
        "matches": fuse_rankings(vector_matches, keyword_matches, k=top_k,
                                 rrf_constant=rrf_constant),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Question to retrieve IRS passages for")
    args = parser.parse_args()
    print(json.dumps(retrieve(args.query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
