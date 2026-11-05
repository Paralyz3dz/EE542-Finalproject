"""
load_embeddings.py: write precomputed embeddings into chunks.embedding.

Input file: one JSON object per line, {"id": "pub17_2025_ch01_011", "embedding": [0.01, ...]},
computed from each chunk's embed_text (not text).

Usage (from the tax-agent folder)
  python -m src.db.load_embeddings data/processed/embeddings_qwen3-0.6b.jsonl --model Qwen/Qwen3-Embedding-0.6B
"""
import argparse
import json

import numpy as np

from src.db.connection import get_conn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--model", required=True, help="model name, stored next to every vector")
    ap.add_argument("--dim", type=int, default=1024, help="must match vector(N) in schema.sql")
    args = ap.parse_args()

    rows = []
    with open(args.file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            vec = np.asarray(r["embedding"], dtype=np.float32)
            if vec.shape != (args.dim,):
                raise SystemExit(f"{r['id']}: got {vec.shape[0]} dims, expected {args.dim}")
            rows.append((vec, args.model, r["id"]))

    with get_conn(register_vectors=True) as conn:
        known = {r[0] for r in conn.execute("SELECT id FROM chunks").fetchall()}
        unknown = [r[2] for r in rows if r[2] not in known]
        if unknown:
            print(f"warning: {len(unknown)} ids not in the database (re-run load_chunks?), e.g. {unknown[:3]}")
        with conn.cursor() as cur:
            cur.executemany("UPDATE chunks SET embedding = %s, embedding_model = %s WHERE id = %s",
                            [r for r in rows if r[2] in known])
        # build the vector index once vectors exist (cosine distance, same as queries.vector_search)
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
                     "ON chunks USING hnsw (embedding vector_cosine_ops)")
        missing = conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]

    print(f"updated {len(rows) - len(unknown)} vectors; chunks still without embedding: {missing}")


if __name__ == "__main__":
    main()
