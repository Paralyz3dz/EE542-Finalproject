import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from src.db.connection import ROOT, get_conn
from src.db.load_chunks import COLS
from src.db.queries import corpus_table


def validate(chunks, tables, vectors):
    ids = {c["id"] for c in chunks}
    expected = {d["name"] + " (2025)" for d in json.loads((ROOT / "config/knowledge_sources.json").read_text(encoding="utf-8"))}
    if len(ids) != len(chunks) or {c["source"] for c in chunks} != expected or set(vectors) != ids:
        raise ValueError("Invalid corpus inventory")
    model = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    for chunk in chunks:
        row = vectors[chunk["id"]]
        digest = hashlib.sha256(chunk["embed_text"].encode()).hexdigest()[:16]
        vector = np.asarray(row["embedding"], dtype=np.float32)
        if (chunk["tax_year"] != 2025 or not chunk["text"] or len(chunk["text"].split()) > 400
                or chunk["content_hash"] != digest or row["content_hash"] != digest
                or row["embedding_model"] != model or vector.shape != (1024,)
                or not np.isfinite(vector).all() or not np.any(vector) or not chunk["provenance"] or type(chunk["retrieval_eligible"]) is not bool
                or not chunk["citation"].startswith(chunk["source"] + ",")):
            raise ValueError("Invalid chunk/vector: " + chunk["id"])
    if len({t["chunk_id"] for t in tables}) != len(tables) or any(t["chunk_id"] not in ids for t in tables):
        raise ValueError("Invalid table links")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/materials-v2")
    parser.add_argument("--corpus", default=json.loads((ROOT / "config/settings.json").read_text(encoding="utf-8"))["corpus_version"])
    args = parser.parse_args()
    load_dotenv(args.env_file)
    target = corpus_table(args.corpus)
    table_target = corpus_table(args.corpus, "tax_tables")
    if args.corpus == "original":
        raise ValueError("The original public corpus must be preserved")
    chunks = json.loads((args.input / "chunks.json").read_text(encoding="utf-8"))
    tables = json.loads((args.input / "tables.json").read_text(encoding="utf-8"))
    vectors = {}
    with (args.input / "embeddings.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["id"] in vectors:
                raise ValueError("Duplicate vector")
            vectors[row["id"]] = row
    validate(chunks, tables, vectors)
    with get_conn() as conn:
        register_vector(conn)
        before = conn.execute("SELECT count(*), md5(string_agg(id || content_hash || md5(embedding::text), ',' ORDER BY id)) FROM public.chunks").fetchone()
        if conn.execute("SELECT to_regclass(%s), to_regclass(%s)", [target, table_target]).fetchone() != (None, None):
            raise ValueError("Target corpus tables already exist; no data was overwritten")
        conn.execute(f"CREATE TABLE {target} (LIKE public.chunks INCLUDING DEFAULTS INCLUDING CONSTRAINTS)")
        conn.execute(f"ALTER TABLE {target} ADD PRIMARY KEY (id), ADD COLUMN provenance JSONB NOT NULL, ADD COLUMN retrieval_eligible BOOLEAN NOT NULL")
        conn.execute(f"CREATE INDEX {args.corpus}_year_type ON {target} (tax_year, type)")
        conn.execute(f"CREATE TABLE {table_target} (LIKE public.tax_tables INCLUDING DEFAULTS INCLUDING CONSTRAINTS)")
        conn.execute(f"ALTER TABLE {table_target} ADD COLUMN metadata JSONB NOT NULL, ADD PRIMARY KEY (chunk_id), ADD FOREIGN KEY (chunk_id) REFERENCES {target}(id) ON DELETE CASCADE")
        columns = COLS + ["embedding", "embedding_model", "provenance", "retrieval_eligible"]
        insert = f"INSERT INTO {target} ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))})"
        with conn.cursor() as cursor:
            cursor.executemany(insert, [[c.get(col) for col in COLS] + [np.asarray(vectors[c["id"]]["embedding"], dtype=np.float32), vectors[c["id"]]["embedding_model"], Jsonb(c["provenance"]), c["retrieval_eligible"]] for c in chunks])
            cursor.executemany(f"INSERT INTO {table_target} (chunk_id, caption, kind, citation, chapter_no, printed_page, pdf_page, rows, metadata) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                               [[t[k] for k in ("chunk_id", "caption", "kind", "citation", "chapter_no", "printed_page", "pdf_page")] + [Jsonb(t["rows"]), Jsonb(t["metadata"])] for t in tables])
        conn.execute(f"CREATE INDEX {args.corpus}_embedding_hnsw ON {target} USING hnsw (embedding vector_cosine_ops) WHERE retrieval_eligible")
        conn.execute(f"ANALYZE {target}")
        after = conn.execute("SELECT count(*), md5(string_agg(id || content_hash || md5(embedding::text), ',' ORDER BY id)) FROM public.chunks").fetchone()
        counts = conn.execute(f"SELECT count(*), count(embedding), count(DISTINCT source) FROM {target}").fetchone()
        table_count = conn.execute(f"SELECT count(*) FROM {table_target}").fetchone()[0]
        source_count = len({chunk["source"] for chunk in chunks})
        if before != after or tuple(counts) != (len(chunks), len(chunks), source_count) or table_count != len(tables):
            raise ValueError("Remote corpus verification failed")
    result = {"corpus": args.corpus, "chunks": counts[0], "vectors": counts[1], "sources": counts[2], "tables": table_count,
              "public_before": before, "public_after": after,
              "files_sha256": {name: hashlib.sha256((args.input / name).read_bytes()).hexdigest() for name in ("chunks.json", "tables.json", "embeddings.jsonl")}}
    (args.input / "remote_load.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
