import json
from collections import Counter

import numpy as np
from psycopg.types.json import Jsonb

from src.db.connection import ROOT, get_conn
from src.db.load_chunks import COLS, UPSERT_CHUNK, UPSERT_TABLE, content_hash
from src.workflow.engine import _doc_key

OUT = ROOT / "data/processed/knowledge_base"


def main():
    chunks = json.loads((OUT / "chunks.json").read_text(encoding="utf-8"))
    tables = json.loads((OUT / "tables.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    expected_sources = {f"{d['name']} (2025)" for d in manifest if d["name"] != "IRS Pub 17"}
    ids = {c["id"] for c in chunks}
    if len(ids) != len(chunks) or {c["source"] for c in chunks} != expected_sources:
        raise ValueError("Duplicate IDs or unexpected document inventory")
    for chunk in chunks:
        if (chunk["tax_year"] != 2025 or not chunk["text"].strip()
                or len(chunk["text"].split()) > 400
                or chunk["type"] not in {"rule", "table", "tip", "caution", "whats_new"}
                or _doc_key(chunk["citation"]) != _doc_key(chunk["source"])):
            raise ValueError(f"Invalid chunk metadata: {chunk['id']}")
    if any(t["chunk_id"] not in ids for t in tables):
        raise ValueError("A table references a missing chunk")
    vectors = {}
    for line in (OUT / "embeddings.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        vector = np.asarray(row["embedding"], dtype=np.float32)
        if vector.shape != (1024,) or not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError(f"Invalid vector: {row['id']}")
        if row["id"] in vectors:
            raise ValueError(f"Duplicate vector: {row['id']}")
        vectors[row["id"]] = row
    if len(vectors) != len(chunks):
        raise ValueError("Vector and chunk counts differ")
    for chunk in chunks:
        chunk["content_hash"] = content_hash(chunk["embed_text"])
        vector = vectors[chunk["id"]]
        if vector["content_hash"] != chunk["content_hash"]:
            raise ValueError(f"Stale vector: {chunk['id']}")
        if vector["embedding_model"] != "Qwen/Qwen3-Embedding-0.6B":
            raise ValueError("Unexpected embedding model")
    expected = Counter(c["source"] for c in chunks)
    with get_conn(register_vectors=True) as conn:
        before = conn.execute("SELECT source, count(*), count(embedding) FROM chunks GROUP BY source ORDER BY source").fetchall()
        (OUT / "database_before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
        with conn.cursor() as cur:
            cur.executemany(UPSERT_CHUNK, [[c.get(col) for col in COLS] for c in chunks])
            cur.executemany(UPSERT_TABLE, [[t["chunk_id"], t["caption"], t["kind"], t["citation"], t["chapter_no"], t["printed_page"], t["pdf_page"], Jsonb(t["rows"])] for t in tables])
            cur.executemany("UPDATE chunks SET embedding = %s, embedding_model = %s WHERE id = %s AND content_hash = %s", [(np.asarray(v["embedding"], dtype=np.float32), v["embedding_model"], v["id"], v["content_hash"]) for v in vectors.values()])
        after = conn.execute("SELECT source, count(*), count(embedding) FROM chunks WHERE tax_year = 2025 GROUP BY source ORDER BY source").fetchall()
        for source, count, embedded in after:
            if source in expected and (count != expected[source] or embedded != count):
                raise ValueError(f"Import verification failed for {source}")
        if len(after) < 21:
            raise ValueError("Fewer than 21 documents present")
    (OUT / "database_after.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
    print(json.dumps(after, indent=2), flush=True)


if __name__ == "__main__":
    main()
