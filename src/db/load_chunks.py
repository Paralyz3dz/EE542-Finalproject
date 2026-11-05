"""
load_chunks.py: sync Pub 17 chunks + table rows from data/processed/ into the database.
Safe to run again after re-chunking:
  - new chunks are inserted, changed chunks updated
  - a chunk whose text changed loses its embedding (it must be recomputed)
  - chunks that no longer exist are deleted

Usage (from the tax-agent folder)
  python src/rag/chunk_pub17.py      # first, to create the JSON files
  python -m src.db.load_chunks
"""
import hashlib
import json

from psycopg.types.json import Jsonb

from src.db.connection import ROOT, get_conn

TAX_YEAR = json.loads((ROOT / "config/settings.json").read_text(encoding="utf-8"))["tax_year"]
CHUNKS = ROOT / f"data/processed/pub17_{TAX_YEAR}_chunks.json"
TABLES = ROOT / f"data/processed/pub17_{TAX_YEAR}_tables.json"

COLS = ["id", "tax_year", "source", "chapter_no", "chapter", "section", "subsection", "minor",
        "runin", "type", "table_kind", "printed_pages", "pdf_pages", "citation", "text",
        "embed_text", "content_hash"]

UPSERT_CHUNK = f"""
INSERT INTO chunks ({", ".join(COLS)}) VALUES ({", ".join(["%s"] * len(COLS))})
ON CONFLICT (id) DO UPDATE SET
  {", ".join(f"{c} = EXCLUDED.{c}" for c in COLS[1:])},
  embedding = CASE WHEN chunks.content_hash = EXCLUDED.content_hash THEN chunks.embedding END,
  embedding_model = CASE WHEN chunks.content_hash = EXCLUDED.content_hash THEN chunks.embedding_model END,
  updated_at = now()
"""

UPSERT_TABLE = """
INSERT INTO tax_tables (chunk_id, caption, kind, citation, chapter_no, printed_page, pdf_page, rows)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET
  caption = EXCLUDED.caption, kind = EXCLUDED.kind, citation = EXCLUDED.citation,
  chapter_no = EXCLUDED.chapter_no, printed_page = EXCLUDED.printed_page,
  pdf_page = EXCLUDED.pdf_page, rows = EXCLUDED.rows
"""


def content_hash(text):
    """Short fingerprint of the embedded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main():
    if not CHUNKS.exists():
        raise SystemExit(f"{CHUNKS} not found: run python src/rag/chunk_pub17.py first")
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    tables = json.loads(TABLES.read_text(encoding="utf-8"))
    for c in chunks:
        c["content_hash"] = content_hash(c["embed_text"])
    source = chunks[0]["source"]

    with get_conn() as conn:                     # one transaction: all or nothing
        conn.execute((ROOT / "src/db/schema.sql").read_text(encoding="utf-8"))
        with conn.cursor() as cur:
            cur.executemany(UPSERT_CHUNK, [[c.get(col) for col in COLS] for c in chunks])
            cur.executemany(UPSERT_TABLE, [[t["chunk_id"], t["caption"], t["kind"], t["citation"],
                                            t["chapter_no"], t["printed_page"], t["pdf_page"],
                                            Jsonb(t["rows"])] for t in tables])
            cur.execute("DELETE FROM chunks WHERE source = %s AND tax_year = %s AND id <> ALL(%s)",
                        [source, TAX_YEAR, [c["id"] for c in chunks]])
            deleted = cur.rowcount
        n, n_emb = conn.execute(
            "SELECT count(*), count(embedding) FROM chunks WHERE tax_year = %s", [TAX_YEAR]).fetchone()
        n_tab = conn.execute("SELECT count(*) FROM tax_tables").fetchone()[0]

    print(f"chunks: {n} (with embedding: {n_emb})   tax_tables: {n_tab}   removed stale: {deleted}")


if __name__ == "__main__":
    main()
