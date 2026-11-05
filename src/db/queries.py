"""
queries.py: read helpers shared by retrieval and tax-rule code.

Example
  from src.db.connection import get_conn
  from src.db.queries import get_table
  with get_conn() as conn:
      citation, rows = get_table(conn, "Table 1-1")
"""


def fetch_chunks(conn, tax_year):
    """All chunks for BM25 or evaluation, same fields as the JSON file."""
    cur = conn.execute("SELECT id, citation, text, embed_text, type FROM chunks "
                       "WHERE tax_year = %s ORDER BY id", [tax_year])
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def vector_search(conn, query_vec, model, tax_year, k=5):
    """Top-k chunks by cosine similarity. query_vec must come from the SAME model
    that produced the stored embeddings (open conn with register_vectors=True)."""
    return conn.execute("""
        SELECT id, citation, text, 1 - (embedding <=> %(q)s) AS score
        FROM chunks
        WHERE tax_year = %(y)s AND embedding_model = %(m)s
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s""", {"q": query_vec, "y": tax_year, "m": model, "k": k}).fetchall()


def get_table(conn, caption_contains):
    """Raw rows of an IRS table, e.g. get_table(conn, 'Table 1-1') -> (citation, rows)."""
    return conn.execute("SELECT citation, rows FROM tax_tables WHERE caption ILIKE %s",
                        [f"%{caption_contains}%"]).fetchone()
