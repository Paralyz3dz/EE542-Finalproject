def fetch_chunks(conn, tax_year):
    cur = conn.execute("SELECT id, citation, text, embed_text, type FROM public.chunks "
                       "WHERE tax_year = %s ORDER BY id", [tax_year])
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def vector_search(conn, query_vec, model, tax_year, k=5):
    return conn.execute("""
        SELECT id, citation, text, 1 - (embedding <=> %(q)s) AS score
        FROM public.chunks
        WHERE tax_year = %(y)s AND embedding_model = %(m)s
          AND embedding IS NOT NULL
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s""", {"q": query_vec, "y": tax_year, "m": model, "k": k}).fetchall()


def get_table(conn, caption_contains):
    return conn.execute("SELECT citation, rows FROM tax_tables WHERE caption ILIKE %s",
                        [f"%{caption_contains}%"]).fetchone()
