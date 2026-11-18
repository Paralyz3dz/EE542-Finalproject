import re


def corpus_table(corpus="original", table="chunks"):
    if not isinstance(corpus, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", corpus):
        raise ValueError("Invalid corpus version")
    if table not in {"chunks", "tax_tables"}:
        raise ValueError("Invalid corpus table")
    return f"public.{table}" if corpus == "original" else f"public.{corpus}_{table}"


def fetch_chunks(conn, tax_year, *, corpus="original"):
    table = corpus_table(corpus)
    extra = ", provenance" if corpus != "original" else ""
    eligibility = " AND retrieval_eligible" if corpus != "original" else ""
    cur = conn.execute(f"SELECT id, citation, text, embed_text, type, content_hash{extra} FROM {table} "
                       f"WHERE tax_year = %s{eligibility} ORDER BY id", [tax_year])
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def vector_search(conn, query_vec, model, tax_year, k=5, *, excluded_ids=None, corpus="original"):
    table = corpus_table(corpus)
    exclusion = "AND id <> ALL(%(excluded_ids)s)" if excluded_ids else ""
    eligibility = "AND retrieval_eligible" if corpus != "original" else ""
    parameters = {"q": query_vec, "y": tax_year, "m": model, "k": k}
    if excluded_ids:
        parameters["excluded_ids"] = excluded_ids
    return conn.execute(f"""
        SELECT id, citation, text, 1 - (embedding <=> %(q)s) AS score
        FROM {table}
        WHERE tax_year = %(y)s AND embedding_model = %(m)s
          AND embedding IS NOT NULL
          {exclusion}
          {eligibility}
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s""", parameters).fetchall()


def get_table(conn, caption_contains, *, corpus="original"):
    table = corpus_table(corpus, "tax_tables")
    return conn.execute(f"SELECT citation, rows FROM {table} WHERE caption ILIKE %s",
                        [f"%{caption_contains}%"]).fetchone()
