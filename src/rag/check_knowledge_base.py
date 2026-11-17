import json
from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi

from src.db.connection import ROOT, get_conn
from src.db.queries import fetch_chunks, vector_search
from src.rag.hybrid import fuse_rankings, tokenize
from src.rag.retriever import QUERY_INSTRUCTION, embed_texts
from src.workflow.engine import _doc_key, load_scenario, start_interview, step

OUT = ROOT / "data/processed/knowledge_base"


def main():
    _, questions = load_scenario("intake")
    query_file = OUT / "query_vectors.json"
    if query_file.exists():
        vectors = json.loads(query_file.read_text(encoding="utf-8"))
    else:
        vectors = []
        for i in range(0, len(questions), 8):
            vectors.extend(embed_texts([f"Instruct: {QUERY_INSTRUCTION}\nQuery: {q['retrieval_query']}" for q in questions[i:i + 8]]).tolist())
        query_file.write_text(json.dumps(vectors), encoding="utf-8")
    with get_conn(register_vectors=True) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        conn.execute("SET LOCAL enable_seqscan = off")
        inventory = conn.execute("SELECT source, count(*), count(embedding) FROM chunks WHERE tax_year = 2025 GROUP BY source ORDER BY source").fetchall()
        chunks = fetch_chunks(conn, 2025)
        index = BM25Okapi([tokenize(c["embed_text"]) for c in chunks])
        results = []
        cache = {}
        for q, vector in zip(questions, vectors):
            rows = vector_search(conn, np.asarray(vector, dtype=np.float32), "Qwen/Qwen3-Embedding-0.6B", 2025, k=5)
            semantic = [{"chunk_id": row[0], "citation": row[1], "text": row[2], "score": row[3]} for row in rows]
            scores = index.get_scores(tokenize(q["retrieval_query"]))
            order = sorted(range(len(chunks)), key=lambda i: (-scores[i], chunks[i]["id"]))[:5]
            keyword = [{"chunk_id": chunks[i]["id"], "citation": chunks[i]["citation"], "text": chunks[i]["text"], "score": float(scores[i])} for i in order if scores[i] > 0]
            matches = fuse_rankings(semantic, keyword, k=5)
            cache[q["retrieval_query"]] = {"matches": matches}
            expected = {_doc_key(s) for s in q["source"].split(";") if s.strip()}
            found = {_doc_key(m["citation"]) for m in matches[:3]}
            results.append({"id": q["id"], "source": q["source"], "hit_at_3": bool(expected & found), "citations": [m["citation"] for m in matches[:3]]})
    case = json.loads((ROOT / "tests/cases/baseline_w2_only.json").read_text(encoding="utf-8"))
    state = start_interview("intake", case.get("priority", []), case.get("background", ""))
    state, output = step(state, search=lambda query: cache[query])
    for _ in range(300):
        if output["type"] == "summary":
            break
        default = "yes" if output["fact"] in ("ca_resident_full_year", "is_us_citizen") else "no" if output["input_type"] == "bool" else output["options"][0] if output["input_type"] == "enum" else "0"
        state, output = step(state, case["answers"].get(output["fact"], default), search=lambda query: cache[query])
    local = json.loads((OUT / "chunks.json").read_text(encoding="utf-8"))
    allowed = {_doc_key(d["name"]) for d in json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))}
    documents = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    batches = {}
    for label, group in (("batch_1", documents[:6]), ("batch_2", documents[6:])):
        sources = {_doc_key(d["name"]) for d in group}
        eligible = [r for r in results if sources & {_doc_key(s) for s in r["source"].split(";")}]
        hits = sum(bool(sources & {_doc_key(s) for s in r["source"].split(";")} & {_doc_key(c) for c in r["citations"]}) for r in eligible)
        batches[label] = {"questions": len(eligible), "hits": hits, "hit_at_3": hits / len(eligible) if eligible else None}
    report = {
        "inventory": inventory,
        "documents": len(inventory),
        "chunks": sum(row[1] for row in inventory),
        "vectors": sum(row[2] for row in inventory),
        "citation_names_valid": all(_doc_key(c["citation"]) in allowed for c in local),
        "short_chunks_preserved_at_heading_boundaries": sum(c["n_words"] < 150 for c in local),
        "maximum_words": max(c["n_words"] for c in local),
        "types": dict(Counter(c["type"] for c in local)),
        "queries": len(results),
        "hit_at_3": sum(r["hit_at_3"] for r in results) / len(results),
        "batches": batches,
        "baseline_summary_references_not_in_kb": output.get("references_not_in_kb"),
        "results": results,
    }
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("results", "inventory")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
