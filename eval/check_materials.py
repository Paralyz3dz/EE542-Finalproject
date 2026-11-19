import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from src.db.connection import ROOT


PROTECTED = (
    ("IRS Pub 590-B (2025)", 70, "total amounts of contributions deducted in prior years"),
    ("IRS Pub 590-A (2025)", 59, "Enter your adjusted gross income"),
    ("IRS Pub 925 (2025)", 27, "TAS has offices in every state"),
    ("IRS Pub 525 (2025)", 39, "TAS has offices in every state"),
    ("IRS Pub 544 (2025)", 61, "Taxpayer Bill of Rights"),
    ("IRS Pub 334 (2025)", 32, "personal part is generally not deductible"),
    ("IRS Pub 505 (2025)", 19, "Who must pay estimated tax"),
    ("IRS Form 8843 (2025)", 3, "you must file Form 8843 to explain the basis"),
    ("IRS Pub 519 (2025)", 5, "you meet the substantial presence test for calendar year 2025"),
    ("IRS Pub 519 (2025)", 91, "I was a resident of the United Kingdom"),
)


def check(chunks, tables, quality):
    by_id = {c["id"]: c for c in chunks}
    expected = {d["name"] + " (2025)" for d in json.loads((ROOT / "config/knowledge_sources.json").read_text(encoding="utf-8"))}
    if len(by_id) != len(chunks) or {c["source"] for c in chunks} != expected:
        raise ValueError("Invalid source/chunk inventory")
    for chunk in chunks:
        if not (0 < chunk["n_words"] == len(chunk["text"].split()) <= 400):
            raise ValueError("Invalid word budget: " + chunk["id"])
        if chunk["content_hash"] != hashlib.sha256(chunk["embed_text"].encode()).hexdigest()[:16]:
            raise ValueError("Stale content fingerprint: " + chunk["id"])
        if re.search(r"[A-Za-z]-\s*\n\s*[a-z]", chunk["text"]):
            raise ValueError("A split word remains: " + chunk["id"])
        if "To help us develop a more useful index" in chunk["text"] or "You may want to see:" in chunk["text"]:
            raise ValueError("Reviewed reference noise remains: " + chunk["id"])
    for table in tables:
        if table["chunk_id"] not in by_id or by_id[table["chunk_id"]]["type"] != "table" or not table["rows"]:
            raise ValueError("Invalid exact table link")
    excluded_pages = {
        "FTB 3514 Instructions (2025)": set(range(13, 21)),
        "FTB 540 Booklet (2025)": set(range(39, 45)) | set(range(47, 53)),
        "IRS Form 8843 (2025)": {1, 2},
        "IRS Pub 519 (2025)": set(range(92, 96)),
    }
    for source, forbidden in excluded_pages.items():
        if any(set(p["pdf_pages"]) & forbidden for c in chunks for p in c["provenance"] if p["source"] == source):
            raise ValueError("An excluded page was included: " + source)
    protected = []
    for source, page, phrase in PROTECTED:
        records = [c for c in chunks if any(p["source"] == source and page in p["pdf_pages"] for p in c["provenance"])]
        matching = [c["id"] for c in records if phrase.casefold() in c["text"].casefold()]
        if not matching:
            matching = [t["chunk_id"] for t in tables if t["pdf_page"] == page and any(p["source"] == source for p in by_id[t["chunk_id"]]["provenance"]) and phrase.casefold() in " ".join(" ".join(row) for row in t["rows"]).casefold()]
        if not matching:
            raise ValueError(f"Protected body/worksheet was lost: {source} p. {page}")
        protected.append({"source": source, "pdf_page": page, "phrase": phrase, "chunk_ids": matching})
    return {"chunks": len(chunks), "eligible_chunks": sum(c["retrieval_eligible"] for c in chunks),
            "numeric_or_oversized_locators_excluded": sum(not c["retrieval_eligible"] for c in chunks), "sources": len(expected), "tables": len(tables),
            "all_chunks_within_400_words": True, "word_boundary_splits": 0, "known_index_prompts": 0,
            "useful_items_catalog_prompts": 0, "form_pages_included": 0, "protected_body_and_worksheet_checks": protected,
            "exact_duplicate_groups": len(quality["canonical_duplicates"]), "short_sections": sum(c["n_words"] < 150 for c in chunks),
            "short_sections_under_40": sum(c["n_words"] < 40 for c in chunks),
            "short_sections_with_bold_runin": sum(c["n_words"] < 150 and bool(c["runin"]) for c in chunks),
            "shortest_non_table_rules": [{"id": c["id"], "citation": c["citation"], "text": c["text"]} for c in chunks if c["n_words"] < 5 and c["type"] != "table"],
            "table_extraction_methods": dict(Counter(t["metadata"]["extraction"] for t in tables))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/materials-v2")
    parser.add_argument("--out", type=Path, default=ROOT / "eval/results/materials-v2-2026-11-17")
    args = parser.parse_args()
    chunks, tables, quality = [json.loads((args.input / name).read_text(encoding="utf-8")) for name in ("chunks.json", "tables.json", "quality.json")]
    audit = check(chunks, tables, quality)
    audit["files_sha256"] = {name: hashlib.sha256((args.input / name).read_bytes()).hexdigest() for name in ("chunks.json", "tables.json", "quality.json")}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "processing_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (args.out / "quality.json").write_bytes((args.input / "quality.json").read_bytes())
    print(json.dumps({k: audit[k] for k in ("chunks", "eligible_chunks", "sources", "tables", "word_boundary_splits")}), flush=True)


if __name__ == "__main__":
    main()
