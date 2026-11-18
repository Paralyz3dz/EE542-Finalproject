import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from xml.sax.saxutils import escape

import pdfplumber
from bs4 import BeautifulSoup
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from src.rag import chunk_pub17 as cutter
from src.rag.corpus_hygiene import filter_chunks

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/knowledge_base"


def html_pdf(path):
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    main = soup.find("main") or soup.find("article")
    if main is None:
        raise ValueError(f"No main content in {path}")
    for tag in main.select("nav, script, style, .breadcrumbs, .feedback"):
        tag.decompose()
    styles = getSampleStyleSheet()
    story = []
    for tag in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        if tag.find_parent(["li", "table"]):
            continue
        text = tag.get_text(" ", strip=True)
        if not text:
            continue
        style = styles["Heading2" if tag.name.startswith("h") else "BodyText"]
        if tag.name == "table":
            rows = [[Paragraph(escape(c.get_text(" ", strip=True)), styles["BodyText"]) for c in row.find_all(["td", "th"])] for row in tag.find_all("tr")]
            rows = [row for row in rows if row]
            width = max(map(len, rows))
            rows = [row + [""] * (width - len(row)) for row in rows]
            table = Table(rows, colWidths=[460 / width] * width)
            table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "black")]))
            story.extend([table, Spacer(1, 5)])
            continue
        else:
            text = escape(text)
        story.extend([Paragraph(text, style), Spacer(1, 5)])
    output = path.with_suffix(".pdf")
    SimpleDocTemplate(str(output)).build(story)
    return output


def ranges(name, count):
    if name == "FTB 3514 Instructions":
        return list(range(3, 11)) + list(range(22, 31))
    if name == "FTB 540 Booklet":
        return list(range(3, 24)) + [25, 26, 27, 29, 30, 31, 32, 33, 35, 36] + list(range(55, 74)) + [75]
    if name == "FTB Pub 1031":
        return list(range(3, count + 1))
    return list(range(1, count + 1))


def process(document):
    name = document["name"]
    if name == "IRS Pub 17":
        return None
    prefix = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    destination = OUT / f"{prefix}_chunks.json"
    if destination.exists():
        return str(destination)
    path = ROOT / document["path"]
    web = document["url"].endswith(".html")
    if web:
        path = html_pdf(path.with_suffix(".html"))
    source = f"{name} (2025)"
    with pdfplumber.open(path) as pdf:
        if "2025" not in (pdf.pages[0].extract_text() or ""):
            raise ValueError(f"Missing 2025 cover: {name}")
        pages = ranges(name, len(pdf.pages))
        sampled = [c for p in pages[:6] for c in pdf.pages[p - 1].chars if c.get("text", "").isalpha() and not any(k in c.get("fontname", "") for k in ("Bold", "Blk", "Black", "Italic"))]
        body_size = Counter(round(c["size"], 1) for c in sampled).most_common(1)[0][0]
        numeric_units = []

        def read_page(page):
            words = page.extract_words(extra_attrs=["size", "fontname"])
            footer = [w for w in words if w["top"] >= page.height - 47]
            footer_text = " ".join(w["text"] for w in footer)
            page_label = re.search(r"\bPage\s+(\d+)\b", footer_text, re.IGNORECASE)
            if page_label:
                footer_text = page_label.group(1)
            if web:
                footer_text = str(page.page_number)
            words = [w for w in words if 20 <= w["top"] < page.height - 47]
            if not words:
                return [], [], footer_text
            raw_text = " ".join(w["text"] for w in words)
            if "THIS PAGE INTENTIONALLY LEFT BLANK" in raw_text or "Table of Contents" in raw_text:
                return [], [], footer_text
            if cutter.numeric_ratio(words) > 0.5:
                rows = [re.split(r"\s{2,}", line.strip()) for line in (page.extract_text(layout=True) or "").splitlines() if line.strip()]
                numeric_units.append({"part": None, "chapter": None, "chapter_no": None, "section": "Tables", "subsection": None, "minor": None, "runin": None, "type": "table", "table_kind": "table", "caption": raw_text[:100], "rows": rows, "text": "\n".join(" | ".join(row) for row in rows), "pdf_pages": [page.page_number], "printed_pages": [cutter.parse_footer(footer_text) or page.page_number]})
                return [], [], footer_text
            icons = []
            for w in words:
                if w["text"] in ("TIP", "!", "CAUTION"):
                    icons.append(dict(w))
                size = w["size"]
                bold = any(k in w["fontname"] for k in ("Bold", "Blk", "Black", "-Bd"))
                if bold and size >= body_size + 5:
                    w["size"] = 15
                elif bold and size >= body_size + 2:
                    w["size"] = 13
                elif bold and size >= body_size + 0.5:
                    w["size"] = 10
                else:
                    w["size"] = 8
            return words, icons, footer_text

        cutter.read_page_words = read_page
        cutter.SKIP_PAGE_NUMERIC_RATIO = 1.1
        units, skipped = cutter.build_units(pdf, pages, 2025)
        units += numeric_units
        if web:
            for unit in units:
                unit["printed_pages"] = []
        for unit in units:
            unit["text"] = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", unit["text"])
            if unit["type"] == "numeric":
                unit.update(type="table", table_kind="table", caption=unit.get("runin") or unit.get("section") or "Table", rows=[re.split(r"\s{2,}|\t", line.strip()) for line in unit["text"].splitlines() if line.strip()])
        chunks, tables = cutter.make_chunks(units, 2025, source, prefix, keep_numeric=True)
    bounded = []
    for chunk in chunks:
        if name.startswith("FTB") and not web:
            chunk["printed_pages"] = chunk["pdf_pages"][:]
            pages_label = f"p. {chunk['printed_pages'][0]}" if len(chunk["printed_pages"]) == 1 else f"pp. {chunk['printed_pages'][0]}-{chunk['printed_pages'][-1]}"
            chunk["citation"] = re.sub(r", (?:p\.|pp\.) [\d-]+$", "", chunk["citation"]) + ", " + pages_label
        words = chunk["text"].split()
        if not words:
            continue
        count = max(1, (len(words) + 399) // 400)
        width = (len(words) + count - 1) // count
        for i in range(count):
            text = " ".join(words[i * width:(i + 1) * width])
            item = dict(chunk, text=text, n_words=len(text.split()))
            if i:
                item["id"] += f"_part{i + 1}"
            heading = chunk["embed_text"].split("\n", 1)[0]
            item["embed_text"] = heading + "\n" + text
            if web:
                item["citation"] += f", web instruction PDF p. {item['pdf_pages'][0]}"
            elif not item["printed_pages"]:
                item["citation"] += f", PDF p. {item['pdf_pages'][0]}"
            bounded.append(item)
    if not bounded:
        raise ValueError(f"No content extracted: {name}")
    destination.write_text(json.dumps(bounded, ensure_ascii=False, indent=2), encoding="utf-8")
    ids = {chunk["id"] for chunk in bounded}
    tables = [table for table in tables if table["chunk_id"] in ids]
    by_id = {chunk["id"]: chunk for chunk in bounded}
    for table in tables:
        table["citation"] = by_id[table["chunk_id"]]["citation"]
        if name.startswith("FTB") and not web:
            table["printed_page"] = table["pdf_page"]
    destination.with_name(destination.name.replace("_chunks", "_tables")).write_text(json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{name}: {len(bounded)} chunks, {len(tables)} tables, pages {pages}", flush=True)
    return str(destination)


def export_retrieval_snapshot(chunks, tables, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    eligible, excluded, stale = filter_chunks(chunks)
    ids = {c["id"] for c in eligible}
    eligible_tables = [t for t in tables if t["chunk_id"] in ids]
    for filename, value in (("retrieval_chunks.json", eligible), ("retrieval_tables.json", eligible_tables),
                            ("hygiene_report.json", {"raw_chunks": len(chunks), "eligible_chunks": len(eligible),
                                                     "excluded_ids": excluded, "stale_ids_retained": stale})):
        (destination / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return eligible, eligible_tables


def main():
    documents = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    with ProcessPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(process, documents))
    chunks = []
    tables = []
    for filename in filter(None, files):
        path = Path(filename)
        chunks.extend(json.loads(path.read_text(encoding="utf-8")))
        tables.extend(json.loads(path.with_name(path.name.replace("_chunks", "_tables")).read_text(encoding="utf-8")))
    (OUT / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (OUT / "tables.json").write_text(json.dumps(tables, ensure_ascii=False), encoding="utf-8")
    export_retrieval_snapshot(chunks, tables, OUT)
    print(f"TOTAL {len(chunks)} chunks, {len(tables)} tables", flush=True)


if __name__ == "__main__":
    main()
