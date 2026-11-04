
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parents[2]

MIN_TABLE_WIDTH = 100    
CAPTION_SEARCH = 45       
FOOTER_TOP = 745          


# ============================================================================
# 1. Find tables on a page
# ============================================================================
def make_caption(heads):
    """Build the caption from the 10pt words above a table.
    Keep the label ("Table 1-1." / "Worksheet 3-1.") and the bold title after it;
    stop at the first regular word after the title, which is the start of an
    explanation such as "Use this table to ..." or "See chapter 3 ..."."""
    heads = sorted(heads, key=lambda w: (round(w["top"]), w["x0"]))
    label = [w for w in heads if re.fullmatch(r"(Table|Worksheet|[\dA-Z]+(-\d+)?\.)", w["text"])
             and "Bold" not in w["fontname"]][:2]
    title = []
    for w in heads:
        if w in label:
            continue
        if "Bold" in w["fontname"]:
            title.append(w["text"])
        elif title:
            break
    text = " ".join([w["text"] for w in label] + title)
    return re.split(r"\s(?:Caution|Note)\.", text)[0].strip()


def find_page_tables(page, words):

    tables, drop = [], set()
    for t in page.find_tables():
        x0, top, x1, bottom = t.bbox
        if x1 - x0 < MIN_TABLE_WIDTH:
            continue
        heads = [w for w in words
                 if top - CAPTION_SEARCH < w["top"] < top
                 and x0 - 5 <= w["x0"] and w["x1"] <= x1 + 5
                 and 9.8 <= w["size"] < 11.5 and "Italic" not in w["fontname"]]
        caption = make_caption(heads)
        cap_top = min((w["top"] for w in heads), default=top)
        region = (x0 - 5, cap_top - 1, x1 + 5, bottom + 1)
        for w in words:
            if (region[0] <= w["x0"] and w["x1"] <= region[2]
                    and region[1] <= w["top"] and w["bottom"] <= region[3]):
                drop.add(id(w))

        tables.append({"bbox": tuple(round(v, 1) for v in t.bbox),
                       "caption": caption, "raw_rows": t.extract()})
    return tables, drop


# ============================================================================
# 2. Clean the cells
# ============================================================================
def split_stacked_rows(raw_rows):
    out = []
    for row in raw_rows:
        cells = ["" if c is None else str(c) for c in row]
        multi = [c.split("\n") for c in cells if "\n" in c]
        counts = {len(m) for m in multi}
        numeric = all(re.fullmatch(r"[$\d,.\s]+", line) for m in multi for line in m)
        if multi and len(counts) == 1 and numeric:
            n = counts.pop()
            for k in range(n):
                out.append([c.split("\n")[k] if "\n" in c else (c if k == 0 else "") for c in cells])
        else:
            out.append(cells)
    return out


def clean_cell(cell):
    if cell is None:
        return ""
    s = re.sub(r"([a-z])-\n([a-z])", r"\1\2", cell)
    s = s.replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()


def clean_rows(raw_rows):
    rows = [[clean_cell(c) for c in r] for r in split_stacked_rows(raw_rows)]
    return [r for r in rows if any(r)]


def tidy(text):
    return re.sub(r"[.*\s]+$", "", text)


# ============================================================================
# 3. Decide what kind of table it is
# ============================================================================
STEP_RE = re.compile(r"\d+[a-z]?\.\s*(Enter|Add|Subtract|Multiply|Divide|Is|Compare)", re.I)


def is_footnote(row):
    return row[0].startswith("*") and not any(row[1:])


def classify(table):
    rows = table["rows"]
    if all(len(r) == 1 for r in rows):
        return "box"
    if "Worksheet" in table["caption"] or any(STEP_RE.match(" ".join(r)) for r in rows):
        return "worksheet"
    if any(r[0].startswith("IF") for r in rows):
        return "decision"
    return "table"


# ============================================================================
# 4. Turn each kind into text
# ============================================================================
def header_index(rows):
    
    for i, row in enumerate(rows):
        if row[0].startswith("IF"):
            return i
    return 0


def forward_fill(rows):
    out, last = [], [""] * len(rows[0]) if rows else []
    for row in rows:
        row = list(row)
        for i in range(len(row) - 1):
            if row[i] == "":
                row[i] = last[i]
            else:
                last[i] = row[i]
        out.append(row)
    return out


def linearize_decision(rows):
    h = header_index(rows)
    header = [tidy(c) for c in rows[h]]
    body = rows[h + 1:]
    notes = [r[0] for r in body if is_footnote(r)]
    data = forward_fill([r for r in body if not is_footnote(r)])

    sentences = []
    for row in data:
        parts = [f"{head} {tidy(value)}" for head, value in zip(header, row) if value]
        if parts:
            sentences.append(" ".join(parts) + ".")
    return "\n".join(sentences + notes)


def linearize_table(rows):
    header = rows[0]
    lines = []
    for row in rows[1:]:
        pairs = [f"{h}: {v}" if h else v for h, v in zip(header, row) if v]
        lines.append(" | ".join(pairs))
    return "\n".join(lines)


def linearize_box(rows):
    return "\n".join(r[0] for r in rows)


def worksheet_pointer(table, max_words=12):
    steps = []
    for row in table["rows"]:
        text = " ".join(c for c in row if c)
        text = re.sub(r"\.{3,}|(\.\s){3,}", " ", text)          # dot leaders "......"
        text = re.sub(r"\s+\d+[a-z]?\.\s*$", "", text).strip()  # line number at the end
        m = re.match(r"(\d+[a-z]?\.)\s*(.*)", text)
        if m and m.group(2):                                     # skip steps with no text
            steps.append(" ".join(text.split()[:max_words]))
    name = table["caption"] or "Worksheet"
    return f"{name}. Steps: " + "; ".join(steps)


def table_to_text(table):
    kind = classify(table)
    if kind == "decision":
        return kind, linearize_decision(table["rows"])
    if kind == "box":
        return kind, linearize_box(table["rows"])
    if kind == "worksheet":
        return kind, worksheet_pointer(table)
    return kind, linearize_table(table["rows"])


# ============================================================================
# 5. Whole document
# ============================================================================
def page_words(page):
    return [w for w in page.extract_words(extra_attrs=["size", "fontname"])
            if 0 <= w["top"] < FOOTER_TOP]


def process_table(t, pdf_page):
    t = dict(t)
    t["rows"] = clean_rows(t.pop("raw_rows"))
    if not t["rows"]:
        return None
    t["kind"], t["text"] = table_to_text(t)
    t["pdf_page"] = pdf_page
    return t


def extract_tables(pdf_path, pages):
    records = []
    with pdfplumber.open(pdf_path) as pdf:
        for pno in pages:
            page = pdf.pages[pno - 1]
            found, _ = find_page_tables(page, page_words(page))
            for t in found:
                rec = process_table(t, pno)
                if rec:
                    records.append(rec)
    return records


# ============================================================================
# Command line
# ============================================================================
def debug_pages(pdf_path, pages):
    for rec in extract_tables(pdf_path, pages):
        print(f"\n=== p{rec['pdf_page']}  [{rec['kind']}]  {rec['caption']!r}")
        print(rec["text"][:700])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")   
    except Exception:
        pass
    settings = json.loads((ROOT / "config" / "settings.json").read_text(encoding="utf-8"))
    tax_year = settings["tax_year"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", default=str(ROOT / "data/raw/irs/p17--2025.pdf"))
    ap.add_argument("--out", default=str(ROOT / f"data/processed/pub17_{tax_year}_tables.json"))
    ap.add_argument("--range", default="3-112", help="PDF pages to scan (same range as chunk_pub17.py)")
    ap.add_argument("--pages", type=int, nargs="+", help="only print these pages, write nothing")
    args = ap.parse_args()

    if args.pages:
        return debug_pages(args.pdf, args.pages)

    a, b = map(int, args.range.split("-"))
    records = extract_tables(args.pdf, range(a, b + 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(records)} tables -> {args.out}")
    print("kinds:", dict(Counter(r["kind"] for r in records)))
    for r in records:
        print(f"  p{r['pdf_page']:<4} {r['kind']:<10} {r['caption'][:70] or '(no caption)'}")


if __name__ == "__main__":
    main()
