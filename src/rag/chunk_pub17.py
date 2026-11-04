"""
chunk_pub17.py: cut an IRS publication PDF into chunks for RAG (built on Pub 17, 2025).

Why the PDF needs special handling
  Pub 17 is printed in 2-4 columns, with headings marked only by font size,
  TIP / CAUTION boxes in the margin, words hyphenated across lines, and bordered
  tables. A plain text extraction mixes all of this together. So we read the
  page word by word (with font size and position) and rebuild the structure:

  step  function            what it does
  1     read_page_words     words + font info, proof header and footer removed
  -     tables.py           bordered tables taken out of the page (see tables.py)
  2     split_columns       find the columns, read them left to right
  3     group_lines         words -> lines, each line tagged (chapter / section / body / tip ...)
  4     build_units         cut the text at every heading and bold run-in heading
  5     make_chunks         merge small pieces, split big ones -> 150-400 word chunks

Every chunk keeps its heading path (Chapter > Section > Subsection > run-in),
its type (rule / tip / caution / whats_new / table) and a citation such as
  IRS Pub 17 (2025), Chapter 1 Filing Information, 'Limit on amount of refund', p. 19

Usage (from the tax-agent folder)
  python src/rag/chunk_pub17.py                       # default PDF and page range
  python src/rag/chunk_pub17.py --pdf <file> --pages 3-112

Output
  data/processed/pub17_<tax_year>_chunks.json   text + table chunks, for RAG
  data/processed/pub17_<tax_year>_tables.json   raw table rows, for code lookups
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pdfplumber

try:                        # run as a package: python -m src.rag.chunk_pub17
    from . import tables as tbl
except ImportError:         # run as a script:  python src/rag/chunk_pub17.py
    import tables as tbl

ROOT = Path(__file__).resolve().parents[2]  # .../tax-agent

# ============================================================================
# Settings
# ============================================================================
FOOTER_TOP = 745          # words below this line are the page footer
MIN_WORDS = 150           # merge units smaller than this with their neighbours
MAX_WORDS = 400           # split units larger than this at sentence ends
TABLE_NUMERIC_RATIO = 0.35  # a text unit with more numbers than this is unbordered table debris
SKIP_PAGE_NUMERIC_RATIO = 0.5  # whole page skipped (Tax Table, worksheets)


# ============================================================================
# Font rules
# ============================================================================
# Font sizes measured on Pub 17 (2025). Other IRS pubs use the same layout system.
#   17.5 Blk = Part, 18 Blk = Chapter, 15 Bold = section, 13 Bold = subsection,
#   10 Bold = minor heading, 8 Regular = body.
#   TIP / CAUTION boxes = 8 Italic text next to a "TIP" or "!" icon in the margin.
def classify_size(size, font):
    """Tag one word by its font: big (part/chapter), section, subsection, minor,
    part_intro or body. Headings in Pub 17 have no numbers or markup, so font size
    is the only reliable signal of the heading level."""
    bold = "Bold" in font or "Blk" in font or "Black" in font
    italic = "Italic" in font
    if size >= 17 and "Blk" in font:
        return "big"            # part or chapter title, decided by the text later
    if 14.5 <= size < 16.5 and bold:
        return "section"
    if 12.5 <= size < 14.5 and bold:
        return "subsection"
    if 9.8 <= size < 11.5 and bold:
        return "minor"
    if 10.2 <= size < 11.5 and italic:
        return "part_intro"     # italic overview under a Part title -> dropped
    return "body"


# ============================================================================
# 1. Read a page
# ============================================================================
def read_page_words(page):
    """All words on the page, minus proof header (top < 0) and footer.
    Returns (words, icons, footer_text)."""
    words = page.extract_words(extra_attrs=["size", "fontname"], keep_blank_chars=False)
    body, icons, footer = [], [], []
    for w in words:
        if w["top"] < 0:                 # proof header printed above the page
            continue
        if w["top"] >= FOOTER_TOP:
            footer.append(w)
            continue
        font = w["fontname"]
        # TIP / CAUTION icons use plain Helvetica (not HelveticaWorld) in the margin
        if "HelveticaWorld" not in font and "HelveticaLTStd" not in font:
            if w["text"] in ("!", "TIP", "CAUTION"):
                icons.append(w)
            continue
        body.append(w)
    footer_text = " ".join(w["text"] for w in sorted(footer, key=lambda w: (round(w["top"]), w["x0"])))
    return body, icons, footer_text


def parse_footer(footer_text):
    """'Publication 17 (2025) Chapter 1 Filing Information 19' -> printed page 19."""
    nums = re.findall(r"(?:^|\s)(\d{1,3})(?=\s|$)", footer_text)
    # the printed page number is the first or last number that is not the year/chapter
    tokens = footer_text.split()
    for t in (tokens[:1] + tokens[-1:]):
        if t.isdigit():
            return int(t)
    return int(nums[-1]) if nums else None


def numeric_ratio(words):
    """Share of words that are numbers. Used to skip the Tax Table pages
    (numbers belong to code, not to the LLM)."""
    if not words:
        return 0.0
    n = sum(1 for w in words if re.fullmatch(r"[\d,.$%()\-–]+", w["text"]))
    return n / len(words)


# ============================================================================
# 2. Find the columns
# ============================================================================
def find_lefts(words):
    """Body text is left-aligned, so every line of a column starts at the same x.
    x positions where many words start = candidate column left edges.
    Indented paragraphs / TIP text start a bit further right, so inside each
    100pt-wide cluster we keep the leftmost x."""
    starts = Counter(round(w["x0"]) for w in words if 7.5 <= w["size"] <= 8.5)
    if not starts:
        return []
    top = starts.most_common(1)[0][1]
    cands = sorted(x for x, n in starts.items() if n >= 0.25 * top)
    lefts = []
    for x in cands:
        if not lefts or x - lefts[-1] >= 100:     # columns are >100pt wide
            lefts.append(x)
    return lefts


def crossings(words, cuts):
    """How many body words would be cut in half by these column boundaries."""
    return sum(1 for w in words if 7.5 <= w["size"] <= 8.5
               for c in cuts if w["x0"] < c < w["x1"])


def split_columns(words, grid=None):
    """Return the page's words as columns (left column first).
    Two guesses for where columns start: this page's own left edges, and the
    document-wide grid. Keep the one that cuts through fewer words."""
    options = [find_lefts(words)]
    if grid:
        options.append(grid)
    best = min(options, key=lambda lefts: (crossings(words, [l - 4 for l in lefts[1:]]), lefts is not grid))
    if not best:
        return [words]
    cuts = [l - 4 for l in best[1:]]
    cols = [[] for _ in range(len(cuts) + 1)]
    for w in words:
        i = sum(1 for c in cuts if w["x0"] >= c)
        cols[i].append(w)
    return cols


# ============================================================================
# 3. Words -> lines
# ============================================================================
def group_lines(col_words):
    """Words of one column -> lines (top-to-bottom), each with a kind."""
    col_words = sorted(col_words, key=lambda w: (w["top"], w["x0"]))
    lines = []
    for w in col_words:
        if lines and abs(w["top"] - lines[-1]["top"]) <= 2.5:
            lines[-1]["words"].append(w)
        else:
            lines.append({"top": w["top"], "words": [w]})
    out = []
    col_left = min((w["x0"] for w in col_words), default=0)
    for ln in lines:
        ws = sorted(ln["words"], key=lambda w: w["x0"])
        real = [w for w in ws if w["text"] != "•"] or ws
        kinds = Counter(classify_size(w["size"], w["fontname"]) for w in real)
        kind = kinds.most_common(1)[0][0]
        text = " ".join(w["text"] for w in ws)
        runin = None
        if kind == "body":
            # bold (not bold-italic) words at the start of a body line = run-in heading
            lead = []
            for w in real:
                f = w["fontname"]
                if "Bold" in f and "Italic" not in f and w["size"] < 9:
                    lead.append(w["text"])
                else:
                    break
            if lead:
                runin = " ".join(lead).rstrip(".:")
        out.append({
            "kind": kind, "text": text, "runin": runin, "top": ln["top"],
            "bottom": max(w["bottom"] for w in ws),
            "size": max(w["size"] for w in real),
            "indent": ws[0]["x0"] - col_left > 8 and ws[0]["text"] != "•",
            "bullet": ws[0]["text"] == "•",
            "all_bold": runin is not None and len(runin.split()) >= len(real) - 1,
            "italic": sum("Italic" in w["fontname"] for w in real) > len(real) / 2,
        })
    # merge multi-line headings (e.g. "1." / "Filing" / "Information")
    merged = []
    for ln in out:
        prev = merged[-1] if merged else None
        if (prev and ln["kind"] == prev["kind"] and ln["kind"] in ("big", "section", "subsection", "minor")
                and ln["top"] - prev["bottom"] < 2.2 * ln["size"]):
            prev["text"] += " " + ln["text"]
            prev["bottom"] = ln["bottom"]
        else:
            merged.append(ln)
    return merged


# ============================================================================
# 4. Lines -> units (cut at headings)
# ============================================================================
def join_text(a, b, new_para=False):
    """Append line b to text a, fixing hyphenation across line breaks."""
    if not a:
        return b
    if new_para:
        return a + "\n" + b
    if re.search(r"[A-Za-z]-$", a) and re.match(r"[a-z]", b):
        return a[:-1] + b               # "af-" + "fecting" -> "affecting"
    return a + " " + b


def place_tables(cols, page_tables):
    """[tables] Decide which column each table belongs to, by its left edge.
    Returns {column index: [tables]}; key None = page has no text columns."""
    starts = [min(w["x0"] for w in c) if c else None for c in cols]
    events = {}
    for t in page_tables:
        idx = None
        for i, x in enumerate(starts):
            if x is not None and x <= t["bbox"][0] + 8:
                idx = i
        if idx is None and any(x is not None for x in starts):
            idx = next(i for i, x in enumerate(starts) if x is not None)
        events.setdefault(idx, []).append(t)
    return events


def build_units(pdf, pages, tax_year):
    """Walk the pages in reading order and cut the text into units.

    A new unit starts at every heading, every bold run-in heading, every TIP /
    CAUTION box and every table. `state` remembers the current heading path, so
    each unit knows which chapter / section / subsection it belongs to.

    Two passes:
      pass 1  read each page once, take the tables out, learn the column grid
              of the whole document (more reliable than guessing page by page)
      pass 2  walk columns and lines, update `state`, fill the units
    """
    state = {"part": None, "chapter": None, "chapter_no": None,
             "section": None, "subsection": None, "minor": None}
    units, cur = [], None
    skipped = []

    def start(kind_type, runin=None):
        """Close the current unit (if it has text) and open a new one."""
        nonlocal cur
        if cur and cur["text"].strip():
            units.append(cur)
        cur = {**state, "runin": runin, "type": kind_type, "text": "",
               "pdf_pages": [], "printed_pages": []}

    def touch(pdf_page, printed):
        """Record that the current unit has text on this page (for citations)."""
        if pdf_page not in cur["pdf_pages"]:
            cur["pdf_pages"].append(pdf_page)
            if printed is not None:
                cur["printed_pages"].append(printed)

    def emit_table(t, pdf_page, printed):
        """[tables] Close the current text unit, add the table as its own unit,
        then continue with a fresh text unit under the same headings."""
        start("rule")
        # a table without its own caption is named after the heading it sits under
        caption = t["caption"] or state["minor"] or state["subsection"] or state["section"] or "Table"
        units.append({**state, "runin": None, "type": "table", "text": t["text"],
                      "table_kind": t["kind"], "caption": caption, "rows": t["rows"],
                      "pdf_pages": [pdf_page], "printed_pages": [printed] if printed else []})

    # pass 1: read every page once, take the tables out, learn the column grid
    cache = {}
    for pno in pages:
        page = pdf.pages[pno - 1]
        words, icons, footer = read_page_words(page)
        if numeric_ratio(words) > SKIP_PAGE_NUMERIC_RATIO:
            skipped.append(pno)
            continue
        # [tables] find bordered tables, remove their words from the running text
        found, drop = tbl.find_page_tables(page, words)
        words = [w for w in words if id(w) not in drop]
        page_tables = [r for r in (tbl.process_table(t, pno) for t in found) if r]
        cache[pno] = (words, icons, footer, page_tables)
    all_body = [w for ws, _, _, _ in cache.values() for w in ws]
    grid = find_lefts(all_body)

    # pass 2: walk pages in reading order
    start("rule")
    pending_runin = False
    for pno, (words, icons, footer, page_tables) in cache.items():
        printed = parse_footer(footer)
        used_icons = set()   # each TIP / CAUTION icon opens only one box
        cols = split_columns(words, grid)
        events = place_tables(cols, page_tables)
        for ci, col in enumerate(cols):
            col_icons = [i for i in icons if col and min(w["x0"] for w in col) - 40 <= i["x0"] <= max(w["x1"] for w in col)]
            col_tables = sorted(events.get(ci, []), key=lambda t: t["bbox"][1])
            for ln in group_lines(col):
                # [tables] a table that sits above this line comes first
                while col_tables and col_tables[0]["bbox"][1] < ln["top"]:
                    emit_table(col_tables.pop(0), pno, printed)
                    pending_runin = False
                k = ln["kind"]
                if k == "part_intro":
                    continue
                if k == "big":
                    m = re.match(r"(\d+)\.\s*(.+)", ln["text"])
                    if m:
                        state.update(chapter_no=int(m.group(1)), chapter=m.group(2).strip(),
                                     section=None, subsection=None, minor=None)
                    else:
                        state.update(part=ln["text"], chapter=None, chapter_no=None,
                                     section=None, subsection=None, minor=None)
                    start("rule")
                    continue
                if k == "section":
                    state.update(section=ln["text"], subsection=None, minor=None)
                    start("rule"); continue
                if k == "subsection":
                    state.update(subsection=ln["text"], minor=None)
                    start("rule"); continue
                if k == "minor":
                    state.update(minor=ln["text"])
                    start("rule"); continue
                # TIP / CAUTION box: italic text whose first line sits next to an icon
                if ln["italic"]:
                    near = [i for i in col_icons if 0 <= i["top"] - ln["top"] <= 15
                            and id(i) not in used_icons]
                    if near:
                        used_icons.update(id(i) for i in col_icons if abs(i["top"] - near[0]["top"]) < 20)
                        is_caution = any(i["text"] in ("!", "CAUTION") for i in near)
                        start("caution" if is_caution else "tip")
                    if cur["type"] in ("tip", "caution"):
                        touch(pno, printed)
                        cur["text"] = join_text(cur["text"], ln["text"])
                        continue
                # body line
                if cur["type"] in ("tip", "caution"):
                    start("rule")
                if ln["runin"]:
                    if pending_runin:      # run-in heading wrapped onto a 2nd line
                        cur["runin"] = join_text(cur["runin"], ln["runin"])
                    else:
                        start("rule", runin=ln["runin"])
                pending_runin = ln["all_bold"]
                touch(pno, printed)
                cur["text"] = join_text(cur["text"], ln["text"],
                                        new_para=(ln["indent"] or ln["bullet"]) and bool(cur["text"]))
            for t in col_tables:            # tables below the last line of the column
                emit_table(t, pno, printed)
        for t in events.get(None, []):      # page with tables but no running text
            emit_table(t, pno, printed)
    if cur and cur["text"].strip():
        units.append(cur)

    for u in units:
        sec = (u["section"] or "").replace("’", "'")
        if u["type"] == "rule" and sec.startswith("What's New"):
            u["type"] = "whats_new"
        if u["type"] != "table" and numeric_ratio([{"text": t} for t in u["text"].split()]) > TABLE_NUMERIC_RATIO:
            u["type"] = "numeric"
    return units, skipped


# ============================================================================
# 5. Units -> chunks
# ============================================================================
def n_words(t):
    """Word count, used for all size limits."""
    return len(t.split())


def same_place(a, b):
    """True if two units sit under the same headings and have the same type,
    so merging them does not mix different topics."""
    keys = ("chapter", "section", "subsection", "minor")
    return all(a[k] == b[k] for k in keys) and a["type"] == b["type"]


def merge_units(a, b):
    """Join two units: text, run-in labels and page lists."""
    out = dict(a)
    out["text"] = a["text"] + "\n" + b["text"]
    out["runin"] = "; ".join(x for x in (a["runin"], b["runin"]) if x) or None
    out["pdf_pages"] = sorted(set(a["pdf_pages"] + b["pdf_pages"]))
    out["printed_pages"] = sorted(set(a["printed_pages"] + b["printed_pages"]))
    return out


def split_big(u):
    """Cut a unit longer than MAX_WORDS at sentence ends into ~300-word pieces.
    Long chunks dilute the embedding; cutting mid-sentence would break the rule."""
    if n_words(u["text"]) <= MAX_WORDS:
        return [u]
    sentences = re.split(r"(?<=[.;:])\s+(?=[A-Z(•])", u["text"])
    target = int(MAX_WORDS * 0.75)
    pieces, buf = [], ""
    for s in sentences:
        if buf and n_words(buf) + n_words(s) > target:
            pieces.append(buf)
            buf = s
        else:
            buf = (buf + " " + s).strip()
    if buf:
        if pieces and n_words(buf) < MIN_WORDS / 2:
            pieces[-1] += " " + buf
        else:
            pieces.append(buf)
    return [{**u, "text": p} for p in pieces]


def make_chunks(units, tax_year, source, prefix, keep_numeric=False):
    """Units -> final chunks.

    1. drop "numeric" units (unbordered table debris)
    2. merge small neighbours under the same headings (target 150-400 words);
       a short section intro joins its first subsection
    3. split anything over MAX_WORDS
    4. add id, citation and embed_text (= heading path + text; only this is embedded)
    Returns (chunks, table_records); table_records keep the raw rows for code."""
    units = [u for u in units if keep_numeric or u["type"] != "numeric"]
    # merge small neighbours that sit under the same headings
    merged = []
    for u in units:
        prev = merged[-1] if merged else None
        intro = (prev and prev["type"] == u["type"] == "rule" and n_words(prev["text"]) < 60
                 and prev["chapter"] == u["chapter"] and prev["section"] == u["section"])
        if intro:   # short section intro joins its first subsection
            merged[-1] = {**merge_units(prev, u), **{k: u[k] for k in ("subsection", "minor", "runin")}}
        elif (prev and same_place(prev, u) and prev["type"] in ("rule", "whats_new")
                and n_words(prev["text"]) < MIN_WORDS
                and n_words(prev["text"]) + n_words(u["text"]) <= MAX_WORDS):
            merged[-1] = merge_units(prev, u)
        else:
            merged.append(u)
    # a tiny leftover at the end of a section joins the chunk before it
    final = []
    for u in merged:
        prev = final[-1] if final else None
        if (prev and same_place(prev, u) and u["type"] in ("rule", "whats_new")
                and n_words(u["text"]) < 60 and n_words(prev["text"]) + n_words(u["text"]) <= MAX_WORDS):
            final[-1] = merge_units(prev, u)
        else:
            final.append(u)

    chunks, table_records, counter = [], [], Counter()
    for u in final:
        if n_words(u["text"]) < 15 and u["type"] not in ("tip", "caution", "table"):
            continue  # stray fragments
        for k, piece in enumerate(split_big(u)):
            ch = piece["chapter_no"] or 0
            counter[ch] += 1
            is_table = piece["type"] == "table"
            path = [f"Chapter {piece['chapter_no']} {piece['chapter']}" if piece["chapter"] else piece["part"],
                    piece["section"], piece["subsection"], piece["minor"], piece["runin"],
                    piece.get("caption") if is_table else None]
            path = [p for p in path if p]
            if is_table:
                label = piece["caption"]
            else:
                label = piece["runin"] or piece["minor"] or piece["subsection"] or piece["section"] or ""
            pages = piece["printed_pages"]
            page_str = (f"p. {pages[0]}" if len(pages) == 1 else f"pp. {pages[0]}-{pages[-1]}") if pages else ""
            citation = ", ".join(x for x in [
                source,
                f"Chapter {piece['chapter_no']} {piece['chapter']}" if piece["chapter"] else None,
                f"'{label}'" if label else None, page_str] if x)
            chunks.append({
                "id": f"{prefix}_{tax_year}_ch{ch:02d}_{counter[ch]:03d}",
                "tax_year": tax_year,
                "source": source,
                "part": piece["part"],
                "chapter_no": piece["chapter_no"],
                "chapter": piece["chapter"],
                "section": piece["section"],
                "subsection": piece["subsection"],
                "minor": piece["minor"],
                "runin": piece["runin"],
                "type": piece["type"],
                "table_kind": piece.get("table_kind"),
                "printed_pages": pages,
                "pdf_pages": piece["pdf_pages"],
                "citation": citation,
                "n_words": n_words(piece["text"]),
                "text": piece["text"],
                "embed_text": " > ".join(path) + "\n" + piece["text"],
            })
            if is_table and k == 0:      # [tables] raw rows for code, linked to the chunk
                table_records.append({
                    "chunk_id": chunks[-1]["id"], "caption": piece["caption"],
                    "kind": piece["table_kind"], "citation": citation,
                    "chapter_no": piece["chapter_no"], "printed_page": pages[0] if pages else None,
                    "pdf_page": piece["pdf_pages"][0], "rows": piece["rows"]})
    return chunks, table_records


# ============================================================================
# Command line
# ============================================================================
def parse_pages(spec, n):
    """'3-112' -> [3, 4, ..., 112], capped at the PDF length."""
    a, b = spec.split("-")
    return list(range(int(a), min(int(b), n) + 1))


def main():
    settings = json.loads((ROOT / "config" / "settings.json").read_text(encoding="utf-8"))
    tax_year = settings["tax_year"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", default=str(ROOT / "data/raw/irs/p17--2025.pdf"))
    ap.add_argument("--out", default=str(ROOT / f"data/processed/pub17_{tax_year}_chunks.json"))
    ap.add_argument("--pages", default="3-112",
                    help="PDF page range to read. Pub 17 2025: 3-112 (skips cover, TOC, Tax Table, index)")
    ap.add_argument("--source", default=f"IRS Pub 17 ({tax_year})")
    ap.add_argument("--prefix", default="pub17")
    ap.add_argument("--tables-out", default=str(ROOT / f"data/processed/pub17_{tax_year}_tables.json"))
    ap.add_argument("--keep-numeric", action="store_true",
                    help="keep text units that are mostly numbers (unbordered table debris)")
    args = ap.parse_args()

    with pdfplumber.open(args.pdf) as pdf:
        pages = parse_pages(args.pages, len(pdf.pages))
        units, skipped = build_units(pdf, pages, tax_year)
    chunks, table_records = make_chunks(units, tax_year, args.source, args.prefix, args.keep_numeric)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.tables_out).write_text(json.dumps(table_records, ensure_ascii=False, indent=2), encoding="utf-8")

    sizes = sorted(c["n_words"] for c in chunks)
    print(f"pages read: {len(pages)}, skipped as tables: {skipped}")
    print(f"units: {len(units)} -> chunks: {len(chunks)}  -> {args.out}")
    print(f"words per chunk: min {sizes[0]}, median {sizes[len(sizes)//2]}, max {sizes[-1]}")
    print("types:", dict(Counter(c["type"] for c in chunks)))
    print(f"tables: {len(table_records)} -> {args.tables_out}",
          dict(Counter(t["kind"] for t in table_records)))
    print("chapters:", sorted({(c['chapter_no'], c['chapter']) for c in chunks if c['chapter_no']}))


if __name__ == "__main__":
    main()
