import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup

from src.db.connection import ROOT
from src.rag.build_knowledge_base import ranges


VERSION = "materials-v2"
OUT = ROOT / "data/processed/materials-v2"
ALLOWED_TYPES = {"rule", "tip", "caution", "whats_new", "table"}


def fingerprint(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def dehyphenate(text):
    compounds = {"tax-free", "tax-exempt", "self-employed", "self-employment", "full-time", "part-time", "long-term", "short-term", "year-end", "after-tax", "before-tax", "employer-provided", "government-related"}
    return re.sub(r"([A-Za-z]+)-\s*\n\s*([a-z]+)", lambda m: m[1] + ("-" if (m[1] + "-" + m[2]).lower() in compounds else "") + m[2], text)


def clean(text):
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "").replace("\xa0", " ")
    text = dehyphenate(text)
    text = re.sub(r"(?:\s*\.\s*){3,}", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def line_text(line):
    result = ""
    previous = None
    for span in line["spans"]:
        text = span["text"]
        if previous and result and text and not result[-1].isspace() and not text[0].isspace():
            if span["bbox"][0] - previous["bbox"][2] > 0.7:
                result += " "
        result += text
        previous = span
    return result


def box_text(box):
    return clean("\n".join(line_text(line) for line in (box.get("textlines") or [])))


def sentences(text):
    boundary = r"(?<=[.!?])\s+(?=[A-Z\d\"\u201c(])(?!(?:\d+[.)]\s))"
    abbreviation = r"\b(?:Pub|Pubs|Mr|Mrs|Ms|Dr|Prof|Sr|Jr|U\.S|U\.K|D\.C|e\.g|i\.e|vs|Inc|Corp|Ltd|Sec|Reg|Rev|Fig|Art|Ch)\.$"
    pieces, start = [], 0
    for match in re.finditer(boundary, text):
        prefix = text[max(0, match.start() - 20):match.start()]
        if re.search(abbreviation, prefix, re.I) or re.search(r"\b[A-Z]\.$|\b\d{1,2}\.$", prefix):
            continue
        if re.search(r"\bNo\.$", prefix) and text[match.end():match.end() + 1].isdigit():
            continue
        pieces.append(text[start:match.start()])
        start = match.end()
    pieces.append(text[start:])
    return pieces


def split_passage(paragraphs, maximum=400):
    pieces = []
    for paragraph, pages in paragraphs:
        if len(paragraph.split()) <= maximum:
            pieces.append((paragraph, pages))
        else:
            for sentence in sentences(paragraph):
                if len(sentence.split()) <= maximum:
                    pieces.append((sentence, pages))
                else:
                    clauses = re.split(r"(?<=[;:])\s+", sentence)
                    if any(len(clause.split()) > maximum for clause in clauses):
                        raise ValueError("An indivisible passage exceeds the word budget: " + sentence[:100])
                    pieces.extend((clause, pages) for clause in clauses)
    groups = []
    text, page_set = [], set()
    for piece, pages in pieces:
        if text and sum(len(t.split()) for t in text) + len(piece.split()) > maximum:
            groups.append((dehyphenate("\n\n".join(text)), sorted(page_set)))
            text, page_set = [], set()
        text.append(piece)
        page_set.update(pages)
    if text:
        groups.append((dehyphenate("\n\n".join(text)), sorted(page_set)))
    if len(groups) > 1 and len(groups[-1][0].split()) < 150:
        before, tail = groups[-2:]
        units = sentences(before[0])
        while len(units) > 1 and len(tail[0].split()) < 150:
            candidate = units[-1] + "\n\n" + tail[0]
            if len(candidate.split()) > maximum or len(" ".join(units[:-1]).split()) < 150:
                break
            tail = (candidate, sorted(set(tail[1] + before[1])))
            units.pop()
        groups[-2:] = [(" ".join(units), before[1]), tail]
    return [(dehyphenate(text), pages) for text, pages in groups]


def valid_heading(text):
    return (bool(re.search(r"[A-Za-z]{2}", text)) and len(text.split()) <= 28
            and not re.fullmatch(r"[\d\W]+", text) and len(text) >= 3)


def runin_heading(box):
    spans = [s for line in (box.get("textlines") or []) for s in line["spans"] if s["text"].strip()]
    heading = []
    for span in spans:
        if re.search(r"Bold|Black|Blk|-Bd", span["font"], re.I):
            heading.append(span["text"])
        else:
            break
    title = clean("\n".join(heading))
    if title in {"Yes.", "No.", "Yes:", "No:", "STOP", "Continue"}:
        return None
    if valid_heading(title) and len(title.split()) <= 22 and re.search(r"[.:?]$", title):
        return title
    return None


def html_elements(path):
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    main = soup.find("main") or soup.find("article")
    if main is None:
        raise ValueError("Missing main content: " + str(path))
    for tag in main.select("nav,script,style,.breadcrumbs,.feedback"):
        tag.decompose()
    if "2025" not in main.get_text(" ", strip=True):
        raise ValueError("Missing 2025 source year: " + str(path))
    elements = []
    for tag in main.find_all(["h1", "h2", "h3", "h4", "h5", "p", "li", "table"]):
        if tag.find_parent(["table", "li"]):
            continue
        text = clean(tag.get_text(" ", strip=True))
        if not text:
            continue
        element = {"text": text, "page": None, "printed": None, "kind": "text", "level": 0}
        if tag.name.startswith("h"):
            element.update(kind="heading", level=int(tag.name[1]))
        elif tag.name == "table":
            rows = [[clean(c.get_text(" ", strip=True)) for c in row.find_all(["td", "th"])]
                    for row in tag.find_all("tr")]
            element.update(kind="table", rows=[row for row in rows if row])
        else:
            first = tag.find(["strong", "b"], recursive=False)
            if first and text.startswith(clean(first.get_text(" ", strip=True))):
                title = clean(first.get_text(" ", strip=True))
                if valid_heading(title) and len(title.split()) <= 18:
                    if title == text:
                        element.update(kind="heading", level=5)
                    else:
                        element["runin"] = title
        elements.append(element)
    return elements


def is_real_table(box, raw_page):
    table = box.get("table")
    if not table or table["row_count"] < 2 or table["col_count"] < 2:
        return False
    rows = table["extract"]
    text = clean(" ".join(str(cell or "") for row in rows for cell in row))
    numbers = len(re.findall(r"\b\d[\d,.]*\b", text))
    if numbers > len(text.split()) * 0.25:
        return True
    if re.search(r"\bIF\b.*\bTHEN\b|\bIf you\b.*\bThen\b", text):
        return True
    rect = pymupdf.Rect(box["x0"], box["y0"], box["x1"], box["y1"])
    above = raw_page.get_text("text", clip=pymupdf.Rect(max(0, rect.x0 - 10), max(0, rect.y0 - 45), min(raw_page.rect.width, rect.x1 + 10), rect.y0))
    if rect.height > 25 and re.search(r"\b(?:Table|Worksheet|Chart|Figure)\s+[\dA-Z]", above):
        return True
    vertical, horizontal = set(), set()
    for drawing in raw_page.get_drawings():
        for item in drawing["items"]:
            if item[0] != "l":
                continue
            start, end = item[1:3]
            if not rect.intersects(pymupdf.Rect(min(start.x, end.x) - 1, min(start.y, end.y) - 1, max(start.x, end.x) + 1, max(start.y, end.y) + 1)):
                continue
            if abs(start.x - end.x) < 1 and abs(start.y - end.y) > rect.height * 0.35:
                vertical.add(round(start.x))
            if abs(start.y - end.y) < 1 and abs(start.x - end.x) > rect.width * 0.35:
                horizontal.add(round(start.y))
    return len(vertical) >= 2 and len(horizontal) >= 2




def numeric_bands(page, box):
    rect = pymupdf.Rect(box["x0"], box["y0"], box["x1"], box["y1"])
    grouped = defaultdict(list)
    for word in page.get_text("words", clip=rect):
        grouped[round(word[1] / 2)].append(word)
    rows = []
    for words in grouped.values():
        run = []
        for word in sorted(words, key=lambda w: w[0]) + [None]:
            if word is not None and re.fullmatch(r"[\d,$.%()\u2212\u2013\u2014+-]+", word[4]) and re.search(r"\d", word[4]):
                run.append(word)
            else:
                if len(run) >= 4:
                    rows.append(run)
                run = []
    rows.sort(key=lambda row: (row[0][1], row[0][0]))
    bands = []
    for row in rows:
        bounds = [min(w[0] for w in row), min(w[1] for w in row), max(w[2] for w in row), max(w[3] for w in row)]
        if bands and abs(bands[-1]["x0"] - bounds[0]) < 20 and abs(bands[-1]["x1"] - bounds[2]) < 20 and 0 <= bounds[1] - bands[-1]["y1"] < 20:
            band = bands[-1]
            band["table"]["extract"].append([w[4] for w in row])
            band["metadata"]["row_cell_boxes"].append([list(w[:4]) for w in row])
            band["y1"] = bounds[3]
            band["x0"] = min(band["x0"], bounds[0])
            band["x1"] = max(band["x1"], bounds[2])
        else:
            bands.append({"x0": bounds[0], "y0": bounds[1], "x1": bounds[2], "y1": bounds[3],
                          "table": {"extract": [[w[4] for w in row]], "col_count": len(row)},
                          "metadata": {"extraction": "numeric_geometry", "row_cell_boxes": [[list(w[:4]) for w in row]]}})
    for band in bands:
        edges = []
        for value in sorted(cell[2] for row in band["metadata"]["row_cell_boxes"] for cell in row):
            if not edges or value - edges[-1] > 3:
                edges.append(value)
        aligned, positions = [], []
        for row, boxes in zip(band["table"]["extract"], band["metadata"]["row_cell_boxes"]):
            cells, bounds = [""] * len(edges), [None] * len(edges)
            for value, box in zip(row, boxes):
                column = min(range(len(edges)), key=lambda i: abs(edges[i] - box[2]))
                if cells[column]:
                    raise ValueError("Ambiguous numeric table column alignment")
                cells[column], bounds[column] = value, box
            aligned.append(cells)
            positions.append(bounds)
        band["table"]["extract"] = aligned
        band["table"]["col_count"] = len(edges)
        band["metadata"]["row_cell_boxes"] = positions
        band["metadata"]["column_right_edges"] = edges
        band["table"]["row_count"] = len(band["table"]["extract"])
        above = pymupdf.Rect(band["x0"], max(18, band["y0"] - 50), band["x1"], band["y0"])
        band["metadata"]["header_text"] = clean(page.get_text("text", clip=above, sort=True))
    return [band for band in bands if band["table"]["row_count"] >= 2]

def grid_tables(page):
    vertical, horizontal = set(), set()
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "re":
                rect = item[1]
                if rect.height > 35 and rect.width > 40:
                    vertical.update((round(rect.x0), round(rect.x1)))
                    horizontal.update((round(rect.y0), round(rect.y1)))
            elif item[0] == "l":
                start, end = item[1:3]
                if abs(start.x - end.x) < 1 and abs(start.y - end.y) > 35:
                    vertical.add(round(start.x))
                if abs(start.y - end.y) < 1 and abs(start.x - end.x) > 40:
                    horizontal.add(round(start.y))
    if len(vertical) < 2 or len(horizontal) < 3:
        return []
    result = []
    for table in page.find_tables(strategy="lines_strict").tables:
        if table.row_count < 2 or table.col_count < 2:
            continue
        rows = [[clean(page.get_text("text", clip=pymupdf.Rect(cell), sort=True)) if cell else "" for cell in row.cells] for row in table.rows]
        x0, y0, x1, y1 = table.bbox
        result.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "table": {"extract": rows, "row_count": len(rows), "col_count": table.col_count}, "metadata": {"extraction": "pdf_grid"}})
    return result

def page_nodes(raw_page, layout_page, body_size, clip_bottom=None, grids=None, *, clip_top=None):
    blocks = raw_page.get_text("dict")["blocks"]
    tables = [{**box, "metadata": {"extraction": "pdf_grid"}} for box in (grids or []) if clip_bottom is None or box["y1"] < clip_bottom + 2]
    for box in layout_page["boxes"]:
        if clip_bottom is not None and box["y1"] >= clip_bottom + 2:
            continue
        rect = pymupdf.Rect(box["x0"], box["y0"], box["x1"], box["y1"])
        overlaps = any((rect & pymupdf.Rect(grid["x0"], grid["y0"], grid["x1"], grid["y1"])).get_area() > rect.get_area() * 0.35 for grid in tables)
        if not overlaps and is_real_table(box, raw_page):
            if box["table"].get("cells"):
                rows = [[clean(raw_page.get_text("text", clip=pymupdf.Rect(cell), sort=True)) if cell else "" for cell in row] for row in box["table"]["cells"]]
                box = {**box, "table": {**box["table"], "extract": rows}}
            raw_text = " ".join(str(cell or "") for row in box["table"]["extract"] for cell in row)
            numeric = len(re.findall(r"\b\d[\d,.]*\b", raw_text)) > len(raw_text.split()) * 0.4
            bands = numeric_bands(raw_page, box) if numeric else []
            tables.extend(bands if bands else [box])
    toc_blocks = {i for i, block in enumerate(blocks) if raw_page.number < 2 and any(re.search(r"(?:\.\s*){3,}\s*\d+\s*$", line_text(line)) for line in block.get("lines", []))}
    lines = [(i, line) for i, block in enumerate(blocks) for line in block.get("lines", []) if i not in toc_blocks]
    if clip_top is not None:
        lines = [(i, line) for i, line in lines if line["bbox"][1] >= clip_top]
    lines = [(i, line) for i, line in lines if (clip_bottom is None or line["bbox"][1] < clip_bottom) and not re.search(r"To help us develop a more useful index|See .*Comments and Suggestions.*Introduction", clean(line_text(line)))]
    lefts = Counter()
    for _, line in lines:
        if line["bbox"][1] >= raw_page.rect.height - 45:
            continue
        regular = sum(len(s["text"]) for s in line["spans"] if abs(s["size"] - body_size) < 0.3 and not re.search(r"Bold|Black|Blk", s["font"], re.I))
        if regular:
            lefts[round(line["bbox"][0] / 6) * 6] += regular
    if not lefts:
        for _, line in lines:
            if 20 < line["bbox"][1] < raw_page.rect.height - 45 and max(s["size"] for s in line["spans"]) <= body_size + 3:
                lefts[round(line["bbox"][0] / 6) * 6] += len(line_text(line))
    margin = min((line["bbox"][0] for _, line in lines if 20 <= line["bbox"][0] and 20 < line["bbox"][1] < raw_page.rect.height - 45), default=42)
    margin = round(margin / 6) * 6
    candidates = []
    for count, gap in ((2, 18), (3, 12)):
        step = (raw_page.rect.width - 2 * margin + gap) / count
        grid = [margin + i * step for i in range(count)]
        loss = sum(weight * min(min(80, max(0, x - left - 42) + max(0, left - x - 10) * 3) for left in grid) for x, weight in lefts.items())
        candidates.append((loss, count, grid))
    _, _, anchors = min(candidates)
    wide = [line for _, line in lines if line["bbox"][0] < margin + 20 and line["bbox"][2] - line["bbox"][0] > raw_page.rect.width * 0.62]
    if wide and sum(len(line_text(line)) for line in wide) > sum(lefts.values()) * 0.65:
        anchors = [margin]
    step = min((b - a for a, b in zip(anchors, anchors[1:])), default=raw_page.rect.width)
    geometric = defaultdict(list)
    raw_icons = []
    for block_index, line in lines:
        x0, y0, x1, y1 = line["bbox"]
        text = clean(line_text(line))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if not text or y0 < 18 or y0 >= raw_page.rect.height - 45:
            continue
        if re.search(r"THIS PAGE INTENTIONALLY LEFT BLANK|The type and rule above prints|Draft Ok to Print|Userid:|XSL/XML", text):
            continue
        if any(box["x0"] - 1 <= cx <= box["x1"] + 1 and box["y0"] - 1 <= cy <= box["y1"] + 1 for box in tables):
            continue
        column = max((i for i, left in enumerate(anchors) if left <= x0 + 12), default=0)
        if text in {"DIRECT DEPOSIT", "Simple. Safe. Secure."}:
            continue
        if text in {"CAUTION", "TIP", "!"}:
            raw_icons.append({"text": text, "kind": "text", "level": 0, "bbox": list(line["bbox"]), "column": column, "spanning": False})
            continue
        baseline = round(line["spans"][0]["origin"][1])
        geometric[column, baseline].append((block_index, line))
    unified = []
    for (column, baseline), fragments in geometric.items():
        spans = sorted([s for _, line in fragments for s in line["spans"]], key=lambda span: span["bbox"][0])
        bbox = [min(s["bbox"][0] for s in spans), min(s["bbox"][1] for s in spans), max(s["bbox"][2] for s in spans), max(s["bbox"][3] for s in spans)]
        unified.append({"spans": spans, "bbox": bbox, "column": column, "blocks": {i for i, _ in fragments}})
    groups = []
    current = None
    for line in sorted(unified, key=lambda line: (line["column"], line["bbox"][1], line["bbox"][0])):
        text = clean(line_text(line))
        spans = [s for s in line["spans"] if s["text"].strip()]
        size = max(s["size"] for s in spans)
        bold = all(re.search(r"Bold|Black|Blk|-Bd", span["font"], re.I) for span in spans)
        heading = bold and size > body_size + 0.4
        category = (heading, round(size, 1) if heading else 0, line["column"])
        previous = current["lines"][-1] if current else None
        previous_text = clean(line_text(previous)) if previous else ""
        continuation = previous and not previous_text.endswith((".", ":", "?", "!")) and (previous_text.endswith("-") or text[:1].islower())
        gap = line["bbox"][1] - previous["bbox"][3] if previous else 999
        same_block = previous and bool(previous["blocks"] & line["blocks"])
        starts_runin = runin_heading({"textlines": [line]}) is not None
        split = bool(re.fullmatch(r"\d{1,2}\.", previous_text)) or gap > 6 or (not same_block and not continuation and not heading) or (starts_runin and not continuation)
        if current and current["category"] == category and (not split or heading and gap < 25 and not re.fullmatch(r"\d{1,2}\.", previous_text)):
            current["lines"].append(line)
        else:
            current = {"category": category, "lines": [line]}
            groups.append(current)
    nodes = []
    for group in groups:
        text = box_text({"textlines": group["lines"]})
        heading, size, column = group["category"]
        bbox = [min(l["bbox"][0] for l in group["lines"]), min(l["bbox"][1] for l in group["lines"]), max(l["bbox"][2] for l in group["lines"]), max(l["bbox"][3] for l in group["lines"])]
        level = 1 if size >= body_size + 9 else 2 if size >= body_size + 6 else 3 if size >= body_size + 3 else 4
        kind = "heading" if heading else "text"
        if heading and re.fullmatch(r"\d{1,2}\.", text) and size >= max(16, body_size + 7):
            kind = "chapter_number"
        nodes.append({"text": text, "kind": kind, "level": level, "bbox": bbox, "size": size,
                      "column": column, "spanning": bbox[2] - bbox[0] > step * 1.35, "textlines": group["lines"]})
    nodes.extend(raw_icons)
    for box in tables:
        nodes.append({"text": "", "kind": "table", "level": 0, "rows": box["table"]["extract"],
                      "bbox": [box[k] for k in ("x0", "y0", "x1", "y1")], "column": min(range(len(anchors)), key=lambda i: abs(anchors[i] - box["x0"])),
                      "spanning": box["x1"] - box["x0"] > step * 1.35, "table_metadata": box.get("metadata", {"extraction": "layout"})})
    icons = [node for node in nodes if node["text"] in {"CAUTION", "TIP", "!"}]
    for icon in icons:
        targets = [n for n in nodes if n not in icons and n["kind"] == "text" and n["column"] == icon["column"]
                   and n.get("textlines") and n["textlines"][0]["bbox"][0] > icon["bbox"][0] + 10 and abs(n["bbox"][1] - icon["bbox"][1]) < 65]
        if targets and icon["text"] != "!":
            target = min(targets, key=lambda n: (not (n["bbox"][1] <= icon["bbox"][1] <= n["bbox"][3] + 5), abs(n["bbox"][1] - icon["bbox"][1])))
            target["text"] = icon["text"] + ". " + target["text"]
            target["special"] = icon["text"].lower()
        nodes.remove(icon)
    barriers = sorted([node for node in nodes if node["spanning"]], key=lambda n: (n["bbox"][1], n["bbox"][0]))
    remaining = [node for node in nodes if not node["spanning"]]
    ordered = []
    for barrier in barriers:
        before = [n for n in remaining if n["bbox"][1] < barrier["bbox"][3]]
        before.sort(key=lambda n: (n["column"], round(n["bbox"][1], 1), n["bbox"][0]))
        ordered.extend(before)
        ordered.append(barrier)
        remaining = [n for n in remaining if n not in before]
    ordered.extend(sorted(remaining, key=lambda n: (n["column"], round(n["bbox"][1], 1), n["bbox"][0])))
    result = []
    for node in ordered:
        if result and node["kind"] == result[-1]["kind"] == "heading":
            previous = result[-1]
            if abs(previous["size"] - node["size"]) < 0.1 and abs(previous["bbox"][0] - node["bbox"][0]) < 10 and 0 <= node["bbox"][1] - previous["bbox"][3] < 25:
                previous["text"] += " " + node["text"]
                previous["bbox"][3] = node["bbox"][3]
                continue
        if node["kind"] == "heading" and not valid_heading(node["text"]):
            node["kind"] = "text"
        if node["kind"] == "text" and not node.get("special"):
            title = runin_heading(node)
            if title:
                node["runin"] = title
        standalone_bold = node["kind"] == "text" and 2 <= len(node["text"].split()) <= 18 and (not node["text"].endswith((".", ",", ";", "-")) or bool(re.match(r"^Exception\s+\d", node["text"]))) and all(re.search(r"Bold|Black|Blk|-Bd", span["font"], re.I) for line in node.get("textlines", []) for span in line["spans"] if span["text"].strip())
        if node["kind"] == "text" and (standalone_bold or re.fullmatch(r"Lines? [\dA-Za-z, \u2013-]+", node["text"])):
            node.update(kind="heading", level=5)
            node.pop("runin", None)
        if node["kind"] == "heading" and re.match(r"^(?:Table|Worksheet|Figure|Chart)\s+[\dA-Z]", node["text"]):
            node["kind"] = "caption"
        if node["text"] in {"2025", "TAX YEAR", "For use in preparing 2025 Returns", "Keep for Your Records", "Keep for Your Records:", "Before you begin:"}:
            continue
        result.append(node)
    return result


def pdf_elements(document, destination, policy):
    path = ROOT / document["path"]
    cache = destination / (path.stem + "_layout.json")
    with pymupdf.open(path) as pdf:
        if "2025" not in pdf[0].get_text():
            raise ValueError("Missing 2025 source year: " + str(path))
        pages = ranges(document["name"], len(pdf))
        if document["name"] == "IRS Pub 17":
            pages = list(range(3, 113))
        start_page = document.get("pdf_start_page", 1)
        pages = [page for page in pages if page >= start_page]
        identity = {"source": fingerprint(path.read_bytes().hex()), "pages": pages, "parser": pymupdf.VersionBind, "ocr": False}
        if cache.exists():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached["identity"] != identity:
                raise ValueError("Stale layout cache: " + str(cache))
            parsed = cached["parsed"]
        else:
            parsed = json.loads(pdf.to_json(pages=[p - 1 for p in pages], use_ocr=False))
            cache.write_text(json.dumps({"identity": identity, "parsed": parsed}), encoding="utf-8")
        fonts = Counter()
        for sample in pages[:8]:
            for block in pdf[sample - 1].get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        if 25 < span["bbox"][1] < 745 and not re.search(r"Bold|Black|Blk|Italic", span["font"], re.I):
                            fonts[round(span["size"], 1)] += len(span["text"])
        body_size = fonts.most_common(1)[0][0] if fonts else 10
        source = document["name"] + " (2025)"
        index = next((r for r in policy["index_regions"] if r["source"] == source), None)
        elements, removed = [], Counter()
        grid_path = destination / (path.stem + "_grids.json")
        grid_cache = json.loads(grid_path.read_text(encoding="utf-8")) if grid_path.exists() else {"identity": identity, "pages": {}}
        if grid_cache["identity"] != identity:
            raise ValueError("Stale grid cache: " + str(grid_path))
        for page in parsed["pages"]:
            number = page["page_number"]
            if index and number > index["pdf_page"]:
                if document["name"] != "IRS Form 1040 Instructions" or number <= 124:
                    removed["index_or_reference_directory_pages"] += 1
                    continue
            raw_page = pdf[number - 1]
            labels = [clean(line_text(line)) for block in raw_page.get_text("dict")["blocks"] for line in block.get("lines", []) if line["bbox"][1] > raw_page.rect.height - 45]
            label = next((re.fullmatch(r"(?:Page\s+)?(?:C-)?(\d{1,3})", t, re.I) for t in labels if re.fullmatch(r"(?:Page\s+)?(?:C-)?(\d{1,3})", t, re.I)), None)
            printed = int(label.group(1)) if label else None
            if document["name"].startswith("FTB"):
                printed = number
            clip_bottom = None
            if index and number == index["pdf_page"]:
                headers = [span for block in raw_page.get_text("dict")["blocks"] for line in block.get("lines", []) for span in line["spans"] if span["text"].strip() in {"Index", "Index to Instructions"} and re.search("Bold|Black|Blk", span["font"])]
                if len(headers) != 1 or abs(headers[0]["bbox"][1] - index["top"]) > 10:
                    raise ValueError("The reviewed index boundary changed: " + source)
                clip_bottom = headers[0]["bbox"][1]
                removed["index_boundary_applied"] += 1
            if str(number) not in grid_cache["pages"]:
                grid_cache["pages"][str(number)] = grid_tables(raw_page)
                grid_path.write_text(json.dumps(grid_cache), encoding="utf-8")
            nodes = page_nodes(raw_page, page, body_size, clip_bottom, grid_cache["pages"][str(number)],
                               clip_top=document.get("pdf_content_top"))
            for node in nodes:
                node.update(page=number, printed=printed)
                elements.append(node)
    return elements, dict(removed)


def make_document(document, elements):
    name = document["name"]
    source = name + " (2025)"
    prefix = re.sub(r"[^a-z0-9_]+", "_", Path(document["path"]).stem.lower().replace("--2025", "").replace("2025-", "ftb_"))
    chunks, tables, audit = [], [], Counter()
    headings = {}
    runin = None
    chapter_no = None
    chapter = None
    part = None
    pending_chapter = None
    paragraphs = []
    printed = {}
    reference = False
    caption = None
    kind = "rule"
    table_metadata = {}

    def emit(text, pages, *, table_rows=None, table_caption=None, row_range=None, store_table=True, retrieval_eligible=True):
        sections = list(headings.values())
        if sections and sections[0] == chapter:
            sections = sections[1:]
        path = list(headings.values())
        chapter_label = f"Chapter {chapter_no} {chapter}" if chapter_no is not None else None
        if chapter_label and (not path or path[0] != chapter):
            path.insert(0, chapter_label)
        elif chapter_label and path:
            path[0] = chapter_label
        if runin and runin not in path:
            path.append(runin)
        heading = " > ".join(path) or "Instructions"
        if table_caption:
            heading += " > " + table_caption
        embed = heading + "\n" + text
        digest = fingerprint(embed)[:16]
        page_label = "web section" if not pages else "PDF " + ("p. " if len(pages) == 1 else "pp. ") + "-".join(map(str, [pages[0]] if len(pages) == 1 else [pages[0], pages[-1]]))
        printed_pages = sorted({printed[p] for p in pages if printed.get(p) is not None})
        citation = f"{source}, '{heading}', {page_label}"
        if name == "IRS Schedule C Instructions" and printed_pages:
            citation += ", printed C-" + "-C-".join(map(str, [printed_pages[0]] if len(printed_pages) == 1 else [printed_pages[0], printed_pages[-1]]))
        elif printed_pages and printed_pages != pages:
            citation += ", printed " + "-".join(map(str, [printed_pages[0]] if len(printed_pages) == 1 else [printed_pages[0], printed_pages[-1]]))
        if row_range:
            citation += f", table rows {row_range[0]}-{row_range[1]}"
        item = {"id": f"{prefix}_2025_ch{chapter_no or 0:02d}_{digest}" + ("_" + fingerprint(json.dumps([table_rows, row_range]))[:10] if table_rows is not None else ""), "tax_year": 2025, "source": source,
                "chapter_no": chapter_no, "chapter": chapter, "part": part, "section": sections[0] if sections else None,
                "subsection": sections[1] if len(sections) > 1 else None, "minor": " > ".join(sections[2:]) or None, "runin": runin,
                "type": "table" if table_rows is not None else kind, "table_kind": "table" if table_rows is not None else None,
                "printed_pages": printed_pages, "pdf_pages": pages, "citation": citation, "text": text,
                "embed_text": embed, "content_hash": digest, "n_words": len(text.split()),
                "source_url": document["url"], "retrieval_eligible": retrieval_eligible, "short_reason": "section_boundary" if len(text.split()) < 150 else None}
        chunks.append(item)
        if table_rows is not None and store_table:
            tables.append({"chunk_id": item["id"], "caption": table_caption or heading, "kind": "table",
                           "citation": citation.split(", table rows")[0], "chapter_no": chapter_no, "printed_page": printed_pages[0] if printed_pages else None,
                           "pdf_page": pages[0] if pages else None, "rows": table_rows, "metadata": {**table_metadata, "source_url": document["url"]}})

    def flush():
        nonlocal paragraphs
        for text, pages in split_passage(paragraphs):
            fragment = (len(text.split()) == 1 and not text.endswith(".")) or bool(re.fullmatch(r"(?:Firm.s name|COPY [A-Z].*|RECORDS\.)", text))
            if fragment or not re.search(r"[A-Za-z]{2}", text) or re.fullmatch(r"(?:Form )?1040(?:-SR)?|Sample Table|Enter earlier year(?: \d{4})?", text):
                audit["non_rule_labels_removed"] += 1
                continue
            emit(text, pages)
        paragraphs = []

    for element in elements:
        text = element["text"]
        page = element["page"]
        if page:
            printed[page] = element["printed"]
        if element["kind"] == "chapter_number":
            flush()
            pending_chapter = int(text.rstrip("."))
            continue
        if element["kind"] == "heading" and re.search(r"^(?:Useful Items|Publications? and Forms|Other IRS Publications|Table of Contents|Contents)$", text, re.I):
            flush()
            reference = "contents" if text in {"Contents", "Table of Contents"} else "references"
            audit["reference_headings"] += 1
            continue
        if reference:
            if ((element["kind"] == "heading" and element["level"] < 5 and not re.search(r"\d+$", text)) or (reference != "contents" and element.get("runin"))) and text not in {"Publication", "Publications", "Form (and Instructions)", "Forms (and Instructions)", "Form", "Forms"}:
                reference = False
            else:
                audit["reference_list_boxes"] += 1
                continue
        if element["kind"] == "heading":
            flush()
            runin = None
            level = element["level"]
            if re.match(r"^Part (?:One|Two|Three|Four|Five|Six|[IVX]+)\.", text):
                chapter_no, chapter, part = None, None, text
                headings = {}
            if re.match(r"^Appendix [A-Z][—–-]", text):
                chapter_no, chapter, part = None, None, None
                headings = {}
                level = 1
            if pending_chapter is not None:
                chapter_no, chapter = pending_chapter, text
                pending_chapter = None
                headings = {}
                level = 1
            for key in list(headings):
                if key >= level:
                    del headings[key]
            headings[level] = text
            kind = "whats_new" if re.search(r"What.s New", text, re.I) else "rule"
            caption = None
            continue
        if element["kind"] == "caption":
            flush()
            runin = None
            caption = text
            continue
        if element["kind"] == "table":
            flush()
            table_metadata = element.get("table_metadata", {"extraction": "html" if page is None else "layout"})
            rows = [[clean(cell or "") for cell in row] for row in element["rows"]]
            rows = [row for row in rows if any(row)]
            catalog_rows = sum(bool(re.search(r"\bPub(?:lication)?\.?\s+\d+|IRS\.gov/(?:\w+)", " ".join(row), re.I)) for row in rows)
            if len(rows) >= 5 and catalog_rows >= len(rows) * 0.6:
                audit["reference_tables_removed"] += 1
                continue
            if not rows:
                continue
            table_caption = caption or next(reversed(headings.values()), "Table")
            numeric = sum(len(re.findall(r"\b\d[\d,.]*\b", " ".join(row))) for row in rows)
            words = sum(len(" ".join(row).split()) for row in rows)
            if numeric > words * 0.45:
                header_cells = [cell for row in rows[:2] for cell in row if re.search(r"[A-Za-z]{3}", cell) and len(cell.split()) <= 18 and len(re.findall(r"\d+", cell)) < 3]
                title = table_caption + (". Columns: " + " | ".join(header_cells[:4]) if header_cells else ". Numeric table")
                title += ". See the cited official table for values."
                emit(title, [page] if page else [], table_rows=rows, table_caption=table_caption, retrieval_eligible=False)
                audit["numeric_tables"] += 1
            else:
                header = next((row for row in rows[:2] if sum(bool(cell) for cell in row) >= 2 and len(" ".join(row).split()) <= 80 and not re.fullmatch(r"[\d\W]+", row[0] or "")), [])
                table_prefix = table_caption + ("\nColumns: " + " | ".join(header) if header else "")
                budget = 395 - len(table_prefix.split())
                groups, group, count = [], [], 0
                table_start = len(tables)
                for number, row in enumerate(rows, 1):
                    rendered = " | ".join(row)
                    size = len(rendered.split())
                    if size > budget:
                        emit(table_caption + ". Row " + str(number) + " is in the cited official table.", [page] if page else [], table_rows=rows, table_caption=table_caption, row_range=(number, number), store_table=len(tables) == table_start, retrieval_eligible=False)
                        audit["oversized_rows_stored_exactly"] += 1
                        continue
                    if group and count + size > budget:
                        groups.append(group)
                        group, count = [], 0
                    group.append((number, row))
                    count += size
                if group:
                    groups.append(group)
                for group in groups:
                    body = table_prefix + "\n" + "\n".join(" | ".join(row) for _, row in group)
                    emit(body, [page] if page else [], table_rows=rows, table_caption=table_caption, row_range=(group[0][0], group[-1][0]), store_table=len(tables) == table_start)
                audit["semantic_tables"] += 1
            caption = None
            continue
        title = element.get("runin")
        special = element.get("special") or ("caution" if re.search(r"\bCAUTION\b", text) else "tip" if re.search(r"\bTIP\b", text) else None)
        if title or special:
            flush()
            runin = title
        if special:
            previous_kind = kind
            kind = special
            paragraphs.append((text, [page] if page else []))
            flush()
            kind = previous_kind
            continue
        paragraphs.append((text, [page] if page else []))
    flush()
    unique, duplicates = {}, []
    for chunk in chunks:
        if chunk["id"] in unique:
            old = unique[chunk["id"]]
            extra = sorted(set(chunk["pdf_pages"]) - set(old["pdf_pages"]))
            old["pdf_pages"] = sorted(set(old["pdf_pages"] + chunk["pdf_pages"]))
            old["printed_pages"] = sorted(set(old["printed_pages"] + chunk["printed_pages"]))
            if extra:
                old["citation"] += ", also PDF pp. " + ", ".join(map(str, extra))
            duplicates.append(chunk["id"])
        else:
            unique[chunk["id"]] = chunk
    table_by_id = {t["chunk_id"]: t for t in tables}
    audit["exact_duplicate_chunks_removed"] = len(duplicates)
    return list(unique.values()), list(table_by_id.values()), dict(audit)



def canonicalize(chunks, tables):
    groups = defaultdict(list)
    for chunk in chunks:
        key = fingerprint(clean(chunk["text"]).casefold()) if chunk["type"] != "table" and chunk["n_words"] >= 40 else chunk["id"]
        groups[key].append(chunk)
    result, merged = [], []
    for group in groups.values():
        group.sort(key=lambda c: ("Booklet" in c["source"], c["source"], c["id"]))
        canonical = dict(group[0])
        canonical["provenance"] = [{k: c[k] for k in ("source", "citation", "pdf_pages", "printed_pages", "source_url")} for c in group]
        result.append(canonical)
        if len(group) > 1:
            merged.append({"canonical": canonical["id"], "aliases": [c["id"] for c in group[1:]], "provenance": canonical["provenance"]})
    return result, tables, merged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "config/knowledge_sources.json")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    documents = json.loads(args.manifest.read_text(encoding="utf-8"))
    policy = json.loads((ROOT / "config/retrieval_exclusions.json").read_text(encoding="utf-8"))
    chunks, tables, report = [], [], {}
    for document in documents:
        path = ROOT / document["path"]
        if document["url"].endswith(".html"):
            elements, removed = html_elements(path.with_suffix(".html")), {}
        else:
            elements, removed = pdf_elements(document, args.out, policy)
        records, raw, audit = make_document(document, elements)
        report[document["name"]] = {"chunks": len(records), "tables": len(raw), "pages": sorted({p for c in records for p in c["pdf_pages"]}), **removed, **audit}
        chunks.extend(records)
        tables.extend(raw)
        print(document["name"], len(records), "chunks", len(raw), "tables", flush=True)
    expected = {d["name"] + " (2025)" for d in documents}
    if {c["source"] for c in chunks} != expected or len({c["id"] for c in chunks}) != len(chunks):
        raise ValueError("Invalid source or chunk inventory")
    for chunk in chunks:
        if chunk["type"] not in ALLOWED_TYPES or not re.fullmatch(r"[a-z0-9_]+_2025_ch\d{2}_[a-f0-9]{16}(?:_[a-f0-9]{10})?", chunk["id"]) or not chunk["text"] or chunk["n_words"] > 400 or chunk["embed_text"].split("\n", 1)[-1] != chunk["text"]:
            raise ValueError("Invalid chunk: " + chunk["id"])
    chunks, tables, merged = canonicalize(chunks, tables)
    duplicate_texts = defaultdict(list)
    for chunk in chunks:
        duplicate_texts[fingerprint(clean(chunk["text"]).casefold())].append(chunk["id"])
    report = {"version": VERSION, "parser": pymupdf.VersionBind, "documents": report, "chunks": len(chunks), "eligible_chunks": sum(c["retrieval_eligible"] for c in chunks), "tables": len(tables),
              "canonical_duplicates": merged,
              "short_sections": sum(c["n_words"] < 150 for c in chunks),
              "cross_source_duplicate_groups": [ids for ids in duplicate_texts.values() if len(ids) > 1],
              "source_files": {str(Path(d["path"]).with_suffix(".html")) if d["url"].endswith(".html") else d["path"]: hashlib.sha256((ROOT / d["path"]).with_suffix(".html").read_bytes() if d["url"].endswith(".html") else (ROOT / d["path"]).read_bytes()).hexdigest() for d in documents},
              "chunks_sha256": fingerprint(json.dumps(chunks, sort_keys=True, ensure_ascii=False)),
              "tables_sha256": fingerprint(json.dumps(tables, sort_keys=True, ensure_ascii=False))}
    for name, value in (("chunks.json", chunks), ("tables.json", tables), ("quality.json", report)):
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("TOTAL", len(chunks), len(tables), flush=True)


if __name__ == "__main__":
    main()
