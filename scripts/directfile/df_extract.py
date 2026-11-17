"""Read Direct File (CC0) flow + fact dictionary + English text into one question list.
Only reads files; never executes repository code."""
import json, re, sys
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
TAX = ROOT / "backend/src/main/resources/tax"
APP = ROOT / "df-client/df-client-app/src"

facts = {}
for xml in TAX.glob("*.xml"):
    if xml.name == "flow.xml":
        continue
    try:
        tree = ET.parse(xml)
    except ET.ParseError:
        continue
    for f in tree.iter("Fact"):
        w = f.find("Writable")
        if w is None or len(w) == 0:
            continue
        t = w[0]
        facts[f.get("path")] = {
            "module": xml.stem,
            "type": t.tag,
            "options_path": t.get("optionsPath"),
            "description": " ".join((f.findtext("Description") or "").split()),
        }

en = yaml.safe_load((APP / "locales/en.yaml").read_text(encoding="utf-8"))
headings, fields = en.get("headings", {}), en.get("fields", {})


def clean(text):
    if not isinstance(text, str):
        return None
    text = re.sub(r"\{\{/taxYear\}\}", "last year", text)
    text = re.sub(r"\{\{[^}]*firstName\}\}", "[name]", text)
    text = re.sub(r"\{\{[^}]*\}\}", "[...]", text)
    text = re.sub(r"<[^>]+>", "", text)
    return " ".join(text.split())


def options_for(path, meta):
    op = meta.get("options_path") if meta else None
    for cand in ([fields.get(path, {}).get(op)] if op and isinstance(fields.get(path), dict) else []) + [fields.get(op)]:
        if isinstance(cand, dict):
            if "long" in cand and isinstance(cand["long"], dict):
                cand = cand["long"]
            return {k: clean(v) if isinstance(v, str) else v for k, v in cand.items()}
    return None


COND_OBJ = re.compile(r"\{\s*(?:operator:\s*`(\w+)`\s*,\s*)?condition:\s*`([^`]+)`\s*\}")
COND_STR = re.compile(r"condition\s*=\s*['\"`]([^'\"`]+)['\"`]")


def conditions(attrs):
    attrs = attrs.replace("\\/", "/")
    out = []
    for op, path in COND_OBJ.findall(attrs):
        out.append({"fact": path, "operator": op or "isTrue"})
    for path in COND_STR.findall(attrs):
        out.append({"fact": path, "operator": "isTrue"})
    m = re.search(r"conditions=\{\[(.*?)\]\}", attrs, re.S)
    if m:
        for path in re.findall(r"(?<![:\w])[`'\"](/[^`'\"]+)[`'\"]", m.group(1)):
            if not any(c["fact"] == path for c in out):
                out.append({"fact": path, "operator": "isTrue"})
    return out


def attr(attrs, name):
    attrs = attrs.replace("\\/", "/")
    m = re.search(rf"{name}\s*=\s*\{{?\s*['\"`]([^'\"`]+)['\"`]", attrs)
    return m.group(1) if m else None


def is_mfj_or_spouse(attrs, key=None):
    if key and (key.endswith("-mfj") or key.endswith("_spouse")):
        return True
    if re.search(r"i18nKeySuffixContext\s*=\s*['\"](mfj|spouse)", attrs):
        return True
    for c in conditions(attrs):
        if c["fact"] in ("/isFilingStatusMFJ",) and c["operator"] == "isTrue":
            return True
    return False


TAG = re.compile(r"<(/?)(Subcategory|SubSubcategory|CollectionLoop|Gate|Screen|Heading|Boolean|Dollar|Enum|MultiEnum|"
                 r"DatePicker|LimitingString|GenericString|KnockoutButton|Tin|Address)\b((?:[^<>]|=>)*?)(/?)>", re.S)
INPUTS = {"Boolean", "Dollar", "Enum", "MultiEnum", "DatePicker", "LimitingString", "GenericString", "Tin", "Address"}

screens = []
for tsx in sorted((APP / "flow/flow-chunks").rglob("*.tsx")):
    src = tsx.read_text(encoding="utf-8")
    stack, cur = [], None
    for m in TAG.finditer(src):
        closing, tag, attrs, selfclose = m.group(1), m.group(2), m.group(3), m.group(4)
        if tag in ("Subcategory", "SubSubcategory", "CollectionLoop", "Gate", "Screen"):
            if closing:
                while stack and stack[-1]["tag"] != tag:
                    stack.pop()
                if stack:
                    stack.pop()
                if tag == "Screen" and cur:
                    screens.append(cur)
                    cur = None
                continue
            conds = [] if tag in ("Subcategory", "SubSubcategory") else conditions(attrs)
            node = {"tag": tag, "route": attr(attrs, "route"), "conditions": conds,
                    "collection": attr(attrs, "collectionContext") or attr(attrs, "loopName")}
            if tag == "Screen":
                gates = [c for n in stack for c in n["conditions"]]
                cur = {"file": str(tsx.relative_to(APP / "flow/flow-chunks")),
                       "section": "/".join(n["route"] for n in stack if n["tag"] in ("Subcategory", "SubSubcategory") and n["route"]),
                       "screen": node["route"], "show_when": gates + node["conditions"],
                       "headings": [], "inputs": [], "knockout": False,
                       "mfj_only": any(c["fact"] == "/isFilingStatusMFJ" and c["operator"] == "isTrue"
                                       for c in gates + node["conditions"])}
            if not selfclose:
                stack.append(node)
            continue
        if cur is None or closing:
            continue
        if tag == "Heading":
            key = attr(attrs, "i18nKey")
            if key and not is_mfj_or_spouse(attrs, key):
                text = clean(headings.get(key))
                if text and text not in cur["headings"]:
                    cur["headings"].append(text)
        elif tag == "KnockoutButton":
            cur["knockout"] = True
        elif tag in INPUTS:
            path = attr(attrs, "path")
            if not path or is_mfj_or_spouse(attrs) or "secondaryFiler" in path:
                continue
            meta = facts.get(path)
            cur["inputs"].append({"fact": path, "type": tag, "required": "required={false}" not in attrs,
                                  "options": options_for(path, meta) if tag in ("Enum", "MultiEnum") else None,
                                  "module": meta["module"] if meta else None,
                                  "description": meta["description"] if meta else None,
                                  "condition": conditions(attrs)})

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "directfile_writable_facts.json").write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")
(OUT / "directfile_screens.json").write_text(json.dumps(screens, indent=1, ensure_ascii=False), encoding="utf-8")
asked = [s for s in screens if s["inputs"]]
print(f"writable facts: {len(facts)}  screens: {len(screens)}  question screens: {len(asked)}  "
      f"knockout screens: {sum(s['knockout'] for s in screens)}")
