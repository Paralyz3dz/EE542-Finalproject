"""Split Direct File screens into interview questions, document-entry fields and knockouts (out of scope)."""
import csv, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(sys.argv[1])
screens = json.load(open(OUT / "directfile_screens.json", encoding="utf-8"))

DOC_PATH = re.compile(r"^/(formW2s|form1099Rs|socialSecurityReports|interestReports|form1099Gs|form1099Miscs|"
                      r"hsaDistributions|w2|filers/\*/(tin|firstName|lastName|middleInitial|suffix|dateOfBirth|address|"
                      r"phone|email|occupation|identityPin))")
DOC_TEXT = re.compile(r"\bbox(es)?\b|Recipient|Payer|employer’s|EIN|TIN|address|phone|email|name", re.I)
ADMIN = ("sign-and-submit/", "your-taxes/PaymentMethod", "your-taxes/OtherPreferences")

rows, knockouts = [], []
for s in screens:
    if s["mfj_only"] or s["file"].startswith(ADMIN):
        continue
    q = s["headings"][0] if s["headings"] else ""
    if s["knockout"]:
        knockouts.append({"section": s["file"].split("/")[-1].replace(".tsx", ""), "screen": s["screen"],
                          "message": q, "triggered_when": "; ".join(f'{c["operator"]} {c["fact"]}' for c in s["show_when"])})
        continue
    for i in s["inputs"]:
        kind = "document_field" if (DOC_PATH.match(i["fact"]) or DOC_TEXT.search(q)) and i["type"] != "Boolean" \
            or i["type"] in ("Tin", "Address", "LimitingString", "GenericString") else "interview"
        rows.append({"kind": kind, "section": s["file"].split("/")[-1].replace("Subcategory.tsx", "").replace(".tsx", ""),
                     "screen": s["screen"], "fact": i["fact"], "type": i["type"], "question": q,
                     "options": " | ".join(str(v) for v in (i["options"] or {}).values())[:300],
                     "show_when": "; ".join(f'{c["operator"]} {c["fact"]}' for c in s["show_when"] + i["condition"]),
                     "description": i["description"] or ""})

seen, uniq = set(), []
for r in rows:
    key = (r["fact"], r["question"])
    if key not in seen:
        seen.add(key); uniq.append(r)
for name, data in [("directfile_interview_questions.csv", [r for r in uniq if r["kind"] == "interview"]),
                   ("directfile_document_fields.csv", [r for r in uniq if r["kind"] == "document_field"]),
                   ("directfile_knockouts.csv", knockouts)]:
    with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)

c = Counter((r["kind"], r["section"]) for r in uniq)
by = defaultdict(dict)
for (k, sec), n in c.items():
    by[sec][k] = n
print(f"interview questions: {sum(1 for r in uniq if r['kind']=='interview')}, document fields: "
      f"{sum(1 for r in uniq if r['kind']=='document_field')}, knockouts: {len(knockouts)}")
for sec, d in sorted(by.items(), key=lambda x: -sum(x[1].values())):
    print(f"  {sec:34s} interview={d.get('interview',0):3d}  document={d.get('document_field',0):3d}")
