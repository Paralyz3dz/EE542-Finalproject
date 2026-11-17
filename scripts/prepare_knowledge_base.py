import json
import re
import shutil
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed/knowledge_base"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in (ROOT / "docs/knowledge-base.md").read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < 7 or not cells[1].startswith(("IRS ", "FTB ")):
            continue
        name, relative = cells[1], cells[4].strip("`")
        url = re.search(r"https://\S+", cells[5]).group().rstrip(")")
        path = RAW / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            matching = {
                "IRS Form 1040 Instructions": RAW / "irs/instructions/2025 Instruction 1040.pdf",
                "IRS Pub 503": RAW / "irs/publications/2025 Publication 503.pdf",
                "IRS Pub 970": RAW / "irs/publications/2025 Publication 970.pdf",
            }.get(name)
            if matching and matching.exists():
                shutil.copyfile(matching, path)
            else:
                response = requests.get(url, timeout=90)
                response.raise_for_status()
                if response.content.startswith(b"%PDF"):
                    path.write_bytes(response.content)
                else:
                    path = path.with_suffix(".html")
                    path.write_bytes(response.content)
        rows.append({"name": name, "path": str(path.relative_to(ROOT)), "url": url})
        print(name, path.name, flush=True)
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
