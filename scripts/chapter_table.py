"""Print the README chapter table (markdown) from site/data.js, so counts always match the repo.

    python scripts/chapter_table.py            # table rows
    python scripts/chapter_table.py --totals   # one-line totals
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
raw = (ROOT / "site" / "data.js").read_text(encoding="utf-8")
data = json.loads(raw.split("=", 1)[1].rstrip().rstrip(";"))

if "--totals" in sys.argv:
    t = data["totals"]
    print(f"{t['chapters']} chapters | {t['sections']} sections | {t['code']} code blocks | {t['words']:,} words")
    sys.exit(0)

print("| # | Chapter | Level | Episodes | Code | Starring | PDF |")
print("|--:|---|---|--:|--:|---|---|")
for c in data["chapters"]:
    cast = ", ".join(c["cast"][:3])
    print(f"| {c['num']:02d} | [{c['title']}]({c['md'].replace('chapters/', 'chapters/')}) | {c['level']} | "
          f"{len(c['sections'])} | {c['codeBlocks']} | {cast} | [PDF]({c['pdf']}) |")
