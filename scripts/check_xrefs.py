"""Check cross-references between chapters.

Verifies references of the forms
    "Module 05 section 3.4"   -> chapter 05 must contain a '### 3.4 ...' sub-heading or a '## 5.3/3.4' style section
    "section 14.5" / "see 8.7" / "§ 13.2"  where the first number is a chapter that has an explicit '## N.M' section
and reports references whose target does not exist. Ignores code fences. Prints a summary; exit 1 if any broken.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
chapters = {}
for md in sorted((ROOT / "chapters").glob("*.md")):
    n = int(md.name[:2])
    text = md.read_text(encoding="utf-8")
    prose, fence = [], False
    for ln in text.split("\n"):
        if ln.lstrip().startswith("```"):
            fence = not fence
            continue
        if not fence:
            prose.append(ln)
    body = "\n".join(prose)
    chapters[n] = {
        "name": md.name,
        "prose": body,
        "h2": set(re.findall(r"^## (\d+\.\d+)\s", body, flags=re.M)),                   # explicit N.M sections
        "legacy_h2": set(re.findall(r"^## (\d+)\.\s", body, flags=re.M)),               # '## 3.' legacy sections
        "h3": set(re.findall(r"^#{3,4} (\d+\.\d+(?:\.\d+)?)\s", body, flags=re.M)),     # '### 3.4' sub-headings
    }

broken, checked = [], 0
for n, c in chapters.items():
    for i, ln in enumerate(c["prose"].split("\n"), 1):
        if ln.startswith("#"):
            continue
        # Module NN section X.Y
        for m in re.finditer(r"Module\s+(\d{1,2})\s+(?:sections?|§)\s+(\d+(?:\.\d+)+)", ln):
            tgt, sec = int(m.group(1)), m.group(2)
            checked += 1
            t = chapters.get(tgt)
            if t is None:
                broken.append((c["name"], ln.strip()[:90], f"module {tgt} missing"))
            elif "." in sec and sec not in t["h3"] and sec not in t["h2"]:
                broken.append((c["name"], ln.strip()[:90], f"Module {tgt} has no {sec}"))
            elif "." not in sec and sec not in t["legacy_h2"]:
                broken.append((c["name"], ln.strip()[:90], f"Module {tgt} has no section {sec}"))
        # plain "section N.M" / "see N.M" / "§N.M" where N is a chapter with explicit sections
        for m in re.finditer(r"(?:section|sections|see|§)\s*(\d{1,2})\.(\d{1,2})\b(?!\.\d)", ln):
            tgt, sub = int(m.group(1)), m.group(2)
            if "Module" in ln[max(0, m.start() - 12):m.start()]:
                continue
            t = chapters.get(tgt)
            sec = f"{tgt}.{sub}"
            if t is None or not t["h2"]:
                continue                      # chapter without explicit sections: legacy numbering, skip
            checked += 1
            if sec not in t["h2"] and sec not in t["h3"]:
                broken.append((c["name"], ln.strip()[:90], f"no section {sec}"))

# Bare "N.M" tokens, e.g. "(9.17)", "see 9.17 and 15.1": only for chapters that have explicit N.M sections, and only
# when M looks like a section index of that chapter (not a decimal such as "15.5 GB" or "0.95").
UNIT = re.compile(r"^\s*(?:ms|s|sec|gb|mb|kb|tb|x|%|k|m|h|d|usd|\$|per|tokens?|steps?|epochs?|gpus?|b)\b", re.I)
for n, c in chapters.items():
    if not c["h2"]:
        continue
    for i, ln in enumerate(c["prose"].split("\n"), 1):
        if ln.startswith("#") or ln.lstrip().startswith("|"):
            continue
        for m in re.finditer(r"(?<![\d.$/\w-])(\d{1,2})\.(\d{1,2})(?![\d.]*\d)", ln):
            tgt, sub = int(m.group(1)), int(m.group(2))
            t = chapters.get(tgt)
            if t is None or not t["h2"] or tgt == n and False:
                continue
            tail = ln[m.end():m.end() + 8]
            if UNIT.match(tail) or re.match(r"^\s*[%×≈±]", tail) or re.match(r"^\s*\*\*\s*$", tail):
                continue
            if re.search(r"[×≈±=*/]\s*$", ln[max(0, m.start() - 3):m.start()]):    # operand of an arithmetic expression
                continue
            members = sorted(int(x.split(".")[1]) for x in t["h2"])
            if sub < members[0] - 1 or sub > members[-1] + 3:
                continue                       # outside the range of real sections: treat as a decimal
            checked += 1
            key = f"{tgt}.{sub}"
            if key not in t["h2"] and key not in t["h3"] and key not in c["h3"]:   # c["h3"]: local legacy sub-section
                broken.append((c["name"], ln.strip()[:90], f"no section {key}"))

print(f"cross-references checked: {checked}; broken: {len(broken)}")
for b in broken[:60]:
    print(" BROKEN", b)
sys.exit(1 if broken else 0)
