"""Structural audit for the chapters (the 'Agent 21' guardrail).

Config: scripts/project.json  (expected_chapters, required_parts, reference_chapters)
        scripts/required_concepts.json  ({chapter_stem: [keywords...]})

Rules
  1. Every explicitly numbered concept section (`## N.M Title`) in a non-reference chapter must contain
     each required part heading and at least one fenced code block.
  2. Numbered sections `N.M` must be unique within a chapter.
  3. Every ```python block must parse (ast).
  4. Each chapter must mention every required-concept keyword (case-insensitive).
Legacy sections written as `## 3. Title` (no N.M) are not format-checked; they are covered by 3 and 4.
"""
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mdutil import split_sections  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "scripts" / "project.json").read_text(encoding="utf-8"))
REQUIRED = CFG.get("required_parts", ["Conceptual Deep Dive", "Cross-Connection Scenarios",
                                      "Production Anti-Patterns", "Minimal Working Code"])
REFERENCE = set(CFG.get("reference_chapters", []))


def sections(text):
    return [(head, head + "\n" + body) for head, body in split_sections(text)]


def main():
    concepts_path = ROOT / "scripts" / "required_concepts.json"
    concepts = json.loads(concepts_path.read_text(encoding="utf-8")) if concepts_path.exists() else {}
    files = sorted((ROOT / "chapters").glob("*.md"))
    exp = CFG.get("expected_chapters")
    problems = 0
    if exp and len(files) != exp:
        print(f"[WARN] expected {exp} chapters, found {len(files)}")
        problems += 1
    for md in files:
        text = md.read_text(encoding="utf-8")
        low = text.lower()
        issues, seen_nums = [], set()
        for title, body in sections(text):
            m = re.match(r"(\d+\.\d+)\s", title)
            if not m:
                continue
            if m.group(1) in seen_nums:
                issues.append(f"duplicate section number {m.group(1)}")
            seen_nums.add(m.group(1))
            if md.stem in REFERENCE:
                continue
            for req in REQUIRED:
                if req.lower() not in body.lower():
                    issues.append(f"section '{title}' missing '{req}'")
            if "```" not in body:
                issues.append(f"section '{title}' has no code block")
        for m in re.finditer(r"```python\n(.*?)```", text, flags=re.S):
            try:
                ast.parse(m.group(1))
            except SyntaxError as e:
                issues.append(f"python block syntax error line {e.lineno}: {e.msg}")
        for kw in concepts.get(md.stem, []):
            if kw.lower() not in low:
                issues.append(f"missing required concept: {kw}")
        words = len(text.split())
        print(f"{'PASS' if not issues else 'FAIL'} {md.name} ({words} words)")
        for i in issues:
            print("   -", i)
        problems += len(issues)
    print(f"\nTotal issues: {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
