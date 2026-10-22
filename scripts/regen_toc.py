"""Regenerate the '**Contents**' list of each chapter that has one.

The numbered legacy entries are rebuilt from the '## k. Title' headings; explicit '## N.M Title' sections
(extensions) are listed after them as bullets. Anchors follow GitHub's slug rules. Idempotent.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mdutil import split_sections  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def slug(heading: str) -> str:
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)      # drop punctuation (keeps letters, digits, _ and -)
    return re.sub(r"\s", "-", s)


def main() -> None:
    changed = 0
    for md in sorted((ROOT / "chapters").glob("*.md")):
        text = md.read_text(encoding="utf-8")
        lines = text.split("\n")
        try:
            start = next(i for i, ln in enumerate(lines) if ln.strip() == "**Contents**")
        except StopIteration:
            continue
        end = start + 1
        while end < len(lines) and (re.match(r"^(\d+\.|-)\s+\[", lines[end]) or not lines[end].strip() and end == start + 1):
            end += 1
        legacy, ext = [], []
        for head, _ in split_sections(text):
            m = re.match(r"(\d+)\.\s+(.+)$", head)
            if m:
                legacy.append(f"{m.group(1)}. [{m.group(2)}](#{slug(head)})")
                continue
            if re.match(r"\d+\.\d+\s", head):
                ext.append(f"- [{head}](#{slug(head)})")
        block = legacy + ext
        new_lines = lines[:start + 1] + block + lines[end:]
        if new_lines != lines:
            md.write_text("\n".join(new_lines), encoding="utf-8", newline="\n")
            changed += 1
            print(f"{md.name}: {len(legacy)} legacy + {len(ext)} extension entries")
    print(f"updated {changed} chapter(s)")


if __name__ == "__main__":
    main()
