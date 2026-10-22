"""Append extensions/NN.md, NNb.md, ... to the matching chapters/NN_*.md (idempotent).

An extension is skipped if its first '## ' heading already exists in the chapter,
so re-running never duplicates content.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    for ext in sorted((ROOT / "extensions").glob("*.md")):
        nn = ext.stem[:2]           # 07.md, 07b.md, 07c.md ... all extend chapter 07
        targets = list((ROOT / "chapters").glob(f"{nn}_*.md"))
        if len(targets) != 1:
            print(f"SKIP {ext.name}: no unique chapter")
            continue
        body = ext.read_text(encoding="utf-8")
        first = re.search(r"^## .*$", body, flags=re.M)
        chapter = targets[0].read_text(encoding="utf-8")
        if first and first.group(0) in chapter:
            print(f"already merged {ext.name}")
            continue
        marker = "<!-- deep:end -->"           # chapters with a nav footer: insert new sections before it
        if marker in chapter:
            i = chapter.rindex(marker)
            merged = chapter[:i].rstrip() + "\n" + body.rstrip() + "\n\n" + chapter[i:]
        else:
            merged = chapter.rstrip() + "\n" + body
        targets[0].write_text(merged, encoding="utf-8", newline="\n")
        print(f"merged {ext.name} -> {targets[0].name}")


if __name__ == "__main__":
    main()
