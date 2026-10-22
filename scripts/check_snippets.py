"""Run each chapter section's Python blocks (shared namespace) in a clean temp dir and report failures.

Only sections whose imports are limited to the standard library (+ numpy) are executed; sections that need
third-party packages, GPUs, servers or network are skipped (they are still syntax-checked by audit.py).
"""
import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mdutil import split_sections  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = set(sys.stdlib_module_names) | {"numpy"}
NETWORK_HINTS = ("urlopen", "uvicorn", "requests.", "httpx.", "asyncio.run(main", "input(")


def main() -> int:
    ok, skip, fails = 0, 0, []
    for md in sorted((ROOT / "chapters").glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for title, sec in split_sections(text):
            blocks = re.findall(r"```python\n(.*?)```", sec, flags=re.S)
            if not blocks:
                continue
            code = "\n".join(blocks)
            fut = [ln for ln in code.split("\n") if ln.startswith("from __future__")]
            if fut:                      # concatenated blocks: future imports must come first
                code = "\n".join(fut) + "\n" + "\n".join(ln for ln in code.split("\n") if not ln.startswith("from __future__"))
            mods = set()
            try:                         # parse imports properly: handles "import a, b" and imports inside functions
                for node in ast.walk(ast.parse(code)):
                    if isinstance(node, ast.Import):
                        mods |= {a.name.split(".")[0] for a in node.names}
                    elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                        mods.add(node.module.split(".")[0])
            except SyntaxError:
                mods = set(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", code, flags=re.M))
            local = set(re.findall(r"^\s*(?:def|class)\s+(\w+)", code, flags=re.M))
            if (mods - ALLOWED - local) or any(h in code for h in NETWORK_HINTS):
                skip += 1
                continue
            with tempfile.TemporaryDirectory() as d:
                try:
                    p = subprocess.run([sys.executable, "-I", "-c", code], cwd=d, capture_output=True,
                                       text=True, timeout=25)
                except subprocess.TimeoutExpired:
                    fails.append((md.name, title, "TIMEOUT"))
                    continue
            if p.returncode == 0:
                ok += 1
            else:
                last = (p.stderr.strip().splitlines() or ["?"])[-1][:160]
                fails.append((md.name, title, last))
    print(f"executed ok={ok} skipped(third-party/network)={skip} failed={len(fails)}")
    for f in fails:
        print(" FAIL", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
