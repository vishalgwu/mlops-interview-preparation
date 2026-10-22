"""Generate site/data.js (+ site/index.html from the template) and copy chapters and PDFs into site/.

Config: scripts/site_meta.json
  {"config": {...brand/year/storage/repo/...}, "chapters": {"1": {"title","tagline","level","genres","cast"}},
   "rows": [{"id","title","chapters":[...]}], "top10": [...], "featured": [...], "profiles": [...]}

Run after audit / build_pdfs:  python scripts/build_site_data.py
"""
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mdutil import split_sections  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
META = json.loads((ROOT / "scripts" / "site_meta.json").read_text(encoding="utf-8"))
CFG = META["config"]
ROADMAP_REPO = "vishalgwu/ultimate-ai-engineering-roadmap-2026"


def sections_of(text: str, chapter: int):
    out = []
    for head, body in split_sections(text):
        m = re.match(r"(\d+)(?:\.(\d+))?\.?\s+(.+)$", head.strip())
        if not m:
            continue                      # Table of Contents and other unnumbered headings
        num = f"{m.group(1)}.{m.group(2)}" if m.group(2) else f"{chapter}.{m.group(1)}"
        out.append({"num": num, "title": m.group(3).strip(), "words": len(body.split())})
    return out


def patch_index():
    tpl = SITE / "index.template.html"
    html = tpl.read_text(encoding="utf-8")
    repo_path = CFG["repo"].replace("https://github.com/", "")
    html = html.replace(ROADMAP_REPO, repo_path)
    title = f"{CFG['title']} - Binge the Series"
    html = re.sub(r"<title>.*?</title>", f"<title>{title}</title>", html, flags=re.S)
    desc = CFG.get("description", CFG["title"])
    html = re.sub(r'(<meta name="description" content=")[^"]*(")', lambda m: m.group(1) + desc + m.group(2), html)
    html = re.sub(r'(<meta property="og:title" content=")[^"]*(")', lambda m: m.group(1) + CFG["title"] + m.group(2), html)
    html = re.sub(r'(<meta property="og:description" content=")[^"]*(")', lambda m: m.group(1) + desc + m.group(2), html)
    html = html.replace(">AI ENGINEERING<", f">{CFG['brand']}<")
    (SITE / "index.html").write_text(html, encoding="utf-8", newline="\n")


def main() -> None:
    (SITE / "chapters").mkdir(parents=True, exist_ok=True)
    (SITE / "pdfs").mkdir(parents=True, exist_ok=True)
    for old in (SITE / "chapters").glob("*.md"):
        old.unlink()
    if (ROOT / "assets").exists():
        shutil.copytree(ROOT / "assets", SITE / "assets", dirs_exist_ok=True)
    chapters, concepts = [], []
    for md in sorted((ROOT / "chapters").glob("*.md")):
        num = int(md.name[:2])
        text = md.read_text(encoding="utf-8")
        meta = META["chapters"].get(str(num), {})
        h1 = re.search(r"^# (.+)$", text, flags=re.M)
        title = meta.get("title") or (h1.group(1).strip() if h1 else md.stem)
        secs = sections_of(text, num)
        for s in secs:
            concepts.append({"chapter": num, "num": s["num"], "title": s["title"]})
        words = len(text.split())
        # chapters live in chapters/, the site serves them from site/chapters/ with assets in site/assets/
        site_text = text.replace("(../assets/", "(assets/").replace('src="../assets/', 'src="assets/')
        (SITE / "chapters" / md.name).write_text(site_text, encoding="utf-8", newline="\n")
        pdf = ROOT / "pdfs" / (md.stem + ".pdf")
        if pdf.exists():
            shutil.copyfile(pdf, SITE / "pdfs" / pdf.name)
        chapters.append({
            "num": num, "slug": md.stem, "title": title,
            "tagline": meta.get("tagline", ""), "level": meta.get("level", "All levels"),
            "genres": meta.get("genres", []), "cast": meta.get("cast", []),
            "sections": secs, "words": words, "minutes": max(1, round(words / 220)),
            "codeBlocks": len(re.findall(r"```(?!\s*$)\w+", text)),
            "md": f"chapters/{md.name}", "pdf": f"pdfs/{md.stem}.pdf",
        })
    nums = {c["num"] for c in chapters}
    rows = [{**r, "chapters": [n for n in r["chapters"] if n in nums]} for r in META["rows"]]
    data = {
        "config": CFG, "chapters": chapters, "rows": rows,
        "top10": [n for n in META["top10"] if n in nums], "featured": [n for n in META["featured"] if n in nums],
        "profiles": [{**p, "rec": [n for n in p["rec"] if n in nums]} for p in META["profiles"]],
        "concepts": concepts,
        "totals": {"chapters": len(chapters), "sections": len(concepts),
                   "words": sum(c["words"] for c in chapters), "code": sum(c["codeBlocks"] for c in chapters)},
    }
    (SITE / "data.js").write_text("window.ROADMAP = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    patch_index()
    print(f"site data: {data['totals']}")


if __name__ == "__main__":
    main()
