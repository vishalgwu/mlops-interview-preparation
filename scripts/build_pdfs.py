"""Compile every chapters/*.md into a styled PDF in pdfs/ (markdown + xhtml2pdf, pure Python)."""
import html
import json
import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = ROOT / "chapters"
PDFS = ROOT / "pdfs"
CFG = json.loads((ROOT / "scripts" / "project.json").read_text(encoding="utf-8"))
FOOTER = CFG.get("pdf_footer", CFG.get("title", "Guide"))

FONT_DIR = Path("C:/Windows/Fonts")
# Built-in PDF fonts lack Greek/math glyphs; embed Unicode TTFs when available (Windows).
FONTS = {"body": ("arial.ttf", "Helvetica"), "mono": ("consola.ttf", "Courier")}
# Glyphs missing even from Arial/Consolas -> ASCII equivalents.
ASCII_FALLBACK = {"ᵀ": "^T", "ⁿ": "^n", "₁": "1", "₂": "2", "ℝ": "R", "‖": "||", "▁": "_", "²": "^2"}


def font_css() -> str:
    css, fam = "", {}
    for key, (fname, default) in FONTS.items():
        p = FONT_DIR / fname
        if p.exists():
            fam[key] = f"{key}u"
            css += f"@font-face {{ font-family: {key}u; src: url('{p.as_posix()}'); }}\n"
        else:
            fam[key] = default
    return css + f"body {{ font-family: {fam['body']}; }} code, pre {{ font-family: {fam['mono']}; }}\n"


CSS = """
@page { size: A4; margin: 2cm 1.8cm; @frame footer { -pdf-frame-content: foot; bottom: 0.6cm; height: 0.8cm; left: 1.8cm; right: 1.8cm; } }
body { font-family: Helvetica; font-size: 10pt; line-height: 1.35; color: #1b1f24; }
h1 { font-size: 22pt; color: #b20710; border-bottom: 2px solid #b20710; padding-bottom: 4px; }
h2 { font-size: 15pt; color: #b20710; margin-top: 18px; }
h3 { font-size: 12pt; color: #333; margin-top: 12px; }
h4 { font-size: 10.5pt; color: #333; }
code { font-family: Courier; font-size: 8.5pt; background-color: #f0f2f5; }
pre { font-family: Courier; font-size: 8pt; background-color: #f6f8fa; border: 1px solid #d0d7de; padding: 6px; }
table { border: 1px solid #999; width: 100%; }
th { background-color: #f3e3e4; border: 1px solid #999; padding: 3px; font-size: 9pt; }
td { border: 1px solid #bbb; padding: 3px; font-size: 9pt; }
blockquote { border-left: 3px solid #b20710; margin-left: 4px; padding-left: 8px; color: #444; }
"""


def render(md_path: Path, pdf_path: Path) -> bool:
    text = md_path.read_text(encoding="utf-8")
    for src, dst in ASCII_FALLBACK.items():
        text = text.replace(src, dst)
    body = markdown.markdown(text, extensions=["fenced_code", "tables", "sane_lists"])
    out, in_pre = [], False
    for line in body.split("\n"):          # xhtml2pdf does not wrap long code lines
        if "<pre>" in line:
            in_pre = True
        if in_pre and len(line) > 105:
            line = "\n".join(line[i:i + 105] for i in range(0, len(line), 105))
        if "</pre>" in line:
            in_pre = False
        out.append(line)
    body = "\n".join(out).replace("<pre><code>", "<pre>").replace("</code></pre>", "</pre>")
    doc = (f"<html><head><meta charset='utf-8'><style>{CSS}{font_css()}</style></head><body>{body}"
           f"<div id='foot'>{html.escape(FOOTER)} - {html.escape(md_path.stem)} - page <pdf:pagenumber/></div></body></html>")
    with open(pdf_path, "wb") as fh:
        result = pisa.CreatePDF(doc, dest=fh, encoding="utf-8")
    return not result.err


def main() -> int:
    PDFS.mkdir(exist_ok=True)
    failed = 0
    for md in sorted(CHAPTERS.glob("*.md")):
        pdf = PDFS / (md.stem + ".pdf")
        ok = render(md, pdf)
        print(("OK   " if ok else "FAIL ") + pdf.name)
        failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
