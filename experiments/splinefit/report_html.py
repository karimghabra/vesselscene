"""The report (experiments/REPORT.md) as one self-contained HTML page: the figures are embedded as data URIs.

    python -m experiments.splinefit.report_html experiments/REPORT.md report.html

Needs the ``markdown`` package (pip install markdown).  Nested lists use 2-space indents, as the READMEs do.
"""
from __future__ import annotations

import base64
import html
import os
import re
import sys

CSS = """
:root { --bg: #ffffff; --fg: #1d2127; --muted: #5b6573; --rule: #d9dee5; --head: #f3f5f8; --accent: #1f6feb;
        --code: #f3f5f8; --cap: #48515c; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
        --bg: #0f1216; --fg: #e3e7ec; --muted: #9aa5b1; --rule: #2c333c; --head: #181d23; --accent: #6aa8ff;
        --code: #181d23; --cap: #b3bcc6; } }
:root[data-theme="dark"] { --bg: #0f1216; --fg: #e3e7ec; --muted: #9aa5b1; --rule: #2c333c; --head: #181d23;
        --accent: #6aa8ff; --code: #181d23; --cap: #b3bcc6; }
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--fg); }
body { font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.65rem; line-height: 1.25; margin: 0.2em 0 0.6em; }
h2 { font-size: 1.3rem; margin: 2.2em 0 0.6em; padding-top: 0.6em; border-top: 1px solid var(--rule); }
h3 { font-size: 1.08rem; margin: 1.8em 0 0.5em; }
a { color: var(--accent); }
p, li { max-width: 78ch; }
ul, ol { padding-left: 1.4em; }
li { margin: 0.2em 0; }
code { background: var(--code); padding: 0.05em 0.3em; border-radius: 4px; font-size: 0.92em; }
pre { background: var(--code); padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.85em; }
pre code { background: none; padding: 0; }
.table-wrap { overflow-x: auto; margin: 0.8em 0 1.2em; }
table { border-collapse: collapse; font-size: 0.86em; font-variant-numeric: tabular-nums; }
th, td { border: 1px solid var(--rule); padding: 4px 8px; text-align: left; white-space: nowrap; }
th { background: var(--head); }
td:first-child, th:first-child { white-space: normal; min-width: 9em; }
td:last-child, th:last-child { white-space: normal; min-width: 12em; }
td.long { white-space: normal; min-width: 14em; }
figure { margin: 1em 0 0.4em; }
figure img { display: block; width: 100%; height: auto; border-radius: 4px; background: #fff; }
p.caption, p > em:only-child { color: var(--cap); }
p.caption { font-size: 0.92em; margin-top: 0.3em; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 12px; }
.gallery figure { margin: 0; }
.gallery figcaption { font-size: 0.85em; color: var(--muted); }
"""


def _data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "svg": "image/svg+xml"}[ext]
    with open(path, "rb") as fh:
        return f"data:{mime};base64," + base64.b64encode(fh.read()).decode()


def _list_breaks(text: str) -> str:
    """A blank line before a list that follows a top-level paragraph line: GitHub starts a list there,
    python-markdown does not."""
    item = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")
    out, fence = [], False
    for ln in text.splitlines():
        if ln.startswith("```"):
            fence = not fence
        prev = out[-1] if out else ""
        if (not fence and item.match(ln) and prev.strip() and not prev[0].isspace() and not item.match(prev)
                and not prev.startswith(("|", "#", "<"))):
            out.append("")
        out.append(ln)
    return "\n".join(out) + "\n"


def build(md_path: str, out_path: str):
    import markdown
    base = os.path.dirname(os.path.abspath(md_path))
    with open(md_path) as fh:
        text = fh.read()
    title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), "Report")
    text = _list_breaks(text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"], tab_length=2)

    body = re.sub(r'<p>(<img [^>]*>)</p>', r"<figure>\1</figure>", body)
    body = re.sub(r'src="(?!data:|https?:)([^"]+)"', lambda m: f'src="{_data_uri(os.path.join(base, m.group(1)))}"',
                  body)
    body = re.sub(r"<p><em>(.*?)</em></p>", r'<p class="caption"><em>\1</em></p>', body, flags=re.S)
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    # long cells wrap wherever they are (short numeric cells keep nowrap)
    body = re.sub(r"<td>(.*?)</td>", lambda m: (f'<td class="long">{m.group(1)}</td>'
                                                if len(re.sub(r"<[^>]+>", "", m.group(1))) > 40 else m.group(0)),
                  body, flags=re.S)
    # links to other markdown files of the repo are not part of this page: keep their text only
    body = re.sub(r'<a href="(?!https?:)[^"]*">(.*?)</a>', r"\1", body)
    page = (f"<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body><main>\n{body}\n</main></body></html>\n")
    with open(out_path, "w") as fh:
        fh.write(page)
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    build(sys.argv[1], sys.argv[2])
