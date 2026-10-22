# Sinh lại index.html sau khi sửa markdown:
#   uv run --no-project --with markdown python api-docs/build_html.py api-docs
"""Build api-docs/index.html from the markdown files (single self-contained page)."""
import html, re, sys
from pathlib import Path
import markdown

docs = Path(sys.argv[1])
pages = [("overview", "Giới thiệu & quy ước", docs / "README.md"),
         ("system-overview", "Tổng quan hệ thống", docs / "system-overview.md"),
         ("database-schema", "Database schema", docs / "database-schema.md")]
for f in sorted((docs / "endpoints").glob("*.md")):
    title = f.read_text().splitlines()[0].lstrip("# ").strip()
    pages.append((f"ep-{f.stem}", title, f))
anchor = {p.name: slug for slug, _, p in pages}

def fix_links(text):
    def repl(m):
        label, target = m.group(1), m.group(2)
        name = target.split("#")[0].split("/")[-1]
        if name in anchor:
            return f"[{label}](#{anchor[name]})"
        if name == "index.html":
            return label
        return m.group(0)
    return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", repl, text)

md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"])
sections, nav = [], {"Tài liệu": [], "System": [], "Predictions": [], "Batches": []}
for slug, title, path in pages:
    body = md.reset().convert(fix_links(path.read_text()))
    sections.append(f'<section id="{slug}" class="page">{body}</section>')
    m = re.match(r"(GET|POST|PUT|DELETE)\s+(\S+)", title)
    if m:
        method, route = m.groups()
        group = "System" if route in ("/health", "/model") else ("Batches" if route.startswith("/batches") else "Predictions")
        nav[group].append(f'<a href="#{slug}"><span class="m m-{method.lower()}">{method}</span><code>{html.escape(route)}</code></a>')
    else:
        nav["Tài liệu"].append(f'<a href="#{slug}">{html.escape(title)}</a>')
nav_html = "".join(f"<h4>{g}</h4>" + "".join(items) for g, items in nav.items())

page = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PlateReader API Docs</title>
<style>
:root{{--bg:#fbfbfa;--panel:#f2f1ee;--text:#1d1d1b;--muted:#64625d;--line:#dedbd4;--code:#efede8;--accent:#2f5d8a;
--get:#2f7d4f;--post:#2f5d8a;--put:#9a6a12;--delete:#a63d3d}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#161615;--panel:#1e1e1c;--text:#e9e7e2;--muted:#a19e97;--line:#34332f;--code:#262522;--accent:#86b4e0;--get:#6cc08f;--post:#86b4e0;--put:#e0b35c;--delete:#e48a8a}}}}
:root[data-theme="dark"]{{--bg:#161615;--panel:#1e1e1c;--text:#e9e7e2;--muted:#a19e97;--line:#34332f;--code:#262522;--accent:#86b4e0;--get:#6cc08f;--post:#86b4e0;--put:#e0b35c;--delete:#e48a8a}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}}
.layout{{display:grid;grid-template-columns:320px minmax(0,1fr);min-height:100vh}}
nav{{position:sticky;top:0;height:100vh;overflow:auto;background:var(--panel);border-right:1px solid var(--line);padding:20px 14px}}
nav .brand{{font-weight:700;font-size:16px;margin:0 6px 4px}} nav .sub{{color:var(--muted);font-size:12px;margin:0 6px 12px}}
nav h4{{margin:18px 6px 6px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}
nav a{{display:flex;gap:8px;align-items:center;padding:5px 6px;border-radius:6px;color:var(--text);text-decoration:none;font-size:13.5px}}
nav a:hover{{background:var(--code)}} nav a code{{background:none;padding:0;font-size:12.5px}}
.m{{flex:0 0 52px;font:700 10.5px/1 ui-monospace,Menlo,monospace;text-align:center;padding:4px 0;border-radius:4px;color:#fff}}
.m-get{{background:var(--get)}} .m-post{{background:var(--post)}} .m-put{{background:var(--put)}} .m-delete{{background:var(--delete)}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]) .m{{color:#111}}}} :root[data-theme="dark"] .m{{color:#111}}
main{{padding:28px 40px 80px;max-width:1080px}}
.page{{border-bottom:1px solid var(--line);padding:8px 0 36px;margin-bottom:24px;scroll-margin-top:12px}}
h1{{font-size:26px;margin:16px 0 10px}} h2{{font-size:19px;margin:28px 0 8px}} h3{{font-size:16px;margin:20px 0 6px}}
a{{color:var(--accent)}}
code{{font:13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--code);padding:1px 5px;border-radius:4px}}
pre{{background:var(--code);border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow:auto;line-height:1.45}}
pre code{{background:none;padding:0;font-size:12.5px}}
table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:13.5px;display:block;overflow-x:auto}}
th,td{{border:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top}}
th{{background:var(--panel)}}
blockquote{{margin:10px 0;padding:6px 14px;border-left:3px solid var(--accent);color:var(--muted);background:var(--panel);border-radius:0 6px 6px 0}}
.toggle{{position:fixed;top:12px;right:14px;border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:6px;padding:4px 10px;cursor:pointer;font-size:12px}}
@media (max-width:820px){{.layout{{grid-template-columns:1fr}} nav{{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line)}} main{{padding:16px}}}}
</style></head>
<body><button class="toggle" onclick="var r=document.documentElement;r.dataset.theme=(r.dataset.theme==='dark'||(!r.dataset.theme&&matchMedia('(prefers-color-scheme: dark)').matches))?'light':'dark'">Sáng / Tối</button>
<div class="layout"><nav><div class="brand">PlateReader LR API</div><div class="sub">v1.0.0 · base <code>/api/v1</code></div>{nav_html}</nav>
<main>{''.join(sections)}</main></div></body></html>"""
(docs / "index.html").write_text(page)
print("ok", len(pages), "pages", len(page)//1024, "KB")
