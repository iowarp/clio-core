#!/usr/bin/env python3
"""Build the CLIO Core static site.

Reads page sources from ``site/pages`` and writes a complete, relocatable
static site (relative links only, so it works under a GitHub Pages project
path such as ``/clio-core/`` and straight from disk).

Each page source starts with a metadata comment::

    <!--
    title: Composing the context filesystem
    description: One sentence for search engines and social cards.
    eyebrow: Tutorial
    lede: The answer, shown under the title.
    facts: Written for | Anyone mounting CLIO ; Time | 15 minutes
    -->

followed by body HTML. ``<pre data-label="bash">`` blocks hold raw text: the
builder escapes and lightly highlights them and adds a copy control. Links
written as ``href="/docs/installation.html"`` are rewritten relative to the
page. ``<h2>``/``<h3>`` headings feed the "On this page" rail.

Usage:
    python3 site/build.py                 # writes site/_site
    python3 site/build.py --out DIR
    python3 site/build.py --serve [PORT]  # build, then serve on localhost
"""

from __future__ import annotations

import argparse
import html
import http.server
import functools
import os
import re
import shutil
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent
PAGES = SITE / "pages"
ASSETS = SITE / "assets"
REPO_URL = "https://github.com/iowarp/clio-core"

# Section navigation: (group title, [(page path, label), ...]).
NAV = {
    "docs": [
        ("Start here", [
            ("docs/index.html", "Introduction"),
            ("docs/installation.html", "Installation"),
            ("docs/basic-usage.html", "Basic usage"),
        ]),
        ("Reference", [
            ("docs/concepts.html", "Core concepts"),
            ("docs/cli.html", "CLI and environment"),
            ("docs/python-api.html", "Python API"),
        ]),
    ],
    "tutorials": [
        ("Overview", [
            ("tutorials/index.html", "All tutorials"),
        ]),
        ("Compose", [
            ("tutorials/compose.html", "Compose local tiers"),
            ("tutorials/operators.html", "Compose data operators"),
            ("tutorials/cloud-storage.html", "Amazon S3 and Google Cloud"),
        ]),
        ("Mount", [
            ("tutorials/fuse-linux.html", "FUSE on Linux"),
            ("tutorials/fuse-macos.html", "FUSE on macOS"),
            ("tutorials/fuse-windows.html", "FUSE on Windows"),
        ]),
        ("Explore", [
            ("tutorials/dashboard.html", "Navigate the dashboard"),
            ("tutorials/python-semantic-query.html", "Query files from Python"),
        ]),
    ],
    "develop": [
        ("Develop", [
            ("develop/index.html", "Custom data operators"),
            ("develop/operator.html", "Write a chain operator"),
            ("develop/extend-storage.html", "Extend the storage system"),
        ]),
    ],
    "design": [
        ("Design", [
            ("design/index.html", "Overview and principles"),
            ("design/runtime.html", "Runtime and block devices"),
            ("design/cte-core.html", "CTE data model and path"),
            ("design/placement.html", "Placement and tiering"),
            ("design/durability.html", "Durability and recovery"),
            ("design/chain.html", "The interposition chain"),
            ("design/filesystem.html", "Filesystem and FUSE"),
            ("design/adapters.html", "I/O adapters"),
        ]),
    ],
}

PRIMARY = [
    ("index.html", "Overview", "overview"),
    ("docs/index.html", "Docs", "docs"),
    ("tutorials/index.html", "Tutorials", "tutorials"),
    ("develop/index.html", "Develop", "develop"),
    ("design/index.html", "Design", "design"),
]

# Old URLs that keep working as redirects.
REDIRECTS = {
    "docs.html": "docs/index.html",
    "tutorials.html": "tutorials/index.html",
    "install/index.html": "docs/installation.html",
    "tutorials/fuse-adapters.html": "tutorials/fuse-linux.html",
}

ICON_GITHUB = ('<svg class="filled" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8'
               'c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94'
               '-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66'
               '.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 '
               '.67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 '
               '2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 '
               '0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>')
ICON_MOON = '<svg class="moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/></svg>'
ICON_SUN = ('<svg class="sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/>'
            '<path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>')
ICON_MENU = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>'

# Applied before first paint so the saved theme never flashes.
THEME_BOOT = ("<script>try{var t=new URLSearchParams(location.search).get('theme')||localStorage.getItem('clio-core-theme');"
              "if(t==='light'||t==='dark')document.documentElement.dataset.theme=t;}catch(e){}"
              "document.documentElement.classList.add('js');</script>")


def parse_page(path: Path) -> tuple[dict, str]:
    """Split a page source into its metadata dict and body HTML.

    Args:
        path: Page source file.

    Returns:
        (metadata, body) where metadata keys are lower-case strings.
    """
    text = path.read_text(encoding="utf-8")
    meta: dict = {}
    m = re.match(r"\s*<!--(.*?)-->", text, re.S)
    if m:
        key = None
        for line in m.group(1).splitlines():
            km = re.match(r"^([a-z_]+):\s?(.*)$", line.strip())
            if km:
                key = km.group(1)
                meta[key] = km.group(2).strip()
            elif key and line.strip():
                meta[key] += " " + line.strip()
        text = text[m.end():]
    return meta, text


def rel_prefix(page: str) -> str:
    """Return the ``../`` prefix that leads from a page back to the site root.

    Args:
        page: Output path relative to the site root, e.g. ``docs/cli.html``.
    """
    return "../" * page.count("/")


def relativize(body: str, page: str) -> str:
    """Rewrite root-absolute ``href``/``src`` values to page-relative ones.

    Args:
        body: HTML to rewrite.
        page: Output path of the page the HTML belongs to.
    """
    prefix = rel_prefix(page)

    def fix(m: re.Match) -> str:
        target = m.group(2)
        if target == "":
            target = "index.html"
        return f'{m.group(1)}="{prefix}{target}'

    return re.sub(r'(?<=\s)(href|src|poster)="/(?!/)([^"]*)', fix, body)


def slugify(text: str) -> str:
    """Make a URL fragment id from heading text."""
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "section"


HL_RULES = {
    "shell": [(r"(^|\n)(\s*#[^\n]*)", r"\1<span class=\"c\">\2</span>"),
              (r"(^|\n)(\$ |PS&gt; |&gt;&gt;&gt; )", r"\1<span class=\"p\">\2</span>")],
    "yaml": [(r"(^|\n)(\s*#[^\n]*)", r"\1<span class=\"c\">\2</span>"),
             (r"(\s#[^\n]*)", r"<span class=\"c\">\1</span>"),
             (r"(^|\n)(\s*(?:- )?)([A-Za-z_][\w.-]*)(:)", r"\1\2<span class=\"k\">\3</span>\4")],
    "python": [(r"(^|\n)(\s*#[^\n]*)", r"\1<span class=\"c\">\2</span>"),
               (r"(  #[^\n]*)", r"<span class=\"c\">\1</span>"),
               (r"\b(import|from|for|in|print|def|return|with|as|if|else)\b", r"<span class=\"k\">\1</span>"),
               (r"(^|\n)(&gt;&gt;&gt; )", r"\1<span class=\"p\">\2</span>")],
}
HL_ALIASES = {"bash": "shell", "sh": "shell", "powershell": "shell", "console": "shell",
              "terminal": "shell", "yaml": "yaml", "yml": "yaml", "python": "python", "py": "python"}


def highlight(code: str, lang: str) -> str:
    """Escape a raw code block and add minimal syntax spans.

    Args:
        code: Raw (unescaped) code text.
        lang: Label given on the block, used to choose highlighting rules.
    """
    out = html.escape(code.strip("\n"), quote=False)
    rules = HL_RULES.get(HL_ALIASES.get(lang.split()[0].lower() if lang else "", ""), [])
    for pattern, repl in rules:
        # Never highlight inside a span that is already there.
        parts = re.split(r"(<span class=\"[a-z]\">.*?</span>)", out, flags=re.S)
        out = "".join(p if p.startswith("<span") else re.sub(pattern, repl, p) for p in parts)
    return out


COPY_BTN = '<button class="copy" type="button" hidden aria-label="Copy code">Copy</button>'


def render_code_blocks(body: str) -> str:
    """Turn ``<pre data-label=...>`` raw blocks into labelled, copyable code.

    Args:
        body: Page body HTML.
    """
    def raw_block(m: re.Match) -> str:
        label = m.group(1)
        code = highlight(m.group(2), label)
        head = f'<div class="code-label"><span>{html.escape(label)}</span>{COPY_BTN}</div>' if label else ""
        tail = "" if label else COPY_BTN
        return f'<div class="code">{head}<pre><code>{code}</code></pre>{tail}</div>'

    # Legacy <pre><code> blocks (already escaped) just get the frame.
    body = re.sub(r'<pre><code>(.*?)</code></pre>',
                  lambda m: f'<div class="code"><pre><code>{m.group(1)}</code></pre>{COPY_BTN}</div>',
                  body, flags=re.S)
    body = re.sub(r'<pre data-label="([^"]*)">(.*?)</pre>', raw_block, body, flags=re.S)
    return body


def wrap_tables(body: str) -> str:
    """Put every table in a horizontally scrollable frame."""
    return re.sub(r'(?<!<div class="table-wrap">)(<table\b.*?</table>)', r'<div class="table-wrap">\1</div>',
                  body, flags=re.S)


def add_heading_ids(body: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Give h2/h3 headings ids and collect them for the page TOC.

    Args:
        body: Page body HTML.

    Returns:
        (body with ids, [(level, id, text), ...]).
    """
    toc: list[tuple[int, str, str]] = []
    seen: set[str] = set()

    def fix(m: re.Match) -> str:
        level, attrs, inner = int(m.group(1)), m.group(2) or "", m.group(3)
        idm = re.search(r'id="([^"]+)"', attrs)
        hid = idm.group(1) if idm else slugify(inner)
        base, n = hid, 2
        while hid in seen and not idm:
            hid, n = f"{base}-{n}", n + 1
        seen.add(hid)
        if not idm:
            attrs = f' id="{hid}"' + attrs
        if "data-notoc" not in attrs:
            toc.append((level, hid, re.sub(r"<[^>]+>", "", inner)))
        return f"<h{level}{attrs}>{inner}</h{level}>"

    body = re.sub(r"<h([23])(\s[^>]*)?>(.*?)</h\1>", fix, body, flags=re.S)
    return body, toc


def header_html(section: str) -> str:
    """Render the sticky site header with primary navigation."""
    links = "".join(
        f'<a href="/{href}"{" aria-current=\"page\"" if key == section else ""}>{label}</a>'
        for href, label, key in PRIMARY)
    return f"""<header class="site-header">
  <div class="header-inner">
    <a class="brand" href="/index.html" aria-label="CLIO Core home">
      <img class="mark-dark" src="/assets/brand/clio-core-mark.svg" width="30" height="30" alt="">
      <img class="mark-light" src="/assets/brand/clio-core-mark-light.svg" width="30" height="30" alt="">
      <span class="brand-name">CLIO <em>Core</em></span>
    </a>
    <nav class="primary-nav" id="primary-nav" aria-label="Primary">{links}</nav>
    <div class="header-tools">
      <a class="icon-btn" href="{REPO_URL}" aria-label="CLIO Core on GitHub">{ICON_GITHUB}</a>
      <button class="icon-btn theme-toggle" type="button" aria-label="Switch color theme">{ICON_MOON}{ICON_SUN}</button>
      <button class="icon-btn menu-toggle" type="button" aria-expanded="false" aria-controls="primary-nav" aria-label="Menu">{ICON_MENU}</button>
    </div>
  </div>
</header>"""


def footer_html() -> str:
    """Render the site footer."""
    return f"""<footer class="site-footer">
  <div class="footer-inner">
    <div class="footer-brand">
      <a class="brand" href="/index.html"><img class="mark-dark" src="/assets/brand/clio-core-mark.svg" width="30" height="30" alt=""><img class="mark-light" src="/assets/brand/clio-core-mark-light.svg" width="30" height="30" alt=""><span class="brand-name">CLIO <em>Core</em></span></a>
      <p>Tiered storage for scientific data, mounted as an ordinary filesystem and searchable from Python. Part of the Clio ecosystem.</p>
    </div>
    <div><h2>Docs</h2><ul>
      <li><a href="/docs/installation.html">Installation</a></li>
      <li><a href="/docs/basic-usage.html">Basic usage</a></li>
      <li><a href="/docs/concepts.html">Core concepts</a></li>
      <li><a href="/docs/python-api.html">Python API</a></li>
    </ul></div>
    <div><h2>Tutorials</h2><ul>
      <li><a href="/tutorials/compose.html">Compose tiers</a></li>
      <li><a href="/tutorials/fuse-linux.html">Mount with FUSE</a></li>
      <li><a href="/tutorials/dashboard.html">Dashboard</a></li>
      <li><a href="/tutorials/python-semantic-query.html">Semantic query</a></li>
    </ul></div>
    <div><h2>Project</h2><ul>
      <li><a href="{REPO_URL}">Source on GitHub</a></li>
      <li><a href="{REPO_URL}/issues">Report an issue</a></li>
      <li><a href="https://clio.iowarp.ai">Clio Agent</a></li>
      <li><a href="https://coder.iowarp.ai">Clio Coder</a></li>
      <li><a href="https://iowarp.ai">Part of IOWarp</a></li>
    </ul></div>
  </div>
  <div class="footer-legal">
    <span>Copyright 2026 iowarp.ai · <a href="{REPO_URL}/blob/main/LICENSE">BSD 3-Clause License</a></span>
    <a href="https://grc.iit.edu">Developed by Gnosis Research Center</a>
  </div>
</footer>"""


def head_html(meta: dict, page: str) -> str:
    """Render the document head."""
    title = meta.get("title", "CLIO Core")
    full = title if page == "index.html" else f"{title} · CLIO Core"
    desc = html.escape(meta.get("description", ""), quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(full)}</title>
<meta name="description" content="{desc}">
<meta property="og:title" content="{html.escape(full, quote=True)}">
<meta property="og:description" content="{desc}">
<meta name="theme-color" content="#000000">
<link rel="icon" href="/assets/brand/clio-core-mark.svg" type="image/svg+xml">
<link rel="preload" href="/assets/fonts/plex-sans.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="/assets/fonts/news-normal-500.woff2" as="font" type="font/woff2" crossorigin>
<link rel="stylesheet" href="/assets/css/site.css">
{THEME_BOOT}
</head>"""


def side_nav_html(section: str, page: str) -> str:
    """Render the left-hand section navigation."""
    groups = []
    for title, items in NAV[section]:
        lis = "".join(
            f'<li><a href="/{href}"{" aria-current=\"page\"" if href == page else ""}>{label}</a></li>'
            for href, label in items)
        groups.append(f"<h2>{title}</h2><ul>{lis}</ul>")
    label = section.capitalize()
    return (f'<nav class="side-nav" aria-label="{label}"><details open><summary>{label} menu</summary>'
            f'<div>{"".join(groups)}</div></details></nav>')


def toc_html(toc: list[tuple[int, str, str]]) -> str:
    """Render the right-hand "On this page" rail."""
    if len(toc) < 2:
        return '<aside class="page-toc" aria-hidden="true"></aside>'
    lis = "".join(f'<li class="{"sub" if lvl == 3 else ""}"><a href="#{hid}">{text}</a></li>'
                  for lvl, hid, text in toc if lvl == 2 or len(toc) < 14)
    return f'<aside class="page-toc"><nav aria-label="On this page"><h2>On this page</h2><ul>{lis}</ul></nav></aside>'


def neighbors(section: str, page: str) -> tuple:
    """Find the previous and next pages in a section's reading order."""
    flat = [item for _, items in NAV[section] for item in items]
    idx = next((i for i, (href, _) in enumerate(flat) if href == page), None)
    if idx is None:
        return None, None
    prev = flat[idx - 1] if idx > 0 else None
    nxt = flat[idx + 1] if idx + 1 < len(flat) else None
    return prev, nxt


def page_nav_html(section: str, page: str) -> str:
    """Render previous / next links at the foot of a guide."""
    prev, nxt = neighbors(section, page)
    parts = []
    if prev:
        parts.append(f'<a class="prev" href="/{prev[0]}"><span class="dir">← Previous</span><span class="title">{prev[1]}</span></a>')
    if nxt:
        parts.append(f'<a class="next" href="/{nxt[0]}"><span class="dir">Next →</span><span class="title">{nxt[1]}</span></a>')
    return f'<nav class="page-nav" aria-label="Pagination">{"".join(parts)}</nav>' if parts else ""


def facts_html(meta: dict) -> str:
    """Render the ruled row of facts under a guide title."""
    if not meta.get("facts"):
        return ""
    items = []
    for pair in meta["facts"].split(";"):
        if "|" in pair:
            k, v = pair.split("|", 1)
            items.append(f"<div><dt>{k.strip()}</dt><dd>{v.strip()}</dd></div>")
    return f'<dl class="facts">{"".join(items)}</dl>'


def render_guide(meta: dict, body: str, section: str, page: str) -> str:
    """Wrap a Docs/Tutorials/Design page in the three-column guide layout."""
    body, toc = add_heading_ids(body)
    lede = f'<p class="lede">{meta["lede"]}</p>' if meta.get("lede") else ""
    eyebrow = f'<p class="eyebrow">{meta["eyebrow"]}</p>' if meta.get("eyebrow") else ""
    return f"""<div class="guide">
  {side_nav_html(section, page)}
  <article class="article">
    <header>{eyebrow}<h1>{meta.get("heading", meta.get("title", ""))}</h1>{lede}{facts_html(meta)}</header>
    <div class="prose">
{body}
    </div>
    {page_nav_html(section, page)}
  </article>
  {toc_html(toc)}
</div>"""


def render_page(src: Path, out_root: Path) -> str:
    """Render one page source into the output tree.

    Args:
        src: Page source under ``site/pages``.
        out_root: Output directory.

    Returns:
        The output path relative to the site root.
    """
    page = src.relative_to(PAGES).as_posix()
    meta, body = parse_page(src)
    section = page.split("/")[0] if "/" in page else "overview"
    body = wrap_tables(render_code_blocks(body))
    if section in NAV and meta.get("layout") != "landing":
        main = render_guide(meta, body, section, page)
    else:
        main = body
    doc = (head_html(meta, page) +
           f'\n<body class="{section}">\n<a class="skip" href="#content">Skip to content</a>\n' +
           header_html(section) + f'\n<main id="content">\n{main}\n</main>\n' + footer_html() +
           '\n<script src="/assets/js/site.js" defer></script>\n</body>\n</html>\n')
    out = out_root / page
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(relativize(doc, page), encoding="utf-8")
    return page


def write_redirect(out_root: Path, old: str, new: str) -> None:
    """Write a small HTML redirect so an old URL keeps working."""
    target = rel_prefix(old) + new
    out = out_root / old
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f'<!doctype html><meta charset="utf-8"><title>Moved</title>'
        f'<meta http-equiv="refresh" content="0; url={target}">'
        f'<link rel="canonical" href="{target}"><p>This page moved to <a href="{target}">{new}</a>.</p>\n',
        encoding="utf-8")


def check_links(out_root: Path) -> list[str]:
    """Return a list of broken internal links in the built site."""
    broken = []
    for f in out_root.rglob("*.html"):
        text = f.read_text(encoding="utf-8")
        ids = set(re.findall(r'\sid="([^"]+)"', text))
        for target in re.findall(r'\s(?:href|src)="([^"]+)"', text):
            if re.match(r"^(https?:|mailto:|data:)", target):
                continue
            path, _, frag = target.partition("#")
            if not path:
                if frag and frag not in ids:
                    broken.append(f"{f.relative_to(out_root)} -> #{frag}")
                continue
            dest = (f.parent / path.split("?")[0]).resolve()
            if not dest.exists():
                broken.append(f"{f.relative_to(out_root)} -> {target}")
            elif frag and dest.suffix == ".html":
                if f'id="{frag}"' not in dest.read_text(encoding="utf-8"):
                    broken.append(f"{f.relative_to(out_root)} -> {target}")
    return broken


def build(out_root: Path) -> int:
    """Build the whole site into ``out_root``.

    Returns:
        Process exit status: 0 on success, 1 if any internal link is broken.
    """
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    shutil.copytree(ASSETS, out_root / "assets")
    pages = [render_page(src, out_root) for src in sorted(PAGES.rglob("*.html"))]
    for old, new in REDIRECTS.items():
        write_redirect(out_root, old, new)
    (out_root / ".nojekyll").write_text("", encoding="utf-8")
    broken = check_links(out_root)
    print(f"Built {len(pages)} pages and {len(REDIRECTS)} redirects into {out_root}")
    for b in broken:
        print(f"  broken link: {b}", file=sys.stderr)
    return 1 if broken else 0


def main() -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=SITE / "_site", help="output directory")
    ap.add_argument("--serve", nargs="?", const=8000, type=int, metavar="PORT",
                    help="serve the built site on localhost after building")
    args = ap.parse_args()
    status = build(args.out.resolve())
    if args.serve is not None:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(args.out))
        print(f"Serving on http://127.0.0.1:{args.serve}/  (Ctrl+C to stop)")
        http.server.ThreadingHTTPServer(("127.0.0.1", args.serve), handler).serve_forever()
    return status


if __name__ == "__main__":
    sys.exit(main())
