# CLIO Core website

The public website for CLIO Core: **Overview**, **Docs** (installation, basic
usage, concepts, CLI, Python API), **Tutorials** (composing the context
filesystem from RAM/NVMe/SSD/disk and S3/GCS, FUSE on Linux/macOS/Windows, the
dashboard, Python semantic query, AI agents over MCP) and **Design** (how the
storage stack works).

It follows the editorial style of the sibling Clio sites
([clio-coder](https://github.com/iowarp/clio-coder) and
[clio-agent](https://github.com/iowarp/clio-agent)): black and warm ivory,
cyan accent, copper secondary, Newsreader headings, IBM Plex text and code,
dark by default with a light theme. It is separate from, and narrower than,
the Docusaurus documentation in the [`docs/`](../docs) submodule.

## Build and preview

No dependencies beyond Python 3.8+:

```bash
python3 site/build.py            # writes site/_site and checks every internal link
python3 site/build.py --serve    # build, then serve on http://127.0.0.1:8000/
```

`.github/workflows/pages.yml` runs the same build on pull requests and deploys
`main` to GitHub Pages. Enable it once under Settings → Pages → Source:
"GitHub Actions". All links are relative, so the site works at
`https://iowarp.github.io/clio-core/`, on a custom domain, or from disk.

## Layout

```
site/
  build.py              Wraps pages in the shared layout, writes _site/, checks links
  pages/                Page sources (one HTML fragment per page)
    index.html          Overview (landing layout)
    docs/               Introduction, installation, basic usage, concepts, CLI, Python API
    tutorials/          Compose, cloud storage, FUSE x3, dashboard, Python query, AI agents
    design/             Runtime, CTE, placement, durability, chain, filesystem, adapters
  assets/
    css/site.css        All styling; colors only through semantic variables
    js/site.js          Theme toggle, phone menu, copy buttons, tabs, TOC highlight
    brand/              CLIO Core mark (dark and light) plus Clio Agent, Clio Coder, IOWarp marks
    fonts/              Self-hosted Newsreader and IBM Plex, with their OFL licenses
```

## Writing a page

Each file in `pages/` starts with a metadata comment, then body HTML:

```html
<!--
title: FUSE on Linux
heading: Mount on <em>Linux.</em>
description: One sentence for search results and link previews.
eyebrow: Tutorial · Mount
lede: The answer, in one or two sentences, under the title.
facts: Time | 10 minutes ; Driver | libfuse3
-->
<h2>1. Install FUSE 3</h2>
<pre data-label="shell">
$ sudo apt install fuse3
</pre>
```

- Add the page to `NAV` in `build.py` so it appears in the sidebar and in the
  previous/next links.
- Write internal links root-relative (`href="/docs/cli.html"`). The build
  rewrites them relative to each page.
- `<pre data-label="...">` blocks hold raw text. The build escapes them, adds
  light highlighting for shell, YAML and Python, and adds a copy button.
- `<h2>` and `<h3>` headings get ids automatically and feed "On this page".
- Tabs: `<div class="tabs"><section class="tab-panel" data-tab="Linux">…`.
  Add `data-platform` to open the visitor's OS first.
- Other components: `.callout` (and `.callout.warn`), `ol.steps`,
  `ul.checks`, `.card-list`, `dl.notes`.

## The logo

`assets/brand/clio-core-mark.svg` is a "C" built from three stacked storage
tiers (RAM, NVMe, disk) with a copper block of data at its core. It uses the
Clio family colors: cyan `#00d4db` and copper `#ee852f`. The light-theme
variant uses deep teal `#0a666c` and a darker copper.

## Keeping content accurate

Commands, flags, config keys and API names were checked against the code
when written: `context-runtime/util/clio_run*.cc`,
`context-runtime/config/clio_default.yaml`,
`context-transfer-engine/core/src/core_config.cc`,
`context-transfer-engine/adapter/libfuse/`,
`context-transfer-engine/wrapper/python/core_bindings.cc`,
`context-exploration-engine/api/src/python_bindings.cc`,
`context-exploration-engine/iowarp-cei-mcp/` and `CMakePresets.json`.
Update the matching page when those change. Nothing regenerates
automatically.
