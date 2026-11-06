"""MkDocs hooks (``hooks:`` in mkdocs.yml): write llms.txt and llms-full.txt.

Both follow https://llmstxt.org: llms.txt lists every nav page with a link and
a description, llms-full.txt holds every nav page's markdown in nav order.
The markdown is captured after mkdocs-macros has rendered it: mkdocs appends
hooks to the plugin list after the configured plugins, so this
on_page_markdown runs after the macros plugin's (the form reference is its
generated tables, never ``{{ form_index() }}``).
"""

import re
from pathlib import Path
from urllib.parse import urljoin

import lispy  # noqa: F401  (the .lpy import hook, for lispy.form_docs)
from lispy.form_docs import BUILTIN_MACROS_NOTE

_pages = {}  # src_uri -> rendered markdown, for the current build
_nav = []


def on_pre_build(config):
    # mkdocs serve rebuilds in the same process.
    _pages.clear()
    _nav.clear()


def on_nav(nav, config, files):
    _nav[:] = nav.pages


def on_page_markdown(markdown, page, config, files):
    _pages[page.file.src_uri] = markdown
    return markdown


def _url(page, config):
    url = page.url or "./"
    return urljoin(config.site_url, url) if config.site_url else url


def _description(page, markdown):
    """The page's ``description`` front matter, else its second-level
    headings: what an LLM can expect to find there."""
    if page.meta.get("description"):
        return page.meta["description"]
    headings = []
    fenced = False
    for line in markdown.splitlines():
        if re.match(r"^\s*(```|~~~)", line):
            fenced = not fenced
        elif not fenced and line.startswith("## "):
            headings.append(line[3:].strip())
    return ", ".join(headings)


def on_post_build(config):
    pages = [p for p in _nav if p.file.src_uri in _pages]
    site_dir = Path(config.site_dir)
    header = [
        f"# {config.site_name}",
        "",
        f"> {config.site_description}",
        "",
        BUILTIN_MACROS_NOTE,
    ]
    index = [
        *header,
        "",
        "Every page below in one file: [llms-full.txt](llms-full.txt).",
        "",
        "## Docs",
        "",
    ]
    full = list(header)
    for page in pages:
        markdown = _pages[page.file.src_uri]
        url = _url(page, config)
        description = _description(page, markdown)
        entry = f"- [{page.title}]({url})"
        index.append(f"{entry}: {description}" if description else entry)
        full += ["", "---", "", f"Page: {page.title}", f"URL: {url}", ""]
        full.append(markdown.strip())
    (site_dir / "llms.txt").write_text("\n".join(index) + "\n", encoding="utf-8")
    (site_dir / "llms-full.txt").write_text("\n".join(full) + "\n", encoding="utf-8")
