"""MkDocs hooks (``hooks:`` in mkdocs.yml): write llms.txt and llms-full.txt.

Both follow https://llmstxt.org: llms.txt lists every nav page with a link and
a description, llms-full.txt holds every nav page's markdown in nav order.
The markdown is captured after mkdocs-macros has rendered it: mkdocs appends
hooks to the plugin list after the configured plugins, so this
on_page_markdown runs after the macros plugin's (the form reference is its
generated tables, never ``{{ form_index() }}``).
"""

import posixpath
import re
from pathlib import Path
from urllib.parse import urljoin, urlsplit

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
    _pages[page.file.src_uri] = _site_links(markdown, page, config, files)
    return markdown


def _site_url(url, config):
    """A site-relative URL as llms.txt lists it: absolute under site_url."""
    url = url or "./"
    return urljoin(config.site_url, url) if config.site_url else url


def _url(page, config):
    return _site_url(page.url, config)


# [text](target) and [text](target "title"); the target has no space or paren.
_LINK = re.compile(r"(?<=\]\()([^)\s]+)(?=[)\s])")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def _site_links(markdown, page, config, files):
    """The page markdown with every internal link to a .md source (the
    links mkdocs rewrites to page URLs) pointing at that page's site URL,
    which is what llms-full.txt can resolve; mkdocs' own file map decides
    where a source path lands. Links in fenced code are left as written."""

    def rewrite(match):
        target = urlsplit(match.group(1))
        if target.scheme or target.netloc or not target.path.endswith(".md"):
            return match.group(1)
        if target.path.startswith("/"):
            src = target.path.lstrip("/")
        else:
            src = posixpath.join(posixpath.dirname(page.file.src_uri), target.path)
        file = files.get_file_from_path(posixpath.normpath(src))
        if file is None:
            return match.group(1)
        url = _site_url(file.url, config)
        return url + (f"#{target.fragment}" if target.fragment else "")

    out = []
    fence = None
    for line in markdown.split("\n"):
        opener = _FENCE.match(line)
        if opener:
            marker = opener.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
        out.append(line if fence or opener else _LINK.sub(rewrite, line))
    return "\n".join(out)


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
