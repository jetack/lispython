"""MkDocs hooks (``hooks:`` in mkdocs.yml): write llms.txt and llms-full.txt.

Both follow https://llmstxt.org: llms.txt lists every nav page with a link and
a description, llms-full.txt holds every nav page's markdown in nav order.
The markdown is captured after mkdocs-macros has rendered it: mkdocs appends
hooks to the plugin list after the configured plugins, so this
on_page_markdown runs after the macros plugin's (the form reference is its
generated tables, never ``{{ form_index() }}``).
"""

import importlib.util
import posixpath
import re
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import lispy  # noqa: F401  (the .lpy import hook, for lispy.form_docs)
from lispy.form_docs import BUILTIN_MACROS_NOTE

# mkdocs loads a hook by file path without putting docs/ on sys.path, so
# ``import main`` would fail (or find another top-level ``main``): load the
# sibling module by path, as the macros plugin does for ``docs/main``.
_spec = importlib.util.spec_from_file_location(
    "lispy_docs_main", Path(__file__).with_name("main.py")
)
docs_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(docs_main)

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


def _site_links(markdown, page, config, files):
    """The page markdown with every internal link to a .md source (the
    links mkdocs rewrites to page URLs) pointing at that page's site URL,
    which is what llms-full.txt can resolve; mkdocs' own file map decides
    where a source path lands. An anchor-only link points into this page's
    URL. Links in fenced blocks and inline code spans are left as written."""

    def rewrite(match):
        target = urlsplit(match.group(1))
        if not (target.scheme or target.netloc or target.path) and target.fragment:
            return _site_url(page.url, config) + f"#{target.fragment}"
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

    def rewrite_line(line):
        parts, last = [], 0
        for span in docs_main.CODE_SPAN.finditer(line):
            parts.append(_LINK.sub(rewrite, line[last : span.start()]))
            parts.append(span.group(0))
            last = span.end()
        parts.append(_LINK.sub(rewrite, line[last:]))
        return "".join(parts)

    return "\n".join(
        line if in_code else rewrite_line(line)
        for line, in_code in docs_main.markdown_lines(markdown)
    )


def _description(page, markdown):
    """The page's ``description`` front matter, else its second-level
    headings: what an LLM can expect to find there."""
    if page.meta.get("description"):
        return page.meta["description"]
    headings = [
        line[3:].strip()
        for line, in_code in docs_main.markdown_lines(markdown)
        if not in_code and line.startswith("## ")
    ]
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
