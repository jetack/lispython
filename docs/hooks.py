"""MkDocs hooks (``hooks:`` in mkdocs.yml): write llms.txt, llms-full.txt and
llms-syntax.txt.

The first two follow https://llmstxt.org: llms.txt lists every nav page with a
link and a description, llms-full.txt holds every nav page's markdown in nav
order. llms-syntax.txt holds only the pages listed in ``extra.llms_syntax_pages``
(mkdocs.yml, the one home of that set), in that order: what an agent needs to
write lispython. A listed path that is not a nav page fails the build.
The markdown is captured after mkdocs-macros has rendered it: mkdocs appends
hooks to the plugin list after the configured plugins, so this
on_page_markdown runs after the macros plugin's (the form reference is its
generated tables, never ``{{ form_index() }}``).
"""

import importlib.util
import logging
import posixpath
import re
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from mkdocs.exceptions import PluginError

import lispy  # noqa: F401  (the .lpy import hook, for lispy.form_docs)
from lispy.form_docs import BUILTIN_MACROS_NOTE

SYNTAX_PREAMBLE = (
    "LisPython is Python written as s-expressions: a `.lpy` file compiles to a "
    "Python AST and runs on CPython, so Python's semantics, built-ins and "
    "standard library apply unchanged. This file covers only what differs from "
    "Python."
)

# mkdocs loads a hook by file path without putting docs/ on sys.path, so
# ``import main`` would fail (or find another top-level ``main``): load the
# sibling module by path, as the macros plugin does for ``docs/main``.
_spec = importlib.util.spec_from_file_location(
    "lispy_docs_main", Path(__file__).with_name("main.py")
)
docs_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(docs_main)

# src_uri -> rendered markdown. Never cleared, so a rebuild in the same process
# (mkdocs serve --dirty skips on_page_markdown for unchanged pages) keeps their
# last render. A cold ``build --dirty`` starts with it empty: on_post_build then
# leaves the llms files as the last full build wrote them. Entries are
# overwritten whenever a page renders, and the output is selected through the
# current nav, so a page removed from the nav never appears.
_pages = {}
_log = logging.getLogger("mkdocs.hooks.llms")
_nav = []


def on_pre_build(config):
    # mkdocs serve rebuilds in the same process.
    _nav.clear()


def on_nav(nav, config, files):
    _nav[:] = nav.pages


def on_page_markdown(markdown, page, config, files):
    _pages[page.file.src_uri] = _site_links(markdown, page, config, files)
    return markdown


def _site_url(url, config):
    """A site-relative URL as llms.txt lists it: absolute under site_url."""
    url = url or "./"
    if not config.site_url:
        return url
    # mike sets site_url to <site_url><version> with no trailing slash, and
    # urljoin would then replace the version segment (as mkdocs' canonical_url
    # also guards against).
    return urljoin(config.site_url.rstrip("/") + "/", url)


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
    nav_by_src = {p.file.src_uri: p for p in _nav}
    listed = config.extra.get("llms_syntax_pages")
    if not listed:
        raise PluginError("extra.llms_syntax_pages is missing or empty")
    syntax = []
    for src in listed:
        if src not in nav_by_src:
            raise PluginError(
                f"extra.llms_syntax_pages lists {src!r}, which is not a nav page"
            )
        if nav_by_src[src] in syntax:
            raise PluginError(f"extra.llms_syntax_pages lists {src!r} twice")
        syntax.append(nav_by_src[src])
    # The files are derived from every nav page's rendered markdown: a build
    # that has not rendered them all (a cold --dirty build skips unchanged
    # pages) cannot produce them, so it leaves the previous ones alone.
    missing = [p for p in _nav if p.file.src_uri not in _pages]
    if missing:
        _log.info(
            "llms*.txt left as the last full build wrote them: %d of %d pages "
            "were not rendered in this build (--dirty skips unchanged pages)",
            len(missing),
            len(_nav),
        )
        return
    site_dir = Path(config.site_dir)
    header = [
        f"# {config.site_name}",
        "",
        f"> {config.site_description}",
        "",
    ]
    index = [
        *header,
        BUILTIN_MACROS_NOTE,
        "",
        "The syntax reference for writing LisPython, in one file: "
        f"[llms-syntax.txt]({_site_url('llms-syntax.txt', config)}).",
        "",
        "Every page below in one file: "
        f"[llms-full.txt]({_site_url('llms-full.txt', config)}).",
        "",
        "## Docs",
        "",
    ]
    for page in _nav:
        description = _description(page, _pages[page.file.src_uri])
        entry = f"- [{page.title}]({_url(page, config)})"
        index.append(f"{entry}: {description}" if description else entry)
    full = [*header, BUILTIN_MACROS_NOTE]
    for page in _nav:
        full += _page_block(page, config)
    syntax_file = [
        f"# {config.site_name} syntax reference",
        "",
        f"> {config.site_description}",
        "",
        SYNTAX_PREAMBLE,
        "",
        BUILTIN_MACROS_NOTE,
    ]
    for page in syntax:
        syntax_file += _page_block(page, config)
    for name, lines in (
        ("llms.txt", index),
        ("llms-full.txt", full),
        ("llms-syntax.txt", syntax_file),
    ):
        (site_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _page_block(page, config):
    """One page as llms-full.txt and llms-syntax.txt carry it."""
    return [
        "",
        "---",
        "",
        f"Page: {page.title}",
        f"URL: {_url(page, config)}",
        "",
        _pages[page.file.src_uri].strip(),
    ]
