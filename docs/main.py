"""mkdocs-macros module (``module_name: docs/main`` in mkdocs.yml).

Besides the ``version`` variable it generates the form reference
(``{{ form_index() }}`` on syntax/reference.md, ``{{ macro_doc("cond") }}`` on
macros.md) from ``lispy.form_docs``, the
documentation the LSP hover shows too, and owns the rule that finds a form's
section on the hand-written pages (``form_section``), which both the reference
links and tests/test_docs.py use. Nothing here needs mkdocs, so the test
imports it directly; only the anchors need Python-Markdown (``markdown``).
"""

import functools
import posixpath
import re
import tomllib
from pathlib import Path
from typing import NamedTuple

import lispy  # noqa: F401  (the .lpy import hook, for lispy.form_docs)
from lispy import form_docs
from lispy.core.compiler.utils import HEADLESS_FORMS, STATEMENT_HEADS
from lispy.core.utils import augassignop_dict, binop_dict

DOCS_DIR = Path(__file__).resolve().parent

# The hand-written pages a form's section is looked for in, in nav order.
FORM_PAGES = ("syntax/statements.md", "syntax/expressions.md", "macros.md")
REFERENCE_PAGE = "syntax/reference.md"


# Forms documented inside a section that is not headed by the form itself.
# Every other form's section is the first heading, in FORM_PAGES order, with
# the form as a code span (## `return`, ### `->>`). A heading here that no
# longer exists counts as no section, so the coverage test reports it.
GROUP_SECTIONS = {
    **{op: ("syntax/expressions.md", "Binary Operators") for op in binop_dict},
    **{op: ("syntax/statements.md", "Augmented Assignment") for op in augassignop_dict},
    "and": ("syntax/expressions.md", "Boolean Operators"),
    "or": ("syntax/expressions.md", "Boolean Operators"),
    "ife": ("syntax/expressions.md", "`if` expression"),
    "yield-from": ("syntax/expressions.md", "`yield from`"),
    ".": ("syntax/expressions.md", "Attribute"),
    "sub": ("syntax/expressions.md", "Subscription"),
    ",": ("syntax/expressions.md", "Tuple"),
    "f-string": ("syntax/expressions.md", "Formatted String Literals"),
    "def": ("syntax/statements.md", "Function Definitions"),
    "async-def": ("syntax/statements.md", "Async Function Definitions"),
    "class": ("syntax/statements.md", "Class Definition"),
    "type": ("syntax/statements.md", "Type Alias"),
    "=": ("syntax/statements.md", "Assignment"),
    "async-for": ("syntax/statements.md", "Asynchronous `for`"),
    "async-with": ("syntax/statements.md", "Asynchronous `with`"),
    "except": ("syntax/statements.md", "`try`"),
    "else": ("syntax/statements.md", "`try`"),
    "finally": ("syntax/statements.md", "`try`"),
    "case": ("syntax/statements.md", "`match`"),
    "lazy-import": ("syntax/statements.md", "Lazy imports"),
    "lazy-from": ("syntax/statements.md", "Lazy imports"),
    "defmacro": ("macros.md", "Macro Definition"),
    "'": ("macros.md", "quote"),
    "`": ("macros.md", "syntax-quote"),
    "~@": ("macros.md", "unquote-splicing"),
}


class Heading(NamedTuple):
    level: int
    text: str  # as written, code spans included
    line: int


class Section(NamedTuple):
    page: str
    heading: Heading
    rank: tuple  # (page index, line): reading order across FORM_PAGES


_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_ATX = re.compile(r"^(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")


def markdown_lines(text):
    """``(line, in_code)`` for each line of a page: ``in_code`` is true for a
    fenced block's lines, fences included. The one home of the fence rule
    (CommonMark: a closer is the same character, at least as long)."""
    fence = None
    for line in text.split("\n"):
        opener = _FENCE.match(line)
        if opener:
            marker = opener.group(1)
            if fence is None:
                fence = marker
                yield line, True
                continue
            if marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
                yield line, True
                continue
        yield line, fence is not None


def _stamp(page, docs_dir):
    """A page's file and its mtime: the cache key, so an edit under
    ``mkdocs serve`` is read again (the process outlives the build)."""
    path = Path(docs_dir) / page
    return path, path.stat().st_mtime_ns


def page_headings(page, docs_dir=DOCS_DIR):
    """The ATX headings of a docs page, outside fenced code blocks."""
    return _headings(*_stamp(page, docs_dir))


@functools.cache
def _headings(path, _mtime_ns):
    headings = []
    text = path.read_text(encoding="utf-8")
    for number, (line, in_code) in enumerate(markdown_lines(text)):
        match = None if in_code else _ATX.match(line)
        if match:
            headings.append(Heading(len(match.group(1)), match.group(2), number))
    return tuple(headings)


def code_spans(text):
    """The contents of a heading's code spans."""
    return [match.group(2).strip() for match in CODE_SPAN.finditer(text)]


def form_section(form, docs_dir=DOCS_DIR):
    """The hand-written section documenting ``form``, or None.

    The one rule for where a form is documented: its GROUP_SECTIONS heading,
    else the first heading in FORM_PAGES with the form as a code span."""
    if form in GROUP_SECTIONS:
        page, text = GROUP_SECTIONS[form]
        for heading in page_headings(page, docs_dir):
            if heading.text == text:
                return Section(page, heading, (FORM_PAGES.index(page), heading.line))
        return None
    for index, page in enumerate(FORM_PAGES):
        for heading in page_headings(page, docs_dir):
            if form in code_spans(heading.text):
                return Section(page, heading, (index, heading.line))
    return None


def canonical_forms():
    """Every form the docs must cover: each documented form (headless kinds
    excluded), every statement head, and every built-in macro."""
    return (
        (set(form_docs.ALL_FORMS) - set(HEADLESS_FORMS))
        | set(STATEMENT_HEADS)
        | set(form_docs.BUILTIN_MACRO_NAMESPACE)
    )


def missing_sections(docs_dir=DOCS_DIR):
    """The canonical forms with no hand-written section, sorted."""
    return sorted(f for f in canonical_forms() if form_section(f, docs_dir) is None)


def page_anchors(page, docs_dir=DOCS_DIR):
    """{heading line: id} as Python-Markdown's toc extension assigns them:
    its own slugify and de-duplication over the heading text as rendered."""
    return _anchors(*_stamp(page, docs_dir), page, docs_dir)


@functools.cache
def _anchors(_path, _mtime_ns, page, docs_dir):
    from markdown.extensions.toc import slugify, unique

    used = set()
    anchors = {}
    for heading in page_headings(page, docs_dir):
        rendered = CODE_SPAN.sub(lambda m: m.group(2).strip(), heading.text)
        anchors[heading.line] = unique(slugify(rendered, "-"), used)
    return anchors


def _code(text):
    """``text`` as a markdown code span, whatever backticks it holds."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _cell(text):
    """Text for one table cell: one line, its pipes escaped outside code."""
    text = " ".join(text.split())
    parts = re.split(r"(`+.+?`+)", text)
    return "".join(p if p.startswith("`") else p.replace("|", "\\|") for p in parts)


def _table_sections():
    # (section, title, optional intro paragraph)
    return (
        (form_docs.STATEMENT, "Statements"),
        (form_docs.EXPRESSION, "Expressions"),
        (form_docs.MACRO_SYSTEM, "Macro system"),
        (
            form_docs.BUILTIN_MACRO,
            "Built-in macros",
            form_docs.BUILTIN_MACROS_NOTE,
        ),
    )


def form_index_markdown(page=REFERENCE_PAGE, docs_dir=DOCS_DIR):
    """The form reference as markdown: a table per section (statements,
    expressions, macro system, built-in macros), a row per form in the
    order the hand-written pages document them, each form linked to its
    section there."""
    out = []
    for section, title, *intro in _table_sections():
        entries = [
            (name, doc, form_section(name, docs_dir))
            for name, doc in form_docs.ALL_FORMS.items()
            if doc.section == section
        ]
        order = {name: i for i, (name, _, _) in enumerate(entries)}
        entries.sort(
            key=lambda e: (e[2] is None, e[2].rank if e[2] else (), order[e[0]])
        )
        out += [f"## {title}", "", *intro, *([""] if intro else [])]
        out += ["| Form | Usage | Description |", "| --- | --- | --- |"]
        for name, doc, found in entries:
            label = _code(name)
            if found is not None:
                target = posixpath.relpath(found.page, posixpath.dirname(page))
                anchor = page_anchors(found.page, docs_dir)[found.heading.line]
                label = f"[{label}]({target}#{anchor})"
            usage = "<br>".join(_code(line) for line in doc.usage)
            out.append(f"| {label} | {usage} | {_cell(doc.prose)} |")
        out.append("")
    return "\n".join(out)


def macro_doc_markdown(name):
    """A built-in macro's usage block and description, from the docstring the
    LSP hover and the reference table read (lispy.form_docs.BUILTIN_MACROS)."""
    doc = form_docs.BUILTIN_MACROS[name]
    return "\n".join(["```python", *doc.usage, "```", "", doc.prose])


def define_env(env):
    try:
        with open("pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        version = pyproject["project"]["version"]
    except Exception:
        version = "dev"

    env.variables["version"] = version

    @env.macro
    def macro_doc(name):
        return macro_doc_markdown(name)

    @env.macro
    def form_index():
        return form_index_markdown(env.page.file.src_uri)
