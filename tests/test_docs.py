"""Every form and built-in macro is documented on the docs site.

The canonical set (docs/main.py ``canonical_forms``) is every form in
lispy.form_docs, every statement head and every built-in macro; each needs a
section on the hand-written pages, found by the same rule the form reference
links with (``form_section``).
"""

import importlib.util
import inspect
import os
from pathlib import Path

import pytest

import lispy.macros

DOCS = Path(__file__).resolve().parent.parent / "docs"


@pytest.fixture(scope="module")
def docs_main():
    spec = importlib.util.spec_from_file_location("docs_main", DOCS / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_form_has_a_docs_section(docs_main):
    missing = docs_main.missing_sections()
    assert not missing, (
        "no section on docs/syntax/*.md or docs/macros.md (a heading with the "
        f"form as a code span, or a GROUP_SECTIONS entry) for: {missing}"
    )


def test_every_group_section_heading_exists(docs_main):
    stale = sorted(
        form
        for form in docs_main.GROUP_SECTIONS
        if docs_main.form_section(form) is None
    )
    assert not stale, f"GROUP_SECTIONS names headings that do not exist: {stale}"


def test_every_built_in_macro_has_a_usage_docstring():
    namespace = lispy.macros.__macro_namespace
    undocumented = sorted(
        name for name, macro in namespace.items() if not inspect.getdoc(macro)
    )
    assert not undocumented, f"built-in macros without a docstring: {undocumented}"
    for name, macro in namespace.items():
        usage = inspect.getdoc(macro).split("\n\n")[0]
        assert usage.startswith(f"({name}"), (name, usage)


def test_form_section_skips_headings_in_code_and_reads_code_spans(docs_main, tmp_path):
    page = tmp_path / "macros.md"
    page.write_text("# Macros\n```python\n## `zap`\n```\n### Using `zop` here\n")
    (tmp_path / "syntax").mkdir()
    for name in ("statements.md", "expressions.md"):
        (tmp_path / "syntax" / name).write_text("# Empty\n")
    assert docs_main.form_section("zap", tmp_path) is None
    found = docs_main.form_section("zop", tmp_path)
    assert (found.page, found.heading.text) == ("macros.md", "Using `zop` here")


def test_macros_page_takes_each_macro_description_from_its_docstring(docs_main):
    page = (DOCS / "macros.md").read_text(encoding="utf-8")
    assert "render_macros: true" in page.split("---")[1]
    missing = [
        name
        for name in docs_main.form_docs.BUILTIN_MACROS
        if f'{{{{ macro_doc("{name}") }}}}' not in page
    ]
    assert not missing, f"macros.md sections without macro_doc: {missing}"


def test_macro_doc_renders_usage_then_the_docstring_description(docs_main):
    doc = docs_main.form_docs.BUILTIN_MACROS["->>"]
    expected = "\n".join(["```python", *doc.usage, "```", "", doc.prose])
    assert docs_main.macro_doc_markdown("->>") == expected


def test_page_headings_are_read_again_after_the_file_changes(docs_main, tmp_path):
    page = tmp_path / "p.md"
    page.write_text("# One\n")
    assert [h.text for h in docs_main.page_headings("p.md", tmp_path)] == ["One"]
    page.write_text("# Two\n## Extra\n")
    os.utime(page, ns=(1, 1))  # a rewrite within the same mtime tick is no edit
    assert [h.text for h in docs_main.page_headings("p.md", tmp_path)] == [
        "Two",
        "Extra",
    ]
    assert list(docs_main.page_anchors("p.md", tmp_path).values()) == ["two", "extra"]
