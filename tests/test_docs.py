"""Every form and built-in macro is documented on the docs site.

The canonical set (docs/main.py ``canonical_forms``) is every form in
lispy.form_docs, every statement head and every built-in macro; each needs a
section on the hand-written pages, found by the same rule the form reference
links with (``form_section``).
"""

import importlib.util
import inspect
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
