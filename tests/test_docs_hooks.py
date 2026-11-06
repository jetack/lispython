"""docs/hooks.py rewrites internal links so llms-full.txt, which joins every
page into one file, still resolves them. Built from mkdocs' own File, Files,
Page and config objects, so the site URLs are the ones mkdocs computes."""

import importlib.util
from pathlib import Path

import pytest
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import File, Files
from mkdocs.structure.pages import Page

HOOKS = Path(__file__).resolve().parent.parent / "docs" / "hooks.py"
SITE = "https://example.org/docs/"


@pytest.fixture(scope="module")
def hooks():
    spec = importlib.util.spec_from_file_location("docs_hooks", HOOKS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def site():
    config = MkDocsConfig()
    config.load_dict({"site_name": "t", "site_url": SITE})
    names = ("index.md", "macros.md", "syntax/statements.md", "syntax/reference.md")
    files = Files([File(name, "/src", "/out", True) for name in names])
    return config, files


def rewrite(hooks, site, markdown, page="syntax/reference.md"):
    config, files = site
    page = Page(None, files.get_file_from_path(page), config)
    return hooks._site_links(markdown, page, config, files)


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        ("[a](../macros.md#cond)", f"[a]({SITE}macros/#cond)"),
        ("[a](statements.md)", f"[a]({SITE}syntax/statements/)"),
        ("[a](/macros.md#x)", f"[a]({SITE}macros/#x)"),
        ('[a](statements.md#t "Title")', f'[a]({SITE}syntax/statements/#t "Title")'),
        # An anchor stays on the page it was written on, not on the joined file.
        ("[a](#built-in)", f"[a]({SITE}syntax/reference/#built-in)"),
        ("[a](https://example.com/x.md#a)", "[a](https://example.com/x.md#a)"),
        ("[a](other.md)", "[a](other.md)"),  # no such source: mkdocs warns
        ("[a](notes.txt)", "[a](notes.txt)"),
    ],
)
def test_internal_links_point_at_site_urls(hooks, site, link, expected):
    assert rewrite(hooks, site, link) == expected


def test_links_in_fenced_blocks_and_code_spans_are_left_as_written(hooks, site):
    text = "\n".join(
        [
            "```python",
            "[a](../macros.md#cond) [b](#top)",
            "```",
            "~~~",
            "[c](statements.md)",
            "~~~",
            "see `[d](statements.md)` and [e](statements.md) and ``[f](#z)``",
        ]
    )
    assert rewrite(hooks, site, text).split("\n") == [
        "```python",
        "[a](../macros.md#cond) [b](#top)",
        "```",
        "~~~",
        "[c](statements.md)",
        "~~~",
        f"see `[d](statements.md)` and [e]({SITE}syntax/statements/) and ``[f](#z)``",
    ]


def test_a_longer_fence_is_not_closed_by_a_shorter_one(hooks, site):
    text = "````\n```\n[a](statements.md)\n```\n````\n[b](statements.md)"
    assert rewrite(hooks, site, text).split("\n")[2] == "[a](statements.md)"
    assert rewrite(hooks, site, text).split("\n")[-1] == (
        f"[b]({SITE}syntax/statements/)"
    )


@pytest.mark.parametrize("opener", ["`", "``"])
def test_a_backtick_run_closes_a_span_only_against_a_run_of_equal_length(
    hooks, site, opener
):
    # CommonMark: no span here, because the only later run has another length,
    # so the link between them is live text and must be rewritten.
    closer = "```" if opener == "``" else "``"
    text = f"{opener}a [l](statements.md) {closer}"
    assert rewrite(hooks, site, text) == (
        f"{opener}a [l]({SITE}syntax/statements/) {closer}"
    )
