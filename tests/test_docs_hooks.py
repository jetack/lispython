"""docs/hooks.py rewrites internal links so llms-full.txt, which joins every
page into one file, still resolves them. Built from mkdocs' own File, Files,
Page and config objects, so the site URLs are the ones mkdocs computes."""

import importlib.util
from pathlib import Path

import pytest
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.exceptions import PluginError
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


# The real hook sequence, as mkdocs drives it, writing into tmp_path.
NAV = [
    ("index.md", "Home"),
    ("syntax/statements.md", "Statements"),
    ("syntax/expressions.md", "Expressions"),
    ("usage/cli.md", "CLI"),
    ("macros.md", "Macros"),
]
LISTED = ["macros.md", "syntax/statements.md", "syntax/expressions.md"]


def build(hooks, tmp_path, listed, site_url=SITE):
    config = MkDocsConfig()
    config.load_dict(
        {
            "site_name": "t",
            "site_url": site_url,
            "site_description": "d",
            "site_dir": str(tmp_path),
            "extra": {"llms_syntax_pages": listed},
        }
    )
    files = Files([File(src, "/src", "/out", True) for src, _ in NAV])
    pages = [Page(title, files.get_file_from_path(src), config) for src, title in NAV]
    hooks.on_pre_build(config)
    hooks.on_nav(type("Nav", (), {"pages": pages})(), config, files)
    for page in pages:
        hooks.on_page_markdown(f"BODY {page.file.src_uri}", page, config, files)
    hooks.on_post_build(config)
    return {p.name: p.read_text(encoding="utf-8") for p in tmp_path.iterdir()}


def page_lines(text):
    return [line for line in text.splitlines() if line.startswith("Page: ")]


def test_syntax_file_holds_the_listed_pages_in_list_order(hooks, tmp_path):
    out = build(hooks, tmp_path, LISTED)
    syntax = out["llms-syntax.txt"]
    assert page_lines(syntax) == [
        "Page: Macros",
        "Page: Statements",
        "Page: Expressions",
    ]
    assert syntax.index("BODY macros.md") < syntax.index("BODY syntax/statements.md")
    assert "BODY index.md" not in syntax and "BODY usage/cli.md" not in syntax
    # llms-full.txt keeps meaning every page, in nav order.
    assert page_lines(out["llms-full.txt"]) == [f"Page: {t}" for _, t in NAV]


def test_index_links_the_syntax_file_before_the_full_file(hooks, tmp_path):
    index = build(hooks, tmp_path, LISTED)["llms.txt"]
    syntax_link = f"[llms-syntax.txt]({SITE}llms-syntax.txt)"
    full_link = f"[llms-full.txt]({SITE}llms-full.txt)"
    assert 0 <= index.index(syntax_link) < index.index(full_link)


@pytest.mark.parametrize("listed", [["macros.md", "gone.md"], ["macros.md"] * 2])
def test_a_listed_path_that_is_no_nav_page_or_repeats_fails_the_build(
    hooks, tmp_path, listed
):
    with pytest.raises(PluginError):
        build(hooks, tmp_path, listed)


def test_a_versioned_site_url_without_a_trailing_slash_keeps_its_version(
    hooks, tmp_path
):
    # mike sets site_url to <site_url><version>, no trailing slash.
    out = build(hooks, tmp_path, LISTED, site_url="https://example.org/docs/0.4.6")
    base = "https://example.org/docs/0.4.6/"
    assert f"URL: {base}macros/" in out["llms-syntax.txt"]
    assert f"({base}llms-syntax.txt)" in out["llms.txt"]
