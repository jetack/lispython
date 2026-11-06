"""The lispython Pygments lexer (lispy.highlight), as the docs' ```lispython
fences and any Pygments user reach it: through the entry point, by alias."""

import pytest
from pygments.lexers import get_lexer_by_name, get_lexer_for_filename
from pygments.token import Comment, Keyword, Name, Number, Operator, String

import lispy  # noqa: F401  (the .lpy import hook, for lispy.form_docs)
from lispy import form_docs


@pytest.fixture(scope="module")
def lexer():
    return get_lexer_by_name("lispython")


def tokens(lexer, src):
    """[(token type, text)] of ``src``, whitespace dropped."""
    return [(t, v) for t, v in lexer.get_tokens(src) if v.strip()]


def kind_of(lexer, src, text):
    """The token type of every token spelled ``text`` in ``src`` (one)."""
    [kind] = {t for t, v in tokens(lexer, src) if v == text}
    return kind


def test_found_by_alias_and_filename(lexer):
    assert lexer.name == "LisPython"
    assert get_lexer_by_name("lpy").name == "LisPython"
    assert get_lexer_for_filename("app/main.lpy").name == "LisPython"


def test_semicolon_comment(lexer):
    src = "(= x 1) ; set x\n(print x)"
    assert kind_of(lexer, src, "; set x") is Comment.Single
    assert kind_of(lexer, src, "print") is Name.Builtin


def test_comment_glued_to_a_symbol(lexer):
    # The reader ends a symbol at ";": a;c is the symbol a and a comment.
    assert tokens(lexer, "a;c") == [(Name, "a"), (Comment.Single, ";c")]


def test_keyword_argument_and_walrus(lexer):
    src = '(f x :sep ", " (:= y 2))'
    assert kind_of(lexer, src, ":sep") is String.Symbol
    assert kind_of(lexer, src, ":=") is Operator


def test_forms_keywords_and_names(lexer):
    src = (
        "(def f [x] (return (cond (is x None) True\n"
        "                       (isinstance x ValueError) (-> x (g 1)))))\n"
        "(import numpy as np)"
    )
    assert kind_of(lexer, src, "def") is Keyword
    assert kind_of(lexer, src, "return") is Keyword
    assert kind_of(lexer, src, "is") is Keyword
    # A Python keyword that is not a lispython form.
    assert kind_of(lexer, src, "as") is Keyword
    assert kind_of(lexer, src, "cond") is Keyword.Pseudo
    assert kind_of(lexer, src, "->") is Keyword.Pseudo
    assert kind_of(lexer, src, "None") is Keyword.Constant
    assert kind_of(lexer, src, "True") is Keyword.Constant
    assert kind_of(lexer, src, "isinstance") is Name.Builtin
    assert kind_of(lexer, src, "ValueError") is Name.Exception
    assert kind_of(lexer, src, "g") is Name.Function
    assert kind_of(lexer, src, "x") is Name


def test_prefixes_wrap_a_name(lexer):
    assert tokens(lexer, "[^int n *args **kw]")[1:-1] == [
        (Operator, "^"),
        (Name.Builtin, "int"),
        (Name, "n"),
        (Operator, "*"),
        (Name, "args"),
        (Operator, "**"),
        (Name, "kw"),
    ]


def test_quote_operators(lexer):
    src = "(return `(do ~@body (while ~pred 'x)))"
    for text in ["`", "~@", "~", "'"]:
        assert kind_of(lexer, src, text) is Operator
    assert kind_of(lexer, src, "body") is Name


@pytest.mark.parametrize(
    ["text", "kind"],
    [
        ["42", Number.Integer],
        ["-1", Number.Integer],
        ["1_000", Number.Integer],
        ["1.5", Number.Float],
        ["1e-3", Number.Float],
        ["0xFF", Number.Hex],
        ["0o17", Number.Oct],
        ["0b101", Number.Bin],
        ["2j", Number],
        ["1+2j", Number],
    ],
)
def test_numbers(lexer, text, kind):
    assert tokens(lexer, f"(f {text})")[2] == (kind, text)


def test_strings(lexer):
    plain = tokens(lexer, '(print "a\\tb\\"c")')
    assert (String.Escape, "\\t") in plain
    assert (String.Escape, '\\"') in plain
    raw = tokens(lexer, '(re.compile r"\\d+")')
    assert (String.Affix, "r") in raw
    assert not [v for t, v in raw if t is String.Escape]
    # A string spans lines; a ; or a paren in it is text.
    multi = tokens(lexer, '(= s """one\n(two) ; three\n""") (f)')
    assert (String, "one\n(two) ; three\n") in multi
    assert (Name.Function, "f") in multi


def test_f_string_fields(lexer):
    src = 'f"{{x}} {(len xs)!r:>{width}} {(f :a 1)}"'
    toks = tokens(lexer, src)
    assert (String.Escape, "{{") in toks
    assert (String.Escape, "}}") in toks
    assert (Name.Builtin, "len") in toks
    assert (String.Interpol, "!r") in toks
    assert (Name, "width") in toks
    # Inside a field's brackets :a is a keyword argument, not a format spec.
    assert (String.Symbol, ":a") in toks
    assert toks[-1] == (String, '"')


def test_string_in_a_field_with_the_same_quote(lexer):
    toks = tokens(lexer, 'f"{(.get d "k")} x" y')
    assert (Name.Function, ".get") in toks
    assert toks[-1] == (Name, "y")


def test_derived_form_names(lexer):
    """Every built-in macro and every lettered special form, as a head."""
    for name in form_docs.BUILTIN_MACROS:
        assert kind_of(lexer, f"({name} x)", name) is Keyword.Pseudo, name
    for name in form_docs.SPECIAL_FORMS:
        if any(c.isalpha() for c in name):
            assert kind_of(lexer, f"({name} x)", name) is Keyword, name
