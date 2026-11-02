import ast
import re
import sys
import time
import warnings

import pytest

from lispy.core.compiler import features
from lispy.core.errors import LispyCompileError, compile_errors
from lispy.core.nodes import FStrExpr, Paren, Symbol, atom_text
from lispy.core.parser import parse, tokenize
from lispy.tools import macroexpand_then_compile
from tests.utils import stmt_to_dump

T_STRINGS = (
    tuple(features.running_version())[:2] >= features.FEATURES["t-string"].min_version
)
NEEDS_3_14 = pytest.mark.skipif(not T_STRINGS, reason="t-strings need Python 3.14")
NOT_CONVERSION = (
    "f-string: invalid conversion character {!r}: expected 's', 'r', or 'a'"
)
CONVERSION_GAP = "f-string: conversion type must come right after the exclamation mark"
UNTERMINATED_STRING = "unterminated string literal (detected at line {})"
SPEC_NEWLINE = (
    "{0}: newlines are not allowed in format specifiers for single quoted {0}s"
)


def error_position(read, src):
    with pytest.raises(SyntaxError) as info:
        read(src)
    return (info.value.msg, info.value.lineno, info.value.offset)


class TestParser:
    def test_arrow_macro_definition(self):
        rst = parse("(defmacro -> [x *fs] x)")
        p = rst.pop()
        assert isinstance(p, Paren)
        assert str(p.list.__getitem__(1)) == "->"
        assert isinstance(p.list.__getitem__(1), Symbol)

    def test_arrow_macro_call(self):
        rst = parse("(-> 1 (+ 2))")
        p = rst.pop()
        assert isinstance(p, Paren)
        assert str(p.list.__getitem__(0)) == "->"
        assert isinstance(p.list.__getitem__(0), Symbol)


class TestCommaRemoval:
    """Test that commas are removed except after { or ( with optional whitespace."""

    def test_comma_removed_in_paren(self):
        """(a, b, c) should parse same as (a b c)"""
        with_commas = parse("(a, b, c)")
        without_commas = parse("(a b c)")
        assert str(with_commas) == str(without_commas)

    def test_comma_removed_numbers(self):
        """(1,2,3) should parse same as (1 2 3)"""
        with_commas = parse("(1,2,3)")
        without_commas = parse("(1 2 3)")
        assert str(with_commas) == str(without_commas)

    def test_comma_kept_after_brace(self):
        """{,} should keep the comma"""
        tokens = tokenize("{,}")
        token_strs = [t[0] for t in tokens]
        assert "," in token_strs

    def test_comma_kept_after_paren(self):
        """(,) should keep the comma"""
        tokens = tokenize("(,)")
        token_strs = [t[0] for t in tokens]
        assert "," in token_strs

    def test_comma_kept_after_brace_with_space(self):
        """{ ,} should keep the comma"""
        tokens = tokenize("{ ,}")
        token_strs = [t[0] for t in tokens]
        assert "," in token_strs

    def test_comma_kept_after_paren_with_space(self):
        """( ,) should keep the comma"""
        tokens = tokenize("( ,)")
        token_strs = [t[0] for t in tokens]
        assert "," in token_strs

    def test_comma_kept_after_brace_with_multiple_spaces(self):
        """{   ,} should keep the comma"""
        tokens = tokenize("{   ,}")
        token_strs = [t[0] for t in tokens]
        assert "," in token_strs

    def test_mixed_commas(self):
        """{, a, b} should keep first comma, remove second"""
        tokens = tokenize("{, a, b}")
        token_strs = [t[0] for t in tokens]
        assert token_strs.count(",") == 1


class TestBrackets:
    """Unbalanced brackets get CPython's messages, never an IndexError or a
    silently dropped form."""

    @pytest.mark.parametrize(
        ["src", "msg", "col"],
        [
            ("(a b))", "unmatched ')'", 6),
            (")", "unmatched ')'", 1),
            (
                "(a b]",
                "closing parenthesis ']' does not match opening parenthesis '('",
                5,
            ),
            (
                "(a [b c)",
                "closing parenthesis ')' does not match opening parenthesis '['",
                8,
            ),
            ("(a (b c)", "'(' was never closed", 1),
            ("(a [b (c d)", "'[' was never closed", 4),
            ('f"{)}"', "unmatched ')'", 4),
        ],
    )
    def test_refused_like_python(self, src, msg, col):
        with pytest.raises(LispyCompileError) as info:
            parse(src)
        assert info.value.msg == msg
        assert (info.value.lineno, info.value.offset) == (1, col)

    @pytest.mark.parametrize(
        ["src", "line", "opening_line"],
        [("(\n]", 2, 1), ("a\n(\n\n]", 4, 2), ('f"{(\n]}"', 2, 1)],
    )
    def test_a_mismatch_names_the_opening_s_other_line(self, src, line, opening_line):
        assert error_position(parse, src) == (
            "closing parenthesis ']' does not match opening parenthesis '(' "
            f"on line {opening_line}",
            line,
            1,
        )


class TestWrapperPrefix:
    """An opening or meta-indicator takes one of ^, *, ** as its wrapper; ^
    followed by stars there has no meaning and is refused at the prefix,
    never a KeyError or a symbol that swallows the opening."""

    @pytest.mark.parametrize(
        ["form", "prefix", "indicator"],
        [
            ("^*(a)", "^*", "("),
            ("^**[a]", "^**", "["),
            ("^*{a}", "^*", "{"),
            ("^*'a", "^*", "'"),
            ("^**~a", "^**", "~"),
            ("^*~@a", "^*", "~@"),
            ("^*`a", "^*", "`"),
            # A space after the quote does not make it a symbol.
            ("^*' a", "^*", "'"),
            ("^**` a", "^**", "`"),
            ("^*~ a", "^*", "~"),
            ("^**~@ a", "^**", "~@"),
            # ^ and stars alone: nothing to annotate or unpack.
            ("^*", "^*", ""),
            ("^**", "^**", ""),
            ("^* a", "^*", ""),
            ("^*,a", "^*", ""),
        ],
    )
    def test_refused_at_the_prefix(self, form, prefix, indicator):
        src = "(g)\n  (f " + form + ")"
        with pytest.raises(LispyCompileError) as info:
            parse(src)
        assert info.value.msg == (
            f"{prefix} prefix has no meaning before {indicator!r}"
            if indicator
            else f"{prefix} has no meaning on its own"
        )
        assert (info.value.lineno, info.value.offset) == (2, 6)
        token_end = 6 + len(prefix) + len(indicator)
        assert (info.value.end_lineno, info.value.end_offset) == (2, token_end)
        assert [str(f) for f in parse(src, partial=True)] == ["(g)"]

    @pytest.mark.parametrize(
        ["src", "parsed"],
        [
            ("(f ^*Ts)", "Paren(Sym(f), Ann(Star(Sym(Ts))))"),
            ("(f ^**P)", "Paren(Sym(f), Ann(DStar(Sym(P))))"),
            ("(f ^(a))", "Paren(Sym(f), Ann(Paren(Sym(a))))"),
            ("(f *(a))", "Paren(Sym(f), Star(Paren(Sym(a))))"),
            ("(f **{a})", "Paren(Sym(f), DStar(Brace(Sym(a))))"),
            ("(f *'a)", "Paren(Sym(f), Star(Quote(Sym(a))))"),
            ("(f ^~a)", "Paren(Sym(f), Ann(Unquote(Sym(a))))"),
        ],
    )
    def test_single_wrappers_still_parse(self, src, parsed):
        [form] = parse(src)
        assert repr(form) == parsed


class TestSemicolonEndsToken:
    """A ; ends a symbol or number, so a comment needs no space before it."""

    @pytest.mark.parametrize(
        ["src", "parsed"],
        [
            ("(f a;c\n b)", "Paren(Sym(f), Sym(a), Sym(b))"),
            ("(f 1;c\n 2)", "Paren(Sym(f), Const('1'), Const('2'))"),
            ("(f *;c\n a)", "Paren(Sym(f), Sym(*), Sym(a))"),
            ("(f x);c", "Paren(Sym(f), Sym(x))"),
            ('(f "a;b")', "Paren(Sym(f), Str('\"a;b\"'))"),
        ],
    )
    def test_comment_follows_without_space(self, src, parsed):
        [form] = parse(src)
        assert repr(form) == parsed

    def test_bare_annotated_star_before_comment_is_refused(self):
        with pytest.raises(LispyCompileError) as info:
            parse("(def f [^*;c\n] 1)")
        assert info.value.msg == "^* has no meaning on its own"
        assert (info.value.lineno, info.value.offset) == (1, 9)


class TestCommentEndsAtAnyLineBreak:
    """CPython ends a comment at LF, CRLF or a lone CR."""

    @pytest.mark.parametrize("brk", ["\n", "\r\n", "\r"])
    def test_the_form_after_a_comment_is_kept(self, brk):
        assert repr(parse(f"; c{brk}(f 1)")) == "[Paren(Sym(f), Const('1'))]"

    def test_a_comment_in_a_field_ends_at_a_lone_cr(self):
        assert str(parse('f"{(f x ;c\r y)}"')[0].list[2].value) == "(f x y)"

    @pytest.mark.parametrize("brk", ["\n", "\r\n", "\r"])
    def test_a_comment_at_a_fields_own_level_ends_at_any_line_break(self, brk):
        # Not in brackets: the comment ends at the break, leaving a second
        # expression, which is refused (CPython's wording varies by version).
        with pytest.raises(LispyCompileError) as info:
            parse(f'f"""{{x ;c{brk} y}}"""')
        assert info.value.msg == (
            "f-string: expecting a single expression in a replacement field"
        )

    @pytest.mark.parametrize(
        ["src", "msg", "lineno"],
        [
            ("; c\r(f 1", "'(' was never closed", 2),
            ("; c\r(f 1)\r)", "unmatched ')'", 3),
        ],
    )
    def test_error_lines_count_the_lone_cr(self, src, msg, lineno):
        with pytest.raises(SyntaxError) as info:
            compile(src.replace(";", "#", 1), "<s>", "exec")
        assert (info.value.msg, info.value.lineno) == (msg, lineno)
        with pytest.raises(LispyCompileError) as ours:
            parse(src)
        assert (ours.value.msg, ours.value.lineno) == (msg, lineno)


class TestDanglingQuote:
    """A quote, quasiquote or unquote-splice (wrapped or not) with no form
    right after it is refused at the token, never a symbol, a quote of the
    next form or an unbalanced closing."""

    @pytest.mark.parametrize(
        "token",
        ["'", "`", "~@", "*'", "**'", "^'", "*`", "^~@"],
    )
    @pytest.mark.parametrize("after", [" a)", ")", ",a)", ";c\n a)", "\n a)", ""])
    def test_refused_at_the_token(self, token, after):
        src = "(g)\n  (f " + token + after
        with pytest.raises(LispyCompileError) as info:
            parse(src)
        assert info.value.msg == f"{token} must be directly followed by a form"
        assert (info.value.lineno, info.value.offset) == (2, 6)
        assert (info.value.end_lineno, info.value.end_offset) == (2, 6 + len(token))
        assert [str(f) for f in parse(src, partial=True)] == ["(g)"]

    @pytest.mark.parametrize(
        ["src", "parsed"],
        [
            ("(~ x)", "Paren(Sym(~), Sym(x))"),
            ("(f ~ x)", "Paren(Sym(f), Sym(~), Sym(x))"),
            ("'[a b]", "Quote(Bracket(Sym(a), Sym(b)))"),
            ('\'"s"', "Quote(Str('\"s\"'))"),
            ("'(a)", "Quote(Paren(Sym(a)))"),
            ("`(a ~@b)", "QuasiQuote(Paren(Sym(a), UnquoteSplice(Sym(b))))"),
            ("(f *'a)", "Paren(Sym(f), Star(Quote(Sym(a))))"),
            ("(f ^*Ts)", "Paren(Sym(f), Ann(Star(Sym(Ts))))"),
            # ~ with no form after it is the symbol ~, as before a space.
            ("(f ~)", "Paren(Sym(f), Sym(~))"),
            ("(f ~;c\n a)", "Paren(Sym(f), Sym(~), Sym(a))"),
            ("(f ~,a)", "Paren(Sym(f), Sym(~), Sym(a))"),
            ("~;c", "Sym(~)"),
        ],
    )
    def test_quotes_with_a_form_still_parse(self, src, parsed):
        [form] = parse(src)
        assert repr(form) == parsed


class TestPrefixedOperandType:
    """The operand after ^, *, ** or : keeps its own token type, so a number
    there is a Constant (as it is standing alone), not a Symbol named 1."""

    @pytest.mark.parametrize(
        ["src", "parsed"],
        [
            ("*1", "Star(Const('1'))"),
            ("**1", "DStar(Const('1'))"),
            (":1", "Kwd(Const('1'))"),
            ("^1", "Ann(Const('1'))"),
            (":1.5e3", "Kwd(Const('1.5e3'))"),
            ("*2j", "Star(Const('2j'))"),
            ("*0x1f", "Star(Const('0x1f'))"),
            ("*-1", "Star(Paren(Sym(-), Const('1')))"),
            ("*a1", "Star(Sym(a1))"),
            (":a", "Kwd(Sym(a))"),
            ("**kw", "DStar(Sym(kw))"),
            ("*True", "Star(Const('True'))"),
        ],
    )
    def test_operand_type(self, src, parsed):
        [form] = parse(src)
        assert repr(form) == parsed

    def test_operand_position(self):
        [form] = parse("**1")
        assert (form.value.lineno, form.value.col_offset) == (1, 2)


def nested(opening, depth, closing='}"', before="", inner="x"):
    """``depth`` literals opened by ``opening``, nested through one field, in
    the parens ``before`` opens."""
    return before + opening * depth + inner + closing * depth + ")" * len(before)


class TestNestedFStrings:
    """Nests past the 149 literals CPython's lexer takes in Python source:
    the AST has no such limit, and lispython leaves it to CPython."""

    DEPTH = 200

    @pytest.mark.parametrize(
        ["opening", "closing", "before"],
        [
            ('f"{', '}"', ""),
            ('f"""{', '}"""', ""),
            ('rf"{', '}"', ""),
            ('f"{\n', '}"', ""),
            ('f"{', '}"', "((("),
        ],
        ids=["single", "triple", "raw", "multiline", "in-a-form"],
    )
    def test_a_nest_past_cpython_s_lexer_limit_parses(self, opening, closing, before):
        assert len(parse(nested(opening, self.DEPTH, closing, before))) == 1

    def test_a_nest_in_a_format_spec_parses(self):
        assert len(parse('f"{x:{' + nested('f"{', self.DEPTH) + '}}"')) == 1


def spec_chain(form):
    """The fields of a format-spec chain, outermost first, each as (its
    columns, its spec's columns or None), with the innermost value."""
    fields = []
    node = form
    while len(node.list) > 2:  # the literal, then a field
        field = node.list[2]
        assert isinstance(field, FStrExpr)
        spec = field.format_spec
        fields.append(
            (
                (field.col_offset, field.end_col_offset),
                None if spec is None else (spec.col_offset, spec.end_col_offset),
            )
        )
        if spec is None:
            return fields, field.value
        node = spec
    raise AssertionError("no innermost field")


class TestFormatSpecChains:
    """A format spec holds fields, whose specs hold fields, to any depth:
    lispython adds no limit of its own (compile() applies the compiler's)."""

    def test_a_spec_chain_parses(self):
        [form] = parse('f"{x:{x:{x}}}"')
        assert atom_text(form.list[0]) == "f-string"
        [fields, leaf] = spec_chain(form)
        # Each spec runs from its ":" to the "}" closing its field.
        assert fields == [((2, 13), (4, 12)), ((5, 12), (7, 11)), ((8, 11), None)]
        assert (type(leaf), atom_text(leaf), leaf.col_offset) == (Symbol, "x", 9)

    @pytest.mark.parametrize(
        ["src", "depth"],
        [
            ('f"{x:{x:{x:{x}}}}"', 4),
            ('f"{x:{x:{x}{x:{x}}}}"', 3),
            ('rf"{x:{x:{x:{x}}}}"', 4),
            ('f"""{x:{x:{x:\n\nab{x}}}}"""', 4),
        ],
        ids=["chain", "after-a-sibling", "raw", "lines-below"],
    )
    def test_a_chain_past_cpython_s_three_fields_parses(self, src, depth):
        [form] = parse(src)
        assert len(spec_chain(form)[0]) == depth

    @pytest.mark.skipif(not T_STRINGS, reason="t-strings need Python 3.14")
    def test_a_t_string_chain_parses(self):
        [form] = parse('t"{x:{x:{x:{x}}}}"')
        assert atom_text(form.list[0]) == "t-string"
        assert len(spec_chain(form)[0]) == 4

    def test_braces_inside_an_expression_are_not_fields(self):
        [form] = parse('f"{ {1 {2 {3 {4 5}}}} }"')
        assert type(form.list[2].value).__name__ == "Brace"

    @pytest.mark.parametrize(
        ["lpy", "py"],
        [
            ('f"{x:{{}}}"', None),
            ('f"{x:{{}} }"', None),
            ('f"{x:a{{}}b}"', None),
            pytest.param(
                'f"{x:{x:{{}}}}"',
                None,
                marks=pytest.mark.skipif(
                    sys.version_info < (3, 12),
                    reason="3.11's parser refuses this nesting depth: no reference AST",
                ),
            ),
            ('f"{x:}}}"', None),
            ('f"{x:{{1 2}}}"', 'f"{x:{ {1: 2} }}"'),
            pytest.param(
                't"{x:{{}}}"',
                None,
                marks=pytest.mark.skipif(not T_STRINGS, reason="needs Python 3.14"),
            ),
        ],
        ids=[
            "empty-dict",
            "empty-dict-space",
            "text-around",
            "nested",
            "close-then-literal",
            "dict-pairs",
            "t-string",
        ],
    )
    def test_no_brace_is_doubled_in_a_format_spec(self, lpy, py):
        # CPython has no {{ or }} escape in a spec: { opens a field, } ends it.
        assert stmt_to_dump(lpy) == ast.dump(ast.parse(py or lpy))

    @pytest.mark.parametrize(["src", "col"], [('f"{x:{{"', 8), ('f"{x:{{}"', 9)])
    def test_an_unclosed_doubled_brace_in_a_spec_is_a_field(self, src, col):
        # The quote is a string in the inner field's expression.
        assert error_position(parse, src) == (UNTERMINATED_STRING.format(1), 1, col)

    def test_a_chain_of_empty_dicts_past_cpython_s_nesting_limit_parses(self):
        # CPython's tokenizer refuses this as "nested too deeply"; lispython
        # adds no limit of its own, as with the chains above.
        [form] = parse('f"{x:{x:{x:{{}}}}}"')
        assert len(spec_chain(form)[0]) == 4

    @pytest.mark.parametrize(
        ["src", "msg", "at"],
        [
            ('f"{x:{x:{x:{x}}}} }"', "f-string: single '}' is not allowed", (1, 19)),
        ],
        ids=["single-brace"],
    )
    def test_an_error_after_a_deep_chain_is_reported_where_it_is(self, src, msg, at):
        with pytest.raises(LispyCompileError) as info:
            parse(src)
        assert info.value.msg == msg
        assert (info.value.lineno, info.value.offset) == at

    def test_a_string_after_a_deep_chain_is_read_where_it_is(self):
        assert error_position(parse, 'f"{x:{x:{x:{x}}}} {"') == (
            UNTERMINATED_STRING.format(1),
            1,
            20,
        )

    def test_the_first_of_two_errors_is_reported(self):
        # CPython's message, at the name.
        with pytest.raises(LispyCompileError) as info:
            parse('f"{x!z} {x:{x:{x:{x}}} }"')
        assert info.value.msg == (
            "f-string: invalid conversion character 'z': expected 's', 'r', or 'a'"
        )
        assert (info.value.lineno, info.value.offset) == (1, 6)

    def test_an_earlier_literal_error_comes_first(self):
        with pytest.raises(LispyCompileError) as info:
            parse('f"}"\nf"{x:{x:{x:{x}}}}"')
        assert info.value.msg == "f-string: single '}' is not allowed"
        assert (info.value.lineno, info.value.offset) == (1, 3)


ESCAPED_QUOTE_DEPRECATED = r'write a plain " in a replacement field, not \\"'


def f_string_reference(expr):
    """The AST of Python's f"{expr}". Python 3.11 refuses a quote or a
    backslash in a field, so there ``expr`` is parsed alone and put in the
    field of f"{_}"."""
    if sys.version_info >= (3, 12):
        return ast.parse(f'f"{{{expr}}}"')
    tree = ast.parse('f"{_}"')
    tree.body[0].value.values[0].value = ast.parse(expr, mode="eval").body
    return tree


class TestFieldsLexedInPlace:
    """A replacement field is lexed where it stands, as CPython 3.12+ lexes it."""

    def test_a_quote_in_a_field_opens_a_string(self):
        # CPython 3.12+ reads f"{f"{"}"}"}" as two literals around "}".
        [form] = parse('f"{f"{"}"}"}"')
        inner = form.list[2].value
        assert atom_text(inner.list[0]) == "f-string"
        assert repr(inner.list[2].value) == "Str('\"}\"')"
        assert inner.list[2].expr_text == '"}"'

    def test_a_string_in_a_field_is_at_its_own_column(self):
        [form] = parse('f"{(.get d "k")}"')
        field = form.list[2]
        [_, _, key] = field.value.list
        assert repr(key) == "Str('\"k\"')"
        assert field.expr_text == '(.get d "k")'
        assert (key.col_offset, key.end_col_offset) == (11, 14)
        assert (field.value.end_col_offset, field.end_col_offset) == (15, 16)

    @pytest.mark.parametrize(
        ["src", "expr"],
        [
            # A field holds strings opened with a bare quote, and CPython's
            # escapes apply in them: \" is a quote, \\ a backslash.
            pytest.param(r'f"{(g "a\\")}"', r'g("a\\")', id="escaped-backslash-closes"),
            pytest.param(
                r'f"{(g "a\\\"b")}"', r'g("a\\\"b")', id="escaped-backslash-then-quote"
            ),
            pytest.param(r'f"{(g r"a\"b")}"', r'g(r"a\"b")', id="raw-escaped-quote"),
            pytest.param(r'f"{(g r"a\\")}"', r'g(r"a\\")', id="raw-backslash-closes"),
            pytest.param(
                r"""f"{(g r'''a\"b''')}" """.strip(),
                r"g(r'''a\"b''')",
                id="raw-triple-single-quoted-keeps-backslash",
            ),
            pytest.param(
                r'f"{(g "say \"hi\"")}"', r'g("say \"hi\"")', id="escaped-quote"
            ),
            pytest.param(
                r'f"{f"x\"y"}"', r'f"x\"y"', id="nested-f-string-escaped-quote"
            ),
        ],
    )
    def test_a_bare_quoted_string_in_a_field_reads_as_python(self, src, expr):
        assert stmt_to_dump(src) == ast.dump(f_string_reference(expr))

    def test_a_bare_quoted_string_in_a_field_is_not_deprecated(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert stmt_to_dump(r'f"{(g "say \"hi\"")}"') == ast.dump(
                f_string_reference(r'g("say \"hi\"")')
            )

    def test_an_echo_is_the_field_source_text(self):
        [stmt] = macroexpand_then_compile(parse(r'f"{(g "say \"hi\"")=}"'))
        assert stmt.value.values[0].value == r'(g "say \"hi\"")='

    def test_an_echo_shows_an_escaped_quote_as_written(self):
        with pytest.warns(SyntaxWarning, match=ESCAPED_QUOTE_DEPRECATED):
            [stmt] = macroexpand_then_compile(parse(r'f"{(.get d \"k\")=}"'))
        assert stmt.value.values[0].value == r"(.get d \"k\")="

    @pytest.mark.parametrize(
        ["src", "py", "count"],
        [
            # The deprecated spelling of a field's string: opened and closed
            # with \", it reads as the same string with plain quotes.
            pytest.param(r'f"{(.get d \"k\")}"', "f\"{d.get('k')}\"", 1, id="escaped"),
            pytest.param(
                r'f"{(g \"\"\"a\"\"\")}"', "f\"{g('''a''')}\"", 1, id="triple"
            ),
            pytest.param(
                r'f"{(g \"a\\"b\")}"',
                """f'''{g('a"b')}'''""",
                1,
                id="backslash-escaped-quote-is-a-quote",
            ),
            pytest.param(
                r'f"{(g \"a\" \"b\")}"', "f\"{g('a', 'b')}\"", 2, id="two-strings"
            ),
            pytest.param(r'f"{f\"{x}\"}"', "f\"{f'{x}'}\"", 1, id="nested-f-string"),
            pytest.param(
                r'f"{f\"{(g \"a\")}\"}"',
                """f\"\"\"{f'{g("a")}'}\"\"\"""",
                2,
                id="escaped-string-in-nested-f-string",
            ),
            pytest.param(
                r'f"{x:{(g \"a\")}}"', "f\"{x:{g('a')}}\"", 1, id="in-a-format-spec"
            ),
        ],
    )
    def test_an_escaped_string_is_deprecated_and_reads_as_plain(self, src, py, count):
        with pytest.warns(SyntaxWarning, match=ESCAPED_QUOTE_DEPRECATED) as rec:
            dump = stmt_to_dump(src)
        assert dump == ast.dump(ast.parse(py))
        assert [w.category for w in rec] == [SyntaxWarning] * count
        assert all(str(w.message).startswith("f-string: ") for w in rec)

    @pytest.mark.skipif(not T_STRINGS, reason="t-strings need Python 3.14")
    def test_an_escaped_string_in_a_t_string_is_deprecated(self):
        with pytest.warns(
            SyntaxWarning, match="^t-string: " + ESCAPED_QUOTE_DEPRECATED + "$"
        ):
            [stmt] = macroexpand_then_compile(parse(r't"{(g \"a\")}"'))
        [interpolation] = stmt.value.values
        assert ast.dump(interpolation.value) == ast.dump(
            ast.parse("g('a')", mode="eval").body
        )
        # The expression text is the source as written.
        assert interpolation.str == r"(g \"a\")"

    def test_the_deprecation_names_the_file_and_line_of_the_opener(self):
        src = '(print\n  f"{(g \\"a\\")}")'
        with pytest.warns(SyntaxWarning) as rec:
            with compile_errors("demo.lpy", src):
                parse(src)
        [w] = rec
        assert re.fullmatch("f-string: " + ESCAPED_QUOTE_DEPRECATED, str(w.message))
        assert (w.filename, w.lineno) == ("demo.lpy", 2)

    def test_the_deprecation_outside_compile_errors_names_no_file(self):
        # compile_errors resets the file it set when it exits.
        with pytest.warns(SyntaxWarning):
            with compile_errors("demo.lpy", r'f"{(g \"a\")}"'):
                parse(r'f"{(g \"a\")}"')
        with pytest.warns(SyntaxWarning) as rec:
            parse(r'f"{(g \"a\")}"')
        assert [w.filename for w in rec] == ["<unknown>"]

    def test_the_deprecation_made_an_error_is_a_compile_error(self):
        # As CPython turns a compile-time warning a filter makes an error
        # into a SyntaxError at the warning, with the warning's message.
        src = r'f"{(.get d \"k\")}"'
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(LispyCompileError) as info:
                with compile_errors("demo.lpy", src):
                    parse(src)
        err = info.value
        assert re.fullmatch("f-string: " + ESCAPED_QUOTE_DEPRECATED, err.msg)
        assert (err.filename, err.lineno, err.offset) == ("demo.lpy", 1, 12)
        assert (err.end_lineno, err.end_offset) == (1, 14)

    def test_a_partial_parse_never_warns(self):
        # The LSP's parse: a warning made an error would cut its forms short.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            [form] = parse(r'(g f"{(h \"a\")}")', partial=True)
        assert atom_text(form.list[0]) == "g"

    @pytest.mark.parametrize(
        ["src", "kind", "at"],
        [
            # Only \" closes a string opened with \": a bare quote in it, or
            # the end of the source, is an error at the opener.
            pytest.param(r'f"{(g \"a")}"', "f", (1, 7), id="closed-bare"),
            pytest.param(r'f"{(g \"\"\"a""")}"', "f", (1, 7), id="triple-closed-bare"),
            pytest.param(
                r'f"{(g \"\"\"a\"\"")}"', "f", (1, 7), id="triple-closed-mixed"
            ),
            pytest.param(r'f"{(g \"""a""")}"', "f", (1, 7), id="bare-after-opener"),
            pytest.param(r'f"{(g \""")=}"', "f", (1, 7), id="bare-triple-after"),
            pytest.param(r'f"{(g \"say "hi"\")}"', "f", (1, 7), id="bare-content"),
            pytest.param(r'f"{(g \"a)}"', "f", (1, 7), id="literal-quote-ends-it"),
            pytest.param(r'f"{(g \"a', "f", (1, 7), id="unterminated"),
            pytest.param(r'f"{f\"a{x}"}"', "f", (1, 5), id="nested-f-string"),
            pytest.param(
                r't"{(g \"a")}"',
                "t",
                (1, 7),
                marks=pytest.mark.skipif(not T_STRINGS, reason="needs Python 3.14"),
                id="t-string",
            ),
        ],
    )
    def test_an_escaped_string_closes_only_with_an_escaped_quote(self, src, kind, at):
        with pytest.raises(LispyCompileError) as info:
            parse(src)
        assert info.value.msg == (
            f'{kind}-string: a string opened with \\" closes only with \\"'
        )
        assert (info.value.lineno, info.value.offset) == at

    def test_a_bad_prefix_before_an_escaped_quote_in_a_field_is_reported_first(self):
        with pytest.raises(LispyCompileError) as info:
            parse(r'f"{^*\"}"')
        assert info.value.msg == "^* has no meaning on its own"
        assert (info.value.lineno, info.value.offset) == (1, 4)

    @pytest.mark.parametrize(
        ["src", "form"],
        [
            ('f"{x ;c}"', "x"),
            ('f"{(f x ;c\n y)}"', "(f x y)"),
            # Inside brackets a "}" does not end it (CPython reads # alike).
            ('f"{(x ;c}\n)}"', "(x)"),
        ],
        ids=["before-the-closing-brace", "to-the-end-of-the-line", "in-brackets"],
    )
    def test_a_comment_in_a_field_ends_at_a_brace_or_the_line(self, src, form):
        assert str(parse(src)[0].list[2].value) == form


class TestSourceEndingInAFieldExpression:
    @pytest.mark.parametrize(
        ["src", "msg", "line", "col"],
        [
            ('f"{(g "a\\")}"', "'(' was never closed", 1, 4),
            ('f"{(', "'(' was never closed", 1, 4),
            ('f"{x', "'{' was never closed", 1, 3),
            ('(f"{x', "'{' was never closed", 1, 4),
            ('f"{x:{(', "'(' was never closed", 1, 7),
            ('f"{x} {y', "'{' was never closed", 1, 7),
            ('f"{x!r', "'{' was never closed", 1, 3),
            ('f"{x!', "'{' was never closed", 1, 3),
            ('f"{x! ', "'{' was never closed", 1, 3),
            ('f"""\n{\n[\n(x', "'(' was never closed", 4, 1),
            ('f"{x\n\n', "'{' was never closed", 1, 3),
            ('f"ab\\\n{x\n', "'{' was never closed", 2, 1),
            ('f"""{(\n x\n', "'(' was never closed", 1, 6),
            ('f"""{x\r\n y\r\n', "'{' was never closed", 1, 5),
            ('f"ab\\\r\n{x', "'{' was never closed", 2, 1),
            ('rf"a\\\n{x\n', "'{' was never closed", 2, 1),
            ('f"a\\\r{x', "'{' was never closed", 2, 1),
            ('f"""{x\r y\r', "'{' was never closed", 1, 5),
            ('f"""{x\r\n\r\r\n \r', "'{' was never closed", 1, 5),
            ('f"{f"{(', "'(' was never closed", 1, 7),
            pytest.param('t"{(', "'(' was never closed", 1, 4, marks=NEEDS_3_14),
        ],
    )
    def test_the_innermost_open_bracket_is_never_closed(self, src, msg, line, col):
        assert error_position(parse, src) == (msg, line, col)

    def test_a_spec_chain_past_cpython_s_nesting_limit_still_names_its_brace(self):
        # CPython refuses the chain as nested too deeply; lispython adds no
        # nesting limit of its own.
        assert error_position(parse, 'f"{x:{x:{x:{x') == ("'{' was never closed", 1, 12)


class TestSourceEndingInALiteral:
    """The end of the source inside a literal, a format spec, a conversion
    or a plain string."""

    @pytest.mark.parametrize(
        ["src", "msg", "line", "col"],
        [
            ('s = "abc', "unterminated string literal (detected at line 1)", 1, 5),
            ('s = b"abc', "unterminated string literal (detected at line 1)", 1, 5),
            ('s = "a\\\nb', "unterminated string literal (detected at line 2)", 1, 5),
            ('s = "a\\\n', "unterminated string literal (detected at line 1)", 1, 5),
            ('s = "a\\\r\n', "unterminated string literal (detected at line 1)", 1, 5),
            (
                's = """a\r\nb\r\n',
                "unterminated triple-quoted string literal (detected at line 2)",
                1,
                5,
            ),
            (
                's = """a\rb\r',
                "unterminated triple-quoted string literal (detected at line 2)",
                1,
                5,
            ),
            (
                's = """a\\\r\n',
                "unterminated triple-quoted string literal (detected at line 1)",
                1,
                5,
            ),
            ('f"""{("a', "unterminated string literal (detected at line 1)", 1, 7),
            ('f"ab\n', "unterminated f-string literal (detected at line 1)", 1, 1),
            ('f"{x}ab\n', "unterminated f-string literal (detected at line 1)", 1, 1),
            ('s = f"abc', "unterminated f-string literal (detected at line 1)", 1, 5),
            ('s = rf"abc', "unterminated f-string literal (detected at line 1)", 1, 5),
            (
                's = f"""abc\ny\n',
                "unterminated triple-quoted f-string literal (detected at line 2)",
                1,
                5,
            ),
            (
                'x\ns = f"abc\n',
                "unterminated f-string literal (detected at line 2)",
                2,
                5,
            ),
            ('s = f"a\\\n', "unterminated f-string literal (detected at line 1)", 1, 5),
            ('f"a\\\n', "unterminated f-string literal (detected at line 1)", 1, 1),
            (
                'f"""a\\\n',
                "unterminated triple-quoted f-string literal (detected at line 1)",
                1,
                1,
            ),
            ('f"""{f"abc', "unterminated f-string literal (detected at line 1)", 1, 6),
            (
                'f"""{x:abc',
                "unterminated triple-quoted f-string literal (detected at line 1)",
                1,
                1,
            ),
            (
                'f"""{x:',
                "unterminated triple-quoted f-string literal (detected at line 1)",
                1,
                1,
            ),
            ('f"{x:{y}abc', "unterminated f-string literal (detected at line 1)", 1, 1),
            ('f"{x:{y}', "unterminated f-string literal (detected at line 1)", 1, 1),
            (
                'f"{x:{y:{z}a',
                "unterminated f-string literal (detected at line 1)",
                1,
                1,
            ),
            ('f"{x\n}ab', "unterminated f-string literal (detected at line 2)", 1, 1),
            ('f"{x!r}', "unterminated f-string literal (detected at line 1)", 1, 1),
            ('f"{x!r', "'{' was never closed", 1, 3),
            ('f"{x!r ', "'{' was never closed", 1, 3),
            ('f"{x!r\n', "'{' was never closed", 1, 3),
            ('f"{x!r\\', "'{' was never closed", 1, 3),
            ('f"{x!r\\\n', "'{' was never closed", 1, 3),
            ('f"{x!', "'{' was never closed", 1, 3),
            ('f"{x!\\', "'{' was never closed", 1, 3),
            (
                'f"{x! z',
                "f-string: conversion type must come right after the exclamation mark",
                1,
                5,
            ),
            (
                'f"{x! z ',
                "f-string: conversion type must come right after the exclamation mark",
                1,
                5,
            ),
            (
                'f"{x! z\n',
                "f-string: conversion type must come right after the exclamation mark",
                1,
                5,
            ),
            (
                'f"{x! r\\\n',
                "f-string: conversion type must come right after the exclamation mark",
                1,
                5,
            ),
            (
                'f"{x! r ',
                "f-string: conversion type must come right after the exclamation mark",
                1,
                5,
            ),
            ('f"{x:a\\', "unterminated f-string literal (detected at line 1)", 1, 1),
            ('f"{x:\\', "unterminated f-string literal (detected at line 1)", 1, 1),
            ('rf"{x:a\\', "unterminated f-string literal (detected at line 1)", 1, 1),
            ('f"{x!r:\\', "unterminated f-string literal (detected at line 1)", 1, 1),
            ('f"{x:{y:a\\', "unterminated f-string literal (detected at line 1)", 1, 1),
            (
                'f"{x! z:',
                "f-string: conversion type must come right after the exclamation mark",
                1,
                5,
            ),
            (
                'f"{x!z}',
                "f-string: invalid conversion character 'z': expected 's', 'r', or 'a'",
                1,
                6,
            ),
            (
                'f"{x!z}a\nb',
                "f-string: invalid conversion character 'z': expected 's', 'r', or 'a'",
                1,
                6,
            ),
            ('f"{x!r ;', "f-string: expecting ':' or '}'", 1, 8),
            ('f"{x!r (', "f-string: expecting ':' or '}'", 1, 8),
            ('f"{x!r=', "f-string: expecting ':' or '}'", 1, 7),
            ('f"a}', "f-string: single '}' is not allowed", 1, 4),
            ('f"""a}', "f-string: single '}' is not allowed", 1, 6),
            ('f"a}\nb"', "f-string: single '}' is not allowed", 1, 4),
            ('f"a\\"}', "f-string: single '}' is not allowed", 1, 6),
            # A plain string in a field never closes the literal, its quote
            # the literal's or not.
            ('f"{(b"a', UNTERMINATED_STRING.format(1), 1, 5),
            (
                'f"{("""a',
                "unterminated triple-quoted string literal (detected at line 1)",
                1,
                5,
            ),
            (
                'f"""{("""a',
                "unterminated triple-quoted string literal (detected at line 1)",
                1,
                7,
            ),
            ('f"{(rb"a', UNTERMINATED_STRING.format(1), 1, 5),
            # x" is one string token, with the prefix x.
            ('f"ab{x"', UNTERMINATED_STRING.format(1), 1, 6),
            ('f"{("a\nb', UNTERMINATED_STRING.format(2), 1, 5),
            ('f"a{(b"x\nc', UNTERMINATED_STRING.format(2), 1, 6),
            ('x = f"{(\nb"a', UNTERMINATED_STRING.format(2), 2, 1),
            ('f"{x}{\n"a', UNTERMINATED_STRING.format(2), 2, 1),
            ('f"{\n{x}{\n"a', UNTERMINATED_STRING.format(3), 3, 1),
            ('f"{f"abc', "unterminated f-string literal (detected at line 1)", 1, 4),
            (
                'f"{f"""abc',
                "unterminated triple-quoted f-string literal (detected at line 1)",
                1,
                4,
            ),
            ('f"{f"{f"abc', "unterminated f-string literal (detected at line 1)", 1, 7),
            ('f"{x} {"', UNTERMINATED_STRING.format(1), 1, 8),
            pytest.param(
                't"{(t"a',
                "unterminated t-string literal (detected at line 1)",
                1,
                5,
                marks=NEEDS_3_14,
            ),
            pytest.param(
                't"{x!z',
                "t-string: invalid conversion character 'z': expected 's', 'r', or 'a'",
                1,
                6,
                marks=NEEDS_3_14,
            ),
            pytest.param(
                't"{x:a\n}',
                SPEC_NEWLINE.format("t-string"),
                1,
                7,
                marks=NEEDS_3_14,
            ),
        ],
    )
    def test_the_error_is_where_the_source_ends(self, src, msg, line, col):
        assert error_position(parse, src) == (msg, line, col)

    @pytest.mark.parametrize(
        ["src", "msg", "at"],
        [
            # A literal's own text, and a plain string in a field, may span
            # lines: an error after the break is the one where it is.
            ('f"a\n}', "f-string: single '}' is not allowed", (2, 1)),
            ('f"a\n{x!z}"', NOT_CONVERSION.format("z"), (2, 4)),
            ('f"a\n{x!}', "f-string: missing conversion character", (2, 4)),
            ('f"a\n{)', "unmatched ')'", (2, 2)),
            ('f"ab\n{x', "'{' was never closed", (2, 1)),
            ('f"a\n{x:a\n}}', SPEC_NEWLINE.format("f-string"), (2, 5)),
            ('f"a\nb{x:c\n}', SPEC_NEWLINE.format("f-string"), (2, 6)),
            ('f"{f"a\nb"}}', "f-string: single '}' is not allowed", (2, 4)),
            ('f"{"a\nb"}{)}"', "unmatched ')'", (2, 5)),
            ('f"""{"a\nb"}{)}"""', "unmatched ')'", (2, 5)),
            ('f"{"a\r\nb"}{)}"', "unmatched ')'", (2, 5)),
            # The source ends in the outer literal's text.
            (
                'f"{"a\nb"}',
                "unterminated f-string literal (detected at line 2)",
                (1, 1),
            ),
            (
                'f"{f"a\nb"}c',
                "unterminated f-string literal (detected at line 2)",
                (1, 1),
            ),
        ],
    )
    def test_an_error_after_a_literal_s_line_break_is_where_it_is(self, src, msg, at):
        assert error_position(parse, src) == (msg, *at)

    @pytest.mark.parametrize("src", ['f"a\n{\\"b"}"', 'f"a\n{\\"b'])
    def test_an_escaped_string_s_error_stays_ahead_of_a_line_break(self, src):
        # The \" spelling is lispython's own: no compile() text to match.
        assert error_position(parse, src) == (
            'f-string: a string opened with \\" closes only with \\"',
            2,
            2,
        )

    def test_a_literal_with_a_line_break_and_no_error_parses(self):
        assert len(parse('f"{x}a\n}}"')) == 1
        assert len(parse('f"{"a\nb"}"')) == 1

    def test_a_plain_string_continued_by_a_backslash_is_read_on(self):
        assert error_position(parse, 'f"{"a\\\nb"}{x!z}"') == (
            NOT_CONVERSION.format("z"),
            2,
            7,
        )


class TestFieldErrorPosition:
    """An empty field expression is refused at its delimiter (a leading "="
    at the "="), a missing conversion at the token after the "!": never at
    the field's "{" or the "!"."""

    @pytest.mark.parametrize(
        ["src", "msg", "line", "col"],
        [
            ('f"{}"', "f-string: valid expression required before '}'", 1, 4),
            ('f"{ }"', "f-string: valid expression required before '}'", 1, 5),
            ('f"{ = }"', "f-string: valid expression required before '='", 1, 5),
            ('f"{!r}"', "f-string: valid expression required before '!'", 1, 4),
            ('f"{ :x}"', "f-string: valid expression required before ':'", 1, 5),
            ('f"""{\n}"""', "f-string: valid expression required before '}'", 2, 1),
            ('f"""{ \n }"""', "f-string: valid expression required before '}'", 2, 2),
            ('f"{x:{ }}"', "f-string: valid expression required before '}'", 1, 8),
            ('f"{}', "f-string: valid expression required before '}'", 1, 4),
            ('f"{ }', "f-string: valid expression required before '}'", 1, 5),
            ('f"{!', "f-string: valid expression required before '!'", 1, 4),
            ('f"{:', "f-string: valid expression required before ':'", 1, 4),
            pytest.param(
                't"{ }"',
                "t-string: valid expression required before '}'",
                1,
                5,
                marks=NEEDS_3_14,
            ),
        ],
    )
    def test_an_empty_expression_is_refused_at_its_delimiter(self, src, msg, line, col):
        assert error_position(parse, src) == (msg, line, col)

    @pytest.mark.parametrize(
        ["src", "msg", "line", "col"],
        [
            ('f"{=}"', "f-string: valid expression required before '='", 1, 4),
            ('f"{=x}"', "f-string: valid expression required before '='", 1, 4),
            ('f"{=!r}"', "f-string: valid expression required before '='", 1, 4),
            ('f"{=:x}"', "f-string: valid expression required before '='", 1, 4),
            ('f"{=!}"', "f-string: valid expression required before '='", 1, 4),
            ('f"{ =}"', "f-string: valid expression required before '='", 1, 5),
            ('f"x{=}y"', "f-string: valid expression required before '='", 1, 5),
            ('f"{x:{=}}"', "f-string: valid expression required before '='", 1, 7),
            ('f"{=\\\n}"', "f-string: valid expression required before '='", 1, 4),
            ('f"""{\n=}"""', "f-string: valid expression required before '='", 2, 1),
            ('f"{=', "f-string: valid expression required before '='", 1, 4),
            ('f"{=}', "f-string: valid expression required before '='", 1, 4),
            ('f"{x:{=', "f-string: valid expression required before '='", 1, 7),
            ('f"""{=', "f-string: valid expression required before '='", 1, 6),
            pytest.param(
                't"{=}"',
                "t-string: valid expression required before '='",
                1,
                4,
                marks=NEEDS_3_14,
            ),
        ],
    )
    def test_a_leading_equals_is_an_empty_expression(self, src, msg, line, col):
        assert error_position(parse, src) == (msg, line, col)

    def test_a_comment_before_a_leading_equals_emits_no_token(self):
        assert error_position(parse, 'f"""{;c\n=}"""') == (
            "f-string: valid expression required before '='",
            2,
            1,
        )

    @pytest.mark.parametrize(
        ["src", "msg", "line", "col"],
        [
            ('f"{x!}"', "f-string: missing conversion character", 1, 6),
            ('f"{x! }"', "f-string: missing conversion character", 1, 7),
            ('f"{x!  }"', "f-string: missing conversion character", 1, 8),
            ('f"{x! :x}"', "f-string: missing conversion character", 1, 7),
            ('f"""{x!\n}"""', "f-string: missing conversion character", 2, 1),
            ('f"{x!\\\n}"', "f-string: missing conversion character", 2, 1),
            ('f"{x=!}"', "f-string: missing conversion character", 1, 7),
        ],
    )
    def test_a_missing_conversion_is_refused_at_the_token_after_the_bang(
        self, src, msg, line, col
    ):
        assert error_position(parse, src) == (msg, line, col)


class TestFirstErrorInSourceOrder:
    """Only the first error in the source is reported: the lexer stops at
    its first error, and an earlier one the parser finds comes first."""

    @pytest.mark.parametrize(
        ["src", "msg", "at"],
        [
            # Nothing after a literal's error is read.
            ('f"{x!z}" "b', NOT_CONVERSION.format("z"), (1, 6)),
            ('f"{x!z}" )', NOT_CONVERSION.format("z"), (1, 6)),
            ('f"{x!r y)}" "b', "f-string: expecting ':' or '}'", (1, 8)),
            ('f"{x:ab" "a', "f-string: expecting '}', or format specs", (1, 8)),
            ('f"{=}" )', "f-string: valid expression required before '='", (1, 4)),
            # An unmatched bracket before the literal comes first.
            ('(a b)) f"{x!z}"', "unmatched ')'", (1, 6)),
        ],
    )
    def test_the_first_error_is_reported(self, src, msg, at):
        assert error_position(parse, src) == (msg, *at)


class TestConversion:
    """A conversion is s, r or a right after the "!", then only blanks
    before the ":" or "}"; each error is refused where it is."""

    @pytest.mark.parametrize(
        ["src", "msg", "at"],
        [
            ('f"{x!z}"', NOT_CONVERSION.format("z"), (1, 6)),
            ('f"{x!zz}"', NOT_CONVERSION.format("zz"), (1, 6)),
            ('f"{x!z:{y!q}}"', NOT_CONVERSION.format("z"), (1, 6)),
            ('f"{x!f"a"}"', NOT_CONVERSION.format("f"), (1, 6)),
            ('f"{x!"a"}"', "f-string: missing conversion character", (1, 6)),
            ('f"""{x!" """', "f-string: missing conversion character", (1, 8)),
            (r'f"{f\"{x!"r"}\"}"', "f-string: missing conversion character", (1, 10)),
            ('f"{x! r}"', CONVERSION_GAP, (1, 5)),
            ('f"{x!r"a"}"', "f-string: expecting ':' or '}'", (1, 7)),
            ('f"{x!r a}"', "f-string: expecting ':' or '}'", (1, 8)),
            ('f"{x!r (}"', "f-string: expecting ':' or '}'", (1, 8)),
        ],
    )
    def test_a_malformed_conversion_is_refused_where_it_is(self, src, msg, at):
        assert error_position(parse, src) == (msg, *at)

    @pytest.mark.parametrize("src", ['f"{x!r}"', 'f"{x!r }"', 'f"{x!r :>3}"'])
    def test_blanks_may_follow_the_name(self, src):
        [field] = macroexpand_then_compile(parse(src))[0].value.values
        assert field.conversion == ord("r")


class TestQuoteOrLineBreakInFormatSpec:
    """The literal's closing quote, or a single-quoted literal's line break,
    inside a format spec: refused where it is, on every interpreter."""

    @pytest.mark.parametrize(
        ["src", "msg", "line", "col"],
        [
            ('f"ab{x:{y}"', "f-string: expecting '}', or format specs", 1, 11),
            ('f"{x:{y}z"', "f-string: expecting '}', or format specs", 1, 10),
            ('f"{x:a{y}b"', "f-string: expecting '}', or format specs", 1, 11),
            ('f"{x:a}b{y:c"', "f-string: expecting '}', or format specs", 1, 13),
            ('f"{x:ab"', "f-string: expecting '}', or format specs", 1, 8),
            ('f"{x:"', "f-string: expecting '}', or format specs", 1, 6),
            ('f"{x!r:"', "f-string: expecting '}', or format specs", 1, 8),
            ('f"{x!r:{y}"', "f-string: expecting '}', or format specs", 1, 11),
            ('f"{x:{y:a}"', "f-string: expecting '}', or format specs", 1, 11),
            ('f"{x:{y}}{z:"', "f-string: expecting '}', or format specs", 1, 13),
            ('f"""{x:{y}"""', "f-string: expecting '}', or format specs", 1, 11),
            ('f"""{x:a"""', "f-string: expecting '}', or format specs", 1, 9),
            ('f"{x:{y}" + 1', "f-string: expecting '}', or format specs", 1, 9),
            ('f"{x:{y:{z}}"', "f-string: expecting '}', or format specs", 1, 13),
        ],
    )
    def test_the_closing_quote_in_a_spec_is_refused_where_it_is(
        self, src, msg, line, col
    ):
        assert error_position(parse, src) == (msg, line, col)

    @pytest.mark.parametrize(
        ["lpy", "py"],
        [
            # A field is code: a line break stands wherever whitespace may.
            ('f"{\nx\n}"', 'f"{x}"'),
            ('f"{x\n!r}"', 'f"{x!r}"'),
            ('f"{x!r\n}"', 'f"{x!r}"'),
            ('f"{x\n:>3}"', 'f"{x:>3}"'),
            ('f"{x\n=}"', 'f"x\\n={x!r}"'),
            ('f"{x=\n}"', 'f"x=\\n{x!r}"'),
            # A nested field in a spec is code too.
            ('f"{x:{\ny}}"', 'f"{x:{y}}"'),
            ('f"{x:{y\n}}"', 'f"{x:{y}}"'),
            # A triple-quoted literal's spec text reads a line break as text.
            ('f"""{x:a\nb}"""', 'f"{x:a\\nb}"'),
            ('f"""{x:{y}\n}"""', 'f"{x:{y}\\n}"'),
        ],
    )
    def test_a_line_break_in_a_field_s_code_is_whitespace(self, lpy, py):
        assert stmt_to_dump(lpy) == ast.dump(ast.parse(py))

    @pytest.mark.parametrize(
        ["lpy", "py"],
        [
            ('f"{x:a\\\nb}"', 'f"{x:a\\\nb}"'),
            ('f"{x:\\\n}"', 'f"{x:\\\n}"'),
            ('f"{x:\\\r\n}"', 'f"{x:\\\r\n}"'),
            ('f"{x:{y}\\\n}"', 'f"{x:{y}\\\n}"'),
            ('f"{x:\\\\\\\n}"', 'f"{x:\\\\\\\n}"'),
            # A raw spec keeps the backslash and the break as text. Written
            # with escapes, because 3.12's compile() drops that text.
            ('rf"{x:\\\n}"', 'f"{x:\\\\\\n}"'),
            ('rf"{x:a\\\nb}"', 'f"{x:a\\\\\\nb}"'),
        ],
    )
    def test_a_backslash_continued_line_in_spec_text_is_no_line_break(self, lpy, py):
        # An odd run of backslashes before the break continues the line, as
        # Python reads the same source.
        assert stmt_to_dump(lpy) == ast.dump(ast.parse(py))

    @pytest.mark.parametrize(
        ["src", "kind", "at"],
        [
            ('f"{x:\n>3}"', "f-string", (1, 6)),
            ('f"{x:a\n}"', "f-string", (1, 7)),
            ('f"{x:a\nb}"', "f-string", (1, 7)),
            ('f"{x:a\r}"', "f-string", (1, 7)),
            ('f"{x:a\r\n}"', "f-string", (1, 7)),
            ('f"{x!r:a\nb}"', "f-string", (1, 9)),
            # An even run of backslashes escapes only itself, raw or not.
            ('f"{x:\\\\\n}"', "f-string", (1, 8)),
            ('rf"{x:\\\\\n}"', "f-string", (1, 9)),
            # In any segment: after a nested field, before the "}".
            ('f"{x:{y}a\n}"', "f-string", (1, 10)),
            ('f"{x:{y}\n}"', "f-string", (1, 9)),
            ('f"{x:{y:a\n}}"', "f-string", (1, 10)),
            # The first error in the source, the end of it included.
            ('f"{x:abc\n', "f-string", (1, 9)),
            ('f"{x:a\n}{1 1}"', "f-string", (1, 7)),
            pytest.param(
                't"{x:a\n}"',
                "t-string",
                (1, 7),
                marks=pytest.mark.skipif(not T_STRINGS, reason="needs 3.14"),
            ),
        ],
    )
    def test_a_line_break_in_single_quoted_spec_text_is_refused(self, src, kind, at):
        assert error_position(parse, src) == (SPEC_NEWLINE.format(kind), *at)

    @pytest.mark.parametrize(
        ["src", "triple", "at"],
        [
            ('f"{x:abc', "", (1, 1)),
            ('f"{x:', "", (1, 1)),
            ('f"{x!r:', "", (1, 1)),
            ('f"{x:{y:ab', "", (1, 1)),
            ('f"""{x:a\nb', "triple-quoted ", (1, 1)),
        ],
    )
    def test_the_source_ending_in_spec_text_leaves_the_literal_open(
        self, src, triple, at
    ):
        line = src.count("\n") + 1
        assert error_position(parse, src) == (
            f"unterminated {triple}f-string literal (detected at line {line})",
            *at,
        )

    def test_a_line_break_right_after_the_exclamation_mark_is_refused(self):
        assert error_position(parse, 'f"{x!\nr}"') == (
            "f-string: conversion type must come right after the exclamation mark",
            1,
            5,
        )


def _read_and_compile(src):
    return macroexpand_then_compile(parse(src))


def _spec_parts(field):
    """A compiled field's format spec as a list: text as str, a nested field
    as (its expression's name, its own spec parts or None)."""
    parts = []
    for value in field.format_spec.values:
        if isinstance(value, ast.Constant):
            parts.append(value.value)
        else:
            nested = None if value.format_spec is None else _spec_parts(value)
            parts.append((value.value.id, nested))
    return parts


class TestBraceAfterNestedFieldInSpec:
    """In a format spec, after a nested field has opened, "{{" is a literal
    "{" and "}" ends the spec even doubled; before one, every "{" opens a
    field. The same reading on every interpreter (3.11's compile() opens a
    field at every "{" in a spec)."""

    @pytest.mark.parametrize(
        ["src", "spec", "after"],
        [
            ('f"{x:{y}{{}"', [("y", None), "{"], []),
            ('f"{x:{y}{z}{{}"', [("y", None), ("z", None), "{"], []),
            ('f"{x:{y}{{{z}}"', [("y", None), "{", ("z", None)], []),
            ('f"{x:{y!r}{{}"', [("y", None), "{"], []),
            ('f"""{x:{y}{{}"""', [("y", None), "{"], []),
            ('f"{x:{y}{{ }"', [("y", None), "{ "], []),
            # "}" ends the spec, then "}}" is the literal's doubled brace.
            ('f"{x:{y}{{}}}"', [("y", None), "{"], ["}"]),
            ('f"{x:{y}}}}"', [("y", None)], ["}"]),
            ('f"{x:a}}}"', ["a"], ["}"]),
        ],
    )
    def test_a_doubled_brace_after_a_nested_field_is_text(self, src, spec, after):
        [stmt] = _read_and_compile(src)
        [field, *rest] = stmt.value.values
        assert _spec_parts(field) == spec
        assert [value.value for value in rest] == after

    @pytest.mark.parametrize(
        ["src", "col"],
        [
            ('f"{x:{y}a{{b}}c}"', 14),
            ('f"{x:{y}{{{z}}}"', 15),
            ('f"{x:{y:{z}{{}}}"', 16),
            ('f"{x:{y}{{}}"', 12),
            ('f"{x:{y}}}"', 10),
        ],
    )
    def test_a_doubled_closing_brace_after_a_nested_field_is_not_text(self, src, col):
        assert error_position(parse, src) == (
            "f-string: single '}' is not allowed",
            1,
            col,
        )


class TestNumberLexingCost:
    """A long digit run that is not a number costs time linear in its length."""

    # Quadratic, 8000 digits take ~5 s; linear, a few ms. One run, with a
    # budget two orders above the linear cost.
    DIGITS = 8000
    BUDGET = 1.0

    @pytest.mark.parametrize(
        "template",
        ["{}", 'f"{{{}}}"', 'f"{{[{}]}}"'],
        ids=["top-level", "field", "bracketed-field"],
    )
    @pytest.mark.parametrize("tail", ["a(", "-", "("])
    def test_digits_before_a_non_number_char_lex_in_linear_time(self, template, tail):
        src = template.format("1" * self.DIGITS + tail)
        start = time.perf_counter()
        tokenize(src)  # only the lexing cost is measured, not validity
        assert time.perf_counter() - start < self.BUDGET

    @pytest.mark.parametrize(
        "src, kind",
        [
            ("1" * 50 + "a", "number"),
            ("1__2", "symbol"),
            ("1_000", "number"),
            ("1e5", "number"),
            ("1j", "number"),
            ("1e5j", "number"),
            ("1_0.5e-3+2_0j", "number"),
            ("0xFF", "number"),
            ("1x_2", "number"),
            ("1.5a", "symbol"),
        ],
    )
    def test_digit_tokens_keep_their_kind(self, src, kind):
        [[text, token_kind, *_]] = tokenize(src)
        assert (token_kind, text) == (kind, src)
