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
IN_PLACE = features.fstring_fields_are_tokens()


def error_position(read, src):
    with pytest.raises(SyntaxError) as info:
        read(src)
    return (info.value.msg, info.value.lineno, info.value.offset)


def python_error_position(src):
    return error_position(lambda s: compile(s, "<s>", "exec"), src)


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

    @pytest.mark.skipif(not IN_PLACE, reason="3.11 ends the literal at the last quote")
    @pytest.mark.parametrize("src", ['f"{x:{{"', 'f"{x:{{}"'])
    def test_an_unclosed_doubled_brace_in_a_spec_is_a_field(self, src):
        # The quote is a string in the inner field's expression.
        assert error_position(parse, src) == python_error_position(src)

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

    @pytest.mark.skipif(not IN_PLACE, reason="3.11 ends the literal at the last quote")
    def test_a_string_after_a_deep_chain_is_read_where_it_is(self):
        # CPython refuses the chain as nested too deeply, so the expected
        # error is that of a field after one with no spec, at the quote.
        [msg, _, _] = python_error_position('f"{x} {"')
        assert error_position(parse, 'f"{x:{x:{x:{x}}}} {"') == (msg, 1, 20)

    def test_the_first_of_two_errors_is_reported(self):
        # CPython's message, at the name.
        with pytest.raises(LispyCompileError) as info:
            parse('f"{x!z} {x:{x:{x:{x}}} }"')
        assert info.value.msg == (
            "f-string: invalid conversion character 'z': expected 's', 'r', or 'a'"
        )
        assert (info.value.lineno, info.value.offset) == (1, 6)

    @pytest.mark.parametrize(
        ["src", "msg", "at"],
        [
            # Not the closing quote: a string where the name should be.
            ('f"""{x!" """', "f-string: invalid conversion character", (1, 8)),
            # The conversion slot is no body text: a closed escaped literal
            # with a string there is the same error, not the quote mix.
            (r'f"{f\"{x!"r"}\"}"', "f-string: invalid conversion character", (1, 10)),
        ],
        ids=[
            "string-in-triple",
            "escaped-literal-string",
        ],
    )
    def test_a_literal_ending_after_the_exclamation_mark(self, src, msg, at):
        with pytest.raises(LispyCompileError) as info:
            parse(src)
        assert info.value.msg == msg
        assert (info.value.lineno, info.value.offset) == at

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
        ["lpy", "py"],
        [
            # The bare-quoted string takes \" as an escape, so it swallows )}
            # and the closing quote. (g, ...) keeps the "(" in lispython's
            # column; 3.11 lexes that Python text as other tokens.
            pytest.param(
                r'f"{(g "a\")}"',
                r'f"{(g, "a\")}"',
                marks=pytest.mark.skipif(
                    not IN_PLACE, reason="3.11 reads the Python text differently"
                ),
            ),
            ('f"{(', None),
            ('f"{x', None),
            ('(f"{x', None),
            ('f"{x:{(', None),
            ('f"{x} {y', None),
            ('f"{x!r', None),
            ('f"{x!', None),
            ('f"{x! ', None),
            ('f"""\n{\n[\n(x', None),
            ('f"{x\n\n', None),
            ('f"ab\\\n{x\n', None),
            # 3.11's line rules: a final \n starts no line, a final \r\n does;
            # a backslash escapes a whole \r\n, and escapes in a raw string too.
            ('f"""{(\n x\n', None),
            ('f"""{x\r\n y\r\n', None),
            ('f"ab\\\r\n{x', None),
            ('rf"a\\\n{x\n', None),
            # A lone CR ends a line too (CRLF is one break).
            ('f"a\\\r{x', None),
            ('f"""{x\r y\r', None),
            ('f"""{x\r\n\r\r\n \r', None),
            pytest.param(
                'f"{f"{(',
                None,
                marks=pytest.mark.skipif(
                    not IN_PLACE, reason="3.11 ends the string at the inner quote"
                ),
            ),
            pytest.param(
                't"{(',
                None,
                marks=pytest.mark.skipif(not T_STRINGS, reason="needs Python 3.14"),
            ),
        ],
        ids=[
            "bare-string-swallows-the-field",
            "innermost-bracket",
            "field-brace",
            "inside-a-bracket",
            "spec-field",
            "second-field",
            "conversion",
            "exclamation-mark",
            "space-after-the-exclamation-mark",
            "triple-multiline",
            "single-newline",
            "backslash-newline",
            "final-lf",
            "final-crlf",
            "backslash-crlf",
            "raw-backslash-newline",
            "backslash-lone-cr",
            "triple-lone-cr",
            "mixed-breaks",
            "nested-literal",
            "t-string",
        ],
    )
    def test_the_error_is_cpython_s(self, lpy, py):
        assert error_position(parse, lpy) == python_error_position(py or lpy)

    @pytest.mark.parametrize(
        ["src", "in_place", "whole_literal"],
        [
            # CPython 3.12+ refuses the chain as nested too deeply; lispython
            # adds no nesting limit of its own.
            ('f"{x:{x:{x:{x', (1, 12), (1, 1)),
        ],
        ids=["past-the-nesting-limit"],
    )
    def test_where_cpython_differs_the_field_rule_still_holds(
        self, src, in_place, whole_literal
    ):
        expected = (
            ("'{' was never closed", *in_place)
            if IN_PLACE
            else ("unterminated string literal (detected at line 1)", *whole_literal)
        )
        assert error_position(parse, src) == expected

    @pytest.mark.parametrize("in_place", [True, False], ids=["3.12+", "3.11"])
    @pytest.mark.parametrize(
        ["src", "errors"],
        [
            (
                '(f"{x',
                {
                    True: ("'{' was never closed", 1, 4),
                    False: ("unterminated string literal (detected at line 1)", 1, 2),
                },
            ),
            (
                'f"""{(\n x\n',
                {
                    True: ("'(' was never closed", 1, 6),
                    False: (
                        "unterminated triple-quoted string literal "
                        "(detected at line 2)",
                        1,
                        1,
                    ),
                },
            ),
            (
                'f"""{x\r\n y\r\n',
                {
                    True: ("'{' was never closed", 1, 5),
                    False: (
                        "unterminated triple-quoted string literal "
                        "(detected at line 3)",
                        1,
                        1,
                    ),
                },
            ),
            (
                'f"ab\\\r\n{x',
                {
                    True: ("'{' was never closed", 2, 1),
                    False: ("unterminated string literal (detected at line 2)", 1, 1),
                },
            ),
            (
                'rf"a\\\n{x\n',
                {
                    True: ("'{' was never closed", 2, 1),
                    False: ("unterminated string literal (detected at line 2)", 1, 1),
                },
            ),
        ],
        ids=["single", "triple-final-lf", "final-crlf", "backslash-crlf", "raw"],
    )
    def test_each_tokenizer_s_error_is_produced(
        self, monkeypatch, in_place, src, errors
    ):
        # Both branches on every interpreter: CPython's measured messages.
        monkeypatch.setattr(features, "fstring_fields_are_tokens", lambda: in_place)
        # 3.11, the one tokenizer counting a line after a final \r\n.
        monkeypatch.setattr(features, "final_crlf_starts_a_line", lambda: not in_place)
        assert error_position(parse, src) == errors[in_place]

    def test_3_11_ends_an_unescaped_string_line_at_a_lone_cr(self, monkeypatch):
        # Measured on 3.11's compile(): the line ends at the CR, so it is line
        # 1 (a backslash-CR continues it, which the CPython-oracle test covers).
        # Pins the body regex together with the locator's CR counting.
        monkeypatch.setattr(features, "fstring_fields_are_tokens", lambda: False)
        assert error_position(parse, 'f"ab\r{x') == (
            "unterminated string literal (detected at line 1)",
            1,
            1,
        )


def _read_on_3_12_plus(srcs):
    """Sources 3.11 reads otherwise: it ends a literal at the first quote of
    its own kind, so a same-quote string or literal in a field closes it,
    and it lexes a whole literal before reading its fields."""
    mark = pytest.mark.skipif(
        not IN_PLACE, reason="3.11 ends a literal at its first same quote"
    )
    return [pytest.param(src, marks=mark) for src in srcs]


def _python_error_position_spelled(src):
    # 3.12 spells one message "exclamanation"; lispython keeps the corrected
    # spelling CPython 3.13+ use.
    [msg, *at] = python_error_position(src)
    return (msg.replace("exclamanation", "exclamation"), *at)


class TestSourceEndingInALiteral:
    """The end of the source inside a literal, a format spec, a conversion
    or a plain string: CPython's error on the running interpreter."""

    @pytest.mark.parametrize(
        "src",
        [
            # A plain string, no literal open.
            's = "abc',
            's = b"abc',
            's = "a\\\nb',
            's = "a\\\n',
            's = "a\\\r\n',
            's = """a\r\nb\r\n',
            's = """a\rb\r',
            's = """a\\\r\n',
            # A string in a field, of another quote than the literal's.
            'f"""{("a',
            'x = f"{(\nb"a',
            'f"{x}{\n"a',
            'f"{\n{x}{\n"a',
            # A line break in a single-quoted literal's text.
            'f"ab\n{x',
            'f"ab\r{x',
            'f"ab\r\n{x',
            'f"ab\n',
            'f"{x}ab\n',
            'f"{x}ab\n{y',
            'f"{x:{y}a\nb',
            # ... or in a format spec.
            'f"{x:abc\n',
            'f"{x:abc\ny',
            'f"{x:a\n{y}',
            'f"{x:\n',
            'f"{x:abc\n(',
            'f"{x:abc\n\ny',
            'f"{x:abc\n ',
            'f"{x:abc\n{y',
            'f"{x:a\\\nb',
            'f"{x:\n\n{y}a\nb',
            'f"{x:ab\n{y\n',
            # Literal text.
            's = f"abc',
            's = rf"abc',
            's = f"""abc\ny\n',
            'x\ns = f"abc\n',
            's = f"a\\\n',
            'f"a\\\n',
            'f"""a\\\n',
            'f"""{f"abc',
            'f"""{x:abc',
            'f"""{x:',
            'f"{x:{y}abc',
            'f"{x:{y}',
            'f"{x:{y:{z}a',
            'f"{x\n}ab',
            'f"{x!r}',
            # A format spec's first segment.
            'f"{x:abc',
            'f"{x:',
            'f"{x:{y:ab',
            'f"{x!r:',
            'f"{x!r:a\n',
            # A conversion.
            'f"{x!r',
            'f"{x!r ',
            'f"{x!r\n',
            'f"{x!r\\',
            'f"{x!r\\\n',
            'f"{x!',
            'f"{x!\\',
            'f"{x! z',
            'f"{x! z ',
            'f"{x! z\n',
            'f"{x! r\\\n',
            'f"{x! r ',
            'f"{x!z',
            'f"{x!z ',
            'f"{x!z\n',
            'f"{x!zz',
            'f"{x!z\\',
            'f"{x!z:',
            'f"{x!z:{y',
            # A backslash escaping the end of the source in a spec.
            'f"{x:a\\',
            'f"{x:\\',
            'f"{x:\\\n',
            'f"{x:\\\r\n',
            'f"{x:a\\\r',
            'rf"{x:a\\',
            'f"{x!r:\\',
            'f"{x:{y:a\\',
            'f"{x:a\\\n',
            'f"{x:a\\\nb\\',
            'f"{x:a\\\nb',
            'f"{x:a\\\nb\n',
            'f"{x:a\\b',
            'f"{x:a\\\\',
            # After a spec's line break 3.12 skips comments and continuations.
            'f"{x:a\n#c',
            'f"{x:a\n\\',
            'f"{x:a\n\\\n',
            'f"{x:a\n#c\nb',
            'f"{x:a\n\\\nb',
            # An error inside a literal 3.11 finds unterminated as a whole.
            'f"{x! z:',
            'f"{x!z}',
            'f"{x!z}a\nb',
            'f"{x!r ;',
            'f"{x!r (',
            'f"{x!r=',
            'f"{x!z (',
            'f"{x! z',
            'f"a}',
            'f"""a}',
            'f"a}\nb"',
            'f"a\\"}',
            'f"""{x!""',
            # A line break in a field or literal that closed before the end.
            'f"{x:a\n}',
            'f"{x:a\n}b',
            'f"{x:a\n#c\n}',
            'f"{x:a\n}{y:b\n}',
            'f"{x:{y:a\n}}',
            'f"{x:a\n}\n',
            'f"{x:a\n}b\nc',
            'f"{x:a\n}{',
            'f"{x:a\n}{y',
            'f"{x:{y:a\n}',
            'f"ab{x:a\n}',
            'f"a\nb{x:c\n}',
            'f"a\nb{x:c\n',
            'f"{x:a\n}{y:c\n',
            'f"{x:a\nb}',
            'f"{x:a\nb}c',
            'f"{x:a\n b}',
            'f"{x:a\n{y}}',
            'f"{x:a\n{y}b}',
            'f"{x:a\n}{f"b\nc"}',
            *_read_on_3_12_plus(
                [
                    'f"{(b"a',
                    'f"{("""a',
                    'f"""{("""a',
                    'f"{(rb"a',
                    'f"ab{x"',
                    'f"{("a\nb',
                    'f"a{(b"x\nc',
                    'f"{f"abc',
                    'f"{f"""abc',
                    'f"{f"{f"abc',
                    'f"{f"{x:ab',
                    'f"{x!r"',
                    'f"{x!b"',
                    'f"{x!rb"',
                    'f"{x!r "',
                    'f"{x! r"',
                    'f"{x!f"',
                    'f"{x!f"a',
                    'f"{x!t"',
                    'f"{x!z"',
                    'f"{x! z"',
                    'f"{x! r "',
                    'f"{x!"',
                    'f"{x! "',
                    'f"{x!\n"',
                    'f"{x} {"',
                    'f"{f"a\nb"}',
                    'f"{f"a\nb"}c',
                    'f"{f"{x:a\n}"}',
                    'f"{f"{x:a\n}"',
                    'f"{x:{f"{y:a\n}"}}',
                    'f"{f"a\nb"}{x:c\n}',
                    'f"{f"a\nb{x:c\n}',
                    'f"""a\n{f"b\nc{x:d\n}"',
                ]
            ),
            *[
                pytest.param(
                    src,
                    marks=pytest.mark.skipif(not T_STRINGS, reason="needs 3.14"),
                )
                for src in ['t"{x:abc', 't"ab\n{x', 't"{(t"a', 't"{x!z', 't"{x:a\n}']
            ],
        ],
    )
    def test_the_error_is_cpython_s(self, src):
        assert error_position(parse, src) == _python_error_position_spelled(src)

    @pytest.mark.parametrize(
        "src",
        [
            # Lexed errors, the source ending in the literal or not.
            'f"a\n}',
            'f"a\n}"',
            'f"a\nb}',
            'f"{x:{y}a\n}}',
            'f"a\n{x:a\n}}',
            'f"a\n{x:a\n"',
            # A conversion checked when its field closes.
            'f"a\n{x!z}',
            'f"a\n{x!z}"',
            'f"a\n{x=!z}',
            'f"a\n{x!}',
            # Raised by the parser.
            'f"a\n{)',
            'f"a\n{x:{)',
            'f"a\n{x!r:{)',
            # An error before the break stays.
            'f"{x!z}a\n}',
            *_read_on_3_12_plus(['f"{f"a\nb"}}', 'f"{f"a\nb"} \n}']),
        ],
    )
    def test_an_error_after_a_literal_s_line_break_is_that_break_s(self, src):
        assert error_position(parse, src) == _python_error_position_spelled(src)

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

    @pytest.mark.skipif(not IN_PLACE, reason="3.11 reads its own messages here")
    @pytest.mark.parametrize(
        "src",
        [
            'f"{x!z}"',
            'f"{x!z:>3}"',
            'f"{x!z:{y!q}}"',
            'f"{x!r"a" }"',
            'f"{x!r "a"}"',
            'f"{x!z "a"}"',
        ],
    )
    def test_a_closed_field_s_conversion_is_cpython_s(self, src):
        assert error_position(parse, src) == _python_error_position_spelled(src)

    @pytest.mark.parametrize(
        ["src", "triple"],
        [
            ('f"{}', ""),
            ('f"{x!}', ""),
            ('f"{a b}', ""),
            ('f"{x!z:{y!q}}', ""),
            ("f'''{}", "triple-quoted "),
        ],
    )
    def test_3_11_reports_an_unclosed_whole_literal_first(
        self, monkeypatch, src, triple
    ):
        # Measured on 3.11's compile(): the error inside (an empty field, a
        # missing conversion, ...) never comes. The 3.11 branch runs on
        # every interpreter, whose own columns for those errors differ.
        monkeypatch.setattr(features, "fstring_fields_are_tokens", lambda: False)
        assert error_position(parse, src) == (
            f"unterminated {triple}string literal (detected at line 1)",
            1,
            1,
        )

    def test_3_11_reads_inside_a_literal_its_token_closes(self, monkeypatch):
        # 3.11's token ends at the first same quote, so the error inside
        # stands (a single "}" is one on every interpreter).
        monkeypatch.setattr(features, "fstring_fields_are_tokens", lambda: False)
        with pytest.raises(LispyCompileError) as info:
            parse('f"a}"')
        assert info.value.msg == "f-string: single '}' is not allowed"

    def test_a_spec_newline_error_exists_only_where_fields_are_tokens(self):
        # 3.11 refuses the probe's text too, as an unterminated string.
        assert features.fstring_fields_are_tokens() or not (
            features.single_quoted_spec_refuses_newline()
        )

    NEWLINE_IN_SPEC = (
        "f-string: newlines are not allowed in format specifiers for single "
        "quoted f-strings"
    )
    NOT_CONVERSION = (
        "f-string: invalid conversion character {!r}: expected 's', 'r', or 'a'"
    )

    @pytest.mark.parametrize(
        ["probe", "src", "errors"],
        [
            (
                "single_quoted_spec_refuses_newline",
                'f"{x:abc',
                {True: (NEWLINE_IN_SPEC, 1, 9), False: ("'{' was never closed", 1, 3)},
            ),
            (
                "conversion_name_checked_first",
                'f"{x!z:',
                {
                    True: (NOT_CONVERSION.format("z"), 1, 6),
                    False: ("'{' was never closed", 1, 3),
                },
            ),
            (
                "conversion_name_checked_first",
                'f"{x!z:{y!q}}"',
                {
                    True: (NOT_CONVERSION.format("z"), 1, 6),
                    False: (NOT_CONVERSION.format("q"), 1, 11),
                },
            ),
            (
                "expecting_brace_names_its_line",
                'x = f"{(\nb"a',
                {
                    True: ("f-string: expecting '}' to close '{' on line 1", 2, 1),
                    False: ("f-string: expecting '}'", 2, 1),
                },
            ),
            (
                "final_crlf_starts_a_line",
                's = """a\r\nb\r\n',
                {
                    True: (
                        "unterminated triple-quoted string literal "
                        "(detected at line 3)",
                        1,
                        5,
                    ),
                    False: (
                        "unterminated triple-quoted string literal "
                        "(detected at line 2)",
                        1,
                        5,
                    ),
                },
            ),
        ],
        ids=[
            "spec-newline",
            "conversion-then-spec",
            "conversion-in-spec",
            "line",
            "crlf",
        ],
    )
    @pytest.mark.parametrize("value", [True, False])
    def test_each_tokenizer_s_error_is_produced(
        self, monkeypatch, probe, src, errors, value
    ):
        # Both branches on every interpreter, the other probes at 3.12's
        # values: CPython's measured messages.
        for name in [
            "single_quoted_spec_refuses_newline",
            "conversion_name_checked_first",
            "expecting_brace_names_its_line",
            "final_crlf_starts_a_line",
        ]:
            monkeypatch.setattr(features, name, lambda: False)
        monkeypatch.setattr(features, "fstring_fields_are_tokens", lambda: True)
        monkeypatch.setattr(features, probe, lambda: value)
        # 3.12 reads a spec on after its line break, 3.13+ refuse the break.
        refuses = features.single_quoted_spec_refuses_newline()
        monkeypatch.setattr(
            features, "spec_line_break_returns_to_code", lambda: not refuses
        )
        assert error_position(parse, src) == errors[value]


class TestFieldErrorPosition:
    """Where CPython 3.12+ place an empty field expression or a missing
    conversion: at the delimiter or the token after the "!", not at the
    field's "{" or the "!". (3.11 places its own messages elsewhere in a
    closed literal; a source ending in one is the whole literal's error.)"""

    @pytest.mark.parametrize(
        "src",
        [
            *_read_on_3_12_plus(
                [
                    'f"{}"',
                    'f"{ }"',
                    'f"{ = }"',
                    'f"{!r}"',
                    'f"{ :x}"',
                    'f"""{\n}"""',
                    'f"""{ \n }"""',
                    'f"{x:{ }}"',
                ]
            ),
            'f"{}',
            'f"{ }',
            'f"{!',
            'f"{:',
            pytest.param(
                't"{ }"', marks=pytest.mark.skipif(not T_STRINGS, reason="needs 3.14")
            ),
        ],
    )
    def test_an_empty_expression_is_refused_at_its_delimiter(self, src):
        assert error_position(parse, src) == python_error_position(src)

    @pytest.mark.parametrize(
        "src",
        _read_on_3_12_plus(
            [
                'f"{x!}"',
                'f"{x! }"',
                'f"{x!  }"',
                'f"{x! :x}"',
                'f"""{x!\n}"""',
                'f"{x!\\\n}"',
                'f"{x=!}"',
            ]
        ),
    )
    def test_a_missing_conversion_is_refused_at_the_token_after_the_bang(self, src):
        assert error_position(parse, src) == python_error_position(src)


class TestTokenizerErrorAfterAParserError:
    """After an f/t-string error of CPython's parser, CPython tokenizes the
    rest of the source, and a tokenizer error raised with no literal open
    (3.11: outside the literal holding the parser's) is reported instead."""

    @pytest.mark.parametrize(
        "src",
        [
            'f"{x!z}" "b',
            'f"""{x!z}""""',
            'f"{x!}" "b',
            'f"{x!:}" "b',
            'f"{}" "b',
            'f"{ }" "b',
            'f"{!r}" "b',
            'f"{:x}" "b',
            'f"{x! r}" "b',
            'f"{x!r a}" "b',
            'f"{x!r"a"}" "b',
            'f"{x!"a"}" "b',
            'f"{x!f"a"}" "b',
            'f"{x!z:a}" "b',
            'f"{x!z:{y!q}}" "b',
            'f"{x!z}" rb"b',
            'f"{x!z}" b"""b',
            'f"{x!z}" x"b',
            'f"{x!z}" "a" "b',
            'f"{x!z}" f"{y}" "b',
            'f"{x!z}" f"{y!q}" "b',
            'f"{x!z}"\n\n"b',
            'f"{x!z}" \\\n"b',
            '"a" f"{x!z}" "b',
            # lispython's "expecting a single expression" is CPython's
            # parser error for the same text too.
            'f"{a b}" "c',
            'f"{(a) b}" "c',
        ],
    )
    def test_a_later_tokenizer_error_outside_a_literal_is_reported(self, src):
        assert error_position(parse, src) == python_error_position(src)

    CLOSED = pytest.mark.skipif(
        not IN_PLACE, reason="3.11 reads its own messages in a closed literal"
    )

    @pytest.mark.parametrize(
        ["lpy", "py"],
        [
            # 3.11 reports these later literals' whole tokens (below).
            ('f"{x!z}" f"b', None),
            ('f"{x!z}" f"{y', None),
            ('f"{x!z}" f"{y\n"b', None),
            ('f"{x!}', None),
            pytest.param('f"{x!z}" f"""b', None, marks=CLOSED),
            pytest.param('f"{f"{x!z}" "b"}"', None, marks=CLOSED),
            # A comment: lispython's ";" is Python's "#".
            pytest.param('f"{x!z}" ;"b', 'f"{x!z}" #"b', marks=CLOSED),
        ],
    )
    def test_the_parser_error_stands_otherwise(self, lpy, py):
        assert error_position(parse, lpy) == python_error_position(py or lpy)

    # 3.11 differs from lispython on these, apart from the bracket: it checks
    # a nested f-string's brackets, and refuses a single-quoted f-string's
    # line break as its whole token unterminated.
    NOT_ON_3_11 = pytest.mark.skipif(not IN_PLACE, reason="a separate 3.11 gap")

    @pytest.mark.parametrize(
        "src",
        [
            # After the parser error, a closing bracket with none open in the
            # field, or of the wrong kind, stops 3.12+'s tokenizer inside the
            # literal: the parser error stands, never the "b after it.
            'f"{x!r y)}" "b',
            'f"{x!r y]}" "b',
            'f"{x!r (]}" "b',
            'f"{x!z (]}" "b',
            'f"{x!r y(]}" "b',
            'f"{x!r [)]}" "b',
            'f"{(f"{x!r y}" ]}" "b',
            'f"""{x!r (\n]}""" "b',
            'f"{}{)}" "c',
            'f"{x!z}{)}" "c',
            'f"{x!}{)}" "c',
            pytest.param('f"{x!r f"{)}"}" "b', marks=NOT_ON_3_11),
            # Right after a conversion it is the parser's next token: the
            # tokenizer raises first (3.14+ check an invalid name before).
            'f"{x!r )}" "b',
            'f"{x!z )}" "b',
            'f"{x!z ]}" "b',
            'f"{x! )}" "b',
            'f"{x!r ]}" "b',
            'f"{x!r )}"\n"b',
            'f"{x!r\\\n)}" "b',
            'f"{x!r \\q}" "b',
            # After the code 3.12 reads in a spec past a line break.
            pytest.param('f"{x:a\nb)}" "c', marks=NOT_ON_3_11),
            pytest.param('f"{x:a\nb]}" "c', marks=NOT_ON_3_11),
        ],
    )
    def test_a_closing_bracket_out_of_place_in_a_field(self, src):
        assert error_position(parse, src) == python_error_position(src)

    @pytest.mark.parametrize("in_place", [True, False], ids=["3.12+", "3.11"])
    @pytest.mark.parametrize("src", ['f"{x!z}" f"b', 'f"{x!z}" f"{y'])
    def test_3_11_reports_a_later_literal_s_whole_token(
        self, monkeypatch, in_place, src
    ):
        # Measured on 3.12 and 3.11: a later literal that never closes is a
        # tokenizer error inside a literal on 3.12+, one string token on 3.11.
        monkeypatch.setattr(features, "fstring_fields_are_tokens", lambda: in_place)
        assert error_position(parse, src) == (
            (TestSourceEndingInALiteral.NOT_CONVERSION.format("z"), 1, 6)
            if in_place
            else ("unterminated string literal (detected at line 1)", 1, 10)
        )


SPEC_REFUSES_NEWLINE = features.single_quoted_spec_refuses_newline()
SPEC_BREAK_TO_CODE = features.spec_line_break_returns_to_code()


class TestQuoteOrLineBreakInFormatSpec:
    """The literal's closing quote, or a single-quoted literal's line break,
    inside a format spec: CPython's error on the running interpreter."""

    @pytest.mark.skipif(not IN_PLACE, reason="3.11 reads the whole literal first")
    @pytest.mark.parametrize(
        "src",
        [
            'f"ab{x:{y}"',
            'f"{x:{y}z"',
            'f"{x:a{y}b"',
            'f"{x:a}b{y:c"',
            'f"{x:ab"',
            'f"{x:"',
            'f"{x!r:"',
            'f"{x!r:{y}"',
            'f"{x:{y:a}"',
            'f"{x:{y}}{z:"',
            'f"""{x:{y}"""',
            'f"""{x:a"""',
            'f"{x:{y}" + 1',
            'f"{x:{y:{z}}"',
        ],
    )
    def test_the_closing_quote_in_a_spec_is_cpython_s(self, src):
        assert error_position(parse, src) == python_error_position(src)

    @pytest.mark.skipif(
        not SPEC_REFUSES_NEWLINE, reason="3.12 reads the spec after it as code"
    )
    @pytest.mark.parametrize(
        "src",
        [
            'f"{x:ab\n}"',
            'f"{x:\n}"',
            'f"{x!r:a\nb}"',
            'f"{x:{y:\n}}"',
            'f"{x:ab\r}"',
            'f"{x:ab\r\n}"',
            'f"{x:ab\n"',
            'f"{x:ab\n}" f"{"',
            'f"{x:a\n}{1 1}"',
            'f"{f"{x:a\n}"}"',
            'f"{x:a\n{y}}"',
            pytest.param(
                't"{x:ab\n}"',
                marks=pytest.mark.skipif(not T_STRINGS, reason="needs 3.14"),
            ),
        ],
    )
    def test_a_line_break_in_a_spec_is_refused_where_it_is(self, src):
        assert error_position(parse, src) == python_error_position(src)

    NEWLINE_IN_SPEC = TestSourceEndingInALiteral.NEWLINE_IN_SPEC

    @pytest.mark.parametrize(
        ["src", "at"],
        [
            ('f"{x:ab\n}"', (1, 8)),
            ('f"{x:a\n}{1 1}"', (1, 7)),
        ],
    )
    def test_3_13_refuses_a_spec_line_break(self, monkeypatch, src, at):
        # 3.13's measured messages, on every interpreter.
        monkeypatch.setattr(features, "fstring_fields_are_tokens", lambda: True)
        monkeypatch.setattr(
            features, "single_quoted_spec_refuses_newline", lambda: True
        )
        monkeypatch.setattr(features, "spec_line_break_returns_to_code", lambda: False)
        assert error_position(parse, src) == (self.NEWLINE_IN_SPEC, *at)

    # 3.12 ends a single-quoted spec's text at a line break in its first
    # segment and reads the rest of the field as code.
    CODE_AFTER_BREAK = pytest.mark.skipif(
        not SPEC_BREAK_TO_CODE, reason="3.12 reads the spec after it as code"
    )
    # [lispython source, Python source building the AST 3.12 builds for it]
    CODE_SPECS = [
        ('f"{x:ab\n}"', 'f"{x:ab}"'),
        ('f"{x:ab\n  \n}"', 'f"{x:ab}"'),
        ('f"{x:ab\n{y}}"', 'f"{x:ab{y}}"'),
        ('f"{x:ab\n {y} {z} \n}"', 'f"{x:ab{y}{z}}"'),
        ('f"{x:ab\n{y:}}"', 'f"{x:ab{y:}}"'),
        ('f"{x:ab\n{y:\n}}"', 'f"{x:ab{y:}}"'),
        ('f"{x:ab\n{y:{z}}}"', 'f"{x:ab{y:{z}}}"'),
        ('f"{x:\n}"', 'f"{x:}"'),
        ('f"{x:ab\n#c\n}"', 'f"{x:ab}"'),
        ('f"{x:ab\n{y!r:{z}}}"', 'f"{x:ab{y!r:{z}}}"'),
    ]

    @staticmethod
    def _dump(src):
        [stmt] = macroexpand_then_compile(parse(src))
        return ast.dump(stmt.value)

    @CODE_AFTER_BREAK
    @pytest.mark.parametrize("src", [lpy for [lpy, _] in CODE_SPECS])
    def test_3_12_builds_the_spec_after_a_line_break_as_cpython(self, src):
        assert self._dump(src) == ast.dump(ast.parse(src).body[0].value)

    @CODE_AFTER_BREAK
    @pytest.mark.parametrize(
        "src",
        [
            # A token where the field must end, CPython's parser's error.
            'f"{x:a\nb}"',
            'f"{x:a\n b}"',
            'f"{x:a\n:b}"',
            'f"{x:a\n!r}"',
            'f"{x:a\n[}"',
            'f"{x:a\n(}"',
            'f"{x:a\n$}"',
            'f"{x:a\n{y:b}}"',
            'f"{x:a\n{y}b}"',
            'f"{x:a\n{y}"z"}"',
            'f"{x:a\n"z"}"',
            'f"{x:a\nf"b',
            'f"{x:a\nrb"z"}"',
            'f"{x:a\nx"b',
            'f"{x:a\n{y:a\nb}}"',
            'f"{x:a\n{y:a\n#}}"',
            'f"{x:a\n{y:b',
            'f"{x:a\n{y:b}',
            # ... after which the field reads on as an expression.
            'f"{x:a\nb}" "c',
            'f"{x:a\n{y:b}}" "c',
            # ... behind a line break CPython stopped at before.
            'f"a\nb{x:c\nd}"',
            'f"a\nb{x:c\nd',
            'f"{x:a\n{f"b\nc"}d',
            # The tokenizer's own error at that token.
            'f"{x:a\n\\x',
            'f"{x:a\n\\x}"',
            'f"{x:a\n \\x}"',
            'f"{x:a\nb \\x}"',
            'f"{x:a\n)}"',
            'f"{x:a\n]"',
            'f"{x:a\n"',
            'f"{x:a\nb"',
            "f\"{x:a\n'b",
            # The end of the source: the innermost "{" is never closed.
            'f"{x:a\n#}',
            'f"{x:a\n#c"',
            'f"{x:a\n#}"',
            'f"{x:a\n{y}#}"',
            'f"{f"{x:a\n#}"}"',
            'f"{x:a\n#}" "c',
            'f"{x:a\n{y:',
            'f"{x:a\n{y',
        ],
    )
    def test_3_12_refuses_what_follows_a_spec_s_line_break_as_cpython(self, src):
        assert error_position(parse, src) == python_error_position(src)

    @staticmethod
    def _as_3_12(monkeypatch):
        for [probe, value] in [
            ["fstring_fields_are_tokens", True],
            ["single_quoted_spec_refuses_newline", False],
            ["spec_line_break_returns_to_code", True],
            ["conversion_name_checked_first", False],
            ["expecting_brace_names_its_line", False],
        ]:
            monkeypatch.setattr(features, probe, lambda value=value: value)

    @pytest.mark.skipif(not IN_PLACE, reason="3.11 builds another spec AST")
    @pytest.mark.parametrize(["lpy", "py"], CODE_SPECS)
    def test_the_3_12_spec_after_a_line_break_is_built(self, monkeypatch, lpy, py):
        # 3.12's AST, on every interpreter (measured: each pair's sources
        # build the same AST on 3.12).
        self._as_3_12(monkeypatch)
        assert self._dump(lpy) == ast.dump(ast.parse(py).body[0].value)

    @pytest.mark.parametrize(
        ["src", "error"],
        [
            ('f"{x:a\nb}"', ("f-string: expecting '}', or format specs", 2, 1)),
            ('f"{x:a\n{y:b}}"', ("f-string: expecting '}', or format specs", 2, 4)),
            (
                'f"{x:a\nb}" "c',
                ("unterminated string literal (detected at line 2)", 2, 5),
            ),
            ('f"{x:a\n#}"', ("'{' was never closed", 1, 3)),
            ('f"{x:a\n)}"', ("f-string: unmatched ')'", 2, 1)),
            (
                'f"{x:a\n\\x}"',
                ("unexpected character after line continuation character", 2, 2),
            ),
            (
                'f"a\nb{x:c\nd}"',
                ("unterminated f-string literal (detected at line 1)", 1, 1),
            ),
        ],
    )
    def test_3_12_s_errors_after_a_spec_s_line_break(self, monkeypatch, src, error):
        # 3.12's measured messages, on every interpreter.
        self._as_3_12(monkeypatch)
        assert error_position(parse, src) == error


def _read_and_compile(src):
    return macroexpand_then_compile(parse(src))


class TestBraceAfterNestedFieldInSpec:
    """In a format spec, after a nested field has opened, "{{" is a literal
    "{" (3.12+) and "}" ends the spec even doubled; before one, every "{"
    opens a field. compile() on the running interpreter is the oracle."""

    @pytest.mark.parametrize(
        "src",
        [
            *_read_on_3_12_plus(
                [
                    'f"{x:{y}{{}"',
                    'f"{x:{y}{{"',
                    'f"{x:{y}{z}{{}"',
                    'f"{x:{y}{{{z}}"',
                    'f"{x:{y!r}{{}"',
                    'f"""{x:{y}{{}"""',
                    'f"{x:{y}{{ }"',
                    'f"{x:{y}a{{b}}c}"',
                    'f"{x:{y}{{{z}}}"',
                    'f"{x:{y:{z}{{}}}"',
                    'f"{x:{y}{{}}"',
                    'f"{x:{y}}}"',
                ]
            ),
            # 3.11 reads these as CPython does too.
            'f"{x:{y}{{}}}"',
            'f"{x:{y}}}}"',
            'f"{x:a}}}"',
        ],
    )
    def test_reads_as_cpython_does(self, src):
        try:
            [expected] = ast.parse(src).body
        except SyntaxError:
            assert error_position(_read_and_compile, src) == python_error_position(src)
            return
        [stmt] = _read_and_compile(src)
        assert ast.dump(stmt.value) == ast.dump(expected.value)

    def test_3_11_style_refuses_the_doubled_brace(self, monkeypatch):
        # Measured on 3.12-3.15 before the probe: the doubled "{" opened a
        # field.
        monkeypatch.setattr(features, "fstring_fields_are_tokens", lambda: True)
        monkeypatch.setattr(features, "spec_doubles_brace_after_field", lambda: False)
        assert error_position(parse, 'f"{x:{y}{{}"') == (
            "f-string: expecting '}'",
            1,
            12,
        )

    def test_3_12_style_reads_the_doubled_brace(self, monkeypatch):
        monkeypatch.setattr(features, "fstring_fields_are_tokens", lambda: True)
        monkeypatch.setattr(features, "spec_doubles_brace_after_field", lambda: True)
        parse('f"{x:{y}{{}"')


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
