import ast
import sys
import time

import pytest

from lispy.core.compiler import features
from lispy.core.errors import LispyCompileError
from lispy.core.nodes import FStrExpr, Paren, Symbol, atom_text
from lispy.core.parser import parse, tokenize
from lispy.tools import macroexpand_then_compile
from tests.utils import stmt_to_dump

T_STRINGS = (
    tuple(features.running_version())[:2] >= features.FEATURES["t-string"].min_version
)


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

    @pytest.mark.parametrize("src", ['f"{x:{{"', 'f"{x:{{}"'])
    def test_an_unclosed_doubled_brace_in_a_spec_is_a_field(self, src):
        with pytest.raises(LispyCompileError, match="f-string: expecting '}'"):
            parse(src)

    def test_a_chain_of_empty_dicts_past_cpython_s_nesting_limit_parses(self):
        # CPython's tokenizer refuses this as "nested too deeply"; lispython
        # adds no limit of its own, as with the chains above.
        [form] = parse('f"{x:{x:{x:{{}}}}}"')
        assert len(spec_chain(form)[0]) == 4

    @pytest.mark.parametrize(
        ["src", "msg", "at"],
        [
            ('f"{x:{x:{x:{x}}}} }"', "f-string: single '}' is not allowed", (1, 19)),
            ('f"{x:{x:{x:{x}}}} {"', "f-string: expecting '}'", (1, 19)),
            ('f"{x:{x:{x:{x', "f-string: expecting '}'", (1, 12)),
        ],
        ids=["single-brace", "open-field", "unterminated"],
    )
    def test_an_error_after_a_deep_chain_is_reported_where_it_is(self, src, msg, at):
        with pytest.raises(LispyCompileError) as info:
            parse(src)
        assert info.value.msg == msg
        assert (info.value.lineno, info.value.offset) == at

    def test_the_first_of_two_errors_is_reported(self):
        # CPython's message; lispython places it at the "!", CPython one
        # column later.
        with pytest.raises(LispyCompileError) as info:
            parse('f"{x!z} {x:{x:{x:{x}}} }"')
        assert info.value.msg == (
            "f-string: invalid conversion character 'z': expected 's', 'r', or 'a'"
        )
        assert (info.value.lineno, info.value.offset) == (1, 5)

    @pytest.mark.parametrize(
        ["src", "msg", "at"],
        [
            # CPython 3.12 and 3.15: no conversion is missing, a "}" is.
            ('f"{x!"', "f-string: expecting '}'", (1, 6)),
            ('f"{x! "', "f-string: expecting '}'", (1, 7)),
            ('f"{x! r"', "f-string: expecting '}'", (1, 7)),
            ('f"{x!r "', "f-string: expecting '}'", (1, 8)),
            # CPython 3.12's reading (3.15 checks the name first).
            ('f"{x!z"', "f-string: expecting '}'", (1, 7)),
            ('f"{x!\n"', "f-string: expecting '}'", (2, 1)),
            # The source ends: as for f"{x" and f"{x!r", at the "{" (CPython:
            # "'{' was never closed" there).
            ('f"{x!', "f-string: expecting '}'", (1, 3)),
            ('f"{x! ', "f-string: expecting '}'", (1, 3)),
            ('f"{x!z ', "f-string: expecting '}'", (1, 3)),
            # Not the closing quote: a string where the name should be.
            ('f"""{x!" """', "f-string: invalid conversion character", (1, 8)),
            ('f"""{x!""', "f-string: invalid conversion character", (1, 8)),
        ],
        ids=[
            "quote",
            "space-quote",
            "space-name-quote",
            "name-space-quote",
            "bad-name-quote",
            "newline-quote",
            "end",
            "space-end",
            "bad-name-space-end",
            "string-in-triple",
            "quotes-in-triple",
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


class TestFieldsLexedInPlace:
    """A replacement field is lexed where it stands, as CPython 3.12+ lexes it."""

    def test_a_quote_in_a_field_opens_a_string(self):
        # CPython 3.12+ reads f"{f"{"}"}"}" as two literals around "}".
        [form] = parse('f"{f"{"}"}"}"')
        inner = form.list[2].value
        assert atom_text(inner.list[0]) == "f-string"
        assert repr(inner.list[2].value) == "Str('\"}\"')"
        assert inner.list[2].expr_text == '"}"'

    def test_an_escaped_quote_in_a_field_is_a_quote_at_its_own_column(self):
        [form] = parse('f"{(.get d \\"k\\")}"')
        field = form.list[2]
        [_, _, key] = field.value.list
        assert repr(key) == "Str('\"k\"')"
        assert field.expr_text == '(.get d "k")'
        # Columns are the source's: \"k\" is 5 characters wide.
        assert (key.col_offset, key.end_col_offset) == (11, 16)
        assert (field.value.end_col_offset, field.end_col_offset) == (17, 18)

    @pytest.mark.parametrize(
        ["src", "py"],
        [
            # Since PEP 701 a field holds strings opened with a bare quote, and
            # CPython's escapes apply in them: \" is a quote, \\ a backslash.
            # On 3.11 the reference is a spelling without a backslash in the
            # field where one exists; a case with none is skipped there.
            *[
                pytest.param(
                    src,
                    py,
                    marks=pytest.mark.skipif(
                        sys.version_info < (3, 12),
                        reason="3.11 refuses a backslash in a field: no reference AST",
                    ),
                    id=name,
                )
                for [name, src, py] in [
                    ["escaped-backslash-closes", r'f"{(g "a\\")}"', r'f"{g("a\\")}"'],
                    [
                        "escaped-backslash-then-quote",
                        r'f"{(g "a\\\"b")}"',
                        r'f"{g("a\\\"b")}"',
                    ],
                    ["escaped-backslash-inside", r'f"{(g "a\\b")}"', r'f"{g("a\\b")}"'],
                    ["raw-escaped-quote", r'f"{(g r"a\"b")}"', r'f"{g(r"a\"b")}"'],
                    ["raw-backslash-closes", r'f"{(g r"a\\")}"', r'f"{g(r"a\\")}"'],
                ]
            ],
            pytest.param(
                r'f"{(g "say \"hi\"")}"',
                r'f"{g("say \"hi\"")}"'
                if sys.version_info >= (3, 12)
                else """f'''{g('say "hi"')}'''""",
                id="escaped-quote",
            ),
            pytest.param(
                r'f"{f"x\"y"}"',
                r'f"{f"x\"y"}"'
                if sys.version_info >= (3, 12)
                else """f'''{f'x"y'}'''""",
                id="nested-f-string-escaped-quote",
            ),
            pytest.param(
                r"""f"{(g '''a\"b''')}" """.strip(),
                r"""f"{g('''a\"b''')}" """.strip()
                if sys.version_info >= (3, 12)
                else """f'''{g('a"b')}'''""",
                id="triple-single-quoted-escaped-quote",
            ),
            # A string opened with \" keeps lispython's field-quote escape.
            pytest.param(
                r'f"{(.get d \"k\")}"',
                """f"{d.get('k')}\"""",
                id="opened-with-escaped-quote",
            ),
            pytest.param(
                r'f"{(g \"a\\"b\")}"',
                """f'''{g('a"b')}'''""",
                id="escaped-quote-in-opened-with-escaped-quote",
            ),
        ],
    )
    def test_a_bare_quoted_string_in_a_field_reads_as_python(self, src, py):
        assert stmt_to_dump(src) == ast.dump(ast.parse(py))

    @pytest.mark.parametrize(
        ["src", "echo"],
        [
            (r'f"{(g "say \"hi\"")=}"', r'(g "say \"hi\"")='),
            (r'f"{(.get d \"k\")=}"', '(.get d "k")='),
            (r'f"{(g \"a\\"b\" "c\"d")=}"', r'(g "a\"b" "c\"d")='),
        ],
        ids=["bare-quoted-keeps-escape", "opened-with-escaped-quote", "both"],
    )
    def test_an_echo_drops_only_the_field_quote_escapes(self, src, echo):
        [stmt] = macroexpand_then_compile(parse(src))
        assert stmt.value.values[0].value == echo

    def test_an_unterminated_bare_string_in_a_field_is_an_error(self):
        with pytest.raises(LispyCompileError) as info:
            parse(r'f"{(g "a\")}"')
        assert info.value.msg == "f-string: expecting '}'"

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
