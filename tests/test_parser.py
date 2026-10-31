import pytest

from lispy.core.compiler import features
from lispy.core.errors import LispyCompileError
from lispy.core.nodes import FStrExpr, Paren, Symbol, atom_text
from lispy.core.parser import FSTRING_FIELD_NESTING_LIMIT, parse, tokenize

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


class TestFormatSpecNestingLimit:
    """CPython's lexer refuses a replacement field opened while three fields
    of the same literal are open (MAX_EXPR_NESTING); positions are CPython
    3.12-3.15's (lineno, offset, end_lineno, end_offset) for the same text."""

    MESSAGE = "f-string: expressions nested too deeply"

    def test_the_limit_is_cpython_s(self):
        assert FSTRING_FIELD_NESTING_LIMIT == 3

    def test_the_deepest_spec_chain_parses(self):
        [form] = parse('f"{x:{x:{x}}}"')
        assert atom_text(form.list[0]) == "f-string"
        [fields, leaf] = spec_chain(form)
        # Each spec runs from its ":" to the "}" closing its field.
        assert fields == [((2, 13), (4, 12)), ((5, 12), (7, 11)), ((8, 11), None)]
        assert (type(leaf), atom_text(leaf), leaf.col_offset) == (Symbol, "x", 9)

    @pytest.mark.parametrize(
        ["src", "at"],
        [
            ('f"{x:{x:{x:{x}}}}"', (1, 11)),
            # The count is of fields open: a closed sibling is not counted.
            ('f"{x:{x:{x}{x:{x}}}}"', (1, 14)),
            ('f"{x:{x:{x:abc{x}}}}"', (1, 14)),
            ('rf"{x:{x:{x:{x}}}}"', (1, 12)),
            ('f"{f"{x:{x:{x:{x}}}}"}"', (1, 14)),
            ('f"""{x:{x:{x:\n\nab{x}}}}"""', (3, 2)),
            # A "{" at column 0: CPython's offset is then 0.
            ('f"""{x:{x:{x:\n{x}}}}"""', (2, 0)),
            ('f"{x:{x:{f"{y:{z:{w:{v}}}}"}}}"', (1, 20)),
        ],
        ids=[
            "chain",
            "after-a-sibling",
            "after-text",
            "raw",
            "nested-literal",
            "lines-below",
            "column-0",
            "inner-literal",
        ],
    )
    def test_one_more_field_is_refused_where_cpython_refuses_it(self, src, at):
        with pytest.raises(LispyCompileError) as info:
            parse(src)
        error = info.value
        assert error.msg == self.MESSAGE
        position = (error.lineno, error.offset, error.end_lineno, error.end_offset)
        assert position == (*at, *at)

    @pytest.mark.skipif(not T_STRINGS, reason="t-strings need Python 3.14")
    def test_a_t_string_is_refused_by_its_own_name(self):
        [form] = parse('t"{x:{x:{x}}}"')
        assert len(spec_chain(form)[0]) == 3
        with pytest.raises(LispyCompileError) as info:
            parse('t"{x:{x:{x:{x}}}}"')
        assert info.value.msg == "t-string: expressions nested too deeply"
        assert (info.value.lineno, info.value.offset) == (1, 11)

    def test_braces_inside_an_expression_are_not_fields(self):
        [form] = parse('f"{ {1 {2 {3 {4 5}}}} }"')
        assert type(form.list[2].value).__name__ == "Brace"

    @pytest.mark.parametrize(
        "src",
        ['f"{x:{f"{y:{z:{w}}}"}}"', 'f"{x:{x:{f"{y:{z:{w}}}"}}}"'],
        ids=["in-the-second-field", "in-the-third-field"],
    )
    def test_a_nested_literal_counts_its_own_fields(self, src):
        assert len(parse(src)) == 1

    @pytest.mark.parametrize(
        "src",
        [
            'f"{x:{x:{x:{x}}}} }"',
            'f"{x:{x:{x:{x}}}} {"',
            'f"{x:{x:{x:{x}}}} {x!z}"',
            'f"{x:{x:{x:{x',
            'f"{x:{x:{x:{x}}}}"\nf"}"',
        ],
        ids=["single-brace", "open-field", "conversion", "unterminated", "next-line"],
    )
    def test_the_refusal_comes_before_any_later_error(self, src):
        with pytest.raises(LispyCompileError) as info:
            parse(src)
        assert info.value.msg == self.MESSAGE
        assert (info.value.lineno, info.value.offset) == (1, 11)

    def test_an_earlier_conversion_error_comes_first(self):
        # CPython's message; lispython places it at the "!", CPython one
        # column later.
        with pytest.raises(LispyCompileError) as info:
            parse('f"{x!z} {x:{x:{x:{x}}}}"')
        assert info.value.msg == (
            "f-string: invalid conversion character 'z': expected 's', 'r', or 'a'"
        )
        assert (info.value.lineno, info.value.offset) == (1, 5)

    def test_an_earlier_literal_error_comes_first(self):
        with pytest.raises(LispyCompileError) as info:
            parse('f"}"\nf"{x:{x:{x:{x}}}}"')
        assert info.value.msg == "f-string: single '}' is not allowed"
        assert (info.value.lineno, info.value.offset) == (1, 3)

    def test_partial_parse_keeps_the_forms_before_the_refusal(self):
        forms = parse('a f"{x:{x:{x:{x}}}}" b', partial=True)
        assert [str(form) for form in forms] == ["a"]


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
        ["src", "form"],
        [('f"{x ;c}"', "x"), ('f"{(f x ;c\n y)}"', "(f x y)")],
        ids=["before-the-closing-brace", "to-the-end-of-the-line"],
    )
    def test_a_comment_in_a_field_ends_at_a_brace_or_the_line(self, src, form):
        assert str(parse(src)[0].list[2].value) == form
