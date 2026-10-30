import pytest

from lispy.core.errors import LispyCompileError
from lispy.core.nodes import Paren, Symbol
from lispy.core.parser import parse, tokenize


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
