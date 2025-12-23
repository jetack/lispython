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

