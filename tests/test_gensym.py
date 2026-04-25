from lispy.core.builtins import gensym, reset_gensym_counter
from lispy.core.nodes import Symbol
from lispy.tools import src_to_python_org


class TestGensym:
    def setup_method(self):
        reset_gensym_counter()

    def test_returns_symbol(self):
        result = gensym()
        assert isinstance(result, Symbol)

    def test_default_prefix(self):
        result = gensym()
        assert result.name == "__gensym_0"

    def test_increments_counter(self):
        a = gensym()
        b = gensym()
        assert a.name == "__gensym_0"
        assert b.name == "__gensym_1"

    def test_custom_prefix(self):
        result = gensym("tmp")
        assert result.name == "__tmp_0"

    def test_custom_prefix_increments(self):
        a = gensym("x")
        b = gensym("y")
        assert a.name == "__x_0"
        assert b.name == "__y_1"

    def test_unique_across_prefixes(self):
        a = gensym("a")
        b = gensym("b")
        assert a.name != b.name

    def test_symbol_prefix(self):
        result = gensym(Symbol("tmp"))
        assert result.name == "__tmp_0"

    def test_rejects_invalid_prefix(self):
        import pytest

        with pytest.raises(TypeError):
            gensym(123)


class TestGensymInMacro:
    def setup_method(self):
        reset_gensym_counter()

    def test_gensym_with_quoted_symbol(self):
        src = """
(defmacro test-quoted []
  (= tmp (gensym 'tmp))
  (return `(= ~tmp 1)))

(test-quoted)
"""
        result = src_to_python_org(src)
        assert "__tmp_0 = 1" in result

    def test_swap_macro_uses_gensym(self):
        src = """
(defmacro swap [a b]
  (= tmp (gensym "tmp"))
  (return `(do (= ~tmp ~a)
               (= ~a ~b)
               (= ~b ~tmp))))

(swap x y)
"""
        result = src_to_python_org(src)
        assert "__tmp_0 = x" in result
        assert "x = y" in result
        assert "y = __tmp_0" in result
