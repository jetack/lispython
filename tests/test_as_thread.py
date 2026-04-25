from lispy.core.builtins import reset_gensym_counter
from lispy.tools import src_to_python_org

PREAMBLE = "(require lispy.macros *)\n"


class TestAsThread:
    def setup_method(self):
        reset_gensym_counter()

    def test_no_forms(self):
        assert src_to_python_org(PREAMBLE + "(as-> 42 x)") == "42"

    def test_single_form(self):
        result = src_to_python_org(PREAMBLE + "(as-> 0 x (+ x 1))")
        assert "__x_0 = 0" in result
        assert "__x_0 = __x_0 + 1" in result

    def test_multiple_forms(self):
        result = src_to_python_org(
            PREAMBLE + "(as-> 0 x (+ x 10) (* 2 x) (str x))"
        )
        assert "__x_0 = 0" in result
        assert "__x_0 = __x_0 + 10" in result
        assert "__x_0 = 2 * __x_0" in result
        assert "__x_0 = str(__x_0)" in result

    def test_method_call(self):
        result = src_to_python_org(
            PREAMBLE + '(as-> "hello" s (.upper s))'
        )
        assert "__s_0 = 'hello'" in result
        assert "__s_0 = __s_0.upper()" in result

    def test_flexible_placement(self):
        result = src_to_python_org(
            PREAMBLE + "(as-> 1 x (+ 10 x) (- x 5))"
        )
        assert "__x_0 = 10 + __x_0" in result
        assert "__x_0 = __x_0 - 5" in result

    def test_does_not_shadow_outer_variable(self):
        result = src_to_python_org(
            PREAMBLE + "(= x 100)\n(as-> 0 x (+ x 10))\n(print x)"
        )
        assert "x = 100" in result
        assert "print(x)" in result
        assert "__x_0 = 0" in result
        assert "__x_0 = __x_0 + 10" in result
