from lispy.tools import src_to_python_org

PREAMBLE = "(require lispy.macros *)\n"


class TestAsThread:
    def test_no_forms(self):
        assert src_to_python_org(PREAMBLE + "(as-> 42 x)") == "42"

    def test_single_form(self):
        result = src_to_python_org(PREAMBLE + "(as-> 0 x (+ x 1))")
        assert "x = 0" in result
        assert "x = x + 1" in result

    def test_multiple_forms(self):
        result = src_to_python_org(
            PREAMBLE + "(as-> 0 x (+ x 10) (* 2 x) (str x))"
        )
        assert "x = 0" in result
        assert "x = x + 10" in result
        assert "x = 2 * x" in result
        assert "x = str(x)" in result

    def test_method_call(self):
        result = src_to_python_org(
            PREAMBLE + '(as-> "hello" s (.upper s))'
        )
        assert "s = 'hello'" in result
        assert "s = s.upper()" in result

    def test_flexible_placement(self):
        result = src_to_python_org(
            PREAMBLE + "(as-> 1 x (+ 10 x) (- x 5))"
        )
        assert "x = 10 + x" in result
        assert "x = x - 5" in result
