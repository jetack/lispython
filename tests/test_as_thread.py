from lispy.core.builtins import reset_gensym_counter
from lispy.tools import src_to_python_org

PREAMBLE = "(require lispy.macros *)\n"


def run(src):
    """Compile lispython source, exec it, and return the resulting namespace."""
    scope = {}
    exec(src_to_python_org(PREAMBLE + src), scope)
    return scope


class TestAsThread:
    def setup_method(self):
        reset_gensym_counter()

    def test_no_forms(self):
        assert src_to_python_org(PREAMBLE + "(as-> 42 x)") == "42"

    def test_single_form(self):
        result = src_to_python_org(PREAMBLE + "(as-> 0 x (+ x 1))")
        assert result == "((__x_0 := 0), __x_0 + 1)[-1]"

    def test_multiple_forms(self):
        result = src_to_python_org(
            PREAMBLE + "(as-> 0 x (+ x 10) (* 2 x) (str x))"
        )
        assert result == (
            "((__x_0 := 0), (__x_0 := (__x_0 + 10)), (__x_0 := (2 * __x_0)), "
            "str(__x_0))[-1]"
        )

    def test_method_call(self):
        result = src_to_python_org(PREAMBLE + '(as-> "hello" s (.upper s))')
        assert result == "((__s_0 := 'hello'), __s_0.upper())[-1]"

    def test_flexible_placement(self):
        result = src_to_python_org(PREAMBLE + "(as-> 1 x (+ 10 x) (- x 5))")
        assert result == "((__x_0 := 1), (__x_0 := (10 + __x_0)), __x_0 - 5)[-1]"

    def test_does_not_shadow_outer_variable(self):
        src = "(= x 100)\n(= y (as-> 0 x (+ x 10)))"
        result = src_to_python_org(PREAMBLE + src)
        assert "x = 100" in result
        assert "__x_0 := 0" in result
        scope = run(src)
        assert scope["x"] == 100
        assert scope["y"] == 10


class TestAsThreadInExpressionPosition:
    def setup_method(self):
        reset_gensym_counter()

    def test_return_value(self):
        scope = run(
            """
(from functools [reduce])
(def solution [nums]
  (= k K (- (** 2 17) 1))
  (return (as-> nums X
                (enumerate X)
                (filter (fn [x] (!= (sub x 0) (sub x 1))) X)
                (reduce (fn [x y] (& x (sub y 1))) X k)
                (% X K))))
"""
        )
        solution = scope["solution"]
        assert solution([0, 3, 2, 1]) == 1
        assert solution([0, 1, 3, 2]) == 2
        assert solution([3, 2, 1, 0]) == 0

    def test_call_argument(self):
        scope = run("(= out (abs (as-> 1 x (- x 5) (* x 10))))")
        assert scope["out"] == 40

    def test_lambda_body(self):
        scope = run("(= f (fn [n] (as-> n x (+ x 1) (* x 2))))")
        assert scope["f"](3) == 8

    def test_nested(self):
        scope = run("(= y (as-> 1 x (+ x (as-> 10 z (* z 2))) (* x 2)))")
        assert scope["y"] == 42

    def test_each_form_evaluated_once_in_order(self):
        scope = run(
            """
(= log [])
(def step [tag v] (log.append tag) (return (+ v 1)))
(= y (as-> 0 x (step "a" x) (step "b" x) (step "c" x)))
"""
        )
        assert scope["log"] == ["a", "b", "c"]
        assert scope["y"] == 3
