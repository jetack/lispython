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


class TestAsThreadRenamesOnlyReferences:
    """as-> renames the threading name only where it is read as a variable."""

    def setup_method(self):
        reset_gensym_counter()

    def compile(self, src):
        return src_to_python_org(PREAMBLE + src)

    def test_keyword_arg_name_kept(self):
        assert self.compile("(as-> 1 x (f :x x))") == "((__x_0 := 1), f(x=__x_0))[-1]"

    def test_quote_kept(self):
        assert (
            self.compile("(as-> 1 x (f 'x x))")
            == "((__x_0 := 1), f(Symbol('x'), __x_0))[-1]"
        )

    def test_quasiquote_renames_only_unquote(self):
        assert (
            self.compile("(as-> 1 x (f `(x ~x) x))")
            == "((__x_0 := 1), f(Paren(Symbol('x'), __x_0), __x_0))[-1]"
        )

    def test_quote_call_is_an_ordinary_call(self):
        assert (
            self.compile("(as-> 1 x (quote x))")
            == "((__x_0 := 1), quote(__x_0))[-1]"
        )

    def test_shadowing_lambda_body_kept(self):
        assert (
            self.compile("(as-> 1 x (f (lambda [x] x) x))")
            == "((__x_0 := 1), f(lambda x: x, __x_0))[-1]"
        )

    def test_shadowing_fn_body_kept(self):
        cases = {
            "[x]": "lambda x: x",
            "[*x]": "lambda *x: x",
            "[**x]": "lambda **x: x",
            "[:x 1]": "lambda x=1: x",
            "[y * :x 1]": "lambda y, *, x=1: x",
        }
        for params, lam in cases.items():
            reset_gensym_counter()
            assert (
                self.compile(f"(as-> 1 x (f (fn {params} x) x))")
                == f"((__x_0 := 1), f({lam}, __x_0))[-1]"
            )

    def test_shadowing_param_default_renamed(self):
        assert (
            self.compile("(as-> 1 x (f (fn [:x x] x)))")
            == "((__x_0 := 1), f(lambda x=__x_0: x))[-1]"
        )

    def test_non_shadowing_fn_default_and_body_renamed(self):
        assert (
            self.compile("(as-> 1 x (f (fn [y :z x] (+ x y)) x))")
            == "((__x_0 := 1), f(lambda y, z=__x_0: __x_0 + y, __x_0))[-1]"
        )

    def test_runtime_keyword_arg(self):
        assert run("(= out (as-> 5 x (dict :x x)))")["out"] == {"x": 5}

    def test_runtime_shadowing_fn(self):
        scope = run("(= out (as-> 5 x (list (map (fn [x] (* x 10)) [1 x]))))")
        assert scope["out"] == [10, 50]


class TestAsThreadDottedAndFormatSpec:
    def setup_method(self):
        reset_gensym_counter()

    def compile(self, src):
        return src_to_python_org(PREAMBLE + src)

    def test_dotted_symbol_renamed(self):
        src = "(= x 100.5)\n(= out (as-> 3 x (+ x.real 1)))"
        assert "((__x_0 := 3), __x_0.real + 1)[-1]" in self.compile(src)
        assert run(src)["out"] == 4

    def test_multi_segment_dotted_symbol_renamed(self):
        src = "(= x 100.5)\n(= out (as-> 3 x (+ x.real.real 1)))"
        assert "__x_0.real.real + 1" in self.compile(src)
        assert run(src)["out"] == 4

    def test_prefix_sharing_symbols_not_renamed(self):
        assert (
            self.compile("(as-> 1 x (f xs.foo x_y.z x.real))")
            == "((__x_0 := 1), f(xs.foo, x_y.z, __x_0.real))[-1]"
        )

    def test_method_symbol_not_matched(self):
        assert (
            self.compile('(as-> "a" s (.upper s))')
            == "((__s_0 := 'a'), __s_0.upper())[-1]"
        )

    def test_fstring_format_spec_renamed(self):
        src = "(= x 9)\n(= out (as-> 5 x f\"{1:{x}}\"))"
        assert "{1:{__x_0}}" in self.compile(src)
        assert run(src)["out"] == "    1"

    def test_fstring_format_spec_dotted_renamed(self):
        src = "(= x 9)\n(= out (as-> 5 x f\"{1:{x.real}}\"))"
        assert "{1:{__x_0.real}}" in self.compile(src)
        assert run(src)["out"] == "    1"

    def test_quoted_dotted_symbol_not_renamed(self):
        out = self.compile("(as-> 1 x (f 'x.real x.real))")
        assert out == "((__x_0 := 1), f(Symbol('x.real'), __x_0.real))[-1]"

    def test_quasiquote_renames_only_unquoted(self):
        assert self.compile("(as-> 5 x `(f ~x x.real))") == (
            "((__x_0 := 5), Paren(Symbol('f'), __x_0, Symbol('x.real')))[-1]"
        )

    def test_quasiquoted_format_spec_kept(self):
        out = self.compile('(as-> 5 x `f"{1:{x}}")')
        assert "FStrExpr(Symbol('x')" in out
        assert "__x_0 :=" in out and "FStrExpr(__x_0" not in out

    def test_unquoted_format_spec_renamed(self):
        out = self.compile('(as-> 5 x `f"{1:{~x}}")')
        assert "FStrExpr(__x_0, conversion=-1, format_spec=None" in out
        assert "Symbol('x')" not in out

    def test_shadowed_body_keeps_dotted(self):
        assert (
            self.compile("(as-> 5 x [(fn [x] x.real) x.real])")
            == "((__x_0 := 5), [lambda x: x.real, __x_0.real])[-1]"
        )
