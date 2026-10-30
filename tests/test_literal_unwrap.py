import ast
import math

import pytest

from lispy.core.compiler.expr import expr_compile
from lispy.core.compiler.literal import string_compile
from lispy.core.errors import LispyCompileError
from lispy.core.nodes import Constant, String, Symbol, data_to_generator_expression
from lispy.tools import macroexpand_then_compile, parse, src_to_python_org

PREAMBLE = "(require lispy.macros *)\n"


class TestLiteralUnwrapInMacro:
    def test_number_is_int(self):
        src = PREAMBLE + """
(defmacro repeat-n [n *body]
  (= stmts [])
  (for _ in (range n)
    (stmts.extend body))
  (return `(do ~@stmts)))

(repeat-n 3 (print "hello"))
"""
        result = src_to_python_org(src)
        assert result.count("print('hello')") == 3

    def test_string_is_str(self):
        src = PREAMBLE + """
(defmacro greet [name]
  (= upper (.upper name))
  (return `(print ~upper)))

(greet "world")
"""
        result = src_to_python_org(src)
        assert "WORLD" in result

    def test_bool_is_bool(self):
        src = PREAMBLE + """
(defmacro if-flag [flag *body]
  (if flag
      (return `(do ~@body))
      (return `(pass))))

(if-flag True (print "yes"))
"""
        result = src_to_python_org(src)
        assert "print('yes')" in result

    def test_none_is_none(self):
        src = PREAMBLE + """
(defmacro maybe-wrap [val]
  (if (is val None)
      (return `(print "nothing"))
      (return `(print ~val))))

(maybe-wrap None)
"""
        result = src_to_python_org(src)
        assert "nothing" in result

    def test_symbol_stays_as_node(self):
        src = PREAMBLE + """
(defmacro my-inc [x]
  (return `(+ ~x 1)))

(my-inc y)
"""
        result = src_to_python_org(src)
        assert "y + 1" in result

    def test_expression_stays_as_node(self):
        src = PREAMBLE + """
(defmacro my-twice [expr]
  (return `(+ ~expr ~expr)))

(my-twice (+ a b))
"""
        result = src_to_python_org(src)
        assert "a + b + (a + b)" in result


class TestRawValueInExpansion:
    """A literal argument reaches the macro as a Python value; a macro that
    puts it straight into an Expression's list (bypassing the constructor)
    still compiles, because the expander converts data anywhere in the
    expansion."""

    @pytest.mark.parametrize(
        ("src", "expected"),
        [
            ("(-> 1 (f))", "f(1)"),
            ("(->> 1 (f 2))", "f(2, 1)"),
            ('(-> "a" (f))', "f('a')"),
            ("(-> 1 (f) (g 2))", "g(f(1), 2)"),
        ],
    )
    def test_threaded_literal_into_paren(self, src, expected):
        assert src_to_python_org(PREAMBLE + src) == expected

    def test_raw_value_inserted_into_nested_paren(self):
        src = PREAMBLE + """
(defmacro put-in [x]
  (= inner `(g))
  (inner.list.insert 1 x)
  (return `(f ~inner)))

(put-in 5)
"""
        assert src_to_python_org(src) == "f(g(5))"

    def test_converted_value_takes_the_macro_call_position(self):
        stl = macroexpand_then_compile(parse(PREAMBLE + "\n  (-> 1 (f))"))
        [call] = [n for n in ast.walk(stl[-1]) if isinstance(n, ast.Call)]
        [arg] = call.args
        assert (arg.lineno, arg.col_offset) == (3, 2)
        assert (arg.end_lineno, arg.end_col_offset) == (3, 12)


TRICKY_STRINGS = [
    "",
    "a\\b",
    "\\",
    'q"q',
    "'",
    "x\ny",
    "\r",
    "\r\n",
    "\t",
    "\x00",
    "\x1b[0m",
    "\x7f",
    "\x85",
    " ",
    "é한😀",
    "\ud800",
    "{x}",
    '"""',
]
TRICKY_BYTES = [b"", b"x", b"\\", b'"', b"\n\r\t", b"\x00\xff", b"\x7f", b"\x80"]


class TestDataRoundTrip:
    """Python data a macro returns compiles to exactly that value."""

    @pytest.mark.parametrize("value", TRICKY_STRINGS + TRICKY_BYTES, ids=ascii)
    def test_data_compiles_to_the_same_value(self, value):
        node = data_to_generator_expression(value)
        compiled = string_compile(node)
        assert type(compiled.value) is type(value) and compiled.value == value

    @pytest.mark.parametrize("value", TRICKY_STRINGS, ids=ascii)
    def test_quoted_string_and_symbol_keep_their_text(self, value):
        # A quoted String/Symbol is rebuilt from its token text at run time.
        for node in (String(value), Symbol(value)):
            assert string_compile(node.operands_generate(False)).value == value

    @pytest.mark.parametrize(
        ("src", "expected"),
        [
            ('(-> "a\\\\b" (f))', "f('a\\\\b')"),
            ('(-> b"x" (f))', "f(b'x')"),
            ("(= s 'a\\b)", "s = Symbol('a\\\\b')"),
        ],
    )
    def test_end_to_end(self, src, expected):
        assert src_to_python_org(PREAMBLE + src) == expected


INF = float("inf")
NAN = float("nan")
TRICKY_SCALARS = [
    None,
    True,
    False,
    ...,
    0,
    -5,
    10**5000,
    -(10**5000),
    1.5,
    -0.0,
    5e-324,
    1e308,
    INF,
    -INF,
    NAN,
    1 + 2j,
    -1j,
    complex(0.0, -0.0),
    complex(-0.0, 0.0),
    complex(-0.0, -0.0),
    complex(INF, NAN),
    complex(1.0, -INF),
]


def same_value(a, b):
    """Equal in type and value, telling -0.0 from 0.0 and nan from all else."""
    if type(a) is not type(b):
        return False
    if isinstance(a, complex):
        return same_value(a.real, b.real) and same_value(a.imag, b.imag)
    if isinstance(a, float):
        return repr(a) == repr(b) and math.copysign(1, a) == math.copysign(1, b)
    return a == b


def compiled_value(node):
    expr = ast.Expression(expr_compile(node))
    return eval(compile(ast.fix_missing_locations(expr), "<data>", "eval"), {})


class TestScalarDataRoundTrip:
    """Every scalar a macro may return compiles to that very value."""

    @pytest.mark.parametrize(
        "value",
        TRICKY_SCALARS,
        # repr refuses ints over 4300 digits.
        ids=lambda v: f"int-{v.bit_length()}-bits" if type(v) is int else repr(v),
    )
    def test_scalar_compiles_to_the_same_value(self, value):
        node = data_to_generator_expression(value)
        assert same_value(compiled_value(node), value)

    @pytest.mark.parametrize(
        ("src", "value"),
        [("(-> ... (f))", ...), ("(-> 1e400 (f))", INF), ("(-> None (f))", None)],
    )
    def test_threaded_scalar(self, src, value):
        stl = macroexpand_then_compile(parse(PREAMBLE + src))
        [call] = [n for n in ast.walk(stl[-1]) if isinstance(n, ast.Call)]
        [arg] = call.args
        assert same_value(eval(compile(ast.Expression(arg), "<arg>", "eval")), value)


# No literal spells these; each must compile to its very value, never to a
# call of float/complex, which a user binding would shadow.
INEXACT_DEFS = PREAMBLE + """
(import math)
(defmacro ret-nan [] (return math.nan))
(defmacro ret-signed-cx [] (return (complex 0.0 -0.0)))
"""


INEXACT = [
    NAN,
    complex(0.0, -0.0),
    complex(-0.0, 1.0),
    complex(INF, NAN),
    complex(1.0, -INF),
]


def define_inexact():
    scope = {}
    stl = macroexpand_then_compile(parse(INEXACT_DEFS), scope)
    exec(compile(ast.Module(stl, []), "<defs>", "exec"), scope)
    return scope


class TestInexactScalarData:
    def test_compiles_to_the_value_with_float_and_complex_rebound(self):
        scope = define_inexact()
        src = "(= float int)\n(= complex int)\n(= got [(ret-nan) (ret-signed-cx)])"
        stl = macroexpand_then_compile(parse(src), scope)
        assert [type(e) for e in stl[-1].value.elts] == [ast.Constant] * 2
        exec(compile(ast.Module(stl, []), "<use>", "exec"), scope)
        [nan, signed] = scope["got"]
        assert same_value(nan, NAN) and same_value(signed, complex(0.0, -0.0))

    def test_is_a_value_pattern_in_a_match_case(self):
        scope = define_inexact()
        src = "(match v (case (ret-signed-cx) (= hit True)) (case _ (= hit False)))"
        [match] = macroexpand_then_compile(parse(src), scope)
        pattern = match.cases[0].pattern
        assert type(pattern) is ast.MatchValue
        assert same_value(pattern.value.value, complex(0.0, -0.0))
        module = ast.fix_missing_locations(ast.Module([match], []))
        scope["v"] = 0j
        exec(compile(module, "<match>", "exec"), scope)
        assert scope["hit"] is True

    def test_signed_in_a_match_case_is_refused(self):
        scope = define_inexact()
        defs = "(defmacro neg-nan [] (return `(- ~math.nan)))"
        stl = macroexpand_then_compile(parse(defs), scope)
        exec(compile(ast.Module(stl, []), "<defs>", "exec"), scope)
        src = "(match v (case (neg-nan) (= hit True)))"
        with pytest.raises(LispyCompileError) as info:
            macroexpand_then_compile(parse(src), scope)
        assert info.value.msg.startswith(
            "pattern can sign only a number literal, got (- "
        )

    @pytest.mark.parametrize("value", INEXACT, ids=repr)
    def test_display_text_reads_back_as_the_value(self, value):
        # macroexpand output shows this text; read back where float and
        # complex are not rebound, it is the same value.
        text = str(data_to_generator_expression(value))
        [form] = parse(text)
        assert same_value(compiled_value(form), value), text

    @pytest.mark.parametrize(
        "value",
        TRICKY_SCALARS + [1e16, -1e-7, 2j, 1e16j, 1.5 + 2.5j, -2 - 3j, -1 + 1j, 1 - 1j],
        ids=lambda v: f"int-{v.bit_length()}-bits" if type(v) is int else repr(v),
    )
    def test_every_scalars_display_text_reads_back_as_the_value(self, value):
        # A complex with a real part used to show as "(1+2j)", which reads back
        # as a call of the number 1+2j.
        text = str(data_to_generator_expression(value))
        [form] = parse(text)
        assert same_value(compiled_value(form), value), text[:60]

    @pytest.mark.parametrize("value", INEXACT, ids=repr)
    def test_quasiquote_rebuild_keeps_the_value(self, value):
        # A quoted Constant is rebuilt at run time; the rebuild must keep the
        # value exact without naming float or complex.
        rebuilt = data_to_generator_expression(value).generator_expression()
        tree = ast.fix_missing_locations(ast.Expression(expr_compile(rebuilt)))
        assert {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} == {
            "Constant"
        }
        scope = {"Constant": Constant, "float": int, "complex": int}
        made = eval(compile(tree, "<rebuild>", "eval"), scope)
        assert same_value(compiled_value(made), value)
