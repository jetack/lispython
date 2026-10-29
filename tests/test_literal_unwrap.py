import ast

import pytest

from lispy.core.compiler.literal import string_compile
from lispy.core.nodes import String, Symbol, data_to_generator_expression
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
