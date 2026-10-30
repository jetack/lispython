import ast

import pytest

from lispy.core.errors import LispyCompileError
from lispy.tools import parse, src_to_python_org
from tests.utils import stmt_to_dump


class TestStarredSubscript:
    """``t[*Ts]`` is Subscript(slice=Tuple([Starred])) on every supported CPython."""

    @pytest.mark.parametrize(
        "lpy, py",
        [
            ("(sub tuple *Ts)", "tuple[*Ts]"),
            ("(= (sub x *a) 1)", "x[*a] = 1"),
            ("(+= (sub x *a) 1)", "x[*a] += 1"),
            ("(del (sub x *a))", "del x[*a]"),
            ("(sub x *a *b)", "x[*a][*b]"),
            ("(sub x (, *a 1))", "x[*a, 1]"),
        ],
    )
    def test_matches_python(self, lpy, py):
        assert stmt_to_dump(lpy) == ast.dump(ast.parse(py))


class TestUnaryOpMethods:
    def test_unaryop(self):
        assert src_to_python_org("+2") == "+2"


class TestBinOpMethods:
    def test_binop(self):
        assert src_to_python_org("(+ 2 3)") == "2 + 3"


class TestBoolOpMethods:
    def test_boolop(self):
        assert src_to_python_org("(and 1 2)") == "1 and 2"


class TestCompareMethods:
    def test_eq(self):
        assert src_to_python_org("(== 1 2 3)") == "1 == 2 == 3"

    def test_noteq(self):
        assert src_to_python_org("(!= 1 2 3)") == "1 != 2 != 3"

    def test_lt(self):
        assert src_to_python_org("(< 1 2 3)") == "1 < 2 < 3"

    def test_lte(self):
        assert src_to_python_org("(<= 1 2 3)") == "1 <= 2 <= 3"

    def test_gt(self):
        assert src_to_python_org("(> 1 2 3)") == "1 > 2 > 3"



class TestNumberAfterPrefix:
    """``*1``, ``**1`` and ``:1`` read a Constant, so every consumer of a
    prefixed operand either handles it or refuses at it: never a raw
    AttributeError/IndexError, nor a symbol named ``1``."""

    @pytest.mark.parametrize(
        ("src", "msg", "col"),
        [
            ("(f :1 x)", "expected a name, got 1", 5),
            ("(f :1.5 x)", "expected a name, got 1.5", 5),
            ("(f x :2j 3)", "expected a name, got 2j", 7),
            ("(def g [:1 2] 1)", "expected a name, got 1", 10),
            ("(def g [a :1] 1)", "expected a name, got 1", 12),
            ("(def g [*1] 1)", "expected a name, got 1", 10),
            ("(def g [**1] 1)", "expected a name, got 1", 11),
            ("(lambda [*1] 1)", "expected a name, got 1", 11),
            ("(class A [:1 2] 1)", "expected a name, got 1", 12),
            ("(= x :1)", ":1 is only valid before a value", 6),
            ("(with [x :1] 1)", ":1 is only valid before a value", 10),
        ],
    )
    def test_refused_at_the_operand_or_prefix(self, src, msg, col):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump(src)
        assert info.value.msg.startswith(msg)
        assert (info.value.lineno, info.value.offset) == (1, col)

    @pytest.mark.parametrize(
        ("src", "msg", "col"),
        [
            ("(= x :a)", ":a is only valid before a value", 6),
            ("(del :a)", ":a is only valid before a value", 6),
            ("(f :a :b 1)", ":b is only valid before a value", 7),
            ("[**1]", "**1 is only valid in a call, a dict", 2),
            ("(= x **a)", "**a is only valid in a call, a dict", 6),
        ],
    )
    def test_bare_prefixed_operand_refused(self, src, msg, col):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump(src)
        assert info.value.msg.startswith(msg)
        assert (info.value.lineno, info.value.offset) == (1, col)

    @pytest.mark.parametrize(
        ("src", "py"),
        [("(def g [a ^1] 1)", "def g(a: 1):\n    1"), ("(f *1)", "f(*1)"), ("(f **1)", "f(**1)"), ("[*1]", "[*1]")],
    )
    def test_star_of_a_number_compiles(self, src, py):
        assert src_to_python_org(src) == py


class TestKeywordWithoutValue:
    """A ``:name`` that ends a call, def or class argument list has no value
    to pop: a positioned LispyCompileError, not an IndexError."""

    @pytest.mark.parametrize(
        ("src", "col"),
        [
            ("(f :a)", 4),
            ("(f *a :b)", 7),
            ("(def g [a :b] 1)", 11),
            ("(def g [* a :b] 1)", 13),
            ("(class A [b :c] 1)", 13),
        ],
    )
    def test_refused(self, src, col):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump(src)
        assert info.value.msg.endswith("has no value after it")
        assert (info.value.lineno, info.value.offset) == (1, col)
