import ast

import pytest

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

