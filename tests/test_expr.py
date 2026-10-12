import ast

from lispy.tools import macroexpand_then_compile, parse, src_to_python_org


class TestStarredSubscript:
    """``t[*Ts]`` is Subscript(slice=Tuple([Starred])) on every supported CPython."""

    @staticmethod
    def compile_to_dump(src):
        [stmt] = macroexpand_then_compile(parse(src), {"__macro_namespace": {}})
        return ast.dump(stmt)

    def test_load_matches_python(self):
        expected = ast.dump(ast.parse("tuple[*Ts]").body[0])
        assert self.compile_to_dump("(sub tuple *Ts)") == expected

    def test_store_matches_python(self):
        expected = ast.dump(ast.parse("x[*a] = 1").body[0])
        assert self.compile_to_dump("(= (sub x *a) 1)") == expected

    def test_mixed_tuple_slice_unchanged(self):
        expected = ast.dump(ast.parse("x[*a, 1]").body[0])
        assert self.compile_to_dump("(sub x (, *a 1))") == expected


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

