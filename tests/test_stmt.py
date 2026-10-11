import ast

import pytest

from lispy.core.errors import LispyCompileError
from lispy.tools import src_to_python_org

from .utils import stmt_to_dump


class TestMatchMethods:
    def test_match_value(self):
        assert stmt_to_dump("""
(match x
  (case "Relevant" ...))""") == ast.dump(ast.parse("""
match x:
    case "Relevant":
        ..."""))

    def test_match_singleton(self):
        assert stmt_to_dump("""
(match x
  (case None
     ...))""") == ast.dump(ast.parse("""
match x:
    case None:
        ..."""))

    def test_match_sequence(self):
        assert stmt_to_dump("""
(match x
  (case [1 2]
     ...))""") == ast.dump(ast.parse("""
match x:
    case [1, 2]:
        ..."""))

    def test_match_star(self):
        assert stmt_to_dump("""
(match x
  (case [1 2 *rest]
     ...)
  (case [*_]
    ...))""") == ast.dump(ast.parse("""
match x:
    case [1, 2, *rest]:
        ...
    case [*_]:
        ..."""))

    def test_match_mapping(self):
        assert stmt_to_dump("""
(match x
  (case {1 _ 2 _}
    ...)
  (case {**rest}
    ...))""") == ast.dump(ast.parse("""
match x:
    case {1: _, 2: _}:
        ...
    case {**rest}:
        ..."""))

    def test_match_class(self):
        assert stmt_to_dump("""
(match x
  (case (Point2D 0 0)
    ...)
  (case (Point3D :x 0 :y 0 :z 0)
    ...))""") == ast.dump(ast.parse("""
match x:
    case Point2D(0, 0):
        ...
    case Point3D(x=0, y=0, z=0):
        ..."""))

    def test_match_as(self):
        assert stmt_to_dump("""
(match x
  (case [x] as y
    ...)
  (case _
    ...))""") == ast.dump(ast.parse("""
match x:
    case [x] as y:
        ...
    case _:
        ..."""))

    def test_match_or(self):
        assert stmt_to_dump("""
(match x
  (case (| [x] y)
    ...))""") == ast.dump(ast.parse("""
match x:
    case [x] | (y):
        ..."""))



class TestSignedLiteralPatterns:
    """Signed and complex literals build the AST ast.parse builds for a case."""

    @pytest.mark.parametrize(
        ["lpy", "py"],
        [
            ("-1", "-1"),
            ("-1.5", "-1.5"),
            ("-1j", "-1j"),
            ("1+2j", "1 + 2j"),
            ("-1-2j", "-1 - 2j"),
            ("1e3-.34j", "1e3 - .34j"),
            ("1e-3j", "1e-3j"),
            ("[-1 2 *rest]", "[-1, 2, *rest]"),
            ("{-1 _ 1+2j _}", "{-1: _, 1 + 2j: _}"),
            ("(Point -1 :y -2.5)", "Point(-1, y=-2.5)"),
            ("(| -1 0 1)", "-1 | 0 | 1"),
        ],
    )
    def test_same_ast_as_python(self, lpy, py):
        assert stmt_to_dump(f"(match x (case {lpy} ...))") == ast.dump(
            ast.parse(f"match x:\n    case {py}:\n        ...")
        )

    def test_runtime(self):
        src = """
(def classify [n]
  (match n
    (case -1 (return "minus one"))
    (case -1-2j (return "complex"))
    (case [-2.5 *_] (return "list"))
    (case _ (return "other"))))
"""
        scope = {}
        exec(src_to_python_org(src), scope)
        classify = scope["classify"]
        assert classify(-1) == "minus one"
        assert classify(complex(-1, -2)) == "complex"
        assert classify([-2.5, 0]) == "list"
        assert classify(1) == "other"

    @pytest.mark.parametrize("pattern", ["(- x)", "(- True)", "(- (- 1))"])
    def test_only_numbers_are_signed(self, pattern):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump(f"(match x\n  (case {pattern} ...))")
        assert info.value.msg == (
            "only a number literal can be signed in a match pattern"
        )
        assert (info.value.lineno, info.value.offset) == (2, 9)
