import ast

import pytest

from lispy.core.errors import LispyCompileError
from lispy.tools import macroexpand_then_compile, parse, src_to_python_org

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
            ("(- 1)", "-1"),
            ("0x1e-2j", "0x1e - 2j"),
            ("-0x1e+2j", "-0x1e + 2j"),
            ("1E+5-2J", "1E+5 - 2J"),
            ("-0o7-1j", "-0o7 - 1j"),
            ("0b11", "0b11"),
            ("1_000j", "1_000j"),
            ("1_0e-2j", "1_0e-2j"),
        ],
    )
    def test_same_ast_as_python(self, lpy, py):
        assert stmt_to_dump(f"(match x (case {lpy} ...))") == ast.dump(
            ast.parse(f"match x:\n    case {py}:\n        ...")
        )

    def test_runtime(self):
        """Runs the AST lispython built, not a re-parse of its unparsed text."""
        src = """
(def classify [n]
  (match n
    (case -1 (return "minus one"))
    (case -1-2j (return "complex"))
    (case -0x1e+2j (return "hex complex"))
    (case [-2.5 *_] (return "list"))
    (case _ (return "other"))))
"""
        scope = {}
        module = ast.Module(macroexpand_then_compile(parse(src), {}), [])
        exec(compile(module, "<lpy>", "exec"), scope)
        classify = scope["classify"]
        assert classify(-1) == "minus one"
        assert classify(complex(-1, -2)) == "complex"
        assert classify(complex(-30, 2)) == "hex complex"
        assert classify(complex(30, 2)) == "other"
        assert classify([-2.5, 0]) == "list"
        assert classify(1) == "other"

    @pytest.mark.parametrize(
        ["pattern", "msg"],
        [
            ("1j+2j", "real number required in complex literal"),
            ("-1j-2j", "real number required in complex literal"),
            ("(- 1+2j)", "only a number literal can be signed in a match pattern"),
        ],
    )
    def test_complex_literal_rules(self, pattern, msg):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump(f"(match x\n  (case {pattern} ...))")
        assert info.value.msg == msg
        assert (info.value.lineno, info.value.offset) == (2, 9)


class TestSignedNumberExpressions:
    """In expressions a signed complex token has the value Python gives it."""

    @pytest.mark.parametrize(
        ["lpy", "value"],
        [
            ("-1+2j", -1 + 2j),
            ("-0x1e+2j", -0x1E + 2j),
            ("--1+2j", --1 + 2j),
            ("1e3-.34j", 1e3 - 0.34j),
            ("0x1e-2j", 0x1E - 2j),
            ("-1", -1),
            ("0b11", 3),
            ("1_000", 1000),
        ],
    )
    def test_value(self, lpy, value):
        [stmt] = macroexpand_then_compile(parse(lpy), {})
        result = eval(compile(ast.Expression(stmt.value), "<lpy>", "eval"))
        assert result == value and type(result) is type(value)

    def test_signed_complex_prints_like_python(self):
        assert src_to_python_org("-1+2j") == "(-1+2j)"

    @pytest.mark.parametrize("pattern", ["(- x)", "(- True)", "(- (- 1))"])
    def test_only_numbers_are_signed(self, pattern):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump(f"(match x\n  (case {pattern} ...))")
        assert info.value.msg == (
            "only a number literal can be signed in a match pattern"
        )
        assert (info.value.lineno, info.value.offset) == (2, 9)


class TestImportNames:
    """Every import form checks its names in one place."""

    @pytest.mark.parametrize(
        ["lpy", "py"],
        [
            ("(from . [a])", "from . import a"),
            ("(from .. [a b as c])", "from .. import a, b as c"),
            ("(from .m [a])", "from .m import a"),
            ("(import os.path as p)", "import os.path as p"),
        ],
    )
    def test_same_ast_as_python(self, lpy, py):
        assert stmt_to_dump(lpy) == ast.dump(ast.parse(py))

    @pytest.mark.parametrize(
        ["lpy", "msg", "at"],
        [
            ('(import "x")', '(import ...) expects a name, got "x"', '"'),
            ("(import)", "(import ...) needs at least one module name", "("),
            ("(import os as a.b)", "(import ...) expects a name, got a.b", "a."),
            ("(from json)", "(from ...) takes module and [names] pairs", "("),
            (
                '(from json "dumps")',
                '(from ...) expects [names] or * after a module, got "dumps"',
                '"',
            ),
            ("(from json [1])", "(from ...) expects a name, got 1", "1"),
            ('(from "json" [a])', '(from ...) expects a name, got "json"', '"'),
            (
                "(import json[x])",
                "json[...] is only valid as a def/class/type name; "
                "use (sub json ...)",
                "j",
            ),
        ],
    )
    def test_refused(self, lpy, msg, at):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump(lpy)
        assert info.value.msg == msg
        assert (info.value.lineno, info.value.offset) == (1, lpy.index(at) + 1)


class TestEmptyLiteralOperands:
    """An empty [] () {} is an Expression of length 0, hence falsy: every form
    with an optional operand must test presence by length, not truthiness."""

    @pytest.mark.parametrize(
        ("lispy", "python"),
        [
            ("(return [])", "return []"),
            ("(return {})", "return {}"),
            ("(return)", "return"),
            ("(return (list))", "return list()"),
            ("(yield [])", "yield []"),
            ("(yield {})", "yield {}"),
            ("(yield)", "yield"),
            ("(raise [])", "raise []"),
            ("(assert x [])", "assert x, []"),
            ("(assert x {})", "assert x, {}"),
            ("(= x [])", "x = []"),
            ("(def f [:a [] *args :b {}] ...)", "def f(a=[], *args, b={}): ..."),
            ("(def f [:a []] (return []))", "def f(a=[]):\n    return []"),
            ("(lambda [:a []] [])", "lambda a=[]: []"),
        ],
    )
    def test_matches_ast_parse(self, lispy, python):
        assert stmt_to_dump(lispy) == ast.dump(ast.parse(python))

    def test_return_takes_one_operand(self):
        with pytest.raises(LispyCompileError):
            stmt_to_dump("(return [] [])")
