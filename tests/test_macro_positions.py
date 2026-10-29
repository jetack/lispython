"""Nodes a macro expansion builds take the macro call's source position.

Without one they would carry the Node default (line 0), so a traceback or a
compile error in macro-generated code would point nowhere."""

import ast

import pytest

from lispy.core.macro import macroexpand_then_compile
from lispy.core.parser import parse


@pytest.fixture(scope="module")
def scope():
    scope = {}
    macroexpand_then_compile(parse("(require lispy.macros *)"), scope)
    return scope


def compile_at_line_3(src, scope):
    return macroexpand_then_compile(parse("\n\n" + src), scope)


def nodes(stl):
    return [n for st in stl for n in ast.walk(st) if hasattr(n, "lineno")]


def calls_to(stl, name):
    return [
        n
        for n in nodes(stl)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == name
    ]


@pytest.mark.parametrize(
    "src",
    [
        "(-> 1 f)",
        "(->> 1 f print)",
        "(when True (f))",
        "(cond a (b) c (d))",
        "(as-> 1 x (f x))",
        "(fn [x] (f x))",
        "(inc (f))",
        "(++ x)",
        "(conde a (b) (c))",
        "(-> (read) solution print)",
        'f"{(-> 1 g):>{(inc w)}}"',
    ],
)
def test_macro_expansion_has_no_line_0(src, scope):
    stl = compile_at_line_3(src, scope)
    assert [type(n).__name__ for n in nodes(stl) if n.lineno == 0] == []


def test_generated_call_takes_the_macro_call_position(scope):
    stl = compile_at_line_3("  (-> 1 f)", scope)
    [call] = calls_to(stl, "f")
    assert (call.lineno, call.col_offset) == (3, 2)
    assert (call.end_lineno, call.end_col_offset) == (3, 10)


def test_generated_if_takes_the_macro_call_position(scope):
    [stmt] = compile_at_line_3("(when True (f))", scope)
    assert isinstance(stmt, ast.If)
    assert stmt.lineno == 3


def test_unquoted_user_arguments_keep_their_own_position(scope):
    stl = compile_at_line_3("(when True\n    (g))", scope)
    [stmt] = stl
    [call] = calls_to(stl, "g")
    assert stmt.lineno == 3
    assert (call.lineno, call.col_offset) == (4, 4)


def test_threaded_symbol_keeps_its_own_position(scope):
    stl = compile_at_line_3("(-> 1\n    f)", scope)
    [call] = calls_to(stl, "f")
    assert call.lineno == 3
    assert (call.func.lineno, call.func.col_offset) == (4, 4)


def define(src, scope):
    stl = macroexpand_then_compile(parse(src), scope)
    exec(compile(ast.Module(body=stl, type_ignores=[]), "<defs>", "exec"), scope)


def compile_at_line(line, src, scope):
    stl = macroexpand_then_compile(parse("\n" * (line - 1) + src), scope)
    compile(ast.Module(body=stl, type_ignores=[]), "<test>", "exec")
    return stl


def test_shared_template_takes_each_call_position(scope):
    # The macro returns one template object on every call; the first call
    # must not pin its position on the later ones.
    define("(= T `(sub [] 1))\n(defmacro shared-template [] (return T))", scope)
    for line in (10, 50):
        stl = compile_at_line(line, "(shared-template)", scope)
        assert [n.lineno for n in nodes(stl) if isinstance(n, ast.Subscript)] == [line]


def test_shared_node_beside_fresh_ones_keeps_a_valid_range(scope):
    # A fresh attribute form ending in a shared symbol: a stale symbol line
    # would end the attribute before it starts, which compile() rejects.
    define("(= A `y)\n(defmacro shared-attr [] (return `(. x ~A)))", scope)
    for line in (10, 50):
        stl = compile_at_line(line, "(shared-attr)", scope)
        [attr] = [n for n in nodes(stl) if isinstance(n, ast.Attribute)]
        assert (attr.lineno, attr.end_lineno) == (line, line)
