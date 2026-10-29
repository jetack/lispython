"""Nodes a macro expansion builds take the macro call's source position.

Without one they would carry the Node default (line 0), so a traceback or a
compile error in macro-generated code would point nowhere."""

import ast
from collections import deque

import pytest

from lispy.core.errors import LispyCompileError
from lispy.core.macro import macroexpand_then_compile
from lispy.core.nodes import (
    Bracket,
    Constant,
    Paren,
    String,
    Symbol,
    data_to_generator_expression,
    macro_expansion_nodes,
)
from lispy.core.parser import parse


def call_at(position):
    return Paren(Symbol("m"), **position)


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


# Filling positions copies the expansion. The copy must not spend Python frames
# per nesting level: a macro call nests as deeply as the same code without one
# (about 245 levels at the default recursion limit), where copy.deepcopy gave
# up near 160 with a raw RecursionError.
DEEP = 230


@pytest.mark.parametrize(
    "src",
    [
        "(when c " + "(f " * DEEP + "1" + ")" * DEEP + ")",
        "(-> x " + " ".join(f"f{i}" for i in range(DEEP)) + ")",
    ],
    ids=["nested-argument", "long-thread"],
)
def test_deeply_nested_macro_call_compiles(src, scope):
    compile_at_line(3, src, scope)


def test_position_fill_copies_and_keeps_sharing():
    shared = Symbol("x")
    placed = Symbol("y", lineno=7, col_offset=1, end_lineno=7, end_col_offset=2)
    tree = Paren(Symbol("f"), shared, shared, placed)
    tree.extra = (shared, deque([shared]))
    position = {"lineno": 3, "col_offset": 0, "end_lineno": 3, "end_col_offset": 9}
    copy = macro_expansion_nodes(tree, call_at(position))
    assert copy is not tree and shared.lineno == 0 and tree.lineno == 0
    [head, first, second, kept] = copy.list
    assert first is second is copy.extra[0] is copy.extra[1][0]
    assert first is not shared
    assert type(copy.extra) is tuple and type(copy.extra[1]) is deque
    assert [n.position_info for n in (copy, head, first)] == [position] * 3
    assert kept.position_info == placed.position_info and kept is not placed


def test_data_in_an_expression_list_becomes_a_positioned_node():
    inner = Paren(Symbol("g"))
    inner.list.append("s")
    tree = Paren(Symbol("f"), inner)
    tree.list.append(1)
    position = {"lineno": 3, "col_offset": 0, "end_lineno": 3, "end_col_offset": 9}
    copy = macro_expansion_nodes(tree, call_at(position))
    assert tree.list[2] == 1 and inner.list[1] == "s"
    [_, copied_inner, one] = copy.list
    assert type(one) is Constant and one.value == "1"
    assert type(copied_inner.list[1]) is String
    assert one.position_info == copied_inner.list[1].position_info == position


def test_a_data_expansion_becomes_a_positioned_node():
    position = {"lineno": 3, "col_offset": 0, "end_lineno": 3, "end_col_offset": 9}
    node = macro_expansion_nodes(7, call_at(position))
    assert type(node) is Constant and node.position_info == position


def nested_list(depth):
    data = [1]
    for _ in range(depth):
        data = [data]
    return data


def bracket_depth(node):
    depth = 0
    while isinstance(node, Bracket):
        [node] = node.list
        depth += 1
    assert type(node) is Constant and node.value == "1"
    return depth


DATA_DEPTH = 5000  # far past the recursion limit


def test_deeply_nested_data_converts_without_recursion():
    position = {"lineno": 3, "col_offset": 0, "end_lineno": 3, "end_col_offset": 9}
    tree = Paren(Symbol("f"))
    tree.list.append(nested_list(DATA_DEPTH))
    copy = macro_expansion_nodes(tree, call_at(position))
    assert bracket_depth(copy.list[1]) == DATA_DEPTH + 1
    assert copy.list[1].position_info == position
    assert bracket_depth(data_to_generator_expression(nested_list(DATA_DEPTH))) == (
        DATA_DEPTH + 1
    )
    assert bracket_depth(Paren(nested_list(DATA_DEPTH)).list[0]) == DATA_DEPTH + 1


@pytest.mark.parametrize(
    ("make", "type_name"),
    [("(dict)", "dict"), ("(set [1])", "set"), ("(object)", "object")],
)
def test_unrepresentable_data_is_refused_at_the_call(make, type_name, scope):
    define(
        f"(defmacro put-raw-{type_name} [] (= e `(f)) (e.list.append {make})"
        " (return e))\n"
        f"(defmacro ret-raw-{type_name} [] (return {make}))",
        scope,
    )
    for name in (f"put-raw-{type_name}", f"ret-raw-{type_name}"):
        with pytest.raises(LispyCompileError) as info:
            compile_at_line(3, f"  ({name})", scope)
        assert info.value.msg == (
            f"macro {name} returned a {type_name} in its expansion, "
            "which has no lispython form"
        )
        assert (info.value.lineno, info.value.offset) == (3, 3)
        assert (info.value.end_lineno, info.value.end_offset) == (3, 5 + len(name))
