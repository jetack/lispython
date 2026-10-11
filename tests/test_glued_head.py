"""``name[params]`` with no space parses to one GluedHead node.

It names a generic def/class/type; anywhere else it is refused, pointing at
``(sub name ...)`` for subscription.
"""

import importlib

import pytest

from lispy.core.errors import LispyCompileError
from lispy.core.macro import macroexpand
from lispy.core.nodes import Bracket, GluedHead, Paren, Symbol
from lispy.tools import macroexpand_then_compile, parse


def compile_src(src):
    return macroexpand_then_compile(parse(src), {"__macro_namespace": {}})


class TestParse:
    def test_glued_symbol_and_bracket_are_one_node(self):
        [node] = parse("f[T *Ts]")
        assert isinstance(node, GluedHead)
        assert type(node.head) is Symbol and node.head.value == "f"
        assert isinstance(node.params, Bracket)
        assert [str(p) for p in node.params] == ["T", "*Ts"]
        assert node.position_info == {
            "lineno": 1,
            "col_offset": 0,
            "end_lineno": 1,
            "end_col_offset": 8,
        }
        assert str(node) == "f[T *Ts]"

    def test_inside_a_form(self):
        [form] = parse("(def my-fn[T] [x] x)")
        assert [type(n) for n in form] == [Symbol, GluedHead, Bracket, Symbol]
        assert str(form) == "(def my-fn[T] [x] x)"
        assert (form[1].col_offset, form[1].end_col_offset) == (5, 13)

    @pytest.mark.parametrize("src", ["f [T]", "(f [T])", "(sub xs [1])", "'f[T]"])
    def test_space_or_prefix_keeps_two_forms(self, src):
        def glued(nodes):
            return any(
                isinstance(n, GluedHead) or (isinstance(n, Paren) and glued(n.list))
                for n in nodes
            )

        assert not glued(parse(src))


class TestMisuse:
    @pytest.mark.parametrize(
        ["src", "line", "col"],
        [
            ("(print xs[0])", 1, 8),
            ("(= a 1)\n(= xs[0] 1)", 2, 4),
            ("(import json[x])", 1, 9),
            ("(global g[T])", 1, 9),
            ("(f :k my-list[1])", 1, 7),
        ],
    )
    def test_refused_with_sub_hint(self, src, line, col):
        with pytest.raises(LispyCompileError) as info:
            compile_src(src)
        name = src.split("\n")[-1][col - 1 :].split("[")[0]
        assert info.value.msg == (
            f"{name}[...] is only valid as a def/class/type name; use (sub {name} ...)"
        )
        assert (info.value.lineno, info.value.offset) == (line, col)


class TestRoundTrip:
    def test_quasiquote_rebuilds_the_node(self):
        scope = {"__macro_namespace": {}}
        src = "(from lispy.core.nodes *)\n(= t 'U)\n(= node `(def f[T ~t] [x] x))"
        code = compile(
            __import__("ast").Module(macroexpand_then_compile(parse(src), scope), []),
            "<lpy>",
            "exec",
        )
        exec(code, scope)
        node = scope["node"]
        assert isinstance(node[1], GluedHead)
        assert str(node) == "(def f[T U] [x] x)"

    def test_macros_expand_inside_the_params(self):
        scope = {"__macro_namespace": {"m": lambda: Symbol("U")}}
        [form] = parse("(def f[T (m)] [x] x)")
        assert str(macroexpand(form, scope)) == "(def f[T U] [x] x)"


class TestLsp:
    SRC = "(def f[T] [x] x)\n(class Box[T] [] (pass))\n(f 1)\n"

    @pytest.fixture
    def server(self):
        # lispy.lsp re-exports the server *object* as ``server``; take the module.
        return importlib.import_module("lispy.lsp.server")

    def test_definitions_are_indexed_by_name(self, server):
        defs = server.collect_definitions(parse(self.SRC), "file:///g.lpy")
        assert defs["f"][0].range.start.character == 5
        assert defs["Box"][0].range.start.line == 1

    def test_document_symbols_use_the_name(self, server):
        names = [s.name for s in server.walk_symbols(parse(self.SRC))]
        assert names == ["f", "Box"]

    def test_hover_and_definition_land_on_the_name(self, server):
        target = server.node_at_position(parse(self.SRC), 0, 5)
        assert type(target) is Symbol and target.value == "f"
        refs = server.collect_references(parse(self.SRC), "f", "file:///g.lpy")
        assert [r.range.start.line for r in refs] == [0, 2]
