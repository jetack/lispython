"""A macro is called by any spelling of its name (core/macro_table.py):
``my-mac`` and ``my_mac`` are one name (name_key), in a call, a qualified
call, a require's selection and the nREPL's docs, and a redefinition under
either spelling replaces the one entry."""

import ast
import importlib
import sys

import pytest

import lispy  # noqa: F401  (registers the .lpy importer)
from lispy import tools
from lispy.core.macro import MacroTable, compile_unit, macroexpand
from lispy.core.macro import macroexpand_then_compile as expand_compile
from lispy.core.parser import parse


def translate(src):
    return tools.src_to_python_org(src, scope={})


@pytest.fixture
def import_dir(tmp_path, monkeypatch):
    """tmp_path on sys.path; modules imported from it are dropped after."""
    monkeypatch.syspath_prepend(str(tmp_path))
    before = set(sys.modules)
    yield tmp_path
    for name in set(sys.modules) - before:
        del sys.modules[name]


@pytest.mark.parametrize(
    ["defined", "called"], [["my_mac", "my-mac"], ["my-mac", "my_mac"]]
)
def test_a_macro_is_called_by_either_spelling(defined, called):
    src = f"(defmacro {defined} [x] (return `(f ~x)))\n({called} 2)"
    assert translate(src) == "f(2)"


@pytest.mark.parametrize(
    ["require", "call"],
    [
        ["(require nm-mod)", "(nm_mod.my-mac 1)"],
        ["(require nm_mod)", "(nm-mod.my-mac 1)"],
        ["(require nm-mod [my-mac])", "(my-mac 1)"],
        ["(require nm-mod *)", "(my-mac 1)"],
    ],
)
def test_a_required_macro_is_called_by_either_spelling(import_dir, require, call):
    (import_dir / "nm_mod.lpy").write_text("(defmacro my_mac [x] (return `(f ~x)))\n")
    assert translate(f"{require}\n{call}") == "f(1)"


def test_a_redefinition_under_the_other_spelling_replaces_the_macro(import_dir):
    # The defining module's runtime table, which every requirer reads, holds
    # the one name once, with the last body, whichever spelling asks for it.
    (import_dir / "nm_redef.lpy").write_text(
        "(defmacro my_mac [] (return 1))\n(defmacro my-mac [] (return 2))\n"
        "(= v [(my_mac) (my-mac)])\n"
    )
    module = importlib.import_module("nm_redef")
    assert module.v == [2, 2]
    assert len(getattr(module, "__macro_namespace")) == 1
    for selection in ["[my_mac]", "[my-mac]"]:
        assert translate(f"(require nm-redef {selection})\n(my_mac)") == "2"


def test_a_hand_written_plain_dict_table_agrees_on_the_last_entry(import_dir):
    # A .py macro module's table is a plain dict, which may hold both
    # spellings; every require form takes the last one in insertion order,
    # the one copying the table keeps.
    (import_dir / "nm_plain.py").write_text(
        "def old():\n    return 1\n\n\ndef new():\n    return 2\n\n\n"
        "__macro_namespace = {'my_mac': old, 'my-mac': new}\n"
    )
    for require, call in [
        ["(require nm-plain [my_mac])", "(my_mac)"],
        ["(require nm-plain [my-mac])", "(my_mac)"],
        ["(require nm-plain *)", "(my_mac)"],
        ["(require nm-plain)", "(nm_plain.my_mac)"],
    ]:
        assert translate(f"{require}\n{call}") == "2", require


def test_a_redefinition_in_a_caller_built_plain_dict_scope_replaces_the_macro():
    # The generated registration makes the scope's table a MacroTable, so
    # the rule holds in a scope whose table started as a plain dict.
    scope = {"__macro_namespace": {}}
    src = "(defmacro my_mac [] (return 1))\n(defmacro my-mac [] (return 2))"
    expand_compile(parse(src), scope)
    assert len(scope["__macro_namespace"]) == 1
    assert ast.unparse(expand_compile(parse("(my_mac)"), scope)[0]) == "2"


def test_a_lookup_before_the_first_unit_leaves_the_scope_to_be_seeded():
    # Expanding in a scope with no table must not create one that passes
    # for seeded, or the next compile_unit gets no built-in macros.
    scope = {}
    macroexpand(parse("(f 1)")[0], scope)
    body = compile_unit(parse("(when x 1)"), scope)
    assert ast.unparse(ast.Module(body, [])) == "if x:\n    1"


@pytest.mark.parametrize("defined", ["my-mac", "my_mac"])
def test_nrepl_docs_and_annotate_find_a_macro_by_either_spelling(defined):
    # annotate names a macro a macro, never the function it is.
    scope = tools.main_scope()
    reply = tools.nrepl_eval(f'(defmacro {defined} [x] "Doc it." (return x))', scope)
    assert reply["error"] is None
    for spelling in ["my-mac", "my_mac"]:
        assert tools.nrepl_docs(spelling, scope)["docs"] == "my-mac [x] - Doc it."
        annotation = tools.nrepl_annotate(spelling, scope)["annotation"]
        assert annotation == "<macro my-mac>"


def test_every_way_of_writing_to_a_table_compares_by_name():
    # A dict subclass would pass update and setdefault to dict's own C code,
    # which stores a second key under the other spelling.
    first, second, third = object(), object(), object()
    table = MacroTable({"my-x": first})
    table.update({"my_x": second})
    table.setdefault("my-x", third)
    assert list(table.items()) == [("my_x", second)]
    assert "my-x" in table and table.pop("my-x") is second and not table
