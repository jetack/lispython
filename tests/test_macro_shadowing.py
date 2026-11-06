"""Binding a name that is an active macro is a compile error.

Macro lookup is by a call's head name with no lexical scope, so a name bound
while a macro of that Python name is active would silently stay a macro call:
``(def inc [x] (+ x 100))`` then ``(inc 1)`` gives 2. Every binding position
refuses it (``check_binding`` in compiler/literal.py).
"""

import ast
import importlib
import sys
import types
from pathlib import Path

import pytest

from lispy import tools
from lispy.core.errors import LispyCompileError
from lispy.core.macro import UnitStart, compile_unit
from lispy.core.parser import parse

COMPILER = (
    Path(__file__).resolve().parent.parent / "src" / "lispy" / "core" / "compiler"
)

BUILT_IN = (
    "cannot bind 'inc': it is the built-in macro 'inc' here, so (inc ...) would "
    "expand "
    "the macro instead of using this name; rename it, or make "
    "(require lispy.macros [...]) the module's first form listing only the "
    "macros it uses"
)


def translate(src, scope=None):
    return tools.src_to_python_org(src, scope={} if scope is None else scope)


def needs(version, feature):
    return pytest.mark.skipif(
        sys.version_info < version, reason=f"{feature} needs {version}"
    )


# Each source binds the built-in macro name inc exactly once, at the first
# `inc` in it, and nowhere else spells it.
BINDING_POSITIONS = [
    ["def name", "(def inc [x] x)"],
    ["async def name", "(async-def inc [] None)"],
    ["class name", "(class inc [])"],
    ["assignment", "(= inc 1)"],
    ["chained assignment", "(= a inc 1)"],
    ["destructuring", "(= [a [b inc]] x)"],
    ["starred destructuring", "(= [a *inc] x)"],
    ["annotated assignment", "(= inc ^int 1)"],
    ["annotation only", "(= inc ^int)"],
    ["augmented assignment", "(+= inc 1)"],
    ["walrus", "(print (:= inc 1))"],
    ["parameter", "(def f [inc] None)"],
    ["parameter default", "(def f [a :inc 1] None)"],
    ["*args", "(def f [*inc] None)"],
    ["**kwargs", "(def f [**inc] None)"],
    ["keyword-only parameter", "(def f [* inc] None)"],
    ["positional-only parameter", "(def f [inc /] None)"],
    ["lambda parameter", "(lambda [a inc] a)"],
    ["for target", "(for inc in xs None)"],
    ["for tuple target", "(for [a inc] in xs None)"],
    ["async for target", "(async-def g [] (async-for inc in xs None))"],
    ["comprehension target", "[x for inc in xs]"],
    ["nested comprehension target", "[x for y in xs for inc in y]"],
    ["with target", "(with [(open p) as inc] None)"],
    ["async with target", "(async-def g [] (async-with [(open p) as inc] None))"],
    ["except name", "(try None (except [ValueError as inc] None))"],
    ["import", "(import inc)"],
    ["dotted import", "(import inc.sub)"],
    ["import alias", "(import a.b as inc)"],
    ["from import", "(from m [inc])"],
    ["from import alias", "(from m [x as inc])"],
    ["capture pattern", "(match x (case inc None))"],
    ["as pattern", "(match x (case 1 as inc None))"],
    ["star pattern", "(match x (case [a *inc] None))"],
    ["mapping rest", "(match x (case {1 a **inc} None))"],
    ["nested capture", "(match x (case (C :k inc) None))"],
    [
        "type parameter",
        pytest.param("(def f[inc] [] None)", marks=needs((3, 12), "PEP 695")),
    ],
    [
        "TypeVarTuple",
        pytest.param("(def f[*inc] [] None)", marks=needs((3, 12), "PEP 695")),
    ],
    ["ParamSpec", pytest.param("(class C[**inc] [])", marks=needs((3, 12), "PEP 695"))],
    [
        "generic def name",
        pytest.param("(def inc[T] [] None)", marks=needs((3, 12), "PEP 695")),
    ],
    [
        "type alias name",
        pytest.param("(type inc int)", marks=needs((3, 12), "PEP 695")),
    ],
    [
        "type alias parameter",
        pytest.param("(type A[inc] int)", marks=needs((3, 12), "PEP 695")),
    ],
]


@pytest.mark.parametrize(
    "src", [src for _, src in BINDING_POSITIONS], ids=[k for k, _ in BINDING_POSITIONS]
)
def test_binding_a_built_in_macro_name_is_an_error(src):
    with pytest.raises(LispyCompileError) as info:
        translate(src)
    assert info.value.msg == BUILT_IN
    # At the binding symbol: the first `inc` in the source.
    assert (info.value.lineno, info.value.offset) == (1, src.index("inc") + 1)


# Each compiles: inc is used here, never bound.
NOT_BINDINGS = [
    ["attribute target", "(= self.inc 1)"],
    ["attribute form target", "(= (. self inc) 1)"],
    ["subscript target", "(= (sub d inc) 1)"],
    ["keyword argument", "(f :inc 1)"],
    ["class base keyword", "(class C [:inc 1])"],
    ["global", "(def f [] (global inc))"],
    ["nonlocal", "(def f [] (= x 1) (def g [] (nonlocal inc)))"],
    ["del", "(del inc)"],
    ["class pattern attribute", "(match x (case (C :inc 1) None))"],
    ["imported name under an alias", "(from m [inc as x])"],
    ["module of an aliased import", "(import inc as x)"],
    ["redefining the macro", "(defmacro inc [x] (return x))"],
    ["opted out of it", "(require lispy.macros [])\n(def inc [x] x)"],
    ["only a qualified key", "(require lispy.macros)\n(= inc 1)"],
    [
        "bound before the macro is defined",
        "(= twice 1)\n(defmacro twice [x] (return x))",
    ],
]


@pytest.mark.parametrize(
    "src", [src for _, src in NOT_BINDINGS], ids=[k for k, _ in NOT_BINDINGS]
)
def test_not_a_binding_of_an_active_macro(src):
    translate(src)


def test_a_user_macro_has_the_plain_message():
    with pytest.raises(LispyCompileError) as info:
        translate("(defmacro twice [x] (return x))\n(def f [twice] None)")
    assert info.value.msg == (
        "cannot bind 'twice': it is the macro 'twice' here, so (twice ...) would "
        "expand the macro instead of using this name; rename it"
    )
    assert (info.value.lineno, info.value.offset) == (2, 9)


def test_a_redefined_prelude_name_is_a_user_macro():
    with pytest.raises(LispyCompileError) as info:
        translate("(defmacro inc [x] (return x))\n(= inc 1)")
    assert info.value.msg.startswith("cannot bind 'inc': it is the macro 'inc' here,")


def test_a_required_macro(monkeypatch):
    mod = types.ModuleType("shadow_fake_macros")
    setattr(mod, "__macro_namespace", {"twice": lambda x: x})
    monkeypatch.setitem(sys.modules, "shadow_fake_macros", mod)
    with pytest.raises(LispyCompileError) as info:
        translate("(require shadow-fake-macros [twice])\n(= twice 1)")
    assert info.value.msg.startswith(
        "cannot bind 'twice': it is the macro 'twice' here,"
    )
    # The module's own require binds nothing a macro named k or v shadows.
    src = "(defmacro k [] (return 1))\n(defmacro v [] (return 1))\n"
    for require in [
        "(require shadow-fake-macros)",
        "(require shadow-fake-macros [twice])",
    ]:
        tools.src_to_python_org(src + require, include_meta=True, scope={})


def test_compared_by_python_name():
    # my-x and my_x are one name; the message spells the binding and the macro
    # each as written.
    with pytest.raises(LispyCompileError) as info:
        translate("(defmacro my-x [] (return 1))\n(= my_x 1)")
    assert info.value.msg.startswith(
        "cannot bind 'my_x': it is the macro 'my-x' here, so (my-x ...) would"
    )


# (macro key, binding spelling, one name?). A hyphenated name and its Python
# name are one name both ways; any other spelling is only itself.
NAME_IDENTITY = [
    ["inc", "inc", True],
    ["my-x", "my_x", True],
    ["my_x", "my-x", True],
    ["as->", "as->", True],
    ["ok?", "ok?", True],
    ["as->", "as_to_", False],
    ["->", "_to_", False],
    ["ok?", "ok_p", False],
    ["ok_p", "ok?", False],
    ["x-", "x_", False],
]


@pytest.mark.parametrize("key, spelling, same", NAME_IDENTITY)
def test_the_compiler_and_the_lsp_agree_on_name_identity(key, spelling, same):
    # Two sides of one rule: the binding clash check and the LSP's name-key
    # (references, completion, macro heads) must call the same pairs one name.
    from lispy.lsp import python_view

    src = f"(require lispy.macros [])\n(defmacro {key} [] (return 1))\n(= {spelling} 1)"
    try:
        translate(src)
        refused = False
    except LispyCompileError as e:
        assert e.msg.startswith(
            f"cannot bind '{spelling}': it is the macro '{key}' here, so ({key} ...)"
        )
        refused = True
    lsp_same = python_view.name_key(key) == python_view.name_key(spelling)
    assert (refused, lsp_same) == (same, same)


# A macro spelled other than as a hyphenated name mangles to a name nobody
# calls it by (-> is _to_), so binding that name hides no macro call.
@pytest.mark.parametrize(
    "src",
    [
        "(= _to_ 1)\n(print _to_)",
        "(= as_to_ 1)",
        "(= __ 1)",
        "(defmacro ok? [] (return 1))\n(= ok_p 1)",
    ],
)
def test_an_operator_macro_does_not_clash_by_python_name(src):
    translate(src)


@pytest.mark.parametrize(
    "src, spelling",
    [
        ["(= -> 1)", "->"],
        ["(= as-> 1)", "as->"],
        ["(= -- 1)", "--"],
        ["(defmacro ok? [] (return 1))\n(= ok? 1)", "ok?"],
    ],
)
def test_an_operator_macro_clashes_with_its_own_spelling(src, spelling):
    with pytest.raises(LispyCompileError) as info:
        translate(src)
    assert info.value.msg.startswith(f"cannot bind '{spelling}': it is the")


def test_a_macro_generated_binding_is_at_the_call():
    src = "(defmacro setinc [] (return '(= inc 1)))\n\n  (setinc)"
    with pytest.raises(LispyCompileError) as info:
        translate(src)
    assert info.value.msg == BUILT_IN
    assert (info.value.lineno, info.value.offset) == (3, 3)


def test_position_on_a_later_line():
    with pytest.raises(LispyCompileError) as info:
        translate("(= a 1)\n(def f [x\n       when]\n  None)")
    assert info.value.msg.startswith("cannot bind 'when': it is the built-in macro")
    assert (info.value.lineno, info.value.offset) == (3, 8)


def test_repl_and_nrepl_scopes():
    scope = {}
    tools.eval_and_print("(= a 1)", scope, filename="<repl>", start=UnitStart.CONTINUE)
    with pytest.raises(LispyCompileError, match="cannot bind 'inc'"):
        tools.eval_and_print(
            "(= inc 1)", scope, filename="<repl>", start=UnitStart.CONTINUE
        )
    error = tools.nrepl_eval("(def dec [x] x)", {})["error"]
    assert "cannot bind 'dec': it is the built-in macro 'dec' here" in error


def test_no_check_outside_a_unit():
    # A scope-less compile (no compile_unit, no prelude) binds freely.
    from lispy.core.compiler import stmt_list_compile

    stmt_list_compile(parse("(= inc 1)"))
    compile_unit(parse("(require lispy.macros [])\n(= inc 1)"), {})


# Functions in expr.py/stmt.py that read a symbol's Python name for a slot
# that binds nothing (or, in import_aliases, check the bound one themselves).
# Any other name read in them must be binding_name or name_compile, which
# refuse an active macro.
NON_BINDING_NAME_READS = {
    "call_args_parse": "a keyword argument names the callee's parameter",
    "attribute_compile": "an attribute",
    "match_class_parse": "a class pattern's keyword names an attribute",
    "scope_names": "global/nonlocal declare a scope, binding nothing",
    "import_aliases": "the imported name; the bound one goes to check_binding",
    "importfrom_parse": "the module a from-import reads",
}


def test_binding_sites_route_through_binding_name():
    readers = {}
    for path in [COMPILER / "expr.py", COMPILER / "stmt.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for top in tree.body:
            if not isinstance(top, ast.FunctionDef):
                continue
            for node in ast.walk(top):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in ("symbol_name", "module_path", "python_name")
                ):
                    readers.setdefault(top.name, set()).add(node.func.id)
    assert set(readers) <= set(NON_BINDING_NAME_READS), readers
    # The allowlist stays current: every entry still reads a name.
    assert set(NON_BINDING_NAME_READS) <= set(readers), readers


# --- LSP diagnostics compile the buffer as a unit ---------------------------


@pytest.fixture
def lsp_server():
    pytest.importorskip("pygls")
    # lispy.lsp re-exports the server *object* as ``server``; take the module.
    return importlib.import_module("lispy.lsp.server")


def test_diagnostics_of_a_clean_buffer_use_the_prelude(lsp_server):
    src = "(= x (inc 1))\n(defmacro twice [a] (return a))\n(= y (twice x))\n"
    assert lsp_server.compute_diagnostics(src, "file:///buffer.lpy") == []


def test_diagnostics_report_a_binding_of_a_macro_at_its_symbol(lsp_server):
    src = "(= a 1)\n(def inc [x] (+ x 100))\n"
    [diagnostic] = lsp_server.compute_diagnostics(src, "file:///buffer.lpy")
    assert BUILT_IN in diagnostic.message
    start = diagnostic.range.start
    assert (start.line, start.character) == (1, 5)


# --- defmacro: its parameters and locals are bindings -----------------------

DEFMACRO_PARAM = "(= a 1)\n(defmacro my-m [x\n               cond]\n  (return x))\n"


def test_a_defmacro_parameter_is_a_binding_in_l2py():
    # l2py (include_meta off) runs the defmacro only at compile time.
    with pytest.raises(LispyCompileError) as info:
        tools.l2py_s(DEFMACRO_PARAM, fresh_scope=True, format=False)
    assert info.value.msg.startswith("cannot bind 'cond': it is the built-in macro")
    assert (info.value.lineno, info.value.offset) == (3, 16)


def test_a_defmacro_parameter_is_a_binding_on_import(tmp_path, monkeypatch):
    # The importer (include_meta on) also compiles the defmacro as module code.
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "defmacro_param_mod.lpy").write_text(DEFMACRO_PARAM)
    monkeypatch.delitem(sys.modules, "defmacro_param_mod", raising=False)
    with pytest.raises(LispyCompileError) as info:
        importlib.import_module("defmacro_param_mod")
    sys.modules.pop("defmacro_param_mod", None)
    assert info.value.msg.startswith("cannot bind 'cond': it is the built-in macro")
    assert (info.value.lineno, info.value.offset) == (3, 16)
