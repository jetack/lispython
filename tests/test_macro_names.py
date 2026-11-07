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


def test_defmacro_and_require_bind_no_global_of_the_module(import_dir):
    # The code they generate runs in the module: a binding it made would
    # replace the user's own value, and a macro of that name would make the
    # next defmacro or require a clash.
    (import_dir / "nm_mod.lpy").write_text("(defmacro my_mac [] (return 1))\n")
    (import_dir / "nm_user.lpy").write_text(
        "(= importlib 5)\n(defmacro foo [] (return 2))\n"
        "(require nm-mod [my-mac])\n(require nm-mod)\n(require nm-mod *)\n"
        "(= v [importlib (foo) (my-mac)])\n"
    )
    assert importlib.import_module("nm_user").v == [5, 2, 1]
    src = (
        "(defmacro importlib [] (return 1))\n(defmacro foo [] (return 2))\n"
        "(require nm-mod *)\n[(importlib) (foo) (my-mac)]"
    )
    assert translate(src) == "[1, 2, 1]"


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


# A builtin the generated code calls, then a head of the code itself.
OWN_PART_NAMES = [
    "(= __import__ 5)",
    "(= globals 5)",
    "(defmacro __import__ [] (return 5))",
    "(defmacro globals [] (return 5))",
    *[
        f"(defmacro {head} [*xs] (return 5))"
        for head in ["lambda", "sub", "do", "def", "=", "del", "from"]
    ],
    # A required head macro is active before the module's first defmacro.
    "(require nm-heads [from])",
]


@pytest.mark.parametrize("rebinding", OWN_PART_NAMES)
def test_defmacro_and_require_survive_user_names_of_their_own_parts(
    import_dir, rebinding
):
    # The generated registration reaches __import__ and globals as builtins,
    # since a global lookup would find the module's own value (at run time)
    # or expand its macro (at compile time); and its own forms are never
    # macro calls (Expression.compiler_form), while the user's body is
    # expanded (uses-foo), sees the node classes (six) and leaves no temp def
    # behind. The defmacro inside the if is walked again by the pass that
    # (foo) beside it makes, and the module's code avoids every head a case
    # redefines.
    (import_dir / "nm_mod.lpy").write_text("(defmacro my_mac [] (return 1))\n")
    (import_dir / "nm_heads.lpy").write_text("(defmacro from [*xs] (return 5))\n")
    body = (
        "(defmacro foo [] (return 2))\n(defmacro uses-foo [] (return (foo)))\n"
        '(defmacro six [] (return (Constant "6")))\n'
        "(require nm-mod [my-mac])\n(require nm-mod)\n"
        "(if True (defmacro bar [] (return 3)) (foo))\n"
        "(.update (vars) :v [(foo) (uses-foo) (six) (my-mac) (nm-mod.my-mac) (bar)])\n"
    )
    (import_dir / "nm_rebound.lpy").write_text(f"{rebinding}\n{body}")
    module = importlib.import_module("nm_rebound")
    assert module.v == [2, 2, 6, 1, 1, 3]
    assert not [name for name in vars(module) if name.startswith("___defmacro")]
    table = getattr(module, "__macro_namespace")
    keys = ["foo", "uses_foo", "six", "bar", "my_mac", "nm_mod.my_mac"]
    assert all(key in table for key in keys)


def test_a_defmacro_binds_no_module_global(import_dir):
    # The body sees the macro tools as closure cells (defmacro-transform), so
    # the module keeps its own Paren and gensym and gains no name it did not
    # write.
    (import_dir / "nm_own_tools.lpy").write_text(
        '(= Paren 5)\n(= gensym "mine")\n'
        "(defmacro m [] (return (String (repr (str (gensym))))))\n(= v (m))\n"
    )
    module = importlib.import_module("nm_own_tools")
    assert (module.Paren, module.gensym) == (5, "mine")
    assert module.v.startswith("__gensym_")
    names = {name for name in vars(module) if not name.startswith("__")}
    assert names == {"Paren", "gensym", "v"}
    assert not [name for name in vars(module) if name.startswith("___defmacro")]


def test_a_required_macro_quasiquotes_past_its_modules_later_bindings(import_dir):
    # The macro runs with its defining module's globals, rebound after the
    # defmacro; its quasiquote's Paren/Symbol calls and gensym still reach
    # the node classes.
    (import_dir / "nm_rebinds.lpy").write_text(
        "(defmacro plus-one [x]\n"
        '  (= g (gensym "t"))\n'
        "  (return `(do (= ~g ~x) (= out (+ ~g 1)))))\n"
        "(= Paren 5)\n(= Symbol None)\n(= gensym None)\n"
    )
    (import_dir / "nm_rebinds_user.lpy").write_text(
        "(require nm-rebinds [plus-one])\n(plus-one 41)\n"
    )
    assert importlib.import_module("nm_rebinds_user").out == 42


def test_every_class_a_quasiquote_calls_is_a_macro_tool():
    # A quasiquote compiles to calls of node classes by their classname; a
    # name missing from the macro tools is a NameError in a defmacro body.
    from pathlib import Path

    from lispy.core import builtins

    source = Path(builtins.__file__).with_name("nodes.py").read_text("utf-8")
    classnames = {
        node.value.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "classname"
        and isinstance(node.value, ast.Constant)
    }
    assert "Paren" in classnames and "UnquoteSplice" in classnames
    assert classnames <= set(builtins.__all__), classnames - set(builtins.__all__)
    assert set(builtins.macro_scope()) == set(builtins.__all__)


@pytest.mark.parametrize("collection", ["[defmacro foo [x] x]", "{defmacro foo [x] x}"])
def test_a_defmacro_head_in_brackets_or_braces_defines_nothing(collection):
    # A meta form is one only at a ( head, as a macro call is.
    scope = {}
    out = tools.src_to_python_org(f"(print {collection})\n(foo 1)", scope=scope)
    assert "foo" not in scope["__macro_namespace"]
    assert out.endswith("foo(1)")


@pytest.mark.parametrize(
    "collection", ["[require nm-mod [my-mac]]", "{require nm-mod}"]
)
def test_a_require_head_in_brackets_or_braces_registers_nothing(import_dir, collection):
    (import_dir / "nm_mod.lpy").write_text("(defmacro my_mac [] (return 1))\n")
    scope = {}
    tools.src_to_python_org(f"(print {collection})", scope=scope)
    table = scope["__macro_namespace"]
    assert not any(key in table for key in ["my_mac", "nm_mod.my_mac"])
