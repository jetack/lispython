"""The name model: every identifier the compiler emits passes through
``python_name`` (``compiler/literal.py``). A symbol that is no Python name in
a name slot is a LispyCompileError at the symbol, never an AST that CPython
accepts but that means nothing (``a.if``, ``f(if=1)``, ``Name("+")``)."""

import pathlib
import re
import sys

import pytest

import lispy.core.compiler.expr as expr_module
import lispy.core.compiler.features as features_module
import lispy.core.compiler.stmt as stmt_module
import lispy.core.compiler.utils as compiler_utils_module
from lispy.core.errors import LispyCompileError
from lispy.tools import src_to_python_org

KEYWORD = "if is a Python keyword and cannot be used as a name"
ASSIGN_DEBUG = "cannot assign to __debug__"


def operator(text):
    return (
        f"{text} is an operator and cannot be used as a name; call it as ({text} ...)"
    )


def invalid(text):
    return f"{text} is not a valid Python name"


def assert_refused(src, msg, at):
    """``src`` is refused with ``msg`` (a prefix ending in ``...`` matches
    the start), spanning ``at`` (``text@n``: its n-th occurrence) on line 1.
    ``at`` is matched exactly when it has no ``@n`` suffix (``~@a``)."""
    with pytest.raises(LispyCompileError) as info:
        src_to_python_org(src)
    err = info.value
    if msg.endswith("..."):
        assert err.msg.startswith(msg[:-3]), err.msg
    else:
        assert err.msg == msg
    [text, sep, nth] = at.rpartition("@")
    if not (sep and text and nth.isdigit()):
        [text, nth] = [at, ""]
    index = -1
    for _ in range(int(nth or 1)):
        index = src.index(text, index + 1)
    assert (err.lineno, err.offset) == (1, index + 1)
    assert (err.end_lineno, err.end_offset) == (1, index + 1 + len(text))


@pytest.mark.parametrize(
    ["src", "msg", "at"],
    [
        # Hard keywords in every name slot.
        ("(. a if)", KEYWORD, "if"),
        ("(= (. a if) 1)", KEYWORD, "if"),
        ("(.if x)", KEYWORD, ".if"),
        ("(f :if 1)", KEYWORD, "if"),
        ("(class C [:if 1] 1)", KEYWORD, "if"),
        ("(def f [if] 1)", KEYWORD, "if"),
        ("(lambda [if] 1)", KEYWORD, "if"),
        ("(def f [*if] 1)", KEYWORD, "if"),
        ("(def f [**if] 1)", KEYWORD, "if"),
        ("(def f [a * if] 1)", KEYWORD, "if"),
        ("(def f [:if 1] 1)", KEYWORD, "if"),
        ("(import if)", KEYWORD, "if"),
        ("(import a as if)", KEYWORD, "if"),
        ("(import a.if)", KEYWORD, "a.if"),
        ("(from a [if])", KEYWORD, "if"),
        ("(from if [b])", KEYWORD, "if"),
        ("(from a [b as if])", KEYWORD, "if"),
        ("(match x (case if 1))", KEYWORD, "if"),
        ("(match x (case [*if] 1))", "sequence pattern expects a name...", "*if"),
        ("(match x (case {**if} 1))", "mapping pattern expects a name...", "**if"),
        (
            "(match x (case (P :if 1) 1))",
            "class pattern expects an attribute name...",
            ":if",
        ),
        (
            "(f (else 1))",
            "else is a Python keyword and cannot be used as a name",
            "else",
        ),
        # Operators as values.
        ("+", operator("+"), "+"),
        ("(f +)", operator("+"), "+"),
        ("(f -)", operator("-"), "-"),
        ("(f ~)", operator("~"), "~"),
        ("(print ~)", operator("~"), "~"),
        ("(f ~ x)", operator("~"), "~"),
        ("(. a +)", operator("+"), "+"),
        ("(f :+ 1)", operator("+"), "+"),
        ("(def f [+] 1)", operator("+"), "+"),
        ("(import +)", operator("+"), "+"),
        ("(match x (case + 1))", operator("+"), "+"),
        # Symbols that are no identifier.
        ("a.", invalid("a."), "a."),
        ("foo!", invalid("foo!"), "foo!"),
        ("(= foo! 1)", invalid("foo!"), "foo!"),
        ("(. a b.c)", invalid("b.c"), "b.c"),
        ("(.+ x)", invalid(".+"), ".+"),
        ("(.a.b x)", invalid(".a.b"), ".a.b"),
        ("(match x (case a. 1))", "pattern expected, got a....", "a."),
        ("(try 1 (except [E as +] 1))", "(except ...) expects one name...", "as"),
        ("(global +)", "(global ...) expects a name, got +...", "+"),
        ("(def + [] 1)", "(def ...) expects a name, got +...", "+"),
        ("(class + [] 1)", "(class ...) expects a name, got +...", "+"),
        ("(def f [(g)] 1)", "expected a name, got (g)", "(g)"),
        # __debug__ where a name binds.
        ("(= __debug__ 1)", ASSIGN_DEBUG, "__debug__"),
        ("(:= __debug__ 1)", ASSIGN_DEBUG, "__debug__"),
        ("(+= __debug__ 1)", ASSIGN_DEBUG, "__debug__"),
        ("(for __debug__ in xs 1)", ASSIGN_DEBUG, "__debug__"),
        ("[x for __debug__ in xs]", ASSIGN_DEBUG, "__debug__"),
        ("(def __debug__ [] 1)", ASSIGN_DEBUG, "__debug__"),
        ("(def f [__debug__] 1)", ASSIGN_DEBUG, "__debug__"),
        ("(lambda [__debug__] 1)", ASSIGN_DEBUG, "__debug__"),
        ("(class __debug__ [] 1)", ASSIGN_DEBUG, "__debug__"),
        ("(class C [:__debug__ 1] 1)", ASSIGN_DEBUG, "__debug__"),
        ("(f :__debug__ 1)", ASSIGN_DEBUG, "__debug__"),
        ("(= a.__debug__ 1)", ASSIGN_DEBUG, "a.__debug__"),
        ("(= (. a __debug__) 1)", ASSIGN_DEBUG, "__debug__"),
        ("(del a.__debug__)", "cannot delete __debug__", "a.__debug__"),
        ("(del __debug__)", "cannot delete __debug__", "__debug__"),
        ("(import __debug__)", ASSIGN_DEBUG, "__debug__"),
        ("(import __debug__.a)", ASSIGN_DEBUG, "__debug__.a"),
        ("(import a.b as __debug__)", ASSIGN_DEBUG, "__debug__"),
        ("(from a [__debug__])", ASSIGN_DEBUG, "__debug__"),
        ("(match x (case __debug__ 1))", ASSIGN_DEBUG, "__debug__"),
        ("(match x (case [*__debug__] 1))", ASSIGN_DEBUG, "__debug__"),
        ("(match x (case {**__debug__} 1))", ASSIGN_DEBUG, "__debug__"),
        ("(match x (case (P :__debug__ 1) 1))", ASSIGN_DEBUG, "__debug__"),
        ("(match x (case 1 as __debug__ 1))", ASSIGN_DEBUG, "__debug__"),
        ("(try 1 (except [E as __debug__] 1))", ASSIGN_DEBUG, "__debug__"),
        # Unquotes and targets that were a raw ValueError/TypeError.
        ("(f ~a)", "~ (unquote) is only allowed inside a quasiquote", "~a"),
        ("(f ~@a)", "~@ (unquote-splice) is only allowed inside a quasiquote", "~@a"),
        ("[x for 1 in xs]", "cannot assign to literal", "1"),
        ("(:= 1 2)", "cannot use assignment expressions with literal", "1"),
        ("(:= a.b 1)", "cannot use assignment expressions with attribute", "a.b"),
        (
            "(:= (sub a 0) 1)",
            "cannot use assignment expressions with subscript",
            "sub a 0",
        ),
        ("(:= (, a b) 1)", "cannot use assignment expressions with tuple", "(, a b)"),
    ],
)
def test_no_python_name_is_refused_at_the_symbol(src, msg, at):
    assert_refused(src, msg, at)


needs_312 = pytest.mark.skipif(sys.version_info < (3, 12), reason="type params: 3.12+")


@needs_312
@pytest.mark.parametrize(
    ["src", "msg", "at"],
    [
        ("(def f[if] [] 1)", KEYWORD, "if"),
        ("(type T[if] int)", KEYWORD, "if"),
        ("(def f[__debug__] [] 1)", ASSIGN_DEBUG, "__debug__"),
        ("(type __debug__ int)", ASSIGN_DEBUG, "__debug__"),
    ],
)
def test_type_parameter_names(src, msg, at):
    assert_refused(src, msg, at)


@pytest.mark.parametrize(
    ["src", "python"],
    [
        ("(+ 1 2)", "1 + 2"),
        ("(~ 1)", "~1"),
        ("(< a b)", "a < b"),
        ("(and a b)", "a and b"),
        ("(match x (case P 1))", "match x:\n    case P:\n        1"),
        ("(match x (case _ 1))", "match x:\n    case _:\n        1"),
        (
            "(match x (case a.__debug__ 1))",
            "match x:\n    case a.__debug__:\n        1",
        ),
        ("(= match 1)", "match = 1"),
        ("(= type 1)", "type = 1"),
        ("(= _ 1)", "_ = 1"),
        ("(f :match 1)", "f(match=1)"),
        ("(.foo-bar x)", "x.foo_bar()"),
        ("(f :a-b 1)", "f(a_b=1)"),
        ("a?", "a_p"),
        ("a->b", "a_to_b"),
        ("__debug__", "__debug__"),
        ("(. a __debug__)", "a.__debug__"),
        ("a.__debug__", "a.__debug__"),
        ("(.__debug__ x)", "x.__debug__()"),
        ("(import a.__debug__)", "import a.__debug__"),
        ("(import __debug__ as x)", "import __debug__ as x"),
        ("(from __debug__ [a])", "from __debug__ import a"),
        ("(from a [__debug__ as x])", "from a import __debug__ as x"),
        ("(global __debug__)", "global __debug__"),
        ("(from . [x])", "from . import x"),
        ("(from ..a [x])", "from ..a import x"),
        ("(from a *)", "from a import *"),
        ("(lambda [a :b 1 *c d **e] 1)", "lambda a, b=1, *c, d, **e: 1"),
    ],
)
def test_python_names_still_compile(src, python):
    assert src_to_python_org(src) == python


def test_macro_with_a_non_identifier_name_defines_and_expands():
    """defmacro's temporary def must be a Python name whatever the macro is
    called; two such macros in one module do not clash."""
    src = (
        "(defmacro ->>> [x] (return `(+ ~x 1)))\n"
        "(defmacro ++ [x] (return `(* ~x 2)))\n"
        "(f (->>> 1) (++ 2))"
    )
    assert src_to_python_org(src).endswith("f(1 + 1, 2 * 2)")


# Modules that emit identifiers: they take names from literal.py's home.
NAME_EMITTING_MODULES = (
    expr_module,
    stmt_module,
    compiler_utils_module,
    features_module,
)


@pytest.mark.parametrize("module", NAME_EMITTING_MODULES, ids=lambda m: m.__name__)
def test_names_are_read_only_through_the_home(module):
    source = pathlib.Path(module.__file__).read_text()
    hits = [
        f"{n}: {line.strip()}"
        for [n, line] in enumerate(source.splitlines(), 1)
        if re.search(r"\.name\b", line)
    ]
    assert not hits, (
        "a symbol's mangled name is read outside the name model; take it "
        "through symbol_name/name_compile/module_path/method_name "
        "(compiler/literal.py) so every identifier is checked:\n" + "\n".join(hits)
    )
