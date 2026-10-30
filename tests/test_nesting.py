"""Deeply nested forms: compile cost linear in the nesting, and nesting past
what Python can compile is a positioned LispyCompileError, never a
RecursionError."""

import ast
import importlib
import subprocess
import sys

import pytest

import lispy.tools as tools
from lispy.core.compiler import features
from lispy.core.compiler.utils import compile_nesting_errors
from lispy.core.errors import LispyCompileError, nesting_error
from lispy.core.macro import macroexpand, macroexpand_then_compile
from lispy.core.nodes import (
    Annotation,
    Brace,
    Bracket,
    Constant,
    DoubleStarred,
    FStrExpr,
    GluedHead,
    Keyword,
    Paren,
    QuasiQuote,
    Quote,
    Starred,
    String,
    Symbol,
    Unquote,
    UnquoteSplice,
    atom_text,
    contains_unquote,
)
from lispy.core.parser import parse
from lispy.core.utils import trampoline

DEEP = 5000  # far past the recursion limit, for every recursive stage
NESTING_MESSAGE = nesting_error(Symbol("x")).msg


@pytest.fixture(scope="module")
def scope():
    scope = {}
    macroexpand_then_compile(parse("(require lispy.macros *)"), scope)
    return scope


def nest(make, depth, leaf):
    """``depth`` levels of ``make`` around ``leaf``, built without recursion."""
    node = leaf
    for _ in range(depth):
        node = make(node)
    return node


def compile_src(src, scope):
    body = macroexpand_then_compile(parse(src), scope)
    compile(ast.Module(body, type_ignores=[]), "<nesting>", "exec")
    return body


def list_depth(node):
    """How many ast.List nest in ``node``, with the innermost element."""
    depth = 0
    while isinstance(node, ast.List):
        [node] = node.elts
        depth += 1
    return depth, node


# Against renders of the whole subtree at every level, depth 40 is 2**40
# renders: only the alarm would end it.
@pytest.mark.usefixtures("time_bound")
def test_nested_lists_compile_in_linear_time(scope):
    src = "[" * 40 + "1" + "]" * 40
    [stmt] = compile_src(src, scope)
    [depth, leaf] = list_depth(stmt.value)
    assert depth == 40 and leaf.value == 1
    assert str(parse(src)[0]) == src


@pytest.mark.usefixtures("time_bound")
def test_nested_lists_in_a_macro_call_compile_in_linear_time(scope):
    src = "(when c " + "[" * 40 + "1" + "]" * 40 + ")"
    [stmt] = compile_src(src, scope)
    assert isinstance(stmt, ast.If)
    [depth, leaf] = list_depth(stmt.body[0].value)
    assert depth == 40 and leaf.value == 1


def counting(cls, bound):
    """A ``cls`` leaf counting how often its text is produced, failing past
    ``bound``. Both the old (get_source) and the new (_source) seams count,
    so a render per level fails here by count rather than by running on."""
    renders = []

    def produced():
        renders.append(None)
        assert len(renders) <= bound, f"leaf text produced over {bound} times"

    class Counting(cls):
        def get_source(self, indent=0):
            produced()
            return super().get_source(indent)

        def _source(self, indent):
            produced()
            return super()._source(indent)

    return Counting, renders


@pytest.mark.parametrize(
    "make, leaf_class, leaf_text",
    [
        (Bracket, Constant, "1"),
        (lambda e: Paren(Symbol("f"), e), Constant, "1"),
        (Paren, Symbol, "f"),
        (lambda e: Brace(Symbol(","), e), Constant, "1"),
        (lambda e: Bracket(e, Symbol("for"), Symbol("x"), Symbol("xs")), Symbol, "y"),
    ],
    ids=["bracket", "paren-arg", "paren-op", "brace-set", "comprehension"],
)
def test_leaf_text_is_produced_o_depth_times(make, leaf_class, leaf_text, scope):
    depth = 30
    bound = 4 * depth
    [cls, renders] = counting(leaf_class, bound)
    tree = nest(make, depth, cls(leaf_text, lineno=1, end_lineno=1))
    str(tree)
    assert len(renders) <= bound
    renders.clear()
    body = macroexpand_then_compile([tree], scope)
    assert len(renders) <= bound
    assert len(body) == 1


@pytest.mark.parametrize(
    "make, opener, closer",
    [(Paren, "(", ")"), (Bracket, "[", "]"), (Brace, "{", "}")],
)
def test_a_form_nested_past_the_recursion_limit_renders(make, opener, closer):
    tree = nest(make, DEEP, Constant("1"))
    expected = opener * DEEP + "1" + closer * DEEP
    assert str(tree) == expected
    assert tree.get_source(7) == expected


def test_a_deep_nest_renders_its_indentation():
    # Ten elements fill a line, so each level's last element starts a new
    # line one column right of its own bracket: the indent threads through
    # every level. (Depth past the recursion limit; the text is quadratic.)
    depth = 1500
    row = " ".join(["a"] * 10)
    tree = nest(
        lambda e: Bracket(*[Symbol("a") for _ in range(10)], e), depth, Symbol("x")
    )
    expected = "x"
    for column in range(depth - 1, -1, -1):
        expected = "[" + row + "\n" + " " * (column + 1) + expected + "]"
    assert str(tree) == expected
    assert str(nest(lambda e: Bracket(Symbol("a"), e), 2, Symbol("x"))) == "[a [a x]]"


@pytest.mark.parametrize("wrap", ["{}", "(when c {})"], ids=["plain", "macro"])
def test_macroexpand_of_a_form_nested_past_the_recursion_limit(wrap, scope):
    inner = "[" * DEEP + "1" + "]" * DEEP
    [form] = parse(wrap.format(inner))
    expanded = macroexpand(form, scope)
    if wrap == "{}":
        assert str(expanded) == inner
    else:
        assert str(expanded) == f"(if c (do {inner}))"


def plus(operands):
    """``(+ 1 1 ...)``: one lispython level, a BinOp chain ``operands`` deep."""
    return "(+" + " 1" * operands + ")"


COMPILE_BOUND = features.compile_depth_follows_recursion_limit()
needs_bound_compile = pytest.mark.skipif(
    not COMPILE_BOUND, reason="compile() depth does not follow the recursion limit"
)
needs_unbound_compile = pytest.mark.skipif(
    COMPILE_BOUND, reason="compile() depth follows the recursion limit"
)


def run_cli(argv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as info:
        tools.run()
    out, err = capsys.readouterr()
    return info.value.code, out, err


def attribute_chain(names):
    """``(. a b b ...)``: one lispython level, an Attribute chain ``names``
    deep. compile() takes as deep a chain as a BinOp chain, and our compiler
    builds it in about half the time of ``plus``."""
    return "(. a" + " b" * names + ")"


# On line 2, past compile()'s own limit on every version: 3.11 refuses it
# first (refuse_deep_ast); 3.12+ compile() raises RecursionError past 9995
# levels (3.12/3.13), 47370 (3.14) and 26067 (3.15).
TOO_DEEP = "\n" + attribute_chain(100_000)


@pytest.mark.usefixtures("time_bound")
def test_lpy_reports_a_too_deep_form_compiler_style(monkeypatch, capsys):
    [code, out, err] = run_cli(["lpy", "-c", TOO_DEEP], monkeypatch, capsys)
    assert code == 1
    assert err.startswith("<string>:2:")
    assert f": error: {NESTING_MESSAGE}\n" in err
    assert "Traceback" not in out + err


@pytest.mark.usefixtures("time_bound")
def test_importing_a_too_deep_module_is_a_positioned_error(tmp_path, monkeypatch):
    module = tmp_path / "lpy_too_deep_module.lpy"
    module.write_text("(= x 1)" + TOO_DEEP)
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with pytest.raises(LispyCompileError) as info:
            importlib.import_module("lpy_too_deep_module")
    finally:
        sys.modules.pop("lpy_too_deep_module", None)
    err = info.value
    assert (err.msg, err.lineno, err.filename) == (NESTING_MESSAGE, 2, str(module))


@pytest.mark.usefixtures("time_bound")
def test_the_repl_reports_a_too_deep_form():
    with pytest.raises(LispyCompileError) as info:
        tools.eval_and_print(TOO_DEEP, {}, filename="<repl>")
    err = info.value
    assert (err.msg, err.lineno, err.filename) == (NESTING_MESSAGE, 2, "<repl>")


@pytest.mark.usefixtures("time_bound")
def test_the_nrepl_reports_a_too_deep_form():
    error = tools.nrepl_eval(TOO_DEEP, {}, filename="buffer.lpy")["error"]
    assert error.startswith("buffer.lpy:2:")
    assert f": error: {NESTING_MESSAGE}" in error
    assert "Traceback" not in error


@pytest.mark.usefixtures("time_bound")
def test_a_too_deep_defmacro_body_is_a_positioned_error():
    src = "(defmacro m []" + TOO_DEEP + ")"
    with pytest.raises(LispyCompileError) as info:
        macroexpand_then_compile(parse(src), {})
    assert (info.value.msg, info.value.lineno) == (NESTING_MESSAGE, 2)


# A macro whose expansion recurses without end: that RecursionError is the
# macro's own bug, with a traceback into it, never a nesting error.
RUNAWAY_MACRO = """
(defmacro runaway-macro []
  (def f [n] (return (f n)))
  (return (f 1)))
(runaway-macro)
"""


@pytest.mark.usefixtures("time_bound")
def test_a_macro_s_runaway_recursion_keeps_its_recursion_error():
    with pytest.raises(RecursionError) as info:
        macroexpand_then_compile(parse(RUNAWAY_MACRO), {})
    assert not isinstance(info.value, LispyCompileError)


@pytest.mark.usefixtures("time_bound")
def test_lpy_reports_a_macro_s_runaway_recursion_as_its_own(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["lpy", "-c", RUNAWAY_MACRO])
    with pytest.raises(RecursionError) as info:
        tools.run()
    assert not isinstance(info.value, LispyCompileError)
    [out, err] = capsys.readouterr()
    assert NESTING_MESSAGE not in out + err


def at(line, col, end_col=None):
    end = col + 1 if end_col is None else end_col
    return dict(lineno=line, col_offset=col, end_lineno=line, end_col_offset=end)


def reported_at(stmts):
    """Where compile_nesting_errors reports a RecursionError in its block."""
    with pytest.raises(LispyCompileError) as info:
        with compile_nesting_errors(stmts):
            raise RecursionError
    assert info.value.msg == NESTING_MESSAGE
    assert info.value.__suppress_context__
    return info.value.lineno, info.value.offset


def test_compile_nesting_errors_reports_the_first_deepest_path():
    load = ast.Load()
    shallow = ast.Expr(ast.Name("s", load, **at(1, 0)), **at(1, 0))
    deeper = ast.Expr(
        ast.BinOp(
            ast.BinOp(
                ast.Name("x", load, **at(2, 5)),
                ast.Add(),
                ast.Name("y", load, **at(2, 7)),
                **at(2, 3, 9),
            ),
            ast.Add(),
            ast.Constant(1, **at(2, 11)),
            **at(2, 0, 12),
        ),
        **at(2, 0, 12),
    )
    # The second statement is deeper; its paths to x and y tie, so x.
    assert reported_at([shallow, deeper]) == (2, 6)
    # Equally deep statements: the first.
    other = ast.Expr(ast.Name("t", load, **at(3, 0)), **at(3, 0))
    assert reported_at([shallow, other]) == (1, 1)


def test_compile_nesting_errors_reports_an_unpositioned_bottom_at_its_ancestor():
    # (lambda [] y): the empty ``arguments`` ties with the body and comes
    # first, but has no position, so the error is at the lambda.
    no_args = ast.arguments(
        posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
    )
    lam = ast.Lambda(no_args, ast.Name("y", ast.Load(), **at(4, 12)), **at(4, 2, 14))
    assert reported_at([ast.Expr(lam, **at(4, 2, 14))]) == (4, 3)


def test_compile_nesting_errors_leaves_other_errors_alone():
    with pytest.raises(ValueError):
        with compile_nesting_errors([]):
            raise ValueError


DEPTH = 5000  # the compiler's nesting must cost no Python frame per level


LINE_3 = {"lineno": 3, "end_lineno": 3, "col_offset": 0, "end_col_offset": 1}


def fstring(field):
    """``f"{field}"`` as the parser builds it, on line 3."""
    return Paren(
        Symbol("f-string", **LINE_3),
        String("''", **LINE_3),
        field,
        String("''", **LINE_3),
        **LINE_3,
    )


def field(value, spec=None):
    """An f-string replacement field of ``value``, format spec ``spec``."""
    return FStrExpr(
        value, expr_text="x", conversion=-1, format_spec=spec, debug=None, **LINE_3
    )


# (id, source nesting DEPTH levels on its line, or a function building that
# form as nodes; the AST class one level adds exactly once; whether to
# compile() the AST too). Sources are built by repetition, never recursion.
# compile() is skipped where CPython 3.12+ cannot take the AST at this depth
# (two AST levels per level pass its ~10000 limit on 3.12/3.13; nested
# comprehensions crash it) or takes over a second (nested scopes, tuple
# constant folding, ife).
EXPRESSION_DEPTHS = [
    ("bracket", "[" * DEPTH + "1" + "]" * DEPTH, ast.List, True),
    ("call-argument", "(f " * DEPTH + "1" + ")" * DEPTH, ast.Call, True),
    ("call-head", "(" * DEPTH + "f" + ")" * DEPTH, ast.Call, True),
    ("keyword-value", "(f :k " * DEPTH + "1" + ")" * DEPTH, ast.keyword, False),
    ("doublestar-value", "(f **" * DEPTH + "d" + ")" * DEPTH, ast.keyword, False),
    ("starred", "(f *" * DEPTH + "xs" + ")" * DEPTH, ast.Starred, False),
    ("attribute", "(. " * DEPTH + "a" + " b)" * DEPTH, ast.Attribute, True),
    ("method-instance", "(.m " * DEPTH + "a" + ")" * DEPTH, ast.Call, False),
    ("method-argument", "(.m a " * DEPTH + "1" + ")" * DEPTH, ast.Call, True),
    ("binop", "(+ 1 " * DEPTH + "1" + ")" * DEPTH, ast.BinOp, True),
    ("unaryop", "(- " * DEPTH + "1" + ")" * DEPTH, ast.UnaryOp, True),
    ("boolop", "(and a " * DEPTH + "b" + ")" * DEPTH, ast.BoolOp, True),
    ("compare", "(< a " * DEPTH + "b" + ")" * DEPTH, ast.Compare, True),
    ("tuple", "(, " * DEPTH + "1" + ")" * DEPTH, ast.Tuple, False),
    ("set", "{, " * DEPTH + "1" + "}" * DEPTH, ast.Set, True),
    ("dict-key", "{" * DEPTH + "1 2}" + " 2}" * (DEPTH - 1), ast.Dict, True),
    ("dict-value", "{1 " * DEPTH + "2" + "}" * DEPTH, ast.Dict, True),
    ("subscript-value", "(sub " * DEPTH + "a" + " 0)" * DEPTH, ast.Subscript, True),
    ("subscript-slice", "(sub a " * DEPTH + "0" + ")" * DEPTH, ast.Subscript, True),
    (
        "slice-bound",
        "(sub a " + "[: " * DEPTH + "1" + "]" * DEPTH + ")",
        ast.Slice,
        True,
    ),
    ("ife", "(ife c " * DEPTH + "1" + " 2)" * DEPTH, ast.IfExp, False),
    ("walrus-value", "(:= x " * DEPTH + "1" + ")" * DEPTH, ast.NamedExpr, True),
    ("lambda-body", "(lambda [] " * DEPTH + "1" + ")" * DEPTH, ast.Lambda, False),
    (
        "lambda-default",
        "(lambda [:a " * DEPTH + "1" + "] a)" * DEPTH,
        ast.Lambda,
        False,
    ),
    (
        "listcomp-element",
        "[" * DEPTH + "x" + " for x in xs]" * DEPTH,
        ast.ListComp,
        False,
    ),
    (
        "listcomp-iterable",
        "[x for x in " * DEPTH + "xs" + "]" * DEPTH,
        ast.ListComp,
        False,
    ),
    (
        "setcomp-element",
        "{, " * DEPTH + "x" + " for x in xs}" * DEPTH,
        ast.SetComp,
        False,
    ),
    (
        "setcomp-iterable",
        "{, x for x in " * DEPTH + "xs" + "}" * DEPTH,
        ast.SetComp,
        False,
    ),
    (
        "dictcomp-value",
        "{k " * DEPTH + "v" + " for x in xs}" * DEPTH,
        ast.DictComp,
        False,
    ),
    (
        "dictcomp-iterable",
        "{k v for x in " * DEPTH + "xs" + "}" * DEPTH,
        ast.DictComp,
        False,
    ),
    (
        "genexp-element",
        "(" * DEPTH + "x" + " for x in xs)" * DEPTH,
        ast.GeneratorExp,
        False,
    ),
    (
        "genexp-iterable",
        "(x for x in " * DEPTH + "xs" + ")" * DEPTH,
        ast.GeneratorExp,
        False,
    ),
    (
        "yield",
        "(def g [] " + "(yield " * DEPTH + "1" + ")" * DEPTH + ")",
        ast.Yield,
        True,
    ),
    (
        "yield-from",
        "(def g [] " + "(yield-from " * DEPTH + "1" + ")" * DEPTH + ")",
        ast.YieldFrom,
        True,
    ),
    (
        "await",
        "(async-def g [] " + "(await " * DEPTH + "1" + ")" * DEPTH + ")",
        ast.Await,
        True,
    ),
    # The parser recurses into a nested f-string's fields, so these are nodes.
    (
        "fstring-field",
        lambda: nest(lambda e: fstring(field(e)), DEPTH, Symbol("x", **LINE_3)),
        ast.JoinedStr,
        False,
    ),
    (
        "fstring-format-spec",
        lambda: fstring(
            nest(
                lambda s: field(Symbol("x", **LINE_3), fstring(s)),
                DEPTH - 1,
                field(Symbol("x", **LINE_3)),
            )
        ),
        ast.JoinedStr,
        False,
    ),
    (
        "macro-argument",
        "(when c " + "[" * DEPTH + "1" + "]" * DEPTH + ")",
        ast.List,
        True,
    ),
    (
        "assignment-target",
        "(= " + "(, " * DEPTH + "a" + ")" * DEPTH + " x)",
        ast.Tuple,
        True,
    ),
    # A quoted [...] is a (Bracket ...) call per level.
    ("quote", "'" + "[" * DEPTH + "]" * DEPTH, ast.Call, True),
    ("quasiquote-unquote", "`" + "[" * DEPTH + "~x" + "]" * DEPTH, ast.Call, True),
    (
        "quasiquote-unquote-splice",
        "`" + "[" * DEPTH + "~@xs" + "]" * DEPTH,
        ast.Call,
        True,
    ),
]


def calls_to(name):
    """A predicate: the AST node is a call of the bare name ``name``."""
    return lambda node: (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    )


def quoted(make):
    """A function building ``'x`` nested DEPTH levels deep by ``make``, on
    line 3 (the parser cannot spell some of these shapes that deep)."""
    return lambda: Quote(nest(make, DEPTH, Symbol("x", **LINE_3)), **LINE_3)


# Each node class's own quote expansion: (id, make one level, the class its
# expansion calls once per level, whether to compile() it).
QUOTED_NODE_DEPTHS = [
    ("keyword", lambda e: Keyword(e, **LINE_3), "Keyword", True),
    ("starred", lambda e: Starred(e, **LINE_3), "Starred", True),
    ("annotation", lambda e: Annotation(e, **LINE_3), "Annotation", True),
    (
        "glued-head",
        lambda e: GluedHead(Symbol("f", **LINE_3), e, **LINE_3),
        "GluedHead",
        True,
    ),
    ("fstring-field", lambda e: field(e), "FStrExpr", True),
    (
        "fstring-format-spec",
        lambda e: field(Symbol("x", **LINE_3), fstring(e)),
        "FStrExpr",
        False,
    ),
    ("quote", lambda e: Quote(e, **LINE_3), "Quote", True),
    ("unquote-outside-quasiquote", lambda e: Unquote(e, **LINE_3), "Unquote", True),
]


@pytest.mark.usefixtures("time_bound")
@pytest.mark.parametrize(
    "make, name, compiles",
    [case[1:] for case in QUOTED_NODE_DEPTHS],
    ids=[case[0] for case in QUOTED_NODE_DEPTHS],
)
def test_quoted_nodes_nest_past_the_recursion_limit(make, name, compiles, scope):
    assert_compiles_at_depth(quoted(make), calls_to(name), compiles, scope)


def assert_compiles_at_depth(form, cls, compiles, scope, flat=False):
    """``form`` (source, or a function building it as nodes) compiles to an
    AST with DEPTH ``cls`` nodes (an AST class, or a predicate on a node),
    which compile() takes when ``compiles``;
    where compile() is bound by the recursion limit (3.11) it is refused as
    too deep, positioned on line 3, unless the AST is ``flat`` (do, deco
    nest in lispython only). Asserts on plain values only: a failing
    comparison would repr a deep node or AST after the timer stopped."""
    forms = [form()] if callable(form) else parse("\n\n" + form)
    if COMPILE_BOUND and not flat:
        with pytest.raises(LispyCompileError) as info:
            macroexpand_then_compile(forms, scope)
        assert info.value.msg == NESTING_MESSAGE
        assert info.value.lineno == 3
        return
    module = ast.Module(macroexpand_then_compile(forms, scope), type_ignores=[])
    matches = (lambda node: isinstance(node, cls)) if isinstance(cls, type) else cls
    assert sum(1 for node in ast.walk(module) if matches(node)) == DEPTH
    if compiles:
        compile(module, "<nesting>", "exec")


@pytest.mark.usefixtures("time_bound")
@pytest.mark.parametrize(
    "form, cls, compiles",
    [case[1:] for case in EXPRESSION_DEPTHS],
    ids=[case[0] for case in EXPRESSION_DEPTHS],
)
def test_expressions_nest_past_the_recursion_limit(form, cls, compiles, scope):
    assert_compiles_at_depth(form, cls, compiles, scope)


TYPE_PARAMS = (
    tuple(features.running_version())[:2]
    >= features.FEATURES["type-params"].min_version
)

# As EXPRESSION_DEPTHS, and whether the AST is flat. compile() is skipped
# for nested loop, with and try blocks (CPython refuses over 20), nested
# scopes (def, class: seconds) and nested match (3.12/3.13 refuse it).
STATEMENT_DEPTHS = [
    ("do", "(do (f) " * DEPTH + ")" * DEPTH, ast.Call, True, True),
    ("if-then", "(if c " * DEPTH + "(f)" + ")" * DEPTH, ast.If, True, False),
    ("if-else", "(if c (f) " * DEPTH + "(f)" + ")" * DEPTH, ast.If, True, False),
    ("while-body", "(while c " * DEPTH + "(f)" + ")" * DEPTH, ast.While, False, False),
    ("for-body", "(for x in xs " * DEPTH + "(f)" + ")" * DEPTH, ast.For, False, False),
    ("with-body", "(with [a] " * DEPTH + "(f)" + ")" * DEPTH, ast.With, False, False),
    (
        "try-body",
        "(try " * DEPTH + "(f)" + " (finally (g)))" * DEPTH,
        ast.Try,
        False,
        False,
    ),
    (
        "def-body",
        "(def f [] " * DEPTH + "(g)" + ")" * DEPTH,
        ast.FunctionDef,
        False,
        False,
    ),
    (
        "class-body",
        "(class C [] " * DEPTH + "(g)" + ")" * DEPTH,
        ast.ClassDef,
        False,
        False,
    ),
    ("deco", "(deco d " * DEPTH + "(def f [] 1)" + ")" * DEPTH, ast.Name, True, True),
    (
        "case-body",
        "(match x (case 1 " * DEPTH + "(f)" + "))" * DEPTH,
        ast.Match,
        False,
        False,
    ),
    (
        "sequence-pattern",
        "(match x (case " + "[" * DEPTH + "a" + "]" * DEPTH + " 1))",
        ast.MatchSequence,
        True,
        False,
    ),
    (
        "class-pattern",
        "(match x (case " + "(C " * DEPTH + "a" + ")" * DEPTH + " 1))",
        ast.MatchClass,
        True,
        False,
    ),
    (
        "mapping-pattern",
        "(match x (case " + "{1 " * DEPTH + "a" + "}" * DEPTH + " 1))",
        ast.MatchMapping,
        True,
        False,
    ),
    (
        "or-pattern",
        "(match x (case " + "(| 1 " * DEPTH + "2" + ")" * DEPTH + " 1))",
        ast.MatchOr,
        True,
        False,
    ),
    (
        "case-as",
        "(match x (case " + "[" * DEPTH + "a" + "]" * DEPTH + " as y 1))",
        ast.MatchSequence,
        True,
        False,
    ),
    pytest.param(
        "(def f[T ^" + "[" * DEPTH + "int" + "]" * DEPTH + "] [] 1)",
        ast.List,
        True,
        False,
        id="type-param-bound",
        marks=pytest.mark.skipif(not TYPE_PARAMS, reason="type params are 3.12+"),
    ),
]


@pytest.mark.usefixtures("time_bound")
@pytest.mark.parametrize(
    "form, cls, compiles, flat",
    [
        pytest.param(*case[1:], id=case[0]) if type(case) is tuple else case
        for case in STATEMENT_DEPTHS
    ],
)
def test_statements_and_patterns_nest_past_the_recursion_limit(
    form, cls, compiles, flat, scope
):
    assert_compiles_at_depth(form, cls, compiles, scope, flat)


@pytest.mark.parametrize(
    "version, bound", [((3, 11, 9), True), ((3, 12, 0), False), ((3, 15, 0), False)]
)
def test_compile_depth_follows_the_recursion_limit_before_3_12(
    version, bound, monkeypatch
):
    # 3.11 compile() follows the recursion limit; 3.12+ has a C limit
    # (3.12/3.13) or the C stack (3.14+), thousands of levels deeper.
    monkeypatch.setattr(features, "running_version", lambda: version)
    assert features.compile_depth_follows_recursion_limit() is bound


@needs_bound_compile
def test_an_ast_too_deep_for_cpython_is_a_positioned_error(scope):
    with pytest.raises(LispyCompileError) as info:
        compile_src("\n" + plus(3000), scope)
    err = info.value
    assert err.msg == NESTING_MESSAGE
    assert err.lineno == 2 and err.offset >= 1


@needs_unbound_compile
def test_a_deep_ast_compiles_where_compile_is_not_bound(scope):
    [stmt] = compile_src(plus(3000), scope)
    code = compile(ast.Expression(stmt.value), "<nesting>", "eval")
    assert eval(code) == 3000


def compile_in_this_frame(src, scope):
    """Expand, compile to AST and compile() in one frame, as define_macro does
    (the earliest frame refuse_deep_ast allows for)."""
    body = macroexpand_then_compile(parse(src), scope)
    compile(ast.Module(body, type_ignores=[]), "<nesting>", "exec")


@needs_bound_compile
@pytest.mark.parametrize(
    "make",
    [
        plus,
        lambda n: "(+" + " a" * n + ")",
        lambda n: "(. a" + " b" * n + ")",
    ],
    ids=["constant-leaves", "name-leaves", "attribute-chain"],
)
def test_the_check_never_refuses_what_compile_takes(make, scope, monkeypatch):
    # Every attempt runs at the same stack depth, through this one helper.
    def outcome(operands):
        try:
            compile_in_this_frame(make(operands), scope)
            return "compiled"
        except LispyCompileError:
            return "refused"
        except RecursionError:
            return "RecursionError"

    # The deepest sum compile() itself takes, with the check off.
    with monkeypatch.context() as patch:
        patch.setattr(features, "compile_depth_follows_recursion_limit", lambda: False)
        [low, high] = [1, 3000]
        while low < high:
            mid = (low + high + 1) // 2
            compiled = outcome(mid) == "compiled"
            [low, high] = [mid, high] if compiled else [low, mid - 1]
        assert outcome(low + 1) == "RecursionError"
    assert 900 < low < 1000
    # The check never refuses it. It counts Python frames, and C-level calls
    # on the stack (pytest has some) also spend recursion depth, so a few
    # levels past it can still meet compile()'s RecursionError; well past
    # it, the check refuses.
    assert outcome(low) == "compiled"
    assert outcome(low + 20) == "refused"


# Run in a fresh interpreter, whose stack has no C-level calls: there the
# frame count is exact, so the check refuses exactly what compile() rejects.
EXACT_BOUNDARY = """
import ast, sys
from lispy.core.compiler import features
from lispy.core.errors import LispyCompileError
from lispy.core.macro import macroexpand_then_compile
from lispy.core.parser import parse
from lispy.core.utils import trampoline

def outcome(src):
    try:
        body = macroexpand_then_compile(parse(src), {})
        compile(ast.Module(body, type_ignores=[]), "<n>", "exec")
        return "compiled"
    except LispyCompileError:
        return "refused"
    except RecursionError:
        return "RecursionError"

make = eval(sys.argv[1])
check = features.compile_depth_follows_recursion_limit
features.compile_depth_follows_recursion_limit = lambda: False
# A plain loop: a generator expression would run outcome a frame deeper.
low = 900
while low < 2000 and outcome(make(low + 1)) == "compiled":
    low += 1
off = outcome(make(low + 1))
features.compile_depth_follows_recursion_limit = check
print(off, outcome(make(low)), outcome(make(low + 1)))
"""


@needs_bound_compile
@pytest.mark.parametrize(
    "make",
    [
        'lambda n: "(+" + " 1" * n + ")"',
        'lambda n: "(+" + " a" * n + ")"',
        'lambda n: "(. a" + " b" * n + ")"',
    ],
    ids=["constant-leaves", "name-leaves", "attribute-chain"],
)
def test_the_check_refuses_exactly_what_compile_rejects(make):
    result = subprocess.run(
        [sys.executable, "-c", EXACT_BOUNDARY, make],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["RecursionError", "compiled", "refused"]


@pytest.mark.parametrize("bound", [True, False])
def test_the_check_follows_the_version_comparison(bound, scope, monkeypatch):
    monkeypatch.setattr(
        features, "compile_depth_follows_recursion_limit", lambda: bound
    )
    if bound:
        with pytest.raises(LispyCompileError):
            macroexpand_then_compile(parse(plus(3000)), scope)
    else:
        assert len(macroexpand_then_compile(parse(plus(3000)), scope)) == 1


def test_the_check_reports_the_first_too_deep_path(scope, monkeypatch):
    monkeypatch.setattr(features, "compile_depth_follows_recursion_limit", lambda: True)
    src = f"(f\n  {plus(3000)}\n  {plus(3000)})"
    with pytest.raises(LispyCompileError) as info:
        macroexpand_then_compile(parse(src), scope)
    assert (info.value.lineno, info.value.offset) == (2, 3)


def test_trivial_code_compiles_with_a_deep_caller_stack(scope):
    # The budget shrinks with the caller's stack; a shallow statement must
    # still compile as long as the stack leaves room for compile() itself.
    def at_depth(frames):
        if frames:
            return at_depth(frames - 1)
        compile_in_this_frame("(= x (+ 1 2))", scope)
        return True

    assert at_depth(sys.getrecursionlimit() - 200)


def test_a_deep_ast_within_the_limit_compiles(scope):
    [stmt] = compile_src(plus(500), scope)
    code = compile(ast.Expression(stmt.value), "<nesting>", "eval")
    assert eval(code) == 500


def test_l2py_of_an_ast_unparse_cannot_write_is_a_positioned_error(scope):
    # ast.unparse recurses several frames per level and gives up on an AST
    # compile() still takes (the test above).
    src = "(= x 1)\n" + plus(500)
    with pytest.raises(LispyCompileError) as info:
        tools.l2py_s(src, fresh_scope=True, format=False)
    err = info.value
    assert err.msg == NESTING_MESSAGE
    assert (err.lineno, err.offset) == (2, 1)


WORDS = ["for", "in", "a", "(a b)", "[]", ":x", "", "(FStrExpr a)", "f[T]"]

NODES = [
    Symbol("a"),
    Symbol("for"),
    String('"s"'),
    Constant("1"),
    Constant("'1'"),
    Paren(Symbol("a"), Symbol("b")),
    Paren(),
    Bracket(),
    Bracket(Symbol(":"), Symbol("a")),
    Brace(Symbol(","), Symbol("a")),
    FStrExpr(Symbol("a")),
    GluedHead(Symbol("f"), Bracket(Symbol("T"))),
    GluedHead(Symbol("f"), Symbol("T")),
    GluedHead(Paren(Symbol("f")), Symbol("T")),
    Keyword(Symbol("x")),
    Keyword(Paren(Symbol("x"))),
    Annotation(Symbol("int")),
    Starred(Symbol("args")),
    Starred(Bracket(Symbol("a"))),
    DoubleStarred(Symbol("kw")),
    Quote(Symbol("a")),
    Quote(Bracket(Symbol("a"))),
    QuasiQuote(Keyword(Symbol("a"))),
    Unquote(Symbol("a")),
    UnquoteSplice(Paren(Symbol("a"))),
    Keyword(Starred(Quote(Symbol("deep")))),
]


@pytest.mark.parametrize("node", NODES, ids=repr)
def test_atom_text_is_the_text_of_a_form_without_brackets(node):
    text = atom_text(node)
    if text is None:
        assert any(c in str(node) for c in "([{")
    else:
        assert text == str(node)


@pytest.mark.parametrize(
    "node, text",
    [
        (Paren(Symbol("a")), None),
        (Bracket(), None),
        (Brace(), None),
        (FStrExpr(Symbol("a")), None),
        (Keyword(Paren(Symbol("x"))), None),
        (GluedHead(Symbol("f"), Bracket(Symbol("T"))), None),
        (GluedHead(Symbol("f"), Symbol("T")), "fT"),
        (Keyword(Starred(Quote(Symbol("deep")))), ":*'deep"),
        (Constant("'1'"), "1"),
    ],
    ids=repr,
)
def test_atom_text_of_each_shape(node, text):
    assert atom_text(node) == text


@pytest.mark.parametrize("node", NODES, ids=repr)
def test_a_node_equals_exactly_its_rendering(node):
    assert node == str(node)
    assert not node != str(node)
    for word in WORDS:
        assert (node == word) == (str(node) == word)


def test_a_compound_node_equals_its_rendering():
    assert Paren(Symbol("a"), Symbol("b")) == "(a b)"
    assert Keyword(Symbol("x")) == ":x"
    assert Paren(Symbol("a"), Symbol("b")) != "a"
    assert Paren(Symbol("for")) != "for"


# Against a pattern rendered or re-walked at every level, depth 40 is 2**40
# steps: only the alarm would end it.
@pytest.mark.usefixtures("time_bound")
@pytest.mark.parametrize(
    "pattern",
    [
        "[" * 40 + "a" + "]" * 40,
        "(C " * 40 + "a" + ")" * 40,
        "{1 " * 40 + "a" + "}" * 40,
    ],
    ids=["sequence", "class", "mapping"],
)
def test_nested_case_patterns_compile_in_linear_time(pattern, scope):
    [stmt] = compile_src(f"(match x (case {pattern} 1))", scope)
    assert isinstance(stmt, ast.Match)


@pytest.mark.parametrize(
    "target, msg, col",
    [
        # A tuple's elements are all checked before its own starred check...
        ("(, *a *b (f))", "cannot assign to function call", 13),
        # ...which comes before the elements after that tuple.
        ("(, (, *a *b) (f))", "multiple starred expressions in assignment", 13),
        ("(, (f) (, *a *b))", "cannot assign to function call", 7),
        ("[a *(f) 1]", "cannot assign to function call", 8),
    ],
)
def test_a_target_is_refused_where_a_left_to_right_walk_first_fails(target, msg, col):
    with pytest.raises(LispyCompileError) as info:
        macroexpand_then_compile(parse(f"(= {target} x)"), {})
    assert (info.value.msg, info.value.offset) == (msg, col)


@pytest.mark.usefixtures("time_bound")
@pytest.mark.parametrize("unquote", [True, False])
def test_contains_unquote_past_the_recursion_limit(unquote):
    leaf = Unquote(Symbol("x")) if unquote else Symbol("x")
    found = contains_unquote(nest(lambda e: Bracket(Keyword(e)), DEPTH, leaf))
    assert found is unquote  # a plain value: a failing node would be rendered


@pytest.mark.usefixtures("time_bound")
@pytest.mark.parametrize("unquote", [True, False])
def test_contains_unquote_visits_a_shared_node_once(unquote):
    # 2**60 paths through 60 nodes: a walk per path would never end.
    leaf = Unquote(Symbol("x")) if unquote else Symbol("x")
    found = contains_unquote(nest(lambda e: Paren(e, e), 60, leaf))
    assert found is unquote


def test_the_trampoline_runs_steps_as_recursion_would():
    log = []

    def leaf(n):
        if n == 0:
            raise ValueError("leaf")
        return n
        yield

    def step(depth):
        # A non-generator is its own result; a generator is run.
        assert (yield 7) == 7
        if depth == 0:
            return (yield leaf(1)) + (yield leaf(1))
        try:
            return 1 + (yield step(depth - 1))
        finally:
            log.append(depth)

    def catching():
        try:
            yield leaf(0)
        except ValueError as err:
            return f"caught {err}"

    assert trampoline(step(DEPTH)) == DEPTH + 2
    assert log == list(range(1, DEPTH + 1))
    assert trampoline(catching()) == "caught leaf"
    assert trampoline("done") == "done"
    log.clear()

    def failing(depth):
        try:
            return (yield failing(depth - 1)) if depth else (yield leaf(0))
        finally:
            log.append(depth)

    with pytest.raises(ValueError):
        trampoline(failing(DEPTH))
    assert log == list(range(DEPTH + 1))
