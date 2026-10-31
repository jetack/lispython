"""Deeply nested forms: compile cost linear in the nesting, and nesting past
what Python can compile is a positioned LispyCompileError, never a
RecursionError."""

import ast
import importlib
import sys
import time
import tracemalloc

import pytest

import lispy.core.parser as parser_module
import lispy.tools as tools
from lispy.core import nodes
from lispy.core.compiler import features
from lispy.core.compiler import stmt as stmt_module
from lispy.core.compiler.utils import compile_nesting_errors
from lispy.core.errors import LispyCompileError
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
    Node,
    Paren,
    QuasiQuote,
    Quote,
    Starred,
    String,
    Symbol,
    Unquote,
    UnquoteSplice,
    Wrapper,
    atom_text,
    contains_unquote,
)
from lispy.core.parser import parse, tokenize
from lispy.core.utils import trampoline

DEEP = 5000  # far past the recursion limit, for every recursive stage
# The nesting error names the recursion limit only where it is what was hit:
# compile() on 3.11 and ast.unparse everywhere, never compile() on 3.12+.
C_LIMIT_MESSAGE = "form nested too deeply to compile"
RECURSION_LIMIT_MESSAGE = (
    f"{C_LIMIT_MESSAGE} (Python's recursion limit is {sys.getrecursionlimit()})"
)
TRANSLATE_MESSAGE = (
    "form nested too deeply to translate"
    f" (Python's recursion limit is {sys.getrecursionlimit()})"
)


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
    """Expand, build and compile() ``src``, as the entry points do."""
    body = macroexpand_then_compile(parse(src), scope)
    with compile_nesting_errors(body):
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
    # Counts renders and never compile()s: 30 nested comprehensions pass
    # CPython's static block limit, where compile() can crash.
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
# What this interpreter reports for a form past compile()'s limit.
NESTING_MESSAGE = RECURSION_LIMIT_MESSAGE if COMPILE_BOUND else C_LIMIT_MESSAGE
needs_bound_compile = pytest.mark.skipif(
    not COMPILE_BOUND, reason="compile() depth does not follow the recursion limit"
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


# On line 2, past compile()'s own limit on every version: 3.11 compile()
# raises RecursionError near the recursion limit, 3.12+ past 9995 levels
# (3.12/3.13), 47370 (3.14) and 26067 (3.15).
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


# Two statements past compile()'s limit on every version (55,000 levels
# clears 3.14's 47,370), the second deeper: the error is at the first, the
# one compile() fails on, not at the deepest of both.
TWO_TOO_DEEP = "\n" + attribute_chain(55_000) + "\n" + attribute_chain(56_000)


@pytest.mark.usefixtures("time_bound")
def test_the_repl_reports_the_first_too_deep_statement():
    with pytest.raises(LispyCompileError) as info:
        tools.eval_and_print(TWO_TOO_DEEP, {}, filename="<repl>")
    assert (info.value.msg, info.value.lineno) == (NESTING_MESSAGE, 2)


@pytest.mark.usefixtures("time_bound")
def test_the_nrepl_reports_the_first_too_deep_statement():
    error = tools.nrepl_eval(TWO_TOO_DEEP, {}, filename="buffer.lpy")["error"]
    assert error.startswith("buffer.lpy:2:")
    assert f": error: {NESTING_MESSAGE}\n" in error


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


@pytest.mark.parametrize(
    "bound, message",
    [(True, RECURSION_LIMIT_MESSAGE), (False, C_LIMIT_MESSAGE)],
    ids=["recursion-limit-bound", "c-limit-bound"],
)
def test_compile_nesting_errors_names_the_recursion_limit_only_where_it_binds(
    bound, message, monkeypatch
):
    # On 3.12+ compile()'s limit is CPython's own, which raising the
    # recursion limit does not move: the message must not name it there.
    monkeypatch.setattr(
        features, "compile_depth_follows_recursion_limit", lambda: bound
    )
    stmt = ast.Expr(ast.Name("s", ast.Load(), **at(1, 0)), **at(1, 0))
    with pytest.raises(LispyCompileError) as info:
        with compile_nesting_errors([stmt]):
            raise RecursionError
    assert info.value.msg == message


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
# (two AST levels per level pass its ~10000 limit on 3.12/3.13,
# comprehensions nested in an iterable), takes over a second (nested scopes,
# genexps, tuple constant folding, ife), or can crash: list/set/dict
# comprehensions nested in the element pass CPython's static block limit,
# which lispython leaves to CPython.
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
    # Nodes: parsing an f-string nest costs the rest of the text per level.
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
    which compile() takes when ``compiles``; where compile() is bound by the
    recursion limit (3.11) it refuses that AST, a nesting error on line 3
    through ``compile_nesting_errors``, unless the AST is ``flat`` (do, deco
    nest in lispython only). The compiler itself takes the depth on every
    version. Asserts on plain values only: a failing comparison would repr a
    deep node or AST after the timer stopped."""
    forms = [form()] if callable(form) else parse("\n\n" + form)
    matches = (lambda node: isinstance(node, cls)) if isinstance(cls, type) else cls
    body = macroexpand_then_compile(forms, scope)
    module = ast.Module(body, type_ignores=[])
    assert sum(1 for node in ast.walk(module) if matches(node)) == DEPTH
    if not compiles:
        return
    if COMPILE_BOUND and not flat:
        with pytest.raises(LispyCompileError) as info:
            with compile_nesting_errors(body):
                compile(module, "<nesting>", "exec")
        assert info.value.msg == NESTING_MESSAGE
        assert info.value.lineno == 3
    else:
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
    assert err.msg == TRANSLATE_MESSAGE
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


T_STRINGS = (
    tuple(features.running_version())[:2] >= features.FEATURES["t-string"].min_version
)


def nested_literals(openings, depth, newline=""):
    """``depth`` f/t-string literals nested in each other's single field,
    cycling through ``openings`` from the outside; each opening is 3
    characters, then ``newline``."""
    heads = [openings[k % len(openings)] for k in range(depth)]
    return "".join(h + newline for h in heads) + "x" + '}"' * depth


def literal_depth(node):
    """How many f/t-strings nest through their single field, with the
    innermost field's value."""
    depth = 0
    while isinstance(node, Paren) and atom_text(node.list[0]) in (
        "f-string",
        "t-string",
    ):
        node = node.list[2].value
        depth += 1
    return depth, node


# (id, openings, needs t-strings): every literal is counted, f or t alike.
NESTED_LITERALS = [
    ("f-strings", ['f"{'], False),
    ("t-strings", ['t"{'], True),
    ("mixed", ['f"{', 't"{'], True),
]


def skip_without_t_strings(needs_t):
    if needs_t and not T_STRINGS:
        pytest.skip("t-strings need Python 3.14")


# Past the 149 literals CPython's lexer takes in Python source: the AST has
# no such limit, and lispython leaves it to CPython.
NESTED_LITERAL_DEPTH = 200


@pytest.mark.usefixtures("time_bound")
@pytest.mark.parametrize("newline", ["", "\n"], ids=["one-line", "a-line-each"])
@pytest.mark.parametrize(
    "openings, needs_t",
    [case[1:] for case in NESTED_LITERALS],
    ids=[case[0] for case in NESTED_LITERALS],
)
def test_nested_f_strings_past_cpython_s_lexer_limit_compile(
    openings, needs_t, newline, scope, monkeypatch
):
    skip_without_t_strings(needs_t)
    literals = []
    string_parse = parser_module.string_parse

    def counting(*args):
        literals.append(None)
        return string_parse(*args)

    monkeypatch.setattr(parser_module, "string_parse", counting)
    [form] = parse(nested_literals(openings, NESTED_LITERAL_DEPTH, newline))
    assert len(literals) == NESTED_LITERAL_DEPTH
    assert literal_depth(form)[0] == NESTED_LITERAL_DEPTH
    [stmt] = macroexpand_then_compile([form], scope)
    code = compile(ast.Expression(stmt.value), "<nesting>", "eval")
    # Only a pure f-string nest is run: an f-string formats a t-string
    # through its repr, which escapes every inner level again, so a mixed
    # nest's text grows exponentially with its depth.
    if not needs_t:
        assert eval(code, {"x": "v"}) == "v"


@pytest.mark.parametrize(
    "openings, needs_t",
    [case[1:] for case in NESTED_LITERALS],
    ids=[case[0] for case in NESTED_LITERALS],
)
def test_nested_f_strings_parse_on_a_deep_stack(openings, needs_t):
    # A frame per literal would not fit in the room the caller leaves.
    skip_without_t_strings(needs_t)
    depth = NESTED_LITERAL_DEPTH
    src = nested_literals(openings, depth)

    def at_depth(frames):
        return at_depth(frames - 1) if frames else parse(src)

    [form] = at_depth(sys.getrecursionlimit() - 200)
    [found, leaf] = literal_depth(form)
    assert found == depth
    assert (type(leaf), atom_text(leaf), leaf.col_offset) == (Symbol, "x", 3 * depth)


@pytest.mark.usefixtures("time_bound")
def test_a_format_spec_chain_past_cpython_s_limit_is_refused_at_its_fourth_field():
    depth = 1000
    with pytest.raises(LispyCompileError) as info:
        parse('f"' + "{x:" * depth + "x" + "}" * depth + '"')
    assert info.value.msg == "f-string: expressions nested too deeply"
    # The fourth "{" is at column 11; CPython's offset is that column.
    assert (info.value.lineno, info.value.offset) == (1, 11)


def best_parse_time(src, repeats):
    """The fastest of ``repeats`` parses of ``src``, in seconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        parse(src)
        times.append(time.perf_counter() - start)
    return min(times)


# (id, the most nesting CPython takes, a flat source with as many literals,
# fields and tokens). CPython's own limits bound both: 149 nested literals,
# 3 open fields per literal, and 200 open brackets of any kind, fields included
# (so at most 66 literals of 3-field spec chains).
PAYLOAD = "[" + " 1," * 300 + "]"
NESTED_AND_FLAT = [
    (
        "spec-chains",
        'f"{x:{x:{' * 66 + "x" + '}}}"' * 66,
        " ".join(['f"{x:{x:{x}}}"'] * 66),
    ),
    (
        "payload-at-the-deepest-literal",
        'f"{' * 149 + PAYLOAD + '}"' * 149,
        'f"{' + PAYLOAD + '}"' + ' f"{x}"' * 148,
    ),
]


@pytest.mark.usefixtures("time_bound")
@pytest.mark.parametrize(
    "nested, flat",
    [case[1:] for case in NESTED_AND_FLAT],
    ids=[case[0] for case in NESTED_AND_FLAT],
)
def test_nested_literals_cost_what_the_same_literals_side_by_side_do(nested, flat):
    # Lexed in one pass, a character costs the same at any depth; re-lexing
    # each field's text made these 9x and 20x the flat cost.
    assert best_parse_time(nested, 5) < 3 * best_parse_time(flat, 5)


def peak_memory(run):
    """The most memory Python allocated while ``run()`` ran, in bytes."""
    tracemalloc.start()
    try:
        run()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.usefixtures("time_bound")
def test_tokens_hold_no_copy_of_a_field_s_text():
    # tokenize() reads whole LSP buffers: a field's token marks where its
    # expression is, so the 149 fields around a 200 KB string add no copies.
    src = 'f"{' * 149 + '"' + "a" * 200_000 + '"' + '}"' * 149
    assert peak_memory(lambda: tokenize(src)) < 5_000_000


@pytest.mark.usefixtures("time_bound")
def test_a_literal_with_many_fields_costs_what_as_many_literals_do():
    # Positions come from line starts found once; counting lines from the
    # literal's start for every piece made one literal quadratic.
    fields = 6000
    one = 'f"""' + "{x}\n" * fields + '"""'
    separate = "\n".join(['f"{x}"'] * fields)
    assert best_parse_time(one, 2) < best_parse_time(separate, 2)


@pytest.mark.usefixtures("time_bound")
def test_brackets_nested_past_the_recursion_limit_in_one_field_parse():
    [form] = parse('\nf"{' + "[" * DEPTH + "x" + "]" * DEPTH + '}"')
    node = form.list[2].value
    depth = 0
    while isinstance(node, Bracket):
        [node] = node.list
        depth += 1
    assert depth == DEPTH
    position = (node.lineno, node.col_offset, node.end_col_offset)
    assert (type(node), atom_text(node), position) == (
        Symbol,
        "x",
        (2, 3 + DEPTH, 4 + DEPTH),
    )


# (id, the prefixes repeated, the wrappers they make in order).
PREFIX_CHAINS = [
    ("annotation", "^", ["Annotation"]),
    ("keyword", ":", ["Keyword"]),
    ("star", "**", ["DoubleStarred"]),
    ("annotation-star", "^*", ["Annotation", "Starred"]),
    ("keyword-star-annotation", ":*^", ["Keyword", "Starred", "Annotation"]),
]


@pytest.mark.usefixtures("time_bound")
@pytest.mark.parametrize(
    "prefix, wrappers",
    [case[1:] for case in PREFIX_CHAINS],
    ids=[case[0] for case in PREFIX_CHAINS],
)
def test_a_prefix_chain_past_the_recursion_limit_parses(prefix, wrappers, scope):
    repeat = DEPTH // len(prefix)
    src = prefix * repeat + "x"
    [form] = parse(src)
    found = []
    node = form
    while isinstance(node, Wrapper):
        found.append((type(node).__name__, node.col_offset))
        node = node.value
    widths = [2 if name == "DoubleStarred" else 1 for name in wrappers] * repeat
    assert [name for [name, _] in found] == wrappers * repeat
    # Each wrapper starts where its prefix does; the leaf after the last.
    assert [col for [_, col] in found] == running_sums(widths)
    assert (type(node), atom_text(node), node.col_offset) == (Symbol, "x", len(src) - 1)
    assert str(form) == src
    try:
        macroexpand_then_compile([form], scope)
    except LispyCompileError:
        pass  # a refusal is fine; a RecursionError is not


# (id, prefixes repeated, the glued form after them, the wrappers they make):
# the prefixes close over the GluedHead the bracket makes.
GLUED_PREFIX_CHAINS = [
    ("annotation", "^", "a[int]", ["Annotation"]),
    ("keyword", ":", "a[0]", ["Keyword"]),
    ("star", "**", "a[0]", ["DoubleStarred"]),
]


@pytest.mark.usefixtures("time_bound")
@pytest.mark.parametrize(
    "prefix, glued, wrappers",
    [case[1:] for case in GLUED_PREFIX_CHAINS],
    ids=[case[0] for case in GLUED_PREFIX_CHAINS],
)
def test_a_prefix_chain_glued_to_a_bracket_past_the_recursion_limit_parses(
    prefix, glued, wrappers, monkeypatch
):
    # Closing the chain must set each wrapper's end once, not walk the
    # chain below every wrapper again (quadratic, yet within the budget).
    # The GluedHead closes first and sets only itself, so no wrapper here
    # needs to walk the chain at all.
    walked = []
    update_dict = Wrapper.update_dict

    def counting(self, key, value):
        walked.append(len(chain(self)))
        return update_dict(self, key, value)

    monkeypatch.setattr(Wrapper, "update_dict", counting)
    repeat = DEPTH // len(prefix)
    src = "\n" + prefix * repeat + glued
    [form] = parse(src)
    assert walked == []
    found = []
    node = form
    while isinstance(node, Wrapper):
        found.append(type(node).__name__)
        node = node.value
    assert found == wrappers * repeat
    # Every wrapper ends with the bracket; each starts at its prefix.
    ends = {(n.end_lineno, n.end_col_offset) for n in chain(form)}
    assert ends == {(2, DEPTH + len(glued))}
    assert [n.col_offset for n in chain(form)] == list(range(0, DEPTH, len(prefix)))
    assert type(node) is GluedHead
    assert (atom_text(node.head), node.head.col_offset) == ("a", DEPTH)
    assert str(form) == src[1:]


def chain(node):
    """The wrappers from ``node`` down, without recursion."""
    nodes = []
    while isinstance(node, Wrapper):
        nodes.append(node)
        node = node.value
    return nodes


def running_sums(widths):
    sums = []
    total = 0
    for width in widths:
        sums.append(total)
        total += width
    return sums


@pytest.mark.usefixtures("time_bound")
def test_quasiquoted_nested_fields_check_for_an_unquote_once_each():
    # Asking contains_unquote afresh at each field walks the rest of the
    # nest: 4000 levels took about 15 s.
    depth = 4000
    quoted = nest(lambda e: fstring(field(e)), depth, Unquote(Symbol("x")))
    rebuilt = quoted.generator_expression(True)
    # Every field rebuilds its text from the unquote's expansion.
    padded = 0
    pending = [rebuilt]
    while pending:
        node = pending.pop()
        if isinstance(node, list):
            pending.extend(node)
        elif isinstance(node, Node):
            padded += atom_text(node) == ":expr-padding"
            pending.extend(vars(node).values())
    assert padded == depth


def shared_and_cyclic_forms():
    """(id, form, the node to ask about first, then the form itself)."""
    leaf = Unquote(Symbol("x"))
    dag = nest(lambda e: Paren(e, e), 60, leaf)
    plain_dag = nest(lambda e: Paren(e, e), 60, Symbol("x"))
    # The unquote hangs off the cycle's entry: asked from inside the cycle
    # first, a walk that cut the back edge would answer False.
    inner = Paren(Symbol("a"))
    entry = Paren(inner, leaf)
    inner.list.append(entry)
    plain_inner = Paren(Symbol("a"))
    plain_entry = Paren(plain_inner, Symbol("b"))
    plain_inner.list.append(plain_entry)
    return [
        ("dag", dag, dag.list[0]),
        ("plain-dag", plain_dag, plain_dag.list[0]),
        ("cycle", entry, inner),
        ("plain-cycle", plain_entry, plain_inner),
    ]


@pytest.mark.usefixtures("time_bound")
@pytest.mark.parametrize(
    "form, first",
    [case[1:] for case in shared_and_cyclic_forms()],
    ids=[case[0] for case in shared_and_cyclic_forms()],
)
def test_the_memoized_unquote_check_answers_as_contains_unquote(form, first):
    memo = nodes._UNQUOTE_MEMO.set({})
    try:
        answers = [nodes._contains_unquote_memoized(n) for n in [first, form]]
    finally:
        nodes._UNQUOTE_MEMO.reset(memo)
    assert answers == [contains_unquote(first), contains_unquote(form)]


def test_nested_do_collects_each_statement_once(monkeypatch):
    # Inlining each level's statement list into the level above copies the
    # rest of the nest at every level: about DEPTH**2 / 2 statements.
    collected = []
    steps = stmt_module.stmt_list_steps

    def counting(sexp_list):
        stmts = yield steps(sexp_list)
        collected.append(len(stmts))
        return stmts

    monkeypatch.setattr(stmt_module, "stmt_list_steps", counting)
    src = "".join(f"(do {k} " for k in range(DEPTH)) + "(do) end (do 1 (do 2 3) 4)"
    stmts = stmt_module.stmt_list_compile(parse(src + ")" * DEPTH))
    values = [s.value.value if k != DEPTH else s.value.id for k, s in enumerate(stmts)]
    assert values == [*range(DEPTH), "end", 1, 2, 3, 4]
    assert sum(collected) == len(stmts)


def test_a_do_compiled_on_its_own_is_its_statements():
    [form] = parse("(do 1 (do 2) 3)")
    stmts = stmt_module.stmt_compile(form)
    assert [s.value.value for s in stmts] == [1, 2, 3]
