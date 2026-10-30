"""Deeply nested forms: compile cost linear in the nesting, and nesting past
what Python can compile is a positioned LispyCompileError, never a
RecursionError."""

import ast
import subprocess
import sys

import pytest

import lispy.tools as tools
from lispy.core.compiler import features
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
)
from lispy.core.parser import parse

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


@pytest.mark.parametrize(
    "src",
    [
        "[" * DEEP + "1" + "]" * DEEP,
        "(f " * DEEP + "1" + ")" * DEEP,
        "(" * DEEP + "f" + ")" * DEEP,
        "(when c " + "[" * DEEP + "1" + "]" * DEEP + ")",
        "(do " * DEEP + "(f)" + ")" * DEEP,
        "'" + "[" * DEEP + "1" + "]" * DEEP,
    ],
    ids=["bracket", "paren-arg", "paren-op", "macro-arg", "do", "quote"],
)
def test_compiling_past_the_recursion_limit_is_a_positioned_error(src, scope):
    with pytest.raises(LispyCompileError) as info:
        compile_src("\n\n" + src, scope)
    err = info.value
    assert err.msg == NESTING_MESSAGE
    assert str(sys.getrecursionlimit()) in err.msg
    assert err.lineno == 3 and err.offset >= 1


def run_cli(argv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as info:
        tools.run()
    out, err = capsys.readouterr()
    return info.value.code, out, err


def test_lpy_reports_a_too_deep_form_compiler_style(monkeypatch, capsys):
    src = "\n" + "[" * DEEP + "1" + "]" * DEEP
    [code, out, err] = run_cli(["lpy", "-c", src], monkeypatch, capsys)
    assert code == 1
    assert err.startswith("<string>:2:")
    assert f": error: {NESTING_MESSAGE}\n" in err
    assert "Traceback" not in out + err


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
