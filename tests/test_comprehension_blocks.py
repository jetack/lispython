"""List/set/dict comprehensions nested past CPython's static block limit, and
an await at that limit in a generator expression, are a positioned
LispyCompileError, never a crashed interpreter (gh-156091).

Every CPython 3.12+ can segfault in compile() on such an AST, so anything
that may compile() one runs in a subprocess, never in the pytest process.
"""

import ast
import json
import subprocess
import sys

import pytest

from lispy.core.compiler import features
from lispy.core.compiler.utils import MAX_STATIC_BLOCKS, refuse_block_overflow
from lispy.core.errors import BLOCK_OVERFLOW_MESSAGE, LispyCompileError
from lispy.core.macro import macroexpand_then_compile
from lispy.core.parser import parse

RUNNING = tuple(sys.version_info)[:3]
# 3.11 compiles each comprehension as a code unit of its own: no limit.
INLINES = RUNNING[:2] >= (3, 12)
# The first patch release of each line that counts an inlined comprehension
# as a static block (gh-156091); 3.12 is security-only and never got it.
COMPREHENSION_BLOCK_FIX = {(3, 13): 16, (3, 14): 8, (3, 15): 1}


def counts_inlined_comprehensions(version):
    """Whether CPython ``version`` refuses an inlined comprehension past the
    static block limit itself rather than crashing."""
    fix = COMPREHENSION_BLOCK_FIX.get(tuple(version[:2]))
    return version[:2] > (3, 15) or (fix is not None and version[2] >= fix)


FIXED = INLINES and counts_inlined_comprehensions(RUNNING)
needs_inlining = pytest.mark.skipif(
    not INLINES, reason="3.11 does not inline comprehensions"
)
needs_fixed = pytest.mark.skipif(
    not FIXED,
    reason="needs a CPython that refuses the overflow itself (3.13.16+, 3.14.8+)",
)
needs_unfixed = pytest.mark.skipif(
    not INLINES or FIXED,
    reason="this CPython does not crash on a plain comprehension overflow",
)


@pytest.mark.parametrize(
    "version, checked",
    [
        ((3, 11, 13), False),
        ((3, 12, 11), True),
        ((3, 13, 15), True),
        # The fixed releases still crash on an await in a genexp at the limit.
        ((3, 13, 16), True),
        ((3, 14, 7), True),
        ((3, 14, 8), True),
        ((3, 15, 0), True),
        ((3, 15, 1), True),
        ((3, 16, 0), True),
    ],
)
def test_every_inlining_release_gets_the_check(version, checked, monkeypatch):
    monkeypatch.setattr(features, "running_version", lambda: version)
    assert features.inlined_comprehensions_overflow_block_stack() is checked


@pytest.mark.parametrize(
    "version, own_scope",
    [
        ((3, 12, 11), False),
        ((3, 13, 16), False),
        ((3, 14, 0), True),
        ((3, 15, 1), True),
    ],
)
def test_annotations_have_their_own_scope_from_3_14(version, own_scope, monkeypatch):
    monkeypatch.setattr(features, "running_version", lambda: version)
    assert features.annotations_have_own_scope() is own_scope


# Compiles a lispython source through the import pipeline (``_lpy_code``) and
# prints the outcome as JSON. argv: source, then "off" to switch the check off.
COMPILE_LPY = """
import json, sys
from lispy.core.compiler import features
from lispy.core.errors import LispyCompileError
from lispy.core.importer import _lpy_code

if sys.argv[2:] == ["off"]:
    features.inlined_comprehensions_overflow_block_stack = lambda: False
try:
    _lpy_code(sys.argv[1], "<case>", lambda tree: compile(tree, "<case>", "exec"))
except LispyCompileError as err:
    print(json.dumps([err.msg, err.lineno, err.offset]))
else:
    print(json.dumps("compiled"))
"""


def compile_lpy(src, *extra):
    """The outcome of compiling ``src`` in a fresh interpreter; a negative
    return code (a signal) is a crash."""
    result = subprocess.run(
        [sys.executable, "-c", COMPILE_LPY, src, *extra],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        return ("exit", result.returncode, result.stderr[-2000:])
    return json.loads(result.stdout)


# Each comprehension kind as (opener, closer): ``n`` of them nest in the
# element (a set's element, a dict's value or key) as opener * n + leaf +
# closer * n, so the k-th opener is at column (k - 1) * len(opener).
KINDS = {
    "list": ("[", "x", " for x in xs]"),
    "set": ("{, ", "x", " for x in xs}"),
    "dict-value": ("{k ", "v", " for x in xs}"),
    "dict-key": ("{", "k v for x in xs}", " v for x in xs}"),
    # Two blocks a level: the comprehension and its async for.
    "async": ("[", "x", " async-for x in xs]"),
}


def comprehensions(kind, n):
    [opener, leaf, closer] = KINDS[kind]
    if kind == "dict-key":
        return opener * n + leaf + closer * (n - 1)
    return opener * n + leaf + closer * n


# (id, kind, lispython source with "{}" where the comprehensions go, alone on
# their line, and the static blocks the comprehensions sit under there).
# 20 blocks: 3.11 refuses a 21st statement block (its CO_MAXBLOCKS is 20).
WITH_20_ITEMS = "(with [" + " ".join(["a"] * 20) + "]\n"
CONTEXTS = [
    ("module-list", "list", "{}", 0),
    ("module-set", "set", "{}", 0),
    ("module-dict-value", "dict-value", "{}", 0),
    ("module-dict-key", "dict-key", "{}", 0),
    ("def", "list", "(def f []\n{})", 0),
    ("with", "list", "(with [a]\n{})", 1),
    ("with-two-items", "set", "(with [a b]\n{})", 2),
    ("for", "list", "(for y in ys\n{})", 1),
    ("while", "dict-value", "(while c\n{})", 1),
    ("while-test", "list", "(while\n{}\n(f))", 1),
    ("try-body", "list", "(try\n{}\n(except [E] 1))", 1),
    ("try-body-with-finally", "list", "(try\n{}\n(except [E] 1)\n(finally 1))", 2),
    ("except-body", "list", "(try 1\n(except [E]\n{}))", 2),
    ("except-as-body", "set", "(try 1\n(except [E as e]\n{}))", 2),
    ("finally-body", "dict-key", "(try 1\n(finally\n{}))", 1),
    ("async-def", "list", "(async-def f []\n{})", 1),
    ("generator-def", "list", "(def g []\n(yield 1)\n{})", 1),
    ("def-under-a-20-item-with", "list", WITH_20_ITEMS + "(def f []\n{}))", 0),
    ("class-under-a-20-item-with", "list", WITH_20_ITEMS + "(class C []\n{}))", 0),
    # Under 21 comprehensions on line 1, a code unit restarts the count.
    (
        "lambda-under-21",
        "list",
        "[" * 21 + "(lambda []\n{})" + " for x in xs]" * 21,
        0,
    ),
    (
        "genexp-under-21",
        "list",
        "[" * 21 + "(\n{} for y in ys)" + " for x in xs]" * 21,
        0,
    ),
]


def placed(context, n):
    """``context``'s source with ``n`` comprehensions placed; the line and
    column of the placeholder."""
    [_, kind, wrapper, _] = context
    before = wrapper[: wrapper.index("{}")]
    line = before.count("\n") + 1
    column = len(before) - (before.rfind("\n") + 1)
    return wrapper.replace("{}", comprehensions(kind, n)), line, column


def blocks_per_level(kind):
    return 2 if kind == "async" else 1


def at_bound(context):
    """The most comprehensions ``context`` takes."""
    [_, kind, _, blocks] = context
    return (MAX_STATIC_BLOCKS - blocks) // blocks_per_level(kind)


@pytest.mark.parametrize("context", CONTEXTS, ids=[case[0] for case in CONTEXTS])
def test_comprehensions_at_the_bound_compile(context):
    [src, _, _] = placed(context, at_bound(context))
    assert compile_lpy(src) == "compiled"


@pytest.mark.parametrize("context", CONTEXTS, ids=[case[0] for case in CONTEXTS])
def test_one_comprehension_past_the_bound_is_a_positioned_error(context):
    # Refused by lispython where CPython would crash, by CPython where it
    # counts the comprehensions itself: the same error either way.
    n = at_bound(context) + 1
    [src, line, column] = placed(context, n)
    outcome = compile_lpy(src)
    if not INLINES:
        assert outcome == "compiled"
        return
    opener = KINDS[context[1]][0]
    assert outcome == [BLOCK_OVERFLOW_MESSAGE, line, column + (n - 1) * len(opener) + 1]


ASYNC_CONTEXT = ("async-comprehension", "async", "(async-def f []\n{})", 1)


@needs_inlining
def test_async_comprehensions_count_their_async_for():
    # 1 + 2 * 10 blocks fit; the 11th comprehension pushes to 22.
    n = at_bound(ASYNC_CONTEXT)
    assert n == 10
    assert compile_lpy(placed(ASYNC_CONTEXT, n)[0]) == "compiled"
    [src, line, column] = placed(ASYNC_CONTEXT, n + 1)
    assert compile_lpy(src) == [BLOCK_OVERFLOW_MESSAGE, line, column + n + 1]


# A generator expression's body sits under a StopIteration handler CPython
# never counts, and an await (or an async for clause's anext) pushes one more:
# each takes a slot past the blocks it sits under. Every 3.12+ release,
# fixed or not, crashes one comprehension past these bounds.
# (id, the leaf the comprehensions nest around, the most comprehensions taken)
GENEXP_AWAITS = [
    # 20 blocks, then the await's slot is the 21st.
    ("await", "(await z)", 20),
    # 18 blocks, the leaf comprehension's, its async for's, then its anext's.
    ("inner-async-comprehension", "[q async-for q in qs]", 18),
]


def genexp_await(leaf, n):
    """An async def's generator expression whose element nests ``n`` list
    comprehensions around ``leaf``, on line 2 with the leaf at column
    ``n + 1``."""
    return (
        "(async-def f []\n(" + "[" * n + leaf + " for x in xs]" * n + " for y in ys))"
    )


@pytest.mark.parametrize(
    "leaf, bound", [c[1:] for c in GENEXP_AWAITS], ids=[c[0] for c in GENEXP_AWAITS]
)
def test_awaits_in_a_generator_expression_at_the_bound_compile(leaf, bound):
    assert compile_lpy(genexp_await(leaf, bound)) == "compiled"


@pytest.mark.parametrize(
    "leaf, bound", [c[1:] for c in GENEXP_AWAITS], ids=[c[0] for c in GENEXP_AWAITS]
)
def test_an_await_past_the_bound_in_a_generator_expression_is_refused(leaf, bound):
    outcome = compile_lpy(genexp_await(leaf, bound + 1))
    if not INLINES:
        assert outcome == "compiled"
        return
    # At the await, or the comprehension holding the async for.
    assert outcome == [BLOCK_OVERFLOW_MESSAGE, 2, bound + 3]


@needs_inlining
def test_without_the_check_an_await_in_a_generator_expression_crashes():
    # Why the check runs on every 3.12+ release, fixed ones included.
    [_, leaf, bound] = GENEXP_AWAITS[0]
    outcome = compile_lpy(genexp_await(leaf, bound + 1), "off")
    assert outcome[0] == "exit" and outcome[1] < 0, outcome


# Comprehensions in the first iterable are evaluated before the comprehension
# pushes its block: no depth of them counts.
FIRST_ITERABLES = [
    ("module", "[x for x in " * 40 + "xs" + "]" * 40),
    ("set", "{, x for x in " * 40 + "xs" + "}" * 40),
    (
        # 1 + 19 blocks: each comprehension in the chain takes the 21st.
        "under-19-comprehensions-in-a-with",
        "(with [a]\n"
        + "[" * 19
        + "[x for x in " * 40
        + "xs"
        + "]" * 40
        + " for x in xs]" * 19
        + ")",
    ),
]


@pytest.mark.parametrize(
    "src", [case[1] for case in FIRST_ITERABLES], ids=[c[0] for c in FIRST_ITERABLES]
)
def test_first_iterable_nesting_is_never_refused(src):
    assert compile_lpy(src) == "compiled"


@needs_unfixed
def test_without_the_check_the_interpreter_crashes():
    # Why the check exists: this CPython segfaults on the source one
    # comprehension past the one it refuses (the 23rd handler overflows).
    [src, _, _] = placed(CONTEXTS[0], MAX_STATIC_BLOCKS + 2)
    outcome = compile_lpy(src, "off")
    assert outcome[0] == "exit" and outcome[1] < 0, outcome


def force_check(monkeypatch):
    """Apply the check whatever the interpreter."""
    monkeypatch.setattr(
        features, "inlined_comprehensions_overflow_block_stack", lambda: True
    )


@pytest.mark.usefixtures("time_bound")
def test_the_check_takes_a_comprehension_nest_past_the_recursion_limit(monkeypatch):
    force_check(monkeypatch)
    # 3.11's recursion-limit depth check would refuse these first.
    monkeypatch.setattr(features, "compile_depth_follows_recursion_limit", bool)
    depth = 5000
    with pytest.raises(LispyCompileError) as info:
        macroexpand_then_compile(parse(comprehensions("list", depth)), {})
    assert (info.value.msg, info.value.lineno, info.value.offset) == (
        BLOCK_OVERFLOW_MESSAGE,
        1,
        MAX_STATIC_BLOCKS + 1,
    )
    # Accepted all the way down: every level in a first iterable, and a
    # generator def's yield found under thousands of levels.
    deep_iterable = "[x for x in " * depth + "xs" + "]" * depth
    body = macroexpand_then_compile(parse(deep_iterable), {})
    assert refuse_block_overflow(body) is body
    deep_yield = "(def g [] " + "(f " * depth + "(yield 1)" + ")" * depth + ")"
    [generator] = macroexpand_then_compile(parse(deep_yield), {})
    assert isinstance(generator, ast.FunctionDef)


# Python shapes for the model: (id, source with {C} for n comprehensions);
# {A} is the async form. Each covers a rule in _counted_children. None may
# crash fixed CPython in SHAPE_SIZES (an await in a genexp would: those are
# GENEXP_AWAITS), or the ground truth below has no answer for it.
PY_SHAPES = [
    ("module", "{C}"),
    ("def", "def f():\n    {C}"),
    ("async-def", "async def f():\n    {C}"),
    ("generator-def", "def f():\n    yield\n    {C}"),
    ("yield-in-nested-default", "def f():\n    def g(a=(yield)): pass\n    {C}"),
    ("yield-in-nested-lambda", "def f():\n    lambda: (yield)\n    {C}"),
    ("yield-in-nested-def", "def f():\n    def g():\n        yield\n    {C}"),
    (
        "yield-in-nested-class-decorator",
        "def f():\n    @(yield)\n    class C: pass\n    {C}",
    ),
    ("generator-lambda", "lambda: ((yield), {C})"),
    ("class", "class C:\n    {C}"),
    ("with", "with a:\n    {C}"),
    ("with-two-items", "with a, b:\n    {C}"),
    ("with-second-item-expr", "with a, {C}:\n    pass"),
    ("with-target", "with a as b[{C}]:\n    pass"),
    ("for", "for y in ys:\n    {C}"),
    ("for-else", "for y in ys:\n    pass\nelse:\n    {C}"),
    ("for-iter", "for y in {C}:\n    pass"),
    ("for-target", "for y[{C}] in ys:\n    pass"),
    ("while", "while c:\n    {C}"),
    ("while-test", "while {C}:\n    pass"),
    ("async-for-iter", "async def f():\n    async for y in {C}:\n        pass"),
    ("async-for-target", "async def f():\n    async for y[{C}] in ys:\n        pass"),
    ("try-body", "try:\n    {C}\nexcept E:\n    pass"),
    ("try-finally-body", "try:\n    {C}\nfinally:\n    pass"),
    (
        "try-except-finally-body",
        "try:\n    {C}\nexcept E:\n    pass\nfinally:\n    pass",
    ),
    ("except-body", "try:\n    pass\nexcept E:\n    {C}"),
    (
        "except-as-finally",
        "try:\n    pass\nexcept E as e:\n    {C}\nfinally:\n    pass",
    ),
    ("except-type", "try:\n    pass\nexcept {C}:\n    pass"),
    ("except-star-body", "try:\n    pass\nexcept* E:\n    {C}"),
    ("try-else", "try:\n    pass\nexcept E:\n    pass\nelse:\n    {C}"),
    (
        "try-else-finally",
        "try:\n    pass\nexcept E:\n    pass\nelse:\n    {C}\nfinally:\n    pass",
    ),
    ("finally-body", "try:\n    pass\nfinally:\n    {C}"),
    ("async-comprehension", "async def f():\n    {A}"),
    ("async-genexp", "async def f():\n    ({C} async for y in ys)"),
    ("genexp-under-comprehensions", "[[({C} for y in ys) for x in xs] for x in xs]"),
    ("decorator-under-with", "with a:\n    @{C}\n    def f(): pass"),
    (
        "match-guard-under-for",
        "for y in ys:\n    match y:\n        case 1 if {C}:\n            pass",
    ),
    ("type-alias-in-with", "with a:\n    type T = {C}"),
    ("default-under-with", "with a:\n    def f(a={C}): pass"),
    ("kw-default-under-with", "with a:\n    def f(*, a={C}): pass"),
    ("class-base-under-with", "with a:\n    class C({C}): pass"),
    # Type parameters are a scope of their own.
    ("type-var-bound-under-with", "with a:\n    def f[T: {C}](): pass"),
    ("class-type-var-bound-under-with", "with a:\n    class C[T: {C}]: pass"),
    ("type-alias-param-under-with", "with a:\n    type T[U: {C}] = int"),
    ("type-var-default-under-with", "with a:\n    def f[T = {C}](): pass", (3, 13)),
    # Annotations: the enclosing unit's on 3.12/3.13, their own from 3.14.
    ("annotation-under-with", "with a:\n    def f(a: {C}): pass"),
    ("vararg-annotation-under-with", "with a:\n    def f(*a: {C}): pass"),
    ("return-annotation-under-with", "with a:\n    def f() -> {C}: pass"),
]


def shape_params():
    """``PY_SHAPES`` as pytest params; a shape with a third item needs that
    Python to parse."""
    return [
        pytest.param(
            shape[1],
            id=shape[0],
            marks=pytest.mark.skipif(
                RUNNING[:2] < shape[2], reason="syntax this Python lacks"
            )
            if len(shape) > 2
            else (),
        )
        for shape in PY_SHAPES
    ]


def python_shape(shape, n):
    nest = "[" * n + "x" + " for x in xs]" * n
    nest_async = "[" * n + "x" + " async for x in xs]" * n
    return shape.replace("{C}", nest).replace("{A}", nest_async)


# Compiles each Python source in argv[1] (a JSON list), printing per source
# whether compile() refused it for too many blocks; any other error is shown.
COMPILE_PY = """
import json, sys
for src in json.loads(sys.argv[1]):
    try:
        compile(src, "<shape>", "exec")
    except SyntaxError as err:
        print(json.dumps("refused" if err.msg == sys.argv[2] else err.msg))
    else:
        print(json.dumps("compiled"))
    sys.stdout.flush()
"""

SHAPE_SIZES = range(8, 24)


def model_refuses(src, monkeypatch):
    force_check(monkeypatch)
    try:
        refuse_block_overflow(ast.parse(src).body)
    except LispyCompileError as err:
        assert err.msg == BLOCK_OVERFLOW_MESSAGE
        return "refused"
    return "compiled"


@needs_fixed
@pytest.mark.parametrize("shape", shape_params())
def test_the_model_refuses_what_fixed_cpython_refuses(shape, monkeypatch):
    sources = [python_shape(shape, n) for n in SHAPE_SIZES]
    result = subprocess.run(
        [sys.executable, "-c", COMPILE_PY, json.dumps(sources), BLOCK_OVERFLOW_MESSAGE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    cpython = [json.loads(line) for line in result.stdout.splitlines()]
    ours = [model_refuses(src, monkeypatch) for src in sources]
    assert dict(zip(SHAPE_SIZES, ours)) == dict(zip(SHAPE_SIZES, cpython))
    # The sizes straddle the bound.
    assert {"compiled", "refused"} <= set(cpython)


@needs_unfixed
@pytest.mark.parametrize("shape", shape_params())
def test_what_the_model_accepts_compiles_on_a_crashing_cpython(shape, monkeypatch):
    # The most comprehensions the model accepts in each shape: the handler
    # stack holds them where CPython does not count them.
    accepted = [
        n
        for n in SHAPE_SIZES
        if model_refuses(python_shape(shape, n), monkeypatch) == "compiled"
    ]
    assert accepted and max(accepted) < SHAPE_SIZES[-1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            COMPILE_PY,
            json.dumps([python_shape(shape, max(accepted))]),
            BLOCK_OVERFLOW_MESSAGE,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ['"compiled"']
