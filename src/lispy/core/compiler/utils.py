import ast
import contextlib
import sys
from collections import deque

from lispy.core.compiler import features
from lispy.core.errors import LispyCompileError, nesting_error
from lispy.core.nodes import *
from lispy.core.utils import augassignop_dict, boolop_dict, compare_dict


def paren_p(sexp):
    return isinstance(sexp, Paren)


def bracket_p(sexp):
    return isinstance(sexp, Bracket)


def brace_p(sexp):
    return isinstance(sexp, Brace)


def starred_p(sexp):
    return isinstance(sexp, Starred)


def doublestarred_p(sexp):
    return isinstance(sexp, DoubleStarred)


def constant_p(sexp):
    return isinstance(sexp, Constant)


def string_p(sexp):
    return isinstance(sexp, String)


def keyword_arg_p(sexp):
    return isinstance(sexp, Keyword)


def merge_position_infos(*position_infos):
    return {
        "lineno": min(map(lambda x: x["lineno"], position_infos)),
        "col_offset": min(map(lambda x: x["col_offset"], position_infos)),
        "end_lineno": max(map(lambda x: x["end_lineno"], position_infos)),
        "end_col_offset": max(map(lambda x: x["end_col_offset"], position_infos)),
    }


# Heads that only compile as statements (what stmt_compile dispatches).
# `match` and `type` are deliberately absent: they are Python soft keywords
# and stay callable in expressions.  A pin test keeps this in sync with
# stmt_compile.
STATEMENT_HEADS = frozenset(
    {
        "do", "=", "del", "pass", "import", "from", "lazy-import", "lazy-from",
        "if", "while", "for", "break", "continue", "raise", "assert", "try",
        "with", "deco", "def", "return", "global", "nonlocal", "class",
        "async-def", "async-for", "async-with",
    }
    | augassignop_dict.keys()
)

# Where a statement head has an expression-position equivalent, say so.
STATEMENT_HINTS = {
    "=": "use (:= name value)",
    "if": "use (ife test then else)",
    "do": (
        "statement blocks, including what defmacro, require and statement macros "
        "such as when expand to, cannot be values; "
        "to sequence expressions use (sub (, a b c) -1)"
    ),
}


# How every statement and expression form, clause and pattern is written: the
# one home of their arity, quoted by form_error in each refusal and shown as the
# LSP hover usage line. Literal [ ] are lispython brackets.
FORM_USAGE = {
    "if": "(if test then else...)",
    "while": "(while test body...)",
    "for": "(for target in iterable body...)",
    "async-for": "(async-for target in iterable body...)",
    "def": "(def name [args] body...)",
    "async-def": "(async-def name [args] body...)",
    "class": "(class name [bases] body...)",
    "with": "(with [context as target ...] body...), each as target optional",
    "async-with": (
        "(async-with [context as target ...] body...), each as target optional"
    ),
    "try": "(try body... (except ...)... (else ...) (finally ...))",
    "except": (
        "(except [type ... as name] body...) or (except body...), as name optional"
    ),
    "except*": "(except* [type ... as name] body...), as name optional",
    "else": "(else body...)",
    "finally": "(finally body...)",
    "match": "(match subject (case ...)...)",
    "case": "(case pattern as name if guard body...), as name and if guard optional",
    "|": "(| pattern pattern...)",
    "pattern": (
        "a name, _, a literal, a dotted name, [sequence], {mapping}, "
        "(| ...) or (Class ...)"
    ),
    "sequence pattern": "[pattern... *rest pattern...], at most one *rest",
    "mapping pattern": (
        "{key pattern... **rest}, each key a literal or dotted name, **rest last"
    ),
    "class pattern": "(Class pattern... :attr pattern...), positional patterns first",
    "deco": "(deco decorator definition) or (deco [decorator ...] definition)",
    "raise": "(raise), (raise exception) or (raise exception from cause)",
    "assert": "(assert test) or (assert test message)",
    "=": (
        "(= target... value) or (= target ^annotation value), "
        "value optional after an annotation"
    ),
    **{op: f"({op} target value...)" for op in augassignop_dict},
    "del": "(del target...)",
    "global": "(global name...)",
    "nonlocal": "(nonlocal name...)",
    ":=": "(:= name value)",
    ".": "(. value attr...)",
    "method call": "(.method instance args...)",
    "ife": "(ife test then else), else optional",
    "sub": "(sub value index...)",
    "lambda": "(lambda [args] body)",
    "await": "(await value)",
    "yield": "(yield) or (yield value)",
    "yield-from": "(yield-from iterable)",
    **{op: f"({op} value...)" for op in boolop_dict},
    **{op: f"({op} a b...)" for op in compare_dict},
    "slice": (
        "[: stop], [: start stop] or [: start stop step], "
        "_ or None for a default"
    ),
    "dict": "{key value ... **mapping}",
    "comprehension": (
        "element for target in iterable if test..., "
        "then more for or async-for clauses, each in optional"
    ),
}

# FORM_USAGE entries that are no head word, named as they are in a message
# ("class pattern has ..."), not as a form ("(| ...) needs ..."). The LSP
# gives them no hover entry.
HEADLESS_FORMS = (
    "pattern",
    "sequence pattern",
    "mapping pattern",
    "class pattern",
    "method call",
    "slice",
    "dict",
    "comprehension",
)


def form_error(sexp, problem, form=None):
    """The one shape of a malformed-form error: the problem, then usage.

    Positioned at ``sexp``; ``form`` (default: ``sexp``'s own head) names the
    form or headless kind whose usage is quoted, so an error at a clause or
    operand can still quote the enclosing form.
    """
    form = str(sexp.op) if form is None else form
    subject = form if form in HEADLESS_FORMS else f"({form} ...)"
    return LispyCompileError(f"{subject} {problem}: {FORM_USAGE[form]}", sexp)


def form_parts(sexp, header, form=None):
    """``[parts, rest]``: the header operands of ``sexp`` in order, then the rest.

    ``header`` names the required operands; an entry ``(word, name)`` lets an
    optional ``word`` precede that operand (for's ``in``). A missing one is
    "is missing its {name}", at the form.
    """
    q = deque(sexp.operands)
    parts = []
    for entry in header:
        [word, name] = entry if isinstance(entry, tuple) else [None, entry]
        if word is not None and len(q) and q[0] == word:
            q.popleft()
        if not q:
            raise form_error(sexp, f"is missing its {name}", form)
        parts.append(q.popleft())
    return [parts, list(q)]


# Frames from refuse_deep_ast up to the earliest frame that can compile() its
# result: macroexpand_then_compile, then that function's caller.
CHECK_TO_COMPILE_FRAMES = 2


# Leaf singletons compile() does not recurse into: they cost it no depth.
_DEPTH_FREE_NODES = (ast.expr_context, ast.operator, ast.cmpop, ast.unaryop, ast.boolop)


def _stack_depth():
    """Frames on the stack, the caller's included (a module's code is 1)."""
    depth = 0
    frame = sys._getframe(1)
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


def _ast_depths(stmts):
    """Yield ``(depth, positioned)`` for each node of each statement in
    ``stmts``, depth first and left to right (so the first path to reach a
    depth is the leftmost one). ``depth`` counts from the statement as 1,
    leaving out ``_DEPTH_FREE_NODES``; ``positioned`` is the node itself, or
    its nearest ancestor with a position when it has none (an ``arguments``,
    a ``comprehension``). Iterative and linear in the AST's size: the ASTs it
    walks are the ones too deep for recursion."""
    for stmt in stmts:
        # (node, depth, nearest positioned node on the path, itself included)
        pending = [(stmt, 1, stmt)]
        while pending:
            [node, depth, positioned] = pending.pop()
            if getattr(node, "lineno", None):
                positioned = node
            yield depth, positioned
            # Reversed, so the first child is popped (walked) first.
            pending.extend(
                (child, depth + 1, positioned)
                for child in reversed(list(ast.iter_child_nodes(node)))
                if not isinstance(child, _DEPTH_FREE_NODES)
            )


def refuse_deep_ast(stmts):
    """Raise ``nesting_error`` if a statement in ``stmts`` is nested deeper
    than CPython's compile() takes, where that depth follows the recursion
    limit (``features.compile_depth_follows_recursion_limit``); return
    ``stmts``.

    There (3.11) compile() called from a frame at stack depth C accepts an
    AST whose deepest path, counted from the statement as 1, has at most
    ``limit - C - 1`` nodes (measured: 996 nodes from depth 3 at limit 1000,
    2494 from depth 505 at limit 3000), and raises a RecursionError naming
    no line beyond. The earliest frame that can compile the result is
    CHECK_TO_COMPILE_FRAMES above this one, and only ASTs that frame's
    compile() would reject are refused, never one it accepts; trivial code is
    refused only where compile() would fail on it too. The count errs only
    the permissive way: C-level calls on the stack spend recursion depth too
    but have no frame, and an entry point may compile one frame deeper (a
    comprehension), so a few levels past it can still reach compile(), whose
    RecursionError ``compile_nesting_errors`` turns into the same error.

    The error is at the node at the refused depth on the first (leftmost)
    too-deep path, or its nearest positioned ancestor (``_ast_depths``).
    """
    if not features.compile_depth_follows_recursion_limit():
        return stmts
    compile_frame = _stack_depth() - CHECK_TO_COMPILE_FRAMES
    refused = sys.getrecursionlimit() - compile_frame
    for depth, positioned in _ast_depths(stmts):
        if depth >= refused:
            raise nesting_error(positioned)
    return stmts


@contextlib.contextmanager
def compile_nesting_errors(stmts):
    """Run the block, which compile()s the statements ``stmts`` lispython
    built, turning a RecursionError into ``nesting_error`` at the bottom of
    the first (leftmost) deepest path in ``stmts``, or its nearest positioned
    ancestor (``_ast_depths``).

    CPython's compile() recurses over the AST and gives up past its own
    limit with a RecursionError that names no line: on 3.12+ a fixed C limit
    or the C stack (``features.compile_depth_follows_recursion_limit`` has
    the measurements), on 3.11 the few levels ``refuse_deep_ast`` lets
    through. So it applies on every version. Wrap compile() alone, never
    macroexpansion: a RecursionError while a macro runs is that macro's own
    runaway recursion, the user's bug, and must keep its traceback into the
    macro. compile() runs no user code, so a RecursionError from it is
    always nesting; ``compile_errors`` must not catch RecursionError for the
    same reason.
    """
    try:
        yield
    except RecursionError:
        deepest = None
        for depth, positioned in _ast_depths(stmts):
            if deepest is None or depth > deepest[0]:
                deepest = (depth, positioned)
        if deepest is None:
            raise
        raise nesting_error(deepest[1]) from None
