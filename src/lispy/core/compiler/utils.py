import ast
import sys
from collections import deque

from lispy.core.compiler import features
from lispy.core.errors import nesting_error
from lispy.core.nodes import *
from lispy.core.utils import augassignop_dict


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


# Frames from refuse_deep_ast up to the earliest frame that can compile() its
# result: macroexpand_then_compile, then that function's caller.
CHECK_TO_COMPILE_FRAMES = 2


def _stack_depth():
    """Frames on the stack, the caller's included (a module's code is 1)."""
    depth = 0
    frame = sys._getframe(1)
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


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
    comprehension), so a few levels can still meet the raw RecursionError.

    One iterative walk per statement, linear in the AST's size; the error is
    at the node at the refused depth on the first (leftmost) too-deep path,
    or at its nearest positioned ancestor when that node has no position (an
    operator, a context).
    """
    if not features.compile_depth_follows_recursion_limit():
        return stmts
    compile_frame = _stack_depth() - CHECK_TO_COMPILE_FRAMES
    refused = sys.getrecursionlimit() - compile_frame
    for stmt in stmts:
        # (node, depth, nearest positioned node on the path, itself included)
        pending = [(stmt, 1, stmt)]
        while pending:
            [node, depth, positioned] = pending.pop()
            if getattr(node, "lineno", None):
                positioned = node
            if depth >= refused:
                raise nesting_error(positioned)
            # Reversed, so the first child is popped (walked) first.
            pending.extend(
                (child, depth + 1, positioned)
                for child in reversed(list(ast.iter_child_nodes(node)))
            )
    return stmts
