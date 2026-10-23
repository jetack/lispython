import ast
from collections import deque

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
