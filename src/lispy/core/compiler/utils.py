import ast
from collections import deque

from lispy.core.nodes import *


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
