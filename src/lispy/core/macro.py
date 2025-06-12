import ast
import importlib
from typing import Callable, Dict

from lispy.core.compiler import def_args_parse, stmt_list_compile
from lispy.core.meta_functions import defmacro_transform, require_transform
from lispy.core.nodes import *

__macro_namespace: Dict[str, Callable[[Node], ast.AST]] = {}


def define_macro(sexp, scope, include_meta=True):
    transformed = defmacro_transform(sexp)
    eval(
        compile(
            ast.Interactive(body=macroexpand_then_compile([transformed])),
            "macro-defining",
            "single",
        ),
        scope,
    )
    return transformed if include_meta else None


def require_macro(sexp, scope, include_meta=True):
    transformed = require_transform(sexp)
    eval(
        compile(
            ast.Interactive(body=macroexpand_then_compile([transformed])),
            "macro-requiring",
            "single",
        ),
        scope,
    )
    return transformed if include_meta else None


def macroexpand(sexp, scope=globals(), include_meta=True):
    expanded = True
    while expanded:
        sexp, expanded = macroexpand_1_and_check(sexp, scope, include_meta=include_meta)
    return sexp


def macroexpand_1(sexp, scope=globals()):
    return macroexpand_1_and_check(sexp, scope)[0]


def macroexpand_1_and_check(sexp, scope=globals(), in_quasi=False, include_meta=True):
    expanded = False
    if isinstance(sexp, QuasiQuote):
        sexp.value, expanded = macroexpand_1_and_check(sexp.value, scope, in_quasi=True)
    elif isinstance(sexp, Quote):
        pass
    elif isinstance(sexp, Unquote):
        sexp.value, expanded = macroexpand_1_and_check(sexp.value, scope)
    elif isinstance(sexp, Wrapper):
        sexp.value, expanded = macroexpand_1_and_check(
            sexp.value, scope, in_quasi=in_quasi
        )
    elif isinstance(sexp, Expression) and len(sexp) > 0:
        [op, *operands] = sexp.list
        if str(op) == "defmacro" and not in_quasi:
            sexp = define_macro(sexp, scope, include_meta=include_meta)
        elif str(op) == "require" and not in_quasi:
            sexp = require_macro(sexp, scope, include_meta=include_meta)
        elif str(op) in scope.setdefault("__macro_namespace", {}) and not in_quasi:
            sexp = scope["__macro_namespace"][str(op)](*operands)
            expanded = True
        else:
            expanded_list, expanded_feedbacks = zip(
                *map(
                    lambda x: macroexpand_1_and_check(x, scope, in_quasi=in_quasi),
                    sexp.list,
                )
            )
            sexp.list = list(filter(lambda x: not x is None, expanded_list))
            expanded = any(expanded_feedbacks)
    return sexp, expanded


def sexp_list_expand(sexp_list, scope, include_meta=True):
    return filter(
        lambda x: not x is None,
        map(lambda x: macroexpand(x, scope, include_meta=include_meta), sexp_list),
    )


def macroexpand_then_compile(sexp_list, scope=globals(), include_meta=True):
    return stmt_list_compile(
        sexp_list_expand(sexp_list, scope, include_meta=include_meta)
    )
