import ast
import importlib
from typing import Callable, Dict

from lispy.core.compiler import def_args_parse, stmt_list_compile
from lispy.core.compiler.literal import constant_compile, string_node_value
from lispy.core.errors import LispyCompileError
from lispy.core.meta_functions import defmacro_transform, require_transform
from lispy.core.nodes import *
from lispy.core.nodes import macro_expansion_nodes

__macro_namespace: Dict[str, Callable[[Node], ast.AST]] = {}

def _unwrap_literal(node):
    if isinstance(node, Constant):
        try:
            return constant_compile(node).value
        except LispyCompileError:
            return node
    elif isinstance(node, String):
        try:
            return string_node_value(node)
        except LispyCompileError:
            return node
    return node


def define_macro(sexp, scope, include_meta=True):
    name = sexp[1] if len(sexp) > 1 else None
    if isinstance(name, GluedHead):
        raise name.misuse_error()
    if type(name) is not Symbol:
        raise LispyCompileError(f"(defmacro ...) expects a name, got {name}", sexp)
    transformed = defmacro_transform(sexp)
    eval(
        compile(
            ast.Interactive(body=macroexpand_then_compile([transformed], include_meta=include_meta)),
            "macro-defining",
            "single",
        ),
        scope,
    )
    return transformed if include_meta else None


def require_macro(sexp, scope, include_meta=True):
    transformed = require_transform(sexp)
    try:
        eval(
            compile(
                ast.Interactive(body=macroexpand_then_compile([transformed], include_meta=include_meta)),
                "macro-requiring",
                "single",
            ),
            scope,
        )
    except (ModuleNotFoundError, ImportError):
        import sys
        module_name = str(sexp.list[1])
        print(f"l2py: warning: could not load macros from '{module_name}' (module not found)", file=sys.stderr)
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
        sexp.value, expanded = macroexpand_1_and_check(
            sexp.value, scope, in_quasi=True, include_meta=include_meta
        )
    elif isinstance(sexp, Quote):
        pass
    elif isinstance(sexp, Unquote):
        sexp.value, expanded = macroexpand_1_and_check(
            sexp.value, scope, include_meta=include_meta
        )
    elif isinstance(sexp, Wrapper):
        sexp.value, expanded = macroexpand_1_and_check(
            sexp.value, scope, in_quasi=in_quasi, include_meta=include_meta
        )
    elif isinstance(sexp, GluedHead):
        sexp.params, expanded = macroexpand_1_and_check(
            sexp.params, scope, in_quasi=in_quasi, include_meta=include_meta
        )
    elif isinstance(sexp, Expression) and len(sexp) > 0:
        [op, *operands] = sexp.list
        if str(op) == "defmacro" and not in_quasi:
            sexp = define_macro(sexp, scope, include_meta=include_meta)
        elif str(op) == "require" and not in_quasi:
            sexp = require_macro(sexp, scope, include_meta=include_meta)
        elif str(op) in scope.setdefault("__macro_namespace", {}) and not in_quasi and isinstance(sexp, Paren):
            call_position = sexp.position_info
            expansion = scope["__macro_namespace"][str(op)](
                *[_unwrap_literal(o) for o in operands]
            )
            # Python data anywhere in the expansion becomes nodes, and
            # template-built nodes take the call's position, so tracebacks and
            # compile errors point at the macro call, not line 0.
            sexp = macro_expansion_nodes(expansion, call_position)
            expanded = True
        else:
            expanded_list, expanded_feedbacks = zip(
                *map(
                    lambda x: macroexpand_1_and_check(x, scope, in_quasi=in_quasi, include_meta=include_meta),
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
