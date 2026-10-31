import ast
import importlib
from typing import Callable, Dict

from lispy.core.compiler import stmt_list_compile
from lispy.core.compiler.literal import constant_compile, string_node_value
from lispy.core.compiler.utils import compile_nesting_errors
from lispy.core.errors import LispyCompileError
from lispy.core.meta_functions import defmacro_transform, require_transform
from lispy.core.nodes import *
from lispy.core.nodes import atom_text, macro_expansion_nodes
from lispy.core.utils import trampoline

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
        raise LispyCompileError(
            f"(defmacro ...) expects a name, got {name}",
            sexp if name is None else name,
        )
    if "." in name.value:
        # Dotted registry keys belong to (require mod), which qualifies a
        # module's macros as mod.name; defmacro defines a local one.
        raise LispyCompileError(
            f"(defmacro ...) expects an undotted name, got {name}", name
        )
    transformed = defmacro_transform(sexp)
    # Expand outside compile_nesting_errors: a RecursionError there is a
    # macro's own, not nesting.
    body = macroexpand_then_compile([transformed], include_meta=include_meta)
    with compile_nesting_errors(body):
        code = compile(ast.Interactive(body=body), "macro-defining", "single")
    eval(code, scope)
    return transformed if include_meta else None


def require_macro(sexp, scope, include_meta=True):
    transformed = require_transform(sexp)
    try:
        body = macroexpand_then_compile([transformed], include_meta=include_meta)
        with compile_nesting_errors(body):
            code = compile(ast.Interactive(body=body), "macro-requiring", "single")
        eval(code, scope)
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


def _expand_1(sexp, scope, in_quasi, include_meta, expanded):
    """One node's step of macroexpand_1_and_check, as a generator.

    It asks for a child form's expansion with ``new = yield expand(child, in_quasi)``
    and returns the node's replacement; ``expanded[0]`` is set once a macro
    call is expanded. macroexpand_1_and_check runs these steps on the shared
    trampoline (core/utils.py), so the order is a recursive pre-order walk's,
    left to right, but nesting depth costs no Python frames.
    """

    def expand(child, child_in_quasi):
        return _expand_1(child, scope, child_in_quasi, include_meta, expanded)

    if isinstance(sexp, QuasiQuote):
        sexp.value = yield expand(sexp.value, True)
    elif isinstance(sexp, Quote):
        pass
    elif isinstance(sexp, Unquote):
        sexp.value = yield expand(sexp.value, False)
    elif isinstance(sexp, Wrapper):
        sexp.value = yield expand(sexp.value, in_quasi)
    elif isinstance(sexp, GluedHead):
        sexp.params = yield expand(sexp.params, in_quasi)
    elif isinstance(sexp, Expression) and len(sexp) > 0:
        [op, *operands] = sexp.list
        name = atom_text(op)
        if name == "defmacro" and not in_quasi:
            sexp = define_macro(sexp, scope, include_meta=include_meta)
        elif name == "require" and not in_quasi:
            sexp = require_macro(sexp, scope, include_meta=include_meta)
        elif (
            name in scope.setdefault("__macro_namespace", {})
            and not in_quasi
            and isinstance(sexp, Paren)
        ):
            expansion = scope["__macro_namespace"][name](
                *[_unwrap_literal(o) for o in operands]
            )
            # Python data anywhere in the expansion becomes nodes, and
            # template-built nodes take the call's position, so tracebacks and
            # compile errors point at the macro call, not line 0. The
            # expansion is expanded by the next pass, not this one.
            sexp = macro_expansion_nodes(expansion, sexp)
            expanded[0] = True
        else:
            expanded_list = []
            for x in sexp.list:
                expanded_list.append((yield expand(x, in_quasi)))
            sexp.list = [x for x in expanded_list if x is not None]
    return sexp


def macroexpand_1_and_check(sexp, scope=globals(), in_quasi=False, include_meta=True):
    """One expansion pass over ``sexp``: ``(new sexp, whether any macro call
    was expanded)``. Iterative (see _expand_1), so a form nested past the
    recursion limit expands like any other."""
    expanded = [False]
    new = trampoline(_expand_1(sexp, scope, in_quasi, include_meta, expanded))
    return new, expanded[0]


def sexp_list_expand(sexp_list, scope, include_meta=True):
    return filter(
        lambda x: not x is None,
        map(lambda x: macroexpand(x, scope, include_meta=include_meta), sexp_list),
    )


def macroexpand_then_compile(sexp_list, scope=globals(), include_meta=True):
    return stmt_list_compile(
        sexp_list_expand(sexp_list, scope, include_meta=include_meta)
    )
