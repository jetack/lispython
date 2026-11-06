import ast
from collections.abc import Callable

from lispy.core.compiler import stmt_list_compile
from lispy.core.compiler.literal import constant_compile, string_node_value
from lispy.core.compiler.utils import compile_nesting_errors
from lispy.core.errors import LispyCompileError
from lispy.core.meta_functions import defmacro_transform, require_transform
from lispy.core.nodes import *
from lispy.core.nodes import atom_text, macro_expansion_nodes
from lispy.core.utils import trampoline

__macro_namespace: dict[str, Callable[[Node], ast.AST]] = {}


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
    body = macroexpand_then_compile([transformed], scope, include_meta=include_meta)
    with compile_nesting_errors(body):
        code = compile(ast.Interactive(body=body), "macro-defining", "single")
    eval(code, scope)
    return transformed if include_meta else None


def require_macro(sexp, scope, include_meta=True):
    transformed = require_transform(sexp)
    try:
        body = macroexpand_then_compile([transformed], scope, include_meta=include_meta)
        with compile_nesting_errors(body):
            code = compile(ast.Interactive(body=body), "macro-requiring", "single")
        eval(code, scope)
    except (ModuleNotFoundError, ImportError):
        import sys

        module_name = str(sexp.list[1])
        print(
            f"l2py: warning: could not load macros from '{module_name}' "
            "(module not found)",
            file=sys.stderr,
        )
    return transformed if include_meta else None


def macroexpand(sexp, scope, include_meta=True):
    expanded = True
    while expanded:
        sexp, expanded = macroexpand_1_and_check(sexp, scope, include_meta=include_meta)
    return sexp


def macroexpand_1(sexp, scope):
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


def macroexpand_1_and_check(sexp, scope, in_quasi=False, include_meta=True):
    """One expansion pass over ``sexp``: ``(new sexp, whether any macro call
    was expanded)``. Iterative (see _expand_1), so a form nested past the
    recursion limit expands like any other."""
    expanded = [False]
    new = trampoline(_expand_1(sexp, scope, in_quasi, include_meta, expanded))
    return new, expanded[0]


def sexp_list_expand(sexp_list, scope, include_meta=True):
    return filter(
        lambda x: x is not None,
        map(lambda x: macroexpand(x, scope, include_meta=include_meta), sexp_list),
    )


def macroexpand_then_compile(sexp_list, scope, include_meta=True):
    return stmt_list_compile(
        sexp_list_expand(sexp_list, scope, include_meta=include_meta)
    )


# The built-in macros every compilation unit starts with (Clojure's
# clojure.core refer): copied into the unit's compile-time scope only, so
# they emit no code and a module's runtime __macro_namespace holds only its
# own defmacros and explicit requires. The scope keeps the copy under this
# key, which marks it seeded (a persistent scope, such as a REPL session, is
# seeded once) and lets an opt-out remove exactly those entries.
PRELUDE_KEY = "__macro_prelude"
PRELUDE_MODULE = "lispy.macros"


def _prelude_source_p(scope):
    """Whether scope compiles lispy.macros itself (or a submodule), the
    prelude's source: importing it there would be circular."""
    for value in (scope.get("__name__"), scope.get("__package__")):
        if isinstance(value, str) and (
            value == PRELUDE_MODULE or value.startswith(PRELUDE_MODULE + ".")
        ):
            return True
    return False


def seed_prelude(scope):
    """Give scope the prelude macros, once. A name the scope already defines
    keeps its own macro."""
    if PRELUDE_KEY in scope:
        return
    if _prelude_source_p(scope):
        prelude = {}
    else:
        import lispy.macros

        prelude = dict(getattr(lispy.macros, "__macro_namespace"))
    scope[PRELUDE_KEY] = prelude
    namespace = scope.setdefault("__macro_namespace", {})
    for name, macro in prelude.items():
        namespace.setdefault(name, macro)


def _opt_out_form_p(forms):
    """Whether the unit's first form, after one docstring, is
    ``(require lispy.macros ...)``: the unit then gets only that require's
    selection of the built-in macros."""
    forms = list(forms[:2])
    if forms and isinstance(forms[0], String):
        forms = forms[1:]
    if not forms:
        return False
    form = forms[0]
    return (
        isinstance(form, Paren)
        and len(form) >= 2
        and atom_text(form.list[0]) == "require"
        and atom_text(form.list[1]) == PRELUDE_MODULE
    )


def _drop_prelude(scope):
    """Remove the prelude's entries from scope's macros, by identity: a macro
    the unit defined under a prelude name stays."""
    namespace = scope.setdefault("__macro_namespace", {})
    for name, macro in scope.get(PRELUDE_KEY, {}).items():
        if namespace.get(name) is macro:
            del namespace[name]


def compile_unit(forms, scope, *, opt_out=True, include_meta=True):
    """Compile one unit of source (a module, a REPL input, an editor view)
    to a list of Python statements, expanding macros in scope.

    The one entry point of compilation: scope gets the prelude macros first.
    With opt_out (module semantics), a unit whose first form is
    ``(require lispy.macros <selection>)`` keeps only that selection of them;
    elsewhere (a REPL session's later inputs) such a require only adds.
    """
    forms = list(forms)
    seed_prelude(scope)
    if opt_out and _opt_out_form_p(forms):
        _drop_prelude(scope)
    return macroexpand_then_compile(forms, scope, include_meta=include_meta)
