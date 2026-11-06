import ast
import enum
from collections.abc import Callable

from lispy.core.compiler import stmt_list_compile
from lispy.core.compiler.literal import (
    ACTIVE_MACROS,
    constant_compile,
    hyphenated_name_p,
    string_node_value,
)
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
    # Binding checks stay on: a parameter or local named like an active macro
    # is a binding the body's (name ...) would expand past, and lpy and the
    # importer refuse it when they compile this same form as the module's
    # code (include_meta). The generated ___defmacro_temp___ name never
    # clashes.
    body = macroexpand_then_compile([transformed], scope, include_meta=include_meta)
    with compile_nesting_errors(body):
        code = compile(ast.Interactive(body=body), "macro-defining", "single")
    eval(code, scope)
    return transformed if include_meta else None


def require_macro(sexp, scope, include_meta=True):
    transformed = require_transform(sexp)
    try:
        body = macroexpand_then_compile(
            [transformed], scope, include_meta=include_meta, check_bindings=False
        )
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


def active_macro_kind(scope):
    """The check_binding lookup (compiler/literal.py) over scope's macros as
    they stand when a binding compiles: for a binding's Python name and
    spelling, "built-in" (its macro is the prelude's own), "macro" or None. A
    dotted key (lispy.macros.inc, from (require mod)) is never a bare name."""

    def kind(name, spelling):
        namespace = scope.get("__macro_namespace") or {}
        prelude = scope.get(PRELUDE_KEY) or {}
        found = None
        for key, macro in namespace.items():
            if "." in key:
                continue
            # Lookup is by spelling, so a binding hides a macro only when a
            # call meant for the binding could be spelled as the macro. A
            # hyphenated name's Python name is a second spelling of it (my-x
            # and my_x are one name); an operator-spelled key (->, --) mangles
            # to a name nobody calls it by (_to_, __), so only its own
            # spelling clashes.
            if hyphenated_name_p(key):
                if Symbol(key).name != name:
                    continue
            elif key != spelling:
                continue
            if prelude.get(key) is macro:
                return "built-in"
            found = "macro"
        return found

    return kind


def macroexpand_then_compile(sexp_list, scope, include_meta=True, check_bindings=True):
    """Expand and compile each form in turn. With check_bindings, binding a
    name that is a macro in scope at that point is a compile error
    (check_binding); require turns it off for the import it generates and
    runs at compile time."""
    token = ACTIVE_MACROS.set(active_macro_kind(scope) if check_bindings else None)
    try:
        return stmt_list_compile(
            sexp_list_expand(sexp_list, scope, include_meta=include_meta)
        )
    finally:
        ACTIVE_MACROS.reset(token)


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
    """Record the prelude in scope (under PRELUDE_KEY) and add its macros to
    scope's macro table, once: a scope already seeded is left as it is, and a
    name the table already defines keeps its own macro. compile_unit seeds
    every scope it compiles in; how a unit start then reshapes the table is
    UnitStart's."""
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


class UnitStart(enum.Enum):
    """How a unit's forms relate to the macro table of the scope they compile
    in (compile_unit).

    MODULE: the unit is a whole module or file (lpy, the importer, l2py, the
    LSP, an nREPL load-file). Its table starts as the prelude alone, even in
    a persistent scope an earlier unit defined macros in or opted out of, so
    a reload compiles as a fresh module does; a first form (after one
    docstring) ``(require lispy.macros <selection>)`` keeps only that
    selection.

    TOP: the unit is a region at the top of a file whose scope persists (an
    nREPL eval sent at line 1). Only the prelude's own entries are derived
    again, from the region's first form: they are dropped, then put back
    (never over a macro of the same name) unless that form is the opt-out,
    whose require adds its selection back as it compiles. Every other macro
    (the file's defmacros further down, other requires) stays, as
    re-evaluating a Clojure namespace's first form keeps its vars. A region
    with no form beyond a docstring changes nothing.

    CONTINUE: the unit is a later input of a session (a REPL input, an nREPL
    eval elsewhere in a file): macros persist and such a require only adds.
    """

    MODULE = "module"
    TOP = "top"
    CONTINUE = "continue"


def _first_form(forms):
    """The unit's first form after one docstring, or None."""
    forms = list(forms[:2])
    if forms and isinstance(forms[0], String):
        forms = forms[1:]
    return forms[0] if forms else None


def _opt_out_form_p(form):
    """Whether form, a unit's first (_first_form), is
    ``(require lispy.macros ...)``: the unit then gets only that require's
    selection of the built-in macros."""
    return (
        isinstance(form, Paren)
        and len(form) >= 2
        and atom_text(form.list[0]) == "require"
        and atom_text(form.list[1]) == PRELUDE_MODULE
    )


def _start_unit(scope, forms, start):
    """Reshape scope's macro table, in place, for a unit starting as start
    says (UnitStart)."""
    if start is UnitStart.CONTINUE:
        return
    namespace = scope.setdefault("__macro_namespace", {})
    prelude = scope.get(PRELUDE_KEY, {})
    first = _first_form(forms)
    if start is UnitStart.MODULE:
        namespace.clear()
        if not _opt_out_form_p(first):
            namespace.update(prelude)
        return
    if first is None:
        return
    for key, macro in list(namespace.items()):
        if prelude.get(key) is macro:
            del namespace[key]
    if not _opt_out_form_p(first):
        for key, macro in prelude.items():
            namespace.setdefault(key, macro)


def compile_unit(forms, scope, *, start=UnitStart.MODULE, include_meta=True):
    """Compile one unit of source (a module, a REPL input, an editor view or
    region) to a list of Python statements, expanding macros in scope.

    The one entry point of compilation: scope gets the prelude macros first
    (seed_prelude), then start (UnitStart) says how the unit's start
    reshapes its macro table. Only the macro table restarts: a reload keeps
    the Python names an earlier load of the scope defined, stale ones
    included, as a Clojure namespace keeps its vars.
    """
    forms = list(forms)
    seed_prelude(scope)
    _start_unit(scope, forms, start)
    return macroexpand_then_compile(forms, scope, include_meta=include_meta)
