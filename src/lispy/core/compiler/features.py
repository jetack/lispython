"""The single home for everything in the compiler that depends on the
running CPython.

Forms: every form that only some releases can compile is listed in
``FEATURES`` and checked with ``require(name, sexp)`` where it is recognised,
before any node is built.  The version is read only through
``running_version()`` (tests monkeypatch it): a form the interpreter cannot run
must fail here with a lispython error, never as a CPython traceback or a
silent mistranslation.

AST shapes: where releases build the same form differently, a named
capability probe below asks the running ``ast`` module instead of comparing
version numbers, so a backported change is followed automatically.  The
compiler calls the probes through this module (``features.probe()``), which
is also how tests replace them.

No other ``sys.version_info`` comparison belongs in the compiler.  The one
exception in the package is ``_PY312_PLUS`` in core/importer.py: it chooses
the calling convention of ``runpy._get_code_from_file``, a runtime API rather
than a form or an AST shape, and runs before any lispython code is compiled.
"""

import ast
import functools
import sys
from dataclasses import dataclass

from lispy.core.errors import LispyCompileError


@dataclass(frozen=True)
class Feature:
    min_version: tuple
    description: str


# Each description names the construct, then its lispython shape with "..."
# for the parts that vary (never a made-up example name).
FEATURES = {
    "t-string": Feature((3, 14), 't-string t"..."'),
    "comprehension-unpacking": Feature(
        (3, 15), "comprehension unpacking [*... for ...]"
    ),
    "match-unary-plus": Feature((3, 15), "unary plus pattern (case +... ...)"),
    "type-params": Feature((3, 12), "type parameters name[...]"),
    "type-param-default": Feature((3, 13), "type parameter default name[:... ...]"),
    "type-alias": Feature((3, 12), "type alias (type name ...)"),
    "lazy-import": Feature((3, 15), "lazy import (lazy-import ...)"),
    "lazy-from": Feature((3, 15), "lazy from-import (lazy-from ...)"),
}


def running_version():
    return sys.version_info


@functools.cache
def interpolation_keeps_trailing_whitespace():
    """Whether Interpolation.str keeps the whitespace after the expression.

    CPython 3.14 strips it and 3.15 keeps it (``{ x }`` -> " x" vs " x ").
    """
    probe = ast.parse('t"{x }"', mode="eval").body
    return probe.values[0].str == "x "


@functools.cache
def compile_depth_follows_recursion_limit():
    """Whether compile() refuses an AST nested about as deep as the Python
    recursion limit, raising a RecursionError that names no line.

    Measured on ASTs built with explicit positions (fix_missing_locations
    recurses itself), at the default limit 1000: 3.11 accepts an AST nested
    ``limit - stack depth`` levels (994 at the top level; 2492 at limit 3000,
    500 frames deep); 3.12 and 3.13 have a fixed C limit (about 9995 levels
    at any recursion limit); 3.14 and 3.15 run until the C stack overflows
    (16821 to 47381 levels by node kind), both a RecursionError. Only the
    first is a limit lispython code meets, so ``refuse_deep_ast`` in
    compiler/utils.py checks for it only where this probe finds it.

    The probe compiles, once, a list nested 10 levels past the recursion
    limit, which no stack depth lets 3.11 accept. With the limit raised past
    the fixed C limit before the first call, 3.12/3.13 read True too; the
    check then refuses nothing compile() would have taken.
    """
    position = {"lineno": 1, "col_offset": 0, "end_lineno": 1, "end_col_offset": 1}
    node = ast.Constant(1, **position)
    for _ in range(sys.getrecursionlimit() + 10):
        node = ast.List([node], ast.Load(), **position)
    try:
        compile(ast.Expression(node), "<depth-probe>", "eval")
    except RecursionError:
        return True
    return False


def import_has_is_lazy():
    """Whether Import/ImportFrom have ``is_lazy`` (3.15).

    ast.parse sets it to 0 on every plain import where the field exists; it
    must never be passed where it does not.
    """
    return "is_lazy" in ast.Import._fields


def require(name, sexp):
    """Raise ``LispyCompileError`` at ``sexp`` if the interpreter is too old."""
    feature = FEATURES[name]
    running = tuple(running_version())
    if running[:2] < feature.min_version:
        needed = ".".join(map(str, feature.min_version))
        have = ".".join(map(str, running[:3]))
        raise LispyCompileError(
            f"{feature.description} needs Python {needed}+, running {have}", sexp
        )
