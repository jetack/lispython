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
exceptions are ``compile_depth_follows_recursion_limit``,
``inlined_comprehensions_overflow_block_stack`` and
``annotations_have_own_scope``, version comparisons through
``running_version()``: probing the first would overflow the C stack, the
others would crash the interpreter.  The compiler calls the probes through
this module (``features.probe()``), which is also how tests replace them.

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


def compile_depth_follows_recursion_limit():
    """Whether compile() refuses an AST nested about as deep as the Python
    recursion limit, raising a RecursionError that names no line: CPython
    before 3.12, whose compiler recursion counts against the Python limit
    (3.12 moved it to a C-level limit). A fixed fact of the release, so it is
    read from ``running_version()``; compiling a probe AST to find out would
    itself overflow the C stack under a raised recursion limit.

    Measured on ASTs built with explicit positions (fix_missing_locations
    recurses itself), default limit 1000 unless noted:

    ======  ==========================================  ======================
    Python  deepest AST compile() accepts               beyond
    ======  ==========================================  ======================
    3.11    limit - stack depth (994 at the top level;  RecursionError
            2492 at limit 3000, 504 frames deep)
    3.12    about 9995, at any recursion limit          RecursionError (C limit)
    3.13    about 9996, at any recursion limit          RecursionError (C limit)
    3.14    47381 BinOp / 28955 List levels             RecursionError (C stack)
    3.15    26073 BinOp / 16821 List levels             RecursionError (C stack)
    ======  ==========================================  ======================

    Only the first is a depth lispython code meets, so ``refuse_deep_ast`` in
    compiler/utils.py checks the depth only where this is True. Beyond the
    limit on 3.12+, compile()'s RecursionError becomes the same positioned
    nesting error through ``compile_nesting_errors`` (compiler/utils.py).
    """
    return tuple(running_version())[:2] < (3, 12)


def inlined_comprehensions_overflow_block_stack():
    """Whether compile() can crash, instead of refusing, on list/set/dict
    comprehensions nested past CPython's static block limit: every release
    that inlines them (PEP 709, 3.12+). A fixed fact of the release, so it is
    read from ``running_version()``; probing would crash the process.
    ``refuse_block_overflow`` in compiler/utils.py has the mechanism and
    applies the rule before compile() runs.

    ======================  ===============================================
    Python                  past the static block limit in one code unit
    ======================  ===============================================
    3.11                    no inlining: unaffected
    3.12.x                  segfault (gh-156091; security-only, never fixed)
    3.13 < 3.13.16,         segfault (gh-156091)
    3.14 < 3.14.8, 3.15.0
    3.13.16, 3.14.8,        SyntaxError "too many statically nested blocks"
    3.15.1                  at the comprehension pushing past the limit, but
                            still a segfault on an ``await`` (or an
                            ``async for`` clause) inside a generator
                            expression at the limit: a separate CPython
                            bug no release has fixed
    ======================  ===============================================
    """
    return tuple(running_version())[:2] >= (3, 12)


def annotations_have_own_scope():
    """Whether a def's parameter and return annotations compile in a code
    unit of their own (PEP 649's ``__annotate__``, 3.14+) rather than
    eagerly in the enclosing one. A fixed fact of the release, read from
    ``running_version()`` like the block check it serves: compiling a probe
    to find out is what that check guards."""
    return tuple(running_version())[:2] >= (3, 14)


@functools.cache
def type_alias_classes():
    """The ast classes of a ``type`` statement: ``(ast.TypeAlias,)`` from
    3.12, none before."""
    return (ast.TypeAlias,) if hasattr(ast, "TypeAlias") else ()


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
