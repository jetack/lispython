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


FEATURES = {
    "t-string": Feature((3, 14), 't-string t"..."'),
    "comprehension-unpacking": Feature((3, 15), "comprehension unpacking [*x for ...]"),
    "match-unary-plus": Feature((3, 15), "unary plus in a match pattern"),
    "type-params": Feature((3, 12), "type parameters f[T]"),
    "type-param-default": Feature((3, 13), "type parameter defaults f[:T int]"),
    "type-alias": Feature((3, 12), "type alias statement (type ...)"),
    "lazy-import": Feature((3, 15), "lazy import (lazy-import ...)"),
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
