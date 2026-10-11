"""The single home for version gating in the compiler.

Every form that only some CPython releases can compile is listed in
``FEATURES`` and checked with ``require(name, sexp)`` at the point the compile
function recognises the form, before it builds any node.  The running version
is read only through ``running_version()`` (tests monkeypatch it), and no other
``sys.version_info`` comparison belongs in the compiler: a form the interpreter
cannot run must fail here with a lispython error, never as a CPython traceback
or a silent mistranslation.
"""

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
