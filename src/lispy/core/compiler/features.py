"""The single home for everything in the compiler that depends on the
running CPython.

Forms: every form that only some releases can compile is listed in
``FEATURES`` and checked with ``require(name, sexp)`` where it is recognised,
before any node is built.  The version is read only through
``running_version()`` (tests monkeypatch it): a form the interpreter cannot run
must fail here with a lispython error, never as a CPython traceback or a
silent mistranslation.

AST shapes: where releases build the same form differently, a named
capability probe below asks the running ``ast`` (or ``token``) module instead
of comparing version numbers, so a backported change is followed automatically.  The
exception is ``compile_depth_follows_recursion_limit``, a version comparison
through ``running_version()``: probing it would overflow the C stack.  The
compiler calls the probes through
this module (``features.probe()``), which is also how tests replace them.

No other ``sys.version_info`` comparison belongs in the compiler.  The one
exception in the package is ``_PY312_PLUS`` in core/importer.py: it chooses
the calling convention of ``runpy._get_code_from_file``, a runtime API rather
than a form or an AST shape, and runs before any lispython code is compiled.
"""

import ast
import functools
import sys
import token
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
def fstring_fields_are_tokens():
    """Whether the tokenizer lexes f-string fields in place (PEP 701, 3.12).

    Decides the error for a source ending inside a field's expression: where
    True, the field's ``{`` is on the tokenizer's bracket stack, so it is
    "'X' was never closed" at the innermost bracket still open, the ``{``
    counting as one; where False (3.11), the whole f-string is one string
    token, so it is "unterminated string literal" at the outermost one.
    """
    return hasattr(token, "FSTRING_START")


def _compile_error(src):
    """The SyntaxError compile() raises for ``src``, or None."""
    try:
        compile(src, "<probe>", "exec")
    except SyntaxError as error:
        return error
    return None


@functools.cache
def single_quoted_spec_refuses_newline():
    """Whether a line break in a single-quoted f-string's format spec is an
    error of its own: "newlines are not allowed in format specifiers for
    single quoted f-strings" (3.13+). 3.12 accepts one; there the text after
    it is read as the field's code."""
    return _compile_error('f"{x:\n}"') is not None


@functools.cache
def conversion_name_checked_first():
    """Whether an invalid conversion name is refused where it is read, before
    the rest of its field (3.14.8 and 3.15, a backport): f"{x!z: is "invalid
    conversion character 'z'". 3.12 and 3.13 check it only once the field
    has closed, so an error or the end of the source in the field's format
    spec comes first."""
    error = _compile_error('f"{x!z:')
    return error is not None and error.msg.startswith(
        "f-string: invalid conversion character 'z'"
    )


@functools.cache
def expecting_brace_names_its_line():
    """Whether "expecting '}'" for a string closing the literal inside a
    field names the field's line when the string starts on another one (3.15):
    "f-string: expecting '}' to close '{' on line 1"."""
    error = _compile_error('f"{\n"')
    return error is not None and error.msg.endswith("on line 1")


@functools.cache
def final_crlf_starts_a_line():
    """Whether a \\r\\n ending the source starts a line for a string literal's
    "detected at line N" (3.11); 3.12+ count no line after it, as after a
    final \\n or lone \\r on every version."""
    error = _compile_error('s = """a\r\n')
    return error is not None and error.msg.endswith("(detected at line 2)")


@functools.cache
def t_string_prefix_exists():
    """Whether t/T (with r/R) is a string prefix (3.14+): t"..." is then one
    string token, never the name t before a string."""
    return hasattr(token, "TSTRING_START")


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

    Beyond the limit, compile()'s RecursionError becomes a positioned
    nesting error through ``compile_nesting_errors`` (compiler/utils.py),
    whose message names the recursion limit only where this is True.
    """
    return tuple(running_version())[:2] < (3, 12)


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
