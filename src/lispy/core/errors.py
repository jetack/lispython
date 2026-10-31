"""Compile-time errors raised by the lispython compiler.

``LispyCompileError`` is a ``SyntaxError`` so that every consumer that already
understands Python syntax errors (tracebacks, IDEs, the LSP's
``extract-error-position``) reads ``.lineno``/``.offset`` off it unchanged.
``format_compile_error`` is the one place the CLIs, REPL and nREPL turn it
into text.
"""

import contextlib
import sys
from types import SimpleNamespace


class LispyCompileError(SyntaxError):
    """A lispython form the compiler refuses, positioned at its s-expression.

    Offsets follow ``SyntaxError``: ``offset``/``end_offset`` are 1-based
    columns (``col_offset + 1``).  ``filename`` and ``text`` are unknown where
    the error is raised and are filled in by whoever holds the source, through
    ``attach_source``.
    """

    def __init__(self, msg, sexp=None, *, filename=None, text=None):
        pos = sexp.position_info if sexp is not None else {}
        lineno = pos.get("lineno") or None
        col = pos.get("col_offset")
        end_lineno = pos.get("end_lineno") or None
        end_col = pos.get("end_col_offset")
        super().__init__(
            msg,
            (
                filename,
                lineno,
                None if lineno is None or col is None else col + 1,
                text,
                end_lineno,
                None if end_lineno is None or end_col is None else end_col + 1,
            ),
        )

    @classmethod
    def from_syntax_error(cls, err):
        """The same error, from CPython compiling a lispython-built AST.

        Its positions already come from the lispython source, since the AST
        carries them; only the type changes, so every entry point reports it
        the same way.
        """
        new = cls(err.msg)
        for attr in ("filename", "lineno", "offset", "text", "end_lineno"):
            setattr(new, attr, getattr(err, attr))
        new.end_offset = err.end_offset
        if new.text is not None:
            # CPython reads the line with its newline; ours never carry one.
            new.text = new.text.rstrip("\r\n")
        return new


def attach_source(err, filename=None, source=None):
    """Fill ``err.filename`` and ``err.text`` where still unset; return ``err``.

    Only missing values are filled, so the innermost holder of the source
    (e.g. the import of a nested .lpy module) wins over outer ones.
    """
    if err.filename is None and filename is not None:
        err.filename = filename
    # The source line only belongs to an error in this same file (a module
    # imported while expanding macros may raise its own).
    if (
        err.text is None
        and source is not None
        and err.lineno
        and err.filename == filename
    ):
        lines = source.splitlines()
        if 0 < err.lineno <= len(lines):
            err.text = lines[err.lineno - 1]
    return err


@contextlib.contextmanager
def compile_errors(filename, source):
    """Report what goes wrong compiling ``source`` as a LispyCompileError.

    The one place every entry point (import hooks, lpy -c, REPL, nREPL, l2py)
    turns compile failures into lispython errors: a LispyCompileError gets
    ``filename`` and its source line, and a SyntaxError CPython raises while
    compiling the AST lispython built (e.g. a lazy import inside a def) is
    converted, keeping its positions, which come from the lispython source.
    Wrap parsing and compiling only, never running the code: a SyntaxError
    the program raises at run time is not a lispython compile error.
    """
    try:
        yield
    except LispyCompileError as err:
        raise attach_source(err, filename, source)
    except SyntaxError as err:
        converted = LispyCompileError.from_syntax_error(err)
        raise attach_source(converted, filename, source) from err


def format_compile_error(err):
    """``path:line:col: error: msg``, then the source line and a caret."""
    path = err.filename or "<unknown>"
    head = f"{path}:{err.lineno or 0}:{err.offset or 0}: error: {err.msg}"
    if not err.text:
        return head
    text = err.text.rstrip("\r\n")
    caret = " " * max((err.offset or 1) - 1, 0) + "^"
    return f"{head}\n{text}\n{caret}"


def _positioned(at):
    """``at`` as something ``LispyCompileError`` positions an error at: a
    lispython node as it is, a Python AST node through its own
    lineno/col_offset/end_lineno/end_col_offset."""
    if hasattr(at, "position_info"):
        return at
    return SimpleNamespace(
        position_info={
            key: getattr(at, key, None)
            for key in ("lineno", "col_offset", "end_lineno", "end_col_offset")
        }
    )


def nesting_error(at, *, recursion_limit, what="compile"):
    """The one error for a form nested deeper than Python can ``what``
    (compile, or translate for l2py's ``ast.unparse``).

    ``at`` is a lispython node, or a Python AST node (positioned by its own
    lineno/col_offset/end_lineno/end_col_offset). ``recursion_limit`` says
    whether the limit hit is Python's recursion limit, which the message then
    names; it is not for compile() on 3.12+, whose limit is CPython's own (a C
    limit or the C stack), so naming the recursion limit there would suggest
    raising it helps.
    """
    msg = f"form nested too deeply to {what}"
    if recursion_limit:
        msg += f" (Python's recursion limit is {sys.getrecursionlimit()})"
    return LispyCompileError(msg, _positioned(at))

