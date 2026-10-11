"""Compile-time errors raised by the lispython compiler.

``LispyCompileError`` is a ``SyntaxError`` so that every consumer that already
understands Python syntax errors (tracebacks, IDEs, the LSP's
``extract-error-position``) reads ``.lineno``/``.offset`` off it unchanged.
``format_compile_error`` is the one place the CLIs, REPL and nREPL turn it
into text.
"""


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


def attach_source(err, filename=None, source=None):
    """Fill ``err.filename`` and ``err.text`` where still unset; return ``err``.

    Only missing values are filled, so the innermost holder of the source
    (e.g. the import of a nested .lpy module) wins over outer ones.
    """
    if err.filename is None and filename is not None:
        err.filename = filename
    if err.text is None and source is not None and err.lineno:
        lines = source.splitlines()
        if 0 < err.lineno <= len(lines):
            err.text = lines[err.lineno - 1]
    return err


def format_compile_error(err):
    """``path:line:col: error: msg``, then the source line and a caret."""
    path = err.filename or "<unknown>"
    head = f"{path}:{err.lineno or 0}:{err.offset or 0}: error: {err.msg}"
    if not err.text:
        return head
    text = err.text.rstrip("\r\n")
    caret = " " * max((err.offset or 1) - 1, 0) + "^"
    return f"{head}\n{text}\n{caret}"
