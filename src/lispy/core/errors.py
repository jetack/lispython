"""Compile-time errors raised by the lispython compiler.

``LispyCompileError`` is a ``SyntaxError`` so that every consumer that already
understands Python syntax errors (tracebacks, IDEs, the LSP's
``extract-error-position``) reads ``.lineno``/``.offset`` off it unchanged.
``format_compile_error`` is the one place the CLIs, REPL and nREPL turn it
into text.
"""

import ast
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


def nesting_error(at):
    """The one error for a form nested deeper than Python can compile.

    ``at`` is a lispython node, or a Python AST node (positioned by its own
    lineno/col_offset/end_lineno/end_col_offset).
    """
    if not hasattr(at, "position_info"):
        at = SimpleNamespace(
            position_info={
                key: getattr(at, key, None)
                for key in ("lineno", "col_offset", "end_lineno", "end_col_offset")
            }
        )
    return LispyCompileError(
        "form nested too deeply to compile "
        f"(Python's recursion limit is {sys.getrecursionlimit()})",
        at,
    )


# Frames to spare below the recursion limit when checking an AST's depth.
# CPython's compile() accepts an AST nested about recursion-limit minus the
# current Python stack depth (measured on 3.11-3.15: limit 1000 at the top
# level -> 992 levels; limit 3000 at 500 frames deep -> 2486) and raises a
# raw RecursionError beyond. The entry points call compile() a few frames
# away from refuse_deep_ast's caller, so the margin covers that difference
# and the few levels CPython spends itself.
AST_NESTING_MARGIN = 50


def _stack_depth():
    depth = 0
    frame = sys._getframe(1)
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


def refuse_deep_ast(stmts):
    """Raise ``nesting_error`` if any statement in ``stmts`` is nested deeper
    than CPython's compile() can take from here, else return ``stmts``.

    One iterative walk per statement (linear in the AST's size), so a deep
    tree is refused at a lispython position instead of failing in compile()
    with a RecursionError that names no line. The error is at the node at the
    budget's depth on the first too-deep path, or at its nearest positioned
    ancestor when that node has no position (an operator, a context).
    """
    budget = sys.getrecursionlimit() - _stack_depth() - AST_NESTING_MARGIN
    for stmt in stmts:
        # (node, depth, nearest positioned node on the path, itself included)
        pending = [(stmt, 1, stmt)]
        while pending:
            [node, depth, positioned] = pending.pop()
            if getattr(node, "lineno", None):
                positioned = node
            if depth >= budget:
                raise nesting_error(positioned)
            pending.extend(
                (child, depth + 1, positioned) for child in ast.iter_child_nodes(node)
            )
    return stmts
