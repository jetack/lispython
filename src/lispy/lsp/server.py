"""LisPython Language Server — LSP implementation using pygls."""

from __future__ import annotations

import builtins
import inspect
import logging
import re
from typing import Optional

from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer

from lispy.core.nodes import (
    Annotation,
    Bracket,
    Constant,
    Expression,
    Keyword,
    MetaIndicator,
    Node,
    Paren,
    String,
    Symbol,
    Wrapper,
)
from lispy.core.parser import parse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

server = LanguageServer("lpy-lsp", "v0.1.0")

# ---------------------------------------------------------------------------
# Special-form documentation (for hover)
# ---------------------------------------------------------------------------

SPECIAL_FORM_DOCS: dict[str, str] = {
    # Statement forms
    "def": (
        "```\n(def name [params] body...)\n```\n\n"
        "Define a function. Parameters are given in a bracket list. "
        "An optional docstring can follow the parameter list."
    ),
    "async-def": (
        "```\n(async-def name [params] body...)\n```\n\n"
        "Define an async function (coroutine)."
    ),
    "class": (
        "```\n(class Name [bases] body...)\n```\n\n"
        "Define a class. Bases are given in a bracket list."
    ),
    "defmacro": (
        "```\n(defmacro name [params] body...)\n```\n\n"
        "Define a compile-time macro. The macro receives unevaluated "
        "S-expressions and must return an S-expression."
    ),
    "if": (
        "```\n(if condition then-body)\n(if condition then-body else-body)\n```\n\n"
        "Conditional statement. Compiles to Python `if/else`."
    ),
    "while": (
        "```\n(while condition body...)\n```\n\n"
        "While loop."
    ),
    "for": (
        "```\n(for target iterable body...)\n```\n\n"
        "For loop. `target` is bound to each element of `iterable`."
    ),
    "async-for": (
        "```\n(async-for target iterable body...)\n```\n\n"
        "Async for loop — use inside `async-def`."
    ),
    "do": (
        "```\n(do body...)\n```\n\n"
        "Execute a sequence of statements (block). "
        "Used where a single expression is expected but multiple statements are needed."
    ),
    "match": (
        "```\n(match subject case...)\n```\n\n"
        "Structural pattern matching (Python 3.10+)."
    ),
    "try": (
        "```\n(try body except-clauses... [finally-clause])\n```\n\n"
        "Try/except/finally statement."
    ),
    "with": (
        "```\n(with [ctx-expr as-name] body...)\n```\n\n"
        "Context manager statement."
    ),
    "async-with": (
        "```\n(async-with [ctx-expr as-name] body...)\n```\n\n"
        "Async context manager — use inside `async-def`."
    ),
    "import": (
        "```\n(import module)\n(import module :as alias)\n```\n\n"
        "Import a module."
    ),
    "from": (
        "```\n(from module [names...])\n(from module [name :as alias ...])\n```\n\n"
        "Import specific names from a module."
    ),
    "require": (
        "```\n(require module *)\n(require module [names...])\n```\n\n"
        "Import macros from a LisPython macro module at compile time."
    ),
    "=": (
        "```\n(= target value)\n```\n\n"
        "Assignment statement."
    ),
    ":=": (
        "```\n(:= target value)\n```\n\n"
        "Walrus operator (assignment expression)."
    ),
    "return": (
        "```\n(return value)\n```\n\n"
        "Return a value from a function."
    ),
    "raise": (
        "```\n(raise exception)\n```\n\n"
        "Raise an exception."
    ),
    "assert": (
        "```\n(assert condition)\n(assert condition message)\n```\n\n"
        "Assert that a condition is true."
    ),
    "del": (
        "```\n(del target)\n```\n\n"
        "Delete a name or item."
    ),
    "pass": (
        "```\n(pass)\n```\n\n"
        "No-op placeholder statement."
    ),
    "break": "```\n(break)\n```\n\nBreak out of a loop.",
    "continue": "```\n(continue)\n```\n\nContinue to the next loop iteration.",
    "global": (
        "```\n(global name...)\n```\n\n"
        "Declare names as global variables."
    ),
    "nonlocal": (
        "```\n(nonlocal name...)\n```\n\n"
        "Declare names as nonlocal variables."
    ),
    "yield": (
        "```\n(yield value)\n```\n\n"
        "Yield a value from a generator."
    ),
    "yield-from": (
        "```\n(yield-from iterable)\n```\n\n"
        "Yield all values from a sub-generator."
    ),
    "await": (
        "```\n(await expr)\n```\n\n"
        "Await a coroutine."
    ),
    "deco": (
        "```\n(deco decorator (def ...))\n```\n\n"
        "Apply a decorator to the following function/class definition."
    ),
    "lambda": (
        "```\n(lambda [params] body)\n```\n\n"
        "Anonymous function (lambda expression)."
    ),
    # Expression forms
    "ife": (
        "```\n(ife condition then-expr else-expr)\n```\n\n"
        "Ternary conditional expression (`then-expr if condition else else-expr`)."
    ),
    ".": (
        "```\n(. obj attr)\n```\n\n"
        "Attribute access (`obj.attr`)."
    ),
    "sub": (
        "```\n(sub obj index)\n```\n\n"
        "Subscript access (`obj[index]`)."
    ),
    ",": (
        "```\n(, a b c)\n```\n\n"
        "Tuple literal."
    ),
    "fn": (
        "```\n(fn [params] body)\n```\n\n"
        "Lambda alias — same as `lambda`."
    ),
    # Operators
    "+": "Arithmetic addition / unary positive.",
    "-": "Arithmetic subtraction / unary negative.",
    "*": "Arithmetic multiplication / iterable unpacking.",
    "/": "True division.",
    "//": "Floor (integer) division.",
    "%": "Modulo (remainder).",
    "**": "Exponentiation.",
    "@": "Matrix multiplication.",
    "<<": "Bitwise left shift.",
    ">>": "Bitwise right shift.",
    "|": "Bitwise OR.",
    "^": "Bitwise XOR.",
    "&": "Bitwise AND.",
    "~": "Bitwise NOT / unquote (inside quasiquote).",
    "and": "Logical AND (short-circuit).",
    "or": "Logical OR (short-circuit).",
    "not": "Logical NOT.",
    "==": "Equality test.",
    "!=": "Inequality test.",
    "<": "Less than.",
    "<=": "Less than or equal.",
    ">": "Greater than.",
    ">=": "Greater than or equal.",
    "is": "Identity test (`x is y`).",
    "is-not": "Negated identity test (`x is not y`).",
    "in": "Membership test (`x in y`).",
    "not-in": "Negated membership test (`x not in y`).",
    # Macro-related
    "'": "Quote — prevent evaluation; return the form as data.",
    "`": "Quasiquote — like quote but allows unquote (`~`) splicing.",
    "~@": "Unquote-splice — splice a list into a quasiquote.",
    "f-string": (
        "```\nf\"text {expr} more\"\n```\n\n"
        "Formatted string literal (f-string)."
    ),
}

# ---------------------------------------------------------------------------
# Python builtins introspection
# ---------------------------------------------------------------------------


def _build_builtin_info() -> dict[str, dict]:
    """Build a dict of Python builtin names → {signature, doc, source_file, lineno}."""
    info: dict[str, dict] = {}
    for name in dir(builtins):
        if name.startswith("_"):
            continue
        obj = getattr(builtins, name)
        if not callable(obj) and not isinstance(obj, type):
            continue

        entry: dict = {"name": name}

        # Signature
        try:
            sig = inspect.signature(obj)
            entry["signature"] = f"{name}{sig}"
        except (ValueError, TypeError):
            entry["signature"] = name

        # Docstring
        doc = inspect.getdoc(obj)
        if doc:
            # Take first paragraph only for hover
            entry["doc"] = doc.split("\n\n")[0]
        else:
            entry["doc"] = ""

        # Source file (works for Python-implemented builtins, not C ones)
        try:
            source_file = inspect.getfile(obj)
            try:
                _, lineno = inspect.getsourcelines(obj)
                entry["lineno"] = lineno
            except (OSError, TypeError):
                lineno = 1
                entry["lineno"] = lineno
            entry["source_file"] = source_file
        except (TypeError, OSError):
            entry["source_file"] = None
            entry["lineno"] = None

        info[name] = entry

    # Also add common names that are technically in builtins but not callable
    # like True, False, None — skip, they're constants
    return info


BUILTIN_INFO = _build_builtin_info()

# LisPython uses hyphens, so also map hyphenated names
# e.g., "is-instance" → isinstance (not needed here since builtins use no hyphens)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _position(line: int, char: int) -> lsp.Position:
    """Create an LSP Position (0-based)."""
    return lsp.Position(line=line, character=char)


def _range(
    start_line: int, start_char: int, end_line: int, end_char: int
) -> lsp.Range:
    """Create an LSP Range (0-based)."""
    return lsp.Range(
        start=_position(start_line, start_char),
        end=_position(end_line, end_char),
    )


def _node_range(node: Node) -> lsp.Range:
    """Convert a LisPython AST node's position info to an LSP Range.

    The parser uses 1-based line numbers; LSP uses 0-based.
    """
    return _range(
        node.lineno - 1,
        node.col_offset,
        node.end_lineno - 1,
        node.end_col_offset,
    )


def _extract_error_position(exc: Exception, src: str):
    """Best-effort extraction of line/col from a parse or compile error.

    Returns (line_0based, col, end_line_0based, end_col) or None.
    """
    # Some Python exceptions carry lineno/offset
    lineno = getattr(exc, "lineno", None)
    if lineno is not None:
        col = (getattr(exc, "offset", None) or 1) - 1
        return (lineno - 1, col, lineno - 1, col)

    # Try to extract from the message  "line X col Y" patterns
    msg = str(exc)
    m = re.search(r"line\s+(\d+)", msg, re.IGNORECASE)
    if m:
        line = int(m.group(1)) - 1
        col = 0
        mc = re.search(r"col(?:umn)?\s+(\d+)", msg, re.IGNORECASE)
        if mc:
            col = int(mc.group(1))
        return (line, col, line, col)

    # Fallback: mark the first line
    return None


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _compute_diagnostics(
    source: str, uri: str
) -> list[lsp.Diagnostic]:
    diagnostics: list[lsp.Diagnostic] = []

    # 1. Parse
    try:
        tree = parse(source)
    except Exception as exc:
        pos = _extract_error_position(exc, source)
        if pos is None:
            pos = (0, 0, 0, 0)
        diagnostics.append(
            lsp.Diagnostic(
                range=_range(*pos),
                severity=lsp.DiagnosticSeverity.Error,
                source="lpy-lsp",
                message=f"Parse error: {exc}",
            )
        )
        return diagnostics

    # 2. Compile (best-effort)
    try:
        from lispy.core.macro import macroexpand_then_compile

        macroexpand_then_compile(tree)
    except Exception as exc:
        pos = _extract_error_position(exc, source)
        if pos is None:
            pos = (0, 0, 0, 0)
        diagnostics.append(
            lsp.Diagnostic(
                range=_range(*pos),
                severity=lsp.DiagnosticSeverity.Error,
                source="lpy-lsp",
                message=f"Compile error: {exc}",
            )
        )

    return diagnostics


def _publish_diagnostics(ls: LanguageServer, uri: str) -> None:
    doc = ls.workspace.get_text_document(uri)
    diagnostics = _compute_diagnostics(doc.source, uri)
    ls.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics)
    )


# ---------------------------------------------------------------------------
# Document symbols
# ---------------------------------------------------------------------------


def _symbol_name(node: Node) -> str:
    """Extract a string name from a node."""
    if isinstance(node, Symbol):
        return node.value
    if isinstance(node, Annotation):
        return _symbol_name(node.value)
    if isinstance(node, String):
        return node.value.strip("\"'")
    return str(node)


def _walk_symbols(
    nodes: list[Node],
) -> list[lsp.DocumentSymbol]:
    """Walk top-level parsed forms and produce DocumentSymbol entries."""
    symbols: list[lsp.DocumentSymbol] = []

    for node in nodes:
        if not isinstance(node, Paren) or len(node) == 0:
            continue

        op = node.op
        if not isinstance(op, Symbol):
            continue

        form = op.value

        if form in ("def", "async-def"):
            if len(node) < 2:
                continue
            name = _symbol_name(node[1])
            rng = _node_range(node)
            sel = _node_range(node[1])
            children = _walk_symbols(node.list[3:]) if len(node) > 3 else []
            symbols.append(
                lsp.DocumentSymbol(
                    name=name,
                    kind=lsp.SymbolKind.Function,
                    range=rng,
                    selection_range=sel,
                    children=children or None,
                )
            )

        elif form == "class":
            if len(node) < 2:
                continue
            name = _symbol_name(node[1])
            rng = _node_range(node)
            sel = _node_range(node[1])
            # Walk class body for method definitions
            body_nodes = node.list[3:] if len(node) > 3 else node.list[2:]
            children = _walk_symbols(body_nodes)
            symbols.append(
                lsp.DocumentSymbol(
                    name=name,
                    kind=lsp.SymbolKind.Class,
                    range=rng,
                    selection_range=sel,
                    children=children or None,
                )
            )

        elif form == "defmacro":
            if len(node) < 2:
                continue
            name = _symbol_name(node[1])
            rng = _node_range(node)
            sel = _node_range(node[1])
            symbols.append(
                lsp.DocumentSymbol(
                    name=name,
                    kind=lsp.SymbolKind.Function,
                    range=rng,
                    selection_range=sel,
                    detail="macro",
                )
            )

        elif form == "=":
            if len(node) < 2:
                continue
            name = _symbol_name(node[1])
            rng = _node_range(node)
            sel = _node_range(node[1])
            symbols.append(
                lsp.DocumentSymbol(
                    name=name,
                    kind=lsp.SymbolKind.Variable,
                    range=rng,
                    selection_range=sel,
                )
            )

        elif form in ("import", "from"):
            if len(node) < 2:
                continue
            name = _symbol_name(node[1])
            rng = _node_range(node)
            sel = _node_range(node[1])
            symbols.append(
                lsp.DocumentSymbol(
                    name=name,
                    kind=lsp.SymbolKind.Module,
                    range=rng,
                    selection_range=sel,
                )
            )

        elif form == "require":
            if len(node) < 2:
                continue
            name = _symbol_name(node[1])
            rng = _node_range(node)
            sel = _node_range(node[1])
            symbols.append(
                lsp.DocumentSymbol(
                    name=name,
                    kind=lsp.SymbolKind.Module,
                    range=rng,
                    selection_range=sel,
                    detail="macro require",
                )
            )

    return symbols


# ---------------------------------------------------------------------------
# Hover
# ---------------------------------------------------------------------------


def _node_at_position(
    nodes: list[Node], line_0: int, col: int
) -> Optional[Node]:
    """Find the most specific node at the given 0-based position."""
    for node in nodes:
        n_start_line = node.lineno - 1
        n_start_col = node.col_offset
        n_end_line = node.end_lineno - 1
        n_end_col = node.end_col_offset

        # Check if position is inside this node's range
        if (line_0, col) < (n_start_line, n_start_col):
            continue
        if (line_0, col) > (n_end_line, n_end_col):
            continue

        # Position is within this node — try to go deeper
        if isinstance(node, Expression):
            deeper = _node_at_position(node.list, line_0, col)
            if deeper is not None:
                return deeper
        if isinstance(node, (Wrapper, MetaIndicator)):
            deeper = _node_at_position([node.value], line_0, col)
            if deeper is not None:
                return deeper

        return node

    return None


# ---------------------------------------------------------------------------
# Definition index
# ---------------------------------------------------------------------------


def _collect_definitions(
    nodes: list[Node], uri: str, defs: dict[str, list[lsp.Location]] | None = None
) -> dict[str, list[lsp.Location]]:
    """Walk AST and collect all definition locations keyed by symbol name."""
    if defs is None:
        defs = {}

    for node in nodes:
        if not isinstance(node, Paren) or len(node) == 0:
            continue

        op = node.op
        if not isinstance(op, Symbol):
            continue

        form = op.value

        if form in ("def", "async-def", "defmacro"):
            if len(node) >= 2 and isinstance(node[1], Symbol):
                name = node[1].value
                loc = lsp.Location(uri=uri, range=_node_range(node[1]))
                defs.setdefault(name, []).append(loc)
                # Recurse into function body for nested defs
                if len(node) > 3:
                    _collect_definitions(node.list[3:], uri, defs)

        elif form == "class":
            if len(node) >= 2 and isinstance(node[1], Symbol):
                name = node[1].value
                loc = lsp.Location(uri=uri, range=_node_range(node[1]))
                defs.setdefault(name, []).append(loc)
                # Recurse into class body
                body = node.list[3:] if len(node) > 3 else node.list[2:]
                _collect_definitions(body, uri, defs)

        elif form == "=":
            if len(node) >= 2 and isinstance(node[1], Symbol):
                name = node[1].value
                loc = lsp.Location(uri=uri, range=_node_range(node[1]))
                defs.setdefault(name, []).append(loc)

        elif form in ("for", "async-for"):
            # Loop variable binding
            if len(node) >= 2 and isinstance(node[1], Symbol):
                name = node[1].value
                loc = lsp.Location(uri=uri, range=_node_range(node[1]))
                defs.setdefault(name, []).append(loc)
            # Recurse into loop body
            _collect_definitions(node.list[1:], uri, defs)

        elif form in ("import", "from"):
            # Imported names
            for child in node.list[1:]:
                if isinstance(child, Symbol) and child.value not in ("as", "*"):
                    name = child.value
                    loc = lsp.Location(uri=uri, range=_node_range(child))
                    defs.setdefault(name, []).append(loc)
                elif isinstance(child, Bracket):
                    for item in child.list:
                        if isinstance(item, Symbol) and item.value != "as":
                            name = item.value
                            loc = lsp.Location(uri=uri, range=_node_range(item))
                            defs.setdefault(name, []).append(loc)

        else:
            # Recurse into any nested paren forms
            _collect_definitions(
                [c for c in node.list if isinstance(c, Paren)], uri, defs
            )

    return defs


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def _collect_references(
    nodes: list[Node], name: str, uri: str
) -> list[lsp.Location]:
    """Find all occurrences of a symbol name in the AST."""
    refs: list[lsp.Location] = []

    for node in nodes:
        if isinstance(node, Symbol) and node.value == name:
            refs.append(lsp.Location(uri=uri, range=_node_range(node)))

        if isinstance(node, Expression):
            refs.extend(_collect_references(node.list, name, uri))
        elif isinstance(node, (Wrapper, MetaIndicator)) and node.value is not None:
            refs.extend(_collect_references([node.value], name, uri))

    return refs


# ---------------------------------------------------------------------------
# LSP event handlers
# ---------------------------------------------------------------------------


@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params: lsp.DidOpenTextDocumentParams):
    _publish_diagnostics(ls, params.text_document.uri)


@server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: LanguageServer, params: lsp.DidChangeTextDocumentParams):
    _publish_diagnostics(ls, params.text_document.uri)


@server.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: LanguageServer, params: lsp.DidSaveTextDocumentParams):
    _publish_diagnostics(ls, params.text_document.uri)


@server.feature(lsp.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
def document_symbol(
    ls: LanguageServer, params: lsp.DocumentSymbolParams
) -> list[lsp.DocumentSymbol]:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    try:
        tree = parse(doc.source)
    except Exception:
        return []
    return _walk_symbols(tree)


@server.feature(lsp.TEXT_DOCUMENT_HOVER)
def hover(
    ls: LanguageServer, params: lsp.HoverParams
) -> Optional[lsp.Hover]:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    try:
        tree = parse(doc.source)
    except Exception:
        return None

    pos = params.position
    target = _node_at_position(tree, pos.line, pos.character)
    if target is None:
        return None

    # Look up documentation for the symbol
    if isinstance(target, Symbol):
        name = target.value

        # 1. LisPython special forms
        doc_text = SPECIAL_FORM_DOCS.get(name)
        if doc_text:
            return lsp.Hover(
                contents=lsp.MarkupContent(
                    kind=lsp.MarkupKind.Markdown,
                    value=doc_text,
                ),
                range=_node_range(target),
            )

        # 2. Python builtins (try hyphen→underscore too)
        builtin_name = name.replace("-", "_")
        bi = BUILTIN_INFO.get(builtin_name)
        if bi:
            sig = bi["signature"]
            doc_body = bi["doc"]
            hover_md = f"```python\n{sig}\n```\n\n*(Python builtin)*"
            if doc_body:
                hover_md += f"\n\n{doc_body}"
            return lsp.Hover(
                contents=lsp.MarkupContent(
                    kind=lsp.MarkupKind.Markdown,
                    value=hover_md,
                ),
                range=_node_range(target),
            )

    return None


@server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
def definition(
    ls: LanguageServer, params: lsp.DefinitionParams
) -> list[lsp.Location] | None:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    try:
        tree = parse(doc.source)
    except Exception:
        return None

    pos = params.position
    target = _node_at_position(tree, pos.line, pos.character)
    if target is None or not isinstance(target, Symbol):
        return None

    name = target.value
    if name in SPECIAL_FORM_DOCS:
        return None

    # 1. Local definitions in the file
    defs = _collect_definitions(tree, doc.uri)
    local = defs.get(name)
    if local:
        return local

    # 2. Python builtins with source files
    builtin_name = name.replace("-", "_")
    bi = BUILTIN_INFO.get(builtin_name)
    if bi and bi["source_file"]:
        from pathlib import Path
        source_path = Path(bi["source_file"])
        if source_path.exists():
            lineno = (bi["lineno"] or 1) - 1
            return [lsp.Location(
                uri=source_path.as_uri(),
                range=_range(lineno, 0, lineno, 0),
            )]

    return None


@server.feature(lsp.TEXT_DOCUMENT_REFERENCES)
def references(
    ls: LanguageServer, params: lsp.ReferenceParams
) -> list[lsp.Location] | None:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    try:
        tree = parse(doc.source)
    except Exception:
        return None

    pos = params.position
    target = _node_at_position(tree, pos.line, pos.character)
    if target is None or not isinstance(target, Symbol):
        return None

    return _collect_references(tree, target.value, doc.uri)
