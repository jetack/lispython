"""Python source text of an AST, as ``ast.unparse`` writes it, at any depth.

``ast.unparse`` spends several Python frames per AST level and fails near
the recursion limit, far shallower than compile() takes. ``unparse`` reuses
CPython's own unparser, so the text is the same byte for byte, but cuts the
tree into chunks of at most ``CHUNK_DEPTH`` levels: past that depth a node is
not visited but left in the output as a ``_Hole`` carrying the unparser state
at the cut, and ``_fill`` renders the holes from an explicit stack, each with
a fresh unparser started in that state. The stack depth is bounded by one
chunk, whatever the tree's depth.

Only ``ast.expr``, ``ast.stmt`` and ``ast.pattern`` nodes are cut, because
they always write text: other nodes can write nothing, and the parent may
test what was written (``visit_Lambda`` writes ``lambda :`` for a non-empty
buffer, which a hole always is).

Every chunk of one render shares the precedence table, as the stdlib's one
instance does; a hole restores the precedence its node had at the cut. A
hole renders after the rest of its parent chunk, though, so the one
divergence left is on a node shared by two places (a DAG; ast.parse and
the compiler build trees): if a later place sets its precedence and the
place inside the hole does not, the hole reads the later value where
``ast.unparse`` reads the earlier one.

F-strings recurse outside any chunk, in two ways. The stdlib writes an
interpolation's value through a nested unparser instance (``type(self)()``
and its ``visit``), and a format spec's parts through the part writer
(``features.stdlib_fstring_part_writer``) calling itself, ten-odd frames per
level either way. So ``unparse`` first renders, innermost first and as a
throwaway, every f/t-string literal (a format spec is a JoinedStr too), and
memoizes the text of each value (``visit`` on a fresh instance) and of each
format spec (the part writer on a JoinedStr, keyed by its arguments, which
rendering the spec's parent fills through the stdlib's own call path): each
level then finds the one inside it written. An entry is dropped when it
is read: in a tree each is read once, and keeping them would hold every
level's text at once, quadratic in the depth. A node in two places (a DAG)
is then rendered again, with the same text. The memos are class attributes
of a subclass made per ``unparse`` call, the one way to reach the instances
the stdlib creates without module state; they die with the call.

What still recurses is a chunk: it needs a few hundred frames of room, and a
caller with less gets the RecursionError to report (``ast-to-python`` in
tools.lpy).
"""

import ast
import copy

from lispy.core.compiler import features

CHUNK_DEPTH = 24

_CUT = (ast.expr, ast.stmt, ast.pattern)
_UNSET = object()  # a hole's node had no recorded precedence at the cut
_Stdlib = features.stdlib_unparser()
_PART_WRITER = features.stdlib_fstring_part_writer()


class _Hole:
    """A node left for ``_fill``, with the unparser state to render it in."""

    __slots__ = ("node", "state", "precedence", "written")

    def __init__(self, node, state, precedence, written):
        self.node = node
        self.state = state
        self.precedence = precedence
        self.written = written  # whether anything was written before the cut


class _Chunked(_Stdlib):
    _depth = 0  # traverse levels in this instance's chunk
    _rendering = False  # whether _render has started on this instance
    _interpolations = None  # per-call memos, set on the per-call subclass
    _format_specs = None

    def traverse(self, node):
        if isinstance(node, list):
            for item in node:
                self.traverse(item)
            return
        if self._depth >= CHUNK_DEPTH and isinstance(node, _CUT):
            precedence = self._precedences.get(node, _UNSET)
            hole = _Hole(node, self._state_at(), precedence, bool(self._source))
            self._source.append(hole)
            return
        self._depth += 1
        try:
            super().traverse(node)
        finally:
            self._depth -= 1

    def _state_at(self):
        state = {
            key: copy.copy(value)
            for key, value in self.__dict__.items()
            if key not in ("_source", "_precedences", "_depth")
        }
        state["_precedences"] = self._precedences
        return state

    def visit(self, node):
        if self._rendering:
            # NodeVisitor.generic_visit on an instance in use (a node with no
            # visit_ method, such as Interactive before 3.14): as the stdlib.
            self._source = []
            self.traverse(node)
            text = _fill(type(self), self._source)
            self._source = [text]
            return text
        # A fresh instance: the stdlib's nested one for an interpolation value.
        text = self._interpolations.pop(node, None)
        if text is None:
            text = self._interpolations[node] = _render(self, node)
        return text

    def visit_Interpolation(self, node):
        # The compiler sets Interpolation.str to the lispython text, which is
        # what a Template reports at runtime; the stdlib writes str verbatim
        # unless it is None, and Python source needs the Python text. A
        # shallow copy keeps the caller's tree as it was.
        python = copy.copy(node)
        python.str = None
        super().visit_Interpolation(python)


def _write_part(self, node, *args, **kwargs):
    """The stdlib part writer, its text memoized for a format spec."""
    write = getattr(_Stdlib, _PART_WRITER)
    if not isinstance(node, ast.JoinedStr):
        return write(self, node, *args, **kwargs)
    memo = self._format_specs
    key = (node, args, tuple(sorted(kwargs.items())))
    text = memo.pop(key, None)
    if text is None:
        with self.buffered() as buffer:
            write(self, node, *args, **kwargs)
        text = memo[key] = "".join(buffer)
    self.write(text)


setattr(_Chunked, _PART_WRITER, _write_part)


def _render(unparser, node):
    """The text of ``node`` from a fresh ``unparser``."""
    unparser._rendering = True
    unparser._source = []
    unparser.traverse(node)
    return _fill(type(unparser), unparser._source)


def _fill(cls, source):
    """``source`` joined, its holes rendered without recursion."""
    out = []
    stack = [iter(source)]
    while stack:
        for piece in stack[-1]:
            if isinstance(piece, _Hole):
                child = cls.__new__(cls)
                child.__dict__.update(piece.state)
                # maybe_newline/maybe_semicolon test whether anything was written.
                child._source = [""] if piece.written else []
                if piece.precedence is _UNSET:
                    child._precedences.pop(piece.node, None)
                else:
                    child._precedences[piece.node] = piece.precedence
                child.traverse(piece.node)
                stack.append(iter(child._source))
                break
            out.append(piece)
        else:
            stack.pop()
    return "".join(out)


def _fstrings_innermost_first(tree):
    """Every f/t-string literal in ``tree``, format specs included, each
    before the ones it is inside."""
    literals = tuple(literal for literal, _ in features.fstring_kinds())
    found = []
    pending = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, literals):
            found.append(node)
        pending.extend(ast.iter_child_nodes(node))
    found.reverse()  # every node was found before its descendants
    return found


def unparse(tree):
    """``ast.unparse(tree)``, at any depth compile() takes."""
    cls = type(
        "_Translation", (_Chunked,), {"_interpolations": {}, "_format_specs": {}}
    )
    for literal in _fstrings_innermost_first(tree):
        _render(cls(), literal)
    return _render(cls(), tree)
