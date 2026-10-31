"""Python source text of an AST, as ``ast.unparse`` writes it, at any depth.

``ast.unparse`` spends several Python frames per AST level and fails near
the recursion limit, far shallower than compile() takes. ``unparse`` reuses
CPython's own unparser, so the text is the same byte for byte, but cuts the
tree into chunks of at most ``CHUNK_DEPTH`` levels: past that depth a node is
not visited but left in the output as a ``_Hole`` carrying the unparser state
at the cut, and ``_render`` fills the holes from an explicit stack, each with
a fresh unparser started in that state. The stack depth is bounded by one
chunk, whatever the tree's depth.

Only ``ast.expr`` and ``ast.stmt`` nodes are cut, because they always write
text: other nodes can write nothing, and the parent may test what was
written (``visit_Lambda`` writes ``lambda :`` for a non-empty buffer, which a
hole always is).

The stdlib writes an f/t-string interpolation's value through a nested
unparser instance (``type(self)()`` and its ``visit``), ten-odd frames per
nested f-string that no chunk counter sees. So ``unparse`` first renders
every f/t-string innermost first, as a throwaway, and ``visit`` memoizes each
value's text: the outer levels then find their values written. The memo is a
class attribute of a subclass made per ``unparse`` call, the one way to reach
the instances the stdlib creates without module state; it dies with the call.

What still recurses: the stdlib writes an f-string nested in a format spec
within one instance, and a chunk needs a few hundred frames of room. The
RecursionError is then the caller's to report (``ast-to-python`` in
tools.lpy).
"""

import ast
import copy

from lispy.core.compiler import features

CHUNK_DEPTH = 24


class _Hole:
    """A node left for ``_render``, with the unparser state to render it in."""

    __slots__ = ("node", "state")

    def __init__(self, node, state):
        self.node = node
        self.state = state


class _Chunked(features.stdlib_unparser()):
    _depth = 0  # traverse levels in this instance's chunk
    _interpolations = None  # the per-call memo, set on the per-call subclass

    def traverse(self, node):
        if isinstance(node, list):
            for item in node:
                self.traverse(item)
            return
        if self._depth >= CHUNK_DEPTH and isinstance(node, (ast.expr, ast.stmt)):
            self._source.append(_Hole(node, self._state_at(node)))
            return
        self._depth += 1
        try:
            super().traverse(node)
        finally:
            self._depth -= 1

    def _state_at(self, node):
        state = {
            key: copy.copy(value)
            for key, value in self.__dict__.items()
            if key not in ("_source", "_precedences", "_depth")
        }
        # maybe_newline/maybe_semicolon test whether anything was written.
        state["_source"] = [""] if self._source else []
        state["_precedences"] = (
            {node: self._precedences[node]} if node in self._precedences else {}
        )
        return state

    def visit(self, node):
        memo = self._interpolations
        if node not in memo:
            self._source = []
            memo[node] = _render(self, node)
        return memo[node]

    def visit_Interpolation(self, node):
        # The compiler sets Interpolation.str to the lispython text, which is
        # what a Template reports at runtime; the stdlib writes str verbatim
        # unless it is None, and Python source needs the Python text. A
        # shallow copy keeps the caller's tree as it was.
        python = copy.copy(node)
        python.str = None
        super().visit_Interpolation(python)


def _render(unparser, node):
    """The text of ``node`` from ``unparser``, filling holes without recursion."""
    cls = type(unparser)
    unparser.traverse(node)
    out = []
    stack = [iter(unparser._source)]
    while stack:
        for piece in stack[-1]:
            if isinstance(piece, _Hole):
                child = cls.__new__(cls)
                child.__dict__.update(piece.state)
                child.traverse(piece.node)
                stack.append(iter(child._source))
                break
            out.append(piece)
        else:
            stack.pop()
    return "".join(out)


def _fstrings_innermost_first(tree):
    found = []
    pending = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, features.fstring_classes()):
            found.append(node)
        pending.extend(ast.iter_child_nodes(node))
    found.reverse()  # every node was found before its descendants
    return found


def unparse(tree):
    """``ast.unparse(tree)``, at any depth compile() takes."""
    cls = type("_Translation", (_Chunked,), {"_interpolations": {}})
    for fstring in _fstrings_innermost_first(tree):
        _render(cls(), fstring)
    return _render(cls(), tree)
