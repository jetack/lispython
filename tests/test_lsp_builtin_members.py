"""Hover and signature help on a member of a builtin-typed value.

Acceptance tests for TASK-1021 (the LSP's Python features served by
basedpyright): ``x.sort``, ``(. x sort)`` and ``(.sort x)`` on a list show
``list.sort``'s parameters, as pyright and jedi do for Python.  Expected
parameters are read from CPython (``inspect.signature``), never from the
server, so the tests hold for any backend that reports the real signature.
They are strict xfails until the backend lands: remove the marker then.

``|`` in a source marks the cursor; it is removed first.
"""

import importlib
import inspect
from types import SimpleNamespace

import pytest
from pygls.workspace import TextDocument

# lispy.lsp re-exports the server *object* as ``server``; take the module.
server = importlib.import_module("lispy.lsp.server")

URI = "file:///buffer.lpy"

pytestmark = pytest.mark.xfail(
    strict=True, reason="TASK-1021: builtin members need the basedpyright backend"
)


def bound_parameters(cls, member):
    """The parameter names a call through an instance takes (self dropped)."""
    parameters = list(inspect.signature(getattr(cls, member)).parameters)
    return parameters[1:]


def request(marked):
    [before, after] = marked.split("|")
    lines = before.split("\n")
    document = TextDocument(URI, before + after)
    workspace = SimpleNamespace(get_text_document=lambda uri: document, folders={})
    pos = SimpleNamespace(
        line=len(lines) - 1, character=len(lines[-1].encode("utf-16-le")) // 2
    )
    params = SimpleNamespace(text_document=SimpleNamespace(uri=URI), position=pos)
    return SimpleNamespace(workspace=workspace), params


def hover_text(marked):
    result = server.hover(*request(marked))
    return None if result is None else result.contents.value


def signature_labels(marked):
    """Every signature offered: overload order differs between stub versions."""
    result = server.signature_help(*request(marked))
    return [] if result is None else [sig.label for sig in result.signatures]


def shows_any(labels, cls, member):
    errors = []
    for label in labels:
        try:
            assert_shows(label, cls, member)
            return
        except (AssertionError, ValueError) as err:
            errors.append(f"{label!r}: {err!r}")
    raise AssertionError(f"no signature shows {cls.__name__}.{member}: {errors}")


def assert_shows(text, cls, member):
    assert text is not None
    for name in bound_parameters(cls, member):
        assert name in text
    # The sorted builtin also takes key and reverse (after an iterable): a
    # server resolving .sort to it must not pass.
    assert "iterable" not in text
    assert "sorted" not in text
    # list.sort's parameters are keyword-only; a positional rendering would
    # be the wrong signature.
    if (cls, member) == (list, "sort"):
        assert text.index("*") < text.index("key")


# Each receiver form, with the list type coming from each source a type
# checker reads: a literal, an annotation, an assignment, a constructor call.
HOVERS = [
    "(= x [1])\n(print x.so|rt)",
    "(= x [1])\n(print (. x so|rt))",
    "(= x [1])\n(print (.so|rt x))",
    "(print (.so|rt [3 1]))",
    "(def f [x ^list] (.so|rt x))",
    "(def f [x ^list[int]] (.so|rt x))",
    "(= x (list))\n(print x.so|rt)",
]

SIGNATURES = [
    "(= x [1])\n(x.sort |)",
    "(= x [1])\n((. x sort) |)",
    "(= x [1])\n(.sort x |)",
    "(.sort [3 1] |)",
    "(def f [x ^list] (.sort x |))",
    "(def f [x ^list[int]] (.sort x |))",
    "(= x (list))\n(.sort x |)",
    # Mid-edit: the call is still open.
    "(= x [1])\n(.sort x :key |",
    # The innermost call is the method, not the enclosing print.
    "(= x [1])\n(print ((. x sort) |))",
]


@pytest.mark.parametrize("marked", HOVERS)
def test_hover_on_a_list_method_shows_its_parameters(marked):
    text = hover_text(marked)
    assert_shows(text, list, "sort")
    assert "sort" in text


@pytest.mark.parametrize("marked", SIGNATURES)
def test_signature_help_in_a_list_method_call_shows_its_parameters(marked):
    shows_any(signature_labels(marked), list, "sort")


def test_another_type_and_method():
    marked = '(= d {"a" 1})\n(.get d |)'
    shows_any(signature_labels(marked), dict, "get")
