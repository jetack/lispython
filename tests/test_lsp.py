"""The LSP keeps working on a buffer that is mid-edit.

Handlers read the buffer with parse(..., partial=True): the complete forms
before the first error, no version gates.  Diagnostics still use the strict
parse, so the error itself is reported.
"""

import importlib
import sys
from types import SimpleNamespace

import pytest

from lispy.core.compiler import features
from lispy.core.parser import parse

# lispy.lsp re-exports the server *object* as ``server``; take the module.
server = importlib.import_module("lispy.lsp.server")

URI = "file:///buffer.lpy"
COMPLETE = "(def foo [] 1)\n(def bar [x] (foo))\n"
BUFFERS = {
    "unclosed form": COMPLETE + "(fo",
    "unterminated f-string": COMPLETE + '(print f"{foo',
    "mismatched bracket": COMPLETE + "(foo]",
    "t-string on an old Python": COMPLETE + '(print t"{(foo)}")',
}


@pytest.fixture(params=sorted(BUFFERS))
def buffer(request, monkeypatch):
    # The t-string case must be refused by the strict parse on every version.
    monkeypatch.setattr(features, "running_version", lambda: (3, 11, 13))
    return BUFFERS[request.param]


def ls_with(source):
    document = SimpleNamespace(source=source, uri=URI)
    workspace = SimpleNamespace(get_text_document=lambda uri: document, folders={})
    return SimpleNamespace(workspace=workspace)


def position(line, character):
    return SimpleNamespace(
        text_document=SimpleNamespace(uri=URI),
        position=SimpleNamespace(line=line, character=character),
    )


def test_strict_parse_refuses_the_buffer(buffer):
    with pytest.raises(Exception):
        parse(buffer)


def test_partial_parse_keeps_the_complete_forms(buffer):
    assert [str(form) for form in parse(buffer, partial=True)][:2] == [
        "(def foo [] 1)",
        "(def bar [x] (foo))",
    ]


def test_document_symbols(buffer):
    symbols = server.document_symbol(ls_with(buffer), position(0, 0))
    assert [s.name for s in symbols] == ["foo", "bar"]


def test_definition_hover_and_references(buffer):
    ls = ls_with(buffer)
    [location] = server.definition(ls, position(1, 14))
    assert location.range.start.line == 0
    refs = server.references(ls, position(0, 5))
    assert [r.range.start.line for r in refs][:2] == [0, 1]
    assert server.hover(ls, position(0, 1)) is not None


def test_completion(buffer):
    items = server.completion(ls_with(buffer), position(2, 2))
    labels = {item.label for item in items.items}
    assert {"foo", "bar"} <= labels


def test_diagnostics_still_report_the_error(buffer):
    [diagnostic] = server.compute_diagnostics(buffer, URI)
    assert diagnostic.message.startswith("Parse error:")
    assert diagnostic.range.start.line == 2


def test_partial_parse_does_not_leak_into_strict_parse(monkeypatch):
    monkeypatch.setattr(features, "running_version", lambda: (3, 11, 13))
    parse(BUFFERS["t-string on an old Python"], partial=True)
    with pytest.raises(Exception):
        parse('(print t"x")')


@pytest.mark.skipif(sys.version_info < (3, 14), reason="t-strings: 3.14+")
def test_partial_parse_of_a_valid_t_string_is_complete():
    forms = parse('(print t"{x}")', partial=True)
    assert [str(f.op) for f in forms] == ["print"]


ALIASES = (
    "(type Alias int)\n"
    "(type Pair[T] (tuple T T))\n"
    "(def use [x] (type Inner x))\n"
    "(type Pair2[T] (tuple T T)\n"  # mid-edit: unclosed
)
# Shapes that call the builtin `type`: one or three operands, a dotted or
# non-symbol first operand.
CALLS = '(type x)\n(type 1 2 3)\n(type a.b c)\n(type "s" c)\n(type)\n'


def test_type_alias_document_symbols():
    symbols = server.document_symbol(ls_with(ALIASES), position(0, 0))
    assert [s.name for s in symbols] == ["Alias", "Pair", "use"]
    assert [c.name for c in symbols[2].children] == ["Inner"]
    # The selection covers the whole head, as it does for def and class.
    assert symbols[1].selection_range.end.character == 13


def test_type_alias_definition():
    # The references come before the unclosed form, which ends the partial parse.
    ls = ls_with(
        "(type Alias int)\n(type Pair[T] (tuple T T))\n(= y Pair)\n(= z Alias)\n(fo"
    )
    [pair] = server.definition(ls, position(2, 5))
    assert (pair.range.start.line, pair.range.start.character) == (1, 6)
    [alias] = server.definition(ls, position(3, 5))
    assert alias.range.start.line == 0


def test_type_alias_hover_on_the_keyword():
    hover = server.hover(ls_with("(type Alias int)"), position(0, 2))
    assert "type alias" in hover.contents.value


def test_type_call_keeps_the_builtin_hover():
    hover = server.hover(ls_with("(type x)\n(type A int)"), position(0, 2))
    assert "Python builtin" in hover.contents.value


def test_type_call_is_not_a_definition():
    ls = ls_with(CALLS)
    assert server.document_symbol(ls, position(0, 0)) == []
    assert server.collect_definitions(parse(CALLS), URI) == {}
