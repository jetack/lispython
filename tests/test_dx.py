"""Editor features beyond the parse-tolerance contract in test_lsp.py:
signatureHelp, workspace/symbol, nREPL completion kinds and .nrepl-port."""

import importlib
from types import SimpleNamespace

import pytest
from pygls.workspace import TextDocument

# lispy.lsp re-exports the server *object* as ``server``; take the module.
server = importlib.import_module("lispy.lsp.server")

URI = "file:///buffer.lpy"
CURSOR = "|"


def ls_with(document):
    workspace = SimpleNamespace(get_text_document=lambda uri: document, folders={})
    return SimpleNamespace(workspace=workspace)


def signature_at(marked, pygls_document=False):
    """Signature label at the "|" in ``marked`` (removed from the source).

    With ``pygls_document`` the buffer is a real pygls TextDocument and the
    position is sent as a client would: in UTF-16 code units."""
    [before, after] = marked.split(CURSOR)
    source = before + after
    lines = before.split("\n")
    if pygls_document:
        document = TextDocument(URI, source)
        character = len(lines[-1].encode("utf-16-le")) // 2
    else:
        document = SimpleNamespace(source=source, uri=URI)
        character = len(lines[-1])
    pos = SimpleNamespace(line=len(lines) - 1, character=character)
    params = SimpleNamespace(text_document=SimpleNamespace(uri=URI), position=pos)
    result = server.signature_help(ls_with(document), params)
    return None if result is None else result.signatures[0].label


LEN = server.BUILTIN_INFO["len"]["signature"]
PRINT = server.BUILTIN_INFO["print"]["signature"]


@pytest.mark.parametrize(
    ["marked", "expected"],
    [
        ["(print (len |", LEN],
        ["(print (len x) |", PRINT],
        ["(print [1 (len |", LEN],
        ["(print [1 |", PRINT],
        ["(len {1 2 |", LEN],
        ['(print "(len |', PRINT],
        ['(print "(len" |', PRINT],
        ["(print ; (len\n  |", PRINT],
        ['(print r"\\\\" (len |', LEN],
        ['(print "\\"(" |', PRINT],
        ['(print r"(" |', PRINT],
        ['(print """a\n(len" """ |', PRINT],
        ['(print """a " (len""" |', PRINT],
        ["(print '''(len''' |", PRINT],
        ["(print b'''(len''' (len |", LEN],
        ['(print f"{(len x)}" |', PRINT],
        ["(print *(len |", LEN],
        ["(len (print x)\n  (print y)\n  |", LEN],
        ["|(print x)", None],
        ["", None],
        ["(no-such-function |", None],
        ["(print (no-such-function |", None],
        ["(print ((fn [x] x) |", PRINT],
    ],
)
def test_signature_help(marked, expected):
    assert signature_at(marked or CURSOR) == expected


def test_signature_help_of_a_special_form_is_its_documented_form():
    assert signature_at("(ife x |") == "(ife condition then-expr else-expr)"


@pytest.mark.parametrize(
    ["marked", "expected"],
    [
        ['(print "\U0001f600" (len x|))', LEN],
        ['(print "\U0001f600" (len x)|)', PRINT],
        ['(print "\U0001f600")\n(print (len x|))', LEN],
    ],
)
def test_signature_help_reads_the_client_position_as_utf16(marked, expected):
    # The emoji is one code point but two UTF-16 units, so a client column
    # after it is one past the str column.
    assert signature_at(marked, pygls_document=True) == expected


@pytest.mark.parametrize(
    "source",
    ["(", ")", "(f]", '(f "abc', '(f f"{', '(f f"{(g', "(f '", "((((", "]]]"],
)
def test_signature_help_never_raises_on_a_mid_edit_buffer(source):
    for col in range(len(source) + 1):
        signature_at(source[:col] + CURSOR + source[col:])
