"""Editor features beyond the parse-tolerance contract in test_lsp.py:
signatureHelp, workspace/symbol, nREPL completion kinds and .nrepl-port."""

import importlib
from types import SimpleNamespace

import pytest
from pygls.workspace import TextDocument

import lispy.tools as tools

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


WORKSPACE_FILE = """\
(import os)
(from os.path [join :as pjoin])
(def load-config [path] (for line in [1] (pass)) 1)
(async-def fetch [] None)
(defmacro unless-config [x] x)
(class ConfigError [Exception]
  (def describe [self] "x"))
(= config-path "a")
(async-for item in (fetch) (pass))
"""


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    for name in ["WORKSPACE_INDEX", "WORKSPACE_SYMBOLS", "FILE_TREES"]:
        monkeypatch.setattr(server, name, {}, raising=False)
    path = tmp_path / "config.lpy"
    path.write_text(WORKSPACE_FILE)
    uri = path.as_uri()
    server.index_file(uri)
    return uri


def symbols_matching(query):
    result = server.workspace_symbol(None, SimpleNamespace(query=query))
    return {s.name: s.kind for s in result}


def test_workspace_symbol_reports_each_definition_with_its_kind(workspace):
    kind = server.lsp.SymbolKind
    assert symbols_matching("") == {
        "load-config": kind.Function,
        "fetch": kind.Function,
        "unless-config": kind.Function,
        "ConfigError": kind.Class,
        "describe": kind.Function,
        "config-path": kind.Variable,
    }


def test_workspace_symbol_matches_a_case_insensitive_substring(workspace):
    assert set(symbols_matching("CONFIG")) == {
        "load-config",
        "unless-config",
        "ConfigError",
        "config-path",
    }


def test_workspace_symbol_points_at_the_definition(workspace):
    [symbol] = server.workspace_symbol(None, SimpleNamespace(query="fetch"))
    assert symbol.location.uri == workspace
    assert symbol.location.range.start.line == 3


def test_imports_and_loop_variables_stay_in_the_definition_index(workspace):
    # go-to-definition and completion still see every bound name.
    assert {"os", "pjoin", "line", "item"} <= set(server.WORKSPACE_INDEX)
    assert not {"os", "pjoin", "line", "item"} & set(symbols_matching(""))


def test_reindexing_a_file_replaces_its_symbols(workspace, tmp_path):
    (tmp_path / "config.lpy").write_text("(def renamed [] 1)\n")
    server.index_file(workspace)
    assert set(symbols_matching("")) == {"renamed"}
    assert "load-config" not in server.WORKSPACE_INDEX


@pytest.fixture
def scope():
    scope = {"__name__": "__nrepl_test__", "__macro_namespace": {}}
    result = tools.nrepl_eval(
        "(require lispy.macros *)\n"
        "(import os)\n"
        "(= my-var 1)\n"
        "(def my-fn [] 1)\n"
        "(class MyClass [] (pass))",
        scope,
    )
    assert result["error"] is None
    return scope


def complete(prefix, scope):
    result = tools.nrepl_complete(prefix, scope)
    assert set(result["kinds"]) == set(result["completions"])
    return result["kinds"]


@pytest.mark.parametrize(
    ["prefix", "name", "kind"],
    [
        ["in", "int", "class"],
        ["le", "len", "function"],
        ["con", "cond", "macro"],
        ["de", "def", "keyword"],
        ["whi", "while", "keyword"],
        ["my", "my-var", "variable"],
        ["my", "my-fn", "function"],
        ["My", "MyClass", "class"],
        ["os.pa", "os.path", "variable"],
        ["os.pa", "os.pathsep", "variable"],
        ["os.ge", "os.getcwd", "function"],
    ],
)
def test_completion_kinds(scope, prefix, name, kind):
    assert complete(prefix, scope)[name] == kind


class Gadget:
    size = 3

    @property
    def broken(self):
        raise RuntimeError("property failed")

    def build(self):
        pass


def test_attribute_completion_survives_a_raising_property(scope):
    scope["gadget"] = Gadget()
    assert complete("gadget.b", scope) == {
        "gadget.broken": "variable",
        "gadget.build": "function",
    }
    assert complete("gadget.si", scope) == {"gadget.size": "variable"}
