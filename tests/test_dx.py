"""Editor features beyond the parse-tolerance contract in test_lsp.py:
signatureHelp, workspace/symbol, nREPL completion kinds and .nrepl-port."""

import functools
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
        ["(print '(len |", PRINT],
        ["(print `(len |", PRINT],
        ["'(len |", None],
        ["`(len |", None],
        ["(print `(foo ~(len |", LEN],
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


def test_document_and_workspace_symbols_agree_on_kinds(workspace):
    document = SimpleNamespace(source=WORKSPACE_FILE, uri=workspace)
    params = SimpleNamespace(text_document=SimpleNamespace(uri=workspace))
    document_kinds = {
        s.name: s.kind
        for s in server.document_symbol(ls_with(document), params)
        if s.kind != server.lsp.SymbolKind.Module
    }
    workspace_kinds = symbols_matching("")
    assert document_kinds == {
        name: kind for name, kind in workspace_kinds.items() if name in document_kinds
    }
    assert set(document_kinds) == set(workspace_kinds) - {"describe"}


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


def test_a_deleted_file_leaves_no_symbols(workspace, tmp_path):
    (tmp_path / "config.lpy").unlink()
    server.index_file(workspace)
    assert symbols_matching("") == {}
    assert server.WORKSPACE_INDEX == {}


SCOPES_FILE = """\
(def handler []
  (= handler-local 1)
  (when True (= nested-local 2)))
(defmacro expander [x] (= macro-local 1) x)
(class Settings []
  (= class-attr 1)
  (def method [self] (= method-local 1)))
(when True (= guarded-global 1))
"""


def test_locals_inside_def_bodies_are_not_workspace_symbols(workspace, tmp_path):
    path = tmp_path / "scopes.lpy"
    path.write_text(SCOPES_FILE)
    server.index_file(path.as_uri())
    local_names = {"handler-local", "nested-local", "macro-local", "method-local"}
    # go-to-definition still finds them ...
    assert local_names <= set(server.WORKSPACE_INDEX)
    # ... but only module and class-body assignments are searchable.
    symbols = symbols_matching("")
    assert not local_names & set(symbols)
    kind = server.lsp.SymbolKind
    assert symbols["class-attr"] == kind.Variable
    assert symbols["guarded-global"] == kind.Variable
    assert symbols["method"] == kind.Function


def test_required_packages_are_not_workspace_symbols(workspace, tmp_path, monkeypatch):
    site = tmp_path / "site"
    (site / "dx_fake_package").mkdir(parents=True)
    (site / "dx_fake_package" / "__init__.py").write_text("")
    (site / "dx_fake_package" / "macros.lpy").write_text(
        "(defmacro package-macro [x] x)\n"
    )
    monkeypatch.syspath_prepend(str(site))
    user = tmp_path / "user.lpy"
    user.write_text("(require dx_fake_package)\n(def user-fn [] 1)\n")
    server.index_file(user.as_uri())
    # go-to-definition can reach the package's macro ...
    assert "package-macro" in server.WORKSPACE_INDEX
    # ... but the symbol search lists only the user's code.
    assert "package-macro" not in symbols_matching("")
    assert "user-fn" in symbols_matching("")


def test_definition_publishes_symbols_only_for_workspace_files(
    workspace, tmp_path, monkeypatch
):
    ws = tmp_path / "ws"
    # A sibling whose name extends the workspace's: not inside it.
    site = tmp_path / "ws-site"
    ws.mkdir()
    site.mkdir()
    (ws / "dx_ws_module.lpy").write_text("(def ws-helper [] 1)\n")
    (site / "dx_site_module.lpy").write_text("(def site-helper [] 1)\n")
    monkeypatch.syspath_prepend(str(site))
    folder = SimpleNamespace(uri=ws.as_uri())

    def go_to(module, name):
        # A name brought in by (require ...) is resolved through the module
        # file, which definition indexes on first use.
        main = ws / "main.lpy"
        document = SimpleNamespace(
            source=f"(require {module})\n({name})\n", uri=main.as_uri()
        )
        ls = SimpleNamespace(
            workspace=SimpleNamespace(
                get_text_document=lambda uri: document, folders={folder.uri: folder}
            )
        )
        params = SimpleNamespace(
            text_document=SimpleNamespace(uri=main.as_uri()),
            position=SimpleNamespace(line=1, character=2),
        )
        [location] = server.definition(ls, params)
        return location.uri

    assert go_to("dx_ws_module", "ws-helper") == (ws / "dx_ws_module.lpy").as_uri()
    assert (
        go_to("dx_site_module", "site-helper") == (site / "dx_site_module.lpy").as_uri()
    )
    symbols = symbols_matching("helper")
    assert "ws-helper" in symbols
    assert "site-helper" not in symbols


def test_under_workspace_compares_path_components(tmp_path):
    folder = SimpleNamespace(uri=(tmp_path / "ws").as_uri())
    assert server.under_workspace(str(tmp_path / "ws" / "a.lpy"), [folder])
    assert not server.under_workspace(str(tmp_path / "ws-site" / "a.lpy"), [folder])
    assert not server.under_workspace(str(tmp_path / "ws" / "a.lpy"), [])


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
        ["if", "ife", "keyword"],
        ["su", "sub", "keyword"],
        ["yield", "yield-from", "keyword"],
        ["req", "require", "keyword"],
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


@pytest.mark.parametrize(
    ["prefix", "name"],
    [["->", "->"], ["->", "->>"], ["as", "as->"], ["gen", "gensym"]],
)
def test_syntax_without_its_macros_is_a_keyword(prefix, name):
    assert complete(prefix, {"__macro_namespace": {}})[name] == "keyword"


def test_a_scope_binding_shadows_a_builtin(scope):
    assert tools.nrepl_eval("(= list 3)", scope)["error"] is None
    assert complete("lis", scope)["list"] == "variable"


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


class Counted:
    reads = 0

    @property
    def counted(self):
        Counted.reads += 1
        return len

    @functools.cached_property
    def cached(self):
        Counted.reads += 1
        return len

    @staticmethod
    def static_helper():
        pass

    @classmethod
    def class_helper(cls):
        pass

    def method(self):
        pass

    Inner = int


def test_attribute_completion_runs_no_property_code(scope):
    scope["counted"] = Counted()
    kinds = complete("counted.", scope)
    assert {
        name: kinds[f"counted.{name}"]
        for name in [
            "counted",
            "cached",
            "static-helper",
            "class-helper",
            "method",
            "Inner",
            "reads",
        ]
    } == {
        "counted": "variable",
        "cached": "variable",
        "static-helper": "function",
        "class-helper": "function",
        "method": "function",
        "Inner": "class",
        "reads": "variable",
    }
    assert Counted.reads == 0


def test_attribute_kinds_of_builtin_types(scope):
    scope["text"] = "abc"
    assert complete("text.upp", scope) == {"text.upper": "function"}
    assert complete("int.from", scope) == {"int.from-bytes": "function"}


def test_port_file_is_written_and_removed(tmp_path):
    path = tools._write_port_file(7888, str(tmp_path))
    assert (tmp_path / ".nrepl-port").read_text() == "7888"
    tools._remove_port_file(path, 7888)
    assert not (tmp_path / ".nrepl-port").exists()


def test_unwritable_directory_gets_no_port_file(tmp_path, capsys):
    assert tools._write_port_file(7888, str(tmp_path / "missing")) is None
    assert ".nrepl-port" in capsys.readouterr().err
    tools._remove_port_file(None, 7888)


def test_port_file_of_another_server_is_left_alone(tmp_path):
    path = tools._write_port_file(7888, str(tmp_path))
    # A second server in the same directory took the file over.
    tools._write_port_file(7999, str(tmp_path))
    tools._remove_port_file(path, 7888)
    assert (tmp_path / ".nrepl-port").read_text() == "7999"


def test_removing_a_vanished_port_file_does_not_raise(tmp_path):
    path = tools._write_port_file(7888, str(tmp_path))
    (tmp_path / ".nrepl-port").unlink()
    tools._remove_port_file(path, 7888)
