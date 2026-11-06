"""The Python backend's diagnostics in the LSP: mapped onto the .lpy buffer,
gated to the full view's text, merged with lispython's own.

Mapping positions are read off the source and generated text with
``locate`` (1-based line, code-point column), so every assertion names the
word it lands on.  The server-flow tests run the fake backend of
test_lsp_backend, which publishes an Error on each ``undefined_name``; the
last test runs the session's real basedpyright.
"""

import asyncio
import importlib
import itertools
import re
import sys
import time
from types import SimpleNamespace

import pytest
from lsprotocol import types as lsp
from pygls.workspace import TextDocument

import lispy  # noqa: F401  (registers the .lpy importer)
import lispy.core.macro
from tests.test_lsp_backend import FAKE_SERVER, hover_params, run_bounded

backend = importlib.import_module("lispy.lsp.backend")
pv = importlib.import_module("lispy.lsp.python_view")
server = importlib.import_module("lispy.lsp.server")


def locate(text, word, nth=0):
    index = -1
    for _ in range(nth + 1):
        index = text.index(word, index + 1)
    return text.count("\n", 0, index) + 1, index - (text.rfind("\n", 0, index) + 1)


def span(text, word, nth=0):
    """The one-line (start, end) of the nth occurrence of word."""
    line, col = locate(text, word, nth)
    return (line, col), (line, col + len(word))


def mapped(src, word, nth=0, tagged=False):
    """lpy-diagnostic-range of the generated range of the nth match of word
    (a string, or a compiled pattern) in src's view."""
    view = pv.compile_view(src)
    assert view is not None
    pattern = word if isinstance(word, re.Pattern) else re.compile(re.escape(word))
    match = list(pattern.finditer(view.text))[nth]
    start, end = span(
        view.text, match.group(), view.text.count(match.group(), 0, match.start())
    )
    return pv.lpy_diagnostic_range(view, src, start, end, tagged)


# ---------------------------------------------------------------------------
# lpy-diagnostic-range
# ---------------------------------------------------------------------------


def test_a_name_maps_onto_its_spelling():
    # Defect (step a): an undefined-variable error lands on the whole call,
    # or on the generated spelling's columns, not on the .lpy word.
    src = "(print undefined-name)"
    assert mapped(src, "undefined_name") == span(src, "undefined-name")


def test_a_range_that_is_exactly_a_node_maps_onto_it():
    # Defect (step b): an extra call argument (reportCallIssue's range is
    # the Constant) and an unreachable statement (reportUnreachable's is the
    # whole statement, a hint) map nowhere and are dropped.
    src = "(def f [a b] (return a))\n(f 1 2 3)"
    assert mapped(src, "3") == span(src, "3")
    src = "(def f []\n  (return 1)\n  (print 2))"
    assert mapped(src, "print(2)", tagged=True) == span(src, "(print 2)")


AS_THREAD = "(= r (as-> 5 x (len x) (+ x 1)))"

# as->'s gensym, whose number depends on what compiled before.
GENSYM = re.compile(r"__x_\d+")


def test_an_error_in_macro_output_lands_on_the_macro_call():
    # Defect (step c): an error on code a macro generated (here its gensym,
    # which the source never spells) is dropped, or moved off the call.
    start, end = mapped(AS_THREAD, GENSYM, nth=3)
    call_start, call_end = span(AS_THREAD, "(as-> 5 x (len x) (+ x 1))")
    assert start == call_start and end <= call_end


def test_a_hint_maps_only_onto_what_it_marks():
    # Defects: an unused-gensym hint greys the macro call that made it; a
    # hint whose range is no node greys its whole enclosing form.
    assert mapped(AS_THREAD, GENSYM, nth=3, tagged=True) is None
    src = "(-> 5 (len) (print))"
    assert mapped(src, "(5)", tagged=False) is not None
    assert mapped(src, "(5)", tagged=True) is None


def test_plumbing_maps_nowhere():
    # Defect: a diagnostic on code no form wrote (a require's or a
    # defmacro's generated statements, the module's start) is moved onto
    # line 1 or some form rather than dropped.
    src = "(require lispy.macros [when])\n(when True (print 1))"
    assert mapped(src, "___imported_macros", nth=1) is None
    assert mapped(src, "___imported_macros", nth=1, tagged=True) is None
    src = "(defmacro twice [a] (return `(do ~a ~a)))\n(twice (print 1))"
    assert mapped(src, "___defmacro_temp___twice") is None
    view = pv.compile_view(src)
    assert pv.lpy_diagnostic_range(view, src, (1, 0), (1, 0), False) is None


# ---------------------------------------------------------------------------
# Messages in the buffer's spelling
# ---------------------------------------------------------------------------


def test_messages_quote_names_as_the_buffer_spells_them():
    # Defects: a message names the Python spelling the user never wrote; a
    # name the buffer spells two ways is guessed; quoted text that is no
    # name (a type) is rewritten.
    src = "(= my-x 1)\n(= other-y my-x)\n(= amb-z 1)\n(= amb_z 2)"
    spellings = server.buffer_name_spellings(src)
    rename = server.message_to_buffer
    assert (
        rename('"undefined_name" is not defined', "undefined-name", spellings)
        == '"undefined-name" is not defined'
    )
    assert (
        rename('Variable "other_y" is not accessed', None, spellings)
        == 'Variable "other-y" is not accessed'
    )
    assert rename('"amb_z" is unbound', None, spellings) == '"amb_z" is unbound'
    literal = 'Type "Literal[\'s\']" is not assignable to "my_x"'
    assert rename(literal, None, spellings) == literal.replace("my_x", "my-x")


# ---------------------------------------------------------------------------
# Only a full view's diagnostics
# ---------------------------------------------------------------------------


def published(uri, version, diagnostics):
    return lsp.PublishDiagnosticsParams(
        uri=uri, version=version, diagnostics=diagnostics
    )


def test_only_the_full_views_current_version_is_kept(tmp_path):
    # Defects: diagnostics computed over a request's cursor view (text the
    # user does not see, positions of another text) or over an older
    # version are kept and later mapped onto the buffer.
    project = tmp_path / "project"
    project.mkdir()
    client = backend.BackendClient(
        pv.view_root_for(str(project), cache_home=str(tmp_path / "cache")), None
    )
    process = object()
    client._client = process
    uri = (project / "main.lpy").as_uri()
    shadow = client.backend_uri(uri)
    heard = []
    client.on_diagnostics = heard.append
    full, cursor = "print(undefined_name)", "print(undefined)"
    client._full[uri] = (full, ("src", None))
    diagnostic = lsp.Diagnostic(
        range=lsp.Range(
            start=lsp.Position(line=0, character=6),
            end=lsp.Position(line=0, character=20),
        ),
        message="m",
    )
    # A request synced its cursor view as version 3.
    client._docs[uri] = [shadow, 3, cursor, ("src", None)]
    client._diagnostics_published(process, published(shadow, 3, [diagnostic]))
    assert client.diagnostics_for(uri, full) is None
    assert client.diagnostics_for(uri, cursor) is None
    # The full view is back as version 4; a late publish of 3 is stale.
    client._docs[uri] = [shadow, 4, full, ("src", None)]
    client._diagnostics_published(process, published(shadow, 3, [diagnostic]))
    assert client.diagnostics_for(uri, full) is None
    assert heard == []
    client._diagnostics_published(process, published(shadow, 4, [diagnostic]))
    assert client.diagnostics_for(uri, full) == [diagnostic]
    assert heard == [uri]


# ---------------------------------------------------------------------------
# The server's flow, over the fake backend
# ---------------------------------------------------------------------------


class Editor:
    """A language server double for one document: its workspace holds the
    buffer, and publishes are recorded."""

    def __init__(self, uri):
        self.uri = uri
        self.workspace = self
        self.document = None
        self.published = []
        self.versions = itertools.count(1)

    def get_text_document(self, uri):
        return self.document

    def text_document_publish_diagnostics(self, params):
        self.published.append(params.diagnostics)

    def edit(self, text, handler):
        self.document = TextDocument(self.uri, text, version=next(self.versions))
        doc = SimpleNamespace(uri=self.uri, version=self.document.version)
        handler(self, SimpleNamespace(text_document=doc))

    def open(self, text):
        self.edit(text, server.did_open)

    def change(self, text):
        self.edit(text, server.did_change)


async def until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


@pytest.fixture
def fake_served(tmp_path, monkeypatch):
    """(editor, backends) over a project whose backend is the fake, as
    basedpyright, with the server's state fresh."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.lpy").write_text("(= y 1)\n")
    script = tmp_path / "fake_backend.py"
    script.write_text(FAKE_SERVER)
    monkeypatch.setitem(
        backend.BACKENDS, "basedpyright", lambda: [sys.executable, str(script)]
    )
    folder = SimpleNamespace(uri=project.as_uri())
    backends = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "cache"))
    backends.configure(None)
    backends.on_diagnostics = server.backend_diagnostics_arrived
    monkeypatch.setattr(server, "backends", backends)
    monkeypatch.setattr(server, "view_cache", pv.ViewCache())
    monkeypatch.setattr(server, "DIAGNOSTIC_RUNS", {})
    for name in ["WORKSPACE_INDEX", "WORKSPACE_SYMBOLS", "FILE_TREES"]:
        monkeypatch.setattr(server, name, {})
    return Editor((project / "main.lpy").as_uri()), backends


def backend_diagnostics(diagnostics):
    return [d for d in diagnostics if d.source == "basedpyright"]


def test_backend_diagnostics_reach_the_buffer_and_a_compile_error_wins(fake_served):
    # Defects: the backend's diagnostics are dropped, never started for on
    # open, or published at generated positions in Python spelling with its
    # data; and a buffer that no longer parses still shows the backend's
    # set for its last good text, under the parse error.
    editor, backends = fake_served

    async def scenario():
        try:
            editor.open("(print undefined-name)\n")
            await until(
                lambda: editor.published and backend_diagnostics(editor.published[-1])
            )
            [found] = backend_diagnostics(editor.published[-1])
            assert found.message == '"undefined-name" is not defined'
            assert found.code == "reportUndefinedVariable"
            assert found.severity == lsp.DiagnosticSeverity.Error
            assert found.data is None
            assert (found.range.start.line, found.range.start.character) == (0, 7)
            assert (found.range.end.line, found.range.end.character) == (0, 21)
            count = len(editor.published)
            editor.change("(print undefined-name\n")
            await until(lambda: len(editor.published) > count)
            await asyncio.sleep(0.2)  # a late republish would come by now
            [diagnostic] = editor.published[-1]
            assert diagnostic.message.startswith("Parse error:")
        finally:
            await backends.shutdown_all()

    run_bounded(scenario())


def test_a_change_compiles_once(fake_served, monkeypatch):
    # Defect: a change compiles the buffer twice, once for lispython's
    # diagnostics and again for the backend's view, running every macro
    # twice per keystroke.
    editor, backends = fake_served
    calls = []

    def counting(real):
        def compile_unit(*args, **kwargs):
            calls.append(True)
            return real(*args, **kwargs)

        return compile_unit

    monkeypatch.setattr(pv, "compile_unit", counting(pv.compile_unit))
    monkeypatch.setattr(
        lispy.core.macro, "compile_unit", counting(lispy.core.macro.compile_unit)
    )
    uri = editor.uri
    client = backends.client_for(uri)

    async def scenario():
        try:
            params = hover_params(client.backend_uri(uri))
            assert (
                await backends.request(uri, lsp.TEXT_DOCUMENT_HOVER, params) is not None
            )
            calls.clear()
            editor.change("(= my-z 2)\n")
            await until(
                lambda: (
                    editor.published
                    and uri in client._docs
                    and client._docs[uri][2] == "my_z = 2"
                )
            )
        finally:
            await backends.shutdown_all()

    run_bounded(scenario())
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Real basedpyright
# ---------------------------------------------------------------------------


def test_real_basedpyright_reports_on_the_lpy_word(
    run_on_loop, real_backends, backend_project, monkeypatch
):
    # Defects, end to end: basedpyright's diagnostics never reach the
    # buffer, land on generated or prelude code, or keep Python spelling.
    if real_backends.available_client((backend_project / "main.lpy").as_uri()) is None:
        pytest.skip("basedpyright is not installed (the lsp extra)")
    argv = backend.backend_argv("basedpyright")
    if argv is None:
        pytest.skip("basedpyright is not installed (the lsp extra)")
    monkeypatch.setattr(
        real_backends, "on_diagnostics", server.backend_diagnostics_arrived
    )
    monkeypatch.setattr(server, "backends", real_backends)
    monkeypatch.setattr(server, "view_cache", pv.ViewCache())
    monkeypatch.setattr(server, "DIAGNOSTIC_RUNS", {})
    for name in ["WORKSPACE_INDEX", "WORKSPACE_SYMBOLS", "FILE_TREES"]:
        monkeypatch.setattr(server, name, {})
    editor = Editor((backend_project / "main.lpy").as_uri())

    async def scenario():
        try:
            editor.open("(print undefined-name)\n")
            await until(
                lambda: editor.published and backend_diagnostics(editor.published[-1]),
                timeout=30,
            )
            return editor.published[-1]
        finally:
            server.did_close(editor, SimpleNamespace(text_document=editor.document))

    [found] = run_on_loop(scenario())
    assert found.source == "basedpyright"
    assert found.code == "reportUndefinedVariable"
    assert found.severity == lsp.DiagnosticSeverity.Error
    assert found.message == '"undefined-name" is not defined'
    assert (found.range.start.line, found.range.start.character) == (0, 7)
    assert (found.range.end.line, found.range.end.character) == (0, 21)
