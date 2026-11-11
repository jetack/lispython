"""The Python backend's diagnostics in the LSP: mapped onto the .lpy buffer,
gated to the full view's text, merged with lispython's own.

Mapping positions are read off the source and generated text with
``locate`` (1-based line, code-point column), so every assertion names the
word it lands on.  The server-flow tests run the fake backend of
test_lsp_backend, which publishes an Error on each ``undefined_name``; the
last test runs the session's real basedpyright.
"""

import asyncio
import concurrent.futures
import fnmatch
import importlib
import importlib.machinery
import itertools
import pathlib
import re
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from lsprotocol import types as lsp
from pygls.workspace import TextDocument

import lispy  # noqa: F401  (registers the .lpy importer)
import lispy.core.macro
from lispy.core.importer import finder_suffixes
from tests.lsp_backends import basedpyright_only
from tests.test_lsp_backend import FAKE_SERVER, fake_hover, hover_params, run_bounded

backend = importlib.import_module("lispy.lsp.backend")
macro_modules = importlib.import_module("lispy.lsp.macro_modules")
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
    assert mapped(src, "register_required") is None
    assert mapped(src, "register_required", tagged=True) is None
    src = "(defmacro twice [a] (return `(do ~a ~a)))\n(twice (print 1))"
    assert mapped(src, "___defmacro_temp___twice") is None
    view = pv.compile_view(src)
    assert pv.lpy_diagnostic_range(view, src, (1, 0), (1, 0), False) is None


# ---------------------------------------------------------------------------
# Messages in the buffer's spelling
# ---------------------------------------------------------------------------


def test_messages_quote_names_as_the_code_they_mark_spells_them():
    # Defects: a message names the Python spelling the user never wrote; a
    # quoted name the marked code does not spell (a TypedDict key, while
    # my-key is a symbol elsewhere) is rewritten; quoted text that is no
    # name (a type) is rewritten.
    src = '(print undefined-name)\n(= unused-x 1)\n(= d {"my_key" 1})\n(= my-key 2)'
    lines = pv.source_lines(src)

    def rename(message, word):
        return server.message_to_buffer(message, lines, span(src, word))

    assert (
        rename('"undefined_name" is not defined', "undefined-name")
        == '"undefined-name" is not defined'
    )
    assert (
        rename('Variable "unused_x" is not accessed', "(= unused-x 1)")
        == 'Variable "unused-x" is not accessed'
    )
    key = '"my_key" is not a defined key in "Point"'
    assert rename(key, '{"my_key" 1}') == key
    # A key missing from the marked call is quoted as the TypedDict spells
    # it, never as the unrelated symbol my-key.
    missing = '"my_key" is required in "Point"'
    assert rename(missing, "(print undefined-name)") == missing
    literal = 'Type "Literal[\'s\']" is not assignable to "unused_x"'
    assert rename(literal, "(= unused-x 1)") == literal.replace("unused_x", "unused-x")
    # pyrefly quotes in backticks, renamed alike; a name between two
    # different quotes is quoted by neither, so it stays.
    assert (
        rename("Could not find name `undefined_name`", "undefined-name")
        == "Could not find name `undefined-name`"
    )
    mismatched = 'Could not find name "undefined_name`'
    assert rename(mismatched, "undefined-name") == mismatched


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
    assert client.diagnostics_for(uri, full) == [(diagnostic, [])]
    assert heard == [uri]


def test_a_reopened_document_never_reuses_a_version(tmp_path):
    # Defect: a reopen sends didOpen at a version already used, so a late
    # publish for the previous open (its version, or pyrefly's empty publish
    # at a closed document's last version + 1) is kept as the diagnostics of
    # the new text and mapped onto it.
    project = tmp_path / "project"
    project.mkdir()
    client = backend.BackendClient(
        pv.view_root_for(str(project), cache_home=str(tmp_path / "cache")), None
    )

    class Process:
        def __init__(self):
            self.sent = []

        def text_document_did_open(self, params):
            self.sent.append(params.text_document.version)

        def text_document_did_change(self, params):
            self.sent.append(params.text_document.version)

        def text_document_did_close(self, params):
            self.closed.append(params.text_document.uri)

        returncode = None

    process = Process()
    process.closed = []
    client._client = process
    client.process = process
    client.state = "running"
    uri = (project / "main.lpy").as_uri()
    shadow = client.backend_uri(uri)
    diagnostic = lsp.Diagnostic(
        range=lsp.Range(
            start=lsp.Position(line=0, character=0),
            end=lsp.Position(line=0, character=1),
        ),
        message="m",
    )
    text_a, text_b = "a", "b"
    client._sync(uri, text_a, origin=("src", None))
    client._full[uri] = (text_a, ("src", None))
    version_a = process.sent[-1]
    client.close_document(uri)
    assert process.closed == [shadow]
    client._sync(uri, text_b, origin=("src", None))
    client._full[uri] = (text_b, ("src", None))
    version_b = process.sent[-1]
    client._diagnostics_published(process, published(shadow, version_a, [diagnostic]))
    assert client.diagnostics_for(uri, text_b) is None
    client._diagnostics_published(process, published(shadow, version_a + 1, []))
    assert client.diagnostics_for(uri, text_b) is None
    client._diagnostics_published(process, published(shadow, version_b, [diagnostic]))
    assert client.diagnostics_for(uri, text_b) == [(diagnostic, [])]


class _SyncingProcess:
    """A running backend's client that records the documents synced to it:
    sent holds (kind, backend uri, version) per didOpen/didChange."""

    def __init__(self):
        self.sent = []
        self.closed = []

    def text_document_did_open(self, params):
        doc = params.text_document
        self.sent.append(("open", doc.uri, doc.version))

    def text_document_did_change(self, params):
        doc = params.text_document
        self.sent.append(("change", doc.uri, doc.version))

    def text_document_did_close(self, params):
        self.closed.append(params.text_document.uri)

    def version(self, uri):
        """The version last sent for backend document uri."""
        return [version for _, sent, version in self.sent if sent == uri][-1]


def _syncing_client(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    client = backend.BackendClient(
        pv.view_root_for(str(project), cache_home=str(tmp_path / "cache")), None
    )
    process = _SyncingProcess()
    client._client = process
    client.state = "running"
    client.process = SimpleNamespace(returncode=None)
    uri = (project / "main.lpy").as_uri()
    heard = []
    client.on_diagnostics = heard.append
    return client, process, uri, heard


_DIAGNOSTIC = lsp.Diagnostic(
    range=lsp.Range(
        start=lsp.Position(line=0, character=6), end=lsp.Position(line=0, character=20)
    ),
    message="m",
)


def test_a_publish_for_a_text_that_becomes_the_full_view_is_kept(tmp_path):
    # Defect: a request synced text T, the backend published for T before
    # push-buffer made T the full view, so the publish was dropped; the
    # push then sent nothing (the backend already held T), the backend
    # never published again, and the buffer had no backend diagnostics
    # until its next edit.
    client, process, uri, heard = _syncing_client(tmp_path)
    text = "print(undefined_name)"
    shadow = client.backend_uri(uri)
    client._sync(uri, text, origin=("src", None))
    client._diagnostics_published(
        process, published(shadow, process.version(shadow), [_DIAGNOSTIC])
    )
    assert heard == []
    view = SimpleNamespace(text=text)
    assert client.push_buffer(uri, "src", view)
    assert len(process.sent) == 1  # the push sent nothing
    assert client.diagnostics_for(uri, text) == [(_DIAGNOSTIC, [])]
    assert heard == [uri]


def test_a_kept_publish_is_dropped_once_another_text_is_sent(tmp_path):
    # Defects: a publish kept for a request's cursor view is accepted when
    # the full view of another text is sent; one kept for a text the backend
    # has since replaced is accepted once that text comes back as a newer
    # version, which the backend will publish for itself.
    client, process, uri, heard = _syncing_client(tmp_path)
    shadow = client.backend_uri(uri)
    cursor, full = "print(cursor)", "print(full)"

    def publish():
        version = process.version(shadow)
        client._diagnostics_published(
            process, published(shadow, version, [_DIAGNOSTIC])
        )

    client._sync(uri, cursor, origin=("src", None))
    publish()
    assert client.push_buffer(uri, "src", SimpleNamespace(text=full))
    assert client.diagnostics_for(uri, full) is None
    assert client.diagnostics_for(uri, cursor) is None
    # The cursor text again, published for; then another text and the
    # cursor text once more as a newer version, which the full view takes.
    client._sync(uri, cursor, origin=("src", None))
    publish()
    client._sync(uri, "print(other)", origin=("src", None))
    sent = len(process.sent)
    client._sync(uri, cursor, origin=("src", None))
    assert len(process.sent) == sent + 1
    assert client.push_buffer(uri, "src", SimpleNamespace(text=cursor))
    assert client.diagnostics_for(uri, cursor) is None
    assert heard == []


def test_a_kept_publish_is_dropped_once_its_copy_is_closed(tmp_path):
    # Defect: a publish kept for a copy later closed (the document closed or
    # moved while a request ran) is accepted once the document reopens with
    # the same text, as the new copy's diagnostics, though the backend
    # analysed only the closed one and publishes the new one itself.
    client, process, uri, heard = _syncing_client(tmp_path)
    shadow = client.backend_uri(uri)
    text = "print(undefined_name)"
    client._sync(uri, text, origin=("src", None))  # a request's view
    client._diagnostics_published(
        process, published(shadow, process.version(shadow), [_DIAGNOSTIC])
    )
    client.close_document(uri)
    assert process.closed == [shadow]
    assert client.push_buffer(uri, "src", SimpleNamespace(text=text))
    assert process.sent[-1][0] == "open"
    assert client.diagnostics_for(uri, text) is None
    assert heard == []


def test_a_kept_publishs_related_locations_are_placed_when_published(tmp_path):
    # Defect: a publish kept until its text became the full view placed its
    # related locations only then, so a location in another open document
    # mapped through that document's newer view, not the one analysed.
    client, process, uri, heard = _syncing_client(tmp_path)
    other = uri.replace("main.lpy", "other.lpy")
    other_shadow = client.backend_uri(other)
    analysed = ("(= y 1)", object())
    client._sync(other, "y = 1", origin=analysed)
    text = "x = 1"
    client._sync(uri, text, origin=("(= x 1)", object()))
    shadow = client.backend_uri(uri)
    here = lsp.Range(
        start=lsp.Position(line=0, character=0), end=lsp.Position(line=0, character=1)
    )
    related = lsp.DiagnosticRelatedInformation(
        location=lsp.Location(uri=other_shadow, range=here), message="see"
    )
    diagnostic = lsp.Diagnostic(range=here, message="m", related_information=[related])
    client._diagnostics_published(
        process, published(shadow, process.version(shadow), [diagnostic])
    )
    client._sync(other, "y = 2", origin=("(= y 2)", object()))  # an edit of other
    assert client.push_buffer(uri, "(= x 1)", SimpleNamespace(text=text))
    [(_, places)] = client.diagnostics_for(uri, text)
    assert places == [("lpy", other, analysed)]


@pytest.mark.parametrize("order", ["published-first", "in-flight", "kept"])
def test_own_related_locations_map_through_the_runs_view(tmp_path, monkeypatch, order):
    # Defect: the document's own related locations were placed at the origin
    # of one source of the text the backend analysed (an earlier full view,
    # a request's cursor view), while the run maps the ranges through the
    # buffer's latest source: after an edit that leaves the text alone (a
    # comment), they missed by its lines.
    client, process, uri, heard = _syncing_client(tmp_path)
    shadow = client.backend_uri(uri)
    older, latest = "(= x 1)", "; comment\n(= x 1)"
    older_view, latest_view = pv.compile_view(older), pv.compile_view(latest)
    text = latest_view.text
    assert older_view.text == text
    (line, col), (end_line, end_col) = span(text, "x")
    here = lsp.Range(
        start=lsp.Position(line=line - 1, character=col),
        end=lsp.Position(line=end_line - 1, character=end_col),
    )
    related = lsp.DiagnosticRelatedInformation(
        location=lsp.Location(uri=shadow, range=here), message="see"
    )
    diagnostic = lsp.Diagnostic(range=here, message="m", related_information=[related])

    def publish():
        version = process.version(shadow)
        client._diagnostics_published(process, published(shadow, version, [diagnostic]))

    if order == "published-first":
        client.push_buffer(uri, older, older_view)
        publish()
        client.push_buffer(uri, latest, latest_view)  # sends nothing
    else:
        if order == "in-flight":
            client.push_buffer(uri, older, older_view)
        client._in_flight[uri] = 1
        client._sync(uri, text, origin=(older, older_view))  # a request's view
        if order == "kept":
            publish()
        client.push_buffer(uri, latest, latest_view)
        if order == "in-flight":
            publish()
        client._request_done(uri)
    assert [kind for kind, _, _ in process.sent] == ["open"]
    stub = SimpleNamespace(name="pyrefly", backend_diagnostics=client.diagnostics_for)
    monkeypatch.setattr(server, "backends", stub)
    run = server.DiagnosticRun(None, TextDocument(uri, latest, version=2))
    run.view = latest_view
    [found] = asyncio.run(server.mapped_backend_diagnostics(run))
    [info] = found.related_information
    assert found.range.start.line == 1
    assert (info.location.uri, info.location.range) == (uri, found.range)
    assert heard == [uri]


def test_related_locations_are_placed_when_published(tmp_path):
    # Defect: a related location is placed (locate) only when the server
    # maps the set, after a later sync of the document it lies in has
    # replaced the origin of the text the backend analysed, so it maps
    # through the wrong view.
    client, process, uri, heard = _syncing_client(tmp_path)
    other = uri.replace("main.lpy", "other.lpy")
    analysed = ("(= y 1)", object())
    client._sync(other, "y = 1", origin=analysed)
    full = "x = 1"
    client.push_buffer(uri, "(= x 1)", SimpleNamespace(text=full))
    shadow = client.backend_uri(uri)
    here = lsp.Range(
        start=lsp.Position(line=0, character=0), end=lsp.Position(line=0, character=1)
    )
    related = lsp.DiagnosticRelatedInformation(
        location=lsp.Location(uri=client.backend_uri(other), range=here), message="see"
    )
    diagnostic = lsp.Diagnostic(range=here, message="m", related_information=[related])
    client._diagnostics_published(
        process, published(shadow, process.version(shadow), [diagnostic])
    )
    client._sync(other, "y = 2", origin=("(= y 2)", object()))  # an edit of other
    [(_, places)] = client.diagnostics_for(uri, full)
    assert places == [("lpy", other, analysed)]


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

    def params(self):
        return SimpleNamespace(text_document=SimpleNamespace(uri=self.uri))

    def save(self):
        server.did_save(self, self.params())

    def close(self):
        server.did_close(self, self.params())


def fresh_server_state(monkeypatch):
    monkeypatch.setattr(server, "view_cache", pv.ViewCache())
    monkeypatch.setattr(server, "DIAGNOSTIC_RUNS", {})
    monkeypatch.setattr(server, "_COMPILING", set())
    monkeypatch.setattr(server, "_PRESET_FALLBACK_NOTIFIED", set())
    monkeypatch.setattr(server, "_REFRESHED_EPOCH", None, raising=False)
    monkeypatch.setattr(server, "_REFRESH_TARGET", None, raising=False)
    for name in ["WORKSPACE_INDEX", "WORKSPACE_SYMBOLS", "FILE_TREES"]:
        monkeypatch.setattr(server, name, {})


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
    monkeypatch.setitem(backend.BACKENDS, "pyrefly", lambda: None)
    monkeypatch.setitem(
        backend.BACKENDS, "basedpyright", lambda: [sys.executable, str(script)]
    )
    folder = SimpleNamespace(uri=project.as_uri())
    backends = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "cache"))
    backends.configure(None)
    backends.on_diagnostics = server.backend_diagnostics_arrived
    monkeypatch.setattr(server, "backends", backends)
    fresh_server_state(monkeypatch)
    return Editor((project / "main.lpy").as_uri()), backends


def backend_diagnostics(diagnostics, name="basedpyright"):
    return [d for d in diagnostics if d.source == name]


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
        def compile_unit_forms(*args, **kwargs):
            calls.append(True)
            return real(*args, **kwargs)

        return compile_unit_forms

    monkeypatch.setattr(pv, "compile_unit_forms", counting(pv.compile_unit_forms))
    monkeypatch.setattr(
        lispy.core.macro,
        "compile_unit_forms",
        counting(lispy.core.macro.compile_unit_forms),
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


def test_a_change_right_after_the_open_still_starts_the_backend(fake_served):
    # Defect: the open starts the backend only once its compile is done, and
    # a change that comes first supersedes the open's run: no backend, no
    # Python diagnostics until some request starts one.
    editor, backends = fake_served

    async def scenario():
        try:
            editor.open("(print 1)\n")
            editor.change("(print undefined-name)\n")
            await until(
                lambda: editor.published and backend_diagnostics(editor.published[-1])
            )
        finally:
            await backends.shutdown_all()

    run_bounded(scenario())


def test_a_save_compiles_again(fake_served, monkeypatch):
    # Defect: a save reuses the version's cached compile, so a macro that
    # reads the disk (here a flag file) shows its result from before.
    editor, backends = fake_served
    project = pathlib.Path(editor.uri[len("file://") :]).parent
    (project / "flag.txt").write_text("ok")
    (project / "diag_flag_mac.lpy").write_text(
        "(import os.path)\n"
        "(defmacro from-disk []\n"
        '  (with [(open (os.path.join (os.path.dirname __file__) "flag.txt")) as f]\n'
        '    (if (== (.strip (.read f)) "bad")\n'
        '        (raise (ValueError "bad flag"))))\n'
        "  (return 1))\n"
    )
    monkeypatch.delitem(sys.modules, "diag_flag_mac", raising=False)
    backends.configure({"backend": "off"})

    async def scenario():
        editor.open("(require diag_flag_mac [from-disk])\n(print (from-disk))\n")
        await until(lambda: editor.published)
        assert editor.published[-1] == []
        (project / "flag.txt").write_text("bad")
        count = len(editor.published)
        editor.save()
        await until(lambda: len(editor.published) > count)
        [warning] = editor.published[-1]
        assert "bad flag" in warning.message

    run_bounded(scenario())
    sys.modules.pop("diag_flag_mac", None)


class Documents:
    """A language server double for several open documents.  As pygls
    does, a closed document is read back from disk on request."""

    def __init__(self):
        self.workspace = self
        self.documents = {}
        self.published = {}

    def get_text_document(self, uri):
        if uri not in self.documents:
            path = pathlib.Path(uri[len("file://") :])
            return TextDocument(uri, path.read_text())
        return self.documents[uri]

    def text_document_publish_diagnostics(self, params):
        self.published.setdefault(params.uri, []).append(params.diagnostics)

    def open(self, path):
        uri = path.as_uri()
        self.documents[uri] = TextDocument(uri, path.read_text(), version=1)
        doc = SimpleNamespace(uri=uri, version=1)
        server.did_open(self, SimpleNamespace(text_document=doc))
        return uri

    def update(self, path):
        uri = path.as_uri()
        version = self.documents[uri].version + 1
        self.documents[uri] = TextDocument(uri, path.read_text(), version=version)
        return SimpleNamespace(text_document=SimpleNamespace(uri=uri, version=version))

    def save(self, path):
        server.did_save(self, self.update(path))

    def change(self, path):
        server.did_change(self, self.update(path))

    def close(self, path):
        params = SimpleNamespace(text_document=SimpleNamespace(uri=path.as_uri()))
        del self.documents[path.as_uri()]
        server.did_close(self, params)


@pytest.fixture
def macro_documents(fake_served, monkeypatch):
    """(editor, mac, mod): an open macro module and an open document
    requiring it, published, the backend off."""
    _, backends = fake_served
    backends.configure({"backend": "off"})
    project = pathlib.Path(backends.folders()[0].uri[len("file://") :])
    name = "diagmmsave"
    pkg = project / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    mac, mod = pkg / "mac.lpy", pkg / "mod.lpy"
    mac.write_text("(defmacro twice [x] (return `(+ ~x ~x)))\n")
    mod.write_text("(require .mac *)\n(= yy (twice 2))\n")
    for key in [name, name + ".mac"]:
        monkeypatch.delitem(sys.modules, key, raising=False)
    yield Documents(), mac, mod
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]


BOOM = '(defmacro twice [x] (raise (ValueError "diag-boom")))\n'


async def opened(editor, mac, mod):
    mod_uri, mac_uri = editor.open(mod), editor.open(mac)
    await until(lambda: mod_uri in editor.published and mac_uri in editor.published)
    assert editor.published[mod_uri][-1] == []
    return mod_uri


@pytest.mark.parametrize("superseded", [False, True])
def test_a_saved_macro_module_refreshes_the_documents_requiring_it(
    macro_documents, superseded
):
    # Defects: only the saved document is compiled again, so another open
    # document requiring the saved macro module keeps the diagnostics (and
    # backend view) of the old macros until its own next edit; and the
    # refresh hangs on the save's run, which an edit right after the save
    # supersedes before it finishes.
    editor, mac, mod = macro_documents

    async def scenario():
        mod_uri = await opened(editor, mac, mod)
        count = len(editor.published[mod_uri])
        mac.write_text(BOOM)
        editor.save(mac)
        if superseded:
            editor.change(mac)
        await until(lambda: len(editor.published[mod_uri]) > count, timeout=5)
        [warning] = editor.published[mod_uri][-1]
        assert "diag-boom" in warning.message

    run_bounded(scenario())


def test_a_macro_rewriting_what_it_imports_refreshes_a_bounded_number_of_times(
    macro_documents,
):
    # Defect: a macro that rewrites a file it imports (codegen) moves the
    # epoch on every compile, so each refreshed document's run refreshes
    # the others (and itself) again, forever.
    editor, mac, mod = macro_documents
    gen = mac.parent / "gen.py"
    gen.write_text("")
    mac.write_text(
        "(import os.path)\n"
        f"(import {mac.parent.name}.gen)\n"
        "(defmacro twice [x]\n"
        '  (with [(open (os.path.join (os.path.dirname __file__) "gen.py") "a")'
        " as f]\n"
        '    (.write f "#\\n"))\n'
        "  (return `(+ ~x ~x)))\n"
    )
    other = mod.parent / "other.lpy"
    other.write_text(mod.read_text())

    def drained():
        return not server._COMPILING and not server._DIAGNOSTIC_TASKS

    async def scenario():
        uris = [editor.open(mod), editor.open(other)]
        await until(lambda: all(uri in editor.published for uri in uris))
        await until(drained, timeout=5)
        before = {uri: len(editor.published[uri]) for uri in uris}
        editor.change(mod)
        # A refresh loop compiles forever, so it never drains.
        await until(drained, timeout=5)
        after = {uri: len(editor.published[uri]) for uri in uris}
        assert after[uris[1]] > before[uris[1]]  # the change refreshed other
        assert sum(after.values()) - sum(before.values()) < 4

    run_bounded(scenario())


def test_a_document_closed_during_a_refresh_is_left_closed(
    macro_documents, monkeypatch
):
    # Defect: the refresh publishes every outdated uri it read before a
    # close, which reads the closed file back from disk, publishes its
    # diagnostics and keeps a cache entry for it that nobody forgets.
    editor, mac, mod = macro_documents
    real = server.refresh_epochs
    armed, entered, go = threading.Event(), threading.Event(), threading.Event()

    def held():
        now = real()
        if armed.is_set():
            entered.set()
            assert go.wait(10)
        return now

    monkeypatch.setattr(server, "refresh_epochs", held)

    async def scenario():
        mod_uri = await opened(editor, mac, mod)
        armed.set()
        mac.write_text(BOOM)
        editor.save(mac)
        await until(entered.is_set)
        editor.close(mod)
        count = len(editor.published[mod_uri])
        go.set()
        await until(lambda: not server._COMPILING and not server._DIAGNOSTIC_TASKS)
        assert len(editor.published[mod_uri]) == count
        assert editor.published[mod_uri][-1] == []
        assert mod_uri not in server.DIAGNOSTIC_RUNS
        assert mod_uri not in server.view_cache._views

    try:
        run_bounded(scenario())
    finally:
        go.set()


# ---------------------------------------------------------------------------
# A change of module naming publishes the open documents again
# ---------------------------------------------------------------------------


@pytest.fixture
def outside_documents(fake_served, monkeypatch):
    """(editor, outside, mod, folders): a document outside/<pkg>/mod.lpy,
    in no workspace folder (folders, the server's and the backends', which
    a test may change) and with no __init__, requiring the sibling macro
    module relatively, the backend off."""
    _, backends = fake_served
    backends.configure({"backend": "off"})
    project = pathlib.Path(backends.folders()[0].uri[len("file://") :])
    outside = project.parent / "outside"
    name = "diagrepub"
    pkg = outside / name
    pkg.mkdir(parents=True)
    (pkg / "mac.lpy").write_text("(defmacro twice [x] (return `(+ ~x ~x)))\n")
    mod = pkg / "mod.lpy"
    mod.write_text("(require .mac *)\n(= yy (twice 2))\n")
    folders = [SimpleNamespace(uri=project.as_uri())]
    monkeypatch.setattr(server, "workspace_folders", lambda: list(folders))
    # As the server's: the backends read the same folders.
    monkeypatch.setattr(backends, "folders", lambda: list(folders))
    for key in [name, name + ".mac"]:
        monkeypatch.delitem(sys.modules, key, raising=False)
    yield Documents(), outside, mod, folders
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]


async def published_as_script(editor, mod):
    """Open mod; as a script its relative require fails."""
    uri = editor.open(mod)
    await until(lambda: uri in editor.published)
    [warning] = editor.published[uri][-1]
    assert "relative" in warning.message
    return uri


def test_a_created_init_file_republishes_the_open_documents(outside_documents):
    # Defect: an __init__ created beside an open document makes its folder
    # a package, so its relative require now resolves, but the warning
    # published under the old naming stays until the document's next edit.
    editor, _, mod, _ = outside_documents

    async def scenario():
        uri = await published_as_script(editor, mod)
        count = len(editor.published[uri])
        init = mod.parent / "__init__.py"
        init.write_text("")
        files_changed(editor, init)
        await until(lambda: len(editor.published[uri]) > count, timeout=5)
        assert editor.published[uri][-1] == []

    run_bounded(scenario())


@pytest.fixture
def outside_package(outside_documents):
    """(editor, outside, mod, folders) as outside_documents, but mod is
    outside/diagrepub/sub/mod.lpy, inside the regular package sub in the
    namespace directory diagrepub, and requires diagrepub.mac absolutely.
    Its import root, the one directory its compile searches, is outside
    only while diagrepub is a namespace package of a workspace folder
    (`import diagrepub.sub.mod`); else it is outside/diagrepub (`python
    -m sub.mod`), where diagrepub.mac is not found, so swallow stays a call
    and its statement operand fails to compile."""
    editor, outside, _, folders = outside_documents
    pkg = outside / "diagrepub"
    (pkg / "mac.lpy").write_text("(defmacro swallow [x] (return 1))\n")
    sub = pkg / "sub"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    mod = sub / "mod.lpy"
    mod.write_text("(require diagrepub.mac *)\n(= yy (swallow (= a 1)))\n")
    for key in ["sub", "sub.mod"]:
        sys.modules.pop(key, None)
    yield editor, outside, mod, folders
    for key in ["sub", "sub.mod"]:
        sys.modules.pop(key, None)


async def published_unexpanded(editor, mod):
    """Open mod; its require loads no macros, so one form fails."""
    uri = editor.open(mod)
    await until(lambda: uri in editor.published)
    [warning] = editor.published[uri][-1]
    assert "statement" in warning.message
    return uri


def test_an_added_workspace_folder_republishes_the_open_documents(
    outside_package,
):
    # Defect: a workspace folder added around an open document makes the
    # directory above its package a namespace package of that project, so
    # its require now resolves, but the warning published before stays.
    editor, outside, mod, folders = outside_package

    async def scenario():
        uri = await published_unexpanded(editor, mod)
        count = len(editor.published[uri])
        folder = lsp.WorkspaceFolder(uri=outside.as_uri(), name="outside")
        folders.append(folder)
        folders_changed(editor, added=[folder])
        await until(lambda: len(editor.published[uri]) > count, timeout=5)
        assert editor.published[uri][-1] == []

    run_bounded(scenario())


def files_changed(ls, path, kind=lsp.FileChangeType.Created):
    """The client tells the server path was created (or kind)."""
    change = lsp.FileEvent(uri=path.as_uri(), type=kind)
    server.did_change_watched_files(
        ls, lsp.DidChangeWatchedFilesParams(changes=[change])
    )


def folders_changed(ls, added=(), removed=()):
    """The client tells the server its folders changed (pygls has already
    updated them)."""
    event = lsp.WorkspaceFoldersChangeEvent(added=list(added), removed=list(removed))
    server.did_change_workspace_folders(
        ls, lsp.DidChangeWorkspaceFoldersParams(event=event)
    )


@pytest.mark.parametrize("suffix", [".py", importlib.machinery.EXTENSION_SUFFIXES[0]])
def test_a_module_file_ending_a_namespace_package_republishes(outside_package, suffix):
    # Defects: only an __init__ is heard, but a module file created beside
    # a namespace directory of its name ends that package too, so the open
    # document keeps the diagnostics of its old package naming; and of the
    # importer's suffixes only .py/.lpy count, so an extension module
    # (a suffix holding dots) is never heard.
    editor, outside, mod, folders = outside_package
    folders.append(SimpleNamespace(uri=outside.as_uri()))

    async def scenario():
        uri = editor.open(mod)
        await until(lambda: uri in editor.published)
        assert editor.published[uri][-1] == []
        count = len(editor.published[uri])
        module = outside / ("diagrepub" + suffix)
        module.write_text("")
        files_changed(editor, module)
        await until(lambda: len(editor.published[uri]) > count, timeout=5)
        [warning] = editor.published[uri][-1]
        assert "statement" in warning.message

    run_bounded(scenario())


def test_only_a_file_that_can_rename_a_users_module_republishes(
    outside_documents,
):
    # Defect: every file event republishes every open document and sweeps
    # every backend's view root, and a client sends one per file, so an
    # install into a venv or a checkout runs thousands of sweeps.
    editor, outside, mod, folders = outside_documents
    folders.append(SimpleNamespace(uri=outside.as_uri()))
    site = outside / "venv" / "lib" / "site-packages" / "dep"
    site.mkdir(parents=True)
    plain = mod.parent / "plain.py"

    async def scenario():
        uri = editor.open(mod)
        await until(lambda: uri in editor.published)
        run = server.DIAGNOSTIC_RUNS[uri]
        for path in [plain, site / "__init__.py"]:
            path.write_text("")
            files_changed(editor, path)
            assert server.DIAGNOSTIC_RUNS[uri] is run
        files_changed(editor, mod.parent / "__init__.py")  # the control
        assert server.DIAGNOSTIC_RUNS[uri] is not run

    run_bounded(scenario())


@pytest.mark.parametrize("created", ["src/x.py", "shared/acme/m.py"])
def test_a_module_file_flipping_the_src_layout_republishes(outside_documents, created):
    # Defect: only __init__ files and namespace-ending module files are
    # heard, so a module file that makes src a src layout (src/x.py in a
    # flat project, or shared/acme/m.py behind the link src/acme ->
    # ../shared/acme) leaves the remembered layout and the open documents'
    # search dirs as they were.
    editor, outside, _, folders = outside_documents
    folders.append(SimpleNamespace(uri=outside.as_uri()))
    (outside / "mypkg").mkdir()
    (outside / "mypkg" / "__init__.py").write_text("")
    (outside / "shared" / "acme").mkdir(parents=True)
    (outside / "src").mkdir()
    (outside / "src" / "acme").symlink_to(outside / "shared" / "acme")
    scripts = outside / "scripts"
    scripts.mkdir()
    script = scripts / "x.lpy"
    script.write_text("(= x 1)\n")

    async def scenario():
        uri = editor.open(script)
        await until(lambda: uri in editor.published)
        assert server.uri_module_names(uri)[3] == (str(scripts), str(outside))
        count = len(editor.published[uri])
        path = outside / created
        path.write_text("")
        files_changed(editor, path)
        await until(lambda: len(editor.published[uri]) > count, timeout=5)
        assert server.uri_module_names(uri)[3] == (str(scripts), str(outside / "src"))

    run_bounded(scenario())


def test_a_save_sees_a_src_package_that_came_with_no_file_event(outside_documents):
    # Defect: a src package moved in whole (mv ../pkg src/) sends no event
    # for its files, and a save keeps the remembered no-Python answer, so
    # the open document never searches src.
    editor, outside, _, folders = outside_documents
    folders.append(SimpleNamespace(uri=outside.as_uri()))
    (outside / "mypkg").mkdir()
    (outside / "mypkg" / "__init__.py").write_text("")
    scripts = outside / "scripts"
    scripts.mkdir()
    script = scripts / "x.lpy"
    script.write_text("(= x 1)\n")

    async def scenario():
        uri = editor.open(script)
        await until(lambda: uri in editor.published)
        assert server.uri_module_names(uri)[3] == (str(scripts), str(outside))
        (outside / "src" / "pkg").mkdir(parents=True)
        (outside / "src" / "pkg" / "m.py").write_text("")
        editor.save(script)
        assert server.uri_module_names(uri)[3] == (str(scripts), str(outside / "src"))

    run_bounded(scenario())


def test_a_module_file_in_a_pruned_directory_walks_no_src(
    outside_documents, monkeypatch
):
    # Defect: an install into the project's virtualenv (one event per file)
    # asks the project's src layout again for every batch.
    _, outside, _, folders = outside_documents
    folders.append(SimpleNamespace(uri=outside.as_uri()))
    (outside / "venv").mkdir()
    (outside / "venv" / "pyvenv.cfg").write_text("")
    asked = []
    monkeypatch.setattr(server, "refresh_src_layout", lambda root: asked.append(root))
    for rel in [
        ".venv/lib/site-packages/dep/m.py",
        "venv/lib/site-packages/dep/m.py",
        "node_modules/x/m.py",
        "src/site-packages/m.py",
    ]:
        files_changed(SimpleNamespace(), outside / rel)
    assert asked == []
    files_changed(SimpleNamespace(), outside / "src" / "m.py")  # the control
    assert asked == [str(outside)]


def test_an_init_file_in_lispythons_own_package_renames_modules():
    # Defect: naming borrowed the tracking exclusions, which leave out
    # lispython's own package, so an __init__ created in an editable
    # lispython checkout that is the workspace republishes nothing.
    # The checkout's package, never lispy.__file__, which an installed
    # wheel (nox tests_latest) puts in site-packages, a library path.
    checkout = pathlib.Path(__file__).resolve().parents[1]
    created = checkout / "src" / "lispy" / "newpkg" / "__init__.py"
    assert server.renaming_file_p(str(created))


def test_an_lpy_file_created_or_deleted_on_disk_is_indexed(outside_documents):
    # Defect: the index hears of an .lpy file only when the editor opens or
    # saves it, so a file another tool wrote offers none of its
    # definitions, and a deleted one keeps offering them.
    editor, outside, _, folders = outside_documents
    folders.append(SimpleNamespace(uri=outside.as_uri()))
    defs = outside / "defs.lpy"

    def found():
        query = SimpleNamespace(query="far")
        return [s.location.uri for s in server.workspace_symbol(editor, query)]

    async def scenario():
        defs.write_text("(def far-away [] 1)\n")
        files_changed(editor, defs)
        assert found() == [defs.as_uri()]
        defs.unlink()
        files_changed(editor, defs, lsp.FileChangeType.Deleted)
        assert found() == []
        assert defs.as_uri() not in server.FILE_TREES

    run_bounded(scenario())


def test_a_file_the_client_spells_otherwise_is_indexed_once(outside_documents):
    # Defect: the index is keyed by the raw uri, and a client may spell a
    # file's uri otherwise than pathlib (VS Code on Windows: file:///c%3A/),
    # so a file it opened is indexed twice once a watched event reaches it,
    # and a watched deletion leaves the client-spelled copy behind.
    editor, outside, _, folders = outside_documents
    folders.append(SimpleNamespace(uri=outside.as_uri()))
    defs = outside / "far-defs.lpy"
    defs.write_text("(def far-away [] 1)\n")
    spelled = defs.as_uri().replace("far-defs", "far%2Ddefs")
    assert spelled != defs.as_uri()

    def found():
        query = SimpleNamespace(query="far")
        return [s.location.uri for s in server.workspace_symbol(editor, query)]

    async def scenario():
        editor.documents[spelled] = TextDocument(spelled, defs.read_text(), version=1)
        doc = SimpleNamespace(uri=spelled, version=1)
        server.did_open(editor, SimpleNamespace(text_document=doc))
        await until(lambda: spelled in editor.published)
        files_changed(editor, defs)
        assert found() == [defs.as_uri()]
        defs.unlink()
        files_changed(editor, defs, lsp.FileChangeType.Deleted)
        assert found() == []

    run_bounded(scenario())


def test_a_backend_gets_one_sweep_while_one_is_due(tmp_path):
    # Defect: each file event schedules another full sweep of the view root
    # even while one has not begun, which would read the same disk.
    folder = SimpleNamespace(uri=tmp_path.as_uri())
    backends = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "c"))
    swept = []

    class Running:
        running, starting = True, False

        async def saved(self, *uris):
            swept.append(uris)

    backends.clients[str(tmp_path.resolve())] = Running()
    uri = (tmp_path / "main.lpy").as_uri()

    async def scenario():
        backends.files_changed([uri])
        backends.files_changed([uri])
        await asyncio.gather(*backends._tasks)
        backends.files_changed([uri])  # none due: swept again
        await asyncio.gather(*backends._tasks)

    run_bounded(scenario())
    assert swept == [(uri,), (uri,)]


def test_a_folder_change_moves_documents_and_drops_gone_projects(tmp_path):
    # Defects: a document whose project changed stays open in the old
    # project's backend (its stale view overriding the disk there), and a
    # removed folder's backend keeps running; once shut down but kept, the
    # folder added again finds a closed client and never gets a backend.
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    folders = [SimpleNamespace(uri=outer.as_uri())]
    backends = backend.Backends(lambda: list(folders), cache_home=str(tmp_path / "c"))
    uri = (inner / "mod.lpy").as_uri()
    old = backends.client_for(uri)
    old._full[uri] = ("text", None)

    async def scenario():
        folders.append(SimpleNamespace(uri=inner.as_uri()))
        backends.folders_changed([uri])
        assert uri not in old._full
        new = backends.client_for(uri)
        assert new is not old
        folders.pop(0)
        backends.folders_changed([uri])
        await asyncio.gather(*backends._tasks)
        assert old.state == "closed"
        assert list(backends.clients) == [str(inner.resolve())]
        folders.insert(0, SimpleNamespace(uri=outer.as_uri()))
        backends.folders_changed([uri])
        again = backends.client_for((outer / "top.lpy").as_uri())
        assert again is not old and again.state != "closed"

    run_bounded(scenario())


def test_a_module_file_change_sweeps_the_running_backend(fake_served):
    # Defect: a naming change publishes the buffers again but never rewrites
    # the backend's view root (package overlays, import roots), which only a
    # start or a save sweeps, so the backend keeps resolving the old names.
    editor, backends = fake_served
    project = pathlib.Path(editor.uri[len("file://") :]).parent

    async def scenario():
        try:
            editor.open("(= y 1)\n")
            client = backends.client_for(editor.uri)
            await until(lambda: client.running)
            init = project / "pkg" / "__init__.py"
            init.parent.mkdir()
            init.write_text("")
            files_changed(editor, init)
            await asyncio.gather(*backends._tasks)
            params = hover_params(client.backend_uri(editor.uri))
            answer = await backends.request(editor.uri, lsp.TEXT_DOCUMENT_HOVER, params)
            return client.backend_uri(editor.uri), fake_hover(answer)["watched"]
        finally:
            await backends.shutdown_all()

    shadow, watched = run_bounded(scenario())
    assert shadow in watched


def test_an_added_workspace_folder_starts_its_open_documents_backend(
    outside_documents, fake_served
):
    # Defect: only an open starts a backend, so a document that a folder
    # added later puts in a project never gets one, nor its diagnostics.
    editor, outside, mod, folders = outside_documents
    _, backends = fake_served
    backends.configure(None)

    async def scenario():
        try:
            uri = await published_as_script(editor, mod)
            folder = lsp.WorkspaceFolder(uri=outside.as_uri(), name="outside")
            folders.append(folder)
            folders_changed(editor, added=[folder])
            await until(lambda: getattr(backends._existing(uri), "running", False))
        finally:
            await backends.shutdown_all()

    run_bounded(scenario())


def test_the_workspace_index_follows_the_folders(outside_documents):
    # Defect: the workspace is indexed only at initialized, so a folder
    # added later offers none of its definitions and a removed one keeps
    # offering them.
    editor, outside, _, folders = outside_documents
    defs = outside / "defs.lpy"
    defs.write_text("(def far-away [] 1)\n")
    folder = lsp.WorkspaceFolder(uri=outside.as_uri(), name="outside")

    def found():
        query = SimpleNamespace(query="far")
        return [s.location.uri for s in server.workspace_symbol(editor, query)]

    async def scenario():
        folders.append(folder)
        folders_changed(editor, added=[folder])
        assert found() == [defs.as_uri()]
        folders.remove(folder)
        folders_changed(editor, removed=[folder])
        assert found() == []
        indexed = [loc.uri for locs in server.WORKSPACE_INDEX.values() for loc in locs]
        assert defs.as_uri() not in indexed

    run_bounded(scenario())


class Registering:
    """A client double for initialized: dynamic registration of file
    watches advertised or not; every registration request is recorded and
    refused."""

    def __init__(self, dynamic):
        watched = SimpleNamespace(dynamic_registration=dynamic)
        self.client_capabilities = SimpleNamespace(
            workspace=SimpleNamespace(did_change_watched_files=watched)
        )
        self.workspace = SimpleNamespace(folders=None)
        self.requests = []

    def client_register_capability(self, params):
        self.requests.append(params)
        refused = concurrent.futures.Future()
        refused.set_exception(RuntimeError("refused"))
        return refused


def test_module_files_are_watched_only_where_the_client_can_register(caplog):
    # Defects: the watch is registered with a client that cannot take a
    # dynamic registration, or never asked for at all, so no module file
    # created or deleted is ever heard; a refused registration raises out
    # of initialized.
    unable, able = Registering(None), Registering(True)
    server.on_initialized(unable, None)
    server.on_initialized(able, None)
    assert unable.requests == []
    [params] = able.requests
    [registration] = params.registrations
    assert registration.method == lsp.WORKSPACE_DID_CHANGE_WATCHED_FILES
    globs = [w.glob_pattern for w in registration.register_options.watchers]
    for suffix in finder_suffixes():  # every file FileFinder imports
        assert any(fnmatch.fnmatch("ns" + suffix, g.removeprefix("**/")) for g in globs)
    assert "refused" in caplog.text


class ScriptedBackends:
    """A running backend whose published sets the test decides: known maps
    a view text to its (diagnostic, places) pairs."""

    name = "basedpyright"

    def __init__(self):
        self.known = {}
        self.on_diagnostics = None

    def document_opened(self, uri):
        pass

    def document_changed(self, uri, source, view):
        return True

    def document_saved(self, uri):
        pass

    def document_closed(self, uri):
        pass

    def backend_diagnostics(self, uri, text):
        return self.known.get(text)

    def publish(self, src, word):
        """An Error on word in src's view, as the backend publishes it."""
        text = pv.compile_view(src).text
        (line, col), (end_line, end_col) = span(text, word)
        diagnostic = lsp.Diagnostic(
            range=lsp.Range(
                start=lsp.Position(line=line - 1, character=col),
                end=lsp.Position(line=end_line - 1, character=end_col),
            ),
            message=f'"{word}" is not defined',
            code="reportUndefinedVariable",
        )
        self.known[text] = [(diagnostic, [])]


@pytest.fixture
def scripted(tmp_path, monkeypatch):
    """(editor, backends) over ScriptedBackends, the server's state fresh."""
    backends = ScriptedBackends()
    monkeypatch.setattr(server, "backends", backends)
    fresh_server_state(monkeypatch)
    return Editor((tmp_path / "main.lpy").as_uri()), backends


def messages(diagnostics):
    return [d.message for d in diagnostics]


def test_a_late_backend_set_republishes(scripted, monkeypatch):
    # Defect: the backend's set that comes after the change's publish (past
    # DIAGNOSTICS_TIMEOUT, or after a start) is never shown.
    editor, backends = scripted
    monkeypatch.setattr(server, "DIAGNOSTICS_TIMEOUT", 0.05)
    src = "(print undefined-name)\n"

    async def scenario():
        editor.change(src)
        await until(lambda: editor.published)
        assert editor.published == [[]]
        backends.publish(src, "undefined_name")
        server.backend_diagnostics_arrived(editor.uri)
        await until(lambda: len(editor.published) > 1)

    run_bounded(scenario())
    assert messages(editor.published[-1]) == ['"undefined-name" is not defined']


def test_a_newer_change_supersedes_a_waiting_run(scripted, monkeypatch):
    # Defect: a run waiting for the backend's set of an older text publishes
    # after a newer change, so the editor shows the older text's state.
    editor, backends = scripted
    monkeypatch.setattr(server, "DIAGNOSTICS_TIMEOUT", 5)
    newer = "(print undefined-name 2)\n"
    backends.publish(newer, "undefined_name")

    async def scenario():
        editor.change("(print 1)\n")
        await until(lambda: server.DIAGNOSTIC_RUNS[editor.uri].arrived is not None)
        editor.change(newer)
        await until(lambda: editor.published, timeout=2)
        await asyncio.sleep(0.2)  # a superseded run would publish by now

    run_bounded(scenario())
    assert [messages(p) for p in editor.published] == [
        ['"undefined-name" is not defined']
    ]


def test_a_close_clears_the_buffer_and_ends_a_waiting_run(scripted, monkeypatch):
    # Defects: forwarded diagnostics linger in the editor after the buffer
    # closes; a run waiting for the backend publishes after the close.
    editor, backends = scripted
    monkeypatch.setattr(server, "DIAGNOSTICS_TIMEOUT", 5)

    async def scenario():
        editor.change("(print 1)\n")
        await until(lambda: server.DIAGNOSTIC_RUNS[editor.uri].arrived is not None)
        waiting = server.DIAGNOSTIC_RUNS[editor.uri]
        editor.close()
        backends.publish("(print 1)\n", "print")
        server.backend_diagnostics_arrived(editor.uri)
        await until(lambda: waiting.arrived is None, timeout=2)
        await asyncio.sleep(0.1)

    run_bounded(scenario())
    assert editor.published == [[]]


def test_rapid_changes_compile_at_most_twice(scripted, monkeypatch):
    # Defect: every keystroke's compile runs, all its macros included, even
    # once a newer change has made it moot; or the coalesced compile is not
    # of the latest version.
    import threading

    editor, backends = scripted
    gate = threading.Event()
    compiled = []
    real = server.compile_document

    def blocking(uri, version, source, **kwargs):
        compiled.append(source)
        if len(compiled) == 1:
            gate.wait(10)
        return real(uri, version, source, **kwargs)

    monkeypatch.setattr(server, "compile_document", blocking)
    monkeypatch.setattr(server, "DIAGNOSTICS_TIMEOUT", 0)

    async def scenario():
        editor.change("(= a 1)\n")
        await until(lambda: compiled)
        for n in range(2, 7):
            editor.change(f"(= a {n})\n")
            await asyncio.sleep(0)
        editor.change("(= a")  # the latest does not parse
        gate.set()
        await until(lambda: editor.published)
        await asyncio.sleep(0.2)

    run_bounded(scenario())
    assert compiled == ["(= a 1)\n", "(= a"]
    [[diagnostic]] = editor.published
    assert diagnostic.message.startswith("Parse error:")


def test_a_close_before_the_run_goes_on_pushes_nothing(fake_served, monkeypatch):
    # Defect: a close that lands between a change's compile and the rest of
    # its run lets that run push its view: the closed document is opened
    # again in the backend (and on every restart, through _full).
    editor, backends = fake_served
    client = backends.client_for(editor.uri)
    real_schedule = server._schedule_run

    def close_first(coroutine):
        if getattr(coroutine, "__name__", "") == "finish_run":
            editor.close()
        real_schedule(coroutine)

    async def scenario():
        try:
            editor.open("(print undefined-name)\n")
            await until(
                lambda: editor.published and backend_diagnostics(editor.published[-1])
            )
            monkeypatch.setattr(server, "_schedule_run", close_first)
            editor.change("(print 1)\n")
            await until(lambda: editor.published[-1] == [])  # the close's
            await until(lambda: not server._DIAGNOSTIC_TASKS)
            return editor.uri in client._docs, editor.uri in client._full
        finally:
            await backends.shutdown_all()

    assert run_bounded(scenario()) == (False, False)


def test_a_compile_that_raises_never_stops_the_next(scripted, monkeypatch):
    # Defect: a compile that raises ends the document's compile loop, so the
    # change recorded during it is never compiled or published.
    import threading

    editor, backends = scripted
    gate = threading.Event()
    calls = []
    real = server.compile_document

    def failing_first(uri, version, source, **kwargs):
        calls.append(source)
        if len(calls) == 1:
            gate.wait(10)
            raise RuntimeError("compile failed")
        return real(uri, version, source, **kwargs)

    monkeypatch.setattr(server, "compile_document", failing_first)

    async def scenario():
        editor.change("(= a 1)\n")
        await until(lambda: calls)
        editor.change("(= a")
        gate.set()
        await until(lambda: editor.published)

    run_bounded(scenario())
    [[diagnostic]] = editor.published
    assert diagnostic.message.startswith("Parse error:")


def drained():
    return not server._COMPILING and not server._DIAGNOSTIC_TASKS


def test_a_compile_that_never_returns_stops_holding_up_its_document(
    scripted, monkeypatch
):
    # Defect: a compile that never returns (a macro looping forever) keeps
    # its document's compile loop waiting on it, so no later change of the
    # document is ever compiled or published.
    editor, backends = scripted
    gate, started = threading.Event(), threading.Event()
    real = server.compile_document

    def hung_first(uri, version, source, **kwargs):
        if version == 1:
            started.set()
            gate.wait(10)
        return real(uri, version, source, **kwargs)

    monkeypatch.setattr(server, "compile_document", hung_first)
    monkeypatch.setattr(macro_modules, "WAIT_LIMIT", 0.1)
    monkeypatch.setattr(server, "DIAGNOSTICS_TIMEOUT", 0)

    async def scenario():
        editor.change("(= a 1)\n")
        await until(started.is_set)
        editor.change("(= a")
        await until(lambda: editor.published, timeout=3)

    try:
        run_bounded(scenario())
    finally:
        gate.set()
    [[diagnostic]] = editor.published
    assert diagnostic.message.startswith("Parse error:")


def test_a_compile_that_raises_publishes_an_internal_error(scripted, monkeypatch):
    # Defect: a change whose compile raises (outside compile-buffer's
    # stages) publishes nothing, so the editor keeps the diagnostics of an
    # older text.
    editor, backends = scripted

    def failing(uri, version, source, **kwargs):
        raise RuntimeError("cache broke")

    monkeypatch.setattr(server, "compile_document", failing)

    async def scenario():
        editor.change("(= a 1)\n")
        await until(lambda: editor.published, timeout=3)
        await until(drained, timeout=3)

    run_bounded(scenario())
    [[diagnostic]] = editor.published
    assert diagnostic.severity == lsp.DiagnosticSeverity.Error
    assert "cache broke" in diagnostic.message


def test_a_macro_exiting_publishes_an_internal_error(scripted, monkeypatch):
    # Defect: a macro calling sys.exit raises SystemExit out of the compile
    # thread, past the compile loop's except Exception, so nothing is
    # published and the SystemExit reaches the event loop.
    editor, backends = scripted
    real = server.compile_document

    def exiting(uri, version, source, **kwargs):
        if version == 1:
            raise SystemExit(3)
        return real(uri, version, source, **kwargs)

    monkeypatch.setattr(server, "compile_document", exiting)
    monkeypatch.setattr(server, "DIAGNOSTICS_TIMEOUT", 0)

    async def scenario():
        editor.change("(= a 1)\n")
        await until(lambda: editor.published, timeout=3)
        editor.change("(= a")  # the loop still compiles
        await until(lambda: len(editor.published) == 2, timeout=3)

    run_bounded(scenario())
    [internal], [parse] = editor.published
    assert "SystemExit" in internal.message
    assert parse.message.startswith("Parse error:")


def held_runs(editor):
    """editor's document, its uri held as compiling, so a publish only
    records its run."""
    editor.document = TextDocument(editor.uri, "(= a 1)\n", version=1)
    server._COMPILING.add(editor.uri)


def test_a_refresh_keeps_a_pending_changes_recompile_and_refresh(scripted):
    # Defect: a refresh run replacing a save's run that has not compiled
    # yet drops its recompile (the save's macros never reread the disk)
    # and makes it a refresh run, so the save refreshes no other document.
    editor, _ = scripted
    held_runs(editor)
    server.publish_diagnostics(editor, editor.uri, recompile=True)
    server.publish_diagnostics(editor, editor.uri, refresh=True)
    run = server.DIAGNOSTIC_RUNS[editor.uri]
    assert (run.recompile, run.refresh) == (True, False)
    del server.DIAGNOSTIC_RUNS[editor.uri]
    server.publish_diagnostics(editor, editor.uri, refresh=True)
    server.publish_diagnostics(editor, editor.uri, refresh=True)
    run = server.DIAGNOSTIC_RUNS[editor.uri]
    assert (run.recompile, run.refresh) == (False, True)


def refreshes(monkeypatch, runs, now, refreshed_epoch, compiled_epoch):
    """The uris refresh_outdated refreshes for a run of "self" compiled
    under compiled_epoch, with the documents' runs at the epochs runs maps
    them to, the epoch at now and _REFRESHED_EPOCH at refreshed_epoch."""
    refreshed = []
    monkeypatch.setattr(server, "refresh_epochs", lambda: (now, now))
    monkeypatch.setattr(
        server, "publish_diagnostics", lambda ls, uri, **kw: refreshed.append(uri)
    )
    monkeypatch.setattr(server, "_REFRESHED_EPOCH", refreshed_epoch)
    for uri, epoch in runs.items():
        server.DIAGNOSTIC_RUNS[uri] = SimpleNamespace(epoch=epoch)
    compiled = SimpleNamespace(epoch=compiled_epoch)
    asyncio.run(server.refresh_outdated(None, "self", compiled))
    return refreshed


def test_a_refresh_keeps_a_compiling_saves_recompile_and_refresh(scripted):
    # Defect: a refresh run (a republish) replacing a save's run whose
    # compile is in flight drops its recompile and its refresh: the save's
    # refresh-outdated never runs, and if that compile is given up the new
    # run serves the cached view from before the save.
    editor, _ = scripted
    held_runs(editor)
    server.publish_diagnostics(editor, editor.uri, recompile=True)
    server.DIAGNOSTIC_RUNS[editor.uri].compiled = True  # compiling
    server.publish_diagnostics(editor, editor.uri, refresh=True)
    run = server.DIAGNOSTIC_RUNS[editor.uri]
    assert (run.recompile, run.refresh) == (True, False)


def test_a_run_under_an_older_epoch_refreshes_nothing(scripted, monkeypatch):
    # Defect: a slow run finishing under an epoch the documents were
    # already refreshed for refreshes them all again.
    runs = {"self": 5, "old": 3}
    assert refreshes(monkeypatch, runs, 5, 5, 4) == []
    assert refreshes(monkeypatch, runs, 5, 5, 5) == []
    assert refreshes(monkeypatch, runs, 6, 5, 6) == ["old"]


def test_a_run_with_no_epoch_is_not_refreshed(scripted, monkeypatch):
    # Defect: a document whose compile raises (its run has no epoch) counts
    # as outdated at every refresh, so it is compiled again every round.
    runs = {"self": 6, "failing": None, "old": 3}
    assert refreshes(monkeypatch, runs, 6, 5, 6) == ["old"]


def finished(editor, epoch):
    """(run, latest): a change of editor's document whose compile ran under
    epoch, finished (finish_run), and the document's latest run after."""
    server.publish_diagnostics(editor, editor.uri)
    run = server.DIAGNOSTIC_RUNS[editor.uri]
    run.compiled = True
    compiled = server.compile_document(editor.uri, None, editor.document.source)
    compiled.epoch = epoch
    asyncio.run(server.finish_run(run, compiled))
    return run, server.DIAGNOSTIC_RUNS[editor.uri]


def test_a_run_compiled_before_a_refresh_publishes_again(scripted, monkeypatch):
    # Defect: a run compiling while a refresh happened (skipped by it, with
    # no epoch yet) publishes a compile from the older macro environment
    # and stays stale until its next edit.
    editor, _ = scripted
    monkeypatch.setattr(server, "DIAGNOSTICS_TIMEOUT", 0)
    held_runs(editor)
    monkeypatch.setattr(server, "_REFRESHED_EPOCH", 10)
    monkeypatch.setattr(server, "_REFRESH_TARGET", 10)
    run, latest = finished(editor, 3)
    assert latest is not run and latest.refresh and not latest.compiled
    run, latest = finished(editor, 10)  # no older than the refresh: done
    assert latest is run
    assert len(editor.published) == 2


def test_a_rerun_below_the_target_is_not_rerun(scripted, monkeypatch):
    # Defect: a rerun whose compile still runs under an epoch below the
    # refresh target (one no compile reaches: a hung compile blocks the
    # drop, or the staleness vanished) reruns itself forever.
    editor, _ = scripted
    monkeypatch.setattr(server, "DIAGNOSTICS_TIMEOUT", 0)
    held_runs(editor)
    monkeypatch.setattr(server, "_REFRESHED_EPOCH", 10)
    monkeypatch.setattr(server, "_REFRESH_TARGET", 10)
    _, rerun = finished(editor, 3)
    assert rerun.rerun
    rerun.compiled = True
    compiled = server.compile_document(editor.uri, None, editor.document.source)
    compiled.epoch = 3
    asyncio.run(server.finish_run(rerun, compiled))
    assert server.DIAGNOSTIC_RUNS[editor.uri] is rerun


def test_a_rerun_that_missed_a_later_refresh_reruns(scripted, monkeypatch):
    # Defect: a rerun's compile ran past a second drop, so it finishes
    # under an epoch below the newer refresh target too, and, being a
    # rerun, never runs again: the document keeps diagnostics from before
    # the second drop until its next edit.
    editor, _ = scripted
    monkeypatch.setattr(server, "DIAGNOSTICS_TIMEOUT", 0)
    held_runs(editor)
    monkeypatch.setattr(server, "_REFRESHED_EPOCH", 10)
    monkeypatch.setattr(server, "_REFRESH_TARGET", 10)
    _, rerun = finished(editor, 3)
    assert rerun.rerun == 10
    monkeypatch.setattr(server, "_REFRESH_TARGET", 20)  # a second refresh
    rerun.compiled = True
    compiled = server.compile_document(editor.uri, None, editor.document.source)
    compiled.epoch = 3
    asyncio.run(server.finish_run(rerun, compiled))
    again = server.DIAGNOSTIC_RUNS[editor.uri]
    assert again is not rerun and again.rerun == 20 and not again.compiled
    # Still under the target, with no newer one: it settles.
    again.compiled = True
    asyncio.run(server.finish_run(again, compiled))
    assert server.DIAGNOSTIC_RUNS[editor.uri] is again


def test_a_parse_error_under_an_unreached_target_settles(scripted, monkeypatch):
    # Defect: a parse error runs no macro, so it never enters fresh-modules
    # and never reaches a refresh target past its epoch (a drop due): it
    # republishes itself forever.
    editor, _ = scripted
    monkeypatch.setattr(server, "_REFRESH_TARGET", macro_modules.current_epoch() + 5)

    async def scenario():
        editor.change("(= a")
        await until(lambda: editor.published, timeout=3)
        await until(drained, timeout=3)

    run_bounded(scenario())
    [[diagnostic]] = editor.published
    assert diagnostic.message.startswith("Parse error:")


def test_the_refresh_target_is_read_with_the_epoch(monkeypatch):
    # Defect: the epoch and the counter are read under two holds of the
    # lock, so a drop landing between them gives a target past the epoch
    # every later compile runs under, and every finishing run republishes.
    def due_then_dropped():  # a drop due, then made before the next read
        with macro_modules._SHARED.condition:
            macro_modules._SHARED.epoch += 1
        return None

    monkeypatch.setattr(macro_modules, "epoch", due_then_dropped)
    _, target = server.refresh_epochs()
    assert target <= macro_modules.current_epoch()  # no drop is due now


def test_a_run_compiled_before_a_due_drop_publishes_again(scripted, monkeypatch):
    # Defect: a run compiling while a refresh happened is checked against
    # the epoch of the run that started the refresh, not the environment
    # the refresh brought the documents to: with a drop due then, a run
    # under the triggering epoch, or one between it and the drop, keeps a
    # compile from before the drop.
    editor, _ = scripted
    monkeypatch.setattr(server, "DIAGNOSTICS_TIMEOUT", 0)
    monkeypatch.setattr(macro_modules, "epoch_and_target", lambda: (None, 8))
    asyncio.run(server.refresh_outdated(None, "trigger", SimpleNamespace(epoch=7)))
    held_runs(editor)
    run, latest = finished(editor, 7)
    assert latest is not run and latest.refresh


# ---------------------------------------------------------------------------
# The pyrefly preset fallback notice
# ---------------------------------------------------------------------------


def preset_server(
    tmp_path, monkeypatch, name, init_options=None, config=None, refuses=None, argv=None
):
    """(project, backends, shown): a server over one project whose backend
    name (alone installed) is the fake backend, or argv, configured by
    init_options, and the messages it shows. config is (file name, text) of a
    project file, and refuses a pyrefly.toml value the installed pyrefly is
    taken to refuse."""
    project = tmp_path / "project"
    project.mkdir()
    if config is not None:
        (project / config[0]).write_text(config[1])
    if argv is None:
        script = tmp_path / "fake_backend.py"
        script.write_text(FAKE_SERVER)
        argv = [sys.executable, str(script)]
    for other in ("pyrefly", "basedpyright"):
        installed = argv if other == name else None
        monkeypatch.setitem(backend.BACKENDS, other, lambda a=installed: a)
    monkeypatch.setattr(
        backend,
        "pyrefly_config_problem",
        lambda binary, text: (
            f"unknown variant `{refuses}`" if refuses and refuses in text else None
        ),
    )
    folder = SimpleNamespace(uri=project.as_uri())
    backends = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "cache"))
    backends.configure(init_options)
    backends.on_view_prepared = server.notify_preset_fallback
    monkeypatch.setattr(server, "backends", backends)
    fresh_server_state(monkeypatch)
    shown = []
    monkeypatch.setattr(
        server, "server", SimpleNamespace(window_show_message=shown.append)
    )
    return project, backends, shown


async def started(client):
    """Once client's latest start, if any, has ended."""
    await until(lambda: client._starting is None or client._starting.done())


async def open_and_request(project, backends):
    """Open main.lpy and other.lpy, ask a hover (which starts a backend that
    no open started, pythonDiagnostics off), close both, and return the
    project's BackendClient once its start ended."""
    editors = [Editor((project / f).as_uri()) for f in ("main.lpy", "other.lpy")]
    for editor in editors:
        editor.open("(= y 1)\n")
        await until(lambda: editor.published)
    uri = editors[0].uri
    client = backends.client_for(uri)
    await backends.request(
        uri, lsp.TEXT_DOCUMENT_HOVER, hover_params(client.backend_uri(uri))
    )
    await started(client)
    for editor in editors:
        editor.close()
    return client


def shown_after_start(tmp_path, monkeypatch, *args, **kwargs):
    """(project, view root, shown) once a server of preset_server's opened
    two buffers of the project and its backend started."""
    project, backends, shown = preset_server(tmp_path, monkeypatch, *args, **kwargs)

    async def scenario():
        try:
            client = await open_and_request(project, backends)
            assert client.running
            return client.view_root
        finally:
            await backends.shutdown_all()

    return project, run_bounded(scenario()), shown


@pytest.mark.parametrize(
    "config, refuses, reason, names",
    [
        (None, None, "missing", None),
        (("pyproject.toml", '[project]\nname = "p"\n'), None, "missing", None),
        (("pyproject.toml", "[tool.pyrefly\n"), None, "malformed", "pyproject.toml"),
        (("pyrefly.toml", "preset = \n"), None, "malformed", "pyrefly.toml"),
        (
            ("pyrefly.toml", 'untyped-def-behavior = "nope"\n'),
            "nope",
            "refused",
            "pyrefly.toml",
        ),
    ],
    ids=[
        "none",
        "pyproject-without-pyrefly",
        "bad-pyproject",
        "bad-pyrefly",
        "refused",
    ],
)
def test_a_project_whose_pyrefly_config_is_not_used_is_told_once(
    tmp_path, monkeypatch, config, refuses, reason, names
):
    # Defects: the user is never told that the view runs a preset of its own,
    # is told on every open, or is told the project has no config when it
    # has one the view could not read or pyrefly refused (the notice must
    # name that file, the one to fix).
    project, root, shown = shown_after_start(
        tmp_path, monkeypatch, "pyrefly", config=config, refuses=refuses
    )
    assert pv.pyrefly_fallback(root).reason == reason
    [notice] = shown
    assert notice.type == lsp.MessageType.Info
    if names is not None:
        assert str(project / names) in notice.message


@pytest.mark.parametrize(
    "name, init_options, config",
    [
        ("pyrefly", None, ("pyrefly.toml", 'preset = "basic"\n')),
        (
            "pyrefly",
            None,
            ("pyproject.toml", '[tool.pyrefly]\npython-version = "3.12"\n'),
        ),
        ("pyrefly", {"pythonDiagnostics": False}, None),
        ("basedpyright", None, None),
    ],
    ids=["pyrefly.toml", "tool.pyrefly", "no-diagnostics", "basedpyright"],
)
def test_no_notice_where_the_preset_is_not_the_views(
    tmp_path, monkeypatch, name, init_options, config
):
    # Defects: the notice nags a project whose own pyrefly config decides
    # its preset, one whose Python diagnostics are off (nothing is checked,
    # though a request started the backend), or a basedpyright user, whose
    # diagnostics no pyrefly preset decides.
    _, _, shown = shown_after_start(tmp_path, monkeypatch, name, init_options, config)
    assert shown == []


def test_a_failed_start_tells_nothing(tmp_path, monkeypatch):
    # Defect: the notice comes when the view root is written, before the
    # backend process starts, so a backend that never ran (not runnable,
    # shut down while starting) tells of a preset nothing checks with.
    project, backends, shown = preset_server(
        tmp_path, monkeypatch, "pyrefly", argv=[str(tmp_path / "no-such-backend")]
    )

    async def scenario():
        try:
            client = await open_and_request(project, backends)
            assert not client.running
            return client.view_root
        finally:
            await backends.shutdown_all()

    root = run_bounded(scenario())
    # The view was written with the fallback: only the start kept it quiet.
    assert pv.pyrefly_fallback(root).reason == "missing"
    assert shown == []


def test_a_config_that_breaks_later_is_told_at_the_next_start(tmp_path, monkeypatch):
    # Defects: a project whose config was fine at the first start is marked
    # as told, so the config breaking later (the backend restarting over
    # the fallback) is never told; or each start over the fallback tells
    # it again.
    monkeypatch.setattr(backend, "MAX_SPAWNS", 3)
    project, backends, shown = preset_server(
        tmp_path, monkeypatch, "pyrefly", config=("pyrefly.toml", 'preset = "basic"\n')
    )

    async def restart(client):
        process = client.process
        process.kill()
        await process.wait()
        client.start_soon()
        await started(client)
        assert client.running and client.process is not process

    async def scenario():
        try:
            client = await open_and_request(project, backends)
            assert client.running and shown == []
            (project / "pyrefly.toml").write_text("preset = \n")
            await restart(client)
            assert pv.pyrefly_fallback(client.view_root).reason == "malformed"
            assert len(shown) == 1
            await restart(client)
            assert len(shown) == 1
        finally:
            await backends.shutdown_all()

    run_bounded(scenario())
    assert str(project / "pyrefly.toml") in shown[0].message


# ---------------------------------------------------------------------------
# Real backends
# ---------------------------------------------------------------------------


# Each real backend's diagnostic on an undefined name, as (code, message in
# the buffer's spelling, or None where only the backticked buffer spelling is
# pinned: nox installs the newest pyrefly, whose wording may change).
UNDEFINED_NAME = {
    "basedpyright": ("reportUndefinedVariable", '"undefined-name" is not defined'),
    "pyrefly": ("unknown-name", None),
}


def test_a_real_backend_reports_on_the_lpy_word(
    run_on_loop, real_backends, backend_project, monkeypatch
):
    # Defects, end to end: the backend's diagnostics never reach the
    # buffer, land on generated or prelude code, or keep Python spelling.
    name = real_backends.name
    monkeypatch.setattr(
        real_backends, "on_diagnostics", server.backend_diagnostics_arrived
    )
    monkeypatch.setattr(server, "backends", real_backends)
    fresh_server_state(monkeypatch)
    editor = Editor((backend_project / "main.lpy").as_uri())

    async def scenario():
        try:
            editor.open("(print undefined-name)\n")
            await until(
                lambda: (
                    editor.published and backend_diagnostics(editor.published[-1], name)
                ),
                timeout=30,
            )
            return backend_diagnostics(editor.published[-1], name)
        finally:
            server.did_close(editor, SimpleNamespace(text_document=editor.document))

    [found] = run_on_loop(scenario())
    code, message = UNDEFINED_NAME[name]
    assert found.code == code
    assert found.severity == lsp.DiagnosticSeverity.Error
    if message is None:
        assert "`undefined-name`" in found.message
        assert "undefined_name" not in found.message
    else:
        assert found.message == message
    assert (found.range.start.line, found.range.start.character) == (0, 7)
    assert (found.range.end.line, found.range.end.character) == (0, 21)


# The buffer names published_marks opens: one per call, never on disk.
MARKS_BUFFERS = (f"marks_{n}.lpy" for n in itertools.count(1))


def published_marks(
    run_on_loop, real_backends, backend_project, monkeypatch, source, count
):
    """(lispython's, the backend's) diagnostics of source opened as a new
    unsaved buffer of backend_project, each as (line, the .lpy text it
    underlines), once the backend published at least count of its own (30
    s at most).  Each call opens a name of its own (MARKS_BUFFERS): the
    backend process is shared, and a late set it publishes for another
    test's copy of one name could otherwise be taken for this one's."""
    name = real_backends.name
    monkeypatch.setattr(
        real_backends, "on_diagnostics", server.backend_diagnostics_arrived
    )
    monkeypatch.setattr(server, "backends", real_backends)
    fresh_server_state(monkeypatch)
    editor = Editor((backend_project / next(MARKS_BUFFERS)).as_uri())

    async def scenario():
        try:
            editor.open(source)
            await until(
                lambda: (
                    editor.published
                    and len(backend_diagnostics(editor.published[-1], name)) >= count
                ),
                timeout=30,
            )
            return editor.published[-1]
        finally:
            server.did_close(editor, SimpleNamespace(text_document=editor.document))

    lines = source.split("\n")

    def marks(diagnostics):
        return sorted(
            (
                d.range.start.line,
                lines[d.range.start.line][
                    d.range.start.character : d.range.end.character
                ],
            )
            for d in diagnostics
        )

    diagnostics = run_on_loop(scenario())
    own = [d for d in diagnostics if d.source == "lpy-lsp"]
    return marks(own), marks(backend_diagnostics(diagnostics, name))


# A buffer with failed forms: a def with no body (whose signature the
# compiler accepts), a class whose method has none, a user macro's call that
# expands to a failing def, and uses of each beside an undefined name.
FAILED_FORMS_BUFFER = (
    "(def f [x])\n"
    "(class C []\n"
    "  (def m [self]))\n"
    "(defmacro mk [name] (return `(def ~name [] (while))))\n"
    "(mk q)\n"
    "(= c ^C (C 1))\n"
    "(print (f 1) (. (C) attr) c q undefined-name)\n"
    "(f 1 2)\n"
)


def test_a_failed_form_keeps_the_rest_of_the_buffers_backend_diagnostics(
    run_on_loop, real_backends, backend_project, monkeypatch
):
    # Defects: one failed form costs the whole buffer its view, so the
    # backend reports nothing for it (undefined-name is lost); a failed
    # def or class leaves its name undefined at every use, a false
    # diagnostic on f, C and ^C; the class placeholder is a variable, which
    # an annotation refuses; a call of the failed def with too many
    # arguments is no longer reported; a failed macro call declares a name
    # its expansion might bind (q is undefined: the form is left out).
    own, found = published_marks(
        run_on_loop, real_backends, backend_project, monkeypatch, FAILED_FORMS_BUFFER, 3
    )
    assert own == [(0, "(def"), (2, "(def"), (4, "(mk")]
    assert found == [(6, "q"), (6, "undefined-name"), (7, "2")]


@pytest.mark.parametrize(
    "source",
    [
        "(= x (while))\n(def x [a]\n  (return a))\n(print undefined-name)\n",
        "(def x [a]\n  (return a))\n(= x (while))\n(print undefined-name)\n",
    ],
    ids=["placeholder-first", "def-first"],
)
def test_a_failed_assignment_beside_a_def_of_its_name_reports_neither(
    run_on_loop, real_backends, backend_project, monkeypatch, source
):
    # Defect: the failed (= x ...)'s placeholder is a second declaration of
    # the user's def x, so basedpyright reports a redeclaration (and an
    # assignment type error) on one x or the other, though neither is wrong.
    _, found = published_marks(
        run_on_loop, real_backends, backend_project, monkeypatch, source, 1
    )
    assert found == [(3, "undefined-name")]


@basedpyright_only
def test_a_placeholder_is_unreported_in_a_strict_mode(
    run_on_loop, real_backends, monkeypatch, tmp_path
):
    # Defect: what basedpyright's recommended mode reports on a failed
    # form's placeholder (reportAny "Type of x is Any") lands on the user's
    # x, which the placeholder's target maps onto for navigation.
    # A project and backend of its own, beside the session's (bounded: one
    # short-lived basedpyright, stopped in the finally): switching the
    # session's backend to recommended and back through a save leaks the
    # mode into the next tests, since basedpyright applies a config change
    # only some time after it is told, with no signal to wait on.
    project = tmp_path / "strict-project"
    project.mkdir()
    (project / "main.lpy").write_text("(= y 1)\n")
    (project / "pyrightconfig.json").write_text('{"typeCheckingMode": "recommended"}')
    folder = SimpleNamespace(uri=project.as_uri())
    strict = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "cache"))
    strict.configure({"backend": real_backends.name})
    try:
        _, found = published_marks(
            run_on_loop,
            strict,
            project,
            monkeypatch,
            "(= x (while))\n(print undefined-name)\n",
            1,
        )
    finally:
        run_on_loop(strict.shutdown_all())
    # recommended may report undefined-name more than once.
    assert set(found) == {(1, "undefined-name")}


def test_an_importer_of_a_failed_module_sees_its_names(
    run_on_loop, real_backends, backend_project, monkeypatch
):
    # Defect: a module none of whose forms compiles has no shadow, so every
    # importer gets a false unresolved import and loses the arity check of
    # its failed def's calls (broken_mod.lpy, conftest's backend_project).
    source = (
        "(from broken-mod [broken-fn BrokenCls])\n"
        "(print (broken-fn 1) (BrokenCls) undefined-name)\n"
        "(broken-fn 1 2)\n"
    )
    _, found = published_marks(
        run_on_loop, real_backends, backend_project, monkeypatch, source, 2
    )
    assert found == [(1, "undefined-name"), (2, "2")]


# ---------------------------------------------------------------------------
# ; pyright: ignore, ; pyrefly: ignore
# ---------------------------------------------------------------------------

SUPPRESSED = (
    "(print undefined-a) ; pyright: ignore[reportUndefinedVariable, reportX]\n"
    "(print undefined-b) ;pyright:ignore\n"
    "(print undefined-c)\n"
    '(print "; pyright: ignore it" undefined-d)\n'
)


class StubBackends:
    """A backend, basedpyright unless name says otherwise, that published
    diagnostics (code, generated text) for the view's text."""

    def __init__(self, published, name="basedpyright"):
        self.published = published
        self.name = name

    def backend_diagnostics(self, uri, text):
        found = []
        for code, word in self.published:
            (line, col), (end_line, end_col) = span(text, word)
            found.append(
                lsp.Diagnostic(
                    range=lsp.Range(
                        start=lsp.Position(line=line - 1, character=col),
                        end=lsp.Position(line=end_line - 1, character=end_col),
                    ),
                    message="m",
                    code=code,
                )
            )
        return [(diagnostic, []) for diagnostic in found]


def survivors(monkeypatch, published, source=SUPPRESSED, name="basedpyright"):
    """(code, .lpy line) of each backend diagnostic published on source's
    view that the server forwards, with backend name."""
    monkeypatch.setattr(server, "backends", StubBackends(published, name))
    doc = TextDocument("file:///buffer.lpy", source, version=1)
    view = pv.compile_view(source)
    run = server.DiagnosticRun(None, doc)
    run.view = view
    found = asyncio.run(server.mapped_backend_diagnostics(run))
    return [(d.code, d.range.start.line + 1) for d in found]


def test_a_listed_ignore_drops_only_its_rules(monkeypatch):
    # Defect: ignore[rule] drops every diagnostic on its line, so a call
    # error there vanishes along with the undefined name it was asked for.
    published = [
        ("reportUndefinedVariable", "undefined_a"),
        ("reportCallIssue", "print(undefined_a)"),
    ]
    assert survivors(monkeypatch, published) == [("reportCallIssue", 1)]


def test_a_bare_ignore_drops_every_rule_on_its_line(monkeypatch):
    # Defect: a bare ignore is read as an empty rule list and drops nothing,
    # or its unspaced form ;pyright:ignore is not recognised.
    published = [
        ("reportUndefinedVariable", "undefined_b"),
        ("reportCallIssue", "print(undefined_b)"),
    ]
    assert survivors(monkeypatch, published) == []


def test_an_ignore_leaves_the_next_line_alone(monkeypatch):
    # Defect: an ignore reaches past its own line (to the next form).
    published = [("reportUndefinedVariable", "undefined_c")]
    assert survivors(monkeypatch, published) == [("reportUndefinedVariable", 3)]


def test_an_ignore_inside_a_string_is_no_comment(monkeypatch):
    # Defect: the comment is found by searching the line for ";", so a
    # string holding "; pyright: ignore" suppresses its line.
    published = [("reportUndefinedVariable", "undefined_d")]
    assert survivors(monkeypatch, published) == [("reportUndefinedVariable", 4)]


PER_TOOL = (
    "(print undefined-a) ; pyrefly: ignore[unknown-name, bad-call]\n"
    "(print undefined-b) ;pyrefly:ignore\n"
    "(print undefined-c) ; pyright: ignore\n"
)


@pytest.mark.parametrize(
    "name, kept",
    [("pyrefly", [3]), ("basedpyright", [1, 2]), ("fake", [1, 2])],
)
def test_each_backend_honours_only_its_own_suppressions(monkeypatch, name, kept):
    # Defects: pyrefly's own ; pyrefly: ignore comments are ignored, so a
    # pyrefly user cannot silence a line; or a ; pyright: ignore silences
    # pyrefly (or the reverse), which the tool itself never does, so the
    # same line reads differently in the editor and under the tool.
    published = [("unknown-name", f"undefined_{c}") for c in "abc"]
    found = survivors(monkeypatch, published, PER_TOOL, name)
    assert [line for _, line in found] == kept
