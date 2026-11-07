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
import pathlib
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


def test_related_locations_are_placed_when_published(tmp_path):
    # Defect: a related location is placed (locate) only when the server
    # maps the set, after a later sync (a request's cursor view) has
    # replaced the origin of the text the backend analysed, so it maps
    # through the wrong view.
    project = tmp_path / "project"
    project.mkdir()
    client = backend.BackendClient(
        pv.view_root_for(str(project), cache_home=str(tmp_path / "cache")), None
    )
    process = object()
    client._client = process
    uri = (project / "main.lpy").as_uri()
    shadow = client.backend_uri(uri)
    full, full_origin = "x = 1", ("(= x 1)", object())
    client._full[uri] = (full, full_origin)
    client._docs[uri] = [shadow, 2, full, full_origin]
    here = lsp.Range(
        start=lsp.Position(line=0, character=0), end=lsp.Position(line=0, character=1)
    )
    related = lsp.DiagnosticRelatedInformation(
        location=lsp.Location(uri=shadow, range=here), message="see"
    )
    diagnostic = lsp.Diagnostic(range=here, message="m", related_information=[related])
    client._diagnostics_published(process, published(shadow, 2, [diagnostic]))
    client._docs[uri][3] = ("(= x 1)", object())  # a later cursor sync
    [(_, places)] = client.diagnostics_for(uri, full)
    assert places == [("lpy", uri, full_origin)]


# ---------------------------------------------------------------------------
# The server's flow, over the fake backend
# ---------------------------------------------------------------------------


class Editor:
    """A language server double for one document: its workspace holds the
    buffer, and publishes and shown messages are recorded."""

    def __init__(self, uri):
        self.uri = uri
        self.workspace = self
        self.document = None
        self.published = []
        self.shown = []
        self.versions = itertools.count(1)

    def get_text_document(self, uri):
        return self.document

    def text_document_publish_diagnostics(self, params):
        self.published.append(params.diagnostics)

    def window_show_message(self, params):
        self.shown.append(params)

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


# ---------------------------------------------------------------------------
# The pyrefly preset fallback notice
# ---------------------------------------------------------------------------


def shown_on_open(tmp_path, monkeypatch, name, init_options=None, config=None):
    """The messages the server shows while main.lpy and other.lpy of one
    project are opened, with backend name installed (alone) and configured
    by init_options; config is (file name, text) of a project file. The
    backend's start is stubbed (no process): the notice reads the
    configuration, not the backend's answers."""
    project = tmp_path / "project"
    project.mkdir()
    if config is not None:
        (project / config[0]).write_text(config[1])
    for other in ("pyrefly", "basedpyright"):
        argv = ["unused"] if other == name else None
        monkeypatch.setitem(backend.BACKENDS, other, lambda argv=argv: argv)
    monkeypatch.setattr(backend.BackendClient, "start_soon", lambda self: None)
    folder = SimpleNamespace(uri=project.as_uri())
    backends = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "cache"))
    backends.configure(init_options)
    monkeypatch.setattr(server, "backends", backends)
    fresh_server_state(monkeypatch)
    editors = [Editor((project / f).as_uri()) for f in ("main.lpy", "other.lpy")]

    async def scenario():
        for editor in editors:
            editor.open("(= y 1)\n")
        await until(lambda: all(editor.published for editor in editors))
        for editor in editors:
            editor.close()

    run_bounded(scenario())
    return project, [message for editor in editors for message in editor.shown]


def test_a_project_without_pyrefly_config_is_told_once(tmp_path, monkeypatch):
    # Defects: the user is never told that the view runs a preset of its own
    # (pyrefly check over the same project reports far less), or is told on
    # every open of the project's documents.
    project, shown = shown_on_open(tmp_path, monkeypatch, "pyrefly")
    [notice] = shown
    assert notice.type == lsp.MessageType.Info
    assert str(project.resolve()) in notice.message
    assert '"default"' in notice.message
    assert '"basic"' in notice.message
    assert "pyrefly.toml" in notice.message and "[tool.pyrefly]" in notice.message


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
    # its preset, one whose Python diagnostics are off (nothing is checked),
    # or a basedpyright user, whose diagnostics no pyrefly preset decides.
    _, shown = shown_on_open(tmp_path, monkeypatch, name, init_options, config)
    assert shown == []


# ---------------------------------------------------------------------------
# Real backends
# ---------------------------------------------------------------------------


# Each real backend's diagnostic on an undefined name, as (code, message in
# the buffer's spelling).
UNDEFINED_NAME = {
    "basedpyright": ("reportUndefinedVariable", '"undefined-name" is not defined'),
    "pyrefly": ("unknown-name", "Could not find name `undefined-name`"),
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
    assert found.message == message
    assert (found.range.start.line, found.range.start.character) == (0, 7)
    assert (found.range.end.line, found.range.end.character) == (0, 21)


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
