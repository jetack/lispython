"""The LSP's Python backend client (lispy.lsp.backend).

The real-backend tests share one basedpyright process on one event loop in a
background thread (a basedpyright start costs about a second and 400 MB). The
others run a fake backend, a tiny pygls server whose hover answers with its
pid, its copy of the document and the configuration the client gave it, and
which publishes an Error on each undefined_name of a synced text, each on its
own loop; none needs basedpyright installed.
"""

import asyncio
import contextlib
import importlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import textwrap
import threading
import time
import tomllib
from types import SimpleNamespace

import pytest
from lsprotocol import types as lsp
from pygls.lsp.client import LanguageClient

import lispy  # noqa: F401  (registers the .lpy importer)

backend = importlib.import_module("lispy.lsp.backend")
pv = importlib.import_module("lispy.lsp.python_view")

# argv: [--slow-init SECONDS] [--linger].  --slow-init delays the answer to
# initialize; --linger keeps the process alive 30 s after its stdin closes
# with no exit notification, as a backend that does not notice its client is
# gone.
FAKE_SERVER = textwrap.dedent(
    """\
    import asyncio
    import json
    import os
    import sys
    import time

    from lsprotocol import types
    from pygls.lsp.server import LanguageServer

    server = LanguageServer("fake-backend", "0")
    config = {}


    @server.feature(types.INITIALIZE)
    async def initialize(ls, params):
        if "--slow-init" in sys.argv:
            await asyncio.sleep(float(sys.argv[sys.argv.index("--slow-init") + 1]))


    @server.feature(types.INITIALIZED)
    async def initialized(ls, params):
        items = [types.ConfigurationItem(section=s) for s in ("python", "x.y")]
        config["answer"] = await ls.workspace_configuration_async(
            types.ConfigurationParams(items=items)
        )


    @server.feature(types.TEXT_DOCUMENT_HOVER)
    async def hover(ls, params):
        for _ in range(500):
            if "answer" in config:
                break
            await asyncio.sleep(0.01)
        # The column is a delay in tenths of a second before the document is
        # read, so requests can overlap.
        await asyncio.sleep(params.position.character / 10)
        doc = ls.workspace.get_text_document(params.text_document.uri)
        text = doc.source
        answer = {"pid": os.getpid(), "text": text, "config": config.get("answer")}
        return types.Hover(contents=json.dumps(answer))


    def publish(ls, document):
        # As pyright: an Error on each undefined_name, at the synced version.
        text = ls.workspace.get_text_document(document.uri).source
        found = []
        for row, line in enumerate(text.split("\\n")):
            col = line.find("undefined_name")
            while col != -1:
                start = types.Position(line=row, character=col)
                end = types.Position(line=row, character=col + len("undefined_name"))
                found.append(
                    types.Diagnostic(
                        range=types.Range(start=start, end=end),
                        message='"undefined_name" is not defined',
                        severity=types.DiagnosticSeverity.Error,
                        code="reportUndefinedVariable",
                        source="fake",
                        data={"generated": True},
                    )
                )
                col = line.find("undefined_name", col + 1)
        ls.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(
                uri=document.uri, version=document.version, diagnostics=found
            )
        )


    @server.feature(types.TEXT_DOCUMENT_DID_OPEN)
    def did_open(ls, params):
        publish(ls, params.text_document)


    @server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
    def did_change(ls, params):
        publish(ls, params.text_document)


    @server.feature(types.EXIT)
    def exit(ls, *args):
        config["exited"] = True


    server.start_io()
    if "--linger" in sys.argv and "exited" not in config:
        time.sleep(30)
    """
)


def locate(text, word, nth=0):
    """(0-based line, column) of the nth occurrence of word."""
    index = -1
    for _ in range(nth + 1):
        index = text.index(word, index + 1)
    return text.count("\n", 0, index), index - (text.rfind("\n", 0, index) + 1)


def hover_params(uri, line=0, character=0):
    return lsp.HoverParams(
        text_document=lsp.TextDocumentIdentifier(uri=uri),
        position=lsp.Position(line=line, character=character),
    )


@pytest.fixture
def spawned(monkeypatch):
    """Every process the backend client starts (pygls' start_io calls
    asyncio.create_subprocess_exec)."""
    calls = []
    original = asyncio.create_subprocess_exec

    async def counting(*args, **kwargs):
        calls.append(args)
        return await original(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", counting)
    return calls


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "main.lpy").write_text("(= y 1)\n")
    return root


def view_root(project, tmp_path):
    return pv.view_root_for(str(project), cache_home=str(tmp_path / "cache"))


@pytest.fixture
def fake_script(tmp_path):
    script = tmp_path / "fake_backend.py"
    script.write_text(FAKE_SERVER)
    return str(script)


@pytest.fixture
def fake_argv(fake_script):
    return [sys.executable, fake_script]


@pytest.fixture
def fake_default(monkeypatch, fake_argv):
    """The fake stands for basedpyright, the default backend, so Backends
    resolves to it whether or not basedpyright is installed."""
    monkeypatch.setitem(backend.BACKENDS, "basedpyright", lambda: fake_argv)


def fake_hover(answer):
    """The fake backend's hover as a dict (pid, text, config), or None."""
    return None if answer is None else json.loads(answer.contents)


def run_bounded(coroutine, timeout=30):
    async def bounded():
        async with asyncio.timeout(timeout):
            return await coroutine

    return asyncio.run(bounded())


def gone(pid):
    """Whether process pid has ended (absent, or a zombie nobody reaped yet)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True


async def wait_gone(pid, timeout=5):
    deadline = time.monotonic() + timeout
    while not gone(pid):
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.02)
    return True


def kill_quietly(pid):
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# Setting and off
# ---------------------------------------------------------------------------


def test_backend_setting(monkeypatch):
    # Defect: the default picks basedpyright whether or not it is installed,
    # or an explicit backend that is missing falls back to another one.
    monkeypatch.setitem(backend.BACKENDS, "basedpyright", lambda: ["node", "bp"])
    monkeypatch.setitem(backend.BACKENDS, "pyrefly", lambda: None)
    assert backend.resolve_backend(None) == "basedpyright"
    assert backend.resolve_backend({"backend": None}) == "basedpyright"
    assert backend.resolve_backend({"backend": "pyrefly"}) == "off"
    assert backend.resolve_backend({"backend": "no-such"}) == "off"
    monkeypatch.setitem(backend.BACKENDS, "basedpyright", lambda: None)
    assert backend.resolve_backend(None) == "off"
    assert backend.resolve_backend({}) == "off"


@pytest.mark.parametrize(
    "init_options, installed",
    [({"backend": "off"}, True), ({"backend": "no-such"}, True), (None, False)],
)
def test_off_never_spawns(
    init_options, installed, project, tmp_path, fake_argv, spawned, monkeypatch
):
    # Defect: an off or unknown backend starts the installed default in its
    # place, the default starts on a plain install, or a request on an off
    # backend raises.
    argv = fake_argv if installed else None
    monkeypatch.setitem(backend.BACKENDS, "basedpyright", lambda: argv)
    folder = SimpleNamespace(uri=project.as_uri())
    backends = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "cache"))
    backends.configure(init_options)
    uri = (project / "main.lpy").as_uri()
    answer = run_bounded(
        backends.request(uri, lsp.TEXT_DOCUMENT_HOVER, hover_params(uri), text="x")
    )
    assert answer is None
    assert spawned == []


def test_a_document_belongs_to_its_innermost_folder(tmp_path):
    # Defect: a document is put in a workspace folder that does not hold it
    # (it then has no view, so no backend) or in an outer folder rather
    # than the project nested in it; or a file outside every folder (a
    # dependency opened by go-to-definition) gets a backend.
    outer = tmp_path / "ws"
    project = outer / "project"
    project.mkdir(parents=True)
    other = tmp_path / "another-workspace-folder"
    other.mkdir()
    folders = [SimpleNamespace(uri=p.as_uri()) for p in (outer, project, other)]
    backends = backend.Backends(lambda: folders, cache_home=str(tmp_path / "cache"))
    backends.configure(None)
    outside = tmp_path / "elsewhere.lpy"
    assert backends.client_for(outside.as_uri()) is None
    inside = backends.client_for((project / "main.lpy").as_uri())
    assert inside is not None and inside.project_root == os.path.realpath(project)


@pytest.mark.parametrize(
    "name, written, absent",
    [
        ("pyrefly", "pyrefly.toml", "pyrightconfig.json"),
        ("basedpyright", "pyrightconfig.json", "pyrefly.toml"),
    ],
)
def test_the_view_root_gets_the_chosen_backends_config(
    name, written, absent, project, tmp_path, monkeypatch
):
    # Defect: the view root gets another backend's config file than the
    # one the user chose: pyrefly then migrates a pyrightconfig.json, with
    # neither this LSP's interpreter nor the project's pyrefly settings.
    monkeypatch.setitem(backend.BACKENDS, name, lambda: ["unused"])
    folder = SimpleNamespace(uri=project.as_uri())
    backends = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "cache"))
    backends.configure({"backend": name})
    client = backends.client_for((project / "main.lpy").as_uri())
    client._prepare_view_root()
    root = pathlib.Path(client.view_root.path)
    assert (root / written).is_file()
    assert not (root / absent).exists()


@pytest.mark.parametrize("python_diagnostics", [True, False])
def test_only_an_open_with_diagnostics_starts_a_backend(
    python_diagnostics, project, tmp_path, fake_default, spawned
):
    # Defects: editing, saving or closing a buffer spawns the backend (a
    # process per project on a plain edit); opening one with
    # pythonDiagnostics false spawns it all the same; or opening one with
    # them on never starts it, so no diagnostics come until some request.
    folder = SimpleNamespace(uri=project.as_uri())
    backends = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "cache"))
    backends.configure({"pythonDiagnostics": python_diagnostics})
    uri = (project / "main.lpy").as_uri()
    client = backends.client_for(uri)
    view = pv.compile_view("(= y 2)\n")

    async def scenario():
        try:
            backends.document_changed(uri, "(= y 2)\n", view)
            backends.document_saved(uri)
            backends.document_closed(uri)
            others = asyncio.all_tasks() - {asyncio.current_task()}
            await asyncio.gather(*others)
            assert spawned == []
            backends.document_opened(uri)
            if client._starting is not None:
                assert await client._starting
        finally:
            await backends.shutdown_all()

    run_bounded(scenario())
    assert len(spawned) == (1 if python_diagnostics else 0)
    # Nobody reads its diagnostics: the backend checks nothing.
    client._prepare_view_root()
    config = json.loads(
        (pathlib.Path(client.view_root.path) / "pyrightconfig.json").read_text()
    )
    assert config["typeCheckingMode"] == ("standard" if python_diagnostics else "off")


# ---------------------------------------------------------------------------
# Fake backend: starts, crashes, shutdown, buffer sync, configuration
# ---------------------------------------------------------------------------


def fake_client(project, tmp_path, argv):
    client = backend.BackendClient(view_root(project, tmp_path), argv)
    uri = (project / "main.lpy").as_uri()
    params = hover_params(client.backend_uri(uri))

    async def hover(text="t"):
        return fake_hover(
            await client.request(lsp.TEXT_DOCUMENT_HOVER, params, uri=uri, text=text)
        )

    return client, hover


def test_crashed_backend_restarts_once(project, tmp_path, fake_argv, spawned):
    # Defect: a backend that crashed stays dead for the session, or one that
    # keeps crashing is respawned on every request.
    client, hover = fake_client(project, tmp_path, fake_argv)

    async def crash():
        process = client.process
        os.kill(process.pid, signal.SIGKILL)
        await process.wait()

    async def scenario():
        try:
            first = await hover()
            assert (first["pid"], first["text"]) == (client.process.pid, "t")
            await crash()
            second = await hover()
            assert second is not None and second["pid"] != first["pid"]
            await crash()
            assert await hover() is None
            assert await hover() is None
        finally:
            await client.shutdown()

    run_bounded(scenario())
    assert len(spawned) == 2


def test_cancelled_requests_never_cost_a_start(project, tmp_path, fake_argv, spawned):
    # Defect: cancelling a request (the editor's $/cancelRequest) while the
    # backend starts cancels the start: each cancel uses up a spawn and can
    # leave a process nobody stops, and two of them turn the backend off.
    client, hover = fake_client(project, tmp_path, fake_argv + ["--slow-init", "1"])

    async def cancelled_hover():
        task = asyncio.create_task(hover())
        while not spawned:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def scenario():
        try:
            await cancelled_hover()
            await cancelled_hover()
            assert await hover() is not None
        finally:
            await client.shutdown()

    run_bounded(scenario())
    assert len(spawned) == 1


def test_a_slow_start_is_bounded_and_goes_on(
    project, tmp_path, fake_argv, spawned, monkeypatch
):
    # Defect: the first request waits for the whole start (every shadow
    # compiled, then initialize) however long it takes, or a request that
    # gave up waiting stops the start, so no request ever finds it running.
    monkeypatch.setattr(backend, "REQUEST_TIMEOUT", 0.3)
    client, hover = fake_client(project, tmp_path, fake_argv + ["--slow-init", "1"])

    async def scenario():
        try:
            started = time.monotonic()
            assert await hover() is None
            assert time.monotonic() - started < 0.9
            # A request that answers None at once (a backend gone off)
            # never yields to the loop: sleep, so the timeout can fire.
            while (answer := await hover()) is None:
                await asyncio.sleep(0.01)
            assert answer["pid"] == client.process.pid
        finally:
            await client.shutdown()

    run_bounded(scenario(), timeout=10)
    assert len(spawned) == 1


def test_shutdown_ends_the_backend(project, tmp_path, fake_argv, spawned):
    # Defect: the backend process outlives the LSP's shutdown, or a request
    # after shutdown starts a new one that nobody stops.
    client, hover = fake_client(project, tmp_path, fake_argv)
    pids = []

    async def scenario():
        try:
            pids.append((await hover())["pid"])
            process = client.process
            await client.shutdown()
            assert process.returncode is not None
            assert await hover() is None
        finally:
            for pid in pids:
                kill_quietly(pid)

    run_bounded(scenario())
    assert len(spawned) == 1


def test_a_cancelled_discard_still_cleans_up(project, tmp_path, fake_argv):
    # Defect: shutdown cancels a start while it is discarding a client, and
    # that client's pygls tasks and stderr drain are left running, with
    # nobody to stop them, after shutdown has returned.
    client, hover = fake_client(project, tmp_path, fake_argv)

    async def scenario():
        try:
            assert await hover() is not None
            discard = asyncio.create_task(client._discard())
            await asyncio.sleep(0)
            discard.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await discard
        finally:
            await client.shutdown()
        return asyncio.all_tasks() - {asyncio.current_task()}

    assert run_bounded(scenario()) == set()


def test_backend_is_told_the_interpreter(project, tmp_path, fake_argv):
    # Defect: the backend's workspace/configuration request for "python"
    # does not get python.pythonPath, so it resolves imports with whatever
    # interpreter it finds on PATH, not this project's.
    client, hover = fake_client(project, tmp_path, fake_argv)

    async def scenario():
        try:
            return (await hover())["config"]
        finally:
            await client.shutdown()

    assert run_bounded(scenario()) == [{"pythonPath": client.view_root.python}, {}]


def test_a_push_waits_for_the_requests_in_flight(project, tmp_path, fake_argv):
    # Defect: a change pushed while a request waits for its answer flips the
    # backend's text under it, so it is answered over the full view; or the
    # pushed view is never synced afterwards, so its diagnostics never come.
    client = backend.BackendClient(view_root(project, tmp_path), fake_argv)
    uri = (project / "main.lpy").as_uri()
    client.push_buffer(uri, "(= y 5)", pv.compile_view("(= y 5)"))
    changed = pv.compile_view("(print undefined-name)")

    def hover(delay, text):
        params = hover_params(client.backend_uri(uri), character=delay)
        return client.request(lsp.TEXT_DOCUMENT_HOVER, params, uri=uri, text=text)

    async def scenario():
        try:
            assert await hover(0, None) is not None
            slow = asyncio.create_task(hover(5, "cursor"))
            while client._docs[uri][2] != "cursor":
                await asyncio.sleep(0.01)
            assert client.push_buffer(uri, "(print undefined-name)", changed)
            assert fake_hover(await slow)["text"] == "cursor"
            while client.diagnostics_for(uri, changed.text) is None:
                await asyncio.sleep(0.01)
            [(diagnostic, _)] = client.diagnostics_for(uri, changed.text)
            assert diagnostic.code == "reportUndefinedVariable"
        finally:
            await client.shutdown()

    run_bounded(scenario())


def test_a_timed_out_or_cancelled_request_ends_its_flight(project, tmp_path, fake_argv):
    # Defect: a request that times out or is cancelled stays counted in
    # flight, so the buffer's full view is never synced back and its
    # diagnostics never come again.
    client = backend.BackendClient(view_root(project, tmp_path), fake_argv)
    uri = (project / "main.lpy").as_uri()
    full = pv.compile_view("(= y 5)")
    client.push_buffer(uri, "(= y 5)", full)

    def hover(delay, text, timeout=None):
        params = hover_params(client.backend_uri(uri), character=delay)
        return client.request(
            lsp.TEXT_DOCUMENT_HOVER, params, uri=uri, text=text, timeout=timeout
        )

    async def scenario():
        try:
            assert await hover(0, None) is not None
            assert await hover(5, "cursor", timeout=0.1) is None
            assert uri not in client._in_flight
            assert client._docs[uri][2] == full.text
            cancelled = asyncio.create_task(hover(5, "cursor"))
            while client._docs[uri][2] != "cursor":
                await asyncio.sleep(0.01)
            cancelled.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cancelled
            assert uri not in client._in_flight
            assert client._docs[uri][2] == full.text
        finally:
            await client.shutdown()

    run_bounded(scenario())


def test_overlapping_requests_keep_their_text_until_the_last_answers(
    project, tmp_path, fake_argv
):
    # Defect: the first of two requests over one cursor text (completion and
    # signature help) to answer syncs the buffer's full view back while the
    # other still waits, which is then answered over the wrong text; or the
    # full view is never synced back after both.
    client = backend.BackendClient(view_root(project, tmp_path), fake_argv)
    uri = (project / "main.lpy").as_uri()
    full = pv.compile_view("(= y 5)")
    client.push_buffer(uri, "(= y 5)", full)

    def hover(delay, text):
        params = hover_params(client.backend_uri(uri), character=delay)
        return client.request(lsp.TEXT_DOCUMENT_HOVER, params, uri=uri, text=text)

    async def scenario():
        try:
            assert fake_hover(await hover(0, None))["text"] == full.text
            fast, slow = await asyncio.gather(hover(1, "cursor"), hover(5, "cursor"))
            assert fake_hover(fast)["text"] == "cursor"
            assert fake_hover(slow)["text"] == "cursor"
            assert fake_hover(await hover(0, None))["text"] == full.text
        finally:
            await client.shutdown()

    run_bounded(scenario())


def test_running_backend_follows_the_buffer(project, tmp_path, fake_default):
    # Defect: edits to a buffer never reach a running backend, so it answers
    # for the text of the last request; or a closed buffer stays open in it.
    folder = SimpleNamespace(uri=project.as_uri())
    backends = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "cache"))
    uri = (project / "main.lpy").as_uri()
    params = hover_params(backends.client_for(uri).backend_uri(uri))

    async def backend_text():
        answer = await backends.request(uri, lsp.TEXT_DOCUMENT_HOVER, params)
        return None if answer is None else fake_hover(answer)["text"]

    async def scenario():
        try:
            await backends.request(uri, lsp.TEXT_DOCUMENT_HOVER, params, text="old")
            backends.document_changed(uri, "(= my-y 2)", pv.compile_view("(= my-y 2)"))
            while (text := await backend_text()) == "old":
                await asyncio.sleep(0.01)
            assert text == "my_y = 2"
            backends.document_closed(uri)
            # Closed, the backend reads the document's uri from disk: the
            # shadow of main.lpy as saved, not the buffer.
            assert await backend_text() == "y = 1"
        finally:
            await backends.shutdown_all()

    # Bounds the poll for the pushed text too.
    run_bounded(scenario(), timeout=10)


# ---------------------------------------------------------------------------
# The lpy-lsp server process and its backends
# ---------------------------------------------------------------------------

# Runs lispython's LSP with the fake as basedpyright, the default backend, and
# one extra request, lispy-test/backendHover, that hovers through the
# server's own Backends (no feature routes to a backend yet).
LAUNCHER = textwrap.dedent(
    """\
    import sys

    import lispy  # noqa: F401
    from lsprotocol import types

    from lispy.lsp import backend

    fake = [sys.executable, sys.argv[1], "--linger"]
    backend.BACKENDS["basedpyright"] = lambda: fake

    from lispy.lsp.server import backends, server  # noqa: E402


    @server.feature("lispy-test/backendHover")
    async def backend_hover(ls, params):
        uri = params["uri"] if isinstance(params, dict) else params.uri
        client = backends.client_for(uri)
        if client is None:
            return None
        doc = types.TextDocumentIdentifier(uri=client.backend_uri(uri))
        hover = types.HoverParams(
            text_document=doc, position=types.Position(line=0, character=0)
        )
        answer = await backends.request(uri, types.TEXT_DOCUMENT_HOVER, hover)
        return None if answer is None else answer.contents


    server.start_io()
    """
)

pytestmark_lsp = pytest.mark.skipif(
    not os.path.isdir("/proc/self"), reason="reads process state from /proc"
)


class LspSession:
    """lpy-lsp in a subprocess, initialized over project."""

    def __init__(self, tmp_path, fake_script, project, launcher=LAUNCHER):
        self.launcher = tmp_path / "launch_lsp.py"
        self.launcher.write_text(launcher)
        self.fake_script = fake_script
        self.project = project
        self.uri = (project / "main.lpy").as_uri()
        self.client = LanguageClient("test", "0")
        self.client.feature(lsp.TEXT_DOCUMENT_PUBLISH_DIAGNOSTICS)(lambda p: None)
        self.pids = []

    async def start(self, init_options):
        await self.client.start_io(
            sys.executable, str(self.launcher), self.fake_script, cwd=str(self.project)
        )
        folder = lsp.WorkspaceFolder(uri=self.project.as_uri(), name="p")
        await self.client.initialize_async(
            lsp.InitializeParams(
                capabilities=lsp.ClientCapabilities(),
                root_uri=self.project.as_uri(),
                workspace_folders=[folder],
                initialization_options=init_options,
            )
        )
        self.client.initialized(lsp.InitializedParams())
        self.client.text_document_did_open(
            lsp.DidOpenTextDocumentParams(
                text_document=lsp.TextDocumentItem(
                    uri=self.uri, language_id="lpy", version=1, text="(= y 1)\n"
                )
            )
        )

    async def backend_hover(self):
        """The fake's hover through the server, as a dict, or None."""
        answer = await self.client.protocol.send_request_async(
            "lispy-test/backendHover", {"uri": self.uri}
        )
        if answer is None:
            return None
        answer = json.loads(answer)
        if answer["pid"] not in self.pids:
            self.pids.append(answer["pid"])
        return answer

    def change(self, version, text):
        self.client.text_document_did_change(
            lsp.DidChangeTextDocumentParams(
                text_document=lsp.VersionedTextDocumentIdentifier(
                    uri=self.uri, version=version
                ),
                content_changes=[lsp.TextDocumentContentChangeWholeDocument(text=text)],
            )
        )

    def close(self):
        self.client.text_document_did_close(
            lsp.DidCloseTextDocumentParams(
                text_document=lsp.TextDocumentIdentifier(uri=self.uri)
            )
        )

    async def exit(self):
        """Send exit; the server's exit code."""
        self.client.exit(None)
        return await asyncio.wait_for(self.client._server.wait(), 5)

    async def stop(self):
        server = self.client._server
        if server is not None and server.returncode is None:
            server.kill()
        for pid in self.pids:
            kill_quietly(pid)
        await asyncio.wait_for(self.client.stop(), 5)


@pytestmark_lsp
def test_lsp_server_drives_its_backend(tmp_path, fake_script, project):
    # Defect: the server's did-change, did-close or shutdown hooks are not
    # wired: edits never reach the backend, a closed buffer stays open in it,
    # or the backend outlives the server's shutdown.
    session = LspSession(tmp_path, fake_script, project)

    async def scenario():
        try:
            await session.start(None)
            first = await session.backend_hover()
            assert first["text"] == "y = 1"
            session.change(2, "(= my-y 2)")
            while (answer := await session.backend_hover())["text"] == "y = 1":
                await asyncio.sleep(0.01)
            assert answer["text"] == "my_y = 2"
            session.close()
            assert (await session.backend_hover())["text"] == "y = 1"
            await asyncio.wait_for(session.client.shutdown_async(None), 5)
            assert await wait_gone(first["pid"])
            assert await session.exit() == 0
        finally:
            await session.stop()

    run_bounded(scenario())


@pytestmark_lsp
def test_lsp_exit_without_shutdown_kills_the_backend(tmp_path, fake_script, project):
    # Pins the outcome, not the exit hook: after an exit with no shutdown,
    # no backend outlives the server.  Deleting on-exit alone does not fail
    # it, since asyncio's subprocess transport also kills the child when the
    # server's interpreter exits; on-exit is the explicit guarantee.
    session = LspSession(tmp_path, fake_script, project)

    async def scenario():
        try:
            await session.start(None)
            pid = (await session.backend_hover())["pid"]
            await session.exit()
            assert await wait_gone(pid)
        finally:
            await session.stop()

    run_bounded(scenario())


@pytestmark_lsp
def test_lsp_honours_backend_off(tmp_path, fake_script, project):
    # Defect: the server never reads its initializationOptions, so the
    # user's "backend": "off" is ignored and the default backend starts.
    session = LspSession(tmp_path, fake_script, project)

    async def scenario():
        try:
            await session.start({"backend": "off"})
            assert await session.backend_hover() is None
        finally:
            await session.stop()

    run_bounded(scenario())


# ---------------------------------------------------------------------------
# Real backend: basedpyright
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def loop():
    """One event loop in a background thread for the shared real backend."""
    event_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=event_loop.run_forever, daemon=True)
    thread.start()
    yield event_loop
    event_loop.call_soon_threadsafe(event_loop.stop)
    thread.join(10)
    event_loop.close()


def on_loop(event_loop, coroutine, timeout=30):
    return asyncio.run_coroutine_threadsafe(coroutine, event_loop).result(timeout)


@pytest.fixture(scope="session")
def real(loop, tmp_path_factory):
    argv = backend.backend_argv("basedpyright")
    if argv is None:
        pytest.skip("basedpyright is not installed (the lsp extra)")
    root = tmp_path_factory.mktemp("real-project")
    (root / "main.lpy").write_text("(= y 1)\n")
    (root / "helper.lpy").write_text(
        '(def greet [name]\n  "Say hello to name."\n  (return name))\n'
    )
    cache = tmp_path_factory.mktemp("real-cache")
    client = backend.BackendClient(
        pv.view_root_for(str(root), cache_home=str(cache)), argv
    )
    yield client, root
    process = client.process
    on_loop(loop, client.shutdown())
    assert process is None or process.returncode is not None


def real_hover(loop, real, source, word, nth=0):
    """basedpyright's hover text on word in the view of source, sent as
    main.lpy's text."""
    client, root = real
    uri = (root / "main.lpy").as_uri()
    view = pv.compile_view(source)
    line, character = locate(view.text, word, nth)
    params = hover_params(client.backend_uri(uri), line, character)
    answer = on_loop(
        loop, client.request(lsp.TEXT_DOCUMENT_HOVER, params, uri=uri, text=view.text)
    )
    assert answer is not None
    return answer.contents.value


def test_real_backend_analyses_the_synced_text(loop, real):
    # Defect: the text a request carries never reaches the backend (didOpen
    # or didChange missing), so it answers for the file on disk, (= y 1).
    path_hover = real_hover(loop, real, "(import os)\n(= x os.path)", "path")
    assert "path: Module" in path_hover
    sep_hover = real_hover(loop, real, "(import os)\n(= x os.sep)", "sep")
    assert "(variable) sep: LiteralString" in sep_hover


def test_real_backend_resolves_a_sibling_lpy_module(loop, real):
    # Defect: the backend starts before the shadows are written, so an
    # import of another .lpy module of the project does not resolve.
    text = real_hover(loop, real, "(from helper [greet])\n(greet 1)", "greet", 1)
    assert "Say hello to name." in text


def test_locate_places_each_backend_location(project, tmp_path):
    # Defects: a shadow or temporary file of the cache reaches the editor,
    # a linked real file is named by its link in the cache, an open
    # document maps through its disk shadow, or a file outside the view
    # root is touched.  No backend process: locate reads the filesystem and
    # the synced documents only.
    (project / "other.lpy").write_text("(= z 1)\n")
    (project / "real.py").write_text("x = 1\n")
    client = backend.BackendClient(view_root(project, tmp_path), None)
    root = pathlib.Path(client.view_root.path)
    root.mkdir(parents=True)
    for name in ["main.py", "other.py", "orphan.py", "w.py.tmp"]:
        (root / name).write_text("z = 1\n")
    (root / "real.py").symlink_to(project / "real.py")
    main_uri = (project / "main.lpy").as_uri()
    origin = ("(= y 1)\n", object())
    client._docs[main_uri] = [(root / "main.py").as_uri(), 1, "y = 1\n", origin]
    real_project = project.resolve()

    def place(path):
        return client.locate(path.as_uri())

    assert place(root / "main.py") == ("lpy", main_uri, origin)
    assert place(root / "other.py") == (
        "lpy",
        (real_project / "other.lpy").as_uri(),
        None,
    )
    assert place(root / "real.py") == ("file", (real_project / "real.py").as_uri())
    assert place(root / "orphan.py") is None
    assert place(root / "w.py.tmp") is None
    outside = tmp_path / "elsewhere.py"
    assert client.locate(outside.as_uri()) == ("file", outside.as_uri())
    # The requesting document's origin, given by the caller, wins over a
    # later sync's in _docs.
    asked = ("(= y 1)\n", object())
    assert client.locate((root / "main.py").as_uri(), origins={main_uri: asked}) == (
        "lpy",
        main_uri,
        asked,
    )
    # A synced document whose text has no known origin is dropped, never
    # mapped through its shadow on disk (not what the backend read).
    client._docs[main_uri][3] = None
    assert place(root / "main.py") is None


# ---------------------------------------------------------------------------
# Real backend: pyrefly
# ---------------------------------------------------------------------------

needs_pyrefly = pytest.mark.skipif(
    backend.backend_argv("pyrefly") is None,
    reason="pyrefly is not installed (the lsp extra)",
)

# A valid value for every project rule key the view root's pyrefly.toml takes
# over, so the real pyrefly can say whether each is a key it knows.
PYREFLY_RULES = {
    "preset": "strict",
    "python-version": "3.12",
    "python-platform": "linux",
    "replace-imports-with-any": ["foo.*"],
    "ignore-missing-imports": ["bar.*"],
    "replace-untyped-imports-with-any": ["baz.*"],
    "infer-with-first-use": False,
    "check-unannotated-defs": False,
    "infer-return-types": "checked",
    "strict-callable-subtyping": True,
    "treat-all-caps-as-final": True,
    "untyped-def-behavior": "check-and-infer-return-type",
    "permissive-ignores": True,
    "enabled-ignores": ["pyrefly", "type"],
    "type-ignore-unknown-tag-behavior": "suppress",
    "spec-compliant-overloads": True,
    "legacy-overload-expansion": True,
}


@needs_pyrefly
def test_real_pyrefly_reads_the_written_config(project, tmp_path):
    # Defects: the view root's pyrefly.toml holds a key pyrefly does not
    # know or a value it cannot read (it warns and drops the key, or falls
    # back to its own auto configuration), its interpreter is not this
    # LSP's, or its search path is mangled on the way to TOML.
    assert set(PYREFLY_RULES) == pv._PYREFLY_RULE_KEYS
    (project / "pyrefly.toml").write_text(
        # JSON spells these values (str, bool, list of str) as TOML does.
        "".join(
            f"{key} = {json.dumps(value)}\n" for key, value in PYREFLY_RULES.items()
        )
        + '[errors]\nbad-assignment = "warn"\n'
    )
    root = view_root(project, tmp_path)
    pv.write_shadows(str(project), os.fspath(root))
    path = pv.write_pyrefly_config(str(project), root)
    with open(path, "rb") as f:
        written = tomllib.load(f)
    assert {key: written[key] for key in PYREFLY_RULES} == PYREFLY_RULES
    # As an editor starts the LSP, unactivated: pyrefly would otherwise find
    # this venv through VIRTUAL_ENV or the python3 on PATH without the
    # config's interpreter.
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("VIRTUAL_ENV", "CONDA_PREFIX")
    }
    env["PATH"] = os.pathsep.join(
        entry
        for entry in env.get("PATH", "").split(os.pathsep)
        if not entry.startswith(sys.prefix)
    )
    found = subprocess.run(
        [
            backend.backend_argv("pyrefly")[0],
            "dump-config",
            "--color",
            "never",
            "-c",
            path,
        ],
        cwd=os.fspath(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = found.stdout + found.stderr
    assert found.returncode == 0, output
    assert "Extra keys" not in output
    assert "failed to parse" not in output
    assert f"Using interpreter: {sys.executable}\n" in output
    [search] = [
        line.split(": ", 1)[1]
        for line in output.splitlines()
        if "Search path (from config file)" in line
    ]
    assert json.loads(search) == written["search-path"]
    assert written["search-path"][1:] == [str(project.resolve())]


@needs_pyrefly
def test_real_pyrefly_starts_and_follows_the_projects_config(project, tmp_path):
    # Defects: _pyrefly-argv does not start an LSP over stdio, or pyrefly's
    # initialize result fails to read (lsprotocol's notebook filter), so
    # every request answers None; or the view root gets no pyrefly.toml
    # (pyrefly then migrates the pyrightconfig.json beside it), so the
    # project's own pyrefly severities never reach the buffer; or a project
    # .py resolves through search-path to the real file rather than its
    # link in the view, and is analysed under the project's own config
    # (here an interpreter that does not exist, so pygls is unknown in it).
    (project / "pyrefly.toml").write_text(
        'python-interpreter-path = "/nonexistent/python"\n'
        '[errors]\nunknown-name = "warn"\n'
    )
    (project / "helper.py").write_text("import pygls.uris as u\nX = u.from_fs_path\n")
    source = (
        "(import pygls)\n(= y pygls.uris)\n(print undefined-name)\n"
        "(from helper [X])\n(= z X)\n"
    )
    client = backend.BackendClient(
        view_root(project, tmp_path), backend.backend_argv("pyrefly"), name="pyrefly"
    )
    uri = (project / "main.lpy").as_uri()
    view = pv.compile_view(source)

    async def hover(word):
        line, character = locate(view.text, word)
        answer = await client.request(
            lsp.TEXT_DOCUMENT_HOVER,
            hover_params(client.backend_uri(uri), line, character),
            uri=uri,
            text=view.text,
        )
        return None if answer is None else answer.contents.value

    async def scenario():
        heard = asyncio.Event()
        client.on_diagnostics = lambda lpy_uri: heard.set()
        try:
            uris = await hover("uris")
            if uris is None:
                return None, None, None
            helper = await hover("z =")
            client.push_buffer(uri, source, view)
            await asyncio.wait_for(heard.wait(), 20)
            return uris, helper, client.diagnostics_for(uri, view.text)
        finally:
            await client.shutdown()

    uris, helper, diagnostics = run_bounded(scenario(), timeout=60)
    assert uris is not None and "Module[pygls.uris]" in uris
    assert "def from_fs_path(path: str)" in helper
    [found] = [d for d, _ in diagnostics if d.code == "unknown-name"]
    assert found.severity == lsp.DiagnosticSeverity.Warning


# Runs lispython's LSP with its real backends, and one extra request,
# lispy-test/backendPid, the pid of a document's running backend, or None.
REAL_LAUNCHER = textwrap.dedent(
    """\
    import lispy  # noqa: F401

    from lispy.lsp.server import backends, server


    @server.feature("lispy-test/backendPid")
    def backend_pid(ls, params):
        uri = params["uri"] if isinstance(params, dict) else params.uri
        client = backends.client_for(uri)
        if client is None or not client.running:
            return None
        return client.process.pid


    server.start_io()
    """
)


def descendants(pid):
    """Every live descendant pid of process pid, from /proc."""
    children = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(entry))
    found, stack = [], [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            found.append(child)
            stack.append(child)
    return found


def running_binary(pid):
    try:
        return os.path.realpath(os.readlink(f"/proc/{pid}/exe"))
    except OSError:
        return None


@needs_pyrefly
@pytestmark_lsp
def test_real_pyrefly_exits_when_the_lsp_is_killed(tmp_path, project):
    # Defect: pyrefly outlives a killed LSP.  pyrefly ignores the
    # initialize processId and exits only on its stdin's EOF, so a process
    # that keeps the write end of its stdin (a wrapper between the LSP and
    # pyrefly, a leaked descriptor) leaves it running for good.
    # Every pyrefly below the LSP, however it was started, not just the
    # LSP's own child.
    session = LspSession(tmp_path, "unused", project, launcher=REAL_LAUNCHER)
    # The pyrefly executable itself, whatever argv starts it with.
    binary = os.path.realpath(backend._scripts_binary("pyrefly"))

    async def scenario():
        pyreflies = []
        try:
            await session.start({"backend": "pyrefly"})
            while not pyreflies:
                running = await session.client.protocol.send_request_async(
                    "lispy-test/backendPid", {"uri": session.uri}
                )
                if running is not None:
                    pyreflies = [
                        pid
                        for pid in descendants(session.client._server.pid)
                        if running_binary(pid) == binary
                    ]
                await asyncio.sleep(0.05)
            session.client._server.kill()
            for pid in pyreflies:
                assert await wait_gone(pid, timeout=10)
        finally:
            for pid in pyreflies:
                kill_quietly(pid)
            await session.stop()

    run_bounded(scenario(), timeout=60)


@needs_pyrefly
def test_real_pyrefly_refusal_keeps_the_views_settings(project, tmp_path, caplog):
    # Defects: a project value the installed pyrefly cannot read (an error
    # code of another pyrefly release, an unknown variant) is written, so
    # pyrefly drops the whole view config for auto configuration and the
    # search paths, interpreter and preset off are lost; or the check reads
    # pyrefly's output wrong (it exits 1 whether or not it read the file),
    # so a refusal goes unnoticed or every file is refused.
    binary = backend.backend_argv("pyrefly")[0]
    assert (
        backend.pyrefly_config_problem(binary, 'preset = "basic"\nbogus = 1\n') is None
    )
    assert "nope" in backend.pyrefly_config_problem(
        binary, 'untyped-def-behavior = "nope"\n'
    )
    (project / "pyrefly.toml").write_text(
        'preset = "strict"\nuntyped-def-behavior = "nope"\n'
    )
    root = view_root(project, tmp_path)
    write = backend.config_writer("pyrefly")
    with caplog.at_level("WARNING", logger="lispy.lsp.python_view"):
        path = write(str(project), root, check_types=False)
    assert str(project / "pyrefly.toml") in caplog.text
    assert "`nope`" in caplog.text
    # The position pyrefly gives is in the generated text, never the
    # project's file the warning names: left out.
    assert "line " not in caplog.text
    problem = backend.pyrefly_config_problem(
        binary, '[errors]\nno-such-code = "warn"\n'
    )
    assert "unknown variant `no-such-code`" in problem
    assert "line 1, column" not in problem
    text = open(path).read()
    assert backend.pyrefly_config_problem(binary, text) is None
    written = tomllib.loads(text)
    assert written["preset"] == "off"
    assert "errors" not in written
    assert written["search-path"][1:] == [str(project)]
    assert written["python-interpreter-path"] == sys.executable


# pyrefly 1.4.1's initialize result, captured from `pyrefly lsp` answering an
# initialize with empty client capabilities.
PYREFLY_INITIALIZE_RESULT = json.loads(
    """{"capabilities": {"callHierarchyProvider": true, "codeActionProvider":
    {"codeActionKinds": ["quickfix", "refactor.extract", "refactor.rewrite",
    "refactor.delete", "refactor.move", "refactor.inline", "source.fixAll",
    "source.fixAll.pyrefly"]}, "codeLensProvider": {"resolveProvider": false},
    "completionProvider": {"resolveProvider": true, "triggerCharacters": [".",
    "'", "\\""]}, "declarationProvider": true, "definitionProvider": true,
    "documentHighlightProvider": true, "documentSymbolProvider": true,
    "foldingRangeProvider": true, "hoverProvider": true, "implementationProvider":
    true, "inlayHintProvider": true, "notebookDocumentSync": {"notebookSelector":
    [{"cells": [{"language": "python"}]}]}, "positionEncoding": "utf-16",
    "referencesProvider": true, "renameProvider": {"prepareProvider": true},
    "selectionRangeProvider": true, "signatureHelpProvider": {"triggerCharacters":
    ["(", ","]}, "textDocumentSync": {"change": 2, "openClose": true, "save":
    {"includeText": false}}, "typeDefinitionProvider": true,
    "typeHierarchyProvider": true, "workspace": {"fileOperations": {"willRename":
    {"filters": [{"pattern": {"glob": "**/*.{py,pyi}", "matches": "file"},
    "scheme": "file"}]}}, "workspaceFolders": {"changeNotifications": true,
    "supported": true}}, "workspaceSymbolProvider": true}, "serverInfo": {"name":
    "pyrefly-lsp", "version": "1.4.1"}}"""
)


def test_backend_converter_reads_pyrefly_initialize_result():
    # Defect: the backend client's converter cannot structure pyrefly's
    # initialize result (a notebook filter with cells and no notebook), so
    # the pyrefly backend never starts.
    result = backend._converter().structure(
        PYREFLY_INITIALIZE_RESULT, lsp.InitializeResult
    )
    [selector] = result.capabilities.notebook_document_sync.notebook_selector
    assert selector.notebook is None
    assert [cell.language for cell in selector.cells] == ["python"]
