"""The LSP's Python backend client (lispy.lsp.backend).

The real-backend tests share one process per installed backend on one event
loop in a background thread (a basedpyright start costs about a second and
400 MB), one backend at a time. The others run a fake backend, a tiny pygls
server whose hover answers with its pid, its copy of the document and the
configuration the client gave it, and which publishes an Error on each
undefined_name of a synced text, each on its own loop; none needs a real
backend installed.
"""

import asyncio
import contextlib
import importlib
import json
import logging
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
from tests.lsp_backends import REAL_BACKENDS

backend = importlib.import_module("lispy.lsp.backend")
pv = importlib.import_module("lispy.lsp.python_view")
server = importlib.import_module("lispy.lsp.server")

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
        answer = {
            "pid": os.getpid(),
            "text": text,
            "config": config.get("answer"),
            "watched": config.get("watched", []),
        }
        return types.Hover(contents=json.dumps(answer))


    @server.feature(types.WORKSPACE_DID_CHANGE_WATCHED_FILES)
    def did_change_watched_files(ls, params):
        config.setdefault("watched", []).extend(c.uri for c in params.changes)


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
    """The fake stands for basedpyright, the default backend when pyrefly is
    not installed (as here), so Backends resolves to it whatever is
    installed."""
    monkeypatch.setitem(backend.BACKENDS, "pyrefly", lambda: None)
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
    except (FileNotFoundError, ProcessLookupError):
        # A process that exits between open and read makes read raise ESRCH.
        return True


def test_gone_treats_an_exit_between_open_and_read_as_gone(monkeypatch):
    class Exiting:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr("builtins.open", lambda *a, **k: Exiting())
    assert gone(2**22 + 1)


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
    # Defects: with no setting, basedpyright runs where pyrefly is installed
    # too, or the default picks a backend that is not installed; or an
    # explicit backend that is missing falls back to another one.
    monkeypatch.setitem(backend.BACKENDS, "basedpyright", lambda: ["node", "bp"])
    monkeypatch.setitem(backend.BACKENDS, "pyrefly", lambda: ["pyrefly", "lsp"])
    assert backend.resolve_backend(None) == "pyrefly"
    assert backend.resolve_backend({"backend": None}) == "pyrefly"
    assert backend.resolve_backend({"backend": "basedpyright"}) == "basedpyright"
    monkeypatch.setitem(backend.BACKENDS, "pyrefly", lambda: None)
    assert backend.resolve_backend(None) == "basedpyright"
    assert backend.resolve_backend({"backend": "pyrefly"}) == "off"
    assert backend.resolve_backend({"backend": "no-such"}) == "off"
    monkeypatch.setitem(backend.BACKENDS, "basedpyright", lambda: None)
    assert backend.resolve_backend(None) == "off"
    assert backend.resolve_backend({}) == "off"


def test_the_real_backend_tests_cover_the_default_backends():
    # Defect: a backend added to (or reordered in) DEFAULT_BACKENDS is never
    # run by the real-backend tests, which list them without importing the
    # LSP (lsp_backends.REAL_BACKENDS).
    assert REAL_BACKENDS == backend.DEFAULT_BACKENDS


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
    monkeypatch.setitem(backend.BACKENDS, "pyrefly", lambda: None)
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
    client._prepare_view_root([])
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
    client._prepare_view_root([])
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


def test_a_save_tells_the_backend_of_every_shadow_it_rewrote(
    project, tmp_path, monkeypatch
):
    # Defect: a save notifies only the saved file's shadow, so after a macro
    # module's edit the backend keeps analysing its dependents' old shadows,
    # which the sweep rewrote, and never hears of one the sweep removed.
    name = "bkmmsave"
    pkg = project / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mac.lpy").write_text("(defmacro twice [x] (return `(+ ~x ~x)))")
    (pkg / "mod.lpy").write_text("(require .mac *)\n(= yy (twice 2))")
    (pkg / "gone.lpy").write_text("(= g 1)")
    for key in [name, name + ".mac"]:
        monkeypatch.delitem(sys.modules, key, raising=False)
    client = backend.BackendClient(view_root(project, tmp_path), ["unused"])
    sent = []
    client.state = "running"
    client.process = SimpleNamespace(returncode=None)
    client._client = SimpleNamespace(workspace_did_change_watched_files=sent.append)
    try:
        client._prepare_view_root([])
        shadow = pathlib.Path(client.view_root.path) / name / "mod.py"
        assert "yy = 2 + 2" in shadow.read_text()
        (pkg / "mac.lpy").write_text("(defmacro twice [x] (return `(* ~x ~x ~x)))")
        (pkg / "gone.lpy").unlink()
        run_bounded(client.file_saved((pkg / "mac.lpy").as_uri()))
        assert "yy = 2 * 2 * 2" in shadow.read_text()
        [params] = sent
        changes = {(change.uri, change.type) for change in params.changes}
        uri = client.backend_uri
        assert changes == {
            (uri((pkg / "mod.lpy").as_uri()), lsp.FileChangeType.Changed),
            (uri((pkg / "mac.lpy").as_uri()), lsp.FileChangeType.Changed),
            (uri((pkg / "gone.lpy").as_uri()), lsp.FileChangeType.Deleted),
        }
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]


def test_a_save_while_the_backend_starts_reaches_it(
    project, tmp_path, fake_default, monkeypatch
):
    # Defect: a save landing while the backend starts, after the start's
    # sweep read the disk, is dropped: a macro module's dependent shadow
    # stays stale and the backend never hears of it until the next save.
    name = "bkmmstart"
    pkg = project / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mac.lpy").write_text("(defmacro twice [x] (return `(+ ~x ~x)))")
    (pkg / "mod.lpy").write_text("(require .mac *)\n(= yy (twice 2))")
    for key in [name, name + ".mac"]:
        monkeypatch.delitem(sys.modules, key, raising=False)
    folder = SimpleNamespace(uri=project.as_uri())
    backends = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "cache"))
    main = (project / "main.lpy").as_uri()
    mac = (pkg / "mac.lpy").as_uri()
    client = backends.client_for(main)
    params = hover_params(client.backend_uri(main))
    shadow = pathlib.Path(client.view_root.path) / name / "mod.py"
    mod_uri = client.backend_uri((pkg / "mod.lpy").as_uri())
    sweep = client._prepare_view_root

    async def scenario():
        loop = asyncio.get_running_loop()
        swept = asyncio.Event()

        def signalling_sweep(open_paths):
            result = sweep(open_paths)
            loop.call_soon_threadsafe(swept.set)
            return result

        client._prepare_view_root = signalling_sweep
        try:
            backends.document_opened(main)
            await swept.wait()
            assert "yy = 2 + 2" in shadow.read_text()
            # The window: the start's sweep is done, the process not running.
            assert client._starting is not None and not client._starting.done()
            assert not client.running
            (pkg / "mac.lpy").write_text("(defmacro twice [x] (return `(* ~x ~x ~x)))")
            backends.document_saved(mac)
            assert await client._starting
            await asyncio.gather(*backends._tasks)
            # The replay runs after the start, as a task of its own.
            deadline = loop.time() + 10
            while True:
                answer = await backends.request(
                    main, lsp.TEXT_DOCUMENT_HOVER, params, text="t"
                )
                watched = fake_hover(answer)["watched"]
                if mod_uri in watched or loop.time() > deadline:
                    return watched
                await asyncio.sleep(0.05)
        finally:
            await backends.shutdown_all()

    try:
        watched = run_bounded(scenario())
        assert "yy = 2 * 2 * 2" in shadow.read_text()
        assert mod_uri in watched
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]


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


def test_a_start_tells_that_the_backends_input_changed(project, tmp_path, fake_argv):
    # Defect: a backend process that starts (first or after a crash) reads
    # every file anew, saves made while none ran included, yet nobody hears
    # of it, so answers kept from before it serve the rest of a word.
    client, hover = fake_client(project, tmp_path, fake_argv)
    told = []
    client.on_input_changed = told.append

    async def scenario():
        try:
            await hover()
        finally:
            await client.shutdown()

    run_bounded(scenario())
    assert None in told


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

    python, other = run_bounded(scenario())
    assert python["pythonPath"] == client.view_root.python
    assert other == {}


def test_a_save_that_changes_the_config_tells_the_backend(
    project, tmp_path, fake_default
):
    # Defects: a save that rewrote the view root's config file neither tells
    # the running backend the file changed (it keeps the config it read at
    # its start) nor on-view-prepared (a preset fallback that began with
    # the save is never told); or every save does, config unchanged.
    folder = SimpleNamespace(uri=project.as_uri())
    backends = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "cache"))
    uri = (project / "main.lpy").as_uri()
    client = backends.client_for(uri)
    params = hover_params(client.backend_uri(uri))
    config_uri = (pathlib.Path(client.view_root) / pv.PYRIGHT_CONFIG).as_uri()
    told = []
    backends.on_view_prepared = told.append

    async def save():
        # As the server's didSave: the hook schedules the work.
        backends.document_saved(uri)
        await asyncio.gather(*backends._tasks)

    async def scenario():
        try:
            await backends.request(uri, lsp.TEXT_DOCUMENT_HOVER, params, text="t")
            started = len(told)
            await save()
            unchanged = len(told)
            (project / "pyrightconfig.json").write_text(
                '{"typeCheckingMode": "strict"}'
            )
            await save()
            changed = len(told)
            answer = await backends.request(uri, lsp.TEXT_DOCUMENT_HOVER, params)
            return started, unchanged, changed, fake_hover(answer)["watched"]
        finally:
            await backends.shutdown_all()

    started, unchanged, changed, watched = run_bounded(scenario())
    assert (started, unchanged, changed) == (1, 1, 2)
    assert watched.count(config_uri) == 1


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


def test_a_timed_out_request_logs_no_error(project, tmp_path, fake_argv, caplog):
    # Defect: shield + wait_for on the request's future makes Python 3.14+
    # log the backend's RequestCancelled reply to a timed-out request at
    # ERROR (a traceback in the editor's LSP log).
    client = backend.BackendClient(view_root(project, tmp_path), fake_argv)
    uri = (project / "main.lpy").as_uri()
    client.push_buffer(uri, "(= y 5)", pv.compile_view("(= y 5)"))

    def hover(delay, timeout=None):
        params = hover_params(client.backend_uri(uri), character=delay)
        return client.request(lsp.TEXT_DOCUMENT_HOVER, params, uri=uri, timeout=timeout)

    async def scenario():
        try:
            assert await hover(0) is not None
            assert await hover(5, timeout=0.1) is None
            futures = client._client.protocol._request_futures
            for _ in range(200):
                if not futures:
                    break
                await asyncio.sleep(0.01)
            assert not futures
            for _ in range(5):
                await asyncio.sleep(0)
        finally:
            await client.shutdown()

    caplog.set_level(logging.ERROR, logger="asyncio")
    run_bounded(scenario())
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []


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


def test_a_request_never_reopens_a_closed_buffer_for_good(
    project, tmp_path, fake_default
):
    # Defect: a request waiting for the start when its buffer closes (or
    # moves to another project's backend) syncs its text afterwards,
    # reopening the document, and nothing closes it again: the backend
    # answers for that cursor view forever instead of the file on disk;
    # once closed there, a completion kept over that text must not serve on
    # (on-input-changed hears it).
    folder = SimpleNamespace(uri=project.as_uri())
    backends = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "cache"))
    heard = []
    backends.on_input_changed = heard.append
    uri = (project / "main.lpy").as_uri()
    params = hover_params(backends.client_for(uri).backend_uri(uri))

    async def backend_text(text=None):
        answer = await backends.request(uri, lsp.TEXT_DOCUMENT_HOVER, params, text=text)
        return fake_hover(answer)["text"]

    async def scenario():
        try:
            pending = asyncio.ensure_future(backend_text("cursor"))
            await asyncio.sleep(0)  # the request awaits the start
            backends.document_closed(uri)
            assert await pending == "cursor"
            assert heard[-2:] == [uri, uri]  # opened, then closed
            assert await backend_text() == "y = 1"
        finally:
            await backends.shutdown_all()

    run_bounded(scenario(), timeout=10)


def test_exit_kills_a_backend_whose_folder_was_removed(project, tmp_path, fake_default):
    # Defect: a removed folder's backend leaves clients while its shutdown
    # still runs, so an LSP exit then (which ends the loop and that
    # shutdown with it) never kills its process, which outlives the server.
    folders = [SimpleNamespace(uri=project.as_uri())]
    backends = backend.Backends(lambda: list(folders), cache_home=str(tmp_path / "c"))
    uri = (project / "main.lpy").as_uri()
    client = backends.client_for(uri)
    params = hover_params(client.backend_uri(uri))

    async def scenario():
        assert await backends.request(uri, lsp.TEXT_DOCUMENT_HOVER, params)
        pid = client.process.pid
        try:
            folders.clear()
            backends.folders_changed([uri])
            backends.kill_all()
            tasks = list(backends._tasks)
            for task in tasks:  # the exit stops the loop
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            assert await wait_gone(pid)
        finally:
            kill_quietly(pid)

    run_bounded(scenario())


# ---------------------------------------------------------------------------
# The lpy-lsp server process and its backends
# ---------------------------------------------------------------------------

# Runs lispython's LSP with the fake as basedpyright, the default backend
# with pyrefly not installed, and
# one extra request, lispy-test/backendHover, that hovers through the
# server's own Backends (no feature routes to a backend yet).
LAUNCHER = textwrap.dedent(
    """\
    import sys

    import lispy  # noqa: F401
    from lsprotocol import types

    from lispy.lsp import backend

    fake = [sys.executable, sys.argv[1], "--linger"]
    backend.BACKENDS["pyrefly"] = lambda: None
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
# Real backends
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


# Module-scoped and parametrized: pytest stops one backend's process before
# it starts the next, and both before the session's real_backends (conftest).
@pytest.fixture(scope="module", params=REAL_BACKENDS)
def real(request, loop, tmp_path_factory):
    name = request.param
    argv = backend.backend_argv(name)
    if argv is None:
        pytest.skip(f"{name} is not installed (its lsp extra)")
    root = tmp_path_factory.mktemp("real-project")
    (root / "main.lpy").write_text("(= y 1)\n")
    (root / "helper.lpy").write_text(
        '(def greet [name]\n  "Say hello to name."\n  (return name))\n'
    )
    cache = tmp_path_factory.mktemp("real-cache")
    client = backend.BackendClient(
        pv.view_root_for(str(root), cache_home=str(cache)), argv, name=name
    )
    yield client, root
    process = client.process
    on_loop(loop, client.shutdown())
    assert process is None or process.returncode is not None


def real_hover(loop, real, source, word, nth=0):
    """The real backend's hover text on word in the view of source, sent as
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
    assert "sep: LiteralString" in sep_hover


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


# Per real backend: the project's config file and its text setting the
# severity of the undefined-name rule, and the code of that rule.
SEVERITY_CONFIGS = {
    "pyrefly": (
        "pyrefly.toml",
        lambda severity: f'[errors]\nunknown-name = "{severity}"\n',
        "unknown-name",
    ),
    "basedpyright": (
        "pyrightconfig.json",
        lambda severity: json.dumps(
            {"reportUndefinedVariable": "warning" if severity == "warn" else severity}
        ),
        "reportUndefinedVariable",
    ),
}


@pytest.mark.parametrize("name", list(SEVERITY_CONFIGS))
def test_real_backend_follows_a_config_a_save_changed(project, tmp_path, name):
    # Defect: a save rewrites the view root's config file, but the running
    # backend read it at its start and is never told it changed, so the
    # buffer keeps the old severities until the backend restarts.
    argv = backend.backend_argv(name)
    if argv is None:
        pytest.skip(f"{name} is not installed (its lsp extra)")
    config_name, config_text, code = SEVERITY_CONFIGS[name]
    (project / config_name).write_text(config_text("warn"))
    source = "(print undefined-name)\n"
    client = backend.BackendClient(view_root(project, tmp_path), argv, name=name)
    uri = (project / "main.lpy").as_uri()
    view = pv.compile_view(source)

    def severity():
        found = client.diagnostics_for(uri, view.text) or []
        return [d.severity for d, _ in found if d.code == code]

    async def published(wanted, timeout):
        # Each publish of the buffer's diagnostics, until one has wanted.
        deadline = asyncio.get_running_loop().time() + timeout
        while severity() != [wanted]:
            heard.clear()
            left = deadline - asyncio.get_running_loop().time()
            if left <= 0:
                return severity()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(heard.wait(), left)
        return severity()

    async def scenario():
        try:
            await client.request(
                lsp.TEXT_DOCUMENT_HOVER, hover_params(client.backend_uri(uri))
            )
            client.push_buffer(uri, source, view)
            before = await published(lsp.DiagnosticSeverity.Warning, 20)
            (project / config_name).write_text(config_text("error"))
            # Saved, never pushed again: the backend republishes by itself.
            await client.saved(uri)
            return before, await published(lsp.DiagnosticSeverity.Error, 15)
        finally:
            await client.shutdown()

    heard = asyncio.Event()
    client.on_diagnostics = lambda lpy_uri: heard.set()
    before, after = run_bounded(scenario(), timeout=90)
    assert before == [lsp.DiagnosticSeverity.Warning]
    assert after == [lsp.DiagnosticSeverity.Error]


@pytest.mark.parametrize("name", list(REAL_BACKENDS))
def test_real_backend_searches_a_script_directory_only_for_its_files(
    project, tmp_path, name
):
    # Defects: the backend's search paths are one list for the whole
    # project, so scripts/ (searched only by `lpy scripts/x.lpy`) answers
    # a root module's (import helper), which `lpy main.lpy` fails; and the
    # view root, searched first, answers scripts/x.lpy's (import utils)
    # with the root utils.py, where `lpy scripts/x.lpy` imports
    # scripts/utils.py.
    argv = backend.backend_argv(name)
    if argv is None:
        pytest.skip(f"{name} is not installed (its lsp extra)")
    scripts = project / "scripts"
    scripts.mkdir()
    (project / "utils.py").write_text("only_in_root = 1\n")
    (scripts / "utils.py").write_text("only_in_scripts = 1\n")
    (scripts / "helper.py").write_text("h = 1\n")
    sources = {
        project / "main.lpy": "(import helper)\n(print helper.h)\n",
        scripts / "x.lpy": (
            "(import utils)\n(print utils.only-in-scripts utils.only-in-root)\n"
        ),
    }
    for path, source in sources.items():
        path.write_text(source)
    client = backend.BackendClient(view_root(project, tmp_path), argv, name=name)
    views = {path.as_uri(): pv.compile_view(text) for path, text in sources.items()}
    main, script = (path.as_uri() for path in sources)

    def messages(uri):
        found = client.diagnostics_for(uri, views[uri].text)
        return None if found is None else [d.message for d, _ in found]

    def settled():
        # Each buffer's positive control: the error the right resolution
        # gives (scripts/utils has no only_in_root; no helper beside main).
        main_found, script_found = messages(main), messages(script)
        return (
            main_found is not None
            and any("helper" in m for m in main_found)
            and script_found is not None
            and any("only_in_root" in m for m in script_found)
        )

    async def scenario():
        try:
            await client.request(
                lsp.TEXT_DOCUMENT_HOVER, hover_params(client.backend_uri(main))
            )
            for path, text in sources.items():
                client.push_buffer(path.as_uri(), text, views[path.as_uri()])
            deadline = asyncio.get_running_loop().time() + 30
            while not settled():
                heard.clear()
                left = deadline - asyncio.get_running_loop().time()
                if left <= 0:
                    break
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(heard.wait(), left)
            return messages(main), messages(script)
        finally:
            await client.shutdown()

    heard = asyncio.Event()
    client.on_diagnostics = lambda lpy_uri: heard.set()
    main_found, script_found = run_bounded(scenario(), timeout=90)
    assert main_found is not None and any("helper" in m for m in main_found)
    assert script_found is not None
    assert any("only_in_root" in m for m in script_found)
    assert not any("only_in_scripts" in m for m in script_found)


def test_a_save_that_adds_or_drops_a_search_environment_tells_the_backend(
    project, tmp_path
):
    # Defect: a save that writes a new nested pyrefly.toml (a script
    # directory got its first .lpy) or removes one tells the running backend
    # nothing, since only the root config file's text is compared, so it
    # keeps the search paths it read at its start for that directory.
    client = backend.BackendClient(
        view_root(project, tmp_path), ["unused"], name="pyrefly"
    )
    sent = []
    client.state = "running"
    client.process = SimpleNamespace(returncode=None)
    client._client = SimpleNamespace(workspace_did_change_watched_files=sent.append)
    nested = pathlib.Path(client.view_root.path) / "scripts" / pv.PYREFLY_CONFIG
    script = project / "scripts" / "x.lpy"

    def config_events():
        return {
            (change.uri, change.type)
            for params in sent
            for change in params.changes
            if change.uri.endswith(pv.PYREFLY_CONFIG)
        }

    # As a start: the view written and its config files read.
    run_bounded(client.saved((project / "main.lpy").as_uri()))
    sent.clear()
    script.parent.mkdir()
    script.write_text("(= x 1)\n")
    run_bounded(client.saved(script.as_uri()))
    assert nested.is_file()
    assert config_events() == {(nested.as_uri(), lsp.FileChangeType.Changed)}
    sent.clear()
    script.unlink()
    run_bounded(client.saved((project / "main.lpy").as_uri()))
    assert not nested.exists()
    assert config_events() == {(nested.as_uri(), lsp.FileChangeType.Deleted)}


def unsaved_script_backends(project, tmp_path):
    """Backends over project (the fake backend), with scripts/sibling.py
    and the uris of scripts/y.lpy and scripts/z.lpy, neither on disk."""
    scripts = project / "scripts"
    scripts.mkdir()
    (scripts / "sibling.py").write_text("X = 1\n")
    folder = SimpleNamespace(uri=project.as_uri())
    backends = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "cache"))
    return backends, (scripts / "y.lpy").as_uri(), (scripts / "z.lpy").as_uri()


def script_environments(client):
    config = json.loads(
        (pathlib.Path(client.view_root) / pv.PYRIGHT_CONFIG).read_text()
    )
    return [env["root"] for env in config.get("executionEnvironments", [])]


def test_opening_an_unsaved_script_gives_the_running_backend_its_environment(
    project, tmp_path, fake_default
):
    # Defects: a buffer opened before its file exists (scripts/y.lpy, no
    # other .lpy in scripts/) gets no search environment, since only
    # shadowed sources make one, so the backend analyses it under the
    # default list and its (import sibling) stays unresolved until the
    # first save; or the running backend is never told the config changed;
    # or every open sweeps the project again, its environment known.
    backends, script, other = unsaved_script_backends(project, tmp_path)
    main = (project / "main.lpy").as_uri()
    client = backends.client_for(main)
    params = hover_params(client.backend_uri(main))
    config_uri = (pathlib.Path(client.view_root) / pv.PYRIGHT_CONFIG).as_uri()
    view_scripts = os.path.join(client.view_root.path, "scripts")

    async def watched():
        answer = await backends.request(main, lsp.TEXT_DOCUMENT_HOVER, params)
        return fake_hover(answer)["watched"]

    async def scenario():
        try:
            await backends.request(main, lsp.TEXT_DOCUMENT_HOVER, params, text="t")
            assert client.running
            assert script_environments(client) == []
            backends.document_opened(script)
            await asyncio.gather(*backends._tasks)
            first = await watched()
            backends.document_opened(other)
            scheduled = len(backends._tasks)
            await asyncio.gather(*backends._tasks)
            return first, scheduled, await watched()
        finally:
            await backends.shutdown_all()

    first, scheduled, second = run_bounded(scenario())
    assert script_environments(client) == [view_scripts]
    assert first.count(config_uri) == 1
    assert scheduled == 0
    assert second == first


def test_an_unsaved_script_opened_while_the_backend_starts_gets_its_environment(
    project, tmp_path, fake_default
):
    # Defect: an open landing while the backend starts, after the start's
    # sweep read the open documents, is never swept: the backend runs on a
    # config with no environment for the script until the next save.
    backends, script, _ = unsaved_script_backends(project, tmp_path)
    main = (project / "main.lpy").as_uri()
    client = backends.client_for(main)
    params = hover_params(client.backend_uri(main))
    config_uri = (pathlib.Path(client.view_root) / pv.PYRIGHT_CONFIG).as_uri()
    sweep = client._prepare_view_root

    async def scenario():
        loop = asyncio.get_running_loop()
        swept = asyncio.Event()

        def signalling_sweep(open_paths):
            result = sweep(open_paths)
            loop.call_soon_threadsafe(swept.set)
            return result

        client._prepare_view_root = signalling_sweep
        try:
            backends.document_opened(main)
            await swept.wait()
            # The window: the start's sweep is done, the process not running.
            assert client.starting and not client.running
            backends.document_opened(script)
            assert await client._starting
            await asyncio.gather(*backends._tasks)
            deadline = loop.time() + 10
            while True:
                answer = await backends.request(main, lsp.TEXT_DOCUMENT_HOVER, params)
                watched = fake_hover(answer)["watched"]
                if config_uri in watched or loop.time() > deadline:
                    return watched
                await asyncio.sleep(0.05)
        finally:
            await backends.shutdown_all()

    watched = run_bounded(scenario())
    assert config_uri in watched
    assert script_environments(client) == [
        os.path.join(client.view_root.path, "scripts")
    ]


def test_an_open_during_a_sweep_that_missed_it_is_swept_again(
    project, tmp_path, fake_default
):
    # Defect: an open checks for a new environment against the manifest of
    # the last finished sweep, which still has scripts/ (scripts/x.lpy, just
    # deleted), while the sweep in flight read the open documents before
    # scripts/y.lpy was opened: nothing sweeps again, and the config loses
    # scripts/ though y.lpy, unsaved, needs it.
    backends, script, _ = unsaved_script_backends(project, tmp_path)
    (project / "scripts" / "x.lpy").write_text("(= x 1)\n")
    main = (project / "main.lpy").as_uri()
    client = backends.client_for(main)
    params = hover_params(client.backend_uri(main))
    view_scripts = os.path.join(client.view_root.path, "scripts")
    sweep = client._prepare_view_root
    gate = threading.Event()

    async def scenario():
        loop = asyncio.get_running_loop()
        began = asyncio.Event()

        def gated_sweep(open_paths):
            loop.call_soon_threadsafe(began.set)
            gate.wait(10)
            return sweep(open_paths)

        try:
            await backends.request(main, lsp.TEXT_DOCUMENT_HOVER, params, text="t")
            assert script_environments(client) == [view_scripts]
            client._prepare_view_root = gated_sweep
            (project / "scripts" / "x.lpy").unlink()
            backends.files_changed([(project / "scripts" / "x.lpy").as_uri()])
            await began.wait()
            # The sweep in flight read no open scripts/ document.
            backends.document_opened(script)
            gate.set()
            while backends._tasks:
                await asyncio.gather(*backends._tasks)
        finally:
            gate.set()
            await backends.shutdown_all()

    run_bounded(scenario())
    assert script_environments(client) == [view_scripts]


def test_opens_during_a_sweep_add_one_sweep_in_all(project, tmp_path, fake_default):
    # Defect: a sweep waiting for the running one's lock is no longer due,
    # so each later open (its own notification, as a session restore sends
    # them) sees the lock held and schedules one more full sweep: N opens,
    # N serial sweeps, and a real save waits behind them all.
    backends, script, other = unsaved_script_backends(project, tmp_path)
    main = (project / "main.lpy").as_uri()
    third = (project / "scripts" / "w.lpy").as_uri()
    client = backends.client_for(main)
    params = hover_params(client.backend_uri(main))
    sweep = client._prepare_view_root
    gate = threading.Event()
    calls = []

    async def scenario():
        loop = asyncio.get_running_loop()
        began = asyncio.Event()

        def gated_sweep(open_paths):
            calls.append(sorted(open_paths))
            loop.call_soon_threadsafe(began.set)
            gate.wait(10)
            return sweep(open_paths)

        try:
            await backends.request(main, lsp.TEXT_DOCUMENT_HOVER, params, text="t")
            client._prepare_view_root = gated_sweep
            backends.files_changed([main])
            await began.wait()
            for uri in [script, other, third]:
                backends.document_opened(uri)
                # Each open in a loop pass of its own: scheduled sweeps begin.
                for _ in range(5):
                    await asyncio.sleep(0)
            gate.set()
            while backends._tasks:
                await asyncio.gather(*backends._tasks)
        finally:
            gate.set()
            await backends.shutdown_all()

    run_bounded(scenario())
    assert len(calls) == 2
    assert len(calls[1]) == 3
    assert script_environments(client) == [
        os.path.join(client.view_root.path, "scripts")
    ]


def test_opens_before_a_starts_sweep_add_no_sweep(project, tmp_path, fake_default):
    # Defect: an open while a start is in progress but before its sweep
    # began (a session restore opening every buffer at once) is checked
    # against the manifest of no sweep at all, so it asks for a second full
    # sweep after the start, though the start's sweep reads it.
    backends, script, other = unsaved_script_backends(project, tmp_path)
    main = (project / "main.lpy").as_uri()
    client = backends.client_for(main)
    sweep = client._prepare_view_root
    calls = []

    def counting_sweep(open_paths):
        calls.append(len(open_paths))
        return sweep(open_paths)

    client._prepare_view_root = counting_sweep

    async def scenario():
        try:
            for uri in [main, script, other]:
                backends.document_opened(uri)
            assert await client._starting
            while backends._tasks or client._replays:
                await asyncio.gather(*backends._tasks, *client._replays)
        finally:
            await backends.shutdown_all()

    run_bounded(scenario())
    assert calls == [3]
    assert script_environments(client) == [
        os.path.join(client.view_root.path, "scripts")
    ]


def test_a_sweep_queued_behind_shutdown_never_runs(project, tmp_path):
    # Defect: a save waiting for the sweep lock still sweeps once shutdown
    # came (a cancelled sweep released the lock while its thread may still
    # write, and a folder removed and added again gives a new client over
    # the same view root), beside whatever writes that view root now.
    client = backend.BackendClient(view_root(project, tmp_path), ["unused"])
    client.state = "running"
    client.process = SimpleNamespace(returncode=None)
    calls = []
    client._prepare_view_root = calls.append

    async def scenario():
        await client._sweep_lock.acquire()
        saving = asyncio.ensure_future(client.file_saved())
        await asyncio.sleep(0)
        client.state = "closed"
        client._sweep_lock.release()
        await saving

    run_bounded(scenario())
    assert calls == []


def linked_package(tmp_path, init=True):
    """proj/src/acme, a directory link out of the project to
    <tmp>/shared/acme, which holds m.lpy and its .py twin (and __init__.py
    when init); the project and its one workspace folder."""
    shared = tmp_path / "shared" / "acme"
    shared.mkdir(parents=True)
    if init:
        (shared / "__init__.py").write_text("")
    (shared / "m.lpy").write_text("(= m 1)\n")
    (shared / "m.py").write_text("m = 1\n")
    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    (proj / "src" / "acme").symlink_to(shared)
    return proj, [SimpleNamespace(uri=proj.as_uri())]


def test_a_document_behind_a_directory_link_is_in_its_project(tmp_path):
    # Defect: the document's directory is resolved, so proj/src/acme/m.lpy
    # lies in <tmp>/shared, in no workspace folder, and gets no backend and
    # no shadow, while Python imports it through the link as acme.m.
    proj, folders = linked_package(tmp_path)
    doc = str(proj / "src" / "acme" / "m.lpy")
    assert backend.project_root_of(doc, folders) == os.path.realpath(proj)
    assert backend.document_path(doc, folders) == doc
    client = backend.BackendClient(
        view_root(proj, tmp_path), ["unused"], folders=lambda: folders
    )
    assert (
        client.backend_uri(pathlib.Path(doc).as_uri())
        == (pathlib.Path(client.view_root.path) / "src" / "acme" / "m.py").as_uri()
    )


@pytest.mark.parametrize("init", [True, False])
def test_a_linked_package_is_named_as_python_imports_it(tmp_path, init):
    # Defect: with the directory resolved the document has no project, so
    # the namespace package acme behind the link is lost (the script m
    # instead of acme.m from proj/src).  A real python run pins the name
    # and the spelled __file__ the import takes through the link.
    proj, folders = linked_package(tmp_path, init=init)
    src = str(proj / "src")
    name, _, import_root, search_dirs = server.document_module_names(
        str(proj / "src" / "acme" / "m.lpy"), folders=folders
    )
    code = (
        "import json, sys; sys.path.insert(0, sys.argv[1]); import acme.m; "
        "print(json.dumps([acme.m.__name__, acme.m.__file__]))"
    )
    run = subprocess.run(
        [sys.executable, "-I", "-c", code, src],
        capture_output=True,
        text=True,
        check=True,
    )
    py_name, py_file = json.loads(run.stdout)
    assert (name, import_root) == (py_name, src)
    assert py_file == str(proj / "src" / "acme" / "m.py")
    assert src in search_dirs


def test_a_document_through_a_folder_link_is_named_under_the_real_root(tmp_path):
    # Defect: the document keeps the folder link's spelling, so its import
    # root is link/src while the project root and the shadow sweep's
    # sources are spelled by the real root: one file, two namings.
    proj = tmp_path / "proj"
    (proj / "src" / "pkg").mkdir(parents=True)
    (proj / "src" / "pkg" / "__init__.py").write_text("")
    (proj / "src" / "pkg" / "m.lpy").write_text("(= m 1)\n")
    link = tmp_path / "link"
    link.symlink_to(proj)
    folders = [SimpleNamespace(uri=link.as_uri())]
    spelled = str(link / "src" / "pkg" / "m.lpy")
    real = str(proj / "src" / "pkg" / "m.lpy")
    names = server.document_module_names(spelled, folders=folders)
    assert names == server.document_module_names(real, folders=folders)
    assert names[2] == str(proj / "src")
    assert backend.document_path(spelled, folders) == real


def test_the_sweep_shadows_a_package_linked_into_src(tmp_path):
    # Defect: the sweep follows no directory link, so src/acme -> a package
    # outside the project gets no shadow, and the backend resolves acme.m to
    # nothing the user wrote.
    proj, _ = linked_package(tmp_path)
    doc = str(proj / "src" / "acme" / "m.lpy")
    assert pv.project_sources(str(proj)) == [doc]
    view = tmp_path / "view"
    sweep = pv.write_shadows(str(proj), str(view))
    shadow = view / "src" / "acme" / "m.py"
    assert sweep.shadows == [str(shadow)]
    assert shadow.read_text() == "m = 1"
    init = view / "src" / "acme" / "__init__.py"
    assert os.readlink(init) == str(proj / "src" / "acme" / "__init__.py")


def test_a_package_linked_within_the_project_is_listed_by_its_link_too(tmp_path):
    # Defect: a walk entering each real directory once, its real spelling
    # included, enters shared/acme (sorted first) and skips src/acme, so the
    # spelling Python imports as acme.m gets no shadow.  A second link to
    # the same package (src/zeta) is entered no more, and the loop up ends.
    proj = tmp_path / "proj"
    (proj / "shared" / "acme").mkdir(parents=True)
    (proj / "shared" / "acme" / "m.lpy").write_text("")
    (proj / "src").mkdir()
    (proj / "src" / "acme").symlink_to(proj / "shared" / "acme")
    (proj / "src" / "zeta").symlink_to(proj / "shared" / "acme")
    (proj / "src" / "acme" / "up").symlink_to(proj / "src")
    assert pv.project_sources(str(proj)) == [
        str(proj / "shared" / "acme" / "m.lpy"),
        str(proj / "src" / "acme" / "m.lpy"),
    ]


def test_the_sweep_never_walks_up_out_of_the_project(tmp_path, time_bound):
    # Defect: the loop guard holds only the directories below the project
    # root, so proj/up -> .. (or a link to ~ or /) walks the whole parent
    # tree and lists every .lpy in it.
    ws = tmp_path / "ws"
    (ws / "other").mkdir(parents=True)
    (ws / "other" / "x.lpy").write_text("")
    proj = ws / "proj"
    proj.mkdir()
    (proj / "a.lpy").write_text("")
    (proj / "up").symlink_to(ws)
    assert pv.project_sources(str(proj)) == [str(proj / "a.lpy")]


def test_the_sweep_walks_a_dag_of_links_in_linear_time(tmp_path, time_bound):
    # Defect: a directory reached by two links is walked under each, so a
    # chain of directories each linking twice to the next lists 2**n paths.
    proj = tmp_path / "proj"
    count = 8
    for i in range(count):
        (proj / f"d{i}").mkdir(parents=True)
        (proj / f"d{i}" / "m.lpy").write_text("")
    for i in range(count - 1):
        for name in ["a", "b"]:
            (proj / f"d{i}" / name).symlink_to(proj / f"d{i + 1}")
    # Each directory by its own spelling, and each but d0 once through a link.
    assert len(pv.project_sources(str(proj))) == 2 * count - 1


def test_a_linked_documents_shadow_and_environment_are_its_own(project, tmp_path):
    # Defect: the document's file is resolved, so a link scripts/a.lpy ->
    # lib/a.lpy is synced at lib/a.py, where lib/a.lpy's own buffer is, not
    # at the shadow scripts/a.py the sweep writes for it, and its search
    # environment is lib/'s, where `lpy scripts/a.lpy` searches scripts/.
    # The buffer does not compile, so only the open gives an environment.
    (project / "lib").mkdir()
    (project / "lib" / "a.lpy").write_text("(= a\n")
    (project / "scripts").mkdir()
    (project / "scripts" / "a.lpy").symlink_to(project / "lib" / "a.lpy")
    client = backend.BackendClient(view_root(project, tmp_path), ["unused"])
    linked = (project / "scripts" / "a.lpy").as_uri()
    assert (
        client.backend_uri(linked)
        == (pathlib.Path(client.view_root.path) / "scripts" / "a.py").as_uri()
    )
    client.open_document(linked)
    client._prepare_view_root(list(client._open_paths.values()))
    real = os.path.realpath(project)
    assert list(client.manifest.environments) == [
        real,
        os.path.join(real, "scripts"),
    ]


def test_opens_in_one_new_directory_sweep_once(project, tmp_path, fake_default):
    # Defect: each open whose environment is new schedules a sweep of its
    # own, so two opens in one new directory (or a folders change, which
    # reopens every document) compile the whole project once per document.
    backends, script, other = unsaved_script_backends(project, tmp_path)
    main = (project / "main.lpy").as_uri()
    client = backends.client_for(main)
    params = hover_params(client.backend_uri(main))
    sweeps = []
    saved = client.saved

    async def counting_saved(*uris):
        sweeps.append(uris)
        await saved(*uris)

    async def scenario():
        try:
            await backends.request(main, lsp.TEXT_DOCUMENT_HOVER, params, text="t")
            client.saved = counting_saved
            backends.document_opened(script)
            backends.document_opened(other)
            while backends._tasks:
                await asyncio.gather(*backends._tasks)
        finally:
            await backends.shutdown_all()

    run_bounded(scenario())
    assert sweeps == [()]
    assert script_environments(client) == [
        os.path.join(client.view_root.path, "scripts")
    ]


@pytest.mark.parametrize("name", list(REAL_BACKENDS))
def test_real_backend_resolves_a_sibling_of_a_script_opened_unsaved(
    project, tmp_path, name
):
    # Defects: a buffer of scripts/y.lpy not yet on disk, opened while the
    # backend runs, is analysed under the default search list, so its
    # (import sibling) of scripts/sibling.py, which `python scripts/y.py`
    # resolves, is reported unresolved until the first save; or the
    # running backend does not take the config the open's sweep wrote (a
    # new nested pyrefly.toml, a new executionEnvironments entry).
    argv = backend.backend_argv(name)
    if argv is None:
        pytest.skip(f"{name} is not installed (its lsp extra)")
    scripts = project / "scripts"
    scripts.mkdir()
    (scripts / "sibling.py").write_text("X = 1\n")
    source = "(import sibling)\n(import no-such-module-anywhere)\n(print sibling.X)\n"
    view = pv.compile_view(source)
    main_source = "(= y 1)\n"
    main_view = pv.compile_view(main_source)
    client = backend.BackendClient(view_root(project, tmp_path), argv, name=name)
    main = (project / "main.lpy").as_uri()
    script = (scripts / "y.lpy").as_uri()

    def messages(uri, text):
        found = client.diagnostics_for(uri, text)
        return None if found is None else [d.message for d, _ in found]

    async def wait_for(done):
        deadline = asyncio.get_running_loop().time() + 30
        while not done():
            heard.clear()
            left = deadline - asyncio.get_running_loop().time()
            if left <= 0:
                return
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(heard.wait(), left)

    def settled():
        # Positive control: the module no directory holds is reported.
        found = messages(script, view.text) or []
        return any("no_such_module" in m for m in found) and not any(
            "sibling" in m for m in found
        )

    async def scenario():
        try:
            client.open_document(main)
            client.push_buffer(main, main_source, main_view)
            await client.request(
                lsp.TEXT_DOCUMENT_HOVER, hover_params(client.backend_uri(main))
            )
            await wait_for(lambda: messages(main, main_view.text) is not None)
            assert client.running and messages(main, main_view.text) is not None
            # As Backends.document_opened then the open's compile: the sweep
            # runs while the buffer reaches the backend.
            assert client.open_document(script)
            sweep = asyncio.create_task(client.saved())
            client.push_buffer(script, source, view)
            await sweep
            await wait_for(settled)
            return messages(script, view.text)
        finally:
            await client.shutdown()

    heard = asyncio.Event()
    client.on_diagnostics = lambda lpy_uri: heard.set()
    found = run_bounded(scenario(), timeout=90)
    assert found is not None and any("no_such_module" in m for m in found)
    assert not any("sibling" in m for m in found)


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
def test_real_pyrefly_sees_shadows_from_a_pure_py_package(project, tmp_path):
    # Defect: a pure-.py package beside the shadowed packages is not in the
    # view, so pyrefly analyses the real file, which cannot see the shadows:
    # a name it imports from a .lpy module is Unknown.
    for name, text in [
        ("lpkg/__init__.lpy", ""),
        ("lpkg/core.lpy", "(def ffff [a] (return (+ a 1)))\n"),
        ("purepkg/__init__.py", ""),
        ("purepkg/helper.py", "from lpkg.core import ffff\nX = ffff\n"),
    ]:
        (project / name).parent.mkdir(parents=True, exist_ok=True)
        (project / name).write_text(text)
    source = "(from purepkg.helper [X])\n(= z X)\n"
    (project / "main.lpy").write_text(source)
    client = backend.BackendClient(
        view_root(project, tmp_path), backend.backend_argv("pyrefly"), name="pyrefly"
    )
    uri = (project / "main.lpy").as_uri()
    view = pv.compile_view(source)

    async def scenario():
        try:
            line, character = locate(view.text, "z =")
            answer = await client.request(
                lsp.TEXT_DOCUMENT_HOVER,
                hover_params(client.backend_uri(uri), line, character),
                uri=uri,
                text=view.text,
            )
            return None if answer is None else answer.contents.value
        finally:
            await client.shutdown()

    hover = run_bounded(scenario(), timeout=60)
    assert hover is not None and "ffff" in hover


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


@needs_pyrefly
def test_real_pyrefly_config_problem_keeps_a_description_holding_a_bar():
    # Defect: every output line holding a | is taken for the text excerpt,
    # so a description quoting a value with a | in it (unknown variant
    # `a|b`) is dropped and the generic answer given instead.
    binary = backend.backend_argv("pyrefly")[0]
    problem = backend.pyrefly_config_problem(binary, 'untyped-def-behavior = "a|b"\n')
    assert "unknown variant `a|b`" in problem


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
