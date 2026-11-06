"""The LSP's Python backend client (lispy.lsp.backend).

The real-backend tests share one basedpyright process on one event loop in a
background thread (a basedpyright start costs about a second and 400 MB); the
others run a fake backend, a tiny pygls server whose hover answers with its
pid and its copy of the document, each on its own loop.
"""

import asyncio
import importlib
import os
import signal
import sys
import textwrap
import threading
from types import SimpleNamespace

import pytest
from lsprotocol import types as lsp

import lispy  # noqa: F401  (registers the .lpy importer)

backend = importlib.import_module("lispy.lsp.backend")
pv = importlib.import_module("lispy.lsp.python_view")

FAKE_SERVER = textwrap.dedent(
    """\
    import os

    from lsprotocol import types
    from pygls.lsp.server import LanguageServer

    server = LanguageServer("fake-backend", "0")


    @server.feature(types.TEXT_DOCUMENT_HOVER)
    def hover(ls, params):
        doc = ls.workspace.get_text_document(params.text_document.uri)
        return types.Hover(contents=f"{os.getpid()}|{doc.source}")


    server.start_io()
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
def fake_argv(tmp_path):
    script = tmp_path / "fake_backend.py"
    script.write_text(FAKE_SERVER)
    return [sys.executable, str(script)]


def fake_hover(answer):
    """(pid, document text) of a fake backend's hover, or None."""
    if answer is None:
        return None
    pid, _, text = answer.contents.partition("|")
    return int(pid), text


def run_bounded(coroutine, timeout=30):
    async def bounded():
        async with asyncio.timeout(timeout):
            return await coroutine

    return asyncio.run(bounded())


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
    "init_options", [{"backend": "off"}, {"backend": "no-such"}, None]
)
def test_off_never_spawns(init_options, project, tmp_path, spawned, monkeypatch):
    # Defect: an off, unknown or (the default on a plain install) missing
    # backend still starts a process, or a request on it raises.
    monkeypatch.setitem(backend.BACKENDS, "basedpyright", lambda: None)
    folder = SimpleNamespace(uri=project.as_uri())
    backends = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "cache"))
    backends.configure(init_options)
    uri = (project / "main.lpy").as_uri()
    answer = run_bounded(
        backends.request(uri, lsp.TEXT_DOCUMENT_HOVER, hover_params(uri), text="x")
    )
    assert answer is None
    assert spawned == []


def test_document_outside_the_workspace_has_no_backend(project, tmp_path, fake_argv):
    # Defect: a file outside every workspace folder (a dependency opened by
    # go-to-definition) gets a backend over a project it is not in.
    folder = SimpleNamespace(uri=project.as_uri())
    backends = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "cache"))
    backends.configure(None)
    outside = tmp_path / "elsewhere.lpy"
    outside.write_text("(= z 1)\n")
    assert backends.client_for(outside.as_uri()) is None
    inside = backends.client_for((project / "main.lpy").as_uri())
    assert inside is not None and inside.project_root == os.path.realpath(project)


def test_buffer_hooks_never_start_a_backend(project, tmp_path, fake_argv, spawned):
    # Defect: opening, editing or saving a buffer spawns the backend (a
    # process per project on a plain edit) or compiles a view for nobody.
    folder = SimpleNamespace(uri=project.as_uri())
    backends = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "cache"))
    backends.configure(None)
    uri = (project / "main.lpy").as_uri()
    client = backends.client_for(uri)
    client.argv = fake_argv
    read = []

    def source():
        read.append(True)
        return "(= y 2)\n"

    async def scenario():
        backends.document_changed(uri, source)
        backends.document_saved(uri)
        backends.document_closed(uri)
        others = asyncio.all_tasks() - {asyncio.current_task()}
        await asyncio.gather(*others)

    run_bounded(scenario())
    assert spawned == []
    assert read == []


# ---------------------------------------------------------------------------
# Fake backend: crashes, shutdown, buffer sync
# ---------------------------------------------------------------------------


def test_crashed_backend_restarts_once(project, tmp_path, fake_argv, spawned):
    # Defect: a backend that crashed stays dead for the session, or one that
    # keeps crashing is respawned on every request.
    client = backend.BackendClient(view_root(project, tmp_path), fake_argv)
    uri = (project / "main.lpy").as_uri()

    async def hover():
        params = hover_params(client.backend_uri(uri))
        return fake_hover(
            await client.request(lsp.TEXT_DOCUMENT_HOVER, params, uri=uri, text="t")
        )

    async def crash():
        process = client.process
        os.kill(process.pid, signal.SIGKILL)
        await process.wait()

    async def scenario():
        try:
            first = await hover()
            assert first == (client.process.pid, "t")
            await crash()
            second = await hover()
            assert second is not None and second[0] != first[0]
            await crash()
            assert await hover() is None
            assert await hover() is None
        finally:
            await client.shutdown()

    run_bounded(scenario())
    assert len(spawned) == 2


def test_shutdown_ends_the_backend(project, tmp_path, fake_argv, spawned):
    # Defect: the backend process outlives the LSP's shutdown, or a request
    # after shutdown starts a new one that nobody stops.
    client = backend.BackendClient(view_root(project, tmp_path), fake_argv)
    uri = (project / "main.lpy").as_uri()
    params = hover_params(client.backend_uri(uri))

    async def scenario():
        assert await client.request(lsp.TEXT_DOCUMENT_HOVER, params, uri=uri, text="t")
        process = client.process
        await client.shutdown()
        assert process.returncode is not None
        assert await client.request(lsp.TEXT_DOCUMENT_HOVER, params) is None

    run_bounded(scenario())
    assert len(spawned) == 1


def test_running_backend_follows_the_buffer(project, tmp_path, fake_argv):
    # Defect: edits to a buffer never reach a running backend, so it answers
    # for the text of the last request; or a closed buffer stays open in it.
    folder = SimpleNamespace(uri=project.as_uri())
    backends = backend.Backends(lambda: [folder], cache_home=str(tmp_path / "cache"))
    uri = (project / "main.lpy").as_uri()
    client = backends.client_for(uri)
    client.argv = fake_argv
    params = hover_params(client.backend_uri(uri))

    async def backend_text():
        answer = fake_hover(
            await backends.request(uri, lsp.TEXT_DOCUMENT_HOVER, params)
        )
        return None if answer is None else answer[1]

    async def scenario():
        try:
            await backends.request(uri, lsp.TEXT_DOCUMENT_HOVER, params, text="old")
            backends.document_changed(uri, lambda: "(= my-y 2)")
            while (text := await backend_text()) == "old":
                await asyncio.sleep(0.01)
            assert text == "my_y = 2"
            backends.document_closed(uri)
            # Closed, the backend reads the document's uri from disk: the
            # shadow of main.lpy as saved, not the buffer.
            assert await backend_text() == "y = 1"
        finally:
            await backends.shutdown_all()

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
    assert "path: Module" in real_hover(
        loop, real, "(import os)\n(= x os.path)", "path"
    )
    assert "(variable) sep: LiteralString" in real_hover(
        loop, real, "(import os)\n(= x os.sep)", "sep"
    )


def test_real_backend_resolves_a_sibling_lpy_module(loop, real):
    # Defect: the backend starts before the shadows are written, so an
    # import of another .lpy module of the project does not resolve.
    text = real_hover(loop, real, "(from helper [greet])\n(greet 1)", "greet", 1)
    assert "Say hello to name." in text
