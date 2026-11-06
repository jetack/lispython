"""Fixtures shared by the test modules."""

import asyncio
import importlib
import itertools
import signal
import threading
from types import SimpleNamespace

import pytest

# Seconds: ~4x the slowest guarded test (test_nesting's fstring-format-spec
# case, roughly linear in DEPTH, ~1.6 s on 3.11 idle), so a loaded box does
# not trip it while a runaway or quadratic regression still does. It also
# caps a runaway's memory (see time_bound) at about 720 MB; raising it
# raises that cap.
SELF_REFERENCE_BUDGET = 6.0


@pytest.fixture
def time_bound():
    """Fail the test once it runs past SELF_REFERENCE_BUDGET.

    A self-referencing form has no finite copy: without the checks the
    conversion grows memory without bound (about 120 MB/s), fast enough to
    exhaust the machine before any memory error. A nested form that costs
    exponential time to compile never finishes. The timer stops either first.
    Where setitimer is missing the test is skipped, never run unguarded.
    """
    if not hasattr(signal, "setitimer"):
        pytest.skip("needs signal.setitimer to bound a runaway self-reference")

    def expire(signum, frame):
        pytest.fail(f"ran past its {SELF_REFERENCE_BUDGET} s budget")

    old_handler = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, SELF_REFERENCE_BUDGET)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# The LSP server over a real Python backend (test_lsp_completion,
# test_lsp_builtin_members): one basedpyright process for the session, over
# one tmp project, on one event loop in a background thread (a basedpyright
# start costs about a second and 400 MB). The lispy.lsp modules are imported
# in the fixtures, so a run that uses none never loads the server.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def run_on_loop():
    """run_on_loop(coroutine, timeout=60): its result, run on the session's
    background event loop."""
    event_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=event_loop.run_forever, daemon=True)
    thread.start()

    def run(coroutine, timeout=60):
        return asyncio.run_coroutine_threadsafe(coroutine, event_loop).result(timeout)

    yield run
    event_loop.call_soon_threadsafe(event_loop.stop)
    thread.join(10)
    event_loop.close()


@pytest.fixture(scope="session")
def backend_project(tmp_path_factory):
    """The workspace folder of the backend tests; its main.lpy is the file
    every buffer is opened as."""
    root = tmp_path_factory.mktemp("backend-project")
    (root / "main.lpy").write_text("(= y 1)\n")
    return root


@pytest.fixture(scope="session")
def make_backends(backend_project):
    """make_backends(cache, name): a Backends over backend_project alone,
    configured with backend name, its view roots under cache."""
    backend = importlib.import_module("lispy.lsp.backend")

    def make(cache, name):
        backends = backend.Backends(
            lambda: [SimpleNamespace(uri=backend_project.as_uri())],
            cache_home=str(cache),
        )
        backends.configure({"backend": name})
        return backends

    return make


@pytest.fixture(scope="session")
def real_backends(run_on_loop, make_backends, tmp_path_factory):
    """The session's basedpyright Backends (started by its first request)."""
    backends = make_backends(tmp_path_factory.mktemp("cache"), "basedpyright")
    yield backends
    run_on_loop(backends.shutdown_all())


@pytest.fixture
def off_backends(make_backends, tmp_path):
    return make_backends(tmp_path / "cache", "off")


@pytest.fixture
def lsp_buffer(backend_project, real_backends, monkeypatch):
    """lsp_buffer(marked, backends=real_backends, **ls_attrs) -> (ls, uri,
    position): the server pointed at backends, and a language server double
    whose workspace holds marked (``|`` is the cursor, removed) as
    backend_project's main.lpy at a new version; position is the cursor in
    UTF-16 units, as a client sends it. lsp_buffer.document is the last
    buffer, a real pygls TextDocument, so lines and the position codec are
    the ones the server uses. Each test gets a fresh view cache."""
    from lsprotocol import types as lsp
    from pygls.workspace import TextDocument

    server = importlib.import_module("lispy.lsp.server")
    python_view = importlib.import_module("lispy.lsp.python_view")
    monkeypatch.setattr(server, "view_cache", python_view.ViewCache())
    versions = itertools.count(1)
    uri = (backend_project / "main.lpy").as_uri()

    def open_buffer(marked, backends=real_backends, **ls_attrs):
        monkeypatch.setattr(server, "backends", backends)
        before, after = marked.split("|")
        line_start = before.rfind("\n") + 1
        position = lsp.Position(
            line=before.count("\n"),
            character=len(before[line_start:].encode("utf-16-le")) // 2,
        )
        document = TextDocument(uri, before + after, version=next(versions))
        open_buffer.document = document
        workspace = SimpleNamespace(get_text_document=lambda uri: document)
        return SimpleNamespace(workspace=workspace, **ls_attrs), uri, position

    return open_buffer


@pytest.fixture
def backend_requests(backend_project, real_backends, monkeypatch):
    """The methods of the requests sent to the session's backend (sent), and
    a way to set its answer (answer["value"]) or act while it is asked
    (answer["during"])."""
    client = real_backends.available_client((backend_project / "main.lpy").as_uri())
    sent = []
    real_request = client.request
    answer = {}

    async def request(method, params, **kwargs):
        sent.append(method)
        if "during" in answer:
            answer["during"]()
        if "value" in answer:
            return answer["value"]
        return await real_request(method, params, **kwargs)

    monkeypatch.setattr(client, "request", request)
    return SimpleNamespace(sent=sent, answer=answer)
