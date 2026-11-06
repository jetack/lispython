"""Completion through the real Python backend (basedpyright).

The tests share one basedpyright process over one tmp project, on one event
loop in a background thread (a basedpyright start costs about a second and
400 MB); each test points the server at it and gets a fresh view cache. The
document is a real pygls TextDocument, so lines and the position codec are
the ones the server uses. ``|`` in a source marks the cursor.
"""

import asyncio
import importlib
import itertools
import threading
import time
from types import SimpleNamespace

import pytest
from lsprotocol import types as lsp
from pygls.workspace import TextDocument

import lispy  # noqa: F401  (registers the .lpy importer)

server = importlib.import_module("lispy.lsp.server")
backend = importlib.import_module("lispy.lsp.backend")
pv = importlib.import_module("lispy.lsp.python_view")

pytestmark = pytest.mark.skipif(
    backend.backend_argv("basedpyright") is None,
    reason="basedpyright is not installed (the lsp extra)",
)

CLASS = (
    "(class A []\n"
    "  (def __init__ [self]\n"
    "    (= self.my-x 1))\n"
    "  (def f [self]\n"
    "    {}))\n"
)


@pytest.fixture(scope="module")
def loop():
    event_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=event_loop.run_forever, daemon=True)
    thread.start()
    yield event_loop
    event_loop.call_soon_threadsafe(event_loop.stop)
    thread.join(10)
    event_loop.close()


def on_loop(event_loop, coroutine, timeout=60):
    return asyncio.run_coroutine_threadsafe(coroutine, event_loop).result(timeout)


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("completion-project")
    (root / "main.lpy").write_text("(= y 1)\n")
    return root


def make_backends(project, cache, name):
    backends = backend.Backends(
        lambda: [SimpleNamespace(uri=project.as_uri())], cache_home=str(cache)
    )
    backends.configure({"backend": name})
    return backends


@pytest.fixture(scope="module")
def real_backends(loop, project, tmp_path_factory):
    backends = make_backends(project, tmp_path_factory.mktemp("cache"), "basedpyright")
    yield backends
    on_loop(loop, backends.shutdown_all())


@pytest.fixture
def off_backends(project, tmp_path):
    return make_backends(project, tmp_path / "cache", "off")


def capabilities(insert_replace):
    return lsp.ClientCapabilities(
        text_document=lsp.TextDocumentClientCapabilities(
            completion=lsp.CompletionClientCapabilities(
                completion_item=lsp.ClientCompletionItemOptions(
                    insert_replace_support=insert_replace
                )
            )
        )
    )


@pytest.fixture
def complete(loop, project, real_backends, monkeypatch):
    """complete(source) -> (line, character, items) at the | of source."""
    monkeypatch.setattr(server, "view_cache", pv.ViewCache())
    versions = itertools.count(1)
    uri = (project / "main.lpy").as_uri()

    def run(source, backends=real_backends, insert_replace=True):
        monkeypatch.setattr(server, "backends", backends)
        before, after = source.split("|")
        line = before.count("\n")
        character = len(before) - (before.rfind("\n") + 1)
        document = TextDocument(uri, before + after, version=next(versions))
        run.document = document
        ls = SimpleNamespace(
            workspace=SimpleNamespace(get_text_document=lambda uri: document),
            client_capabilities=capabilities(insert_replace),
        )
        params = lsp.CompletionParams(
            text_document=lsp.TextDocumentIdentifier(uri=uri),
            position=lsp.Position(line=line, character=character),
        )
        result = on_loop(loop, server.completion(ls, params))
        return line, character, result.items

    return run


def by_label(items):
    return {item.label: item for item in items}


def test_members_take_the_buffer_spelling(complete):
    # Defects: the backend's Python name is shown (my_x beside or instead of
    # my-x); the filterText drops the item once the user types my- (or
    # keeps only the hyphen spelling when nothing hyphenated was typed).
    items = by_label(complete(CLASS.format("self.|"))[2])
    assert "my-x" in items and "my_x" not in items
    assert "__class__" in items  # inherited from object: only the backend knows
    assert items["my-x"].filter_text == "my_x"
    items = by_label(complete(CLASS.format("self.my-|"))[2])
    assert items["my-x"].filter_text == "my-x"


@pytest.mark.parametrize(
    "source, label, typed_filter",
    [
        (
            "(class A [Exception]\n  (def f [self]\n    self.with-|))",
            "with_traceback",
            "with-traceback",
        ),
        (
            "(from collections [OrderedDict])\n"
            "(class A [OrderedDict]\n  (def f [self]\n    self.move-|))",
            "move_to_end",
            "move-to-end",
        ),
    ],
    ids=["builtin base", "imported base"],
)
def test_a_member_never_spelled_in_the_buffer(complete, source, label, typed_filter):
    # Defect: a hyphen typed toward a member the buffer never spells drops
    # it, since the client filters with-  against with_traceback.
    items = by_label(complete(source)[2])
    assert items[label].filter_text == typed_filter


@pytest.mark.parametrize("source", ["osp.jo|in", "(. osp jo|in)"])
def test_edit_ranges_with_the_cursor_mid_word(complete, source):
    # Defects: the replace range stops at the cursor (accepting leaves the
    # rest of the word behind), or an InsertReplaceEdit goes to a client
    # that cannot take one.
    source = "(import os.path as osp)\n" + source
    line, character, items = complete(source)
    edit = by_label(items)["join"].text_edit
    assert isinstance(edit, lsp.InsertReplaceEdit)
    assert edit.new_text == "join"
    assert (edit.insert.start.character, edit.insert.end.character) == (
        character - 2,
        character,
    )
    assert (edit.replace.start.character, edit.replace.end.character) == (
        character - 2,
        character + 2,
    )
    assert {edit.insert.start.line, edit.replace.end.line} == {line}

    _, _, items = complete(source, insert_replace=False)
    edit = by_label(items)["join"].text_edit
    assert isinstance(edit, lsp.TextEdit)
    assert (edit.range.start.character, edit.range.end.character) == (
        character - 2,
        character,
    )


@pytest.mark.parametrize("access", ["osp.jo|", "(. osp jo|)"])
def test_module_attributes_after_an_alias(complete, access):
    # Regression TASK-1010: attributes of an aliased module offered the
    # general list (print, ...) instead of the module's members.
    labels = set(by_label(complete(f"(import os.path as osp)\n{access}")[2]))
    assert "join" in labels
    assert "print" not in labels


IMPORTED_CLASS = "(from collections [defaultdict])\n(defau|"


def kinds_of(items, label):
    return [item.kind for item in items if item.label == label]


@pytest.fixture
def idle(loop, project, tmp_path):
    """A Backends of its own whose basedpyright has not started."""
    backends = make_backends(project, tmp_path / "idle-cache", "basedpyright")
    yield backends, backends.available_client((project / "main.lpy").as_uri())
    on_loop(loop, backends.shutdown_all())


def wait_running(client, seconds=30):
    """Poll (bounded) until client's backend runs; True when it does."""
    deadline = time.monotonic() + seconds
    while not client.running and time.monotonic() < deadline:
        time.sleep(0.05)
    return client.running


def test_a_non_attribute_completion_never_waits_for_the_backend(complete, idle):
    # Defect: a keystroke off an attribute waits for the backend's start
    # (about a second) for kinds that only refine the local list.
    backends, client = idle
    items = complete(IMPORTED_CLASS, backends)[2]
    assert not client.running  # answered before the backend was up
    assert kinds_of(items, "defaultdict") == [lsp.CompletionItemKind.Variable]


def test_an_imported_class_is_one_class_item(complete, idle, monkeypatch):
    # Regression TASK-1014: an imported defaultdict was a Variable, shown
    # beside the nREPL's class item as a second defaultdict. The first
    # completion starts the backend without waiting; once it runs, the
    # next one has its kinds. A loaded box may answer past KINDS-TIMEOUT,
    # which this test does not pin.
    monkeypatch.setattr(server, "KINDS_TIMEOUT", 30)
    backends, client = idle
    complete(IMPORTED_CLASS, backends)
    assert wait_running(client)
    items = complete(IMPORTED_CLASS, backends)[2]
    assert kinds_of(items, "defaultdict") == [lsp.CompletionItemKind.Class]


def test_auto_import_completions_are_off(complete, loop, project, real_backends):
    # Defect: the backend offers symbols the buffer never imports, with an
    # import edit on a generated line (and their kinds).
    complete("(import os)\nos.pa|")
    client = real_backends.available_client((project / "main.lpy").as_uri())
    document = server.TextDocument(
        (project / "main.lpy").as_uri(), "(= default-x 1)\n(defau)", version=None
    )
    _, items = on_loop(loop, server.backend_answer(client, document, 2, 6, None))
    assert "default_x" in {item.label for item in items}  # it did answer
    assert "defaultdict" not in {item.label for item in items}


LOCAL_MEMBERS = {"__init__", "my-x", "f"}


@pytest.fixture
def requests(project, real_backends, monkeypatch):
    """The completion requests sent to the shared backend, and a way to set
    its answer."""
    client = real_backends.available_client((project / "main.lpy").as_uri())
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


@pytest.mark.parametrize(
    "empty", [[], lsp.CompletionList(is_incomplete=False, items=[])], ids=repr
)
def test_a_backend_with_no_items_gives_the_local_members(complete, requests, empty):
    # Defect: an empty backend answer is returned as is: no completion.
    requests.answer["value"] = empty
    labels = set(by_label(complete(CLASS.format("self.|"))[2]))
    assert requests.sent == [lsp.TEXT_DOCUMENT_COMPLETION]
    assert labels == LOCAL_MEMBERS


def test_backend_off_gives_the_local_members(complete, off_backends, monkeypatch):
    # Defect: an off backend still has a view compiled (the buffer's macros
    # run) for a request that cannot be answered.
    views = []
    monkeypatch.setattr(server, "cursor_view", lambda *a: views.append(a))
    labels = set(by_label(complete(CLASS.format("self.|"), off_backends)[2]))
    assert (labels, views) == (LOCAL_MEMBERS, [])


def test_a_buffer_that_does_not_compile_gives_the_local_members(complete, requests):
    # An earlier refused form: neither the buffer nor its prefix compiles,
    # so there is no view to ask the backend about.
    labels = set(by_label(complete("(def)\n" + CLASS.format("self.|"))[2]))
    assert (labels, requests.sent) == (LOCAL_MEMBERS, [])
    # The backend does answer the same buffer once it compiles.
    labels = set(by_label(complete(CLASS.format("self.|"))[2]))
    assert "__class__" in labels and requests.sent


def test_an_edit_during_the_request_never_moves_the_answer(complete, requests):
    # Defect: the handler reads the live document after its awaits, so an
    # edit made meanwhile (here: the buffer emptied) gives the old cursor
    # column against the new text: the range starts at column 0 and the
    # typed my- is lost from the filterText.
    requests.answer["during"] = lambda: setattr(complete.document, "_source", "")
    line, character, items = complete(CLASS.format("self.my-|"))
    item = by_label(items)["my-x"]
    assert item.filter_text == "my-x"
    assert item.text_edit.insert.start.character == character - 3
