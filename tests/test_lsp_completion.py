"""Completion through the real Python backend (basedpyright).

The tests share the session's basedpyright over one tmp project
(``real_backends``, ``lsp_buffer`` in conftest.py); each test points the
server at it and gets a fresh view cache. ``|`` in a source marks the cursor.
"""

import importlib
import time

import pytest
from lsprotocol import types as lsp

import lispy  # noqa: F401  (registers the .lpy importer)

server = importlib.import_module("lispy.lsp.server")
backend = importlib.import_module("lispy.lsp.backend")

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
def complete(lsp_buffer, run_on_loop, real_backends):
    """complete(source) -> (line, character, items) at the | of source."""

    def run(source, backends=real_backends, insert_replace=True):
        ls, uri, position = lsp_buffer(
            source, backends, client_capabilities=capabilities(insert_replace)
        )
        run.document = lsp_buffer.document
        params = lsp.CompletionParams(
            text_document=lsp.TextDocumentIdentifier(uri=uri), position=position
        )
        result = run_on_loop(server.completion(ls, params))
        return position.line, position.character, result.items

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
def idle(run_on_loop, make_backends, backend_project, tmp_path):
    """A Backends of its own whose basedpyright has not started."""
    backends = make_backends(tmp_path / "idle-cache", "basedpyright")
    yield backends, backends.available_client((backend_project / "main.lpy").as_uri())
    run_on_loop(backends.shutdown_all())


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
    # next one has its items. A loaded box may answer past MERGE-TIMEOUT,
    # which this test does not pin.
    monkeypatch.setattr(server, "MERGE_TIMEOUT", 30)
    backends, client = idle
    complete(IMPORTED_CLASS, backends)
    assert wait_running(client)
    items = complete(IMPORTED_CLASS, backends)[2]
    assert kinds_of(items, "defaultdict") == [lsp.CompletionItemKind.Class]


def test_auto_import_completions_are_off(
    complete, run_on_loop, backend_project, real_backends
):
    # Defect: the backend offers symbols the buffer never imports, with an
    # import edit on a generated line (and their kinds).
    complete("(import os)\nos.pa|")
    client = real_backends.available_client((backend_project / "main.lpy").as_uri())
    document = server.TextDocument(
        (backend_project / "main.lpy").as_uri(),
        "(= default-x 1)\n(defau)",
        version=None,
    )
    _, items, _ = run_on_loop(server.backend_answer(client, document, 2, 6, None))
    assert "default_x" in {item.label for item in items}  # it did answer
    assert "defaultdict" not in {item.label for item in items}


def assert_fallback_list(labels):
    """labels is the local list: a special form, a builtin, the buffer's own
    class, and no attribute of self guessed (only the backend has
    __class__)."""
    assert {"ife", "print", "A"} <= labels
    assert "__class__" not in labels


@pytest.mark.parametrize(
    "empty", [[], lsp.CompletionList(is_incomplete=False, items=[])], ids=repr
)
def test_a_backend_with_no_items_gives_the_fallback_list(
    complete, backend_requests, empty
):
    # Defect: an empty backend answer is returned as is: no completion.
    backend_requests.answer["value"] = empty
    labels = set(by_label(complete(CLASS.format("self.|"))[2]))
    assert backend_requests.sent == [lsp.TEXT_DOCUMENT_COMPLETION]
    assert_fallback_list(labels)


def test_backend_off_gives_the_fallback_list(complete, off_backends, monkeypatch):
    # Defect: an off backend still has a view compiled (the buffer's macros
    # run) for a request that cannot be answered.
    views = []
    monkeypatch.setattr(server, "cursor_view", lambda *a: views.append(a))
    labels = set(by_label(complete(CLASS.format("self.|"), off_backends)[2]))
    assert views == []
    assert_fallback_list(labels)


def test_a_buffer_that_does_not_compile_gives_the_fallback_list(
    complete, backend_requests
):
    # An earlier refused form: neither the buffer nor its prefix compiles,
    # so there is no view to ask the backend about.
    labels = set(by_label(complete("(def)\n" + CLASS.format("self.|"))[2]))
    assert backend_requests.sent == []
    assert_fallback_list(labels)
    # The backend does answer the same buffer once it compiles.
    labels = set(by_label(complete(CLASS.format("self.|"))[2]))
    assert "__class__" in labels and backend_requests.sent


def test_an_edit_during_the_request_never_moves_the_answer(complete, backend_requests):
    # Defect: the handler reads the live document after its awaits, so an
    # edit made meanwhile (here: the buffer emptied) gives the old cursor
    # column against the new text: the range starts at column 0 and the
    # typed my- is lost from the filterText.
    backend_requests.answer["during"] = lambda: setattr(
        complete.document, "_source", ""
    )
    line, character, items = complete(CLASS.format("self.my-|"))
    item = by_label(items)["my-x"]
    assert item.filter_text == "my-x"
    assert item.text_edit.insert.start.character == character - 3
