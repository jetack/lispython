"""Completion through the real Python backends (each installed one, pyrefly
and basedpyright; lsp_backends.py).

The tests share the session's backend over one tmp project
(``real_backends``, ``lsp_buffer`` in conftest.py); each test points the
server at it and gets a fresh view cache. ``|`` in a source marks the cursor.
"""

import importlib
import time

import pytest
from lsprotocol import types as lsp

import lispy  # noqa: F401  (registers the .lpy importer)
from tests.lsp_backends import basedpyright_only, once

server = importlib.import_module("lispy.lsp.server")
backend = importlib.import_module("lispy.lsp.backend")
python_view = importlib.import_module("lispy.lsp.python_view")


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
def idle(run_on_loop, make_backends, real_backends, backend_project, tmp_path):
    """A Backends of its own, of the session's backend, not started yet."""
    backends = make_backends(tmp_path / "idle-cache", real_backends.name)
    yield backends, backends.available_client((backend_project / "main.lpy").as_uri())
    run_on_loop(backends.shutdown_all())


def wait_running(client, seconds=30):
    """Poll (bounded) until client's backend runs; True when it does."""
    deadline = time.monotonic() + seconds
    while not client.running and time.monotonic() < deadline:
        time.sleep(0.05)
    return client.running


def test_a_non_attribute_completion_never_waits_for_the_backend(
    complete, idle, monkeypatch
):
    # Defect: a keystroke off an attribute waits for the backend's start
    # (about a second) for kinds that only refine the local list. The wait
    # is a backend_answer call (bounded by MERGE-TIMEOUT, shorter than the
    # start), so timing alone cannot tell it from the start-soon branch.
    answers = []
    real_answer = server.backend_answer

    async def spy(*args, **kwargs):
        answers.append(args)
        return await real_answer(*args, **kwargs)

    monkeypatch.setattr(server, "backend_answer", spy)
    backends, client = idle
    items = complete(IMPORTED_CLASS, backends)[2]
    assert answers == []  # no request, so no wait on the start
    assert not client.running  # answered before the backend was up
    assert wait_running(client)  # and the start was begun, not skipped
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


def test_an_unresolved_base_offers_no_names(complete):
    # Regression TASK-1117: an unresolved base offered the general list
    # (special forms, builtins, the buffer's names), so heapq.nl with no
    # import showed nonlocal and UnboundLocalError.
    assert complete("(def f [] 1)\n(heapq.nl|")[2] == []
    labels = set(by_label(complete("(import heapq)\n(heapq.nl|")[2]))
    assert {"nlargest", "nsmallest"} <= labels
    assert not labels & {"print", "def"}


def macro_style(items):
    return [(item.label, item.kind, item.label_details.description) for item in items]


LPY_MACS = [
    ("thrice", lsp.CompletionItemKind.Function, "macro"),
    ("twice", lsp.CompletionItemKind.Function, "macro"),
]


@once
def test_a_required_modules_macros_follow_its_name(complete, backend_requests):
    # Defect: the mod.name macros a (require mod) registers are offered
    # nowhere after mod., since no backend knows them (mod is never bound),
    # so the attribute answer is empty.
    backend_requests.answer["value"] = []
    assert macro_style(complete("(require lpy-macs)\n(lpy-macs.|)")[2]) == LPY_MACS


@once
def test_a_required_macro_wins_over_the_backends_name(complete, backend_requests):
    # Defects: the backend's items replace the macros, or a backend item of
    # a macro's name is offered beside it.
    backend_requests.answer["value"] = [
        lsp.CompletionItem(label="twice", kind=lsp.CompletionItemKind.Variable),
        lsp.CompletionItem(label="my_attr", kind=lsp.CompletionItemKind.Variable),
    ]
    items = complete("(require lpy-macs)\n(lpy-macs.|)")[2]
    assert macro_style([i for i in items if i.label != "my_attr"]) == LPY_MACS
    assert "my_attr" in by_label(items)


@once
def test_named_required_macros_are_not_qualified(
    complete, backend_requests, backend_project
):
    # Defect: every macro of the view is offered after the module's name,
    # though (require mod [a]) registers a alone, unqualified.
    view = python_view.compile_view(
        "(require lpy-macs [twice])", import_root=str(backend_project)
    )
    assert "twice" in view.macros and "lpy_macs.twice" not in view.macros
    backend_requests.answer["value"] = []
    assert complete("(require lpy-macs [twice])\n(lpy-macs.|)")[2] == []


@once
@pytest.mark.parametrize("access", ["(nope.|)", "(. lpy-macs |)"])
def test_required_macros_follow_only_their_modules_name(
    complete, backend_requests, access
):
    # Defects: the qualified macros are offered after any base, or in the
    # (. mod |) form, where mod.name is no macro call.
    backend_requests.answer["value"] = []
    assert complete(f"(require lpy-macs)\n{access}")[2] == []


@once
@pytest.mark.parametrize(
    "empty", [[], lsp.CompletionList(is_incomplete=False, items=[])], ids=repr
)
def test_a_backend_with_no_items_at_an_attribute_gives_no_items(
    complete, backend_requests, empty
):
    # Defect: an attribute with no members is offered the general list.
    backend_requests.answer["value"] = empty
    items = complete(CLASS.format("self.|"))[2]
    assert backend_requests.sent == [lsp.TEXT_DOCUMENT_COMPLETION]
    assert items == []


@once
@pytest.mark.parametrize(
    "empty", [[], lsp.CompletionList(is_incomplete=False, items=[])], ids=repr
)
def test_a_backend_with_no_items_off_an_attribute_gives_the_fallback_list(
    complete, backend_requests, real_backends, backend_project, empty
):
    # Defect: an empty backend answer off an attribute becomes an empty list
    # instead of lispython's own. The backend must be running, or the
    # completion returns before any request.
    client = real_backends.available_client((backend_project / "main.lpy").as_uri())
    complete(CLASS.format("self.|"))  # starts the backend
    assert wait_running(client)
    backend_requests.sent.clear()
    backend_requests.answer["value"] = empty
    labels = set(by_label(complete(CLASS.format("(pri|)"))[2]))
    assert backend_requests.sent == [lsp.TEXT_DOCUMENT_COMPLETION]
    assert_fallback_list(labels)


@once
def test_backend_off_gives_the_fallback_list(complete, off_backends, monkeypatch):
    # Defect: an off backend still has a view compiled (the buffer's macros
    # run) for a request that cannot be answered.
    views = []
    monkeypatch.setattr(server, "cursor_view", lambda *a: views.append(a))
    labels = set(by_label(complete(CLASS.format("(pri|)"), off_backends)[2]))
    assert views == []
    assert_fallback_list(labels)
    # An attribute gets nothing: only the backend knows self's members.
    assert complete(CLASS.format("self.|"), off_backends)[2] == []
    assert views == []


@once
def test_a_buffer_that_does_not_compile_gives_the_fallback_list(
    complete, backend_requests
):
    # An earlier refused form: neither the buffer nor its prefix compiles,
    # so there is no view to ask the backend about.
    labels = set(by_label(complete("(def)\n" + CLASS.format("(pri|)"))[2]))
    assert backend_requests.sent == []
    assert_fallback_list(labels)
    # An attribute gets nothing: only the backend knows self's members.
    assert complete("(def)\n" + CLASS.format("self.|"))[2] == []
    assert backend_requests.sent == []
    # The backend does answer the same buffer once it compiles.
    labels = set(by_label(complete(CLASS.format("self.|"))[2]))
    assert "__class__" in labels and backend_requests.sent


@once
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


@pytest.fixture
def merged(complete, real_backends, backend_project, monkeypatch):
    """complete, off an attribute, with the session's backend running (only
    a running backend's items are merged) and the time to answer on a
    loaded box."""
    monkeypatch.setattr(server, "MERGE_TIMEOUT", 30)
    client = real_backends.available_client((backend_project / "main.lpy").as_uri())
    if not client.running:
        complete("(= x 1)\n(pri|)")
        assert wait_running(client)
    return complete


def test_merged_completion_offers_the_backends_names_as_the_buffer_spells_them(
    merged,
):
    # Defects: off an attribute the backend's items are not merged (the
    # parameter, which only the backend's scopes know, is missing), or a
    # name keeps the backend's Python spelling (my_arg).
    items = by_label(merged("(def f [my-arg]\n  (return my-a|))")[2])
    assert "my-arg" in items and "my_arg" not in items
    assert items["my-arg"].filter_text == "my-arg"
    assert "ife" in items  # lispython's own forms stay


@pytest.mark.parametrize(
    "source, label, kind",
    [
        ("(def when [] 1)\n(print whe|)", "when", lsp.CompletionItemKind.Function),
        ("(= is_not 1)\n(print is_n|)", "is-not", lsp.CompletionItemKind.Keyword),
    ],
    ids=["same spelling", "same python name"],
)
def test_a_lispython_form_wins_over_the_backends_name(merged, source, label, kind):
    # Defect: the backend's item for a name lispython has a form for (when;
    # is_not, the Python name of is-not) is offered beside lispython's
    # item, or in its place.
    name = server.py_name(label)
    same = [item for item in merged(source)[2] if server.py_name(item.label) == name]
    assert [(item.label, item.kind) for item in same] == [(label, kind)]


@basedpyright_only
def test_python_keywords_are_dropped_but_the_constants_stay(merged):
    # Defects: Python syntax the backend offers (elif) is offered as if
    # lispython had it; or True, a Keyword item too, goes with it.
    items = by_label(merged("(= x 1)\nel|")[2])
    assert items["eval"].text_edit is not None  # the backend did answer
    assert "elif" not in items
    assert "True" in by_label(merged("(= x 1)\n(print Tr|)")[2])


@once
@pytest.mark.parametrize(
    "marked", ["(= my-x 1)\n(print my-|)", "(= my-x [])\nmy-x.co|"]
)
def test_an_incomplete_backend_list_stays_incomplete(
    merged, backend_requests, run_on_loop, lsp_buffer, marked
):
    # Defect: the backend's partial list is answered as complete, so the
    # client filters it on the next keystrokes instead of asking again.
    backend_requests.answer["value"] = lsp.CompletionList(
        is_incomplete=True,
        items=[lsp.CompletionItem(label="my_x", kind=lsp.CompletionItemKind.Variable)],
    )
    ls, uri, position = lsp_buffer(marked)
    params = lsp.CompletionParams(
        text_document=lsp.TextDocumentIdentifier(uri=uri), position=position
    )
    assert run_on_loop(server.completion(ls, params)).is_incomplete is True


def test_a_backend_member_edit_counts_in_utf16_units(complete):
    # Defect: the edit range counts code points: after the emoji (one code
    # point, two UTF-16 units) it starts one column short.
    line, character, items = complete('(= ab [1])\n(print "\U0001f600" ab.so|)')
    edit = by_label(items)["sort"].text_edit
    assert (edit.insert.start.line, edit.insert.start.character) == (
        line,
        character - 2,
    )


def test_the_built_in_macros_are_offered_once_by_their_own_names(merged):
    # Defect: the view's macros are Python names (_to_ for ->, __ for --),
    # matched against lispython's items by spelling, so each operator macro
    # comes back as a second, junk item.
    labels = set(by_label(merged("(require lispy.macros *)\n(= x 1)\n(print x|)")[2]))
    assert {"->", "->>", "--"} <= labels
    assert not {"_to_", "_to_>", "__"} & labels


@once
def test_a_view_macro_is_styled_as_a_built_in_macro(merged):
    # Defect: the tag or kind of a view macro drifts from a built-in's.
    items = by_label(merged("(defmacro my-mac [x] x)\n(print my|)")[2])

    def style(label):
        return (items[label].kind, items[label].label_details.description)

    assert style("my-mac") == style("->")


@once
def test_a_buffer_macro_wins_over_the_backends_name(merged, backend_requests):
    # Defects: a defmacro of the buffer is not offered, is offered by its
    # Python name (my_mac), or beside the backend's item of that name.
    backend_requests.answer["value"] = [
        lsp.CompletionItem(label="my_mac", kind=lsp.CompletionItemKind.Function),
        lsp.CompletionItem(label="my_var", kind=lsp.CompletionItemKind.Variable),
    ]
    items = merged("(defmacro my-mac [x] x)\n(= my-var 1)\n(print my|)")[2]
    same = [item for item in items if server.py_name(item.label) == "my_mac"]
    assert [
        (item.label, item.kind, item.label_details.description) for item in same
    ] == [("my-mac", lsp.CompletionItemKind.Function, "macro")]
    assert "my-var" in by_label(items)  # the backend's other item stays


@once
def test_a_view_macro_carries_its_tag_in_label_details_not_detail(
    merged, backend_requests
):
    # Defects: the view's macro lacks its tag in labelDetails (the client
    # sets no labelDetailsSupport, as eglot), or has it back in detail,
    # which lpy-mode shows as a signature. lispython-items: test_lsp.py.
    backend_requests.answer["value"] = [
        lsp.CompletionItem(label="my_var", kind=lsp.CompletionItemKind.Variable)
    ]
    items = by_label(merged("(defmacro my-mac [x] x)\n(my|)")[2])
    assert items["my-mac"].label_details.description == "macro"
    assert items["my-mac"].detail is None


@once
def test_a_backend_items_label_details_pass_through(merged, backend_requests):
    # Defect: the backend's own labelDetails or detail (signature, type) is
    # dropped or overwritten the way lispython's own items are tagged.
    sent = lsp.CompletionItemLabelDetails(detail="(x)", description="int")
    backend_requests.answer["value"] = [
        lsp.CompletionItem(
            label="my_var",
            kind=lsp.CompletionItemKind.Variable,
            detail="my_var: int",
            label_details=sent,
        )
    ]
    items = by_label(merged("(= my-var 1)\n(print my|)")[2])
    assert items["my-var"].label_details == sent
    assert items["my-var"].detail == "my_var: int"


def test_a_fallback_view_keeps_the_definitions_after_the_cursor(merged):
    # Only the text before the cursor compiles, so the backend never sees
    # helper. Defect: the merged list offers only what the backend saw.
    labels = set(by_label(merged("(def main []\n  (hel|\n(def helper [] 1)")[2]))
    assert "help" in labels  # the backend did answer
    assert "helper" in labels


AUTO_IMPORT = lsp.CompletionItem(
    label="imported_name",
    kind=lsp.CompletionItemKind.Class,
    additional_text_edits=[
        lsp.TextEdit(
            range=lsp.Range(
                start=lsp.Position(line=0, character=0),
                end=lsp.Position(line=0, character=0),
            ),
            new_text="from m import imported_name\n",
        )
    ],
)


@once
@pytest.mark.parametrize(
    "marked", ["(= my-x [])\n(print my-|)", "(= my-x [])\nmy-x.co|"]
)
def test_an_auto_import_item_never_appears(merged, backend_requests, marked):
    # Defect: a symbol the buffer never binds is offered with an import
    # edit positioned in generated code (merged and attribute paths).
    backend_requests.answer["value"] = [
        AUTO_IMPORT,
        lsp.CompletionItem(label="my_x", kind=lsp.CompletionItemKind.Variable),
    ]
    labels = set(by_label(merged(marked)[2]))
    assert "my-x" in labels
    assert "imported_name" not in labels


@once
def test_the_fallback_list_keeps_the_definitions_after_the_cursor(
    complete, off_backends
):
    # Defect: the form left open at the cursor swallows the rest of the
    # partial parse, so with no backend helper is not offered.
    marked = "(def main []\n  (hel|\n(def helper [] 1)"
    assert "helper" in by_label(complete(marked, off_backends)[2])
