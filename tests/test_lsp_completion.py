"""Completion through the real Python backends (each installed one, pyrefly
and basedpyright; lsp_backends.py).

The tests share the session's backend over one tmp project
(``real_backends``, ``lsp_buffer`` in conftest.py); each test points the
server at it and gets a fresh view cache. ``|`` in a source marks the cursor.
The ``off_complete`` tests need no backend installed (lsp_document).
"""

import asyncio
import importlib
import time
from types import SimpleNamespace

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


def completer(open_buffer, run_on_loop, default_backends):
    def run(source, backends=default_backends, insert_replace=True, before=None):
        ls, uri, position = open_buffer(
            source, backends, client_capabilities=capabilities(insert_replace)
        )
        run.document = open_buffer.document
        if before is not None:
            before(run.document)
        params = lsp.CompletionParams(
            text_document=lsp.TextDocumentIdentifier(uri=uri), position=position
        )
        result = run_on_loop(server.completion(ls, params))
        return position.line, position.character, result.items

    return run


@pytest.fixture
def complete(lsp_buffer, run_on_loop, real_backends):
    """complete(source) -> (line, character, items) at the | of source;
    before(document), if given, runs once the buffer is open."""
    return completer(lsp_buffer, run_on_loop, real_backends)


@pytest.fixture
def off_complete(lsp_document, run_on_loop, off_backends):
    """complete with the backend off, needing none installed."""
    return completer(lsp_document, run_on_loop, off_backends)


def by_label(items):
    return {item.label: item for item in items}


def assert_filters_by(items, typed):
    """Every item keeps the word as typed: its filterText is typed plus the
    rest of its label, so a client's plain prefix match keeps it."""
    for item in items:
        assert item.filter_text == typed + item.label[len(typed) :], item.label


def test_a_member_takes_the_hyphen_spelling(complete):
    # Defects: the backend's Python name is shown (my_x beside or instead of
    # my-x); the item is dropped once the user types my_ or my-.
    items = by_label(complete(CLASS.format("self.|"))[2])
    assert "my-x" in items and "my_x" not in items
    assert "__class__" in items  # inherited from object: only the backend knows
    for typed in ["my_", "my-"]:
        items = complete(CLASS.format(f"self.{typed}|"))[2]
        assert "my-x" in by_label(items), typed
        assert_filters_by(items, typed)


@pytest.mark.parametrize(
    "source, name, label, stem",
    [
        (
            "(class A [Exception]\n  (def f [self]\n    self.{}|))",
            "with_traceback",
            "with-traceback",
            "with",
        ),
        (
            "(from collections [OrderedDict])\n"
            "(class A [OrderedDict]\n  (def f [self]\n    self.{}|))",
            "move_to_end",
            "move-to-end",
            "move",
        ),
    ],
    ids=["builtin base", "imported base"],
)
def test_a_member_never_spelled_in_the_buffer(complete, source, name, label, stem):
    # Defects: a member the buffer never spells is offered in its Python
    # spelling (with_traceback), beside or instead of the hyphen spelling,
    # or accepting it inserts the Python spelling; the item is dropped once
    # a hyphen is typed, or while the user types its _ spelling. with alone
    # is a keyword the view cannot compile as an attribute, so the word
    # typed with no hyphen holds its first _ (with_|).
    for typed in [stem + "_", stem + "-"]:
        answer = complete(source.format(typed))[2]
        items = by_label(answer)
        assert label in items and name not in items
        assert items[label].text_edit.new_text == label
        assert_filters_by(answer, typed)


def test_the_buffers_underscore_spelling_never_wins(complete):
    # Defect: the buffer's own _ spelling of a name (os.get_terminal_size)
    # overrides the hyphen spelling completion offers by default.
    items = by_label(complete("(import os)\n(os.get_terminal_size)\n(os.get|)")[2])
    assert "get-terminal-size" in items and "get_terminal_size" not in items


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
    labels = set(by_label(complete("(import heapq)\n(heapq.n|")[2]))
    assert {"nlargest", "nsmallest"} <= labels
    assert not labels & {"print", "nonlocal"}


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
def test_a_qualified_macro_takes_the_hyphen_spelling(complete, backend_requests):
    # Defect: a macro after its module's name is labelled in a spelling
    # other than the default: its name-key (my_mac) for one defined as
    # my-mac.
    backend_requests.answer["value"] = []
    labels = by_label(complete("(require macmod)\n(macmod.my|)")[2])
    assert "my-mac" in labels and "my_mac" not in labels
    # Defect: a macro defined with an underscore keeps it after mod. (my_umac)
    # instead of the hyphen spelling every name is offered in.
    assert "my-umac" in labels and "my_umac" not in labels


@once
def test_a_required_macro_wins_over_the_backends_name(complete, backend_requests):
    # Defects: the backend's items replace the macros, or a backend item of
    # a macro's name is offered beside it.
    backend_requests.answer["value"] = [
        lsp.CompletionItem(label="twice", kind=lsp.CompletionItemKind.Variable),
        lsp.CompletionItem(label="my_attr", kind=lsp.CompletionItemKind.Variable),
    ]
    items = complete("(require lpy-macs)\n(lpy-macs.|)")[2]
    assert macro_style([i for i in items if i.label != "my-attr"]) == LPY_MACS
    assert "my-attr" in by_label(items)


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


def test_required_macros_follow_their_module_with_no_backend(off_complete):
    # Defects: the mod.name macros a (require mod) registers are offered
    # only when a Python backend answers, though they are lispython's own,
    # so with the backend off (or not installed) the attribute answer is
    # empty; or they are offered in another spelling than the hyphen one.
    assert macro_style(off_complete("(require lpy-macs)\n(lpy-macs.|)")[2]) == LPY_MACS
    labels = by_label(off_complete("(require macmod)\n(macmod.my|)")[2])
    assert set(labels) == {"my-mac", "my-umac"}
    # Defect: the require gate compares the module's raw spelling, so a base
    # written lpy_macs (the same name as lpy-macs) compiles nothing.
    assert macro_style(off_complete("(require lpy-macs)\n(lpy_macs.|)")[2]) == LPY_MACS


def cache_full_view(document):
    """Compile document's full view into the server's cache, as its
    diagnostics would."""
    name, package, import_root = python_view.module_names(document.path)
    server.view_cache.compiled(
        document.uri,
        document.version,
        document.source,
        name=name,
        package=package,
        import_root=import_root,
    )
    assert server.view_cache.cached_view(document.uri, document.version) is not None


def test_required_macros_come_from_the_full_view_with_no_cursor_view(
    off_complete, monkeypatch
):
    # Defect: with no cursor view (its compile gave none or timed out) the
    # attribute answer is empty, though the version's full view, compiled
    # for its diagnostics, holds the module's macros.
    monkeypatch.setattr(server, "cursor_view", lambda *args: None)
    # The cursor before tw, so the full view's text compiles (a bare
    # lpy-macs. would not) and no typed word filters the macros.
    items = off_complete("(require lpy-macs)\n(lpy-macs.|tw)", before=cache_full_view)
    assert macro_style(items[2]) == LPY_MACS


@pytest.mark.parametrize(
    "access", ['"lpy-macs.|"', "; lpy-macs.|", "(print 1) ; lpy-macs.|"]
)
def test_required_macros_are_never_offered_in_a_string_or_comment(
    off_complete, monkeypatch, access
):
    # Defect: mod.| in a string literal or a comment is taken for a macro
    # call, so its macros are offered there: the full view compiles such a
    # buffer fine and holds them (the cursor view gives none there, so it
    # cannot show the defect).
    monkeypatch.setattr(server, "cursor_view", lambda *args: None)
    items = off_complete(f"(require lpy-macs)\n{access}", before=cache_full_view)
    assert items[2] == []


@pytest.mark.parametrize(
    "source",
    [
        CLASS.format("self.|"),
        "(require lpy-macs)\n(import os)\n(os.path.|)",
        "(require lpy-macs [twice])\n(lpy-macs.|)",
        "(require lpy-macs *)\n(lpy-macs.|)",
    ],
    ids=["no require", "another module", "named", "star"],
)
def test_no_backend_compiles_nothing_for_a_base_no_require_qualifies(
    off_complete, monkeypatch, source
):
    # Defect: with no backend, every keystroke after a dot compiles a cursor
    # view (the buffer's macros, a thread, up to REQUEST-TIMEOUT the editor
    # waits on), though only a base a bare (require mod) names can have
    # qualified macros to offer.
    views = []
    monkeypatch.setattr(server, "cursor_view", lambda *args: views.append(args))
    assert off_complete(source)[2] == []
    assert views == []


@pytest.mark.parametrize("access", ["(nope.|)", "(. lpy-macs |)"])
def test_no_backend_offers_nothing_after_another_base(off_complete, access):
    # Defects: with no backend, the qualified macros are offered after any
    # base, or in the (. mod |) form, where mod.name is no macro call; or an
    # unresolved base gets the general list.
    assert off_complete(f"(require lpy-macs)\n{access}")[2] == []


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
    labels = set(by_label(complete(CLASS.format("(|)"))[2]))
    assert backend_requests.sent == [lsp.TEXT_DOCUMENT_COMPLETION]
    assert_fallback_list(labels)


@once
def test_backend_off_gives_the_fallback_list(complete, off_backends, monkeypatch):
    # Defect: an off backend still has a view compiled (the buffer's macros
    # run) for a request that cannot be answered.
    views = []
    monkeypatch.setattr(server, "cursor_view", lambda *a: views.append(a))
    labels = set(by_label(complete(CLASS.format("(|)"), off_backends)[2]))
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
    labels = set(by_label(complete("(def)\n" + CLASS.format("(|)"))[2]))
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
    # typed my_ is lost from the filterText.
    backend_requests.answer["during"] = lambda: setattr(
        complete.document, "_source", ""
    )
    line, character, items = complete(CLASS.format("self.my_|"))
    item = by_label(items)["my-x"]
    assert item.filter_text == "my_x"
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
    # lispython's own forms stay
    assert "ife" in by_label(merged("(def f [i-arg]\n  (return i|))")[2])


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
    items = by_label(merged("(= x 1)\ne|")[2])
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
    labels = set(by_label(merged("(require lispy.macros *)\n(= x 1)\n(print |)")[2]))
    assert {"->", "->>", "--"} <= labels
    assert not {"_to_", "_to_>", "__"} & labels


@pytest.mark.parametrize("selection", ["*", "[my-mac]"])
def test_a_required_macro_takes_the_hyphen_spelling(merged, selection):
    # Defect: a required macro the buffer never spells is labelled by its
    # name-key (my_mac), not the hyphen spelling (my-mac).
    items = by_label(merged(f"(require macmod {selection})\n(print my|)")[2])
    assert "my-mac" in items and "my_mac" not in items


@once
def test_a_macro_defined_with_underscores_is_offered_as_my_mac(merged):
    # Defect: a macro offered in a spelling other than the default (my_mac,
    # as registered) where every other name takes the hyphen spelling.
    items = by_label(merged("(defmacro my_mac [x] x)\n(print my|)")[2])
    assert "my-mac" in items and "my_mac" not in items
    assert items["my-mac"].label_details.description == "macro"


@once
def test_a_view_macro_is_styled_as_a_built_in_macro(merged):
    # Defect: the tag or kind of a view macro drifts from a built-in's.
    items = by_label(merged("(defmacro my-mac [x] x)\n(print |)")[2])

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
    label="my_imported",
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
    "marked", ["(= my-x [])\n(print my-|)", "(= my-x [])\nmy-x.my|"]
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
    assert not {"my-imported", "my_imported"} & labels


@once
def test_the_fallback_list_keeps_the_definitions_after_the_cursor(
    complete, off_backends
):
    # Defect: the form left open at the cursor swallows the rest of the
    # partial parse, so with no backend helper is not offered.
    marked = "(def main []\n  (hel|\n(def helper [] 1)"
    assert "helper" in by_label(complete(marked, off_backends)[2])


@once
def test_the_fallback_list_hyphenates_every_definition(complete, off_backends):
    # Defects: with no backend a buffer's my_val keeps its _ spelling, or is
    # offered twice beside my-val; a defmacro is offered in a spelling other
    # than the default (my_mac).
    source = "(defmacro my_mac [x] x)\n(= my_val 1)\n(= my-val 2)\n(my|)"
    labels = [item.label for item in complete(source, off_backends)[2]]
    assert labels.count("my-val") == 1 and "my_val" not in labels
    assert "my-mac" in labels and "my_mac" not in labels


@once
def test_a_buffer_spelling_outside_the_hyphen_rule_wins(merged, backend_requests):
    # Defects: a name the buffer spells outside the hyphen rule (ok? for
    # ok_p, ->foo for _to_foo) is offered as the hyphen spelling of its
    # Python name (ok-p, _to-foo), so it is dropped while the user types
    # ->f; over a fallback view the buffer's own definition then sits
    # beside it, two items for one name.
    backend_requests.answer["value"] = [
        lsp.CompletionItem(label="ok_p", kind=lsp.CompletionItemKind.Function),
        lsp.CompletionItem(label="_to_foo", kind=lsp.CompletionItemKind.Function),
    ]
    defs = "(def ok? [] 1)\n(def ->foo [] 1)\n"
    for source in [defs + "(print |)", "(def main []\n  (|\n" + defs]:
        labels = [item.label for item in merged(source)[2]]
        assert labels.count("ok?") == 1 and labels.count("->foo") == 1, source
        assert not {"ok-p", "_to-foo", "ok_p", "_to_foo"} & set(labels), source
    for typed, label in [("->f", "->foo"), ("ok?", "ok?")]:
        answer = merged(defs + f"(print {typed}|)")[2]
        assert label in by_label(answer), typed
        assert_filters_by(answer, typed)


@once
def test_a_fallback_definition_and_the_backends_item_share_a_label(
    merged, backend_requests
):
    # Defect: over a fallback view the buffer's definition (ok?) and the
    # backend's item of its Python name (ok_p, which the buffer first
    # writes as ok_p, so ok-p) are labelled by two rules: two items for one
    # name.
    backend_requests.answer["value"] = [
        lsp.CompletionItem(label="ok_p", kind=lsp.CompletionItemKind.Function),
    ]
    source = "(print ok_p)\n(def main []\n  (ok|\n(def ok? [] 1)\n"
    same = [
        item.label for item in merged(source)[2] if server.py_name(item.label) == "ok_p"
    ]
    assert same == ["ok-p"]


@pytest.mark.parametrize(
    "definition, label",
    [("(= ? 1)", "?"), ("(= -?- 1)", "-?-"), ("(= self.ok? 1)", "self.ok?")],
)
def test_the_fallback_list_keeps_a_symbol_only_or_dotted_definition_spelling(
    complete, off_backends, definition, label
):
    # Defect: buffer-spellings keys only dotted parts that spell a name, so
    # these definitions were labelled by the hyphen rule alone (_p, __p_,
    # self.ok-p) and typing their own spelling dropped the item.
    answer = complete(f"{definition}\n(print |)", off_backends)[2]
    labels = [item.label for item in answer]
    assert labels.count(label) == 1
    assert not {"_p", "__p_", "self.ok-p"} & set(labels)


@once
def test_the_fallback_list_hyphenates_a_nested_macro(complete, off_backends):
    # Defect: a defmacro nested in a (do ...), which registers all the same,
    # is offered in a spelling other than the default (in_do).
    labels = by_label(complete("(do (defmacro in_do [x] x))\n(in|)", off_backends)[2])
    assert "in-do" in labels and "in_do" not in labels


@pytest.fixture
def answer_at(lsp_buffer, run_on_loop):
    """answer_at(source, backends=real_backends) -> the completion handler's
    whole CompletionList at the | of source."""

    def run(source, *backends):
        ls, uri, position = lsp_buffer(source, *backends)
        params = lsp.CompletionParams(
            text_document=lsp.TextDocumentIdentifier(uri=uri), position=position
        )
        return run_on_loop(server.completion(ls, params))

    return run


def item(label):
    return lsp.CompletionItem(label=label, kind=lsp.CompletionItemKind.Variable)


KEYSTROKES = ["my-v", "my_v", "my_val-x", "my-val_x", "MY-V"]


@once
@pytest.mark.parametrize("typed", KEYSTROKES)
def test_each_keystroke_is_filtered_on_the_server(
    merged, backend_requests, answer_at, typed
):
    # Defects: the server leaves filtering to the client (names that do not
    # start with the typed word come back, or the answer is complete, so the
    # client filters it on later keystrokes by rules of its own, which drop
    # my-val-x at my_v); a - typed where the label has _ (or the reverse),
    # or another case, drops the item; a filterText other than the typed
    # word plus the rest of the label, which a client's prefix match drops.
    backend_requests.answer["value"] = [item("my_val_x"), item("other_name")]
    answer = answer_at(f"(= my-val-x 1)\n(print {typed}|)")
    labels = set(by_label(answer.items))
    assert answer.is_incomplete is True
    assert "my-val-x" in labels
    assert not {"other-name", "print", "ife"} & labels
    assert_filters_by(answer.items, typed)


@once
@pytest.mark.parametrize("typed", KEYSTROKES)
def test_the_fallback_list_is_filtered_on_the_server(answer_at, off_backends, typed):
    # Defects: as above, for the list given with no backend (lispython's own
    # items, builtins, the buffer's definitions).
    answer = answer_at(f"(= my-val-x 1)\n(= other 1)\n({typed}|)", off_backends)
    labels = set(by_label(answer.items))
    assert answer.is_incomplete is True
    assert "my-val-x" in labels
    assert not {"other", "print", "ife"} & labels
    assert_filters_by(answer.items, typed)


@once
def test_a_leading_separator_is_matched_as_written(answer_at, off_backends):
    # Defect: a - or _ that starts the word matches the other one, so - (an
    # operator macro being typed) offers every _private name, and _ every
    # operator.
    source = "(= _hidden 1)\n({}|)"
    labels = set(by_label(answer_at(source.format("-"), off_backends).items))
    assert "->" in labels and "_hidden" not in labels
    labels = set(by_label(answer_at(source.format("_"), off_backends).items))
    assert "_hidden" in labels and "->" not in labels


@once
def test_keystrokes_in_one_word_ask_the_backend_once(
    merged, backend_requests, answer_at
):
    # Defects: every keystroke compiles a cursor view and asks the backend
    # again (no cache, or one never read); or a cached answer is reused
    # where it no longer holds: after a backspace past the cached word, an
    # edit before the word or after the cursor, or at another word.
    backend_requests.answer["value"] = [item("my_var")]
    backend_requests.sent.clear()

    def requests_after(source):
        assert "my-var" in by_label(answer_at(source).items), source
        return len(backend_requests.sent)

    assert requests_after("(= my-var 1)\n(print my|)") == 1
    assert requests_after("(= my-var 1)\n(print my-v|)") == 1
    assert requests_after("(= my-var 1)\n(print my_va|)") == 1
    assert requests_after("(= my-var 1)\n(print m|)") == 2
    assert requests_after("(= my-var 2)\n(print m|)") == 3
    assert requests_after("(= my-var 2)\n(print m|) 1") == 4
    assert requests_after("(= my-var 2)\n(print m my|) 1") == 5


@once
def test_an_incomplete_list_is_kept(merged, backend_requests, answer_at):
    # Defect: a list the backend marked incomplete is not kept, so with
    # pyrefly and basedpyright, which mark every answer so, each keystroke
    # in a word asks the backend again.
    backend_requests.answer["value"] = lsp.CompletionList(
        is_incomplete=True, items=[item("my_var")]
    )
    backend_requests.sent.clear()
    answer_at("(= my-var 1)\n(print my|)")
    answer = answer_at("(= my-var 1)\n(print my-v|)")
    assert len(backend_requests.sent) == 1
    assert "my-var" in by_label(answer.items)


@once
@pytest.mark.parametrize(
    "first",
    [None, [], lsp.CompletionList(is_incomplete=True, items=[]), "failed"],
    ids=["null", "empty list", "empty CompletionList", "failed"],
)
def test_an_answer_with_no_items_is_not_kept(
    merged, backend_requests, answer_at, first
):
    # Defect: a keystroke the backend did not answer (failed or timed out),
    # answered with null, or with no items (as while it still analyses), is
    # kept as an empty answer, so the rest of the word misses its items.
    if first == "failed":
        backend_requests.answer["replace"] = lambda kwargs: kwargs["failed"]
    else:
        backend_requests.answer["value"] = first
    backend_requests.sent.clear()
    answer_at("(= my-var 1)\n(print my|)")
    backend_requests.answer.clear()
    backend_requests.answer["value"] = [item("my_var")]
    answer = answer_at("(= my-var 1)\n(print my-v|)")
    assert len(backend_requests.sent) == 2
    assert "my-var" in by_label(answer.items)


def test_an_answer_before_the_backend_runs_is_not_kept(complete, idle, monkeypatch):
    # Defect: the answer given while the backend starts (lispython's own
    # list) is kept for the word, so its next keystrokes never ask the
    # backend that has started meanwhile.
    answers = []
    real_answer = server.backend_answer

    async def spy(*args, **kwargs):
        answers.append(args)
        return await real_answer(*args, **kwargs)

    monkeypatch.setattr(server, "backend_answer", spy)
    monkeypatch.setattr(server, "MERGE_TIMEOUT", 30)
    backends, client = idle
    complete("(= my-var 1)\n(print my|)", backends)
    assert answers == []
    assert wait_running(client)
    complete("(= my-var 1)\n(print my-v|)", backends)
    assert len(answers) == 1


@pytest.fixture
def kept_answer(
    merged, backend_requests, answer_at, real_backends, backend_project, run_on_loop
):
    """kept_answer(event) -> the backend requests two keystrokes of one word
    in main.lpy sent, with event (a function of the main.lpy client, run on
    the loop) between them."""
    main = (backend_project / "main.lpy").as_uri()
    client = real_backends.available_client(main)

    def run(event):
        backend_requests.answer["value"] = [item("my_var")]
        backend_requests.sent.clear()
        answer_at("(= my-var 1)\n(print my|)")

        async def on_loop():
            result = event(client)
            if asyncio.iscoroutine(result):
                await result

        run_on_loop(on_loop())
        answer_at("(= my-var 1)\n(print my-v|)")
        return len(backend_requests.sent)

    return run


def push_buffer(uri, path):
    """An event pushing the full view of the .lpy file at path to the
    backend as buffer uri, as a change or an open of that buffer does."""
    source = path.read_text()
    view = python_view.compile_view(source, import_root=str(path.parent))
    return lambda client: client.push_buffer(uri, source, view)


@once
@pytest.mark.parametrize("pushed", ["other", "own"])
def test_a_buffer_pushed_to_the_backend_drops_the_other_kept_answers(
    kept_answer, run_on_loop, real_backends, backend_project, pushed
):
    # Defect: an answer kept for a word in one document outlives an edit or
    # open of another (a module it imports) reaching the backend, so the
    # rest of the word misses a name added there. The document's own text
    # is told by the key, so its own push keeps the entry.
    path = backend_project / ("other.lpy" if pushed == "other" else "main.lpy")
    uri = path.as_uri()
    try:
        sent = kept_answer(push_buffer(uri, path))
    finally:
        if pushed == "other":
            client = real_backends.available_client(uri)

            async def close():
                client.close_document(uri)

            run_on_loop(close())
    assert sent == (2 if pushed == "other" else 1)


@once
def test_a_save_drops_the_kept_answers(kept_answer, backend_project):
    # Defect: an answer kept for a word outlives a save, after which the
    # backend reads the saved file and the shadows it rewrote (a macro
    # module's dependents among them).
    main = (backend_project / "main.lpy").as_uri()
    assert kept_answer(lambda client: client.saved(main)) == 2


@once
def test_a_close_drops_the_kept_answers(kept_answer, backend_project):
    # Defect: an answer kept for a word outlives the close of another
    # document, after which the backend reads that file from disk.
    other = (backend_project / "other.lpy").as_uri()
    ls = SimpleNamespace(text_document_publish_diagnostics=lambda params: None)
    params = lsp.DidCloseTextDocumentParams(
        text_document=lsp.TextDocumentIdentifier(uri=other)
    )
    assert kept_answer(lambda client: server.did_close(ls, params)) == 2


@once
@pytest.mark.parametrize("spared", [False, True], ids=["dropped", "spared"])
def test_an_invalidate_during_the_request_drops_its_answer(
    merged, backend_requests, answer_at, backend_project, spared
):
    # Defects: an answer still in flight when an invalidate drops the
    # document's entry is kept all the same, over the backend's old input;
    # or an invalidate that spares the document (its own buffer pushed, as
    # after every request of its own) refuses its answer, so nothing is
    # ever kept.
    main = (backend_project / "main.lpy").as_uri()
    keep = main if spared else None
    backend_requests.answer["value"] = [item("my_var")]
    backend_requests.answer["during"] = lambda: server.completion_cache.invalidate(
        keep=keep
    )
    backend_requests.sent.clear()
    answer_at("(= my-var 1)\n(print my|)")
    del backend_requests.answer["during"]
    answer_at("(= my-var 1)\n(print my-v|)")
    assert len(backend_requests.sent) == (1 if spared else 2)


@once
def test_a_moved_macro_epoch_drops_the_kept_answer(
    merged, backend_requests, answer_at, monkeypatch
):
    # Defect: an answer kept for a word is reused after a macro module
    # changed (the epoch moved), though its view expanded the old macros.
    epochs = iter([1, 1, 1, 2, 2])
    monkeypatch.setattr(server.macro_modules, "epoch", lambda: next(epochs))
    backend_requests.answer["value"] = [item("my_var")]
    backend_requests.sent.clear()
    answer_at("(= my-var 1)\n(print my|)")  # reads 1 before, 1 after: kept
    answer_at("(= my-var 1)\n(print my-|)")  # reads 1: reused
    assert len(backend_requests.sent) == 1
    answer_at("(= my-var 1)\n(print my-v|)")  # reads 2: asked again
    assert len(backend_requests.sent) == 2


# Many names under one prefix, so a backend that caps its list (and marks
# it incomplete) gives a shorter word fewer of them than a longer one.
MANY_NAMES = "".join(f"(= item-{i:04} {i})\n" for i in range(1500)) + (
    "(= iz-a 1)\n(= iz-b 2)\n"
)


@pytest.mark.parametrize(
    "defs, first, then, expected",
    [
        (
            "(= my-alpha 1)\n(= my-beta 2)\n(= my_gamma 3)\n(= mz 4)\n",
            "m",
            "my-",
            {"my-alpha", "my-beta", "my-gamma"},
        ),
        (MANY_NAMES, "i", "item-1", {"item-1000", "item-1499"}),
    ],
    ids=["shared prefix", "many names"],
)
def test_a_kept_answer_offers_what_a_fresh_request_does(
    merged, backend_requests, answer_at, defs, first, then, expected
):
    # Defect: the backend's answer at a shorter word is not a superset of
    # its answer at the longer one (a capped list, or a filter that is not
    # by prefix), so reusing it offers other names than asking again.
    source = defs + "(print {}|)"
    backend_requests.sent.clear()
    answer_at(source.format(first))
    reused = set(by_label(answer_at(source.format(then)).items))
    assert len(backend_requests.sent) == 1  # the second one was reused
    server.completion_cache.invalidate()
    fresh = set(by_label(answer_at(source.format(then)).items))
    assert len(backend_requests.sent) == 2
    assert expected <= fresh
    assert reused == fresh
