"""Go-to-definition and references through the real Python backend
(basedpyright).

The backend answers over the compiled views; every location it gives is
mapped back: one in a .lpy file's view to that .lpy (the open buffer's to
its own uri, another file's through its shadow on disk), one in a real .py
passes (in the editor's units), one in code the compiler made up is dropped.
Every Python name goes to the backend, the buffer's own bindings too, since
it knows scopes; lispython's forms and macros keep the local answer, as do
a buffer with no backend or no Python view, and references over a view of
only the text before the cursor.

The tests share the session's basedpyright over one tmp project
(``real_backends``, ``lsp_buffer``, ``backend_project`` in conftest.py:
main.lpy is the buffer, other.lpy and helpers.py its neighbours). ``|`` in a
source marks the cursor; it is removed first.
"""

import importlib

import pytest
from lsprotocol import types as lsp

import lispy  # noqa: F401  (registers the .lpy importer)

server = importlib.import_module("lispy.lsp.server")
backend = importlib.import_module("lispy.lsp.backend")
python_view = importlib.import_module("lispy.lsp.python_view")

pytestmark = pytest.mark.skipif(
    backend.backend_argv("basedpyright") is None,
    reason="basedpyright is not installed (the lsp extra)",
)


@pytest.fixture
def navigate(lsp_buffer, run_on_loop, real_backends):
    """navigate(handler, marked, backends=real_backends) -> the handler's
    answer at the | of marked (definition or references)."""

    def run(handler, marked, backends=real_backends):
        ls, uri, position = lsp_buffer(marked, backends)
        document = lsp.TextDocumentIdentifier(uri=uri)
        if handler is server.references:
            params = lsp.ReferenceParams(
                text_document=document,
                position=position,
                context=lsp.ReferenceContext(include_declaration=True),
            )
        else:
            params = lsp.DefinitionParams(text_document=document, position=position)
        return run_on_loop(handler(ls, params))

    return run


def spans(locations):
    """(file name, start line, start col, end line, end col) of each."""
    return [
        (
            loc.uri.rsplit("/", 1)[-1],
            loc.range.start.line,
            loc.range.start.character,
            loc.range.end.line,
            loc.range.end.character,
        )
        for loc in locations or []
    ]


OTHER = "(from other [other-fn])\n"


def test_definition_in_another_lpy_module(navigate, backend_project):
    # Defects: the shadow's other.py is passed on (a file the user never
    # wrote, in the cache), or its range stays in the generated text
    # (def other_fn at columns 4-12 of line 0, not other-fn at 5-13).
    [location] = navigate(server.definition, OTHER + "(other-f|n 2)")
    assert location.uri == (backend_project / "other.lpy").as_uri()
    assert spans([location]) == [("other.lpy", 0, 5, 0, 13)]


def test_definition_in_a_user_py_module(navigate, backend_project):
    # Defect: a real .py's location is dropped or remapped; it is the file
    # itself, its range the backend's.
    [location] = navigate(
        server.definition, "(from helpers [my-helper])\n(my-hel|per 1 2)"
    )
    assert location.uri == (backend_project / "helpers.py").as_uri()
    assert spans([location]) == [("helpers.py", 0, 4, 0, 13)]


def test_references_across_lpy_files(navigate, backend_project):
    # Defects: the other file's references stay in its shadow, the buffer's
    # come back under the shadow's uri, or columns are generated ones.
    found = set(spans(navigate(server.references, OTHER + "(other-f|n 2)")))
    assert {
        ("other.lpy", 0, 5, 0, 13),
        ("other.lpy", 2, 1, 2, 9),
        ("main.lpy", 0, 13, 0, 21),
        ("main.lpy", 1, 1, 1, 9),
    } <= found
    assert all(name in ("other.lpy", "main.lpy") for name, *_ in found)


MACRO_BUFFER = "(defmacro swap [a b] `(= [~a ~b] [~b ~a]))\n(= my-x 1)\n(print my-|x)\n"


def generated_range(text, word, nth=0):
    """The 0-based LSP range of the nth occurrence of word in text."""
    index = -1
    for _ in range(nth + 1):
        index = text.index(word, index + 1)
    line = text.count("\n", 0, index)
    col = index - (text.rfind("\n", 0, index) + 1)
    return lsp.Range(
        start=lsp.Position(line=line, character=col),
        end=lsp.Position(line=line, character=col + len(word)),
    )


def test_a_location_in_generated_code_is_dropped(
    navigate, backend_requests, real_backends, backend_project
):
    # Defect: a location on code the compiler made up (the defmacro's
    # temporary def) reaches the editor at whatever .lpy text its line
    # number lands on; the location on my_x still maps to my-x.
    uri = (backend_project / "main.lpy").as_uri()
    shadow = real_backends.available_client(uri).backend_uri(uri)

    def answer(request):
        # Ranges over the text actually synced for the request.
        text = request["text"]
        return [
            lsp.Location(
                uri=shadow, range=generated_range(text, "___defmacro_temp___swap")
            ),
            lsp.Location(uri=shadow, range=generated_range(text, "my_x", 1)),
        ]

    backend_requests.answer["replace"] = answer
    found = navigate(server.references, MACRO_BUFFER)
    assert backend_requests.sent == [lsp.TEXT_DOCUMENT_REFERENCES]
    assert spans(found) == [("main.lpy", 2, 7, 2, 11)]


def test_the_buffer_maps_by_the_view_it_was_asked_over(
    navigate, backend_requests, real_backends, backend_project
):
    # Defect: the buffer's locations map through whatever origin the client
    # holds when the answer arrives; a sync while the request was in flight
    # (here: the same text from a source two lines lower) moves them.
    uri = (backend_project / "main.lpy").as_uri()
    client = real_backends.available_client(uri)
    shadow = client.backend_uri(uri)
    moved = "\n\n(= my-x 1)\n(print my-x)"
    moved_view = python_view.compile_view(moved)

    def answer(request):
        assert moved_view.text == request["text"]
        client._sync(uri, request["text"], origin=(moved, moved_view))
        return [
            lsp.Location(uri=shadow, range=generated_range(request["text"], "my_x", 1))
        ]

    backend_requests.answer["replace"] = answer
    found = navigate(server.references, "(= my-x 1)\n(print my-|x)")
    assert spans(found) == [("main.lpy", 1, 7, 1, 11)]


def test_backend_off_gives_the_local_answers(navigate, off_backends):
    # Defect: no backend leaves definition or references empty instead of
    # the local answers: the name the import binds, the buffer's own
    # occurrences.
    found = navigate(server.definition, OTHER + "(other-f|n 2)", off_backends)
    assert spans(found) == [("main.lpy", 0, 13, 0, 21)]
    found = navigate(server.references, OTHER + "(other-f|n 2)", off_backends)
    assert spans(found) == [("main.lpy", 0, 13, 0, 21), ("main.lpy", 1, 1, 1, 9)]


def test_a_buffer_that_does_not_compile_gives_the_local_answers(
    navigate, backend_requests
):
    # An earlier refused form: there is no Python view, so no request, and
    # the local answers stand.
    marked = "(def)\n" + OTHER + "(other-f|n 2)"
    assert spans(navigate(server.definition, marked)) == [("main.lpy", 1, 13, 1, 21)]
    assert spans(navigate(server.references, marked)) == [
        ("main.lpy", 1, 13, 1, 21),
        ("main.lpy", 2, 1, 2, 9),
    ]
    assert backend_requests.sent == []


def test_definition_follows_python_scopes(navigate):
    # Defect: a function's local of the same name answers for the global
    # (a scope-blind lookup of the buffer's bindings): the outer sep is the
    # one (from os [sep]) binds, which the backend follows into os.
    found = navigate(
        server.definition, '(from os [sep])\n(def f [] (= sep "/"))\n(print se|p)'
    )
    assert found
    assert not [span for span in spans(found) if span[:2] == ("main.lpy", 1)]
    assert any(loc.uri.endswith(".pyi") for loc in found)


@pytest.mark.parametrize("handler", [server.definition, server.references])
@pytest.mark.parametrize(
    "marked",
    [
        "(require lispy.macros *)\n(wh|en 1 2)",
        "(defmacro twice [x] (return x))\n(twi|ce 1)",
    ],
    ids=["required macro", "defmacro"],
)
def test_a_macro_never_reaches_the_backend(navigate, backend_requests, handler, marked):
    # Defect: a macro head is sent to the backend, which sees the code it
    # expanded to (or the sentinel's call) and answers for some Python name.
    navigate(handler, marked)
    assert backend_requests.sent == []


def test_references_never_come_from_a_view_of_the_prefix(navigate, backend_requests):
    # The buffer ends mid-form, so only the text before the cursor compiles.
    # Defect: the backend's references over that prefix (lines 0 and 1)
    # replace the local answer, and the use on line 2 is lost.
    found = navigate(
        server.references, "(= my-x 1)\n(print my-|x)\n(print my-x)\n(print (+ 1"
    )
    assert [span[1] for span in spans(found)] == [0, 1, 2]
    assert backend_requests.sent == []


SHADOWED = "(= my-x 1)\n(def f []\n  (= my-x 2))\n(print my-|x)"


@pytest.mark.parametrize("handler", [server.definition, server.references])
def test_an_empty_backend_answer_is_final(navigate, backend_requests, handler):
    # Defect: the backend found nothing, and the local lookup, which
    # matches by name alone, answers anyway with every my-x in the buffer,
    # the function's local among them.
    backend_requests.answer["value"] = []
    assert navigate(handler, SHADOWED) == []
    assert backend_requests.sent == [
        lsp.TEXT_DOCUMENT_DEFINITION
        if handler is server.definition
        else lsp.TEXT_DOCUMENT_REFERENCES
    ]


def test_a_macro_gets_the_local_answers(navigate, backend_requests):
    # Defect: the backend, never asked about a macro, counts as having
    # answered nothing, and the macro's own definition and uses are lost.
    marked = "(defmacro twice [x] (return x))\n(twi|ce 1)"
    assert spans(navigate(server.definition, marked)) == [("main.lpy", 0, 10, 0, 15)]
    assert spans(navigate(server.references, marked)) == [
        ("main.lpy", 0, 10, 0, 15),
        ("main.lpy", 1, 1, 1, 6),
    ]
    assert backend_requests.sent == []


def test_an_answer_that_maps_nowhere_gives_the_local_answer(
    navigate, backend_requests, real_backends, backend_project
):
    # Defect: the backend found the name, but only in code the compiler
    # made up, and the empty mapped list counts as its final answer: the
    # local definition is lost.
    uri = (backend_project / "main.lpy").as_uri()
    shadow = real_backends.available_client(uri).backend_uri(uri)
    backend_requests.answer["replace"] = lambda request: [
        lsp.Location(
            uri=shadow,
            range=generated_range(request["text"], "___defmacro_temp___swap"),
        )
    ]
    found = navigate(server.definition, MACRO_BUFFER)
    assert backend_requests.sent == [
        lsp.TEXT_DOCUMENT_DEFINITION,
        lsp.TEXT_DOCUMENT_TYPE_DEFINITION,
    ]
    assert spans(found) == [("main.lpy", 1, 3, 1, 7)]


def test_an_unmapped_answer_never_jumps_to_another_file(
    navigate, backend_requests, real_backends, backend_project, monkeypatch
):
    # Defect: the backend located the name only in generated code, and the
    # local fallback's workspace step jumps to an unrelated definition of
    # that name in another file; only the buffer's own definitions may answer.
    uri = (backend_project / "main.lpy").as_uri()
    shadow = real_backends.available_client(uri).backend_uri(uri)
    other = lsp.Location(
        uri=(backend_project / "other.lpy").as_uri(),
        range=lsp.Range(
            start=lsp.Position(line=0, character=5),
            end=lsp.Position(line=0, character=13),
        ),
    )
    monkeypatch.setattr(server, "WORKSPACE_INDEX", {"made-x": [other]})
    backend_requests.answer["replace"] = lambda request: [
        lsp.Location(
            uri=shadow,
            range=generated_range(request["text"], "___defmacro_temp___swap"),
        )
    ]
    marked = "(defmacro swap [a b] `(= [~a ~b] [~b ~a]))\n(print made-|x)\n"
    assert navigate(server.definition, marked) is None
    assert backend_requests.sent == [
        lsp.TEXT_DOCUMENT_DEFINITION,
        lsp.TEXT_DOCUMENT_TYPE_DEFINITION,
    ]


def test_a_name_that_mangles_like_a_macro_reaches_the_backend(
    navigate, backend_requests
):
    # Defect: macro heads are compared by Python name, and -> mangles to
    # _to_, so a variable _to_ is taken for the macro and never asked about.
    navigate(server.definition, "(require lispy.macros *)\n(= _to_ 1)\n(print _to|_)")
    assert lsp.TEXT_DOCUMENT_DEFINITION in backend_requests.sent


@pytest.mark.parametrize(
    "marked",
    ["(print (+ nosu|ch 1))", "(import os)\n(print os.nosu|ch)"],
    ids=["undefined name", "unknown attribute"],
)
def test_a_name_with_no_definition_has_none(navigate, marked):
    # Defect: basedpyright answers definition on a name it cannot resolve
    # for the enclosing call, so go-to-definition jumps to print in
    # builtins.pyi, and that answer is final.
    assert navigate(server.definition, marked) == []


@pytest.mark.parametrize(
    "marked",
    ["(from data|classes [dataclass])", "(import os.pa|th)"],
    ids=["from-import module", "dotted import"],
)
def test_a_module_name_in_an_import_keeps_its_definition(navigate, marked):
    # Defect: the check that drops an answer for an enclosing node asks for
    # the word's type definition, which basedpyright never gives a module
    # name in an import, so the module's own definition is dropped.
    found = navigate(server.definition, marked)
    assert found
    assert all(loc.uri.endswith((".pyi", ".py")) for loc in found)


def test_a_defined_name_in_a_call_keeps_its_definition(navigate):
    # Defect: the check that drops an answer for an enclosing node drops
    # the answer of a name that has one, inside the same call.
    found = navigate(server.definition, "(= my-x 1)\n(print my-|x)")
    assert [span[:2] for span in spans(found)] == [("main.lpy", 0)]


def test_a_failed_check_gives_the_local_answer(
    navigate, real_backends, backend_project, monkeypatch
):
    # Defect: the type-definition check that drops an answer for an
    # enclosing node fails (timeout, dead backend), and the failure is
    # taken for an empty check: the definition becomes [] instead of the
    # local lookup's answer.
    client = real_backends.available_client((backend_project / "main.lpy").as_uri())
    real_request = client.request

    async def request(method, params, **kwargs):
        if method == lsp.TEXT_DOCUMENT_TYPE_DEFINITION:
            return kwargs.get("failed")
        return await real_request(method, params, **kwargs)

    monkeypatch.setattr(client, "request", request)
    found = navigate(
        server.definition, "(def f [] (= nosuch 1))\n(print (+ nosu|ch 1))"
    )
    assert spans(found) == [("main.lpy", 0, 13, 0, 19)]
