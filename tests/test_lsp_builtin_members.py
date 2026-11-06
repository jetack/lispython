"""Hover and signature help through the real Python backend (basedpyright).

Acceptance tests for TASK-1021: ``x.sort``, ``(. x sort)`` and ``(.sort x)``
on a list show ``list.sort``'s parameters, as pyright and jedi do for Python.
Expected parameters are read from CPython (``inspect.signature``), never from
the server, so the tests hold for any backend that reports the real
signature. The rest pin the route: lispython's own forms and macros answer
locally, the backend's names take the buffer's spelling, and with no backend
answer the local one stands.

The tests share the session's basedpyright over one tmp project
(``real_backends``, ``lsp_buffer`` in conftest.py). ``|`` in a source marks
the cursor; it is removed first.
"""

import importlib
import inspect

import pytest
from lsprotocol import types as lsp

import lispy  # noqa: F401  (registers the .lpy importer)

# lispy.lsp re-exports the server *object* as ``server``; take the module.
server = importlib.import_module("lispy.lsp.server")
backend = importlib.import_module("lispy.lsp.backend")

pytestmark = pytest.mark.skipif(
    backend.backend_argv("basedpyright") is None,
    reason="basedpyright is not installed (the lsp extra)",
)


@pytest.fixture
def hover(lsp_buffer, run_on_loop, real_backends):
    """hover(marked, backends=real_backends) -> the server's Hover or None."""

    def run(marked, backends=real_backends):
        ls, uri, position = lsp_buffer(marked, backends)
        params = lsp.HoverParams(
            text_document=lsp.TextDocumentIdentifier(uri=uri), position=position
        )
        return run_on_loop(server.hover(ls, params))

    return run


def signature_capabilities(label_offsets):
    """An editor's capabilities, advertising labelOffsetSupport or not."""
    return lsp.ClientCapabilities(
        text_document=lsp.TextDocumentClientCapabilities(
            signature_help=lsp.SignatureHelpClientCapabilities(
                signature_information=lsp.ClientSignatureInformationOptions(
                    parameter_information=lsp.ClientSignatureParameterInformationOptions(
                        label_offset_support=label_offsets
                    )
                )
            )
        )
    )


@pytest.fixture
def signature_help(lsp_buffer, run_on_loop, real_backends):
    """signature_help(marked, backends=real_backends, label_offsets=True) ->
    SignatureHelp or None, for an editor advertising labelOffsetSupport as
    label_offsets says."""

    def run(marked, backends=real_backends, label_offsets=True):
        ls, uri, position = lsp_buffer(
            marked,
            backends,
            client_capabilities=signature_capabilities(label_offsets),
        )
        params = lsp.SignatureHelpParams(
            text_document=lsp.TextDocumentIdentifier(uri=uri), position=position
        )
        return run_on_loop(server.signature_help(ls, params))

    return run


def hover_text(result):
    return None if result is None else result.contents.value


def signature_labels(result):
    """Every signature offered: overload order differs between stub versions."""
    return [] if result is None else [sig.label for sig in result.signatures]


def bound_parameters(cls, member):
    """The parameter names a call through an instance takes (self dropped)."""
    parameters = list(inspect.signature(getattr(cls, member)).parameters)
    return parameters[1:]


def shows_any(labels, cls, member):
    errors = []
    for label in labels:
        try:
            assert_shows(label, cls, member)
            return
        except (AssertionError, ValueError) as err:
            errors.append(f"{label!r}: {err!r}")
    raise AssertionError(f"no signature shows {cls.__name__}.{member}: {errors}")


def assert_shows(text, cls, member):
    assert text is not None
    for name in bound_parameters(cls, member):
        assert name in text
    # The sorted builtin also takes key and reverse (after an iterable): a
    # server resolving .sort to it must not pass.
    assert "iterable" not in text
    assert "sorted" not in text
    # list.sort's parameters are keyword-only; a positional rendering would
    # be the wrong signature.
    if (cls, member) == (list, "sort"):
        assert text.index("*") < text.index("key")


# Each receiver form, with the list type coming from each source a type
# checker reads: a literal, an annotation, an assignment, a constructor call.
HOVERS = [
    "(= x [1])\n(print x.so|rt)",
    "(= x [1])\n(print (. x so|rt))",
    "(= x [1])\n(print (.so|rt x))",
    "(print (.so|rt [3 1]))",
    "(def f [x ^list] (.so|rt x))",
    "(def f [x ^(sub list int)] (.so|rt x))",
    "(= x (list))\n(print x.so|rt)",
]

SIGNATURES = [
    "(= x [1])\n(x.sort |)",
    "(= x [1])\n((. x sort) |)",
    "(= x [1])\n(.sort x |)",
    "(.sort [3 1] |)",
    "(def f [x ^list] (.sort x |))",
    "(def f [x ^(sub list int)] (.sort x |))",
    "(= x (list))\n(.sort x |)",
    # Mid-edit: the call is still open.
    "(= x [1])\n(.sort x :key |",
    # The innermost call is the method, not the enclosing print.
    "(= x [1])\n(print ((. x sort) |))",
]


@pytest.mark.parametrize("marked", HOVERS)
def test_hover_on_a_list_method_shows_its_parameters(hover, marked):
    text = hover_text(hover(marked))
    assert_shows(text, list, "sort")
    assert "sort" in text


@pytest.mark.parametrize("marked", SIGNATURES)
def test_signature_help_in_a_list_method_call_shows_its_parameters(
    signature_help, marked
):
    shows_any(signature_labels(signature_help(marked)), list, "sort")


def test_another_type_and_method(signature_help):
    marked = '(= d {"a" 1})\n(.get d |)'
    shows_any(signature_labels(signature_help(marked)), dict, "get")


# ok? is ok_p in Python: one character longer, so every offset after it in a
# relabelled signature moves.
USER_FUNCTION = "(def my-fn [ok? my-arg] my-arg)\n"


def test_hover_shows_the_buffer_spelling(hover):
    # Defect: the backend's Python names are shown (my_fn, my_arg, ok_p),
    # names the user never wrote.
    text = hover_text(hover(USER_FUNCTION + "(my-f|n 1 2)"))
    assert "my-fn" in text and "my-arg" in text and "ok?" in text
    assert "my_fn" not in text and "my_arg" not in text and "ok_p" not in text


@pytest.mark.parametrize("label_offsets", [True, False], ids=["offsets", "strings"])
def test_signature_help_shows_the_buffer_spelling(signature_help, label_offsets):
    # Defects: the label shows my_arg/ok_p; an offset parameter label keeps
    # the backend's offsets, which point into the Python label and so off
    # by one into the relabelled one (my-arg: Any read as "y-arg: Any,");
    # or an editor that never advertised labelOffsetSupport is sent
    # offsets it cannot read, and highlights no parameter.
    result = signature_help(USER_FUNCTION + "(my-fn 1 |)", label_offsets=label_offsets)
    [signature] = result.signatures
    assert "my-arg" in signature.label and "ok?" in signature.label
    assert "my_arg" not in signature.label and "ok_p" not in signature.label
    texts = []
    for parameter in signature.parameters:
        label = parameter.label
        assert isinstance(label, str) is not label_offsets
        if not isinstance(label, str):
            label = signature.label[label[0] : label[1]]
        texts.append(label.split(":")[0].strip())
    assert texts == ["ok?", "my-arg"]
    assert signature.active_parameter == 1 or result.active_parameter == 1


def test_markdown_prose_is_never_relabelled():
    # Defect: a docstring's ordinary word that matches a buffer name's
    # Python spelling is rewritten (prose is the library's, not the buffer's).
    spellings = {"my_x": "my-x"}
    text = "```python\ndef f(my_x) -> my_x\n```\n---\nSets my_x."
    assert server.markdown_to_buffer(text, spellings) == (
        "```python\ndef f(my-x) -> my-x\n```\n---\nSets my_x."
    )


def test_the_hover_range_is_the_lpy_word(hover):
    # Defect: the backend's range is passed on: it lies in the generated
    # Python (sort is at columns 8-12 of `print(x.sort)`), so the client
    # highlights the wrong text.
    result = hover("(= x [1])\n(print (. x so|rt))")
    start, end = result.range.start, result.range.end
    assert (start.line, start.character, end.line, end.character) == (1, 12, 1, 16)
    result = hover("(= ab [1])\n(print ab.so|rt)")
    start, end = result.range.start, result.range.end
    assert (start.line, start.character, end.line, end.character) == (1, 10, 1, 14)


PRINT_HOVER = server.BUILTIN_INFO["print"]["signature"]


def test_backend_off_gives_the_local_answer(hover, signature_help, off_backends):
    # Defect: no backend answer leaves hover or signature help empty
    # instead of lispython's builtin documentation.
    assert PRINT_HOVER in hover_text(hover("(pri|nt 1)", off_backends))
    assert signature_labels(signature_help("(print |)", off_backends)) == [PRINT_HOVER]


def test_a_buffer_that_does_not_compile_gives_the_local_answer(
    hover, signature_help, backend_requests
):
    # An earlier refused form: neither the buffer nor its prefix compiles,
    # so there is no view and no request; the builtin answer stands.
    assert PRINT_HOVER in hover_text(hover("(def)\n(pri|nt 1)"))
    assert signature_labels(signature_help("(def)\n(print |)")) == [PRINT_HOVER]
    assert backend_requests.sent == []


@pytest.mark.parametrize(
    "marked",
    [
        "(require lispy.macros *)\n(wh|en 1 2)",
        "(require lispy.macros *)\n(co|nd 1 2)",
        "(defmacro twice [x] (return x))\n(twi|ce 1)",
        "(i|fe 1 2 3)",
    ],
    ids=["when", "cond", "defmacro", "special form"],
)
def test_lispython_forms_never_reach_the_backend_on_hover(
    hover, backend_requests, marked
):
    # Defect: a macro head is sent to the backend, which sees the sentinel's
    # call in its place (when(1, 2)) and documents an unknown name or some
    # Python object of that name.
    hover(marked)
    assert backend_requests.sent == []


@pytest.mark.parametrize(
    "marked",
    [
        "(require lispy.macros *)\n(when x |)",
        "(require lispy.macros *)\n(print (when x |))",
        "(require lispy.macros *)\n(print (-> x |))",
        "(require lispy.macros *)\n(cond x |)",
        "(defmacro twice [x] (return x))\n(twice |)",
        "(print (ife x |))",
    ],
    ids=[
        "when",
        "when in a call",
        "macro in a call",
        "cond",
        "defmacro",
        "special form",
    ],
)
def test_lispython_forms_never_reach_the_backend_on_signature_help(
    signature_help, backend_requests, marked
):
    # Defect: the call around the form (print) or the form's sentinel call
    # is sent to the backend, which answers for a call the user is not in.
    signature_help(marked)
    assert backend_requests.sent == []


def test_a_call_inside_a_form_reaches_the_backend(signature_help, backend_requests):
    # The gate is the innermost open form's head only: a call typed inside
    # a macro's body is a call (the counterpart of the cases above).
    signature_help("(require lispy.macros *)\n(when x (.sort [1] |))")
    assert backend_requests.sent == [lsp.TEXT_DOCUMENT_SIGNATURE_HELP]
