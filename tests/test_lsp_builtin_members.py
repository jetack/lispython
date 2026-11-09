"""Hover and signature help through the real Python backends (each
installed one, pyrefly and basedpyright; lsp_backends.py).

Acceptance tests for TASK-1021: ``x.sort``, ``(. x sort)`` and ``(.sort x)``
on a list show ``list.sort``'s parameters, as pyright and jedi do for Python.
Expected parameters are read from CPython (``inspect.signature``), never from
the server, so the tests hold for any backend that reports the real
signature. The rest pin the route: lispython's own forms and macros answer
locally, the backend's names take completion's spelling, and with no backend
answer the local one stands.

The tests share the session's backend over one tmp project
(``real_backends``, ``lsp_buffer`` in conftest.py). ``|`` in a source marks
the cursor; it is removed first.
"""

import ast
import importlib
import inspect
import re
import shutil

import pytest
from lsprotocol import types as lsp

import lispy  # noqa: F401  (registers the .lpy importer)
from lispy.core.compiler.literal import hyphen_spelling
from tests.lsp_backends import once

# lispy.lsp re-exports the server *object* as ``server``; take the module.
server = importlib.import_module("lispy.lsp.server")
backend = importlib.import_module("lispy.lsp.backend")


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


def assert_parameters(text, names):
    """text shows each Python parameter name in names as completion spells
    it (follow_symlinks as follow-symlinks)."""
    assert text is not None
    for name in names:
        assert re.search(rf"\b{hyphen_spelling(name)}\b", text), (name, text)


# shutil.copyfile: its runtime parameters are the stub's too (os.path.join's
# are not: a, *p at runtime, a, *paths in typeshed).
STDLIB = "(import shutil as sh)\n"
COPYFILE = list(inspect.signature(shutil.copyfile).parameters)


def test_hover_on_a_stdlib_function_through_a_module_alias(hover):
    # Defect: a module attribute reached through an alias gets the local
    # answer (none: copyfile is no builtin), or the alias itself.
    text = hover_text(hover(STDLIB + '(sh.copy|file "a" "b")'))
    assert_parameters(text, COPYFILE)
    assert "copyfile" in text


def test_signature_help_in_a_stdlib_call(signature_help, real_backends):
    result = signature_help(STDLIB + '(sh.copyfile "a" |)')
    assert_parameters(signature_labels(result)[0], COPYFILE)
    # pyrefly (1.4.1) gives a generic function's signature no parameters.
    if real_backends.name == "basedpyright":
        active = result.signatures[0].active_parameter
        assert active == 1 or result.active_parameter == 1


def parameter_texts(signature):
    """Each parameter's label text, whether sent as a string or as offsets
    into the signature's label."""
    return [
        label if isinstance(label, str) else signature.label[label[0] : label[1]]
        for label in (parameter.label for parameter in signature.parameters)
    ]


def helper_parameters(backend_project):
    """my_helper's parameter names, read from the project's helpers.py."""
    tree = ast.parse((backend_project / "helpers.py").read_text())
    [function] = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    return function.name, [arg.arg for arg in function.args.args]


HELPER = "(from helpers [my-helper])\n"


def test_hover_on_a_user_py_module_function(hover, backend_project):
    # Defects: a name from the project's own .py module is not resolved
    # (the view root lacks it), or shows its Python spelling my_helper.
    name, parameters = helper_parameters(backend_project)
    text = hover_text(hover(HELPER + "(my-hel|per 1 2)"))
    assert_parameters(text, parameters)
    assert name.replace("_", "-") in text and name not in text


def test_signature_help_in_a_user_py_module_call(signature_help, backend_project):
    _, parameters = helper_parameters(backend_project)
    result = signature_help(HELPER + "(my-helper 1 |)")
    [signature] = result.signatures
    assert_parameters(signature.label, parameters)
    labels = parameter_texts(signature)
    assert [label.split(":")[0].strip() for label in labels] == parameters
    assert signature.active_parameter == 1 or result.active_parameter == 1


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
    # offsets it cannot read, and highlights no parameter.  basedpyright
    # sends offsets (asked for them), pyrefly strings.
    result = signature_help(USER_FUNCTION + "(my-fn 1 |)", label_offsets=label_offsets)
    [signature] = result.signatures
    assert "my-arg" in signature.label and "ok?" in signature.label
    assert "my_arg" not in signature.label and "ok_p" not in signature.label
    if not label_offsets:
        assert all(isinstance(p.label, str) for p in signature.parameters)
    texts = [label.split(":")[0].strip() for label in parameter_texts(signature)]
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


def test_every_name_takes_the_completion_spelling_and_strings_stay():
    # Defects: a name the buffer never wrote keeps its Python spelling
    # (my_x, _T_co), unlike the completion item that offered it; a string
    # literal's text is relabelled (Literal["a_b"] read as "a-b", a value the
    # library does not take); or an offset past a buffer spelling of another
    # length (ok? for ok_p) lands one character off.
    text = "def f(my_x: Literal[\"a_b\"], ok_p: _T_co = 'c\\'d_e') -> my_x"
    new, mapped = server.python_to_buffer(text, {"ok_p": "ok?"})
    assert new == "def f(my-x: Literal[\"a_b\"], ok?: _T-co = 'c\\'d_e') -> my-x"
    spelled = {
        '"a_b"': '"a_b"',
        "ok_p": "ok?",
        "_T_co": "_T-co",
        "'c\\'d_e'": "'c\\'d_e'",
        "-> my_x": "-> my-x",
    }
    for old, expected in spelled.items():
        start = text.index(old)
        assert new[mapped(start) : mapped(start + len(old))] == expected


JSON_DUMPS = "(import json)\n"


def test_hover_spells_a_library_name_as_completion_does(hover):
    # Defect: a parameter the buffer never wrote shows its Python spelling
    # (ensure_ascii) though completion offers it as ensure-ascii, so the
    # hover shows a spelling the user does not type; or the docstring after
    # the code block is relabelled, though it is the library's prose.
    text = hover_text(hover(JSON_DUMPS + "(json.dum|ps 1)"))
    code, _, prose = text.partition("\n```")
    assert "ensure-ascii" in code and "check-circular" in code
    assert "ensure_ascii" not in code and "check_circular" not in code
    assert "ensure-ascii" not in prose
    assert re.search(r"ensure\\?_ascii", prose), prose


@pytest.mark.parametrize("label_offsets", [True, False], ids=["offsets", "strings"])
def test_signature_help_spells_a_library_name_as_completion_does(
    signature_help, label_offsets
):
    # Defect: the label and parameters show ensure_ascii, a spelling the
    # user never types, though completion offers ensure-ascii.  An exact
    # parameter text also fails if an offset label no longer covers exactly
    # its parameter in the relabelled label.
    result = signature_help(
        JSON_DUMPS + "(json.dumps 1 |)", label_offsets=label_offsets
    )
    texts = [t for s in result.signatures for t in parameter_texts(s)]
    assert "ensure-ascii: bool = True" in texts
    assert "check-circular: bool = True" in texts
    names = [t.split(":")[0].split("=")[0].strip().lstrip("*") for t in texts]
    assert "ensure-ascii" in names and "sort-keys" in names
    assert not [name for name in names if "_" in name.strip("_")], names
    assert not [s.label for s in result.signatures if "ensure_ascii" in s.label]


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


def assert_local_print(hover_result, signature_result):
    """The answers are lispython's own builtin documentation of print (its
    signature as inspect reads it differs between Python versions)."""
    text = hover_text(hover_result)
    assert text.startswith("```python\nprint") and "*(Python builtin)*" in text
    [label] = signature_labels(signature_result)
    assert label.startswith("print") and label in text


@once
def test_backend_off_gives_the_local_answer(hover, signature_help, off_backends):
    # Defect: no backend answer leaves hover or signature help empty
    # instead of lispython's builtin documentation.
    assert_local_print(
        hover("(pri|nt 1)", off_backends), signature_help("(print |)", off_backends)
    )


@once
def test_the_local_builtin_answer_spells_names_as_completion_does(
    hover, signature_help, off_backends
):
    # Defect: with no backend, the builtin signature keeps inspect's Python
    # spelling (class_or_tuple) where a backend's answer shows
    # class-or-tuple.
    text = hover_text(hover("(isinst|ance 1 int)", off_backends))
    [label] = signature_labels(signature_help("(isinstance |)", off_backends))
    for shown in (text, label):
        assert "class-or-tuple" in shown and "class_or_tuple" not in shown


@once
def test_a_buffer_that_does_not_compile_gives_the_local_answer(
    hover, signature_help, backend_requests
):
    # A refused form before the cursor in the form holding it: neither the
    # buffer nor its prefix compiles there, so there is no view and no
    # request; the builtin answer stands.
    assert_local_print(
        hover("(do (def)\n  (pri|nt 1))"), signature_help("(do (def)\n  (print |))")
    )
    assert backend_requests.sent == []


@once
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


@once
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


@once
def test_a_call_inside_a_form_reaches_the_backend(signature_help, backend_requests):
    # The gate is the innermost open form's head only: a call typed inside
    # a macro's body is a call (the counterpart of the cases above).
    signature_help("(require lispy.macros *)\n(when x (.sort [1] |))")
    assert backend_requests.sent == [lsp.TEXT_DOCUMENT_SIGNATURE_HELP]


@once
def test_a_required_macro_answers_signature_help_before_the_backend(
    signature_help, backend_requests
):
    # Defect: the backend is asked first and its answer (print's signature)
    # wins over the usage of the macro the cursor is in.
    from lispy.form_docs import BUILTIN_MACROS

    result = signature_help("(require lispy.macros *)\n(print (when x |))")
    assert result is not None
    assert [s.label for s in result.signatures] == list(BUILTIN_MACROS["when"].usage)
    assert backend_requests.sent == []


@once
def test_a_headless_form_falls_back_to_the_enclosing_macro_usage(
    signature_help, backend_requests
):
    # Defect: with no backend answer for the headless innermost form ((g) |),
    # the enclosing macro's usage is dropped. The backend is asked first here
    # (the view compiles), so the request must have been sent; its answer is
    # none (pyrefly would answer for g, basedpyright does not).
    from lispy.form_docs import BUILTIN_MACROS

    backend_requests.answer["value"] = None
    result = signature_help("(require lispy.macros *)\n(when x ((g) |))")
    assert backend_requests.sent == [lsp.TEXT_DOCUMENT_SIGNATURE_HELP]
    assert result is not None
    assert [s.label for s in result.signatures] == list(BUILTIN_MACROS["when"].usage)
