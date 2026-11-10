"""The LSP keeps working on a buffer that is mid-edit.

Handlers read the buffer with parse(..., partial=True): the complete forms
before the first error, no version gates.  Diagnostics still use the strict
parse, so the error itself is reported.
"""

import asyncio
import importlib
import importlib.machinery
import importlib.util
import pathlib
import re
import sys
import threading
from types import SimpleNamespace

import pytest
from lsprotocol import types as lsp

import lispy.macros
from lispy import form_docs
from lispy.core import nodes
from lispy.core.compiler import features, literal
from lispy.core.compiler.utils import FORM_USAGE, HEADLESS_FORMS
from lispy.core.macro import macroexpand_then_compile
from lispy.core.parser import parse

# lispy.lsp re-exports the server *object* as ``server``; take the module.
server = importlib.import_module("lispy.lsp.server")
macro_modules = importlib.import_module("lispy.lsp.macro_modules")
pv = importlib.import_module("lispy.lsp.python_view")

URI = "file:///buffer.lpy"
COMPLETE = "(def foo [] 1)\n(def bar [x] (foo))\n"
BUFFERS = {
    "unclosed form": COMPLETE + "(fo",
    "unterminated f-string": COMPLETE + '(print f"{foo',
    "mismatched bracket": COMPLETE + "(foo]",
    "t-string on an old Python": COMPLETE + '(print t"{(foo)}")',
    "meaningless wrapper prefix": COMPLETE + "(foo ^*(a))",
}


@pytest.fixture(params=sorted(BUFFERS))
def buffer(request, monkeypatch):
    # The t-string case must be refused by the strict parse on every version.
    monkeypatch.setattr(features, "running_version", lambda: (3, 11, 13))
    return BUFFERS[request.param]


def ls_with(source):
    document = SimpleNamespace(source=source, uri=URI)
    workspace = SimpleNamespace(get_text_document=lambda uri: document, folders={})
    return SimpleNamespace(workspace=workspace)


def run_completion(ls, params):
    """The completion handler's answer (it is a coroutine)."""
    return asyncio.run(server.completion(ls, params))


def run_hover(ls, params):
    """The hover handler's answer (it is a coroutine)."""
    return asyncio.run(server.hover(ls, params))


def run_definition(ls, params):
    """The definition handler's answer (it is a coroutine)."""
    return asyncio.run(server.definition(ls, params))


def run_references(ls, params):
    """The references handler's answer (it is a coroutine)."""
    return asyncio.run(server.references(ls, params))


def position(line, character):
    return SimpleNamespace(
        text_document=SimpleNamespace(uri=URI),
        position=SimpleNamespace(line=line, character=character),
    )


def test_strict_parse_refuses_the_buffer(buffer):
    with pytest.raises(Exception):
        parse(buffer)


def test_partial_parse_keeps_the_complete_forms(buffer):
    assert [str(form) for form in parse(buffer, partial=True)][:2] == [
        "(def foo [] 1)",
        "(def bar [x] (foo))",
    ]


def test_document_symbols(buffer):
    symbols = server.document_symbol(ls_with(buffer), position(0, 0))
    assert [s.name for s in symbols] == ["foo", "bar"]


def test_lazy_imports_are_outline_modules():
    symbols = server.document_symbol(
        ls_with("(lazy-import json)\n(lazy-from os [sep])\n"), position(0, 0)
    )
    assert [(s.name, s.kind) for s in symbols] == [
        ("json", server.lsp.SymbolKind.Module),
        ("os", server.lsp.SymbolKind.Module),
    ]


def test_definition_hover_and_references(buffer):
    ls = ls_with(buffer)
    [location] = run_definition(ls, position(1, 14))
    assert location.range.start.line == 0
    refs = run_references(ls, position(0, 5))
    assert [r.range.start.line for r in refs][:2] == [0, 1]
    assert run_hover(ls, position(0, 1)) is not None


def test_a_parameter_is_no_definition_of_a_module_name():
    # Defect: the name-only local lookup lists every parameter spelled like
    # the name, so the module's x also jumps into g's parameter list.
    ls = ls_with("(= x 1)\n(def g [x] x)\n(print x)\n(fo")
    locations = run_definition(ls, position(2, 7))
    assert [loc.range.start.line for loc in locations] == [0]


@pytest.mark.parametrize("typed", ["", "_", "___lpy"])
def test_a_name_the_view_generated_is_never_offered(typed):
    # Defect: the failed forms' Any import (___lpy_any, labelled in hyphen
    # spelling as every backend name is) and the cursor sentinel are module
    # names of the view, so the backend offers them at a typed _.
    items = [
        lsp.CompletionItem(label=label)
        for label in ["___lpy-any", "___lpy_cursor___", "_private"]
    ]
    kept = server.filtered_answer(typed, items).items
    expected = {"": ["_private"], "_": ["_private"], "___lpy": []}[typed]
    assert [item.label for item in kept] == expected


def test_completion(buffer):
    # Right after the last line's (, so no typed word filters the list.
    items = run_completion(ls_with(buffer), position(2, 1))
    labels = {item.label for item in items.items}
    assert {"foo", "bar"} <= labels


def test_diagnostics_still_report_the_error(buffer):
    [diagnostic] = server.compute_diagnostics(buffer, URI)
    assert diagnostic.message.startswith("Parse error:")
    assert diagnostic.range.start.line == 2


@pytest.mark.parametrize(
    ("source", "line", "underlined"),
    [
        # An indented form on a later line, the reported case: the body is
        # being typed after the head.
        ("(for i in (range 3)\n  (while (and busy)))", 1, "(while"),
        ("(def f [x])", 0, "(def"),
        ("(+ 1 2))", 0, ")"),
    ],
)
def test_lispython_diagnostics_underline_the_form_head(source, line, underlined):
    # Defect: a zero-width range at the form, which eglot places a character
    # early and VS Code widens to a word of its own, and a message carrying
    # str(SyntaxError)'s "(line N)" suffix.
    [diagnostic] = server.compute_diagnostics(source, URI)
    start, end = diagnostic.range.start, diagnostic.range.end
    assert start.line == end.line == line
    assert source.split("\n")[line][start.character : end.character] == underlined
    assert not re.search(r"(\(|, )line \d+\)$", diagnostic.message)


def test_each_failed_form_has_its_own_diagnostic():
    # Defect: the compile stops at the first failed form, so a later one
    # is reported only once the first is fixed, or a later form's failure
    # replaces the first's.
    source = "(while)\n(= y 1)\n(for i in (range 3)\n  (def f [x]))"
    lines = source.split("\n")
    found = [
        (
            d.range.start.line,
            lines[d.range.start.line][d.range.start.character : d.range.end.character],
            d.severity,
        )
        for d in server.compute_diagnostics(source, URI)
    ]
    warning = server.lsp.DiagnosticSeverity.Warning
    assert found == [(0, "(while", warning), (3, "(def", warning)]


def assert_placed_in_required_module(diagnostic, module, line, start, end):
    rng = diagnostic.range
    assert (rng.start.line, rng.start.character) == (0, 0)
    assert (rng.end.line, rng.end.character) == (0, len("(require"))
    # The full path, never the basename alone, which two packages may share.
    assert f"({module}:{line + 1})" in diagnostic.message
    [related] = diagnostic.related_information
    assert related.location.uri == module.as_uri()
    there = related.location.range
    assert (there.start.line, there.start.character) == (line, start)
    assert (there.end.line, there.end.character) == (line, end)


def test_an_error_in_a_required_module_is_not_placed_in_the_buffer(tmp_path):
    # Defects: a required module's compile error carries that module's line
    # (3, its (while)), which the buffer's diagnostic took as its own: it
    # underlined the buffer's third form, (=, with a bare message naming no
    # file. It is placed at the require that raised it. Its message then
    # named the module by basename alone ("lspbrokenreq.lpy, line 3"), and
    # no related location led to the line there.
    module = tmp_path / "lspbrokenreq.lpy"
    module.write_text("(= a 1)\n(= b 2)\n  (while)\n")
    source = "(require lspbrokenreq)\n(= a 1)\n(= b 2)"
    try:
        [diagnostic] = server.compute_diagnostics(
            source, (tmp_path / "main.lpy").as_uri()
        )
    finally:
        sys.modules.pop("lspbrokenreq", None)
    # Its (while head on line 3, after two spaces.
    assert_placed_in_required_module(diagnostic, module, 2, 2, 8)


def test_a_required_python_modules_syntax_error_names_its_file(tmp_path):
    # Defect: CPython's SyntaxError of a required .py reached the message as
    # its str, "invalid syntax (lspbrokenpy.py, line 2)", by basename alone.
    module = tmp_path / "lspbrokenpy.py"
    module.write_text("x = 1\nx = = 2\n")
    try:
        [diagnostic] = server.compute_diagnostics(
            "(require lspbrokenpy)", (tmp_path / "main.lpy").as_uri()
        )
    finally:
        sys.modules.pop("lspbrokenpy", None)
    assert diagnostic.message.startswith("Compile warning: invalid syntax (")
    # The second = of "x = = 2", where CPython places it.
    assert_placed_in_required_module(diagnostic, module, 1, 4, 5)


def test_a_position_in_text_a_macro_parsed_is_not_the_buffers():
    # Defect: an include-like macro parses other text, whose (while) error
    # carries filename None and that text's line 3, which the diagnostic took
    # as the buffer's third line and underlined its unrelated (= b 2).
    source = (
        "(defmacro inc [] (import lispy.core.parser)"
        ' (return (sub (lispy.core.parser.parse "\\n\\n(while)") 0)))\n'
        "(= a 1)\n"
        "(= b 2)\n"
        "(inc)"
    )
    [diagnostic] = server.compute_diagnostics(source, URI)
    rng = diagnostic.range
    # The (inc) call raised it, on the buffer's fourth line (index 3).
    assert (rng.start.line, rng.start.character, rng.end.character) == (3, 0, 4)


def test_an_internal_error_range_is_not_zero_width():
    # Defect: (0,0)-(0,0), which eglot places a character early and VS Code
    # widens to a word of its choosing.
    [diagnostic] = server.internal_error_diagnostics(RuntimeError("boom"))
    rng = diagnostic.range
    assert (rng.start.line, rng.start.character) != (rng.end.line, rng.end.character)


def test_a_required_modules_related_range_counts_in_the_editors_units(tmp_path):
    # Defect: the related range's columns measured over the buffer's lines,
    # not the module's line, so the astral character before the module's
    # (while (code point 10, UTF-16 unit 11) went uncounted.
    from pygls.workspace import PositionCodec, TextDocument

    (tmp_path / "lspastralreq.lpy").write_text('(= s "😀") (while)\n')
    source = "(require lspastralreq)"
    uri = (tmp_path / "main.lpy").as_uri()
    doc = TextDocument(uri, source, position_codec=PositionCodec(encoding="utf-16"))
    try:
        [diagnostic] = server.compute_diagnostics(source, uri, doc=doc)
    finally:
        sys.modules.pop("lspastralreq", None)
    there = diagnostic.related_information[0].location.range
    assert (there.start.character, there.end.character) == (11, 17)


def other_file_error(tmp_path, kind):
    module = tmp_path / "other.lpy"
    module.write_text("(= a 1)\n")
    return {
        # A name no file holds, relative to the server's working directory.
        "relative name": SyntaxError("boom", ("rel/other.lpy", 1, 1, "(= a 1)")),
        "no lineno": SyntaxError("boom", (str(module), None, None, None)),
        "no msg": SyntaxError(None, (str(module), 1, 1, "(= a 1)")),
        "non-str msg": SyntaxError(42, (str(module), 1, 1, "(= a 1)")),
    }[kind]


@pytest.mark.parametrize(
    "kind", ["relative name", "no lineno", "no msg", "non-str msg"]
)
def test_another_files_error_names_only_what_it_carries(tmp_path, kind):
    # Defects: a relative fake filename was resolved against the working
    # directory into a file that does not exist; an error with no lineno
    # named line 1; a None msg read "None (...)", and a None or non-str msg
    # made its related information raise TypeError.
    exc = other_file_error(tmp_path, kind)
    diagnostic = server.error_diagnostic(
        exc,
        "(require other)",
        (tmp_path / "main.lpy").as_uri(),
        lsp.DiagnosticSeverity.Warning,
        "Compile warning: ",
    )
    if kind == "relative name":
        assert diagnostic.related_information is None
        assert diagnostic.message == f"Compile warning: {exc}"
    elif kind == "no lineno":
        assert diagnostic.message.endswith(f"({tmp_path / 'other.lpy'})")
    elif kind == "no msg":
        assert "None" not in diagnostic.message
        [related] = diagnostic.related_information
        assert related.message == "SyntaxError"
    else:
        [related] = diagnostic.related_information
        assert related.message == "42"


def test_an_error_with_no_position_marks_the_form_that_raised_it():
    # Defects: an error naming no position took the head of the first form,
    # (defmacro, as if that form were at fault; or every such error stacked
    # on the buffer's first character, one per failed form.
    source = '(defmacro boom [] (raise (ValueError "boom")))\n(= a 1)\n(boom)\n  (boom)'
    lines = source.split("\n")
    found = [
        (
            d.range.start.line,
            lines[d.range.start.line][d.range.start.character : d.range.end.character],
            d.message,
        )
        for d in server.compute_diagnostics(source, URI)
    ]
    message = "Compile warning: boom"
    assert found == [(2, "(boom", message), (3, "(boom", message)]


def test_another_exceptions_line_is_not_a_buffer_position():
    # Defect: a macro's JSONDecodeError carries lineno 4, a line of the JSON
    # text it parsed, which the diagnostic took as the buffer's fourth line
    # and underlined (= d 4) there.
    source = (
        '(defmacro bad [] (import json) (json.loads "{\\n\\n\\n  bad}"))\n'
        "(= b 2)\n"
        "(= c 3)\n"
        "(= d 4)\n"
        "(bad)"
    )
    [diagnostic] = server.compute_diagnostics(source, URI)
    rng = diagnostic.range
    # The (bad) call raised it, on the buffer's fifth line (index 4).
    assert (rng.start.line, rng.start.character, rng.end.character) == (4, 0, 4)
    assert "line 4" in diagnostic.message


def test_a_lone_carriage_return_ends_the_measured_line():
    # Defect: the head was measured on the line split at LF alone, while the
    # reader (and the error's line) also ends a line at a lone CR, so
    # (while on the second line was measured on another line's text.
    source = "(f x)\r  (while)"
    [diagnostic] = server.compute_diagnostics(source, URI)
    rng = diagnostic.range
    assert rng.start.line == 1
    assert source.split("\r")[1][rng.start.character : rng.end.character] == "(while"


def test_a_diagnostic_range_counts_in_the_editors_units():
    # Defect: lispython's own diagnostic went out in code points, so a
    # UTF-16 editor underlined one unit early per astral character before
    # the form (here the head (while starts at code point 10, unit 11).
    from pygls.workspace import PositionCodec, TextDocument

    source = '(= s "😀") (while)'
    doc = TextDocument(URI, source, position_codec=PositionCodec(encoding="utf-16"))
    [diagnostic] = server.compute_diagnostics(source, URI, doc=doc)
    assert (diagnostic.range.start.character, diagnostic.range.end.character) == (
        11,
        17,
    )


def drive(handler, *args):
    """Run a pygls builtin handler (a generator) and the user handler it
    yields, as pygls' protocol does; its return value."""
    steps = handler(*args)
    try:
        user_handler, user_args, _ = next(steps)
        user_handler(*user_args)
        steps.send(None)
    except StopIteration as stop:
        return stop.value
    raise AssertionError("handler yielded twice")


@pytest.fixture
def served(monkeypatch):
    """The real server's protocol after initialize: open(source) -> the
    document its workspace holds after didOpen; change(range, text) -> that
    document after an incremental didChange.  pygls' builtins run, the user
    didOpen/didChange handlers do not (they compile); the protocol is
    restored after."""
    protocol = server.server.protocol
    saved = dict(vars(protocol))
    monkeypatch.setattr(server.backends, "name", server.backends.name)
    monkeypatch.setattr(
        server.backends, "python_diagnostics", server.backends.python_diagnostics
    )
    drive(
        protocol.lsp_initialize,
        lsp.InitializeParams(capabilities=lsp.ClientCapabilities()),
    )

    def builtin(handler, params):
        # Only the builtin's own workspace update, before the user handler.
        steps = handler(params)
        next(steps, None)
        steps.close()

    def open_(source):
        item = lsp.TextDocumentItem(uri=URI, language_id="lpy", version=1, text=source)
        builtin(
            protocol.lsp_text_document__did_open,
            lsp.DidOpenTextDocumentParams(text_document=item),
        )
        return protocol.workspace.get_text_document(URI)

    def change(rng, text):
        partial = lsp.TextDocumentContentChangePartial(range=rng, text=text)
        builtin(
            protocol.lsp_text_document__did_change,
            lsp.DidChangeTextDocumentParams(
                text_document=lsp.VersionedTextDocumentIdentifier(uri=URI, version=2),
                content_changes=[partial],
            ),
        )
        return protocol.workspace.get_text_document(URI)

    yield SimpleNamespace(open=open_, change=change)
    vars(protocol).clear()
    vars(protocol).update(saved)


def test_an_incremental_edit_lands_on_the_readers_line(served):
    # Defect: the workspace kept pygls' TextDocument, whose lines split on
    # str.splitlines() (\x0c, \x1c-\x1e, \x85, U+2028 too): an incremental
    # edit after one of them was applied to the wrong line, corrupting the
    # buffer the server compiles.
    edit = lsp.Range(
        start=lsp.Position(line=1, character=1), end=lsp.Position(line=1, character=6)
    )
    for char in "\x0c\x1c\x85 ":
        served.open(f'(= s "a{char}b")\n(print 1)\n')
        assert served.change(edit, "len").source == f'(= s "a{char}b")\n(len 1)\n'


def test_a_position_after_a_final_line_break_is_on_the_empty_last_line(served):
    # Defect: pygls' lines have no empty line after a final break, so the
    # codec clamped a position there to the end of the line before it.
    doc = served.open("(print 1)\n")
    assert server.cursor_of(doc, lsp.Position(line=1, character=0)) == (2, 0)


def test_a_diagnostic_range_ignores_line_breaks_the_reader_does_not_count(served):
    # Defect: columns were converted over str.splitlines()' lines, which
    # break at \x0c, \x1c-\x1e, \x85 and U+2028 too, so every range after
    # one of them on its line was converted over the wrong line.
    for char in "\x0c\x1c\x85 ":
        source = f'(= s "a{char}b")\n  (while)'
        doc = served.open(source)
        [diagnostic] = server.compute_diagnostics(source, URI, doc=doc)
        rng = diagnostic.range
        assert (rng.start.line, rng.start.character, rng.end.character) == (1, 2, 8)


def test_the_cursor_word_is_found_on_the_readers_line_after_a_lone_cr():
    # Defect: the buffer was split on "\n" only, so after a lone \r (a line
    # break to the reader) the cursor's line was missing or another one:
    # no cursor view, no completion context.
    source = "(x)\r(fo)\r\n(y)"
    marked, _, start, end, partial = pv.marked_source(source, 2, 3)
    assert (marked, start, end, partial) == (
        f"(x)\r({pv.CURSOR_SENTINEL})\r\n(y)",
        1,
        3,
        "fo",
    )
    assert server.line_text(source, 2) == "(fo)"
    assert pv.source_prefix(source, 2, 3) == "(x)\r(fo"


def test_a_diagnostic_never_underlines_a_crlf_lines_carriage_return():
    # Defect: the line's text kept CRLF's \r, so an error at the end of the
    # line underlined the invisible \r instead of the line's last character.
    source = "(f x\r\n(g)\r\n"
    exc = SyntaxError("boom", (None, 1, 5, "(f x", 1, 5))
    diagnostic = server.error_diagnostic(
        exc, source, URI, lsp.DiagnosticSeverity.Error, "Parse error: "
    )
    rng = diagnostic.range
    assert source.split("\n")[0][rng.start.character : rng.end.character] == "x"


def test_a_diagnostics_parse_warning_names_the_document_file():
    with pytest.warns(SyntaxWarning) as rec:
        server.compute_diagnostics('(print f"{(g \\"a\\")}")', URI)
    assert [w.filename for w in rec] == ["/buffer.lpy"]


def test_partial_parse_does_not_leak_into_strict_parse(monkeypatch):
    monkeypatch.setattr(features, "running_version", lambda: (3, 11, 13))
    parse(BUFFERS["t-string on an old Python"], partial=True)
    with pytest.raises(Exception):
        parse('(print t"x")')


def test_refused_prefix_form_still_closes_in_the_call_scan():
    # ^*( is a bad-prefix token, not an opening; its ) must close it, not g.
    source = "(g ^*(f) x)"
    heads = server.open_form_heads(source, 1, source.index("x"))
    assert heads == ["g"]
    assert server.innermost_named(heads) == "g"


@pytest.mark.skipif(sys.version_info < (3, 14), reason="t-strings: 3.14+")
def test_partial_parse_of_a_valid_t_string_is_complete():
    forms = parse('(print t"{x}")', partial=True)
    assert [str(f.op) for f in forms] == ["print"]


ALIASES = (
    "(type Alias int)\n"
    "(type Pair[T] (tuple T T))\n"
    "(def use [x] (type Inner x))\n"
    "(type Pair2[T] (tuple T T)\n"  # mid-edit: unclosed
)
# Shapes that call the builtin `type`: one or three operands, a dotted or
# non-symbol first operand.
CALLS = '(type x)\n(type 1 2 3)\n(type a.b c)\n(type "s" c)\n(type)\n'


def test_type_alias_document_symbols():
    symbols = server.document_symbol(ls_with(ALIASES), position(0, 0))
    assert [s.name for s in symbols] == ["Alias", "Pair", "use"]
    assert [c.name for c in symbols[2].children] == ["Inner"]
    # The selection covers the whole head, as it does for def and class.
    assert symbols[1].selection_range.end.character == 13


def test_type_alias_definition():
    # The references come before the unclosed form, which ends the partial parse.
    ls = ls_with(
        "(type Alias int)\n(type Pair[T] (tuple T T))\n(= y Pair)\n(= z Alias)\n(fo"
    )
    [pair] = run_definition(ls, position(2, 5))
    assert (pair.range.start.line, pair.range.start.character) == (1, 6)
    [alias] = run_definition(ls, position(3, 5))
    assert alias.range.start.line == 0


def test_type_alias_hover_on_the_keyword():
    hover = run_hover(ls_with("(type Alias int)"), position(0, 2))
    assert "type alias" in hover.contents.value


def test_type_call_keeps_the_builtin_hover():
    hover = run_hover(ls_with("(type x)\n(type A int)"), position(0, 2))
    assert "Python builtin" in hover.contents.value


@pytest.mark.parametrize("name", ["f-string", "ife"])
def test_a_bound_special_form_name_has_no_form_hover(name):
    # Defect: the form's doc answers for a name the buffer bound, which the
    # compiler reads as the form only at a form head.
    source = f"(= {name} 3)\n(print {name})"
    assert run_hover(ls_with(source), position(0, 4)) is None
    assert run_hover(ls_with(source), position(1, 8)) is None


@pytest.mark.parametrize(
    "source, character, name",
    [('(print f"ab{x}")', 8, "f-string"), ("(ife a b c)", 2, "ife")],
    ids=["f-literal", "ife head"],
)
def test_a_special_form_at_its_head_keeps_the_form_hover(source, character, name):
    hover = run_hover(ls_with(source), position(0, character))
    assert hover.contents.value == server.SPECIAL_FORM_DOCS[name]


@pytest.mark.parametrize(
    "source, character, name",
    [
        ("[x for x in xs if x]", 4, "for"),
        ("[x for x in xs if x]", 16, "if"),
        ("(for x in xs (print x))", 8, "in"),
        ("(def f [a * b] 1)", 10, "*"),
        ("[x async-for x in xs]", 5, "async-for"),
    ],
    ids=["for clause", "if clause", "for's in", "star marker", "async-for clause"],
)
def test_a_form_no_name_can_bind_keeps_its_hover_off_a_head(source, character, name):
    # Defect: every special form answers only at a form head, so a keyword
    # or a clause keyword the compiler reads as the form anywhere loses it.
    hover = run_hover(ls_with(source), position(0, character))
    assert hover.contents.value == server.SPECIAL_FORM_DOCS[name]


def test_type_call_is_not_a_definition():
    ls = ls_with(CALLS)
    assert server.document_symbol(ls, position(0, 0)) == []
    assert server.collect_definitions(parse(CALLS), URI) == {}


class TestRequireSelection:
    """go-to-definition and hover follow the compiler's (require ...) rule:
    absent list = every macro as module.name, * = every macro, [a b] = those,
    [] = none."""

    MACROS = "(defmacro twice [x] x)\n(defmacro thrice [x] x)\n"

    @staticmethod
    def _find(source, name):
        return server.find_import_for_name(parse(source), name)

    @pytest.fixture
    def workspace(self, tmp_path, monkeypatch):
        (tmp_path / "macmod.lpy").write_text(self.MACROS)
        monkeypatch.setattr(server, "WORKSPACE_INDEX", {})
        monkeypatch.setattr(server, "FILE_TREES", {})
        return {"macmod": SimpleNamespace(uri=tmp_path.as_uri())}

    @staticmethod
    def _definition(workspace, source, line, character):
        document = SimpleNamespace(source=source, uri=URI)
        ls = SimpleNamespace(
            workspace=SimpleNamespace(
                get_text_document=lambda uri: document, folders=workspace
            )
        )
        return run_definition(ls, position(line, character))

    def test_absent_list_binds_only_the_qualified_names(self):
        source = "(require macmod)\n(macmod.twice 1)"
        assert self._find(source, "macmod.twice") is not None
        assert self._find(source, "twice") is None

    def test_empty_list_binds_nothing(self):
        source = "(require macmod [])\n(twice 1)"
        assert self._find(source, "twice") is None
        assert self._find(source, "macmod.twice") is None

    def test_named_list_binds_those_names(self):
        source = "(require macmod [twice])\n(twice 1)"
        assert self._find(source, "twice") is not None
        assert self._find(source, "thrice") is None

    def test_star_binds_every_name(self):
        assert self._find("(require macmod *)\n(thrice 1)", "thrice") is not None

    def test_unrelated_name_is_not_bound_by_a_require(self):
        # Previously any require node answered for any name.
        assert self._find("(require macmod [twice])", "print") is None

    def test_definition_through_a_named_require(self, workspace):
        [loc] = self._definition(workspace, "(require macmod [twice])\n(twice 1)", 1, 2)
        assert loc.uri.endswith("macmod.lpy")

    def test_definition_through_a_bare_require(self, workspace):
        [loc] = self._definition(workspace, "(require macmod)\n(macmod.twice 1)", 1, 4)
        assert loc.uri.endswith("macmod.lpy")

    def test_definition_through_an_empty_list_finds_nothing(self, workspace):
        got = self._definition(workspace, "(require macmod [])\n(twice 1)", 1, 2)
        assert not got

    def test_hover_text_covers_bare_and_empty_forms(self):
        text = server.SPECIAL_FORM_DOCS["require"]
        assert "(require module)" in text
        assert "[]" in text and "none" in text

    @pytest.mark.parametrize(
        "form",
        [
            "(require lispy.macros foo)",
            "(require lispy.macros [a] b)",
            '(require lispy.macros "ab")',
            "(require lispy.macros :as x)",
        ],
    )
    def test_malformed_require_binds_nothing_and_does_not_raise(self, form):
        # The module is a built-in macro module and the form opts out of the
        # prelude, so only the malformed selection stands between the buffer
        # and a hover on `cond`.
        tree = parse(form + "\n(cond)")
        assert server.find_import_for_name(tree, "print") is None
        assert server.builtin_macro_doc(tree, "cond") is None

    def test_a_require_naming_no_module_keeps_the_prelude(self):
        # (require) names no lispy.macros, so it is no opt-out.
        tree = parse("(require)\n(cond)", partial=True)
        assert server.find_import_for_name(tree, "print") is None
        assert "(cond test" in server.builtin_macro_doc(tree, "cond")

    def test_bare_require_of_the_built_in_macros_is_qualified_only(self):
        tree = parse("(require lispy.macros)\n(cond)")
        assert server.builtin_macro_doc(tree, "cond") is None
        assert "(cond test" in server.builtin_macro_doc(tree, "lispy.macros.cond")

    def test_an_aliased_require_qualifies_by_the_alias(self):
        # Defects: the alias is ignored (the macros stay module.name), or it
        # is compared by its raw spelling, so my_m.twice misses (... as my-m).
        source = "(require macmod as my-m)\n(my-m.twice 1)"
        assert self._find(source, "my-m.twice") is not None
        assert self._find(source, "my_m.twice") is not None
        assert self._find(source, "macmod.twice") is None

    def test_a_relative_require_qualifies_by_the_last_part(self):
        # Defect: (require ..pkg.macmod) answers for ..pkg.macmod.twice, a
        # name no code can spell, and not for macmod.twice.
        source = "(require ..pkg.macmod)\n(macmod.twice 1)"
        assert self._find(source, "macmod.twice") is not None
        assert self._find(source, "..pkg.macmod.twice") is None

    def test_hover_through_an_aliased_require_of_the_built_in_macros(self):
        # With no view the syntax scan decides; it must read the alias.
        source = "(require lispy.macros as lm)\n(lm.cond x 1)"
        hover = run_hover(ls_with(source), position(1, 5))
        assert "(cond test" in hover.contents.value

    def test_definition_through_an_aliased_require(self, workspace):
        source = "(require macmod as m)\n(m.twice 1)"
        [loc] = self._definition(workspace, source, 1, 4)
        assert loc.uri.endswith("macmod.lpy")

    @pytest.fixture
    def package(self, tmp_path, monkeypatch):
        """A package pkg (holding macmod.lpy) under a directory whose name
        has a space, which a file: URI percent-encodes, beside a top-level
        macmod.lpy outside any package; the workspace folder is that
        directory's parent."""
        monkeypatch.setattr(server, "WORKSPACE_INDEX", {})
        monkeypatch.setattr(server, "FILE_TREES", {})
        root = tmp_path / "my proj"
        pkg = root / "pkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "macmod.lpy").write_text(self.MACROS)
        (root / "macmod.lpy").write_text(self.MACROS)
        return pkg

    @staticmethod
    def _definition_in(source, uri, folder, line, character):
        document = SimpleNamespace(source=source, uri=uri)
        ls = SimpleNamespace(
            workspace=SimpleNamespace(
                get_text_document=lambda uri: document,
                folders={"root": SimpleNamespace(uri=folder.as_uri())},
            )
        )
        return run_definition(ls, position(line, character))

    def test_definition_through_a_relative_require(self, package):
        # Defects: the relative module is looked up from the workspace root
        # (or not at all), or the document's file: URI is read undecoded
        # (my%20proj), so the definition is not found.
        source = "(require .macmod)\n(macmod.thrice 1)"
        uri = (package / "mod.lpy").as_uri()
        assert "%20" in uri
        [loc] = self._definition_in(source, uri, package.parent.parent, 1, 9)
        assert loc.uri == (package / "macmod.lpy").as_uri()
        assert loc.range.start.line == 1

    @pytest.mark.parametrize(
        "form", ["(from .macmod [thrice])", "(from . [macmod])", "(require .macmod)"]
    )
    def test_a_relative_import_target_resolves_in_the_package(self, package, form):
        # Defect: a relative from (sharing the require's lookup) resolves
        # from the workspace root, so it finds the top-level macmod.lpy or,
        # for the package itself, nothing. (Definition never reaches this
        # for a from: the name it binds is the buffer's own definition.)
        (package / "__init__.lpy").write_text("")
        [node] = parse(form)
        origin = str(package / "mod.lpy")
        folders = [SimpleNamespace(uri=package.parent.as_uri())]
        got = server.find_import_target(node, folders, origin=origin)
        expected = package / ("__init__.lpy" if "[macmod]" in form else "macmod.lpy")
        assert got == str(expected)

    @pytest.mark.parametrize("form", ["(require pkg.dual)", "(require .dual)"])
    def test_a_package_wins_over_a_module_file_of_its_name(self, package, form):
        # Defect: dual.lpy is taken where Python's FileFinder imports the
        # package dual/ (its __init__.lpy).
        (package / "dual").mkdir()
        (package / "dual" / "__init__.lpy").write_text("")
        (package / "dual.lpy").write_text("")
        [node] = parse(form)
        origin = str(package / "mod.lpy")
        folders = [SimpleNamespace(uri=package.parent.as_uri())]
        got = server.find_import_target(node, folders, origin=origin)
        assert got == str(package / "dual" / "__init__.lpy")

    @pytest.mark.parametrize("init", ["__init__.py", "__init__.pyc"])
    def test_a_package_of_another_suffix_hides_the_module_file(self, package, init):
        # Defect: only __init__.lpy counts as the package, so dual.lpy is
        # opened where Python imports the package dual/ (a .py one here),
        # whose code is not that file's.
        (package / "dual").mkdir()
        (package / "dual" / init).write_text("")
        (package / "dual.lpy").write_text("")
        [node] = parse("(require .dual)")
        origin = str(package / "mod.lpy")
        assert server.find_import_target(node, [], origin=origin) is None

    @pytest.mark.parametrize("spelling", ["ok?", "->foo"])
    def test_a_required_module_is_looked_up_by_its_import_name(self, package, spelling):
        # Defect: the lookup mangles the name by a rule of its own (each -
        # as _), so it misses the file the import of ok? (ok_p) reads.
        [stmt] = list(
            macroexpand_then_compile(
                parse(f"(import {spelling})"), {"__macro_namespace": {}}
            )
        )
        target = package / f"{stmt.names[0].name}.lpy"
        target.write_text("")
        [node] = parse(f"(require .{spelling})")
        origin = str(package / "mod.lpy")
        assert server.find_import_target(node, [], origin=origin) == str(target)

    @pytest.mark.parametrize("form", ["(require ..macmod)", "(from ..macmod [x])"])
    def test_a_relative_module_beyond_the_top_level_package_resolves_to_nothing(
        self, package, form
    ):
        # Defect: each dot walks up one directory with no package boundary,
        # so ..macmod from pkg/mod.lpy finds the top-level macmod.lpy, which
        # Python's import never reaches.
        [node] = parse(form)
        origin = str(package / "mod.lpy")
        folders = [SimpleNamespace(uri=package.parent.as_uri())]
        assert server.find_import_target(node, folders, origin=origin) is None

    # A module of a namespace package is also the script its directory runs
    # (lpy lspnspkg/a.lpy), so it requires its package's module relatively
    # and its sibling module bare.
    NS_SOURCE = "(require .mac *)\n(require lspnshelper *)\n(= y (twice (thrice 1)))"
    NS_EXPANDED = "y = 1 + 1 + 1 + (1 + 1 + 1)"
    NS_MODULES = ["lspnspkg", "lspnspkg.mac", "lspnshelper"]

    @pytest.fixture
    def namespace(self, tmp_path, monkeypatch):
        """A namespace package lspnspkg (no __init__) holding mac.lpy and
        lspnshelper.lpy, in the workspace folder tmp_path/proj, which the
        server's workspace_folders reports; what a compile imports is
        dropped after the test."""
        monkeypatch.setattr(server, "WORKSPACE_INDEX", {})
        monkeypatch.setattr(server, "FILE_TREES", {})
        monkeypatch.setattr(sys, "path", list(sys.path))
        for name in self.NS_MODULES:
            monkeypatch.delitem(sys.modules, name, raising=False)
        pkg = tmp_path / "proj" / "lspnspkg"
        pkg.mkdir(parents=True)
        (pkg / "mac.lpy").write_text("(defmacro twice [x] (return `(+ ~x ~x)))\n")
        (pkg / "lspnshelper.lpy").write_text(
            "(defmacro thrice [x] (return `(+ ~x ~x ~x)))\n"
        )
        folders = [SimpleNamespace(uri=pkg.parent.as_uri())]
        monkeypatch.setattr(server, "workspace_folders", lambda: folders)
        yield pkg
        for name in self.NS_MODULES:
            sys.modules.pop(name, None)

    def test_a_namespace_module_compiles_its_relative_and_sibling_requires(
        self, namespace
    ):
        # Defects: the document is named as a top-level module, so the
        # relative require fails ("no known parent package"); or only the
        # project root is on sys.path while its macros run, so the sibling
        # require, which `lpy lspnspkg/a.lpy` imports, loads no macros.
        uri = (namespace / "a.lpy").as_uri()
        compiled = server.compile_document(uri, None, self.NS_SOURCE)
        assert compiled.error is None
        assert compiled.view.text.endswith(self.NS_EXPANDED)

    def test_a_cursor_view_compiles_a_namespace_module(self, namespace):
        # Defect: the cursor view names the document by the regular-package
        # rule alone, so its requires fail and the view is None.
        source = self.NS_SOURCE + "\n(print y)"
        doc = SimpleNamespace(uri=(namespace / "a.lpy").as_uri(), version=None)
        doc.source = source

        async def view():
            deadline = asyncio.get_running_loop().time() + 30
            return await server.awaited_cursor_view(doc, 4, 8, deadline)

        got = asyncio.run(view())
        assert got is not None and self.NS_EXPANDED in got.text

    def test_a_disk_view_compiles_a_namespace_module(self, namespace, tmp_path):
        # Defect: the disk view of a file the backend points into names it
        # as a top-level module, so it never compiles to its shadow's text
        # and a location in it is dropped.
        path = namespace / "a.lpy"
        path.write_text(self.NS_SOURCE)
        pv.write_shadows(str(namespace.parent), str(tmp_path / "view"))
        shadow = tmp_path / "view" / "lspnspkg" / "a.py"
        assert shadow.read_text().endswith(self.NS_EXPANDED)
        uri = path.as_uri()
        try:
            origin = server.disk_origin(uri, str(shadow), server.disk_views.generation)
        finally:
            server.disk_views.forget(uri)
        assert origin is not None and origin[0] == self.NS_SOURCE

    def test_an_absolute_require_resolves_beside_a_script_first(self, namespace):
        # Defect: an absolute module is looked up only under the workspace
        # folder (and its src), so the sibling lspnshelper.lpy that the
        # compile imports is missed, or a folder-level one of that name is
        # taken instead.
        project = namespace.parent
        (project / "lspnshelper.lpy").write_text("")
        [node] = parse("(require lspnshelper)")
        origin = str(namespace / "a.lpy")
        folders = server.workspace_folders()
        got = server.find_import_target(node, folders, origin=origin)
        assert got == str(namespace / "lspnshelper.lpy")
        # With no origin the folder's own module is the answer.
        assert server.find_import_target(node, folders) == str(
            project / "lspnshelper.lpy"
        )

    def test_a_file_link_takes_its_directorys_project(self, namespace, tmp_path):
        # Defect: the project is found from the link's target, which lies
        # outside every folder, so the document compiles as the top-level
        # a and its relative require fails, while the shadow sweep (which
        # walks the link where it is) names it lspnspkg.a.
        elsewhere = tmp_path / "elsewhere" / "a.lpy"
        elsewhere.parent.mkdir()
        elsewhere.write_text(self.NS_SOURCE)
        link = namespace / "a.lpy"
        link.symlink_to(elsewhere)
        folders = server.workspace_folders()
        assert server.project_root_of(str(link), folders) == str(
            namespace.parent.resolve()
        )
        compiled = server.compile_document(link.as_uri(), None, self.NS_SOURCE)
        assert compiled.error is None
        assert compiled.view.text.endswith(self.NS_EXPANDED)

    def test_definition_through_a_relative_require_in_a_namespace_package(
        self, namespace
    ):
        # Defect: the relative module resolves against an empty package,
        # so neither the module nor its macro is found.
        source = "(require .mac)\n(mac.twice 1)"
        origin = namespace / "a.lpy"
        [loc] = self._definition_in(source, origin.as_uri(), namespace.parent, 1, 6)
        assert loc.uri == (namespace / "mac.lpy").as_uri()
        [node] = parse("(from .mac [twice])")
        folders = [SimpleNamespace(uri=namespace.parent.as_uri())]
        got = server.find_import_target(node, folders, origin=str(origin))
        assert got == str(namespace / "mac.lpy")

    def test_a_buffer_with_no_file_resolves_no_relative_module(
        self, package, monkeypatch
    ):
        # Defect: an untitled: buffer's path is taken as its URI text, whose
        # dirname is the server's working directory, so a relative module
        # resolves against wherever the server runs.
        monkeypatch.chdir(package)
        source = "(require .macmod)\n(macmod.thrice 1)"
        assert not self._definition_in(source, "untitled:x", package.parent, 1, 9)

    def test_a_cached_view_follows_the_documents_module_naming(
        self, tmp_path, monkeypatch
    ):
        # Defect: the view cache keys a document's compile by version and
        # macro epoch alone, so once an __init__.py makes its directory a
        # package (or removes it), a request on the same version answers
        # from the compile under the old name and package: the relative
        # require still fails, or a view whose require no longer resolves
        # is still served.
        monkeypatch.setattr(server, "view_cache", pv.ViewCache())
        monkeypatch.setattr(server, "workspace_folders", lambda: [])
        monkeypatch.setattr(sys, "path", list(sys.path))
        modules = ["lspnamingpkg", "lspnamingpkg.mac"]
        for name in modules:
            monkeypatch.delitem(sys.modules, name, raising=False)
        pkg = tmp_path / "lspnamingpkg"
        pkg.mkdir()
        (pkg / "mac.lpy").write_text("(defmacro twice [x] (return `(+ ~x ~x)))\n")
        source = "(require .mac *)\n(= y (twice 1))"
        doc = SimpleNamespace(uri=(pkg / "x.lpy").as_uri(), version=1, source=source)
        try:
            assert server.compile_document(doc.uri, 1, source).failures
            (pkg / "__init__.py").write_text("")
            compiled = server.compile_document(doc.uri, 1, source)
            assert compiled.failures == []
            assert compiled.view.text.endswith("y = 1 + 1")
            assert server.compiled_view(doc) is compiled.view
            (pkg / "__init__.py").unlink()
            assert server.compiled_view(doc) is None
        finally:
            for name in modules:
                sys.modules.pop(name, None)

    @staticmethod
    def _target_in(folder, form, origin=None):
        [node] = parse(form)
        folders = [SimpleNamespace(uri=folder.as_uri())]
        return server.find_import_target(node, folders, origin=origin)

    def test_a_workspace_module_that_is_no_lpy_hides_an_installed_one(self, tmp_path):
        # Defect: the workspace lookup finds lispy/tools.py, the module the
        # import reads there, and falls through to sys.path, which opens the
        # installed lispy/tools.lpy, code that import never runs.
        (tmp_path / "lispy").mkdir()
        (tmp_path / "lispy" / "__init__.py").write_text("")
        (tmp_path / "lispy" / "tools.py").write_text("")
        assert self._target_in(tmp_path, "(require lispy.tools)") is None
        # With no such workspace module the installed .lpy files answer,
        # init.lpy standing for the package whose __init__.py re-exports it.
        empty = tmp_path / "empty"
        empty.mkdir()
        tools = self._target_in(empty, "(require lispy.tools)")
        assert tools == str(
            pathlib.Path(lispy.macros.__file__).parents[1] / "tools.lpy"
        )
        macros = self._target_in(empty, "(require lispy.macros)")
        assert macros == str(pathlib.Path(lispy.macros.__file__).parent / "init.lpy")

    def test_a_module_file_ends_the_lookup_of_a_dotted_name(self, tmp_path):
        # Defect: the parents of a dotted name are joined as directories, so
        # util/x.lpy is opened where Python imports the module util.lpy and
        # lspnsutil.x is no module at all.
        (tmp_path / "lspnsutil").mkdir()
        (tmp_path / "lspnsutil" / "x.lpy").write_text("")
        (tmp_path / "lspnsutil.lpy").write_text("")
        assert self._target_in(tmp_path, "(require lspnsutil.x)") is None
        # With no module file beside it the directory is a namespace package.
        (tmp_path / "lspnsutil.lpy").unlink()
        got = self._target_in(tmp_path, "(require lspnsutil.x)")
        assert got == str(tmp_path / "lspnsutil" / "x.lpy")

    def test_an_extension_module_wins_over_a_source_file(self, tmp_path):
        # Defect: the .lpy is tried first, where FileFinder imports the
        # extension module of its name.
        (tmp_path / "lspdual.lpy").write_text("")
        (tmp_path / f"lspdual{importlib.machinery.EXTENSION_SUFFIXES[0]}").write_text(
            ""
        )
        assert self._target_in(tmp_path, "(require lspdual)") is None

    @pytest.mark.parametrize("form", ['(require "lispy.tools")', "(import [a])"])
    def test_a_module_that_is_no_symbol_resolves_to_nothing(self, tmp_path, form):
        assert self._target_in(tmp_path, form) is None

    def test_namespace_portions_merge_across_search_dirs(self, namespace):
        # Defect: the first directory holding the namespace package ends the
        # search, so a portion under a later one is never found.
        project = namespace.parent
        (namespace / "lspnsmerged").mkdir()
        (namespace / "lspnsmerged" / "a.lpy").write_text("")
        (project / "lspnsmerged").mkdir()
        (project / "lspnsmerged" / "b.lpy").write_text("")
        origin = str(namespace / "a.lpy")
        got = self._target_in(project, "(require lspnsmerged.b)", origin=origin)
        assert got == str(project / "lspnsmerged" / "b.lpy")

    def test_the_lookup_imports_no_parent_package(self, tmp_path, monkeypatch):
        # Defect: an installed dotted name is found through
        # importlib.util.find_spec, which imports (runs) its parent package
        # in the server's process.
        site = tmp_path / "site"
        (site / "lspsideeffect").mkdir(parents=True)
        (site / "lspsideeffect" / "__init__.py").write_text("")
        (site / "lspsideeffect" / "mod.lpy").write_text("")
        monkeypatch.setattr(sys, "path", [*sys.path, str(site)])
        workspace = tmp_path / "ws"
        workspace.mkdir()
        try:
            got = self._target_in(workspace, "(require lspsideeffect.mod)")
            assert "lspsideeffect" not in sys.modules
        finally:
            sys.modules.pop("lspsideeffect", None)
        assert got == str(site / "lspsideeffect" / "mod.lpy")

    def test_indexing_a_require_imports_no_parent_package(self, tmp_path, monkeypatch):
        # Defect: indexing a file finds each required package through
        # importlib.util.find_spec, which imports (runs) its parent package
        # in the server's process.
        monkeypatch.setattr(server, "WORKSPACE_INDEX", {})
        monkeypatch.setattr(server, "FILE_TREES", {})
        monkeypatch.setattr(server, "WORKSPACE_SYMBOLS", {})
        monkeypatch.setattr(server, "workspace_folders", lambda: [])
        site = tmp_path / "site"
        macs = site / "lspindexside" / "macs"
        macs.mkdir(parents=True)
        (site / "lspindexside" / "__init__.py").write_text("")
        (macs / "__init__.lpy").write_text(self.MACROS)
        monkeypatch.setattr(sys, "path", [*sys.path, str(site)])
        target = tmp_path / "target.lpy"
        target.write_text("(require lspindexside.macs)\n")
        try:
            server.index_file(target.as_uri())
            assert "lspindexside" not in sys.modules
        finally:
            sys.modules.pop("lspindexside", None)
        assert "thrice" in server.WORKSPACE_INDEX

    def test_a_meta_path_finder_finds_a_module(self, tmp_path, monkeypatch):
        # Defect: only the path-based lookup is modelled, so a module an
        # editable install maps through its sys.meta_path finder, which
        # Python imports, resolves to nothing; or the finder's package is
        # not walked into for a dotted name.
        elsewhere = tmp_path / "elsewhere"
        (elsewhere / "lspedpkg").mkdir(parents=True)
        (elsewhere / "lspedmod.lpy").write_text("")
        (elsewhere / "lspedpkg" / "__init__.py").write_text("")
        (elsewhere / "lspedpkg" / "mac.lpy").write_text("")

        class EditableFinder:
            @staticmethod
            def find_spec(name, path=None, target=None):
                if name == "lspedmod":
                    origin = elsewhere / "lspedmod.lpy"
                    return importlib.util.spec_from_file_location(name, origin)
                if name == "lspedpkg":
                    package = elsewhere / "lspedpkg"
                    return importlib.util.spec_from_file_location(
                        name,
                        package / "__init__.py",
                        submodule_search_locations=[str(package)],
                    )
                return None

        monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, EditableFinder])
        workspace = tmp_path / "ws"
        workspace.mkdir()
        got = self._target_in(workspace, "(require lspedmod)")
        assert got == str(elsewhere / "lspedmod.lpy")
        got = self._target_in(workspace, "(require lspedpkg.mac)")
        assert got == str(elsewhere / "lspedpkg" / "mac.lpy")

    def test_a_finder_before_the_path_lookup_is_never_asked(
        self, tmp_path, monkeypatch
    ):
        # Defect: every sys.meta_path finder is asked, so an import hook
        # placed before PathFinder (setuptools' _distutils_hack imports
        # setuptools, or drops distutils from sys.modules, when asked) acts
        # on a go-to-definition.
        asked = []

        class EarlyHook:
            @staticmethod
            def find_spec(name, path=None, target=None):
                # Only the lookup's own names: the test's process may
                # import other modules meanwhile.
                if name.startswith("lspearly"):
                    asked.append(name)
                return None

        monkeypatch.setattr(sys, "meta_path", [EarlyHook, *sys.meta_path])
        (tmp_path / "lspearly").mkdir()
        (tmp_path / "lspearly" / "mac.lpy").write_text("")
        got = self._target_in(tmp_path, "(require lspearly.mac)")
        assert got == str(tmp_path / "lspearly" / "mac.lpy")
        assert self._target_in(tmp_path, "(require lspearlymissing)") is None
        assert asked == []

    def test_a_path_hook_entry_serves_a_namespace_package(self, tmp_path, monkeypatch):
        # Defect: a sys.path entry that is no directory is skipped, so the
        # namespace package an editable install serves from its path-hook
        # entry (setuptools' __editable__...__path_hook__) is not found, and
        # neither is a module under it.
        sub = tmp_path / "src" / "lspedns" / "sub"
        sub.mkdir(parents=True)
        (sub / "__init__.py").write_text("")
        (sub / "mac.lpy").write_text("")
        hook_entry = "__editable__.lspdemo-0.1.finder.__path_hook__"

        class EditableFinder:
            @staticmethod
            def find_spec(name, path=None, target=None):
                if name == "lspedns.sub":
                    return importlib.util.spec_from_file_location(
                        name, sub / "__init__.py", submodule_search_locations=[str(sub)]
                    )
                return None

        class NamespaceFinder:
            @staticmethod
            def hook(entry):
                if entry == hook_entry:
                    return NamespaceFinder
                raise ImportError

            @staticmethod
            def find_spec(name, target=None):
                if name == "lspedns":
                    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
                    spec.submodule_search_locations = [hook_entry]
                    return spec
                return None

        monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, EditableFinder])
        monkeypatch.setattr(sys, "path_hooks", [*sys.path_hooks, NamespaceFinder.hook])
        monkeypatch.setattr(sys, "path", [*sys.path, hook_entry])
        workspace = tmp_path / "ws"
        workspace.mkdir()
        got = self._target_in(workspace, "(require lspedns.sub.mac)")
        assert got == str(sub / "mac.lpy")

    def test_a_built_in_module_hides_a_file_of_its_name(self, tmp_path):
        # Defect: a workspace sys.lpy is taken where Python imports the
        # built-in module sys, which no file holds.
        (tmp_path / "sys.lpy").write_text("")
        assert self._target_in(tmp_path, "(require sys)") is None

    def test_a_working_directory_that_is_gone_is_skipped(self, tmp_path, monkeypatch):
        # Defect: the '' entry of sys.path is read through os.getcwd, which
        # raises once the directory is gone, out of the definition handler.
        # The module lies on sys.path past '', so the lookup reaches it.
        gone = tmp_path / "gone"
        gone.mkdir()
        site = tmp_path / "site"
        site.mkdir()
        (site / "lspgonemod.lpy").write_text("")
        workspace = tmp_path / "ws"
        workspace.mkdir()
        monkeypatch.chdir(gone)
        gone.rmdir()
        monkeypatch.setattr(sys, "path", ["", *sys.path, str(site)])
        got = self._target_in(workspace, "(require lspgonemod)")
        assert got == str(site / "lspgonemod.lpy")


def test_local_completion_tags_special_forms_in_label_details_not_detail():
    # Defect: the no-backend path drops the tag from labelDetails (the client
    # advertises nothing, as eglot) or leaves it in detail.
    items = {i.label: i for i in run_completion(ls_with("(i"), position(0, 2)).items}
    assert items["ife"].label_details.description == "special form"
    assert items["ife"].detail is None


def test_general_completion_offers_every_built_in_macro_and_no_phantoms():
    items = server.general_items(SimpleNamespace(source="", uri=URI))
    macros = {
        i.label
        for i in items
        if i.label_details and i.label_details.description == "macro"
    }
    assert macros == set(lispy.macros.__macro_namespace)
    # `gensym` is a function inside defmacro bodies, not a macro.
    gensym = next(i for i in items if i.label == "gensym")
    assert gensym.label_details.description == "function"


def test_built_in_macro_is_tagged_macro_and_special_form_keeps_its_tag():
    # Defect: built-ins read "keyword" beside project macros' "macro".
    items = {i.label: i for i in server.lispython_items()}
    assert items["when"].label_details.description == "macro"
    assert items["when"].kind == lsp.CompletionItemKind.Function
    assert items["ife"].label_details.description == "special form"
    assert items["ife"].kind == lsp.CompletionItemKind.Keyword


def test_completion_skips_reader_syntax_doc_keys_that_hover_keeps():
    # Defect: "f-string" and the quote characters were offered as heads.
    labels = {i.label for i in server.lispython_items()}
    assert not {"f-string", "'", "`", "~@"} & labels
    assert {"def", ",", ".", "~"} <= labels
    assert {"f-string", "'", "`", "~@"} <= set(server.SPECIAL_FORM_DOCS)


# Hover usage text for statement forms comes from the compiler's FORM_USAGE,
# the table its error messages quote, so the two cannot drift.
FORM_KEYS = sorted(k for k in FORM_USAGE if k not in HEADLESS_FORMS)


def test_hand_written_docs_do_not_shadow_form_usage():
    assert not (set(form_docs.HAND_WRITTEN_DOCS) & set(FORM_USAGE))
    form_docs.refuse_shadowing(form_docs.HAND_WRITTEN_DOCS, FORM_USAGE)


def test_a_hand_written_entry_shadowing_form_usage_is_refused():
    with pytest.raises(RuntimeError, match="shadows FORM_USAGE entries: .*'del'"):
        form_docs.refuse_shadowing({"del": "Delete."}, {"del": "(del x)"})


@pytest.mark.parametrize("key", FORM_KEYS)
def test_form_hover_usage_is_the_compilers(key):
    doc = server.SPECIAL_FORM_DOCS[key]
    lines = re.search(r"```\n(.+?)\n```", doc, re.DOTALL).group(1).split("\n")
    assert lines.count(FORM_USAGE[key]) == 1
    others = [line for line in lines if line != FORM_USAGE[key]]
    assert others == (
        [form_docs.EXTRA_USAGE[key]] if key in form_docs.EXTRA_USAGE else []
    )


def test_form_prose_must_match_form_usage():
    with pytest.raises(RuntimeError, match="no prose for .*'del'"):
        form_docs.form_docs({"del": "(del x)"}, {}, {})
    with pytest.raises(RuntimeError, match="no usage for .*'zap'"):
        form_docs.form_docs({}, {"zap": "Zap."}, {})


def test_an_extra_usage_key_without_form_usage_is_refused():
    with pytest.raises(RuntimeError, match="EXTRA-USAGE.*'zap'"):
        form_docs.form_docs({"del": "(del x)"}, {"del": "Del."}, {"zap": "(zap)"})


def test_def_hover_shows_the_compilers_arguments():
    hover = run_hover(ls_with("(def f [x] x)"), position(0, 2))
    assert "(def name [args] body...)" in hover.contents.value
    assert "[params]" not in hover.contents.value


def test_or_operator_and_or_pattern_share_one_entry():
    doc = server.SPECIAL_FORM_DOCS["|"]
    block = re.search(r"```\n(.+?)\n```", doc, re.DOTALL).group(1)
    assert block == "(| a b ...)\n" + FORM_USAGE["|"]
    assert "or-pattern" in doc


def test_an_undocumented_statement_head_is_refused():
    with pytest.raises(RuntimeError, match="no form docs: .*'zap'"):
        form_docs.refuse_undocumented({"zap"}, {})


class TestBuiltinMacroHover:
    """Hover on a built-in macro shows its docstring where the buffer has
    that macro by the compiler's rule: the prelude unless the first form opts
    out with (require lispy.macros ...), plus any require."""

    @staticmethod
    def _hover(source, line, character):
        hover = run_hover(ls_with(source), position(line, character))
        return None if hover is None else hover.contents.value

    def test_hover_shows_the_docstring_usage_and_prose(self):
        text = self._hover("(require lispy.macros *)\n(cond x 1 y 2)", 1, 2)
        usage = re.search(r"```\n(.+?)\n```", text, re.DOTALL).group(1)
        assert usage == "(cond test then test then ... else)"
        assert "*(built-in macro)*" in text
        assert "`conde`" in text

    def test_qualified_require_names_the_macro_by_module(self):
        source = "(require lispy.macros)\n(lispy.macros.inc x)"
        assert "(inc value)" in self._hover(source, 1, 3)

    def test_a_buffer_with_no_require_has_the_prelude(self):
        assert "(inc value)" in self._hover("(inc x)", 0, 2)

    @pytest.mark.parametrize(
        "source",
        [
            "(require lispy.macros [when])\n(cond x 1 y 2)",
            '"Docs."\n(require lispy.macros [when])\n(cond x 1 y 2)',
        ],
    )
    def test_an_opt_out_keeps_only_its_selection(self, source):
        line = source.count("\n")
        assert self._hover(source, line, 2) is None
        assert "(when test" in server.builtin_macro_doc(parse(source), "when")

    @pytest.mark.parametrize(
        "source",
        [
            "(import os)\n(require lispy.macros [when])\n(cond x 1 y 2)",
            '"A."\n"B."\n(require lispy.macros [when])\n(cond x 1 y 2)',
            "(require lispy.macros.sugar [when])\n(cond x 1 y 2)",
        ],
    )
    def test_a_require_that_is_no_opt_out_keeps_the_prelude(self, source):
        # Only lispy.macros itself, as the first form after one docstring,
        # replaces the prelude (compile_unit's rule).
        line = source.count("\n")
        assert "(cond test" in self._hover(source, line, 2)


class TestBuiltinMacroDocsOverTheView:
    """With the buffer's full view compiled (the one a change's diagnostics
    cache), hover and signature help show a built-in macro's docs only for
    a name the compile bound to that very macro."""

    @pytest.fixture
    def compiled(self, monkeypatch):
        """compiled(marked) -> (hover text or None, signature label or None)
        at the | of marked, its version's view compiled first."""
        monkeypatch.setattr(server, "view_cache", server.ViewCache())

        def run(marked):
            [before, after] = marked.split("|")
            source = before + after
            assert server.compile_document(URI, 1, source).view is not None
            ls = ls_with(source)
            ls.workspace.get_text_document(URI).version = 1
            lines = before.split("\n")
            params = position(len(lines) - 1, len(lines[-1]))
            hover = asyncio.run(server.hover(ls, params))
            help = asyncio.run(server.signature_help(ls, params))
            return (
                None if hover is None else hover.contents.value,
                None if help is None else help.signatures[0].label,
            )

        return run

    def test_a_defmacro_shadowing_a_built_in_shows_no_built_in_docs(self, compiled):
        # Defect: the syntax scan sees no opt-out, so the buffer's own inc
        # gets the built-in inc's docstring and usage.
        hover, label = compiled('(defmacro inc [x] "my doc" (return x))\n(in|c 1)')
        assert hover is None or "built-in macro" not in hover
        assert label != "(inc value)"

    def test_a_built_in_macro_keeps_its_docs(self, compiled):
        # Defect: the view's macro table misses the built-in (it is not
        # told by identity), so no built-in macro has docs once compiled.
        hover, label = compiled("(in|c 1)")
        assert "(inc value)" in hover
        assert label == "(inc value)"

    def test_an_opt_out_keeps_only_its_selection(self, compiled):
        hover, _ = compiled("(require lispy.macros [when])\n(co|nd x 1 y 2)")
        assert hover is None
        hover, _ = compiled("(require lispy.macros [when])\n(wh|en x 1)")
        assert "(when test" in hover


def test_headless_forms_are_no_hover_or_completion_entry():
    # A headless kind ("method call", "slice") is no word a user types.
    assert not (set(HEADLESS_FORMS) & set(server.SPECIAL_FORM_DOCS))
    assert ".method" not in server.SPECIAL_FORM_DOCS


class TestImportBindings:
    """The definition index and go-to-definition bind exactly the names
    Python binds for an import (import_bindings), keyed by source spelling."""

    BOUND = {
        "(import os.path as p)": {"p"},
        "(import os.path)": {"os"},
        "(from os.path [join as pjoin])": {"pjoin"},
        "(from os.path [join] my-mod *)": {"join"},
        "(from my-mod [a-b as c-d e-f])": {"c-d", "e-f"},
    }
    REFUSED = [
        "(import os.path :as p)",
        "(from os.path [join :as pjoin])",
        "(import)",
        "(from os.path)",
        "(import (f) as p)",
        "(from os.path [join as])",
    ]

    @pytest.mark.parametrize("source", sorted(BOUND))
    def test_index_holds_exactly_the_bound_names(self, source):
        defs = server.collect_definitions(parse(source), URI)
        assert set(defs) == self.BOUND[source]

    @pytest.mark.parametrize("source", sorted(BOUND))
    def test_find_import_for_name_agrees_with_the_index(self, source):
        tree = parse(source)
        candidates = {"os", "os.path", "p", "join", "pjoin", "my-mod"}
        candidates |= {"a-b", "c-d", "e-f", "a_b", "c_d"}
        found = {n for n in candidates if server.find_import_for_name(tree, n)}
        assert found == self.BOUND[source]

    def test_an_alias_is_located_at_the_alias(self):
        [loc] = server.collect_definitions(parse("(import os.path as p)"), URI)["p"]
        assert (loc.range.start.character, loc.range.end.character) == (19, 20)

    @pytest.mark.parametrize("source", REFUSED)
    def test_a_form_the_compiler_refuses_binds_nothing(self, source):
        tree = parse(source, partial=True)
        assert server.collect_definitions(tree, URI) == {}
        assert server.find_import_for_name(tree, "p") is None
        assert server.find_import_for_name(tree, "pjoin") is None

    def test_lazy_import_binds_on_a_python_without_it(self, monkeypatch):
        monkeypatch.setattr(features, "running_version", lambda: (3, 11, 13))
        tree = parse("(lazy-import os.path as p)\n(lazy-from os [sep])", partial=True)
        assert set(server.collect_definitions(tree, URI)) == {"p", "sep"}
        assert server.find_import_for_name(tree, "sep") is not None


def local_references(source, line, character):
    """(line, character) of each local reference at the position (no backend
    serves URI)."""
    from lsprotocol import types as lsp

    params = lsp.ReferenceParams(
        text_document=lsp.TextDocumentIdentifier(uri=URI),
        position=lsp.Position(line=line, character=character),
        context=lsp.ReferenceContext(include_declaration=True),
    )
    found = asyncio.run(server.references(ls_with(source), params))
    return [(loc.range.start.line, loc.range.start.character) for loc in found]


def test_local_references_compare_python_names():
    # Defect: references compare spellings, so my_x misses the my-x it
    # stands for (one Python name).
    source = "(= my-x 1)\n(print my_x)\n(print my-y)"
    assert local_references(source, 1, 8) == [(0, 3), (1, 7)]


def test_an_operator_is_never_the_name_its_mangling_gives():
    # Defect: references compare Python names, and - mangles to _, so
    # references on - list every _ and those on _ list the -.
    source = "(= _ 1)\n(print (- 3 _))"
    assert local_references(source, 1, 8) == [(1, 8)]
    assert local_references(source, 0, 3) == [(0, 3), (1, 12)]


def test_an_operator_never_spells_a_name():
    # Defect: (- a b) records - as the spelling of _ (py-name of -), so a
    # backend member _ is relabelled "-".  ? is a name (the compiler binds
    # it as _p), so it does spell _p.
    spellings = server.buffer_spellings("(- a b)\n(? c)\n(= x._ x._p)")
    assert (spellings.get("_", "_"), spellings["_p"]) == ("_", "?")


# w at line 1 columns 10 (the field's value) and 13 (its format spec's field).
FIELD_SPEC = '(= w 4)\n(print f"{w:{w}}")'


@pytest.mark.parametrize("character", [3, 10, 13], ids=["def", "field", "spec"])
def test_references_include_a_format_specs_fields(character):
    # Defect: the walk reads a field's value only, never its format spec,
    # and the f-string's head spans the whole literal, so a name in a field
    # was that head: the spec's w was missed, or the references were the
    # f-string form's.
    assert local_references(FIELD_SPEC, 0 if character == 3 else 1, character) == [
        (0, 3),
        (1, 10),
        (1, 13),
    ]


@pytest.mark.parametrize("character", [10, 13], ids=["field", "spec"])
def test_definition_of_a_name_in_an_f_string_field(character):
    # Defect: the cursor's node was the f-string's head, whose span is the
    # whole literal, so a name in a field had no definition.
    [location] = run_definition(ls_with(FIELD_SPEC), position(1, character))
    assert (location.range.start.line, location.range.start.character) == (0, 3)


def test_hover_inside_a_format_spec_reads_the_spec():
    # Defect: a name or form head in a format spec answered as the f-string
    # form, or a head there was no form head (form-heads skipped the spec).
    source = '(print f"{1:{len}}" f"{1:{(ife a b c)}}")'
    hover = run_hover(ls_with(source), position(0, 14))
    assert "Python builtin" in hover.contents.value
    hover = run_hover(ls_with(source), position(0, source.index("ife") + 1))
    assert hover.contents.value == server.SPECIAL_FORM_DOCS["ife"]


@pytest.mark.parametrize("character", [8, 9, 11, 14, 15, 18, 20, 22])
def test_hover_on_an_f_strings_own_text_is_the_form(character):
    # Defect: a cursor on the literal's text, braces or delimiters left the
    # head for a string part or the field, so there was no hover.
    # Columns: the quote, a, the field's {, the spec's > and {, the
    # field's closing }, c, the closing quote.
    source = '(print f"ab{x:>{w}} cd")'
    hover = run_hover(ls_with(source), position(0, character))
    assert hover.contents.value == server.SPECIAL_FORM_DOCS["f-string"]


def test_node_at_position_walks_any_depth():
    # Defect: one call per nesting level, so a deep buffer raised
    # RecursionError out of hover.
    # The RecursionError is caught here: pytest formatting its frames, each
    # holding a node whose repr walks the tree, would never finish.
    depth = 5000
    tree = parse("(" * depth + "x" + ")" * depth, partial=True)
    try:
        name = server.node_at_position(tree, 0, depth).value
    except RecursionError:
        name = "RecursionError"
    assert name == "x"


def test_completion_reads_the_view_generation_before_the_hand_off(
    monkeypatch, tmp_path
):
    # Defect: the cursor view's thread reads the cache generation itself,
    # so a document closed after the hand-off but before the thread starts
    # still gets its view stored.
    from pygls.workspace import TextDocument

    pv = importlib.import_module("lispy.lsp.python_view")
    uri = (tmp_path / "main.lpy").as_uri()
    monkeypatch.setattr(server, "view_cache", pv.ViewCache())
    compiled = []
    real_view_at_cursor = pv._view_at_cursor

    def counting(*args, **kwargs):
        compiled.append(args)
        return real_view_at_cursor(*args, **kwargs)

    monkeypatch.setattr(pv, "_view_at_cursor", counting)
    real_in_own_thread = server.in_own_thread

    def close_first(fn, *args):
        server.view_cache.forget(uri)  # the close lands before the worker
        return real_in_own_thread(fn, *args)

    async def no_items(*args, **kwargs):
        return []

    client = SimpleNamespace(backend_uri=lambda u: u, request=no_items)
    document = TextDocument(uri, "(import os)\nos.pa", version=1)
    monkeypatch.setattr(server, "in_own_thread", close_first)
    asyncio.run(server.backend_answer(client, document, 2, 5, None))
    monkeypatch.setattr(server, "in_own_thread", real_in_own_thread)
    asyncio.run(server.backend_answer(client, document, 2, 5, None))
    assert len(compiled) == 2  # the closed document's view was not kept


@pytest.mark.parametrize(
    ["encoding", "start", "end"],
    [("utf-16", 10, 16), ("utf-32", 9, 15), ("utf-8", 13, 19)],
)
def test_a_real_file_range_counts_in_the_editors_units(tmp_path, encoding, start, end):
    # Defect: the backend's UTF-16 columns over a real .py go unchanged to
    # an editor that negotiated another encoding, off by one per character
    # outside the BMP (and more in UTF-8).  "target" follows a two-unit
    # character on the file's second line.
    from lsprotocol import types as lsp
    from pygls.workspace import PositionCodec, TextDocument

    path = tmp_path / "real.py"
    path.write_text('x = 1\nπ = "𝒳"; target = 1\n', encoding="utf-8")
    doc = TextDocument(URI, "", position_codec=PositionCodec(encoding=encoding))
    utf16 = lsp.Range(
        start=lsp.Position(line=1, character=10), end=lsp.Position(line=1, character=16)
    )
    got = server.file_range(doc, path.as_uri(), utf16)
    assert (got.start.line, got.start.character, got.end.character) == (1, start, end)


@pytest.mark.parametrize(
    "line", ["(f -x)", "(f --x y.z)", "(g *args **kw :k)", "(h -1+2j a.b.c)"]
)
def test_the_cursor_word_is_the_word_its_view_restores(line):
    # Defect: on the - of -x the word is the whole token -x, but the view is
    # made with the cursor at the word's end, where the partial is the leaf x,
    # so the offset back over -x lands before x and the backend answers for
    # another name.
    doc = SimpleNamespace(source=line)
    for col in range(len(line) + 1):
        word = server.cursor_word(doc, 1, col)
        if word is None:
            continue
        start, end, _ = word
        assert (start, end) == pv.partial_span(line, end), col
    assert server.cursor_word(doc, 1, len(line) - 1) is not None


def test_a_hover_on_a_non_builtin_name_caches_nothing():
    # Defect: the builtin lookup's cache keeps an entry (None) for every
    # name hovered, so it grows with every name a session hovers.
    caches = [f for f in vars(server).values() if hasattr(f, "cache_info")]

    def cached():
        return sum(f.cache_info().currsize for f in caches)

    run_hover(ls_with("(print 1)"), position(0, 2))  # a builtin may be cached
    before = cached()
    assert run_hover(ls_with("(no-such-name 1)"), position(0, 3)) is None
    assert cached() == before


def test_a_dotted_name_compares_part_by_part():
    # Defect: a dotted symbol is no name spelling as a whole, so it is
    # compared as written and self.my_v misses the self.my-v it stands for.
    source = "(= self.my-v 1)\n(print self.my_v)"
    assert local_references(source, 1, 9) == [(0, 3), (1, 7)]


# ---------------------------------------------------------------------------
# Cursor-view compiles (scheduled-cursor-view): one per document at a time,
# never one for a request nobody waits for any more.
# ---------------------------------------------------------------------------


@pytest.fixture
def slow_views(monkeypatch):
    """cursor_view replaced by one that blocks until release is set (so a
    compile is in flight for as long as a test needs) and records the
    (version, col) of each compile and the most run at once."""
    record = SimpleNamespace(
        calls=[],
        running=0,
        most=0,
        started=threading.Event(),
        release=threading.Event(),
    )
    lock = threading.Lock()

    def cursor_view(uri, version, source, line, col, *rest):
        with lock:
            record.calls.append((version, col))
            record.running += 1
            record.most = max(record.most, record.running)
        record.started.set()
        try:
            record.release.wait(10)
            return SimpleNamespace(version=version, col=col)
        finally:
            with lock:
                record.running -= 1

    monkeypatch.setattr(server, "cursor_view", cursor_view)
    try:
        yield record
    finally:
        record.release.set()


def view_of(tmp_path, version, timeout=None, col=1):
    """backend_request's (view, answer) of a buffer at version and col,
    asking the backend nothing (make-params gives None)."""
    document = server.TextDocument(
        (tmp_path / "m.lpy").as_uri(), "(= x 1)\n", version=version
    )
    client = SimpleNamespace(backend_uri=lambda uri: uri)
    return server.backend_request(
        client, document, 1, col, timeout, "m", lambda *a: None
    )


async def until(condition, limit=5):
    """Poll condition on the loop until it holds, at most limit seconds."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit
    while not condition():
        assert loop.time() < deadline, "condition never held"
        await asyncio.sleep(0.005)


def test_abandoned_keystrokes_queue_no_cursor_view_compiles(slow_views, tmp_path):
    # Defect: every keystroke whose completion gave up waiting (MERGE-TIMEOUT)
    # still compiles its view, all at once in the executor, and the next
    # request that does wait is served only after every one of them.
    async def scenario():
        await view_of(tmp_path, 1, timeout=0.01, col=1)
        await until(slow_views.started.is_set)
        for col in range(2, 7):
            assert await view_of(tmp_path, 1, timeout=0.01, col=col) == (None, None)
        live = asyncio.ensure_future(view_of(tmp_path, 1, col=9))
        await asyncio.sleep(0)
        slow_views.release.set()
        result = await asyncio.wait_for(live, 10)
        await until(lambda: not server.CURSOR_COMPILES)
        return result

    view, answer = asyncio.run(scenario())
    assert (view.version, view.col) == (1, 9)
    assert slow_views.calls == [(1, 1), (1, 9)]
    assert slow_views.most == 1


def test_a_newer_version_supersedes_pending_cursor_views(slow_views, tmp_path):
    # Defects: a view of a version the buffer already left is compiled (its
    # macros run for an answer nobody can use); two callers of one view
    # compile it twice.
    async def scenario():
        asyncio.ensure_future(view_of(tmp_path, 1))
        await until(slow_views.started.is_set)
        stale = asyncio.ensure_future(view_of(tmp_path, 2))
        await asyncio.sleep(0)
        latest = [asyncio.ensure_future(view_of(tmp_path, 3)) for _ in range(2)]
        await asyncio.sleep(0)
        slow_views.release.set()
        result = await asyncio.wait_for(asyncio.gather(stale, *latest), 10)
        await until(lambda: not server.CURSOR_COMPILES)
        return result

    stale, *latest = asyncio.run(scenario())
    assert stale == (None, None)
    assert [view.version for view, _ in latest] == [3, 3]
    assert slow_views.calls == [(1, 1), (3, 1)]


def test_a_cursor_view_that_never_returns_stops_holding_up_its_document(
    monkeypatch, tmp_path
):
    # Defect: a cursor view whose compile never returns (a macro looping
    # forever) keeps every later request of its document waiting behind it.
    started, release = threading.Event(), threading.Event()

    def cursor_view(uri, version, source, line, col, *rest):
        if version == 1:
            started.set()
            release.wait(10)
        return SimpleNamespace(version=version)

    monkeypatch.setattr(server, "cursor_view", cursor_view)
    monkeypatch.setattr(macro_modules, "WAIT_LIMIT", 0.1)

    async def scenario():
        first = asyncio.ensure_future(view_of(tmp_path, 1, timeout=5))
        await until(started.is_set)
        second = await view_of(tmp_path, 2, timeout=3)
        return await first, second

    try:
        (first, _), (second, _) = asyncio.run(scenario())
    finally:
        release.set()
    assert first is None
    assert second.version == 2


def test_a_request_with_no_timeout_still_has_a_deadline(
    slow_views, monkeypatch, tmp_path
):
    # Defect: hover and signature help (timeout None) wait for a cursor
    # view that never comes, forever.
    monkeypatch.setattr(server, "REQUEST_TIMEOUT", 0.2)

    async def scenario():
        return await asyncio.wait_for(view_of(tmp_path, 1), 3)

    assert asyncio.run(scenario()) == (None, None)


def test_given_up_compiles_still_running_are_capped(monkeypatch):
    # Defect: every compile given up on leaves its thread spinning in the
    # macro, so a user typing over a macro that never returns piles up
    # threads without bound; or a parked thread that ends still counts, so
    # the cap holds every later hung compile for good.
    monkeypatch.setattr(server, "MAX_PARKED_COMPILES", 1)
    monkeypatch.setattr(server, "PARKED_COMPILES", server._ParkedCompiles())
    monkeypatch.setattr(macro_modules, "WAIT_LIMIT", 0.05)
    parked, held = threading.Event(), threading.Event()

    async def scenario():
        loop = asyncio.get_running_loop()
        superseded = asyncio.Event()
        superseded.set()
        first = server.in_own_thread(parked.wait, 10)
        given_up = await server.compile_or_give_up(first, superseded, loop.time())
        assert given_up is server.GAVE_UP
        second = asyncio.ensure_future(
            server.compile_or_give_up(
                server.in_own_thread(held.wait, 10), superseded, loop.time()
            )
        )
        await asyncio.sleep(0.3)  # several WAIT_LIMITs
        assert not second.done()  # held at the cap
        parked.set()  # the parked thread ends
        return await asyncio.wait_for(second, 3)

    try:
        assert asyncio.run(scenario()) is server.GAVE_UP
    finally:
        parked.set()
        held.set()


def test_a_cursor_view_error_reaches_its_caller(monkeypatch, tmp_path):
    # Defect: an exception in the compile thread never reaches the waiter,
    # so the request hangs instead of failing into the handler's except.
    def cursor_view(*args):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(server, "cursor_view", cursor_view)

    async def scenario():
        with pytest.raises(RuntimeError, match="compile failed"):
            await asyncio.wait_for(view_of(tmp_path, 1), 10)
        await until(lambda: not server.CURSOR_COMPILES)

    asyncio.run(scenario())


def cursor_view_references(path):
    """The nearest enclosing def's name (None at top level) of every symbol
    in the lispython source at path whose last dotted part is the name
    cursor-view (by name_key, so cursor_view and server.cursor-view count)."""
    stack = [(form, None) for form in parse(path.read_text())]
    found = []
    while stack:
        node, def_name = stack.pop()
        if isinstance(node, nodes.Paren) and len(node) > 1:
            if nodes.atom_text(node[0]) in ("def", "async-def"):
                def_name = nodes.atom_text(node[1])
        if isinstance(node, nodes.Symbol):
            # any spelling of the name: cursor_view, server.cursor-view, ...
            if literal.name_key(
                nodes.atom_text(node).rsplit(".", 1)[-1]
            ) == literal.name_key("cursor-view"):
                found.append(def_name)
        elif isinstance(node, nodes.Expression):
            stack.extend((child, def_name) for child in node)
        elif isinstance(node, nodes.Wrapper):
            stack.append((node.value, def_name))
    return found


def test_cursor_view_is_compiled_only_through_the_scheduler():
    # Defect: a handler running cursor-view in a thread directly brings
    # back one abandoned compile per keystroke, unbounded.
    lsp = pathlib.Path(server.__file__).parent
    sites = sorted(
        (path.name, def_name)
        for path in lsp.glob("*.lpy")
        for def_name in cursor_view_references(path)
    )
    assert sites == [
        ("server.lpy", "cursor-view"),
        ("server.lpy", "run-cursor-compiles"),
    ]


def test_completion_on_an_unreadable_document_gives_the_general_list(tmp_path):
    # Defect: a document that is not open is read from disk by pygls, and a
    # missing file raises out of the handler instead of answering.
    uri = (tmp_path / "missing.lpy").as_uri()
    document = server.TextDocument(uri)
    ls = SimpleNamespace(
        workspace=SimpleNamespace(get_text_document=lambda uri: document, folders={})
    )
    params = SimpleNamespace(
        text_document=SimpleNamespace(uri=uri),
        position=SimpleNamespace(line=0, character=0),
    )
    labels = {item.label for item in run_completion(ls, params).items}
    assert {"ife", "print"} <= labels
