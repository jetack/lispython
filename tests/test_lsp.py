"""The LSP keeps working on a buffer that is mid-edit.

Handlers read the buffer with parse(..., partial=True): the complete forms
before the first error, no version gates.  Diagnostics still use the strict
parse, so the error itself is reported.
"""

import importlib
import re
import sys
from types import SimpleNamespace

import pytest

from lispy.core.compiler import features
from lispy.core.compiler.stmt import FORM_USAGE, PATTERN_KINDS
from lispy.core.parser import parse

# lispy.lsp re-exports the server *object* as ``server``; take the module.
server = importlib.import_module("lispy.lsp.server")

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


def test_definition_hover_and_references(buffer):
    ls = ls_with(buffer)
    [location] = server.definition(ls, position(1, 14))
    assert location.range.start.line == 0
    refs = server.references(ls, position(0, 5))
    assert [r.range.start.line for r in refs][:2] == [0, 1]
    assert server.hover(ls, position(0, 1)) is not None


def test_completion(buffer):
    items = server.completion(ls_with(buffer), position(2, 2))
    labels = {item.label for item in items.items}
    assert {"foo", "bar"} <= labels


def test_diagnostics_still_report_the_error(buffer):
    [diagnostic] = server.compute_diagnostics(buffer, URI)
    assert diagnostic.message.startswith("Parse error:")
    assert diagnostic.range.start.line == 2


def test_partial_parse_does_not_leak_into_strict_parse(monkeypatch):
    monkeypatch.setattr(features, "running_version", lambda: (3, 11, 13))
    parse(BUFFERS["t-string on an old Python"], partial=True)
    with pytest.raises(Exception):
        parse('(print t"x")')


def test_refused_prefix_form_still_closes_in_the_call_scan():
    # ^*( is a bad-prefix token, not an opening; its ) must close it, not g.
    source = "(g ^*(f) x)"
    assert server.find_enclosing_function(source, 1, source.index("x")) == "g"


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
    [pair] = server.definition(ls, position(2, 5))
    assert (pair.range.start.line, pair.range.start.character) == (1, 6)
    [alias] = server.definition(ls, position(3, 5))
    assert alias.range.start.line == 0


def test_type_alias_hover_on_the_keyword():
    hover = server.hover(ls_with("(type Alias int)"), position(0, 2))
    assert "type alias" in hover.contents.value


def test_type_call_keeps_the_builtin_hover():
    hover = server.hover(ls_with("(type x)\n(type A int)"), position(0, 2))
    assert "Python builtin" in hover.contents.value


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
        return server.definition(ls, position(line, character))

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
        "form", ["(require m foo)", "(require m [a] b)", '(require m "ab")', "(require m :as x)"]
    )
    def test_malformed_require_binds_nothing_and_does_not_raise(self, form):
        assert self._find(form + "\n(print 1)", "print") is None


# Hover usage text for statement forms comes from the compiler's FORM_USAGE,
# the table its error messages quote, so the two cannot drift.
FORM_KEYS = sorted(k for k in FORM_USAGE if k not in PATTERN_KINDS)


@pytest.mark.parametrize("key", FORM_KEYS)
def test_form_hover_usage_is_the_compilers(key):
    doc = server.SPECIAL_FORM_DOCS[key]
    assert doc.startswith("```\n" + FORM_USAGE[key] + "\n```\n\n")
    assert re.search(r"```\n(.+?)\n```", doc, re.DOTALL).group(1) == FORM_USAGE[key]


def test_form_prose_must_match_form_usage():
    with pytest.raises(RuntimeError, match="no prose for .*'del'"):
        server.form_docs({"del": "(del x)"}, {})
    with pytest.raises(RuntimeError, match="no usage for .*'zap'"):
        server.form_docs({}, {"zap": "Zap."})


def test_def_hover_shows_the_compilers_arguments():
    hover = server.hover(ls_with("(def f [x] x)"), position(0, 2))
    assert "(def name [args] body...)" in hover.contents.value
    assert "[params]" not in hover.contents.value


def test_or_operator_and_or_pattern_share_one_entry():
    assert "or-pattern" in server.SPECIAL_FORM_DOCS["|"]
