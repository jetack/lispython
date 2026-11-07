"""The LSP keeps working on a buffer that is mid-edit.

Handlers read the buffer with parse(..., partial=True): the complete forms
before the first error, no version gates.  Diagnostics still use the strict
parse, so the error itself is reported.
"""

import asyncio
import importlib
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
from lispy.core.parser import parse

# lispy.lsp re-exports the server *object* as ``server``; take the module.
server = importlib.import_module("lispy.lsp.server")
macro_modules = importlib.import_module("lispy.lsp.macro_modules")

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


def test_completion(buffer):
    items = run_completion(ls_with(buffer), position(2, 2))
    labels = {item.label for item in items.items}
    assert {"foo", "bar"} <= labels


def test_diagnostics_still_report_the_error(buffer):
    [diagnostic] = server.compute_diagnostics(buffer, URI)
    assert diagnostic.message.startswith("Parse error:")
    assert diagnostic.range.start.line == 2


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
    # backend member _ is relabelled "-"; ? (py-name _p) likewise.
    spellings = server.buffer_spellings("(- a b)\n(? c)\n(= x._ x._p)")
    assert (spellings.get("_", "_"), spellings["_p"]) == ("_", "_p")


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
