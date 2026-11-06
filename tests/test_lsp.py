"""The LSP keeps working on a buffer that is mid-edit.

Handlers read the buffer with parse(..., partial=True): the complete forms
before the first error, no version gates.  Diagnostics still use the strict
parse, so the error itself is reported.
"""

import asyncio
import importlib
import re
import sys
from types import SimpleNamespace

import pytest

from lispy.core.compiler import features
from lispy.core.compiler.utils import FORM_USAGE, HEADLESS_FORMS
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


def run_completion(ls, params):
    """The completion handler's answer (it is a coroutine)."""
    return asyncio.run(server.completion(ls, params))


def run_hover(ls, params):
    """The hover handler's answer (it is a coroutine)."""
    return asyncio.run(server.hover(ls, params))


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
    [location] = server.definition(ls, position(1, 14))
    assert location.range.start.line == 0
    refs = server.references(ls, position(0, 5))
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
        "form",
        [
            "(require m foo)",
            "(require m [a] b)",
            '(require m "ab")',
            "(require m :as x)",
        ],
    )
    def test_malformed_require_binds_nothing_and_does_not_raise(self, form):
        assert self._find(form + "\n(print 1)", "print") is None


# Hover usage text for statement forms comes from the compiler's FORM_USAGE,
# the table its error messages quote, so the two cannot drift.
FORM_KEYS = sorted(k for k in FORM_USAGE if k not in HEADLESS_FORMS)


def test_hand_written_docs_do_not_shadow_form_usage():
    assert not (set(server.HAND_WRITTEN_DOCS) & set(FORM_USAGE))
    server.refuse_shadowing(server.HAND_WRITTEN_DOCS, FORM_USAGE)


def test_a_hand_written_entry_shadowing_form_usage_is_refused():
    with pytest.raises(RuntimeError, match="shadows FORM_USAGE entries: .*'del'"):
        server.refuse_shadowing({"del": "Delete."}, {"del": "(del x)"})


@pytest.mark.parametrize("key", FORM_KEYS)
def test_form_hover_usage_is_the_compilers(key):
    doc = server.SPECIAL_FORM_DOCS[key]
    lines = re.search(r"```\n(.+?)\n```", doc, re.DOTALL).group(1).split("\n")
    assert lines.count(FORM_USAGE[key]) == 1
    others = [line for line in lines if line != FORM_USAGE[key]]
    assert others == ([server.EXTRA_USAGE[key]] if key in server.EXTRA_USAGE else [])


def test_form_prose_must_match_form_usage():
    with pytest.raises(RuntimeError, match="no prose for .*'del'"):
        server.form_docs({"del": "(del x)"}, {}, {})
    with pytest.raises(RuntimeError, match="no usage for .*'zap'"):
        server.form_docs({}, {"zap": "Zap."}, {})


def test_an_extra_usage_key_without_form_usage_is_refused():
    with pytest.raises(RuntimeError, match="EXTRA-USAGE.*'zap'"):
        server.form_docs({"del": "(del x)"}, {"del": "Del."}, {"zap": "(zap)"})


def test_def_hover_shows_the_compilers_arguments():
    hover = run_hover(ls_with("(def f [x] x)"), position(0, 2))
    assert "(def name [args] body...)" in hover.contents.value
    assert "[params]" not in hover.contents.value


def test_or_operator_and_or_pattern_share_one_entry():
    doc = server.SPECIAL_FORM_DOCS["|"]
    block = re.search(r"```\n(.+?)\n```", doc, re.DOTALL).group(1)
    assert block == "(| a b ...)\n" + FORM_USAGE["|"]
    assert "or-pattern" in doc


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


class TestMemberCompletion:
    """After a method receiver's dot, completion offers the class's members.

    ``|`` in a source marks the cursor; it is removed before completing.
    """

    BASE = "(class Base []\n  (= inherited 1))\n"
    CLASS = (
        BASE + "(class A [Base]\n"
        "  (= class-var 1)\n"
        "  (def __init__ [self]\n"
        "    (= self.x 1)\n"
        "    (= (. self y) 2)\n"
        "    (+= self.count 1)\n"
        "    (= self.z ^int))\n"
        "  (def before [self] 1)\n"
        "  (def here [self]\n"
        "    {})\n"
        "  (def after [self] 1))\n"
    )
    MEMBERS = {
        "inherited", "class-var", "__init__", "x", "y", "count", "z",
        "before", "here", "after",
    }  # fmt: skip

    @staticmethod
    def complete(source):
        before, after = source.split("|")
        line = before.count("\n")
        character = len(before) - (before.rfind("\n") + 1)
        result = run_completion(ls_with(before + after), position(line, character))
        return line, character, result.items

    def labels(self, source):
        return {item.label for item in self.complete(source)[2]}

    @pytest.mark.parametrize(
        "body, typed",
        [("self.|", 0), ("self.ba|", 2), ("(. self |)", 0), ("(. self ba|)", 2)],
    )
    def test_receiver_dot_offers_exactly_the_class_members(self, body, typed):
        line, character, items = self.complete(self.CLASS.format(body))
        assert {item.label for item in items} == self.MEMBERS
        for item in items:
            start, end = item.text_edit.range.start, item.text_edit.range.end
            assert (start.line, start.character) == (line, character - typed)
            assert (end.line, end.character) == (line, character)
            assert item.text_edit.new_text == item.label

    def test_unclosed_buffer_offers_the_members_written_so_far(self):
        source = "(class A []\n  (= v 1)\n  (def f [self]\n    (= self.a 1)\n    self.|"
        assert self.labels(source) == {"v", "f", "a"}

    def test_the_receiver_is_the_first_parameter_by_name(self):
        source = "(class A []\n  (def f [this]\n    (= this.a 1)\n    this.|))"
        assert self.labels(source) == {"f", "a"}

    @pytest.mark.parametrize(
        "source",
        [
            "(class A []\n  (deco staticmethod (def f [self]\n    self.|)))",
            "(class A []\n  (deco builtins.staticmethod (def f [self]\n    self.|)))",
            "(def f [self]\n  self.|)",
            "(class A []\n  (def f [self]\n    (fn [self] self.|)))",
            "(class A []\n  (def f [self]\n    (. self| )))",
        ],
        ids=[
            "staticmethod",
            "builtins.staticmethod",
            "function",
            "shadowing lambda",
            "touching receiver",
        ],
    )
    def test_no_member_context_keeps_the_general_completion(self, source):
        before, after = source.split("|")
        line = before.count("\n") + 1
        column = len(before) - (before.rfind("\n") + 1)
        document = SimpleNamespace(source=before + after, uri=URI)
        assert server.member_completion(document, line, column) is None
        assert "print" in self.labels(source)

    @pytest.mark.parametrize(
        "statement",
        [
            "(= (, self.p self.q) (, 1 2))",
            "(for [self.p (, _ self.q)] in xs (pass))",
            "(async-for self.p in xs (= self.q 1))",
            "(with [(open x) as self.p (open y) as [self.q]] (pass))",
            "(async-with [(open x) as (, self.p self.q)] (pass))",
        ],
    )
    def test_tuple_loop_and_with_targets_are_attributes(self, statement):
        source = f"(class A []\n  (def f [self]\n    {statement}\n    self.|))"
        assert self.labels(source) == {"f", "p", "q"}

    @pytest.mark.parametrize(
        "method, opening, closing",
        [
            ("__init_subclass__", "(def __init_subclass__ [cls]", ")"),
            ("__init-subclass__", "(def __init-subclass__ [cls]", ")"),
            ("__class_getitem__", "(def __class_getitem__ [cls]", ")"),
            ("__new__", "(def __new__ [cls]", ")"),
            ("__new__", "(deco staticmethod (def __new__ [cls]", "))"),
        ],
        ids=[
            "__init_subclass__",
            "hyphenated __init_subclass__",
            "__class_getitem__",
            "__new__",
            "staticmethod __new__",
        ],
    )
    def test_implicit_class_receivers_skip_instance_attributes(
        self, method, opening, closing
    ):
        source = (
            "(class A []\n"
            "  (def __init__ [self] (= self.instance-attr 1))\n"
            f"  {opening}\n"
            "    (= cls.class-attr 1)\n"
            f"    cls.|){closing}"
        )
        assert self.labels(source) == {"__init__", method, "class-attr"}

    def test_names_compare_as_python_names(self):
        # my-self/my_self, my-attr/my_attr and my-base/my_base are one name
        # each; the first spelling seen is the label.
        source = (
            "(class my-base []\n  (= inherited 1))\n"
            "(class A [my_base]\n"
            "  (def f [my-self]\n"
            "    (= my_self.my-attr 1)\n"
            "    (= (. my-self my_attr) 2)\n"
            "    my_self.|))"
        )
        assert self.labels(source) == {"f", "my-attr", "inherited"}

    @pytest.mark.parametrize(
        "source, expected",
        [
            (
                "(class A []\n"
                "  (def i [self] (= self.instance-attr 1))\n"
                "  (deco builtins.classmethod (def c [cls]\n"
                "    (= cls.class-attr 1)\n"
                "    cls.|)))",
                {"i": "Method", "c": "Method", "class-attr": "Variable"},
            ),
            (
                "(class A []\n"
                "  (deco builtins.property (def p [self] 1))\n"
                "  (def f [self]\n"
                "    self.|))",
                {"p": "Property", "f": "Method"},
            ),
        ],
        ids=["builtins.classmethod", "builtins.property"],
    )
    def test_a_builtins_decorator_reads_as_the_builtin(self, source, expected):
        items = self.complete(source)[2]
        assert {item.label: item.kind.name for item in items} == expected

    def test_the_edit_range_is_in_utf16_units(self):
        from pygls.workspace import TextDocument

        source = '(class A []\n  (def f [self]\n    (print "\U0001f600" self.ab)))'
        document = TextDocument(URI, source)
        ls = SimpleNamespace(
            workspace=SimpleNamespace(get_text_document=lambda uri: document)
        )
        # After "self.ab": 22 code points, 23 UTF-16 units (the emoji is two).
        [item, *_] = run_completion(ls, position(2, 23)).items
        start, end = item.text_edit.range.start, item.text_edit.range.end
        assert (start.line, start.character, end.line, end.character) == (2, 21, 2, 23)

    def test_members_replace_the_word_and_filter_by_the_typed_spelling(self):
        # Defects: accepting mid-word leaves the rest of the word behind (no
        # replace range); my-x is dropped once the user types my_ (no
        # filterText), or once they type my- (filterText always my_x).
        from lsprotocol import types as lsp

        def complete(source):
            before, after = source.split("|")
            line = before.count("\n")
            character = len(before) - (before.rfind("\n") + 1)
            ls = ls_with(before + after)
            ls.client_capabilities = lsp.ClientCapabilities(
                text_document=lsp.TextDocumentClientCapabilities(
                    completion=lsp.CompletionClientCapabilities(
                        completion_item=lsp.ClientCompletionItemOptions(
                            insert_replace_support=True
                        )
                    )
                )
            )
            result = run_completion(ls, position(line, character))
            return character, {item.label: item for item in result.items}

        source = (
            "(class A []\n"
            "  (def __init__ [self] (= self.my-x 1))\n"
            "  (def f [self]\n"
            "    self.{}))"
        )
        character, items = complete(source.format("my|-x"))
        edit = items["my-x"].text_edit
        assert (edit.new_text, items["my-x"].filter_text) == ("my-x", "my_x")
        assert (edit.insert.start.character, edit.insert.end.character) == (
            character - 2,
            character,
        )
        assert (edit.replace.start.character, edit.replace.end.character) == (
            character - 2,
            character + 2,
        )
        _, items = complete(source.format("my-|"))
        assert items["my-x"].filter_text == "my-x"

    def test_a_classmethod_receiver_skips_instance_attributes(self):
        source = (
            "(class A []\n"
            "  (def i [self] (= self.instance-attr 1))\n"
            "  (deco classmethod (def c [cls]\n"
            "    (= cls.class-attr 1)\n"
            "    cls.|)))"
        )
        assert self.labels(source) == {"i", "c", "class-attr"}

    def test_a_closure_sees_the_methods_receiver(self):
        source = "(class A []\n  (def f [self]\n    (def g [] self.|)))"
        assert self.labels(source) == {"f"}


def test_an_operator_never_spells_a_name():
    # Defect: (- a b) records - as the spelling of _ (py-name of -), so a
    # backend member _ is relabelled "-"; ? (py-name _p) likewise.
    spellings = server.buffer_spellings("(- a b)\n(? c)\n(= x._ x._p)")
    assert (spellings.get("_", "_"), spellings["_p"]) == ("_", "_p")


def test_an_auto_import_item_gives_no_kind():
    # Defect: a symbol the buffer never binds (offered with an import edit)
    # sets the kind of the buffer's own name.
    from lsprotocol import types as lsp

    local = lsp.CompletionList(
        is_incomplete=False,
        items=[lsp.CompletionItem(label="foo", kind=lsp.CompletionItemKind.Variable)],
    )
    edit = lsp.TextEdit(
        range=lsp.Range(
            start=lsp.Position(line=0, character=0),
            end=lsp.Position(line=0, character=0),
        ),
        new_text="from m import foo\n",
    )
    imported = lsp.CompletionItem(
        label="foo", kind=lsp.CompletionItemKind.Class, additional_text_edits=[edit]
    )
    [item] = server.with_backend_kinds(local, [imported]).items
    assert item.kind == lsp.CompletionItemKind.Variable


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
    real_view_at_cursor = pv.view_at_cursor

    def counting(*args, **kwargs):
        compiled.append(args)
        return real_view_at_cursor(*args, **kwargs)

    monkeypatch.setattr(pv, "view_at_cursor", counting)
    real_to_thread = asyncio.to_thread

    async def close_first(fn, *args, **kwargs):
        server.view_cache.forget(uri)  # the close lands before the worker
        return await real_to_thread(fn, *args, **kwargs)

    async def no_items(*args, **kwargs):
        return []

    client = SimpleNamespace(backend_uri=lambda u: u, request=no_items)
    document = TextDocument(uri, "(import os)\nos.pa", version=1)
    monkeypatch.setattr(asyncio, "to_thread", close_first)
    asyncio.run(server.backend_answer(client, document, 2, 5, None))
    monkeypatch.setattr(asyncio, "to_thread", real_to_thread)
    asyncio.run(server.backend_answer(client, document, 2, 5, None))
    assert len(compiled) == 2  # the closed document's view was not kept
