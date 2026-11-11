"""The LSP's Python view: a buffer's generated Python and its position map.

Expected positions are read off the source and generated text with
``locate``, so every assertion names the word it lands on.
"""

import builtins
import contextlib
import hashlib
import importlib
import importlib.util
import json
import ntpath
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import warnings
from types import SimpleNamespace

import pytest

import lispy  # noqa: F401  (registers the .lpy importer)
from lispy.core import nodes, parser
from lispy.core.importer import lpy_cache_path, package_root

pv = importlib.import_module("lispy.lsp.python_view")
server = importlib.import_module("lispy.lsp.server")
mm = importlib.import_module("lispy.lsp.macro_modules")


def locate(text, word, nth=0):
    """(1-based line, code-point column) of the nth occurrence of word, on
    the reader's lines (a lone CR ends one too)."""
    index = -1
    for _ in range(nth + 1):
        index = text.index(word, index + 1)
    lines = parser.split_lines(text[:index])
    return len(lines), len(lines[-1])


def plus(pos, cols):
    return pos[0], pos[1] + cols


def at_bar(src):
    """(source, line, col) of src with its | cursor mark removed."""
    index = src.index("|")
    text = src.replace("|", "", 1)
    line = text.count("\n", 0, index) + 1
    return text, line, index - (text.rfind("\n", 0, index) + 1)


def view_of(src):
    view = pv.compile_view(src)
    assert view is not None
    return view


class TestNames:
    def test_kebab_name_maps_both_ways(self):
        src = "(= my-x 1)\n(print my-x)"
        view = view_of(src)
        assert view.text == "my_x = 1\nprint(my_x)"
        gen, lpy = locate(view.text, "my_x", 1), locate(src, "my-x", 1)
        assert view.map.to_lpy(*plus(gen, 2)) == plus(lpy, 2)
        assert view.map.to_generated(*plus(lpy, 2)) == plus(gen, 2)

    def test_mangled_name_of_another_length(self):
        # dd-ee? is 6 characters, dd_ee_p 7: an offset past the spelling
        # clamps to it, and the name after it is not shifted by one.
        src = "(= dd-ee? 1)\n(print dd-ee? zz)"
        view = view_of(src)
        gen, lpy = locate(view.text, "dd_ee_p", 1), locate(src, "dd-ee?", 1)
        assert view.map.to_lpy(*plus(gen, 7)) == plus(lpy, 6)
        assert view.map.to_generated(*plus(lpy, 6)) == plus(gen, 6)
        assert view.map.to_generated(*locate(src, "zz")) == locate(view.text, "zz")
        assert view.map.to_lpy(*locate(view.text, "zz")) == locate(src, "zz")

    def test_non_ascii_before_a_name(self):
        # ast.parse columns are UTF-8 bytes: "é" is two of them.
        src = '(print "é" my-x)'
        view = view_of(src)
        gen, lpy = locate(view.text, "my_x"), locate(src, "my-x")
        assert view.map.to_lpy(*gen) == lpy
        assert view.map.to_generated(*lpy) == gen

    def test_f_string_field_name(self):
        src = '(= my-x 1)\n(print f"{my-x}")'
        view = view_of(src)
        gen, lpy = locate(view.text, "my_x", 1), locate(src, "my-x", 1)
        assert view.map.to_lpy(*plus(gen, 1)) == plus(lpy, 1)
        assert view.map.to_generated(*lpy) == gen

    @pytest.mark.skipif(sys.version_info < (3, 14), reason="t-strings: 3.14+")
    def test_t_string_field_is_written_in_python(self):
        # Interpolation.str holds the lispython text (what a Template reports);
        # written verbatim, the view would read t'{my-x}', a subtraction.
        src = '(= my-x 1)\n(print t"{my-x}" t"{(+ my-x 1)!r:>{my-x}}")'
        view = view_of(src)
        assert view.text == "my_x = 1\nprint(t'{my_x}', t'{my_x + 1!r:>{my_x}}')"
        gen, lpy = locate(view.text, "my_x", 1), locate(src, "my-x", 1)
        assert view.map.to_lpy(*plus(gen, 1)) == plus(lpy, 1)

    def test_nested_past_ast_unparse(self):
        # 1000 operands nest 1000 BinOps: ast.unparse fails a few hundred
        # levels down, while ast.parse and the map's walk take them.
        src = "(= my-x 1)\n(print (+" + " 1" * 1000 + " my-x))"
        view = view_of(src)
        assert view.text == "my_x = 1\nprint(" + "1 + " * 1000 + "my_x)"
        gen, lpy = locate(view.text, "my_x", 1), locate(src, "my-x", 1)
        assert view.map.to_lpy(*plus(gen, 1)) == plus(lpy, 1)
        assert view.map.to_generated(*plus(lpy, 1)) == plus(gen, 1)

    def test_parameter_after_its_colon_and_keyword_argument(self):
        src = "(def f [a :bb-c 1] (g :kw-d 2))"
        view = view_of(src)
        for py, spelled in [("bb_c", "bb-c"), ("kw_d", "kw-d")]:
            gen, lpy = locate(view.text, py), locate(src, spelled)
            assert view.map.to_lpy(*plus(gen, 1)) == plus(lpy, 1)
            assert view.map.to_generated(*plus(lpy, 1)) == plus(gen, 1)


class TestAttributes:
    @pytest.mark.parametrize(
        ["src", "py", "spelled"],
        [
            ("(print a.cc.dd-ee?)", "dd_ee_p", "dd-ee?"),
            ("(print a.bb-c?.dd)", "bb_c_p", "bb-c?"),
            ("(print (. x aa bb))", "bb", "bb"),
            ("(print (. x aa bb))", "aa", "aa"),
            ("(print (.meth-a x 1))", "meth_a", "meth-a"),
        ],
    )
    def test_attribute_name_column(self, src, py, spelled):
        view = view_of(src)
        gen, lpy = locate(view.text, py), locate(src, spelled)
        assert view.map.to_lpy(*plus(gen, 1)) == plus(lpy, 1)
        assert view.map.to_generated(*plus(lpy, 1)) == plus(gen, 1)


class TestMacros:
    def test_when_maps_to_the_macro_call_and_its_operand(self):
        src = "(require lispy.macros *)\n(= my-x 1)\n(when my-x (print 1))"
        view = view_of(src)
        assert view.map.to_lpy(*locate(view.text, "if")) == locate(src, "(when")
        gen, lpy = locate(view.text, "my_x", 1), locate(src, "my-x", 1)
        assert view.map.to_lpy(*gen) == lpy
        assert view.map.to_generated(*lpy) == gen

    def test_threading_method_names(self):
        src = "(require lispy.macros *)\n(-> x (.mm) .nn)"
        view = view_of(src)
        assert view.text.endswith("x.mm().nn()")
        for name in ["mm", "nn"]:
            assert view.map.to_lpy(*locate(view.text, name)) == locate(src, name)

    def test_a_macro_position_outside_the_buffer_maps_nowhere(self):
        # Defect: the map trusted whatever position a macro gave its node
        # (column 50 of the 3-character line 2): reading the buffer there
        # raised IndexError (no view), and a node entry sent the editor to it.
        src = (
            '(defmacro f [] (return (Paren (Symbol "print" :lineno 2'
            " :col_offset 50 :end_lineno 2 :end_col_offset 55))))\n(f)\n"
        )
        view = view_of(src)
        lines = src.split("\n")
        outside = [
            (start, end)
            for [_, _, start, end, _] in view.map.entries
            if start[1] > len(lines[start[0] - 1]) or end[1] > len(lines[end[0] - 1])
        ]
        assert outside == []


def gen_range(text, word, nth=0):
    start = locate(text, word, nth)
    return start, plus(start, len(word))


class TestLpyRange:
    """lpy_range: a backend's range in the generated text, back to .lpy."""

    def test_a_name_maps_character_for_character(self):
        # Defect: the range keeps generated columns or the Python spelling's
        # width (dd_ee_p is one longer than dd-ee?, so its end overshoots).
        src = "(= dd-ee? 1)\n(print dd-ee? obj.dd-ee?)"
        view = view_of(src)
        for nth in range(3):
            got = pv.lpy_range(view, src, *gen_range(view.text, "dd_ee_p", nth))
            assert got == gen_range(src, "dd-ee?", nth)

    @pytest.mark.parametrize("head", ["def", "class"])
    def test_a_definition_name_is_its_lpy_word(self, head):
        # A FunctionDef or ClassDef name has no entry of its own: the
        # defect is the start of the whole form, or None.
        src = f"(= x 1)\n({head} my-fn [] (pass))"
        view = view_of(src)
        got = pv.lpy_range(view, src, *gen_range(view.text, "my_fn"))
        assert got == gen_range(src, "my-fn")

    def test_generated_code_maps_nowhere(self):
        # Defect: a defmacro's temporary def, which the user never wrote,
        # lands on whatever .lpy text its entry or line number names.
        src = "(defmacro swap [a b] `(= [~a ~b] [~b ~a]))\n(= x 1)"
        view = view_of(src)
        assert (
            pv.lpy_range(view, src, *gen_range(view.text, "___defmacro_temp___swap"))
            is None
        )

    @pytest.mark.parametrize("sep", ["\n", "\r\n", "\r"])
    def test_a_name_maps_on_the_readers_lines(self, sep):
        # Defect: the position map read the source's lines split on \n only,
        # so after a lone CR no node was in the buffer and nothing mapped.
        src = f"(= a 1){sep}(= bb 2){sep}(print bb)"
        view = view_of(src)
        assert pv.lpy_range(view, src, *gen_range(view.text, "bb")) == ((2, 3), (2, 5))

    def test_the_module_location_stays_at_its_start(self):
        # A module's own location (the definition of an imported module) is
        # the empty range at its first character, whatever entry is there.
        src = "(= x 1)"
        assert pv.lpy_range(view_of(src), src, (1, 0), (1, 0)) == ((1, 0), (1, 0))


class TestFailures:
    def test_a_parse_error_is_no_view(self):
        assert pv.compile_view("(print 1") is None

    @pytest.mark.parametrize(
        "src",
        [
            "(= 1 2)\n(= y 3)",  # compile error
            "(defmacro boom [] (raise (ValueError)))\n(boom)\n(= y 3)",  # macro fails
        ],
    )
    def test_a_failed_form_is_left_out_of_the_view(self, src):
        # Defect: one form that fails to compile costs the whole buffer its
        # view (so the backend publishes nothing for it), or is not reported.
        compiled = pv.compile_buffer(src)
        assert compiled.view.text.endswith("\ny = 3") or compiled.view.text == "y = 3"
        [(form, error)] = compiled.failures
        assert form.lineno == src.count("\n")
        assert isinstance(error, Exception)


FAILED_FORMS = """"Doc."
(from __future__ [annotations])
(def f [x ^int y :z 1 *a :k 2 **kw])
(async-def af [p])
(class C []
  (def m [self]))
(deco d (def h [a]))
(def g[T] [])
(def bad-params [:x 1 y])
(= v (while))
(= a.b (while))
(defmacro mac [x] (return `(def ~x [] (while))))
(mac q)
(def when [] 1)
(= ok 1)
"""


def module_bindings(view):
    """(kind, name) of each module-level statement of view's tree that binds
    a name, the defmacro's and the placeholders' generated scaffolding
    aside."""
    found = []
    for stmt in view.tree.body:
        if isinstance(stmt, (pv.ast.FunctionDef, pv.ast.AsyncFunctionDef)):
            if not stmt.name.startswith("___defmacro"):
                found.append((type(stmt).__name__, stmt.name))
        elif isinstance(stmt, pv.ast.ClassDef):
            found.append(("ClassDef", stmt.name))
        elif isinstance(stmt, pv.ast.AnnAssign):
            if not stmt.target.id.startswith(pv.GENERATED_PREFIX):
                found.append(("AnnAssign", stmt.target.id))
        elif isinstance(stmt, pv.ast.Assign):
            found += [("Assign", t.id) for t in stmt.targets if hasattr(t, "id")]
    return found


class TestFailedForms:
    def test_a_failed_form_declares_the_names_its_syntax_binds(self):
        # Defects: a failed def/class/= leaves its name undefined for the
        # backend at every use (a false "is not defined", in this file and
        # through the shadow in its importers); a failed user-macro call, a
        # dotted target or a name the compiler refuses (a macro's) is
        # declared anyway.
        view = view_of(FAILED_FORMS)
        assert module_bindings(view) == [
            ("FunctionDef", "f"),
            ("AsyncFunctionDef", "af"),
            ("ClassDef", "C"),
            ("Assign", "h"),
            ("FunctionDef", "g"),
            ("Assign", "bad_params"),
            ("Assign", "v"),
            ("Assign", "ok"),
        ]
        [f] = [s for s in view.tree.body if getattr(s, "name", None) == "f"]
        # The signature stub keeps the arity: x, y and z positional (z with
        # a default), *a, keyword-only k, **kw.
        assert [a.arg for a in f.args.args] == ["x", "y", "z"]
        assert len(f.args.defaults) == 1
        assert (f.args.vararg.arg, f.args.kwarg.arg) == ("a", "kw")
        assert [a.arg for a in f.args.kwonlyargs] == ["k"]

    def test_a_failed_chained_assignment_declares_every_target(self):
        # Defect: only the first target of (= a b v) (a = b = v) is
        # declared, so b is a false "is not defined" at its uses; an
        # annotation is no target.
        view = view_of("(= a b (while))\n(= t ^int (while))\n(print a b t)")
        assert module_bindings(view) == [
            ("Assign", "a"),
            ("Assign", "b"),
            ("Assign", "t"),
        ]

    def test_the_any_import_follows_the_docstring_and_future_imports(self):
        # Defects: the import or the placeholders' value goes first, before
        # a from __future__ import, which Python and the backends refuse
        # anywhere but at the top; or the value is declared after a
        # placeholder reads it (unbound there).
        body = view_of(FAILED_FORMS).tree.body
        docstring, future = body[0], body[1]
        assert isinstance(docstring.value.value, str)
        assert future.module == "__future__"

        def first(predicate):
            return next(i for i, stmt in enumerate(body) if predicate(stmt))

        imported = first(
            lambda stmt: any(
                alias.asname == pv.ANY_ALIAS for alias in getattr(stmt, "names", [])
            )
        )
        declared = first(
            lambda stmt: (
                isinstance(stmt, pv.ast.AnnAssign) and stmt.target.id == pv.ANY_VALUE
            )
        )
        read = [
            i
            for i, stmt in enumerate(body)
            for node in pv.ast.walk(stmt)
            if isinstance(node, pv.ast.Name)
            and node.id == pv.ANY_VALUE
            and isinstance(node.ctx, pv.ast.Load)
        ]
        assert read
        assert 2 <= imported < declared < min(read)

    def test_only_a_declared_name_maps_to_the_buffer(self):
        # Defects: the declared name has no entry (definition at it is
        # lost), or a diagnostic on the placeholder plumbing (basedpyright's
        # reportExplicitAny on the annotation or the class base, reportAny
        # or an unused-parameter hint on a stub's parameter, anything on the
        # import) or on the declared name itself (reportAny on lispython's
        # own binding) lands on the user's code.
        src = FAILED_FORMS
        view = view_of(src)
        start = locate(view.text, "v = ___lpy_any_value")
        assert pv.lpy_range(view, src, start, plus(start, 1)) == (
            locate(src, "v (while)"),
            plus(locate(src, "v (while)"), 1),
        )
        plumbing = [
            gen_range(view.text, "___lpy_any", nth)
            for nth in range(view.text.count("___lpy_any"))
        ] + [
            gen_range(view.text, "from typing import Any as ___lpy_any"),
            gen_range(view.text, "___lpy_any_value: ___lpy_any = ..."),
            (start, plus(start, 1)),
            gen_range(view.text, "class C(___lpy_any):"),
            gen_range(view.text, "x: ___lpy_any"),
            gen_range(view.text, "kw: ___lpy_any"),
            gen_range(view.text, "async def af"),
            gen_range(view.text, "h = ___lpy_any_value"),
        ]
        for start, end in plumbing:
            for tagged in (False, True):
                assert pv.lpy_diagnostic_range(view, src, start, end, tagged) is None


class TestCursor:
    def test_partial_restored_and_map_shifted(self):
        src = "(= my-x 1)\n(print my-x.fo zz-top)"
        view = pv.view_at_cursor(src, *plus(locate(src, "fo"), 2))
        assert view.text == "my_x = 1\nprint(my_x.fo, zz_top)"
        assert view.cursor == plus(locate(view.text, "fo"), 2)
        assert (view.partial, view.fallback) == ("fo", False)
        gen, lpy = locate(view.text, "zz_top"), locate(src, "zz-top")
        assert view.map.to_lpy(*plus(gen, 3)) == plus(lpy, 3)
        assert view.map.to_generated(*plus(lpy, 3)) == plus(gen, 3)
        gen, lpy = locate(view.text, "fo"), locate(src, "fo")
        assert view.map.to_lpy(*plus(gen, 1)) == plus(lpy, 1)

    def test_cursor_mid_word(self):
        # Defect: the rest of the word glues onto the sentinel (one symbol,
        # ___lpy_cursor___-top), so a cursor inside a word has no view.
        src = "(= my-x 1)\n(print my-x.fo-o zz-top)"
        view = pv.view_at_cursor(src, *plus(locate(src, "fo-o"), 2))
        assert view.text == "my_x = 1\nprint(my_x.fo, zz_top)"
        assert view.cursor == plus(locate(view.text, "fo"), 2)
        assert (view.partial, view.attribute) == ("fo", True)
        gen, lpy = locate(view.text, "zz_top"), locate(src, "zz-top")
        assert view.map.to_lpy(*plus(gen, 3)) == plus(lpy, 3)

    def test_mangled_partial(self):
        src = "(= my-x 1)\n(print my-x.dd-e?)"
        view = pv.view_at_cursor(src, *plus(locate(src, "dd-e?"), 5))
        assert view.text == "my_x = 1\nprint(my_x.dd_e_p)"
        assert view.cursor == plus(locate(view.text, "dd_e_p"), 6)

    @pytest.mark.parametrize(
        ["src", "after", "line"],
        [
            ("(= my-x 1)\n(print my-x.)", "my-x.", "print(my_x.)"),
            ("(= obj 1)\n(. obj )", "obj ", "obj."),
        ],
    )
    def test_empty_partial(self, src, after, line):
        last = src.count(after) - 1
        view = pv.view_at_cursor(src, *plus(locate(src, after, last), len(after)))
        assert view.text.split("\n")[1] == line
        assert view.cursor == (2, line.index(".") + 1)
        assert view.partial == ""

    @pytest.mark.parametrize(
        ["src", "fallback"],
        [
            ("(while)\n(= my-x 1)\n(print my-x.fo|)\n(def g)", False),
            # Unbalanced, so the prefix compiles, (while) in it.
            ("(while)\n(def f [grid]\n  (when (== (sub gr|\n", True),
        ],
    )
    def test_a_failed_form_elsewhere_keeps_the_cursor_view(self, src, fallback):
        # Defect: a form failing anywhere in the buffer (or in its prefix)
        # costs the cursor its view, so the backend is never asked there.
        view = pv.view_at_cursor(*at_bar(src))
        assert view is not None and view.fallback is fallback

    def test_the_cursors_own_failed_form_falls_back_to_the_prefix(self):
        # Defect: the form holding the cursor is left out of the view as a
        # failure elsewhere is, so the view lacks the sentinel and there is
        # none; its failure must send the cursor to the prefix and its unwrap.
        src = "(= my-x 1)\n(def f [] (when (== my-x.fo| 1)))\n(= z 2)"
        view = pv.view_at_cursor(*at_bar(src))
        assert view.fallback is True
        assert (view.partial, view.text) == ("fo", "my_x = 1\n\ndef f():\n    my_x.fo")

    def test_unbalanced_buffer_falls_back_to_the_prefix(self):
        src = "(= my-x 1)\n(print my-x.fo\n(def g [] 1)"
        view = pv.view_at_cursor(src, *plus(locate(src, "fo"), 2))
        assert view.fallback is True
        assert view.text == "my_x = 1\nprint(my_x.fo)"
        assert view.cursor == plus(locate(view.text, "fo"), 2)

    def test_a_form_refused_around_the_cursor_is_unwrapped(self):
        # Defect: an (== ...) with one operand and a (when ...) with no
        # body around the cursor leave no view; or the unwrap moves the
        # cursor out of the def, where grid is no name.
        src = "(def f [grid]\n  (when (== (sub gr\n"
        view = pv.view_at_cursor(src, *plus(locate(src, "gr", 1), 2))
        assert view.fallback is True
        assert view.text == "def f(grid):\n    gr"

    @pytest.mark.parametrize(
        ["src", "text", "before"],
        [
            (
                "(def f [grid] (when (all (map (fn [row] (> (len ro|",
                "def f(grid):\n    all(map(lambda row: len(ro)))",
                "lambda row: len(ro",
            ),
            (
                "(def f [grid] (try (for x in grid (print x|",
                "def f(grid):\n    for x in grid:\n        print(x)",
                "        print(x",
            ),
        ],
    )
    def test_the_unwrap_keeps_the_forms_inside_the_refused_one(self, src, text, before):
        # Regression TASK-1181: the unwrap blanked every form between the
        # refused one and the cursor, so the fn's row and the for's x were
        # gone and the backend's final answer never offered them.
        view = pv.view_at_cursor(*at_bar(src))
        assert view.text == text
        line, col = view.cursor
        assert view.text.split("\n")[line - 1][:col].endswith(before)

    @pytest.mark.parametrize("partial", ["", "re"])
    def test_a_glued_base_survives_the_unwrap(self, partial):
        # Regression TASK-1185: the base grid. was blanked with the refused
        # form, so the view asked for the scope's names, not grid's members.
        view = pv.view_at_cursor(*at_bar(f"(def f [grid] (when (== grid.{partial}|"))
        assert view.attribute is True
        line, col = view.cursor
        assert view.text.split("\n")[line - 1][:col] == f"    grid.{partial}"

    def test_the_unwrap_reads_into_a_quasiquote(self):
        # Defect: the walk for the child holding the cursor skips quote
        # forms, so a cursor in an unquote finds no child and gives no view.
        view = pv.view_at_cursor(*at_bar("(def f [grid] (when `(f ~gr|"))
        line, col = view.cursor
        assert view.text.startswith("def f(grid):\n")
        assert view.text.split("\n")[line - 1][:col].endswith(" gr")

    @pytest.mark.parametrize(
        ["src", "text"],
        [
            ("(def f [grid] (when `(a ~(when gr|", "    Paren(Symbol('a'), gr)"),
            ("(def f [grid] (when `(a ~(> (len gr|", "    Paren(Symbol('a'), len(gr))"),
        ],
    )
    def test_a_refused_form_in_an_unquote_stays_unquoted(self, src, text):
        # Defect: the ~ is left before the blanks, where a lone ~ is the
        # invert symbol, so the child is no longer unquoted (or no view).
        view = pv.view_at_cursor(*at_bar(src))
        assert view.text == "def f(grid):\n" + text

    def test_quote_marks_written_back_keep_the_map(self):
        # The child starts a line one column in: the ~ fits before it.
        src = "(def f [grid] (when `(a ~(when\n gr|"
        view = pv.view_at_cursor(*at_bar(src))
        assert view.text == "def f(grid):\n    Paren(Symbol('a'), gr)"
        assert view.map.to_lpy(view.cursor[0], view.cursor[1] - 2) == (2, 1)

    @pytest.mark.parametrize(
        "src",
        [
            "(def f [grid] (when `(a ~(when\ngr|",
            "(def f [grid] (when `(a ~(when\r\ngr|",
            "(def f [grid] (when `(a ~@(when\n gr|",
        ],
    )
    def test_quote_marks_never_overwrite_a_line_break(self, src):
        # Defect: the marks written before a child at the start of a line
        # overwrite the line break, so every later position moves and the
        # word maps to the line above (or nowhere). With no room for them
        # the marks are dropped, which leaves the word quoted: no view.
        view = pv.view_at_cursor(*at_bar(src))
        word = (2, src.split("\n")[-1].index("gr"))
        assert view is None or (
            view.map.to_lpy(view.cursor[0], view.cursor[1] - 2) == word
        )

    @pytest.mark.parametrize(
        ["src", "line", "attribute"],
        [
            ("(def f [grid] (when *gr|", "    gr", False),
            ("(def f [grid] (when (== *grid.|", "    grid.", True),
            ("(def f [grid] (when (== :gr|", "    gr", False),
        ],
    )
    def test_a_child_in_the_refused_forms_place_drops_a_list_prefix(
        self, src, line, attribute
    ):
        # Defect: the child keeps its * ** : prefix where an expression
        # stands, so the view is no Python (*gr as a statement).
        view = pv.view_at_cursor(*at_bar(src))
        assert (view.text.split("\n")[-1], view.attribute) == (line, attribute)

    def test_a_glued_word_that_is_no_attribute_is_no_view(self):
        # Defect: a macro turns grid.re into re, and the view asks for the
        # scope's names where the user typed an attribute of grid.
        src = (
            '(defmacro m [x] (return (Symbol (sub (.split x.name ".") -1))))\n'
            "(def f [grid] (m grid.re|))"
        )
        assert pv.view_at_cursor(*at_bar(src)) is None

    def test_a_lambda_head_with_glued_params_gives_no_view(self):
        # Defect: fn[T] (not legal lispython) has a GluedHead head, which
        # _param-slot read as no slot, so the form was unwrapped to a bare x
        # at module scope and x resolved outside the lambda's parameters.
        assert pv.view_at_cursor(*at_bar("(fn[T] [x] ^x|")) is None

    @pytest.mark.skipif(sys.version_info >= (3, 12), reason="type needs 3.12")
    def test_a_type_alias_value_gives_no_view_where_type_is_refused(self):
        # Defect: on 3.11 the refused type form was unwrapped to the bare
        # value +U, which resolves U outside the alias's type parameters.
        assert pv.view_at_cursor(*at_bar("(type A[U] (+ U|")) is None

    @pytest.mark.skipif(sys.version_info >= (3, 12), reason="type needs 3.12")
    def test_a_type_alias_without_type_parameters_keeps_its_unwrapped_view(self):
        # Defect guarded: refusing every type value on 3.11 also lost the
        # view of an alias that binds no name.
        assert pv.view_at_cursor(*at_bar("(type A x.|)")).text == "x."

    def test_a_type_head_with_glued_params_gives_no_view_in_its_value(self):
        # Defect: type[T] (not legal) was unwrapped to the bare value +U, which
        # resolves U outside the alias's type parameters.
        assert pv.view_at_cursor(*at_bar("(type[T] A[U] (+ U|")) is None

    def test_a_def_head_with_glued_params_gives_no_view_in_its_body(self):
        # Defect: def[T] (not legal) read as no slot, so the body unwrapped
        # to a bare +x outside the def's parameters.
        assert pv.view_at_cursor(*at_bar("(def[T] f [x] (+ x|")) is None

    @pytest.mark.parametrize(
        ["src", "text"], [("(def[T] f [x] ^x|", "x"), ("(class[T] A [B|", "B")]
    )
    def test_a_glued_def_or_class_head_unwraps_as_the_plain_head_does(self, src, text):
        # Defect: the head of def[T]/class[T] was read raw (None), so a return
        # annotation counted as body (no view) and a class's bases bracket was
        # never entered ('[B]').
        assert pv.view_at_cursor(*at_bar(src)).text == text

    @pytest.mark.parametrize(
        ["src", "text"], [("[(when x|", "[x]"), ("((when x|", "x()")]
    )
    def test_the_innermost_refused_form_is_unwrapped(self, src, text):
        # Defect: the outer form, whose head the inner form's bracket is,
        # is unwrapped first, and the inner one in the next round, so the
        # list or call around the cursor is lost.
        assert pv.view_at_cursor(*at_bar(src)).text == text

    def test_a_lone_carriage_return_never_loops_the_unwrap(self, monkeypatch):
        # Defect: a lone CR is a line break to the lexer but not to a split
        # on \n, so the blanked span missed the refused bracket and every
        # round recompiled (running the user's macros) without end. Four
        # open forms allow at most five compiles.
        compiles = []
        compile_buffer = pv.compile_buffer

        def counted(*args, **kwargs):
            compiles.append(args)
            assert len(compiles) <= 5, "the unwrap does not terminate"
            return compile_buffer(*args, **kwargs)

        monkeypatch.setattr(pv, "compile_buffer", counted)
        src = '(def f [grid]\n  (print "a\rb") (when (== (sub gr\n'
        view = pv.view_at_cursor(src, *plus(locate(src, "gr", 1), 2))
        assert len(compiles) <= 5
        # The blank lands on the refused forms, so the cursor stays in f.
        assert view.text == "def f(grid):\n    print('a\\nb')\n    gr"

    @pytest.mark.parametrize(
        "src",
        [
            "(def f [grid y|",
            "(async-def f [grid y|",
            "(defmacro m [grid y|",
            "(def f [*ar|",
            "(def f[T|",
            "(class A[T|",
            "(fn [y|",
            # A binding form among the forms the blank drops, not the one
            # refused, or the one an error at its bracket belongs to.
            "(def f [grid] (lambda [a y|",
            "(def f [grid] (fn [y|",
            "(def f [grid] (if (lambda [x|",
            "(def f [grid] (when (== (lambda [x|",
            "(def f [grid] (for [i (lambda [x|",
            "(def f [grid] ((lambda [x|",
            "(def f [grid] [(lambda [x|",
            "(def f [grid] {(lambda [x|",
            # A parameter in a quasiquote's unquote.
            "(def f [grid] (when (lambda [x] `(a ~(fn [gr|",
            # In the body of a def or lambda the unwrap drops, its
            # parameters are names.
            "(fn [x] x y|",
        ],
    )
    def test_a_cursor_where_a_dropped_form_binds_a_name_is_none(self, src):
        # Defect: unwrapping drops the form whose parameter (or type
        # parameter) is being named, so that name is answered as a name of
        # the scope around it.
        assert pv.view_at_cursor(*at_bar(src)) is None

    @pytest.mark.parametrize(
        ["src", "text"],
        [
            ("(def f [grid] (lambda [a :b gr|", "def f(grid):\n    gr"),
            ("(def f [grid] (lambda [a ^gr|", "def f(grid):\n    gr"),
            # The default's own form stays around the cursor.
            ("(def f [grid] (lambda [a :b (g gr|", "def f(grid):\n    g(gr)"),
            ("(class A [Ba|", "Ba"),
            # A return annotation is no body: it is read around the def.
            ("(def f [grid] (def g [x] ^gr|", "def f(grid):\n    gr"),
            ("(def f [grid] (async-def g [x] ^gr|", "def f(grid):\n    gr"),
            ("(def f [grid] (def g [x] ^(sub list gr|", "def f(grid):\n    list[gr]"),
            ("(def f [grid] (def g [x] ^grid.|", "def f(grid):\n    grid."),
        ],
    )
    def test_a_default_annotation_or_base_is_read_around_the_form(self, src, text):
        # Defect: every cursor in a parameter list gives no view, though a
        # default, an annotation or a class base is evaluated in the scope
        # around the form, where the backend knows the names.
        assert pv.view_at_cursor(*at_bar(src)).text == text

    def test_unclosable_prefix_is_none(self):
        src = '(= my-x 1)\n(print f"{my-x.'
        assert pv.view_at_cursor(src, 2, len('(print f"{my-x.')) is None

    @pytest.mark.parametrize(
        ["src", "partial", "line"],
        [
            ("(print *ar)", "ar", "print(*ar)"),
            ("(print **kw-a)", "kw-a", "print(**kw_a)"),
            ("(f :ke-y 1)", "ke-y", "f(ke_y=1)"),
            ("(print a,bb)", "bb", "print(a, bb)"),
        ],
    )
    def test_prefix_stays_in_the_source(self, src, partial, line):
        view = pv.view_at_cursor(src, *plus(locate(src, partial), len(partial)))
        assert (view.text, view.partial, view.attribute) == (line, partial, False)
        gen = locate(line, pv.py_name(partial))
        assert view.cursor == plus(gen, len(pv.py_name(partial)))
        assert view.map.to_lpy(*plus(gen, 1)) == plus(locate(src, partial), 1)

    @pytest.mark.parametrize(
        ["src", "partial"],
        [
            ("(f :ke)", ":ke"),  # a keyword with no value does not compile
            ("(print ~@xs)", "~@xs"),  # an unquote-splice outside a quasiquote
            ("(print x.class)", "class"),  # restored as a keyword: no Python
            ("(print a+b)", "a+b"),  # an operator symbol, no name
            # an operator is no name, never restored as its mangled text _
            ("(-)", "-"),
            ('(print "my-wo")', "my-wo"),  # inside a string literal
            ('(print f"ab-c{x}")', "ab-c"),  # inside an f-string's text
        ],
    )
    def test_cursor_on_no_identifier_is_none(self, src, partial):
        assert pv.view_at_cursor(src, *plus(locate(src, partial), len(partial))) is None

    def test_sentinel_already_in_the_buffer_is_none(self):
        src = f"(= {pv.CURSOR_SENTINEL} 1)\n(print my-x.fo)"
        assert pv.view_at_cursor(src, *plus(locate(src, "fo"), 2)) is None

    @pytest.mark.skipif(sys.version_info < (3, 14), reason="t-strings: 3.14+")
    def test_t_string_field(self):
        # Interpolation.str is the field's source, so it spells the sentinel
        # too: that text is no second identifier.
        src = '(= my-x 1)\n(print t"{my-x}")'
        view = pv.view_at_cursor(src, *plus(locate(src, "my-x", 1), 4))
        assert view is not None
        assert (view.text, view.partial) == (pv.compile_view(src).text, "my-x")
        assert view.cursor == plus(locate(view.text, "my_x", 1), 4)

    def test_generated_text_is_not_parsed_again(self, monkeypatch):
        # compile_view's parse is the one inside its error handling; a parse
        # of the same text after it would raise out of the request.
        real, seen = pv.ast.parse, set()

        def parse_once(text, *args, **kwargs):
            if isinstance(text, str) and pv.CURSOR_SENTINEL in text:
                if text in seen:
                    raise SyntaxError("parsed twice")
                seen.add(text)
            return real(text, *args, **kwargs)

        monkeypatch.setattr(pv.ast, "parse", parse_once)
        src = "(= my-x 1)\n(print my-x.fo)"
        view = pv.view_at_cursor(src, *plus(locate(src, "fo"), 2))
        assert view is not None and view.text == "my_x = 1\nprint(my_x.fo)"
        assert view.tree is None  # the tree held the sentinel and old columns


class TestCache:
    def test_by_version_latest_only(self, monkeypatch):
        calls = []
        real = pv.compile_buffer

        def counting(source, **kw):
            calls.append(source)
            return real(source, **kw)

        monkeypatch.setattr(pv, "compile_buffer", counting)
        cache = pv.ViewCache()
        first = cache.view("u", 1, "(= a 1)")
        assert cache.view("u", 1, "ignored") is first
        assert cache.view("u", 2, "(= a (") is None
        assert cache.view("u", 2, "(= a (") is None  # None is cached too
        cache.view("u", 1, "(= a 1)")  # only the latest version is kept
        assert len(calls) == 3
        cursor = cache.view_at_cursor("u", 1, "(print a.b)", 1, 10)
        assert cache.view_at_cursor("u", 1, "(print a.b)", 1, 10) is cursor
        cache.view_at_cursor("u", 1, "(print a.b)", 1, 9)
        assert len(calls) == 5

    @pytest.mark.parametrize("method", ["view", "view_at_cursor"])
    def test_a_close_during_a_view_leaves_no_entry(self, monkeypatch, method):
        # Defect: the document closes while its view is compiled in a
        # thread, and the view stores its entry after forget dropped it.
        cache = pv.ViewCache()
        calls = []
        real = pv.compile_buffer

        def closing(source, **kw):
            calls.append(source)
            if len(calls) == 1:
                cache.forget("u")
            return real(source, **kw)

        monkeypatch.setattr(pv, "compile_buffer", closing)
        args = ("u", 1, "(print a.b)") + ((1, 10) if method == "view_at_cursor" else ())
        getattr(cache, method)(*args)
        getattr(cache, method)(*args)
        assert len(calls) == 2  # nothing was cached for the closed document

    @pytest.mark.parametrize("method", ["view", "view_at_cursor"])
    def test_a_late_older_version_keeps_the_newer_entry(self, monkeypatch, method):
        # Defect: an older version's compile (one given up on) that returns
        # after a newer version was cached replaces the newer entry, so the
        # newer version compiles again at its next request.
        cache = pv.ViewCache()
        calls = []
        real = pv.compile_buffer

        def at(version):
            cursor = (1, 10) if method == "view_at_cursor" else ()
            return ("u", version, "(print a.b)") + cursor

        def late(source, **kw):
            calls.append(source)
            if len(calls) == 1:  # version 2 is cached while version 1 compiles
                getattr(cache, method)(*at(2))
            return real(source, **kw)

        monkeypatch.setattr(pv, "compile_buffer", late)
        getattr(cache, method)(*at(1))
        getattr(cache, method)(*at(2))
        assert len(calls) == 2

    @pytest.mark.parametrize(
        ["table", "rest"], [("_views", ()), ("_cursor_views", (1, 10))]
    )
    def test_a_late_older_epoch_keeps_the_newer_entry(self, table, rest):
        # Defect: a late compile (one given up on) at the same version but
        # an older epoch replaces the entry a refresh stored under the newer
        # epoch, so the next request at that epoch compiles again.
        cache = pv.ViewCache()
        entries = getattr(cache, table)
        cache._store(entries, "u", (2, 5) + rest, "newer", cache.generation)
        cache._store(entries, "u", (2, 4) + rest, "late", cache.generation)
        assert entries["u"][1] == "newer"
        cache._store(entries, "u", (2, 5) + rest, "fresh", cache.generation)
        assert entries["u"][1] == "fresh"  # the same key: a fresh compile

    @pytest.mark.parametrize("method", ["view", "view_at_cursor"])
    def test_a_close_before_the_worker_starts_leaves_no_entry(
        self, monkeypatch, method
    ):
        # Defect: the generation is read when the worker starts, after a
        # forget that came between the hand-off and the start.
        cache = pv.ViewCache()
        calls = []
        real = pv.compile_buffer

        def counting(source, **kw):
            calls.append(source)
            return real(source, **kw)

        monkeypatch.setattr(pv, "compile_buffer", counting)
        args = ("u", 1, "(print a.b)") + ((1, 10) if method == "view_at_cursor" else ())
        generation = cache.generation  # read before the hand-off
        cache.forget("u")
        getattr(cache, method)(*args, generation=generation)
        getattr(cache, method)(*args)
        assert len(calls) == 2


class TestShadows:
    def write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_module_names_follow_packages_not_the_project_root(self, tmp_path):
        # src layout: src is no package, so it is the import root of pkg.
        # Defect: a module inside a regular package also searches its own
        # directory (src/pkg), which `import pkg.mod` never searches, so its
        # compile resolves a bare sibling that fails at run time.
        src = tmp_path / "src"
        for parts in [("pkg", "__init__.py"), ("pkg", "a", "__init__.lpy")]:
            self.write(src.joinpath(*parts), "")
        names = [
            pv.module_names(str(tmp_path.joinpath(*parts)))
            for parts in [
                ("top.lpy",),
                ("src", "pkg", "mod.lpy"),
                ("src", "pkg", "a", "__init__.lpy"),
                ("src", "pkg", "a", "m.lpy"),
            ]
        ]
        assert names == [
            ("top", "", str(tmp_path), (str(tmp_path),)),
            ("pkg.mod", "pkg", str(src), (str(src),)),
            ("pkg.a", "pkg.a", str(src), (str(src),)),
            ("pkg.a.m", "pkg.a", str(src), (str(src),)),
        ]

    def test_a_namespace_package_counts_up_to_a_root_above(self, tmp_path):
        # Defect: the walk stops at the first directory with no __init__,
        # so ns/a.lpy gets no package where Python imports it as ns.a, and
        # its relative requires fail ("no known parent package").
        self.write(tmp_path / "ns" / "pkg" / "__init__.py", "")
        roots = [str(tmp_path)]
        assert package_root(str(tmp_path / "ns"), roots) == (str(tmp_path), ["ns"])
        assert package_root(str(tmp_path / "ns" / "pkg"), roots) == (
            str(tmp_path),
            ["ns", "pkg"],
        )
        # A directory that is itself a root is its own import root.
        assert package_root(str(tmp_path), roots) == (str(tmp_path), [])

    def test_a_namespace_package_stops_at_the_nearest_root(self, tmp_path):
        # Defect: the walk goes on to the outer root, naming src/ns/a.lpy
        # src.ns.a where the src layout imports it as ns.a.
        (tmp_path / "src" / "ns").mkdir(parents=True)
        roots = [str(tmp_path), str(tmp_path / "src")]
        assert package_root(str(tmp_path / "src" / "ns"), roots) == (
            str(tmp_path / "src"),
            ["ns"],
        )

    @pytest.mark.parametrize("name", ["my-ns", "class", "1ns"])
    def test_a_directory_that_is_no_identifier_is_no_namespace(self, tmp_path, name):
        # Defect: every directory up to the root becomes a part, giving a
        # package Python can never import (my-ns, a keyword).
        (tmp_path / name / "ns").mkdir(parents=True)
        directory = str(tmp_path / name / "ns")
        assert package_root(directory, [str(tmp_path)]) == (directory, [])

    @pytest.mark.parametrize("module", ["ns.py", "ns.lpy", "ns.pyc"])
    def test_a_module_file_beside_it_hides_the_namespace(self, tmp_path, module):
        # Defect: ns/ is taken as a namespace package where FileFinder
        # imports the module file ns.* instead.
        (tmp_path / "ns").mkdir()
        self.write(tmp_path / module, "")
        directory = str(tmp_path / "ns")
        assert package_root(directory, [str(tmp_path)]) == (directory, [])

    def test_with_no_root_above_only_regular_packages_count(self, tmp_path):
        # Defect: the walk climbs through every directory with no root to
        # stop it, or a root below the directory is taken as one above.
        self.write(tmp_path / "ns" / "pkg" / "__init__.py", "")
        directory = str(tmp_path / "ns" / "pkg")
        assert package_root(directory) == (str(tmp_path / "ns"), ["pkg"])
        assert package_root(directory, [directory + "/inner"]) == (
            str(tmp_path / "ns"),
            ["pkg"],
        )

    def test_a_root_matches_a_symlinked_spelling_of_it(self, tmp_path):
        # Defect: paths compare as spelled, so a workspace folder given by
        # its real path (Backends' project root) never matches a document
        # opened through a symlink, which then loses its namespace package.
        real = tmp_path / "real"
        (real / "ns").mkdir(parents=True)
        (tmp_path / "link").symlink_to(real)
        directory = str(tmp_path / "link" / "ns")
        assert package_root(directory, [str(real)]) == (
            str(tmp_path / "link"),
            ["ns"],
        )

    def test_module_names_count_namespace_packages_under_the_project(self, tmp_path):
        # Defects: the LSP names ns/a.lpy as the top-level module a; or it
        # drops ns/ from the search dirs, though no regular package holds it
        # and `lpy ns/a.lpy` runs it as a script with ns/ on sys.path, so a
        # bare sibling stops resolving; or it also searches the import root
        # (the project root), which that run never puts on sys.path, so a
        # root utils.py answers its `import utils` over src/utils.py; or it
        # gives it the package ns, though that run has no __package__, so a
        # relative require resolves where the run refuses it; or it leaves
        # out src, which an editable install puts on sys.path for every run.
        self.write(tmp_path / "src" / "sns" / "b.lpy", "")
        (tmp_path / "ns").mkdir()
        got = [
            pv.module_names(str(tmp_path.joinpath(*parts)), project_root=str(tmp_path))
            for parts in [("ns", "a.lpy"), ("src", "sns", "b.lpy")]
        ]
        src = tmp_path / "src"
        assert got == [
            ("ns.a", "", str(tmp_path), (str(tmp_path / "ns"), str(src))),
            ("sns.b", "", str(src), (str(src / "sns"), str(src))),
        ]

    def test_view_at_cursor_with_a_relative_require(self, tmp_path, monkeypatch):
        # An open buffer compiles as its shadow does: its import root on
        # sys.path while its macros load, and sys.path as it was after.
        pkg = tmp_path / "src" / "pvcurpkg"
        self.write(pkg / "__init__.py", "")
        self.write(pkg / "mac.lpy", "(defmacro twice [x] (return `(+ ~x ~x)))")
        names = ["pvcurpkg", "pvcurpkg.mac"]
        for name in names:
            monkeypatch.delitem(sys.modules, name, raising=False)
        src = "(require .mac *)\n(= yy (twice 2))"
        path = list(sys.path)
        try:
            view = pv.ViewCache().view_at_cursor(
                "u",
                1,
                src,
                *plus(locate(src, "yy"), 2),
                package="pvcurpkg",
                name="pvcurpkg.mod",
                search_dirs=(str(tmp_path / "src"),),
            )
        finally:
            for name in names:
                sys.modules.pop(name, None)
        assert view is not None and view.text.endswith("\nyy = 2 + 2")
        assert sys.path == path

    def test_relative_require_in_a_src_layout(self, tmp_path, monkeypatch):
        project, root = tmp_path / "project", tmp_path / "view"
        pkg = project / "src" / "pvsrcpkg"
        self.write(pkg / "__init__.py", "")
        self.write(pkg / "mac.lpy", "(defmacro twice [x] (return `(+ ~x ~x)))")
        self.write(pkg / "mod.lpy", "(require .mac *)\n(= y (twice 2))")
        names = ["pvsrcpkg", "pvsrcpkg.mac"]
        for name in names:
            monkeypatch.delitem(sys.modules, name, raising=False)
        path = list(sys.path)
        try:
            pv.write_shadows(str(project), str(root))
        finally:
            for name in names:
                sys.modules.pop(name, None)
        assert "y = 2 + 2" in (root / "src" / "pvsrcpkg" / "mod.py").read_text()
        assert sys.path == path

    def test_a_namespace_module_compiles_and_is_searched_as_its_script(
        self, tmp_path, monkeypatch
    ):
        # Defects: only the project root is on sys.path while its macros
        # run, so its sibling require loads no macros; or the project root is
        # searched too, which `lpy pvnsscripts/rel.lpy` never puts on
        # sys.path, so the relative require (refused by that run: no known
        # parent package) loads pvnsscripts.mac and the project root becomes
        # a shared root and a search path of the script's environment; or
        # scripts/ is no search path of the backend for its own files, so a
        # sibling (import ...) the script run resolves stops resolving; or it
        # is one for every file of the project, so a root module's (import
        # helper) resolves to scripts/helper, which `lpy m.lpy` never finds.
        project, root = tmp_path / "project", tmp_path / "view"
        scripts = project / "pvnsscripts"
        self.write(scripts / "mac.lpy", "(defmacro twice [x] (return `(+ ~x ~x)))")
        self.write(
            scripts / "pvnshelper.lpy", "(defmacro thrice [x] (return `(+ ~x ~x ~x)))"
        )
        self.write(scripts / "rel.lpy", "(require .mac *)\n(= y (twice 2))")
        self.write(scripts / "bare.lpy", "(require pvnshelper *)\n(= y (thrice 2))")
        names = ["pvnsscripts", "pvnsscripts.mac", "pvnshelper"]
        for name in names:
            monkeypatch.delitem(sys.modules, name, raising=False)
        manifest = pv.ShadowManifest()
        try:
            pv.write_shadows(str(project), str(root), manifest=manifest)
        finally:
            for name in names:
                sys.modules.pop(name, None)
        view = root / "pvnsscripts"
        assert (view / "rel.py").read_text().endswith("y = twice(2)")
        assert (view / "bare.py").read_text().endswith("y = 2 + 2 + 2")
        assert manifest.environments == {str(scripts): (str(scripts),)}
        config = json.loads(
            open(
                pv.write_pyright_config(
                    str(project),
                    str(root),
                    environments=manifest.environments,
                )
            ).read()
        )
        assert str(view) not in config["extraPaths"]
        assert str(scripts) not in config["extraPaths"]
        [env] = config["executionEnvironments"]
        assert env["root"] == str(view)
        assert env["extraPaths"] == [str(view), str(scripts)]
        # pyrefly: the nearest pyrefly.toml up from a file is its config.
        top = self.pyrefly_config(
            project,
            root,
            environments=manifest.environments,
        )
        assert str(view) not in top["search-path"]
        assert str(scripts) not in top["search-path"]
        with open(view / "pyrefly.toml", "rb") as f:
            nested = tomllib.load(f)
        assert nested["search-path"] == env["extraPaths"]
        assert {k: v for k, v in nested.items() if k != "search-path"} == {
            k: v for k, v in top.items() if k != "search-path"
        }
        assert pv.config_files(str(root), "pyrefly.toml") == sorted(
            [str(root / "pyrefly.toml"), str(view / "pyrefly.toml")]
        )
        # Once scripts/ holds no shadow, its environment and nested config go.
        for name in ["mac.lpy", "pvnshelper.lpy", "rel.lpy", "bare.lpy"]:
            (scripts / name).unlink()
        (project / "top.lpy").write_text("(= t 1)")
        pv.write_shadows(str(project), str(root), manifest=manifest)
        assert manifest.environments == {str(project): (str(project),)}
        self.pyrefly_config(
            project,
            root,
            environments=manifest.environments,
        )
        assert pv.config_files(str(root), "pyrefly.toml") == [
            str(root / "pyrefly.toml")
        ]

    def test_a_regular_package_in_a_namespace_directory_is_its_package_module(
        self, tmp_path, monkeypatch
    ):
        # Defect: pvtools/pvmypkg/a.lpy (a regular package in a namespace
        # directory) also searches pvtools/ (a root no run of the file has:
        # `import pvtools.pvmypkg.a` searches the project root alone), so
        # (require pvmypkg.mac *) loads macros there and the backend has an
        # environment for pvtools/, where the run fails.
        project, root = tmp_path / "project", tmp_path / "view"
        tools = project / "pvtools"
        pkg = tools / "pvmypkg"
        self.write(pkg / "__init__.py", "")
        self.write(pkg / "mac.lpy", "(defmacro twice [x] (return `(+ ~x ~x)))")
        self.write(pkg / "a.lpy", "(require pvtools.pvmypkg.mac *)\n(= y (twice 2))")
        self.write(pkg / "b.lpy", "(require pvmypkg.mac *)\n(= y (twice 2))")
        assert pv.module_names(str(pkg / "a.lpy"), project_root=str(project)) == (
            "pvtools.pvmypkg.a",
            "pvtools.pvmypkg",
            str(project),
            (str(project),),
        )
        names = ["pvmypkg", "pvmypkg.mac", "pvtools", "pvtools.pvmypkg"]
        names.append("pvtools.pvmypkg.mac")
        for name in names:
            monkeypatch.delitem(sys.modules, name, raising=False)
        manifest = pv.ShadowManifest()
        try:
            pv.write_shadows(str(project), str(root), manifest=manifest)
        finally:
            for name in names:
                sys.modules.pop(name, None)
        view = root / "pvtools" / "pvmypkg"
        assert (view / "a.py").read_text().endswith("y = 2 + 2")
        assert (view / "b.py").read_text().endswith("y = twice(2)")
        assert manifest.environments == {str(project): (str(project),)}
        path = pv.write_pyright_config(
            str(project),
            str(root),
            environments=manifest.environments,
        )
        assert "executionEnvironments" not in json.loads(open(path).read())

    def test_a_package_module_never_searches_its_own_directory(
        self, tmp_path, monkeypatch
    ):
        # Defect: src/pkg/m.lpy also searches src/pkg, so a bare sibling
        # (require pvbarehelper *) loads its macros while `import pkg.m`
        # raises ModuleNotFoundError for it, and the compile and the backend
        # disagree on what the module imports.
        project, root = tmp_path / "project", tmp_path / "view"
        pkg = project / "src" / "pvbarepkg"
        self.write(pkg / "__init__.py", "")
        self.write(pkg / "pvbarehelper.lpy", "(defmacro twice [x] (return `(+ ~x ~x)))")
        self.write(pkg / "bare.lpy", "(require pvbarehelper *)\n(= y (twice 2))")
        self.write(
            pkg / "qualified.lpy", "(require pvbarepkg.pvbarehelper *)\n(= y (twice 2))"
        )
        names = ["pvbarehelper", "pvbarepkg", "pvbarepkg.pvbarehelper"]
        for name in names:
            monkeypatch.delitem(sys.modules, name, raising=False)
        try:
            pv.write_shadows(str(project), str(root), manifest=pv.ShadowManifest())
        finally:
            for name in names:
                sys.modules.pop(name, None)
        view = root / "src" / "pvbarepkg"
        assert (view / "qualified.py").read_text().endswith("y = 2 + 2")
        assert (view / "bare.py").read_text().endswith("y = twice(2)")

    def test_src_layout_config_and_mirrored_py(self, tmp_path):
        # The package interior src/pkg is no search path, of the compile or
        # the backend: it would resolve the module's `import helper` (or a
        # src/pkg/json.py its `import json`) to that package's modules,
        # where Python imports it as pkg.m from src.
        project, root = tmp_path / "project", tmp_path / "view"
        pkg = project / "src" / "pkg"
        self.write(pkg / "__init__.py", "from .helper import h")
        self.write(pkg / "helper.py", "h = 1")
        self.write(pkg / "m.lpy", "(= m 1)")
        self.write(pkg / "notes.txt", "")
        manifest = pv.ShadowManifest()
        pv.write_shadows(str(project), str(root), manifest=manifest)
        src = str(project / "src")
        assert manifest.environments == {src: (src,)}
        for name in ["__init__.py", "helper.py"]:
            assert os.readlink(root / "src" / "pkg" / name) == str(pkg / name)

        path = pv.write_pyright_config(
            str(project),
            str(root),
            environments=manifest.environments,
        )
        config = json.loads(open(path).read())
        assert config["extraPaths"] == [str(root), str(root / "src"), str(project), src]
        assert config["executionEnvironments"] == [
            {
                "root": str(root / "src"),
                "extraPaths": [str(root / "src"), src],
            }
        ]

        (pkg / "helper.py").unlink()
        self.write(pkg / "__init__.lpy", "(= i 1)")  # replaces the linked .py
        pv.write_shadows(str(project), str(root), manifest=manifest)
        assert not os.path.lexists(root / "src" / "pkg" / "helper.py")
        init = root / "src" / "pkg" / "__init__.py"
        assert not init.is_symlink() and init.read_text() == "i = 1"
        assert (pkg / "__init__.py").read_text() == "from .helper import h"

    def test_a_plain_directory_inside_a_package_is_no_backend_search_path(
        self, tmp_path
    ):
        # Defect: src/pkg/res (no __init__ of its own, inside the regular
        # package src/pkg, so pkg.res from src) is searched as a script's
        # own directory, a backend search path: src/pkg/res/json.py would
        # answer `import json`, which `import pkg.res.x` never searches
        # there.  Likewise src/pkg/res/sub one level deeper.
        project, root = tmp_path / "project", tmp_path / "view"
        res = project / "src" / "pkg" / "res"
        self.write(project / "src" / "pkg" / "__init__.py", "")
        self.write(res / "x.lpy", "(= x 1)")
        self.write(res / "sub" / "y.lpy", "(= y 1)")
        manifest = pv.ShadowManifest()
        pv.write_shadows(str(project), str(root), manifest=manifest)
        src = str(project / "src")
        assert manifest.environments == {src: (src,)}
        path = pv.write_pyright_config(
            str(project),
            str(root),
            environments=manifest.environments,
        )
        config = json.loads(open(path).read())
        assert config["extraPaths"] == [str(root), str(root / "src"), str(project), src]
        assert config["executionEnvironments"] == [
            {
                "root": str(root / "src"),
                "extraPaths": [str(root / "src"), src],
            }
        ]

    def test_a_script_directory_is_searched_only_by_its_files(self, tmp_path):
        # Defects: src/pkg/my-res (no __init__, no identifier, inside the
        # regular package src/pkg) can be named by no import, so x.lpy is
        # the script `python src/pkg/my-res/x.py` runs, with my-res and the
        # installed src on sys.path: sharing its environment with every file
        # makes src/pkg/my-res/json.py answer every file's `import json`;
        # searching src/pkg or the project root for it, or dropping my-res or
        # src, mis-resolves its imports.
        project, root = tmp_path / "project", tmp_path / "view"
        src, res = project / "src", project / "src" / "pkg" / "my-res"
        self.write(src / "pkg" / "__init__.py", "")
        self.write(src / "pkg" / "m.lpy", "(= m 1)")
        self.write(res / "x.lpy", "(= x 1)")
        self.write(res / "json.py", "X = 1")
        assert pv.module_names(str(res / "x.lpy"), project_root=str(project)) == (
            "x",
            "",
            str(res),
            (str(res), str(src)),
        )
        manifest = pv.ShadowManifest()
        pv.write_shadows(str(project), str(root), manifest=manifest)
        assert manifest.environments[str(res)] == (str(res), str(src))
        view_res = root / "src" / "pkg" / "my-res"
        path = pv.write_pyright_config(
            str(project),
            str(root),
            environments=manifest.environments,
        )
        config = json.loads(open(path).read())
        assert config["extraPaths"] == [
            str(root),
            str(root / "src"),
            str(project),
            str(src),
        ]
        env, src_env = config["executionEnvironments"]
        assert env["root"] == str(view_res)
        assert env["extraPaths"][0] == str(view_res)
        assert str(res) in env["extraPaths"]
        assert src_env["root"] == str(root / "src")
        assert str(res) not in src_env["extraPaths"]
        assert str(view_res) not in src_env["extraPaths"]
        top = self.pyrefly_config(
            project,
            root,
            environments=manifest.environments,
        )
        assert str(res) not in top["search-path"]
        assert str(view_res) not in top["search-path"]
        with open(view_res / "pyrefly.toml", "rb") as f:
            assert tomllib.load(f)["search-path"] == env["extraPaths"]

    def test_an_import_roots_modules_are_linked_beside_its_packages(self, tmp_path):
        # Defect: an import root holding only shadowed packages has none of
        # its other modules in the view, so a pure-.py package beside them
        # resolves to the real directory, where pyrefly analyses it under
        # the project's own config (or none) and it cannot see the shadows.
        # A namespace package or a hidden directory is never linked.
        project, root = tmp_path / "project", tmp_path / "view"
        src = project / "src"
        for name, text in [
            ("lpkg/__init__.lpy", ""),
            ("lpkg/core.lpy", "(def ffff [a] (return (+ a 1)))"),
            ("purepkg/__init__.py", ""),
            ("purepkg/helper.py", "from lpkg.core import ffff\nX = ffff\n"),
            ("top.py", "t = 1"),
            ("top.pyi", "t: int"),
            ("nsdir/x.py", "x = 1"),
            (".hidden/__init__.py", ""),
        ]:
            self.write(src / name, text)
        view_src = root / "src"
        for _ in range(2):  # the second sweep keeps every link
            pv.write_shadows(str(project), str(root))
            for name in ["purepkg", "top.py", "top.pyi"]:
                assert os.readlink(view_src / name) == str(src / name)
            assert not os.path.lexists(view_src / "nsdir")
            assert not os.path.lexists(view_src / ".hidden")
            lpkg = view_src / "lpkg"
            assert lpkg.is_dir() and not lpkg.is_symlink()
            assert (lpkg / "core.py").is_file() and not (lpkg / "core.py").is_symlink()

    def test_a_package_that_lost_its_last_lpy_is_linked_at_once(self, tmp_path):
        # Defect: the package's view directory from the sweep before, still
        # real, is taken for one this sweep needs, so it is not linked, and
        # the orphan sweep then removes it: the package is missing from the
        # view until a third sweep.
        project, root = tmp_path / "project", tmp_path / "view"
        self.write(project / "top.lpy", "(= t 1)")
        self.write(project / "pkg" / "__init__.py", "")
        self.write(project / "pkg" / "a.py", "a = 1")
        self.write(project / "pkg" / "m.lpy", "(= m 1)")
        pv.write_shadows(str(project), str(root))
        assert (root / "pkg").is_dir() and not (root / "pkg").is_symlink()
        (project / "pkg" / "m.lpy").unlink()
        pv.write_shadows(str(project), str(root))
        assert os.readlink(root / "pkg") == str(project / "pkg")

    def test_a_package_of_failing_lpy_keeps_its_directory_link(
        self, tmp_path, monkeypatch
    ):
        # Defect: every sweep made the view directory real before it knew a
        # shadow would be written, so a package whose .lpy all fail to
        # compile lost its link, stood as an empty directory while compiling,
        # and was linked again afterwards.
        project, root = tmp_path / "project", tmp_path / "view"
        self.write(project / "top.lpy", "(= t 1)")
        self.write(project / "pkg" / "__init__.py", "")
        self.write(project / "pkg" / "a.py", "a = 1")
        self.write(project / "pkg" / "bad.lpy", "(= x")
        pv.write_shadows(str(project), str(root))
        assert os.readlink(root / "pkg") == str(project / "pkg")

        seen = []
        compile_buffer = pv.compile_buffer

        def watching(*args, **kwargs):
            seen.append(os.path.islink(root / "pkg"))
            return compile_buffer(*args, **kwargs)

        monkeypatch.setattr(pv, "compile_buffer", watching)
        pv.write_shadows(str(project), str(root), manifest=pv.ShadowManifest())
        assert seen == [True, True]  # bad.lpy, top.lpy
        assert os.readlink(root / "pkg") == str(project / "pkg")

    def test_a_cached_shadow_under_a_directory_link_is_not_kept(self, tmp_path):
        # Defect: a shadow path that lies under a link into the project is
        # taken for the shadow itself, since the project's file there is a
        # regular file.
        project, root = tmp_path / "project", tmp_path / "view"
        self.write(project / "pkg" / "__init__.py", "")
        self.write(project / "pkg" / "m.lpy", "(= m 1)")
        manifest = pv.ShadowManifest()
        pv.write_shadows(str(project), str(root), manifest=manifest)
        self.write(project / "pkg" / "m.py", "real = 1")
        shutil.rmtree(root / "pkg")
        (root / "pkg").symlink_to(project / "pkg")
        pv.write_shadows(str(project), str(root), manifest=manifest)
        shadow = root / "pkg" / "m.py"
        assert not (root / "pkg").is_symlink() and not shadow.is_symlink()
        assert shadow.read_text() == "m = 1"
        assert (project / "pkg" / "m.py").read_text() == "real = 1"

    def test_sweep_follows_no_directory_link_and_skips_environments(self, tmp_path):
        project, root = tmp_path / "project", tmp_path / "view"
        self.write(project / "top.lpy", "(= a 1)")
        (project / "sub").mkdir()
        (project / "sub" / "up").symlink_to(project)  # a loop
        (project / "alias.lpy").symlink_to(project / "top.lpy")
        self.write(project / "venv" / "pyvenv.cfg", "")
        for skipped in ["venv", "node_modules", "__pycache__", "site-packages"]:
            self.write(project / skipped / "dep.lpy", "(= d 1)")
        # Each path to a file is its own module, so the link gets a shadow too.
        assert pv.write_shadows(str(project), str(root)).shadows == [
            str(root / "alias.py"),
            str(root / "top.py"),
        ]

    def overlay_layout(self, project):
        pkg = project / "src" / "pkg"
        imports = "(import pkg.util.x)\n(import pkg.stub)\n(from pkg.util.x [val])"
        for name, text in [
            ("__init__.py", ""),
            ("m.lpy", imports),
            ("stub.pyi", "s: int"),
            ("py.typed", ""),
            ("fast.cpython-314-x86_64-linux-gnu.so", ""),
            ("util/__init__.py", ""),
            ("util/x.py", "val = 1"),
            ("sub/__init__.py", ""),
            ("sub/other.py", ""),
            ("sub/deep/inner.lpy", "(= i 1)"),
            ("__pycache__/m.cpython-314.pyc", ""),
            (".cache/c.lpy", "(= c 1)"),
        ]:
            self.write(pkg / name, text)
        return pkg

    def test_package_is_overlaid_whole(self, tmp_path):
        # The view's pkg is a regular package that hides the real one, so
        # whatever it lacks of the real package does not resolve.
        project, root = tmp_path / "project", tmp_path / "view"
        pkg = self.overlay_layout(project)
        view = root / "src" / "pkg"
        pv.write_shadows(str(project), str(root))
        linked = ["__init__.py", "stub.pyi", "py.typed", "util", "sub/other.py"]
        for name in linked + ["fast.cpython-314-x86_64-linux-gnu.so"]:
            assert os.readlink(view / name) == str(pkg / name)
        assert not (view / "sub").is_symlink()
        assert (view / "sub" / "deep" / "inner.py").read_text() == "i = 1"
        for name in ["m.lpy", "__pycache__", ".cache", "sub/deep/inner.lpy"]:
            assert not os.path.lexists(view / name)

        # A .lpy in a linked directory makes it a view directory; its shadow
        # is never written through the old link into the project.
        self.write(pkg / "util" / "y.lpy", "(= y 1)")
        pv.write_shadows(str(project), str(root))
        assert not (view / "util").is_symlink()
        assert (view / "util" / "y.py").read_text() == "y = 1"
        assert os.readlink(view / "util" / "x.py") == str(pkg / "util" / "x.py")
        assert sorted(os.listdir(pkg / "util")) == ["__init__.py", "x.py", "y.lpy"]

        # With no .lpy left under them, view directories become links again.
        (pkg / "util" / "y.lpy").unlink()
        (pkg / "sub" / "deep" / "inner.lpy").unlink()
        pv.write_shadows(str(project), str(root))
        assert os.readlink(view / "util") == str(pkg / "util")
        assert os.readlink(view / "sub") == str(pkg / "sub")
        assert sorted(os.listdir(pkg / "sub")) == ["__init__.py", "deep", "other.py"]

        # With no shadow left, every link goes, unlinked and never followed.
        (pkg / "m.lpy").unlink()
        pv.write_shadows(str(project), str(root))
        assert not os.path.lexists(root / "src")
        assert sorted(os.listdir(pkg / "util")) == ["__init__.py", "x.py"]
        assert (pkg / "sub" / "other.py").exists()

    def basedpyright_messages(
        self, tmp_path, project, root, files=("src/pkg/m.py",), **settings
    ):
        """basedpyright's diagnostics on files of the view (src/pkg/m.py),
        under the view root's config as written plus settings."""
        manifest = pv.ShadowManifest()
        pv.write_shadows(str(project), str(root), manifest=manifest)
        path = pv.write_pyright_config(
            str(project),
            str(root),
            environments=manifest.environments,
        )
        config = json.loads(open(path).read())
        config.update(settings, reportMissingImports="error")
        check = tmp_path / "check" / "pyrightconfig.json"
        self.write(check, json.dumps(config))
        out = subprocess.run(
            ["basedpyright", "--outputjson", "-p", str(check)]
            + [str(root / name) for name in files],
            capture_output=True,
            text=True,
            timeout=60,
        )
        report = json.loads(out.stdout)
        assert report["summary"]["filesAnalyzed"] == len(files)
        return [d["message"] for d in report["generalDiagnostics"]]

    @pytest.mark.skipif(not shutil.which("basedpyright"), reason="no basedpyright")
    def test_overlay_resolves_for_basedpyright(self, tmp_path):
        # The config as written, so pkg is found only in the view's src.
        project, root = tmp_path / "project", tmp_path / "view"
        self.overlay_layout(project)
        # pkg.stub is stub-only by design: it resolves, with no source.
        assert (
            self.basedpyright_messages(
                tmp_path, project, root, reportMissingModuleSource="none"
            )
            == []
        )

    @pytest.mark.skipif(not shutil.which("basedpyright"), reason="no basedpyright")
    def test_src_layout_modules_outside_the_view_resolve(self, tmp_path):
        # pkg is overlaid in the view and a pure-.py package, a top-level .py
        # and a stub-only module beside it are linked in, so they resolve,
        # while pkg.m2, which exists only as .lpy, resolves from the view's
        # shadow, and so does pkg.both, whose stale .py beside it in the real
        # src lacks new.
        project, root = tmp_path / "project", tmp_path / "view"
        src = project / "src"
        imports = (
            "(import other)\n(import single)\n(import stubbed)\n(import pkg.m2)\n"
            "(from pkg.both [new])"
        )
        for name, text in [
            ("pkg/__init__.py", ""),
            ("pkg/m.lpy", imports),
            ("pkg/m2.lpy", "(= z 1)"),
            ("pkg/both.lpy", "(= new 1)"),
            ("pkg/both.py", "old = 1"),
            ("other/__init__.py", ""),
            ("single.py", "s = 1"),
            ("stubbed.pyi", "t: int"),
        ]:
            self.write(src / name, text)
        messages = self.basedpyright_messages(
            tmp_path,
            project,
            root,
            reportMissingModuleSource="none",
            reportAttributeAccessIssue="error",
        )
        assert messages == []

    UTILS_FILES = ("scripts/x.py", "src/pkg/m.py", "src/pkg/my-res/x.py")

    def src_utils_layout(self, project):
        """src/pkg/m.lpy and the scripts src/pkg/my-res/x.lpy and
        scripts/x.lpy importing utils, which both the project root and src
        hold; a top-level .lpy gives the root a shadow, so its utils.py is
        linked into the view root."""
        uses = "(import utils)\n(= w utils.SRC-ONLY)"
        for name, text in [
            ("top.lpy", "(= t 1)"),
            ("utils.py", "ROOT_ONLY = 1"),
            ("src/utils.py", "SRC_ONLY = 1"),
            ("src/pkg/__init__.py", ""),
            ("src/pkg/m.lpy", uses),
            ("src/pkg/my-res/x.lpy", uses),
            ("scripts/x.lpy", uses),
        ]:
            self.write(project / name, text)

    def test_a_shared_roots_modules_search_it_before_the_project_root(self, tmp_path):
        # Defects: src (a shared root) takes the default list, which leads
        # with the view root and the project root, so the backend resolves
        # the `import utils` of src/pkg/m.lpy to the root utils.py, where its
        # run (`python -m pkg.m` from src) resolves src/utils.py; or a
        # script (src/pkg/my-res/x.lpy, scripts/x.lpy, top.lpy) makes its
        # import root (the project root) a shared root, or scripts/x.lpy
        # searches it as its own, so a script searches the project root
        # before src, where `lpy` with src installed never searches the
        # project root at all; or an environment searches the project root
        # last though none of its runs does.
        project, root = tmp_path / "project", tmp_path / "view"
        self.src_utils_layout(project)
        manifest = pv.ShadowManifest()
        pv.write_shadows(str(project), str(root), manifest=manifest)
        src, res = str(project / "src"), str(project / "src" / "pkg" / "my-res")
        scripts = str(project / "scripts")
        view_res = str(root / "src" / "pkg" / "my-res")
        view_scripts = str(root / "scripts")
        assert os.readlink(root / "utils.py") == str(project / "utils.py")
        assert pv.shared_import_roots(str(project)) == [src]
        settings = {
            "environments": manifest.environments,
        }
        path = pv.write_pyright_config(str(project), str(root), **settings)
        config = json.loads(open(path).read())
        # A module of the project root still searches it first.
        assert config["extraPaths"] == [str(root), str(root / "src"), str(project), src]
        res_env, scripts_env, src_env = config["executionEnvironments"]
        assert src_env["root"] == str(root / "src")
        assert src_env["extraPaths"] == [str(root / "src"), src]
        assert res_env["root"] == view_res
        assert res_env["extraPaths"] == [view_res, str(root / "src"), res, src]
        assert scripts_env["root"] == view_scripts
        assert scripts_env["extraPaths"] == [
            view_scripts,
            str(root / "src"),
            scripts,
            src,
        ]
        self.pyrefly_config(project, root, **settings)
        for env in [src_env, res_env, scripts_env]:
            with open(os.path.join(env["root"], "pyrefly.toml"), "rb") as f:
                assert tomllib.load(f)["search-path"] == env["extraPaths"]

    @pytest.mark.skipif(not shutil.which("basedpyright"), reason="no basedpyright")
    def test_a_shared_roots_import_resolves_there_for_basedpyright(self, tmp_path):
        project, root = tmp_path / "project", tmp_path / "view"
        self.src_utils_layout(project)
        messages = self.basedpyright_messages(
            tmp_path,
            project,
            root,
            files=self.UTILS_FILES,
            reportAttributeAccessIssue="error",
        )
        assert messages == []

    ROOTONLY_FILES = ("top.py", "scripts/x.py", "src/pkg/m.py", "src/pkg/my-res/x.py")

    def rootonly_layout(self, project):
        """A root rootonly.py, which no regular package beside it makes a
        shared root, and ROOTONLY_FILES' .lpy sources each importing it."""
        self.write(project / "rootonly.py", "R = 1")
        self.write(project / "src" / "pkg" / "__init__.py", "")
        for name in self.ROOTONLY_FILES:
            self.write(project / name.replace(".py", ".lpy"), "(import rootonly)")

    def pyrefly_errors(self, tmp_path, project, root, files):
        """(view-relative path, error name) of pyrefly's errors on files of
        the view, each checked in a run of its own under its nearest
        pyrefly.toml, as the language server takes it, so one file's result
        never depends on what another run analysed."""
        manifest = pv.ShadowManifest()
        pv.write_shadows(str(project), str(root), manifest=manifest)
        pv.write_pyrefly_config(
            str(project), str(root), environments=manifest.environments
        )
        found = []
        for name in files:
            out = subprocess.run(
                ["pyrefly", "check", "--output-format", "json", str(root / name)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=tmp_path,
            )
            found += [
                (os.path.relpath(tmp_path / e["path"], root), e["name"])
                for e in json.loads(out.stdout)["errors"]
            ]
        return sorted(found)

    @pytest.mark.skipif(not shutil.which("basedpyright"), reason="no basedpyright")
    def test_a_root_only_module_is_missing_where_no_run_searches_the_root(
        self, tmp_path
    ):
        # Defect: every environment searches the project root last, so
        # `import rootonly` (a root rootonly.py) resolves for src/pkg/m.lpy,
        # src/pkg/my-res/x.lpy and scripts/x.lpy, whose runs (`python -m
        # pkg.m` from src, `lpy` of the script with src installed) raise
        # ModuleNotFoundError; a root module (top.lpy) still resolves it.
        project, root = tmp_path / "project", tmp_path / "view"
        self.rootonly_layout(project)
        messages = self.basedpyright_messages(
            tmp_path, project, root, files=self.ROOTONLY_FILES
        )
        assert messages == ['Import "rootonly" could not be resolved'] * 3

    @pytest.mark.skipif(not shutil.which("pyrefly"), reason="no pyrefly")
    def test_a_root_only_module_is_missing_where_no_run_searches_the_root_pyrefly(
        self, tmp_path
    ):
        # The same defect under pyrefly's per-file configs.
        project, root = tmp_path / "project", tmp_path / "view"
        self.rootonly_layout(project)
        errors = self.pyrefly_errors(tmp_path, project, root, self.ROOTONLY_FILES)
        assert errors == [(name, "missing-import") for name in self.ROOTONLY_FILES[1:]]

    def test_a_scripts_compile_searches_a_flat_layouts_root(
        self, tmp_path, monkeypatch
    ):
        # Defect: a script's compile searches its own directory alone, while
        # an editable install of a flat layout (a regular package at the
        # root) puts the project root on sys.path for every run, so
        # (require flatpkgrv.mac *) in scripts/x.lpy loads no macros, though
        # the backend resolves the import.
        project, root = tmp_path / "project", tmp_path / "view"
        self.write(project / "flatpkgrv" / "__init__.py", "")
        self.write(
            project / "flatpkgrv" / "mac.lpy",
            "(defmacro twice [x] (return `(+ ~x ~x)))",
        )
        self.write(
            project / "scripts" / "x.lpy", "(require flatpkgrv.mac *)\n(= y (twice 2))"
        )
        names = ["flatpkgrv", "flatpkgrv.mac"]
        for name in names:
            monkeypatch.delitem(sys.modules, name, raising=False)
        manifest = pv.ShadowManifest()
        try:
            pv.write_shadows(str(project), str(root), manifest=manifest)
        finally:
            for name in names:
                sys.modules.pop(name, None)
        assert (root / "scripts" / "x.py").read_text().endswith("y = 2 + 2")
        scripts = str(project / "scripts")
        assert manifest.environments[scripts] == (scripts, str(project))

    def test_a_src_layout_never_shares_the_project_root(self, tmp_path):
        # Defect: a regular package at the root (tests/__init__.py, common in
        # a src layout) shares the project root too, so src/pkg/m.lpy
        # searches it and `import rootonly` resolves, where an editable
        # install of the src layout puts src alone on sys.path and
        # `python -m pkg.m` raises ModuleNotFoundError.
        project = tmp_path / "project"
        self.write(project / "src" / "pkg" / "__init__.py", "")
        self.write(project / "tests" / "__init__.py", "")
        src = str(project / "src")
        assert pv.shared_import_roots(str(project)) == [src]
        names = pv.module_names(
            str(project / "src" / "pkg" / "m.lpy"), project_root=str(project)
        )
        assert names[3] == (src,)
        # With no src the root package makes it a flat layout.
        (project / "src" / "pkg" / "__init__.py").unlink()
        for directory in [project / "src" / "pkg", project / "src"]:
            directory.rmdir()
        assert pv.shared_import_roots(str(project)) == [str(project)]

    # scripts/run.lpy importing a module of a shared root that holds no .lpy:
    # a pure-.py regular package mylib at the root (the flat layout), or
    # nsp.mod from a src holding only the namespace package nsp.
    SHARED_LAYOUTS = {
        "flat": [
            ("mylib/__init__.py", "M = 1"),
            ("scripts/run.lpy", "(import mylib)\n(= v mylib.M)"),
        ],
        "src": [
            ("src/nsp/mod.py", "N = 1"),
            ("scripts/run.lpy", "(import nsp.mod)\n(= v nsp.mod.N)"),
        ],
    }

    def test_a_src_of_no_python_is_no_src_layout(self, tmp_path, monkeypatch):
        # Defect: any src directory makes a src layout, so a mixed repo
        # (src/lib.rs beside the root package mypkg) never shares the root
        # and (require mypkg.mac *) in scripts/x.lpy loads no macros, where
        # its run with the project installed imports mypkg.
        project, root = tmp_path / "project", tmp_path / "view"
        self.write(project / "src" / "lib.rs", "")
        self.write(project / "src" / "core" / "mod.rs", "")
        self.write(project / "src" / "__pycache__" / "x.cpython-315.pyc", "")
        self.write(project / "mypkg" / "__init__.py", "")
        self.write(
            project / "mypkg" / "mac.lpy", "(defmacro twice [x] (return `(+ ~x ~x)))"
        )
        self.write(
            project / "scripts" / "x.lpy", "(require mypkg.mac *)\n(= y (twice 2))"
        )
        assert pv.shared_import_roots(str(project)) == [str(project)]
        names = ["mypkg", "mypkg.mac"]
        for name in names:
            monkeypatch.delitem(sys.modules, name, raising=False)
        try:
            pv.write_shadows(str(project), str(root))
        finally:
            for name in names:
                sys.modules.pop(name, None)
        assert (root / "scripts" / "x.py").read_text().endswith("y = 2 + 2")
        # A module file at any depth makes it one: a namespace package
        # src/acme (no __init__) holding the regular package acme.tool.
        self.write(project / "src" / "acme" / "tool" / "__init__.py", "")
        assert pv.refresh_src_layout(str(project))
        assert pv.shared_import_roots(str(project)) == [str(project / "src")]

    def test_a_package_linked_into_src_makes_a_src_layout(self, tmp_path):
        # Defect: the walk follows no directory link, so src/acme ->
        # ../shared/acme (a package FileFinder imports through the link) is
        # no Python code and the root is shared instead, while
        # renaming-file-p counts a file under it; a link back up must not
        # loop.
        project = tmp_path / "project"
        self.write(project / "shared" / "acme" / "__init__.py", "")
        self.write(project / "tests" / "__init__.py", "")
        (project / "src").mkdir()
        (project / "src" / "acme").symlink_to(project / "shared" / "acme")
        (project / "src" / "loop").symlink_to(project / "src")
        assert pv.shared_import_roots(str(project)) == [str(project / "src")]

    def test_the_src_layout_walks_only_when_its_answer_can_have_moved(
        self, tmp_path, monkeypatch
    ):
        # Defects: every module-names call walks the whole src, on the LSP
        # loop per request (11 ms on a src of 4,800 .tsx files); or a
        # remembered src layout outlives the file that made it one (a
        # deleted src/pkg), so the root is never shared again.
        walks = []
        real_walk = os.walk

        def counted_walk(top, *args, **kwargs):
            walks.append(top)
            return real_walk(top, *args, **kwargs)

        monkeypatch.setattr(pv.os, "walk", counted_walk)
        other, python = tmp_path / "other", tmp_path / "python"
        for project in [other, python]:
            self.write(project / "mypkg" / "__init__.py", "")
        self.write(other / "src" / "lib.rs", "")
        self.write(python / "src" / "pkg" / "m.py", "")
        for _ in range(2):
            assert pv.shared_import_roots(str(other)) == [str(other)]
            assert pv.shared_import_roots(str(python)) == [str(python / "src")]
        assert len(walks) == 2
        (python / "src" / "pkg" / "m.py").unlink()
        assert pv.shared_import_roots(str(python)) == [str(python)]
        assert len(walks) == 3

    def test_a_rechecked_no_python_answer_still_reports_its_flip(self, tmp_path):
        # Defect: an open or save in one project drops every project's
        # no-Python answer, so when b/src/x.py is created afterwards,
        # refresh has nothing to compare and reports no change, and b's
        # open documents keep searching b's root.
        a, b = tmp_path / "a", tmp_path / "b"
        for project in [a, b]:
            self.write(project / "mypkg" / "__init__.py", "")
            self.write(project / "src" / "lib.rs", "")
            assert pv.shared_import_roots(str(project)) == [str(project)]
        pv.forget_false_src_layouts()
        self.write(b / "src" / "x.py", "")
        assert pv.refresh_src_layout(str(b))
        assert pv.shared_import_roots(str(b)) == [str(b / "src")]

    def shared_layout_without_lpy(self, project, layout):
        for name, text in self.SHARED_LAYOUTS[layout]:
            self.write(project / name, text)

    @pytest.mark.parametrize("layout", ["flat", "src"])
    def test_shared_roots_come_from_the_layout_not_the_shadows(self, tmp_path, layout):
        # Defects: the shared roots are read off the .lpy shadows' import
        # roots, so a pure-.py package at the root (mylib) or a src holding
        # only namespace packages is no search path of scripts/run.lpy,
        # where `lpy scripts/run.lpy` with the project installed imports it.
        project, root = tmp_path / "project", tmp_path / "view"
        self.shared_layout_without_lpy(project, layout)
        manifest = pv.ShadowManifest()
        pv.write_shadows(str(project), str(root), manifest=manifest)
        scripts = str(project / "scripts")
        shared = str(project / "src") if layout == "src" else str(project)
        view_shared = str(root / "src") if layout == "src" else str(root)
        assert manifest.environments == {scripts: (scripts, shared)}
        config = json.loads(
            open(
                pv.write_pyright_config(
                    str(project), str(root), environments=manifest.environments
                )
            ).read()
        )
        default = list(dict.fromkeys([str(root), view_shared, str(project), shared]))
        assert config["extraPaths"] == default
        [env] = config["executionEnvironments"]
        assert env["extraPaths"] == [
            str(root / "scripts"),
            view_shared,
            scripts,
            shared,
        ]

    @pytest.mark.skipif(not shutil.which("basedpyright"), reason="no basedpyright")
    @pytest.mark.parametrize("layout", ["flat", "src"])
    def test_shared_roots_from_the_layout_resolve_for_basedpyright(
        self, tmp_path, layout
    ):
        project, root = tmp_path / "project", tmp_path / "view"
        self.shared_layout_without_lpy(project, layout)
        messages = self.basedpyright_messages(
            tmp_path,
            project,
            root,
            files=("scripts/run.py",),
            reportAttributeAccessIssue="error",
        )
        assert messages == []

    @pytest.mark.skipif(not shutil.which("pyrefly"), reason="no pyrefly")
    @pytest.mark.parametrize("layout", ["flat", "src"])
    def test_shared_roots_from_the_layout_resolve_for_pyrefly(self, tmp_path, layout):
        # A deliberate error proves pyrefly checked the file.
        project, root = tmp_path / "project", tmp_path / "view"
        self.shared_layout_without_lpy(project, layout)
        with open(project / "scripts" / "run.lpy", "a") as f:
            f.write('\n(= bad (+ 1 "a"))')
        errors = self.pyrefly_errors(tmp_path, project, root, ["scripts/run.py"])
        assert errors == [("scripts/run.py", "unsupported-operation")]

    @pytest.mark.skipif(not shutil.which("pyrefly"), reason="no pyrefly")
    def test_a_shared_roots_import_resolves_there_for_pyrefly(self, tmp_path):
        # Each file is checked under its nearest pyrefly.toml, as the
        # language server takes it.  A deliberate error in each file must be
        # reported, so a file pyrefly never checked cannot pass as clean.
        project, root = tmp_path / "project", tmp_path / "view"
        self.src_utils_layout(project)
        for name in self.UTILS_FILES:
            with open(project / name.replace(".py", ".lpy"), "a") as f:
                f.write('\n(= bad (+ 1 "a"))')
        manifest = pv.ShadowManifest()
        pv.write_shadows(str(project), str(root), manifest=manifest)
        pv.write_pyrefly_config(
            str(project),
            str(root),
            environments=manifest.environments,
        )
        out = subprocess.run(
            ["pyrefly", "check", "--output-format", "json"]
            + [str(root / name) for name in self.UTILS_FILES],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=tmp_path,
        )
        errors = json.loads(out.stdout)["errors"]
        assert sorted(
            (os.path.relpath(tmp_path / e["path"], root), e["name"]) for e in errors
        ) == [(name, "unsupported-operation") for name in self.UTILS_FILES]

    def snapshot(self, top):
        """Every entry under top, links unfollowed: kind and content."""
        tree = {}
        for dirpath, dirnames, filenames in os.walk(top):
            for name in dirnames + filenames:
                path = os.path.join(dirpath, name)
                if os.path.islink(path):
                    entry = ("link", os.readlink(path))
                elif os.path.isdir(path):
                    entry = ("dir",)
                else:
                    with open(path, "rb") as f:
                        entry = ("file", f.read())
                tree[os.path.relpath(path, top)] = entry
        return tree

    def util_layout(self, project):
        for name, text in [
            ("pkg/__init__.py", ""),
            ("pkg/m.lpy", "(= m 1)"),
            ("pkg/util/__init__.py", ""),
            ("pkg/util/x.py", "val = 1"),
            ("pkg/util/y.lpy", "(= y 1)"),
        ]:
            self.write(project / name, text)

    def test_servers_sharing_a_view_root_never_touch_the_project(
        self, tmp_path, monkeypatch
    ):
        # "fresh" sees pkg/util/y.lpy and writes view/pkg/util/y.py; "stale"
        # does not, so it links view/pkg/util into the project.  "stale"
        # starts while "fresh" pauses with view/pkg/util a real directory:
        # under the lock it waits, so "fresh" writes its shadow (the link
        # guard alone keeps the project intact but loses the shadow).
        real_sources, real_dir = pv.project_sources, pv._real_view_dir
        paused = threading.Event()

        def sources(project_root):
            found = real_sources(project_root)
            if threading.current_thread().name == "stale":
                found = [p for p in found if not p.endswith("y.lpy")]
            return found

        def slow_dir(root, directory):
            real_dir(root, directory)
            util = directory.endswith(os.path.join("pkg", "util"))
            fresh = threading.current_thread().name == "fresh"
            if fresh and util and not paused.is_set():
                paused.set()
                time.sleep(0.15)

        monkeypatch.setattr(pv, "project_sources", sources)
        monkeypatch.setattr(pv, "_real_view_dir", slow_dir)
        for repeat in range(5):
            project, root = tmp_path / f"project{repeat}", tmp_path / f"view{repeat}"
            self.util_layout(project)
            before = self.snapshot(project)
            paused.clear()
            errors, shadows = [], {}

            def run(wait, project=project, root=root):
                try:
                    assert not wait or paused.wait(5)
                    name = threading.current_thread().name
                    shadows[name] = pv.write_shadows(str(project), str(root)).shadows
                except BaseException as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=run, args=(False,), name="fresh"),
                threading.Thread(target=run, args=(True,), name="stale"),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
            assert not any(thread.is_alive() for thread in threads)
            assert errors == []
            assert str(root / "pkg" / "util" / "y.py") in shadows["fresh"]
            assert self.snapshot(project) == before

    def test_no_write_or_removal_through_a_foreign_link(self, tmp_path, monkeypatch):
        # A writer that takes no lock links view/pkg/util into the project
        # right after it was made a real directory.
        project, root = tmp_path / "project", tmp_path / "view"
        self.util_layout(project)
        util, view_util = project / "pkg" / "util", str(root / "pkg" / "util")
        real_dir = pv._real_view_dir

        def foreign(root, directory):
            real_dir(root, directory)
            if directory == view_util:
                shutil.rmtree(directory)
                os.symlink(util, directory)

        monkeypatch.setattr(pv, "_real_view_dir", foreign)
        before = self.snapshot(project)
        pv.write_shadows(str(project), str(root))
        assert self.snapshot(project) == before

    @pytest.mark.parametrize("where", ["same", "inside", "around"])
    def test_overlapping_view_root_is_refused(self, tmp_path, where):
        project = tmp_path / "project"
        root = {
            "same": project,
            "inside": project / ".cache" / "view",
            "around": tmp_path,
        }[where]
        self.write(project / "top.lpy", "(= a 1)")
        self.write(project / "keep.py", "k = 1")
        before = self.snapshot(project)
        with pytest.raises(ValueError, match="overlaps"):
            pv.write_shadows(str(project), str(root))
        with pytest.raises(ValueError, match="overlaps"):
            pv.write_pyright_config(str(project), str(root))
        assert self.snapshot(project) == before

    def test_shadow_replaces_a_link_of_the_same_text(self, tmp_path):
        project, root = tmp_path / "project", tmp_path / "view"
        pkg = project / "pkg"
        self.write(pkg / "__init__.py", "")
        self.write(pkg / "a.lpy", "(= a 1)")
        self.write(pkg / "m.py", "m = 1")
        pv.write_shadows(str(project), str(root))
        shadow = root / "pkg" / "m.py"
        assert shadow.is_symlink()
        self.write(pkg / "m.lpy", "(= m 1)")  # compiles to m.py's very text
        assert str(shadow) in pv.write_shadows(str(project), str(root)).shadows
        assert not shadow.is_symlink() and shadow.read_text() == "m = 1"
        assert (pkg / "m.py").read_text() == "m = 1"

    def test_stale_temporaries_and_empty_directories_go(self, tmp_path):
        project, root = tmp_path / "project", tmp_path / "view"
        self.write(project / "top.lpy", "(= a 1)")
        self.write(project / "lib" / "m.lpy", "(= m 1)")
        pv.write_shadows(str(project), str(root))
        old = time.time() - 2 * 3600
        self.write(root / "tmpdead.tmp", "")
        os.utime(root / "tmpdead.tmp", (old, old))
        (root / ".top.py.dead.tmp").symlink_to(project / "top.lpy")
        os.utime(root / ".top.py.dead.tmp", (old, old), follow_symlinks=False)
        self.write(root / "tmplive.tmp", "theirs")
        (project / "lib" / "m.lpy").unlink()
        pv.write_shadows(str(project), str(root))
        assert sorted(os.listdir(root)) == [".lock", "tmplive.tmp", "top.py"]

    def test_unreadable_source_keeps_its_shadow(self, tmp_path):
        if os.geteuid() == 0:
            pytest.skip("root reads the unreadable file")
        project, root = tmp_path / "project", tmp_path / "view"
        self.write(project / "top.lpy", "(= a 1)")
        shadows = [str(root / "top.py")]
        assert pv.write_shadows(str(project), str(root)).shadows == shadows
        (project / "top.lpy").chmod(0)
        try:
            assert pv.write_shadows(str(project), str(root)).shadows == shadows
        finally:
            (project / "top.lpy").chmod(0o600)
        assert (root / "top.py").read_text() == "a = 1"

    def test_one_unwritable_file_leaves_the_rest(self, tmp_path):
        if os.geteuid() == 0:
            pytest.skip("root writes through the read-only directory")
        project, root = tmp_path / "project", tmp_path / "view"
        self.write(project / "top.lpy", "(= a 1)")
        self.write(project / "locked" / "m.lpy", "(= m 1)")
        (root / "locked").mkdir(parents=True)
        self.write(root / "locked" / "keep.txt", "")  # not an empty dir to sweep
        (root / "locked").chmod(0o500)
        # Another writer's temporary file beside a shadow is not touched.
        self.write(root / "top.py.tmp", "theirs")
        manifest = pv.ShadowManifest()
        try:
            shadows = pv.write_shadows(
                str(project), str(root), manifest=manifest
            ).shadows
        finally:
            (root / "locked").chmod(0o700)
        assert shadows == [str(root / "top.py")]
        assert list(manifest.sources) == [str(project / "top.lpy")]
        assert (root / "top.py.tmp").read_text() == "theirs"
        assert os.listdir(root / "locked") == ["keep.txt"]
        # Writable again: the next sweep writes the file it missed.
        pv.write_shadows(str(project), str(root), manifest=manifest)
        assert (root / "locked" / "m.py").read_text() == "m = 1"

    def test_written_updated_and_removed(self, tmp_path, monkeypatch):
        project, root = tmp_path / "project", tmp_path / "view"
        self.write(project / "pkg" / "mod.lpy", "(= my-x 1)")
        self.write(project / "top.lpy", "(= a 1)")
        self.write(project / "bad.lpy", "(= 1 2)")
        self.write(project / ".hidden" / "h.lpy", "(= h 1)")
        self.write(root / "pyrightconfig.json", "{}")
        manifest = pv.ShadowManifest()
        shadows = pv.write_shadows(str(project), str(root), manifest=manifest).shadows
        mod, top, bad = root / "pkg" / "mod.py", root / "top.py", root / "bad.py"
        # A module whose only form fails still has its (empty) shadow, so
        # an import of it resolves.
        assert shadows == sorted([str(mod), str(top), str(bad)])
        assert mod.read_text() == "my_x = 1"
        assert bad.read_text() == ""

        calls = []
        real = pv.compile_buffer
        monkeypatch.setattr(
            pv, "compile_buffer", lambda s, **kw: calls.append(s) or real(s, **kw)
        )
        stamp = mod.stat().st_mtime_ns
        pv.write_shadows(str(project), str(root), manifest=manifest)
        assert calls == []  # nothing changed, nothing compiled
        # A new mtime with the same text compiles but leaves the shadow alone.
        os.utime(project / "pkg" / "mod.lpy", ns=(stamp + 10**9, stamp + 10**9))
        pv.write_shadows(str(project), str(root), manifest=manifest)
        assert calls == ["(= my-x 1)"]
        assert mod.stat().st_mtime_ns == stamp

        (project / "top.lpy").unlink()
        (project / "bad.lpy").unlink()
        self.write(project / "pkg" / "mod.lpy", "(= 1 1")  # stops parsing
        os.utime(project / "pkg" / "mod.lpy", ns=(stamp + 2 * 10**9,) * 2)
        assert (
            pv.write_shadows(str(project), str(root), manifest=manifest).shadows == []
        )
        assert not mod.exists() and not top.exists() and not bad.exists()
        assert (root / "pyrightconfig.json").read_text() == "{}"

    def test_view_path(self, tmp_path):
        project = str(tmp_path / "p")
        lpy = os.path.join(project, "pkg", "m.lpy")
        assert pv.view_path(project, "/v", lpy) == os.path.join("/v", "pkg", "m.py")
        assert pv.view_path(project, "/v", str(tmp_path / "elsewhere.lpy")) is None

    def test_view_root_for(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        project.mkdir()
        link = tmp_path / "link"
        link.symlink_to(project)
        digest = hashlib.sha256(os.path.realpath(project).encode()).hexdigest()[:16]
        root = pv.view_root_for(str(link), cache_home=str(tmp_path / "c"))
        expected = tmp_path / "c" / "lispython" / "lsp-view" / digest
        assert os.fspath(root) == str(expected)
        assert root.python == sys.executable
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "x"))
        assert os.fspath(pv.view_root_for(str(project))).startswith(
            str(tmp_path / "x" / "lispython")
        )

    def test_pyright_config_leaves_out_import_roots_above_the_project(self, tmp_path):
        # A project inside a package: its modules' import root lies above it.
        project, root = tmp_path / "pkg" / "project", tmp_path / "view"
        project.mkdir(parents=True)
        path = pv.write_pyright_config(
            str(project),
            str(root),
            environments={str(tmp_path): (str(tmp_path),)},
        )
        config = json.loads(open(path).read())
        assert config["extraPaths"] == [str(root), str(project)]
        assert "executionEnvironments" not in config

    def test_overlap_check_takes_two_drives_as_apart(self, monkeypatch):
        monkeypatch.setattr(pv, "osp", ntpath)
        pv._refuse_overlap("C:\\work\\project", "D:\\cache\\view")
        with pytest.raises(ValueError, match="overlaps"):
            pv._refuse_overlap("C:\\work\\project", "c:\\work\\project\\.cache")

    def test_pyright_config_takes_the_projects_rules_never_its_paths(self, tmp_path):
        # Defects: the project's own checking level and rule overrides never
        # reach the backend (the user's strictness is ignored); a path
        # setting is copied, so the backend reads the project's directories
        # instead of the shadows; the project turns the escape rule back on,
        # flagging escapes the compiler made.
        project, root = tmp_path / "project", tmp_path / "view"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            "[tool.pyright]\n"
            'typeCheckingMode = "basic"\n'
            "[tool.basedpyright]\n"
            'typeCheckingMode = "strict"\n'
            'reportUnusedVariable = "none"\n'
            'reportInvalidStringEscapeSequence = "error"\n'
            "strictListInference = true\n"
            'include = ["elsewhere"]\n'
            'extraPaths = ["/not/the/view"]\n'
            'venv = "other"\n'
        )
        config = json.loads(
            open(pv.write_pyright_config(str(project), str(root))).read()
        )
        assert config["typeCheckingMode"] == "strict"
        assert config["reportUnusedVariable"] == "none"
        assert config["strictListInference"] is True
        assert config["reportInvalidStringEscapeSequence"] == "none"
        assert "include" not in config
        assert config["extraPaths"] == [str(root), str(project)]
        assert config.get("venv") != "other"
        # pyrightconfig.json wins over pyproject.toml, as for pyright.
        (project / "pyrightconfig.json").write_text('{"typeCheckingMode": "basic"}')
        config = json.loads(
            open(pv.write_pyright_config(str(project), str(root))).read()
        )
        assert config["typeCheckingMode"] == "basic"
        # A malformed file is no settings, never a failed write.
        (project / "pyrightconfig.json").write_text("{not json")
        config = json.loads(
            open(pv.write_pyright_config(str(project), str(root))).read()
        )
        assert config["typeCheckingMode"] == "standard"

    def test_pyright_config_reads_jsonc_as_pyright_does(self, tmp_path):
        # Defect: a pyrightconfig.json with comments or trailing commas,
        # which pyright accepts, is taken as malformed, so the project's
        # rules silently fall back to the default; or a "//" or "/*" inside
        # a string is cut as a comment.
        project, root = tmp_path / "project", tmp_path / "view"
        project.mkdir()
        (project / "pyrightconfig.json").write_text(
            "{\n"
            "  // the project's level\n"
            '  "typeCheckingMode": "strict", /* inline */\n'
            '  "reportGeneralTypeIssues": "warning",\n'
            '  "pythonPlatform": "a//b /* c */ \\"d\\"",\n'
            '  "ignore": ["x",],\n'
            "}\n"
        )
        config = json.loads(
            open(pv.write_pyright_config(str(project), str(root))).read()
        )
        assert config["typeCheckingMode"] == "strict"
        assert config["reportGeneralTypeIssues"] == "warning"
        assert config["pythonPlatform"] == 'a//b /* c */ "d"'

    def test_pyright_config_checks_nothing_without_diagnostics(self, tmp_path):
        # Defect: with pythonDiagnostics false the backend still type-checks
        # every open buffer, at the project's strictness, for nobody.
        project, root = tmp_path / "project", tmp_path / "view"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            "[tool.basedpyright]\n"
            'typeCheckingMode = "strict"\n'
            'reportUnusedVariable = "error"\n'
            'pythonVersion = "3.12"\n'
        )
        path = pv.write_pyright_config(str(project), str(root), check_types=False)
        config = json.loads(open(path).read())
        assert config["typeCheckingMode"] == "off"
        assert "reportUnusedVariable" not in config
        # Hover and completion still follow the project's Python.
        assert config["pythonVersion"] == "3.12"

    def test_pyright_config(self, tmp_path, monkeypatch):
        project, root = tmp_path / "project", tmp_path / "view"
        site = project / ".venv" / "lib" / "python3.99" / "site-packages"
        site.mkdir(parents=True)
        monkeypatch.setattr(sys, "prefix", "/envs/my-venv")
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        path = pv.write_pyright_config(str(project), str(root))
        assert json.loads(open(path).read()) == {
            # A project with no settings of its own: "standard" checking,
            # never basedpyright's own "recommended", which floods code
            # with no annotations; string escapes are the reader's.
            "typeCheckingMode": "standard",
            "reportInvalidStringEscapeSequence": "none",
            "extraPaths": [str(root), str(project), str(site)],
            "venvPath": "/envs",
            "venv": "my-venv",
        }
        stamp = os.stat(path).st_mtime_ns
        os.utime(path, ns=(stamp - 10**9, stamp - 10**9))
        pv.write_pyright_config(str(project), str(root))
        assert os.stat(path).st_mtime_ns == stamp - 10**9  # unchanged: not rewritten
        monkeypatch.setattr(sys, "prefix", "/usr")
        assert "venv" not in json.loads(
            open(pv.write_pyright_config(str(project), str(root))).read()
        )

    def pyrefly_config(self, project, root, **settings):
        path = pv.write_pyrefly_config(str(project), str(root), **settings)
        with open(path, "rb") as f:
            return tomllib.load(f)

    def test_pyrefly_config_takes_the_projects_rules_never_its_paths(
        self, tmp_path, caplog
    ):
        # Defects: the project's own preset, Python and error severities
        # never reach pyrefly; a path or tooling setting is copied, so
        # pyrefly reads the project's directories, interpreter or baseline
        # instead of the view's; [tool.pyrefly] is not read; a malformed
        # file fails the write instead of reading as no settings, or is
        # logged below WARNING, where the user never sees which file broke.
        project, root = tmp_path / "project", tmp_path / "view"
        project.mkdir()
        (project / "pyrefly.toml").write_text(
            'preset = "strict"\n'
            'python-version = "3.12"\n'
            'ignore-missing-imports = ["vendored.*"]\n'
            "check-unannotated-defs = false\n"
            'search-path = ["/not/the/view"]\n'
            'python-interpreter-path = "/other/python"\n'
            'project-excludes = ["**/gen"]\n'
            'baseline = "baseline.json"\n'
            'min-severity = "error"\n'
            "[errors]\n"
            'bad-assignment = "warn"\n'
            "unknown-name = false\n"
            "[[sub-config]]\n"
            'matches = "tests/**"\n'
        )
        assert self.pyrefly_config(project, root) == {
            "preset": "strict",
            "python-version": "3.12",
            "ignore-missing-imports": ["vendored.*"],
            "check-unannotated-defs": False,
            "errors": {"bad-assignment": "warn", "unknown-name": False},
            "search-path": [str(root), str(project)],
            "python-interpreter-path": sys.executable,
        }
        # pyrefly.toml wins over pyproject.toml, as for pyrefly.
        (project / "pyproject.toml").write_text(
            '[tool.pyrefly]\npreset = "legacy"\npython-platform = "win32"\n'
        )
        assert self.pyrefly_config(project, root)["preset"] == "strict"
        (project / "pyrefly.toml").unlink()
        config = self.pyrefly_config(project, root)
        assert (config["preset"], config["python-platform"]) == ("legacy", "win32")
        # A malformed file is no settings, never a failed write, and the
        # log names the file to fix at a level an editor shows.
        (project / "pyproject.toml").write_text("[tool.pyrefly\n")
        with caplog.at_level("WARNING", logger="lispy.lsp.python_view"):
            assert self.pyrefly_config(project, root)["preset"] == "default"
        assert str(project / "pyproject.toml") in caplog.text
        caplog.clear()
        (project / "pyproject.toml").write_text("[tool]\npyrefly = 1\n")
        with caplog.at_level("WARNING", logger="lispy.lsp.python_view"):
            assert self.pyrefly_config(project, root)["preset"] == "default"
        assert "[tool.pyrefly] is not a table" in caplog.text

    def test_pyrefly_default_preset_only_without_a_project_config(self, tmp_path):
        # Defects: a project with no pyrefly config runs pyrefly's
        # unconfigured basic preset, which hides nearly every type error,
        # or no preset at all; or a project whose pyrefly config sets no
        # preset is pinned to the view's, not the one its own pyrefly runs.
        project, root = tmp_path / "project", tmp_path / "view"
        project.mkdir()
        assert self.pyrefly_config(project, root)["preset"] == "default"
        assert pv.pyrefly_fallback(str(root)).reason == "missing"
        (project / "pyproject.toml").write_text('[project]\nname = "p"\n')
        assert self.pyrefly_config(project, root)["preset"] == "default"
        assert pv.pyrefly_fallback(str(root)).reason == "missing"
        (project / "pyproject.toml").write_text(
            '[tool.pyrefly]\npython-version = "3.12"\n'
        )
        config = self.pyrefly_config(project, root)
        assert pv.pyrefly_fallback(str(root)) is None
        assert "preset" not in config
        assert config["python-version"] == "3.12"

    def test_pyrefly_config_checks_nothing_without_diagnostics(self, tmp_path):
        # Defects: with pythonDiagnostics false pyrefly still checks every
        # open buffer, at the project's preset, or the project's errors
        # table turns codes back on under preset off, for nobody; or the
        # default preset is recorded as in effect, so the user is told of
        # a preset that checks nothing.
        project, root = tmp_path / "project", tmp_path / "view"
        project.mkdir()
        (project / "pyrefly.toml").write_text(
            'preset = "strict"\npython-version = "3.12"\n'
            '[errors]\nunknown-name = "error"\n'
        )
        config = self.pyrefly_config(project, root, check_types=False)
        assert config["preset"] == "off"
        assert "errors" not in config
        # Hover and completion still follow the project's Python.
        assert config["python-version"] == "3.12"
        # A project with no pyrefly config checks nothing too: no preset
        # runs, so none is reported to the user as in effect.
        (project / "pyrefly.toml").unlink()
        assert self.pyrefly_config(project, root, check_types=False)["preset"] == "off"
        assert pv.pyrefly_fallback(str(root)) is None

    def test_pyrefly_search_path_is_pyrights(self, tmp_path):
        # Defect: the two backends' search paths drift apart, so a module
        # one resolves (a shadowed src package, a .venv package) the other
        # does not.  Each list leads with its config's own directory, which
        # pyright searches first as its execution root and pyrefly only
        # after search-path.
        project, root = tmp_path / "project", tmp_path / "view"
        self.write(project / "src" / "pkg" / "__init__.py", "")
        site = project / ".venv" / "lib" / "python3.99" / "site-packages"
        site.mkdir(parents=True)
        tools = str(project / "my-tools")
        dirs = (tools, str(project / "src"), str(tmp_path))
        search = {"environments": {tools: dirs, str(tmp_path): (str(tmp_path),)}}
        config = json.loads(
            open(pv.write_pyright_config(str(project), str(root), **search)).read()
        )
        assert config["extraPaths"] == [
            str(root),
            str(root / "src"),
            str(project),
            str(project / "src"),
            str(site),
        ]
        assert config["executionEnvironments"] == [
            {
                "root": str(root / "my-tools"),
                "extraPaths": [
                    str(root / "my-tools"),
                    str(root / "src"),
                    tools,
                    str(project / "src"),
                    str(site),
                ],
            }
        ]
        pyrefly = self.pyrefly_config(project, root, **search)
        assert pyrefly["search-path"] == config["extraPaths"]
        with open(root / "my-tools" / "pyrefly.toml", "rb") as f:
            nested = tomllib.load(f)
        [env] = config["executionEnvironments"]
        assert nested["search-path"] == env["extraPaths"]

    def test_pyrefly_config_is_toml_for_any_path_and_value(self, tmp_path):
        # Defects: the hand-written TOML breaks on a path TOML must escape
        # (a quote, a backslash, DEL, a control character) or on one JSON
        # escapes as a surrogate pair (a character above U+FFFF), so pyrefly
        # cannot read its config; a project value of a shape the writer does
        # not know is written as something else instead of left out.
        project = tmp_path / 'p "q" \\ é 😀 \x7f\t'
        root = tmp_path / "view"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            "[tool.pyrefly]\n"
            "python-version = 3.12\n"
            'enabled-ignores = ["pyrefly", 1]\n'
            'python-platform = "linux"\n'
            "[tool.pyrefly.errors]\n"
            '"bad-assignment" = "warn"\n'
            '"unknown-name" = 0\n'
            '"code with \\" quote" = true\n'
        )
        config = self.pyrefly_config(project, root, python='/py "x"\\ 😀')
        assert config == {
            "python-platform": "linux",
            "errors": {"bad-assignment": "warn", 'code with " quote': True},
            "search-path": [str(root), str(project)],
            "python-interpreter-path": '/py "x"\\ 😀',
        }

    def test_a_refused_pyrefly_config_keeps_the_views_own_settings(
        self, tmp_path, caplog
    ):
        # Defects: a project value the installed pyrefly refuses is written
        # anyway, so pyrefly drops the whole file and the view loses its
        # search paths, interpreter and preset off; the warning does not say
        # which project file to fix; or pyrefly is run again on every
        # unchanged write (one process per save).
        project, root = tmp_path / "project", tmp_path / "view"
        project.mkdir()
        (project / "pyrefly.toml").write_text(
            'preset = "strict"\nuntyped-def-behavior = "nope"\n'
            '[errors]\nnot-a-code = "warn"\n'
        )
        checked = []

        def check(text):
            checked.append(text)
            for value in ["nope", "not-a-code"]:
                if value in text:
                    return f"unknown variant `{value}`"
            return None

        with caplog.at_level("WARNING", logger="lispy.lsp.python_view"):
            config = self.pyrefly_config(project, root, check_types=False, check=check)
        assert config == {
            "preset": "off",
            "search-path": [str(root), str(project)],
            "python-interpreter-path": sys.executable,
        }
        assert str(project / "pyrefly.toml") in caplog.text
        assert "`nope`" in caplog.text
        # Refused, but checking nothing: no preset is in effect to report.
        assert pv.pyrefly_fallback(str(root)) is None
        # With checking, the errors table is refused too.
        (project / "pyrefly.toml").write_text(
            'preset = "strict"\n[errors]\nnot-a-code = "warn"\n'
        )
        config = self.pyrefly_config(project, root, check=check)
        assert (config["preset"], "errors" in config) == ("default", False)
        assert pv.pyrefly_fallback(str(root)) == (
            "refused",
            str(project / "pyrefly.toml"),
        )
        # Saved again with the same refused config: the file holds the
        # fallback, which differs from the candidate, yet pyrefly is not run
        # again and the warning is not repeated on every save.
        calls, warnings = len(checked), caplog.text.count("refuses")
        with caplog.at_level("WARNING", logger="lispy.lsp.python_view"):
            config = self.pyrefly_config(project, root, check=check)
        assert (config["preset"], "errors" in config) == ("default", False)
        assert (len(checked), caplog.text.count("refuses")) == (calls, warnings)
        assert pv.pyrefly_fallback(str(root)).reason == "refused"
        (project / "pyrefly.toml").write_text('preset = "strict"\n')
        assert self.pyrefly_config(project, root, check=check)["preset"] == "strict"
        assert pv.pyrefly_fallback(str(root)) is None
        calls = len(checked)
        self.pyrefly_config(project, root, check=check)
        assert len(checked) == calls

    def refusing_project(self, tmp_path):
        project, root = tmp_path / "project", tmp_path / "view"
        project.mkdir()

        def check(text):
            return "unknown variant `nope`" if "nope" in text else None

        return project, root, check

    def test_a_refused_pyrefly_config_restored_after_removal_warns_again(
        self, tmp_path, caplog
    ):
        # Defect: the refusal stays remembered while the project has no
        # pyrefly config, so the same refused config put back later is taken
        # for the one already told about: never checked, never warned.
        project, root, check = self.refusing_project(tmp_path)
        refused = 'untyped-def-behavior = "nope"\n'
        (project / "pyrefly.toml").write_text(refused)
        with caplog.at_level("WARNING", logger="lispy.lsp.python_view"):
            self.pyrefly_config(project, root, check=check)
            assert caplog.text.count("refuses") == 1
            (project / "pyrefly.toml").unlink()
            self.pyrefly_config(project, root, check=check)
            (project / "pyrefly.toml").write_text(refused)
            self.pyrefly_config(project, root, check=check)
        assert caplog.text.count("refuses") == 2
        assert pv.pyrefly_fallback(str(root)).reason == "refused"

    def test_a_refusal_is_remembered_for_its_text_only(self, tmp_path, caplog):
        # Defect: a refusal is remembered per view root whatever the text,
        # so another refused config, or the first one back after an accepted
        # one, is never checked and never warned about.
        project, root, check = self.refusing_project(tmp_path)
        first = 'untyped-def-behavior = "nope"\n'
        texts = [
            first,
            'untyped-def-behavior = "nope"\npython-version = "3.12"\n',
            first,
            'python-version = "3.12"\n',
            first,
        ]
        counts = []
        with caplog.at_level("WARNING", logger="lispy.lsp.python_view"):
            for text in texts:
                (project / "pyrefly.toml").write_text(text)
                config = self.pyrefly_config(project, root, check=check)
                counts.append(caplog.text.count("refuses"))
        assert counts == [1, 2, 3, 3, 4]
        assert config["preset"] == "default"

    def test_a_malformed_pyrefly_config_is_told_once_per_problem(
        self, tmp_path, caplog
    ):
        # Defects: the warning about a malformed project config repeats on
        # every write (every backend start and every save); or it is told
        # once for good, so another malformation, or the same one after the
        # config was fixed, is never told.
        project, root = tmp_path / "project", tmp_path / "view"
        project.mkdir()
        config = project / "pyrefly.toml"
        broken, other = 'preset = "strict"\npreset = "basic"\n', "[errors\n"
        fixed = 'preset = "strict"\n'
        counts = []
        with caplog.at_level("WARNING", logger="lispy.lsp.python_view"):
            # The last malformation comes back after the fix: told again.
            for text in [broken, broken, other, fixed, other]:
                config.write_text(text)
                self.pyrefly_config(project, root, check_types=False)
                counts.append(caplog.text.count("cannot read the pyrefly settings"))
        assert counts == [1, 1, 2, 2, 3]

    def test_a_project_config_in_a_package_never_replaces_the_views(self, tmp_path):
        # Defect: a project inside a package has its own pyrefly.toml or
        # pyrightconfig.json linked over the view root's on every shadow
        # write, so the backend reads the project's paths, and each save
        # flips the file between the two.
        project, root = tmp_path / "project", tmp_path / "view"
        self.write(project / "__init__.py", "")
        self.write(project / "m.lpy", "(= m 1)")
        self.write(project / "pyrefly.toml", 'search-path = ["/elsewhere"]\n')
        self.write(project / "pyrightconfig.json", '{"extraPaths": ["/elsewhere"]}')
        writers = [pv.write_pyrefly_config, pv.write_pyright_config]
        for _ in range(2):
            pv.write_shadows(str(project), str(root))
            for write in writers:
                path = write(str(project), str(root))
                assert not os.path.islink(path)
                assert "/elsewhere" not in open(path).read()
        pv.write_shadows(str(project), str(root))
        for name in [pv.PYREFLY_CONFIG, pv.PYRIGHT_CONFIG]:
            assert not os.path.islink(root / name)


def next_tick():
    """Let the filesystem clock move on, as it has by the time a user saves:
    the kernel stamps mtimes from a clock that ticks every few milliseconds,
    so a directory changed within one tick of being searched keeps its
    mtime."""
    time.sleep(0.05)


TWICE = "(defmacro twice [x] (return `(+ ~x ~x)))"
# A different size, so the bytecode cache (mtime to the second + size) cannot
# hand back the old module.
THRICE = "(defmacro twice [x] (return `(* ~x ~x ~x)))"


class TestMacroModules:
    """A module a require imported is imported anew once its file changed:
    the LSP compiles in its own process, where it would otherwise stay in
    sys.modules with the old macros."""

    @pytest.fixture
    def package(self, tmp_path, monkeypatch, request):
        # A package named after the test, so no other test's import of it is
        # in sys.modules; every module under it is dropped afterwards.
        name = "pvmm" + re.sub(r"\W", "", request.node.name).lower()
        root = tmp_path / "project" / "src"
        pkg = root / name
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "mac.lpy").write_text(TWICE)
        for key in [name, name + ".mac"]:
            monkeypatch.delitem(sys.modules, key, raising=False)
        yield SimpleNamespace(name=name, root=root, pkg=pkg)
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]

    def view(self, package):
        return pv.compile_view(
            "(require .mac *)\n(= yy (twice 2))",
            package=package.name,
            name=package.name + ".mod",
            search_dirs=(str(package.root),),
        )

    def test_compile_view_sees_an_edited_macro_module(self, package):
        assert self.view(package).text.endswith("yy = 2 + 2")
        (package.pkg / "mac.lpy").write_text(THRICE)
        assert self.view(package).text.endswith("yy = 2 * 2 * 2")

    def test_write_shadows_recompiles_a_dependent_of_an_edited_module(
        self, package, tmp_path
    ):
        # mod.lpy itself is unchanged, so its stamp alone would keep its shadow.
        (package.pkg / "mod.lpy").write_text("(require .mac *)\n(= yy (twice 2))")
        project, root = package.root.parent, tmp_path / "view"
        shadow = root / "src" / package.name / "mod.py"
        manifest = pv.ShadowManifest()
        pv.write_shadows(str(project), str(root), manifest=manifest)
        assert "yy = 2 + 2" in shadow.read_text()
        (package.pkg / "mac.lpy").write_text(THRICE)
        pv.write_shadows(str(project), str(root), manifest=manifest)
        assert "yy = 2 * 2 * 2" in shadow.read_text()

    def test_view_cache_misses_once_a_macro_module_changed(self, package):
        cache = pv.ViewCache()
        src = "(require .mac *)\n(= yy (twice 2))"
        kwargs = dict(
            package=package.name,
            name=package.name + ".mod",
            search_dirs=(str(package.root),),
        )
        assert cache.view("u", 1, src, **kwargs).text.endswith("yy = 2 + 2")
        (package.pkg / "mac.lpy").write_text(THRICE)
        assert cache.view("u", 1, src, **kwargs).text.endswith("yy = 2 * 2 * 2")

    def test_a_qualified_macro_keeps_its_dotted_spelling(self, package):
        # What a qualified-macro completion labels from: name-key and
        # spelling differ only in the hyphen. A relative require qualifies by
        # the module's last part, the name from . import mac binds.
        (package.pkg / "mac.lpy").write_text("(defmacro my-mac [] 1)")
        view = pv.compile_view(
            "(require .mac)\n(= yy 1)",
            package=package.name,
            name=package.name + ".mod",
            search_dirs=(str(package.root),),
        )
        assert view.macros["mac.my_mac"] == "mac.my-mac"

    def test_diagnostics_see_an_edited_macro_module(self, package):
        # The document's own import root (module-names) puts its package on
        # sys.path for the compile, as compile-buffer does for every view.
        uri = (package.pkg / "mod.lpy").as_uri()
        src = "(require .mac *)\n(= yy (twice 2))"
        assert server.compute_diagnostics(src, uri) == []
        (package.pkg / "mac.lpy").write_text(
            '(defmacro twice [x] (raise (ValueError "pvmm-boom")))'
        )
        [diagnostic] = server.compute_diagnostics(src, uri)
        assert "pvmm-boom" in diagnostic.message

    def test_a_module_from_another_source_root_is_seen(
        self, package, tmp_path, monkeypatch
    ):
        # Defect: only modules under the compiling file's import root are
        # tracked, so a helper from another source root (a workspace member,
        # an editable install, a PYTHONPATH entry) keeps its old instance.
        other = tmp_path / "other"
        other.mkdir()
        helper = package.name + "helper"
        (other / (helper + ".py")).write_text("VALUE = 1\n")
        monkeypatch.syspath_prepend(str(other))
        monkeypatch.delitem(sys.modules, helper, raising=False)
        (package.pkg / "mac.lpy").write_text(
            f"(import {helper})\n(defmacro twice [x] (return {helper}.VALUE))"
        )
        try:
            assert self.view(package).text.endswith("yy = 1")
            (other / (helper + ".py")).write_text("VALUE = 22\n")
            assert self.view(package).text.endswith("yy = 22")
        finally:
            sys.modules.pop(helper, None)

    def test_a_module_in_site_packages_is_never_dropped(
        self, package, tmp_path, monkeypatch
    ):
        # A library a macro imports keeps its one instance (an extension
        # refuses a second load).
        site = tmp_path / "lib" / "site-packages"
        site.mkdir(parents=True)
        ext = package.name + "ext"
        (site / (ext + ".py")).write_text("VALUE = 1\n")
        monkeypatch.syspath_prepend(str(site))
        monkeypatch.delitem(sys.modules, ext, raising=False)
        (package.pkg / "mac.lpy").write_text(f"(import {ext})\n" + TWICE)
        try:
            self.view(package)
            module = sys.modules[ext]
            (site / (ext + ".py")).write_text("VALUE = 22\n")
            (package.pkg / "mac.lpy").write_text(f"(import {ext})\n" + THRICE)
            assert self.view(package).text.endswith("yy = 2 * 2 * 2")
            assert sys.modules[ext] is module
        finally:
            sys.modules.pop(ext, None)

    def test_one_name_from_two_roots_tracks_both_files(self, package, tmp_path):
        # Defect: tracking by module name lets the second root's file replace
        # the first's, so fixing the first (whose import failed) is never
        # seen: the name stays bound to the second root's module.
        name = package.name + "two"
        first, second = tmp_path / "first", tmp_path / "second"
        for root in [first, second]:
            root.mkdir()
        (first / (name + ".lpy")).write_text("(defmacro twice [x]")
        (second / (name + ".lpy")).write_text(THRICE)

        def view(root):
            return pv.compile_view(
                f"(require {name} *)\n(= yy (twice 2))",
                name="pvmmtwo",
                search_dirs=(str(root),),
            )

        try:
            # The require failed and was left out: twice stays a call.
            assert view(first).text.endswith("yy = twice(2)")
            assert view(second).text.endswith("yy = 2 * 2 * 2")
            (first / (name + ".lpy")).write_text(
                "(defmacro twice [x] (return `(- ~x ~x)))"
            )
            assert view(first).text.endswith("yy = 2 - 2")
        finally:
            sys.modules.pop(name, None)

    def consumer(self, kind, package, tmp_path):
        """The text a ViewCache view (one uri and version throughout) or a
        write-shadows sweep (one manifest throughout) makes of mod, or None."""
        src = "(require .mac *)\n(= yy (twice 2))"
        if kind == "view cache":
            cache = pv.ViewCache()

            def text():
                view = cache.view(
                    "u",
                    1,
                    src,
                    package=package.name,
                    name=package.name + ".mod",
                    search_dirs=(str(package.root),),
                )
                return None if view is None else view.text

            return text
        (package.pkg / "mod.lpy").write_text(src)
        project, root = package.root.parent, tmp_path / "view"
        shadow = root / "src" / package.name / "mod.py"
        manifest = pv.ShadowManifest()

        def text():
            pv.write_shadows(str(project), str(root), manifest=manifest)
            return shadow.read_text() if shadow.exists() else None

        return text

    @pytest.mark.parametrize("kind", ["view cache", "shadows"])
    def test_a_fixed_macro_module_is_seen(self, package, tmp_path, kind):
        # A module whose import failed is in no sys.modules, yet is tracked
        # by the file it was read from, so fixing that file is seen.
        text = self.consumer(kind, package, tmp_path)
        (package.pkg / "mac.lpy").write_text("(defmacro twice [x] (return `(+ ~x ~x))")
        # The require failed and was left out: twice stays a call.
        assert text().endswith("yy = twice(2)")
        (package.pkg / "mac.lpy").write_text(THRICE)
        assert "yy = 2 * 2 * 2" in text()

    @pytest.mark.parametrize("kind", ["view cache", "shadows"])
    def test_a_created_macro_module_is_seen(self, package, tmp_path, kind):
        # A missing module is never read, so nothing tracks it; the require
        # only warns, and twice stays a call.
        text = self.consumer(kind, package, tmp_path)
        (package.pkg / "mac.lpy").unlink()
        assert "yy = twice(2)" in text()
        next_tick()
        (package.pkg / "mac.lpy").write_text(THRICE)
        assert "yy = 2 * 2 * 2" in text()

    def test_a_package_created_in_a_namespace_directory_is_seen(self, package):
        # Defect: a require of a directory with no __init__ binds an empty
        # namespace module, which no read tracks, so a later __init__.lpy
        # there is never loaded while the server runs.
        (package.pkg / "nosuch").mkdir()
        src = "(require .nosuch *)\n(= yy (twice 2))"
        kwargs = dict(
            package=package.name,
            name=package.name + ".mod",
            search_dirs=(str(package.root),),
        )
        assert pv.compile_view(src, **kwargs).text.endswith("yy = twice(2)")
        next_tick()
        (package.pkg / "nosuch" / "__init__.lpy").write_text(THRICE)
        assert pv.compile_view(src, **kwargs).text.endswith("yy = 2 * 2 * 2")

    def flat_namespace(self, package, tmp_path):
        """(project, name): a flat layout whose import root holds a
        directory name with no __init__, after a view requiring it."""
        project = tmp_path / "flat"
        name = package.name + "ns"
        (project / name).mkdir(parents=True)
        assert self.flat_view(project, name).text.endswith("yy = twice(2)")
        next_tick()
        return project, name

    def test_a_module_created_beside_a_namespace_directory_is_seen(
        self, package, tmp_path
    ):
        # A module file of the namespace's name replaces it on import.
        project, name = self.flat_namespace(package, tmp_path)
        try:
            (project / (name + ".lpy")).write_text(THRICE)
            assert self.flat_view(project, name).text.endswith("yy = 2 * 2 * 2")
        finally:
            sys.modules.pop(name, None)

    def test_a_namespace_leaves_its_parent_directory_unwatched(self, package, tmp_path):
        # Defect: a top-level namespace stamps its parent directory, the
        # import root of a flat layout, so any file written there drops
        # every module.
        project, name = self.flat_namespace(package, tmp_path)
        try:
            epoch = mm.epoch()
            (project / "uv.lock").write_text("")
            assert mm.epoch() == epoch
        finally:
            sys.modules.pop(name, None)

    def flat_view(self, project, mac):
        return pv.compile_view(
            f"(require {mac} *)\n(= yy (twice 2))",
            name="pvmmflat",
            search_dirs=(str(project),),
        )

    def test_an_incidental_miss_leaves_the_import_root_unwatched(
        self, package, tmp_path
    ):
        # Defect: a macro module's optional import that is missing stamps
        # every sys.path directory, the import root among them, so in a flat
        # layout any file written there (uv.lock) drops every module.
        project = tmp_path / "flat"
        project.mkdir()
        mac = package.name + "flatmac"
        (project / (mac + ".lpy")).write_text(
            "(try (import pvmmnosuchoptional) (except [ImportError] (pass)))\n" + TWICE
        )
        try:
            assert self.flat_view(project, mac).text.endswith("yy = 2 + 2")
            epoch = mm.epoch()
            next_tick()
            (project / "uv.lock").write_text("")
            assert mm.epoch() == epoch
        finally:
            sys.modules.pop(mac, None)

    def test_a_created_top_level_macro_module_is_seen(self, package, tmp_path):
        # A require's own top-level miss still watches sys.path, so a cached
        # view misses once the module is created in the import root.
        project = tmp_path / "flat"
        project.mkdir()
        mac = package.name + "flatmac"
        cache = pv.ViewCache()

        def text():
            return cache.view(
                "u",
                1,
                f"(require {mac} *)\n(= yy (twice 2))",
                name="pvmmflat",
                search_dirs=(str(project),),
            ).text

        try:
            assert text().endswith("yy = twice(2)")
            next_tick()
            (project / (mac + ".lpy")).write_text(THRICE)
            assert text().endswith("yy = 2 * 2 * 2")
        finally:
            sys.modules.pop(mac, None)

    def test_a_same_size_edit_in_the_same_second_is_seen(self, package, monkeypatch):
        # A pyc keeps the source's mtime in whole seconds plus its size, so
        # it would pass for the edited source; the .lpy cache must not.
        monkeypatch.setattr(sys, "dont_write_bytecode", False)
        mac = package.pkg / "mac.lpy"
        second = (int(time.time()) - 10) * 10**9
        os.utime(mac, ns=(second + 100, second + 100))
        assert self.view(package).text.endswith("yy = 2 + 2")
        assert os.path.exists(lpy_cache_path(str(mac)))
        minus = TWICE.replace("(+ ", "(- ")
        assert len(minus) == len(TWICE)
        mac.write_text(minus)
        os.utime(mac, ns=(second + 200, second + 200))
        assert self.view(package).text.endswith("yy = 2 - 2")

    def test_a_stale_wait_is_bounded(self, package, monkeypatch):
        # A macro hung in a running compile holds a waiting one up for
        # WAIT_LIMIT at most; it then goes on with the modules as they are.
        monkeypatch.setattr(mm, "WAIT_LIMIT", 0.2)
        assert self.view(package) is not None
        entered, release = threading.Event(), threading.Event()

        def hold():
            with mm.fresh_modules():
                entered.set()
                release.wait(30)

        def wait():
            with mm.fresh_modules():
                pass

        holder = threading.Thread(target=hold, daemon=True)
        waiter = threading.Thread(target=wait, daemon=True)
        holder.start()
        try:
            assert entered.wait(10)
            (package.pkg / "mac.lpy").write_text(THRICE)
            waiter.start()
            waiter.join(0.2 + 3)
            assert not waiter.is_alive()
        finally:
            release.set()
            holder.join(10)
            if waiter.ident is not None:
                waiter.join(10)

    @contextlib.contextmanager
    def held_compile(self, monkeypatch, limit=0.3):
        """A compile hung in its macros for the whole block (a thread inside
        fresh-modules), WAIT_LIMIT set to limit; yields the list of the
        timed waits on the shared condition made meanwhile."""
        monkeypatch.setattr(mm, "WAIT_LIMIT", limit)
        condition = mm._SHARED.condition
        real, waits = condition.wait, []

        def counting(timeout=None):
            if timeout is not None:
                waits.append(timeout)
            return real(timeout)

        entered, release = threading.Event(), threading.Event()

        def hold():
            with mm.fresh_modules():
                entered.set()
                release.wait(30)

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        try:
            assert entered.wait(10)
            monkeypatch.setattr(condition, "wait", counting)
            yield waits
        finally:
            release.set()
            holder.join(10)
            assert not holder.is_alive()

    def test_a_hung_compile_costs_one_wait_and_drops_resume(self, package, monkeypatch):
        # Defect: a compile hung in a macro keeps every later drop from
        # happening, so no edit is seen again and every compile waits
        # WAIT_LIMIT anew; and epoch waits too, before every cached lookup.
        mac = package.pkg / "mac.lpy"
        assert self.view(package).text.endswith("yy = 2 + 2")
        edits = [
            (THRICE, "yy = 2 * 2 * 2"),
            ("(defmacro twice [x] (return `(- ~x ~x ~x ~x)))", "yy = 2 - 2 - 2 - 2"),
            ("(defmacro twice [x] (return `(+ ~x 1)))", "yy = 2 + 1"),
        ]
        with self.held_compile(monkeypatch) as waits:
            for number, (text, expected) in enumerate(edits):
                next_tick()
                mac.write_text(text)
                if number == 0:
                    assert mm.epoch() is None
                    assert waits == []
                assert self.view(package).text.endswith(expected)
            assert len(waits) == 1

    def test_a_cursor_view_waits_once_for_a_hung_compile(self, package, monkeypatch):
        # Defect: a cursor view waits for the hung compile in epoch and again
        # in each of its compiles (the buffer's, then its prefix's), past the
        # backend's request timeout.  The open form makes the prefix compile.
        cache = pv.ViewCache()
        src = "(require .mac *)\n(= yy (+ (twice 2) zz)"
        kwargs = dict(
            package=package.name,
            name=package.name + ".mod",
            search_dirs=(str(package.root),),
        )
        line, col = plus(locate(src, "zz"), 2)
        first = cache.view_at_cursor("u", 1, src, line, col, **kwargs)
        assert "yy = 2 + 2 + zz" in first.text
        with self.held_compile(monkeypatch) as waits:
            next_tick()
            (package.pkg / "mac.lpy").write_text(THRICE)
            view = cache.view_at_cursor("u", 2, src, line, col, **kwargs)
            assert len(waits) == 1
        assert view.fallback and "yy = 2 * 2 * 2 + zz" in view.text

    @pytest.fixture
    def helper(self, tmp_path, request):
        """A top-level macro module of one name in two directories, a/ and
        b/, whose pick expands to 1 and 2."""
        name = "pvsd" + re.sub(r"\W", "", request.node.name).lower()
        dirs = []
        for number, sub in enumerate(["a", "b"], 1):
            directory = tmp_path / sub
            directory.mkdir()
            (directory / f"{name}.lpy").write_text(
                f"(defmacro pick [] (return {number}))"
            )
            dirs.append(str(directory))
        sys.modules.pop(name, None)
        yield SimpleNamespace(name=name, a=dirs[0], b=dirs[1])
        sys.modules.pop(name, None)

    def picked(self, helper, *dirs):
        """What pick expands to in a compile searching dirs."""
        view = pv.compile_view(
            f"(require {helper.name} *)\n(= yy (pick))", search_dirs=dirs
        )
        return view.text.rpartition("yy = ")[2]

    @contextlib.contextmanager
    def held_with(self, dirs, imported, limit, monkeypatch):
        """A compile searching dirs hung in its macros for the whole block,
        after importing the module imported; WAIT_LIMIT set to limit."""
        monkeypatch.setattr(mm, "WAIT_LIMIT", limit)
        entered, release = threading.Event(), threading.Event()

        def hold():
            with mm.fresh_modules(dirs):
                importlib.import_module(imported)
                entered.set()
                release.wait(30)

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        try:
            assert entered.wait(10)
            yield
        finally:
            release.set()
            holder.join(10)
            assert not holder.is_alive()

    def test_search_dirs_come_first_in_their_order(self, helper, monkeypatch):
        # Defect: a search dir already on sys.path is left where it is while
        # an absent one goes in front, so a module both hold comes from the
        # second dir.
        monkeypatch.setattr(sys, "path", [helper.a, *sys.path])
        assert self.picked(helper, helper.a, helper.b) == "1"

    def test_a_module_of_other_search_dirs_is_imported_anew(self, helper):
        # Defect: a script's sibling module stays in sys.modules, so a script
        # in another directory requiring the same name gets the first one's.
        assert self.picked(helper, helper.a) == "1"
        assert self.picked(helper, helper.b) == "2"

    def test_a_switch_of_search_dirs_waits_for_running_compiles(
        self, helper, monkeypatch
    ):
        # Defect: a compile under other search dirs drops (or reuses) the
        # modules a running compile is using.
        result = []
        with self.held_with((helper.a,), helper.name, 30, monkeypatch):
            other = threading.Thread(
                target=lambda: result.append(self.picked(helper, helper.b)),
                daemon=True,
            )
            other.start()
            other.join(0.5)
            assert other.is_alive()
        other.join(10)
        assert result == ["2"]

    def test_compiles_arriving_while_one_waits_to_drop_wait_behind_it(
        self, helper, monkeypatch
    ):
        # Defect: compiles that do not conflict keep being admitted while one
        # waits to drop, so it waits for them too, times out, marks them hung
        # and moves the epoch with no edit.
        with self.held_with((helper.a,), helper.name, 30, monkeypatch):
            switcher = threading.Thread(
                target=lambda: self.picked(helper, helper.b), daemon=True
            )
            switcher.start()
            switcher.join(0.3)
            assert switcher.is_alive()

            def arrive():
                with mm.fresh_modules((helper.a,)):
                    pass

            arriving = threading.Thread(target=arrive, daemon=True)
            arriving.start()
            arriving.join(0.3)
            assert arriving.is_alive()
        switcher.join(10)
        arriving.join(10)
        assert not switcher.is_alive() and not arriving.is_alive()

    def test_a_switch_past_a_hung_compile_moves_the_epoch(self, helper, monkeypatch):
        # Defect: a switch drops the modules under a hung compile but leaves
        # its result, which the drop may have broken, current.
        with self.held_with((helper.a,), helper.name, 0.2, monkeypatch):
            before = mm.current_epoch()
            compiled = pv.compile_buffer("(= yy 1)", search_dirs=(helper.b,))
        assert compiled.epoch == before + 1

    def raced(self, helper):
        """Two compiles admitted together, nothing tracked yet: one searching
        a/ imports the helper, then the other, searching b/, reuses it.
        Returns the epoch before and after the second one ended."""
        imported, release = threading.Event(), threading.Event()

        def hold():
            with mm.fresh_modules((helper.a,)):
                importlib.import_module(helper.name)
                imported.set()
                release.wait(30)

        with mm.fresh_modules((helper.b,)) as before:
            holder = threading.Thread(target=hold, daemon=True)
            holder.start()
            try:
                assert imported.wait(10)
                assert importlib.import_module(helper.name).__file__.startswith(
                    helper.a
                )
            except BaseException:
                release.set()
                raise
        after = mm.current_epoch()
        release.set()
        holder.join(10)
        assert not holder.is_alive()
        return before, after

    def test_a_module_reused_from_other_search_dirs_moves_the_epoch(self, helper):
        # Defect: the compile searching b/ used a/'s helper, and its result
        # stays current under the epoch.
        before, after = self.raced(helper)
        assert after == before + 1

    def waits_beside(self, helper, monkeypatch):
        """Whether a compile searching a/ waits while one searching b/ runs."""
        monkeypatch.setattr(mm, "WAIT_LIMIT", 30)
        held, release = threading.Event(), threading.Event()

        def hold():
            with mm.fresh_modules((helper.b,)):
                held.set()
                release.wait(30)

        def enter():
            with mm.fresh_modules((helper.a,)):
                pass

        holder = threading.Thread(target=hold, daemon=True)
        other = threading.Thread(target=enter, daemon=True)
        holder.start()
        try:
            assert held.wait(10)
            other.start()
            other.join(0.5)
            waited = other.is_alive()
        finally:
            release.set()
            holder.join(10)
            if other.ident is not None:
                other.join(10)
        assert not other.is_alive()
        return waited

    def test_a_raced_name_keeps_the_two_compiles_apart(self, helper, monkeypatch):
        # Defect: after the race the next compile searching b/ drops the
        # helper, a compile searching a/ is admitted beside it and imports it
        # again, and the race (and its recompile of every document) repeats.
        self.raced(helper)
        assert self.waits_beside(helper, monkeypatch)

    def test_an_edit_frees_a_raced_name(self, helper, monkeypatch):
        # Defect: a raced name keeps the two compiles apart for ever, each
        # dropping every tracked module, after any edit.
        self.raced(helper)
        next_tick()
        (pathlib.Path(helper.a) / (helper.name + ".lpy")).write_text(
            "(defmacro pick [] (return 11))"
        )
        mm.epoch()
        assert not self.waits_beside(helper, monkeypatch)

    def test_a_module_another_finder_loads_conflicts_with_nothing(
        self, tmp_path, monkeypatch
    ):
        # Defect: a module a meta_path finder other than PathFinder loaded
        # (an editable install's) is found nowhere by PathFinder, so its own
        # compile conflicts with it: every compile moves the epoch.
        name = "pvsdfinderloaded"
        (tmp_path / "elsewhere").mkdir()
        source = tmp_path / "elsewhere" / f"{name}.py"
        source.write_text("X = 1\n")

        class Finder:
            @staticmethod
            def find_spec(fullname, path, target=None):
                if fullname != name:
                    return None
                return importlib.util.spec_from_file_location(name, source)

        monkeypatch.setattr(sys, "meta_path", [Finder, *sys.meta_path])
        directory = tmp_path / "d"
        directory.mkdir()
        try:
            with mm.fresh_modules((str(directory),)) as before:
                first = importlib.import_module(name)
            with mm.fresh_modules((str(directory),)):
                assert importlib.import_module(name) is first
            assert mm.current_epoch() == before
        finally:
            sys.modules.pop(name, None)

    def test_a_search_dir_spelled_two_ways_is_one(self, tmp_path, monkeypatch):
        # Defect: a search dir that links to a sys.path entry resolves its
        # module under two spellings, so compiles alternating with another
        # directory drop and import it again every time.
        name = "pvsdlinked"
        (tmp_path / "real").mkdir()
        (tmp_path / "real" / f"{name}.py").write_text("X = 1\n")
        (tmp_path / "link").symlink_to(tmp_path / "real")
        (tmp_path / "other").mkdir()
        monkeypatch.setattr(sys, "path", [*sys.path, str(tmp_path / "real")])
        try:
            with mm.fresh_modules((str(tmp_path / "link"),)):
                first = importlib.import_module(name)
            with mm.fresh_modules((str(tmp_path / "other"),)):
                assert importlib.import_module(name) is first
        finally:
            sys.modules.pop(name, None)

    def test_a_parked_compile_moves_the_epoch_once(self, helper, monkeypatch):
        # Defect: a hung compile's reads keep counting after a drop passed
        # it, so every later compile searching b/ conflicts with them, drops
        # b's helper, imports it again and moves the epoch: every open
        # document recompiles for ever, and no edit stops it.
        with self.held_with((helper.a,), helper.name, 0.2, monkeypatch):
            epochs = []
            for _ in range(3):
                with mm.fresh_modules((helper.b,)):
                    module = importlib.import_module(helper.name)
                    assert module.__file__.startswith(helper.b)
                epochs.append(mm.current_epoch())
        assert epochs[1:] == epochs[:1] * 2

    def test_a_module_an_appended_finder_serves_conflicts_with_nothing(
        self, tmp_path, monkeypatch
    ):
        # Defect: a package an editable install's finder (after PathFinder)
        # serves from src/ is found by PathFinder under (src,) only, so
        # compiles of src/ and tests/ alternating drop it and run it again.
        name = "pvsdeditable"
        package = tmp_path / "src" / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("RUNS.append(1)\n")
        (tmp_path / "tests").mkdir()
        runs = []
        monkeypatch.setattr(builtins, "RUNS", runs, raising=False)

        class Finder:
            @staticmethod
            def find_spec(fullname, path, target=None):
                if fullname != name:
                    return None
                return importlib.util.spec_from_file_location(
                    name,
                    package / "__init__.py",
                    submodule_search_locations=[str(package)],
                )

        monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, Finder])
        try:
            for dirs in [("src",), ("tests",)] * 2:
                with mm.fresh_modules(tuple(str(tmp_path / d) for d in dirs)):
                    importlib.import_module(name)
        finally:
            sys.modules.pop(name, None)
        assert runs == [1]

    def test_a_package_extending_its_path_conflicts_with_nothing(self, tmp_path):
        # Defect: a regular package is compared by its live __path__, which
        # its __init__ extended, so it conflicts with its own search dirs:
        # every compile drops it and moves the epoch, and every open
        # document recompiles for ever.
        name = "pvsdmutpkg"
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text(
            "import os\n__path__.append(os.path.join(os.path.dirname(__file__), 'x'))\n"
        )
        try:
            with mm.fresh_modules((str(tmp_path),)) as before:
                first = importlib.import_module(name)
            for _ in range(2):
                with mm.fresh_modules((str(tmp_path),)):
                    assert importlib.import_module(name) is first
            assert mm.current_epoch() == before
        finally:
            sys.modules.pop(name, None)

    def test_a_module_found_only_under_other_search_dirs_is_not_reused(self, helper):
        # Defect: y in b/ requires a helper only a/ holds, after x in a/
        # compiled: it gets a's macros, which `lpy b/y.lpy` never finds, and
        # creating b's own helper later moves no epoch.
        os.remove(os.path.join(helper.b, helper.name + ".lpy"))
        before = mm.current_epoch()
        for _ in range(2):
            assert self.picked(helper, helper.a) == "1"
            assert self.picked(helper, helper.b) == "pick()"
        # Alternating only drops and imports again: nothing to recompile.
        assert mm.current_epoch() == before
        next_tick()
        (pathlib.Path(helper.b) / (helper.name + ".lpy")).write_text(
            "(defmacro pick [] (return 2))"
        )
        assert self.picked(helper, helper.b) == "2"

    def test_a_namespace_package_keeps_its_search_dir_portion(self, tmp_path):
        # Defect: a namespace package with portions under a search dir and on
        # sys.path recomputes them from sys.path alone on invalidate_caches,
        # so its submodule under the search dir stops importing.
        name = "pvsdns"
        (tmp_path / "d" / name).mkdir(parents=True)
        (tmp_path / "s" / name).mkdir(parents=True)
        path = sys.path[:]
        sys.path.append(str(tmp_path / "s"))
        try:
            with mm.fresh_modules((str(tmp_path / "d"),)):
                importlib.import_module(name)
                importlib.invalidate_caches()
                portions = list(sys.modules[name].__path__)
        finally:
            sys.path[:] = path
            for key in [name, name + ".sub"]:
                sys.modules.pop(key, None)
        assert portions == [str(tmp_path / "d" / name), str(tmp_path / "s" / name)]

    def test_a_missing_module_keeps_the_environment_settled(
        self, package, tmp_path, monkeypatch
    ):
        # A require of a module that does not exist moves the epoch only once
        # its directory changes, so every other view and shadow stays cached.
        bad = "(require .nosuch *)\n(= zz (twice 3))"
        (package.pkg / "bad.lpy").write_text(bad)
        (package.pkg / "mod.lpy").write_text("(require .mac *)\n(= yy (twice 2))")
        cache = pv.ViewCache()
        kwargs = dict(package=package.name, search_dirs=(str(package.root),))

        def good():
            return cache.view(
                "good", 1, "(= a 1)", name=package.name + ".good", **kwargs
            )

        cache.view("bad", 1, bad, name=package.name + ".bad", **kwargs)
        first = good()
        cache.view("bad", 2, bad + "\n", name=package.name + ".bad", **kwargs)
        assert good() is first

        project, root = package.root.parent, tmp_path / "view"
        manifest = pv.ShadowManifest()
        pv.write_shadows(str(project), str(root), manifest=manifest)
        entries = dict(manifest.sources)
        compiled = []
        compile_buffer = pv.compile_buffer
        monkeypatch.setattr(
            pv,
            "compile_buffer",
            lambda *a, **k: compiled.append(a) or compile_buffer(*a, **k),
        )
        pv.write_shadows(str(project), str(root), manifest=manifest)
        assert compiled == []
        assert manifest.sources == entries

        next_tick()
        (package.pkg / "nosuch.lpy").write_text(TWICE)
        pv.write_shadows(str(project), str(root), manifest=manifest)
        shadow = root / "src" / package.name / "bad.py"
        assert "zz = 3 + 3" in shadow.read_text()

    def test_a_macro_a_macro_module_required_is_seen(self, package, monkeypatch):
        # mac's own cache holds what base's macro expanded to when mac was
        # compiled, so an edit to base reaches mac only through a new compile
        # of mac.
        monkeypatch.setattr(sys, "dont_write_bytecode", False)
        (package.pkg / "base.lpy").write_text("(defmacro bump [x] (return `(+ ~x 1)))")
        (package.pkg / "mac.lpy").write_text(
            "(require .base *)\n(defmacro twice [x] (return (bump 10)))"
        )
        assert self.view(package).text.endswith("yy = 11")
        assert os.path.exists(lpy_cache_path(str(package.pkg / "mac.lpy")))
        (package.pkg / "base.lpy").write_text("(defmacro bump [x] (return `(* ~x 30)))")
        assert self.view(package).text.endswith("yy = 300")

    def expansion_helper(self, package):
        # mac's V comes from b's macro, which imports h only as it expands:
        # h is read when mac compiles, never when mac loads from its cache.
        (package.pkg / "h.py").write_text("VALUE = 1\n")
        (package.pkg / "b.lpy").write_text(
            "(defmacro val [] (import importlib)"
            f' (return (. (importlib.import-module "{package.name}.h") VALUE)))'
        )
        (package.pkg / "mac.lpy").write_text(
            "(require .b *)\n(= V (val))\n(defmacro twice [x] (return V))"
        )

    def test_a_module_read_by_a_cached_modules_compile_is_seen(
        self, package, monkeypatch
    ):
        # After a drop, mac loads from its cache, so only its record names h.
        monkeypatch.setattr(sys, "dont_write_bytecode", False)
        self.expansion_helper(package)
        assert self.view(package).text.endswith("yy = 1")
        init = package.pkg / "__init__.py"
        st = os.stat(init)
        os.utime(init, ns=(st.st_atime_ns, st.st_mtime_ns + 10**6))
        assert self.view(package).text.endswith("yy = 1")
        (package.pkg / "h.py").write_text("VALUE = 99\n")
        assert self.view(package).text.endswith("yy = 99")

    def test_a_server_starting_on_a_warm_cache_sees_its_record(
        self, package, monkeypatch
    ):
        # Another process compiled mac, so this one loads it from its cache
        # and never reads h.
        monkeypatch.setattr(sys, "dont_write_bytecode", False)
        self.expansion_helper(package)
        env = dict(os.environ)
        env.pop("PYTHONDONTWRITEBYTECODE", None)
        env["PYTHONPATH"] = str(package.root)
        subprocess.run(
            [sys.executable, "-c", f"import lispy, {package.name}.mac"],
            env=env,
            check=True,
            timeout=120,
        )
        assert os.path.exists(lpy_cache_path(str(package.pkg / "mac.lpy")))
        assert self.view(package).text.endswith("yy = 1")
        (package.pkg / "h.py").write_text("VALUE = 99\n")
        assert self.view(package).text.endswith("yy = 99")

    def test_a_broken_macro_module_keeps_the_environment_settled(
        self, package, monkeypatch
    ):
        # While mac fails to compile, its file alone is tracked: nothing is
        # stale until it changes again.
        monkeypatch.setattr(sys, "dont_write_bytecode", False)
        assert self.view(package).text.endswith("yy = 2 + 2")
        (package.pkg / "mac.lpy").write_text("(defmacro twice [x]")
        with contextlib.suppress(Exception):
            self.view(package)
        epoch = mm.epoch()
        for _ in range(3):
            with contextlib.suppress(Exception):
                self.view(package)
            assert mm.epoch() == epoch

    def test_a_helper_a_macro_module_imported_before_is_seen(
        self, package, monkeypatch
    ):
        # b imports h at top level, after b's own compile, so h is in no
        # record: mac compiles while b is loaded, and only the drop's removal
        # of mac's cache makes the next view compile mac again.
        monkeypatch.setattr(sys, "dont_write_bytecode", False)
        (package.pkg / "h.py").write_text("VALUE = 1\n")
        (package.pkg / "b.lpy").write_text(
            f"(import {package.name}.h as h)\n(defmacro val [] (return h.VALUE))"
        )
        (package.pkg / "mac.lpy").write_text(
            "(require .b *)\n(defmacro twice [x] (return (val)))"
        )
        kwargs = dict(package=package.name, search_dirs=(str(package.root),))
        pv.compile_view(
            "(require .b *)\n(= zz (val))", name=package.name + ".other", **kwargs
        )
        assert self.view(package).text.endswith("yy = 1")
        (package.pkg / "h.py").write_text("VALUE = 77\n")
        assert self.view(package).text.endswith("yy = 77")

    def test_a_second_instance_of_the_module_installs_nothing(
        self, package, monkeypatch
    ):
        # A second import of macro_modules (once a test or a reload dropped it
        # from sys.modules) must neither wrap path_stats again nor blind the
        # windows of the instance already in use.
        loader = importlib.machinery.SourceFileLoader
        stats, finders = loader.path_stats, list(sys.meta_path)
        namespace_exec = importlib.machinery.NamespaceLoader.exec_module
        monkeypatch.delitem(sys.modules, "lispy.lsp.macro_modules")
        second = importlib.import_module("lispy.lsp.macro_modules")
        assert second is not mm
        assert loader.path_stats is stats
        assert sys.meta_path == finders
        assert importlib.machinery.NamespaceLoader.exec_module is namespace_exec
        assert self.view(package).text.endswith("yy = 2 + 2")
        (package.pkg / "mac.lpy").write_text(THRICE)
        assert self.view(package).text.endswith("yy = 2 * 2 * 2")

    def test_a_second_instances_compile_holds_the_first_instances_drop(
        self, package, monkeypatch
    ):
        # Defect: each instance keeps its own condition and compile count, so
        # the first drops modules while a compile of the second imports them.
        assert self.view(package).text.endswith("yy = 2 + 2")
        monkeypatch.delitem(sys.modules, "lispy.lsp.macro_modules")
        second = importlib.import_module("lispy.lsp.macro_modules")
        assert second is not mm
        entered, release = threading.Event(), threading.Event()

        def hold():
            with second.fresh_modules():
                entered.set()
                release.wait(30)

        holder = threading.Thread(target=hold, daemon=True)
        dropper = threading.Thread(target=lambda: self.view(package), daemon=True)
        holder.start()
        try:
            assert entered.wait(10)
            (package.pkg / "mac.lpy").write_text(THRICE)
            dropper.start()
            dropper.join(0.5)
            assert dropper.is_alive()
            release.set()
            dropper.join(10)
            assert not dropper.is_alive()
        finally:
            release.set()
            holder.join(10)
            if dropper.ident is not None:
                dropper.join(10)

    def test_lispython_own_modules_are_never_tracked(self, tmp_path):
        path = os.path.join(os.path.realpath(tmp_path), "m.py")
        assert mm._trackable_p("pvmmown.m", path)
        assert not mm._trackable_p("lispy", path)
        assert not mm._trackable_p("lispy.lsp.m", path)
        assert not mm._trackable_p("pvmmown.m", os.__file__)


def fresh_with_p(node):
    """Whether node is a (with [(fresh-modules ...) ...] ...) form."""
    if not (isinstance(node, nodes.Paren) and len(node) > 1):
        return False
    if nodes.atom_text(node[0]) != "with":
        return False
    items = node[1]
    return isinstance(items, nodes.Expression) and any(
        isinstance(item, nodes.Paren)
        and len(item)
        and nodes.atom_text(item[0]) == "fresh-modules"
        for item in items
    )


# core/macro.py's compile entry points: compile-unit and its form-by-form
# variant, which the LSP's compile uses.
COMPILE_UNIT_HEADS = ("compile-unit", "compile-unit-forms")


def compile_unit_calls(path):
    """(nearest enclosing def's name or None, whether inside the body of a
    (with [(fresh-modules ...)] ...) form) of every call of a compile entry
    point (COMPILE_UNIT_HEADS) in the lispython source at path."""
    stack = [(form, None, False) for form in parser.parse(path.read_text())]
    found = []
    while stack:
        node, def_name, fresh = stack.pop()
        if isinstance(node, nodes.Paren) and len(node):
            head = nodes.atom_text(node[0])
            if head == "def" and len(node) > 1:
                def_name = nodes.atom_text(node[1])
            elif head in COMPILE_UNIT_HEADS:
                found.append((def_name, fresh))
            if fresh_with_p(node):
                stack.extend((child, def_name, True) for child in node[2:])
                stack.extend((child, def_name, fresh) for child in node[:2])
                continue
        if isinstance(node, nodes.Expression):
            stack.extend((child, def_name, fresh) for child in node)
        elif isinstance(node, nodes.Wrapper):
            stack.append((node.value, def_name, fresh))
    return found


def test_every_lsp_compile_tracks_its_modules():
    # Policy: a compile the LSP runs in its own process goes through
    # fresh-modules, or a module its requires import stays stale.
    lsp = pathlib.Path(pv.__file__).parent
    sites = [
        (path.name, def_name, fresh)
        for path in sorted(lsp.glob("*.lpy"))
        for def_name, fresh in compile_unit_calls(path)
    ]
    assert sorted(sites) == [
        ("python_view.lpy", "compile-buffer", True),
    ]


def test_moved_helpers_keep_one_home():
    for name in ["open_form_closers", "source_prefix", "py_name"]:
        assert getattr(server, name) is getattr(pv, name)


def test_concurrent_filter_swaps_leave_the_filters_as_they_were():
    # Defect: two threads inside warnings.catch_warnings at once (a view
    # compiling in a worker while the LSP's loop compiles diagnostics)
    # restore the filters out of order, leaving one's "ignore" in force.
    from lispy.core.errors import filtered_warnings

    original = list(warnings.filters)
    a_in, a_go, b_in, b_go = (threading.Event() for _ in range(4))

    def a():
        with filtered_warnings("ignore"):
            a_in.set()
            a_go.wait(5)

    def b():
        with filtered_warnings("always"):
            b_in.set()
            b_go.wait(5)

    first = threading.Thread(target=a)
    first.start()
    assert a_in.wait(5)
    second = threading.Thread(target=b)
    second.start()
    # Without the lock b is inside now; with it, only once a has left.
    b_in.wait(0.2)
    a_go.set()
    first.join(5)
    b_go.set()
    second.join(5)
    assert warnings.filters == original


SLOW_MACRO = """\
(defmacro slow []
  (= gate (__import__ "lpy_test_gate"))
  (.set gate.entered)
  (.wait gate.release 10)
  (return 1))
(= x (slow))
"""


def test_a_slow_macro_in_a_view_never_holds_up_another_compile(monkeypatch):
    # Defect: compiling a view holds a process-wide lock across the user's
    # macros, so the LSP loop's diagnostics wait at a backslash literal for
    # as long as a worker's macro runs, and a hanging macro freezes the LSP.
    gate = SimpleNamespace(entered=threading.Event(), release=threading.Event())
    monkeypatch.setitem(sys.modules, "lpy_test_gate", gate)
    worker = threading.Thread(target=pv.compile_view, args=(SLOW_MACRO,))
    worker.start()
    try:
        assert gate.entered.wait(10)
        done = threading.Event()

        def diagnostics():
            server.compute_diagnostics('(= s "a\\nb")', "file:///other.lpy")
            done.set()

        threading.Thread(target=diagnostics, daemon=True).start()
        # Released only after: the other compile must finish meanwhile.
        assert done.wait(5)
    finally:
        gate.release.set()
        worker.join(10)


def test_catch_warnings_has_one_home():
    # Rule: every swap of the process-wide warning filters goes through
    # errors.filtered_warnings, whose lock keeps threads from undoing each
    # other's swaps; a second catch_warnings site would bypass it.
    root = os.path.dirname(lispy.__file__)
    allowed = {os.path.join(root, "core", "errors.py")}
    offenders = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if not filename.endswith((".py", ".lpy")) or path in allowed:
                continue
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if re.search(r"warnings\.catch[_-]warnings|import catch", text):
                offenders.append(path)
    assert offenders == []


@pytest.mark.parametrize(
    "src",
    [
        '(= x [1])\n(print f"{(. x |)}")',
        '(= x [1])\n(print f"{(. x so|)}")',
        '(= x [1])\n(print f"""a{(. x so|)}b""")',
        '(= x [1])\n(print f"{x!r:>{(. x so|)}}")',
        '(= x [1])\n(print f"{(. x |)} {x}")',
    ],
)
def test_the_attribute_gate_agrees_with_the_view_inside_a_literal(src):
    """The cheap gate read before a cursor view must say attribute wherever
    the view does, f/t-string fields included."""
    line, col = locate(src, "|")
    source = src.replace("|", "")
    assert pv.view_at_cursor(source, line, col).attribute
    assert server.attribute_context_p(source, line, col)


def test_the_attribute_gate_rejects_a_field_outside_an_attribute():
    src = '(print f"{(len |)}")'
    line, col = locate(src, "|")
    assert not server.attribute_context_p(src.replace("|", ""), line, col)


@pytest.mark.parametrize(
    "src", ['(= x [1])\nf"{x)}"\n(. x so|)', '(= x [1])\nf"{x]}" (. x |']
)
def test_a_stray_closer_in_a_field_never_breaks_the_cursor_view_or_gate(src):
    line, col = locate(src, "|")
    source = src.replace("|", "")
    pv.view_at_cursor(source, line, col)
    server.attribute_context_p(source, line, col)  # must not raise


class TestSpelledConstants:
    """A Constant the unparse spells as an operation (-1, 1+2j, a nan) re-parses
    as a UnaryOp/BinOp.  Without the rule it gets no map entry, so its
    diagnostic falls back to the enclosing statement's span (None if tagged)."""

    def check(self, src, spelled, expected):
        view = view_of(src)
        assert spelled in view.text
        line, col = locate(view.text, spelled)
        start, end = (line, col), (line, col + len(spelled))
        for tagged in (False, True):
            assert pv.lpy_diagnostic_range(view, src, start, end, tagged) == expected

    def test_complex_token(self):
        src = "(print 1+2j x)"
        line, col = locate(src, "1+2j")
        self.check(src, "1+2j", ((line, col), (line, col + 4)))

    def test_macro_negative(self):
        src = "(defmacro neg [] (return (- 0 1)))\n(print (neg) x)"
        line, col = locate(src, "(neg)")
        self.check(src, "-1", ((line, col), (line, col + 5)))

    def test_macro_nan(self):
        src = '(defmacro nan [] (return (float "nan")))\n(print (nan) x)'
        line, col = locate(src, "(nan)")
        self.check(src, "1e309-1e309", ((line, col), (line, col + 5)))


def test_an_operator_never_spells_a_name():
    # Defect: a name run is matched by its mangled text alone, so the
    # operator - is taken for the name _ (Symbol.name of - is _).
    assert pv.spelled_in(["(- a b)"], (1, 0), (1, 7), "_") is None
    assert pv.spelled_in(["(- -x _)"], (1, 0), (1, 8), "x") == ((1, 4), (1, 5))
    assert pv.spelled_in(["(- -x _)"], (1, 0), (1, 8), "_") == ((1, 6), (1, 7))
