"""The LSP's Python view: a buffer's generated Python and its position map.

Expected positions are read off the source and generated text with
``locate``, so every assertion names the word it lands on.
"""

import hashlib
import importlib
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
from lispy.core.importer import lpy_cache_path

pv = importlib.import_module("lispy.lsp.python_view")
server = importlib.import_module("lispy.lsp.server")
mm = importlib.import_module("lispy.lsp.macro_modules")


def locate(text, word, nth=0):
    """(1-based line, code-point column) of the nth occurrence of word."""
    index = -1
    for _ in range(nth + 1):
        index = text.index(word, index + 1)
    line = text.count("\n", 0, index) + 1
    return line, index - (text.rfind("\n", 0, index) + 1)


def plus(pos, cols):
    return pos[0], pos[1] + cols


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

    def test_the_module_location_stays_at_its_start(self):
        # A module's own location (the definition of an imported module) is
        # the empty range at its first character, whatever entry is there.
        src = "(= x 1)"
        assert pv.lpy_range(view_of(src), src, (1, 0), (1, 0)) == ((1, 0), (1, 0))


class TestFailures:
    @pytest.mark.parametrize(
        "src",
        [
            "(print 1",  # parse error
            "(= 1 2)",  # compile error
            "(defmacro boom [] (raise (ValueError)))\n(boom)",  # macro fails
        ],
    )
    def test_compile_failure_is_none(self, src):
        assert pv.compile_view(src) is None


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

    def test_unbalanced_buffer_falls_back_to_the_prefix(self):
        src = "(= my-x 1)\n(print my-x.fo\n(def g [] 1)"
        view = pv.view_at_cursor(src, *plus(locate(src, "fo"), 2))
        assert view.fallback is True
        assert view.text == "my_x = 1\nprint(my_x.fo)"
        assert view.cursor == plus(locate(view.text, "fo"), 2)

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
            ("top", "", str(tmp_path)),
            ("pkg.mod", "pkg", str(src)),
            ("pkg.a", "pkg.a", str(src)),
            ("pkg.a.m", "pkg.a", str(src)),
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
                import_root=str(tmp_path / "src"),
            )
        finally:
            for name in names:
                sys.modules.pop(name, None)
        assert view is not None and view.text.endswith("\nyy = 2 + 2")
        assert sys.path == path

    def test_import_root_stays_on_sys_path_while_any_user_holds_it(self):
        # Two compiles overlapping on one import root, the first ending first:
        # the root stays for the second and goes with it.
        root = "/nonexistent/pv-overlap-root"
        path = list(sys.path)
        first, second = pv._on_sys_path(root), pv._on_sys_path(root)
        first.__enter__()
        second.__enter__()
        try:
            first.__exit__(None, None, None)
            assert sys.path[0] == root
        finally:
            second.__exit__(None, None, None)
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

    def test_src_layout_config_and_mirrored_py(self, tmp_path):
        project, root = tmp_path / "project", tmp_path / "view"
        pkg = project / "src" / "pkg"
        self.write(pkg / "__init__.py", "from .helper import h")
        self.write(pkg / "helper.py", "h = 1")
        self.write(pkg / "m.lpy", "(= m 1)")
        self.write(pkg / "notes.txt", "")
        manifest = pv.ShadowManifest()
        pv.write_shadows(str(project), str(root), manifest=manifest)
        assert manifest.import_roots == [str(project / "src")]
        for name in ["__init__.py", "helper.py"]:
            assert os.readlink(root / "src" / "pkg" / name) == str(pkg / name)

        path = pv.write_pyright_config(
            str(project), str(root), import_roots=manifest.import_roots
        )
        extra = json.loads(open(path).read())["extraPaths"]
        assert extra == [str(root / "src"), str(project / "src"), str(project)]

        (pkg / "helper.py").unlink()
        self.write(pkg / "__init__.lpy", "(= i 1)")  # replaces the linked .py
        pv.write_shadows(str(project), str(root), manifest=manifest)
        assert not os.path.lexists(root / "src" / "pkg" / "helper.py")
        init = root / "src" / "pkg" / "__init__.py"
        assert not init.is_symlink() and init.read_text() == "i = 1"
        assert (pkg / "__init__.py").read_text() == "from .helper import h"

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
        assert pv.write_shadows(str(project), str(root)) == [
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

    def basedpyright_messages(self, tmp_path, project, root, **settings):
        """basedpyright's diagnostics on the view's src/pkg/m.py, under the
        view root's config as written plus settings."""
        manifest = pv.ShadowManifest()
        pv.write_shadows(str(project), str(root), manifest=manifest)
        path = pv.write_pyright_config(
            str(project), str(root), import_roots=manifest.import_roots
        )
        config = json.loads(open(path).read())
        config.update(settings, reportMissingImports="error")
        check = tmp_path / "check" / "pyrightconfig.json"
        self.write(check, json.dumps(config))
        out = subprocess.run(
            ["basedpyright", "--outputjson", "-p", str(check)]
            + [str(root / "src" / "pkg" / "m.py")],
            capture_output=True,
            text=True,
            timeout=60,
        )
        report = json.loads(out.stdout)
        assert report["summary"]["filesAnalyzed"] == 1
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
        # Only pkg is in the view: a pure-.py package, a top-level .py and a
        # stub-only module beside it resolve from the real src, while pkg.m2,
        # which exists only as .lpy, resolves from the view's shadow, and so
        # does pkg.both, whose stale .py beside it in the real src lacks new.
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
                    shadows[name] = pv.write_shadows(str(project), str(root))
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
        assert str(shadow) in pv.write_shadows(str(project), str(root))
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
        assert pv.write_shadows(str(project), str(root)) == shadows
        (project / "top.lpy").chmod(0)
        try:
            assert pv.write_shadows(str(project), str(root)) == shadows
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
            shadows = pv.write_shadows(str(project), str(root), manifest=manifest)
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
        shadows = pv.write_shadows(str(project), str(root), manifest=manifest)
        mod, top = root / "pkg" / "mod.py", root / "top.py"
        assert shadows == sorted([str(mod), str(top)])
        assert mod.read_text() == "my_x = 1"

        calls = []
        real = pv.compile_view
        monkeypatch.setattr(
            pv, "compile_view", lambda s, **kw: calls.append(s) or real(s, **kw)
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
        self.write(project / "pkg" / "mod.lpy", "(= 1 1)")  # stops compiling
        os.utime(project / "pkg" / "mod.lpy", ns=(stamp + 2 * 10**9,) * 2)
        assert pv.write_shadows(str(project), str(root), manifest=manifest) == []
        assert not mod.exists() and not top.exists()
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
            str(project), str(root), import_roots=[str(tmp_path)]
        )
        assert json.loads(open(path).read())["extraPaths"] == [str(project)]

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
        assert config["extraPaths"] == [str(project)]
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
            "extraPaths": [str(project), str(site)],
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

    def test_pyrefly_config_takes_the_projects_rules_never_its_paths(self, tmp_path):
        # Defects: the project's own preset, Python and error severities
        # never reach pyrefly; a path or tooling setting is copied, so
        # pyrefly reads the project's directories, interpreter or baseline
        # instead of the view's; [tool.pyrefly] is not read; a malformed
        # file fails the write instead of reading as no settings.
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
        # A malformed file is no settings, never a failed write.
        (project / "pyproject.toml").write_text("[tool.pyrefly\n")
        assert self.pyrefly_config(project, root)["preset"] == "basic"

    def test_pyrefly_default_preset_only_without_a_project_config(self, tmp_path):
        # Defects: a project with no pyrefly config runs pyrefly's own
        # stricter default instead of the basic preset pyrefly check runs on
        # it; or a project whose pyrefly config sets no preset is pinned to
        # basic, silently weaker than the default its own pyrefly runs.
        project, root = tmp_path / "project", tmp_path / "view"
        project.mkdir()
        assert self.pyrefly_config(project, root)["preset"] == "basic"
        (project / "pyproject.toml").write_text('[project]\nname = "p"\n')
        assert self.pyrefly_config(project, root)["preset"] == "basic"
        (project / "pyproject.toml").write_text(
            '[tool.pyrefly]\npython-version = "3.12"\n'
        )
        config = self.pyrefly_config(project, root)
        assert "preset" not in config
        assert config["python-version"] == "3.12"

    def test_pyrefly_config_checks_nothing_without_diagnostics(self, tmp_path):
        # Defect: with pythonDiagnostics false pyrefly still checks every
        # open buffer, at the project's preset, or the project's errors
        # table turns codes back on under preset off, for nobody.
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

    def test_pyrefly_search_path_is_pyrights(self, tmp_path):
        # Defect: the two backends' search paths drift apart, so a module
        # one resolves (a shadowed src package, a .venv package) the other
        # does not.  pyrefly's leads with the view root, which pyright
        # searches first as its execution root.
        project, root = tmp_path / "project", tmp_path / "view"
        (project / "src" / "pkg").mkdir(parents=True)
        site = project / ".venv" / "lib" / "python3.99" / "site-packages"
        site.mkdir(parents=True)
        roots = [str(project / "src"), str(tmp_path)]
        extra = json.loads(
            open(
                pv.write_pyright_config(str(project), str(root), import_roots=roots)
            ).read()
        )["extraPaths"]
        assert extra == [
            str(root / "src"),
            str(project / "src"),
            str(project),
            str(site),
        ]
        assert self.pyrefly_config(project, root, import_roots=roots)[
            "search-path"
        ] == [str(root), *extra]

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
        # With checking, the errors table is refused too.
        (project / "pyrefly.toml").write_text(
            'preset = "strict"\n[errors]\nnot-a-code = "warn"\n'
        )
        config = self.pyrefly_config(project, root, check=check)
        assert (config["preset"], "errors" in config) == ("basic", False)
        # Saved again with the same refused config: the file holds the
        # fallback, which differs from the candidate, yet pyrefly is not run
        # again and the warning is not repeated on every save.
        calls, warnings = len(checked), caplog.text.count("refuses")
        with caplog.at_level("WARNING", logger="lispy.lsp.python_view"):
            config = self.pyrefly_config(project, root, check=check)
        assert (config["preset"], "errors" in config) == ("basic", False)
        assert (len(checked), caplog.text.count("refuses")) == (calls, warnings)
        (project / "pyrefly.toml").write_text('preset = "strict"\n')
        assert self.pyrefly_config(project, root, check=check)["preset"] == "strict"
        calls = len(checked)
        self.pyrefly_config(project, root, check=check)
        assert len(checked) == calls

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
            import_root=str(package.root),
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
            import_root=str(package.root),
        )
        assert cache.view("u", 1, src, **kwargs).text.endswith("yy = 2 + 2")
        (package.pkg / "mac.lpy").write_text(THRICE)
        assert cache.view("u", 1, src, **kwargs).text.endswith("yy = 2 * 2 * 2")

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

    def test_a_module_outside_the_import_root_is_never_dropped(
        self, package, tmp_path, monkeypatch
    ):
        # Only modules under the import root are dropped: a library a macro
        # imports keeps its one instance (an extension refuses a second load).
        outside = tmp_path / "outside"
        outside.mkdir()
        ext = "pvmmoutsideext"
        (outside / (ext + ".py")).write_text("VALUE = 1\n")
        monkeypatch.syspath_prepend(str(outside))
        monkeypatch.delitem(sys.modules, ext, raising=False)
        (package.pkg / "mac.lpy").write_text(f"(import {ext})\n" + TWICE)
        try:
            self.view(package)
            module = sys.modules[ext]
            (outside / (ext + ".py")).write_text("VALUE = 22\n")
            (package.pkg / "mac.lpy").write_text(f"(import {ext})\n" + THRICE)
            assert self.view(package).text.endswith("yy = 2 * 2 * 2")
            assert sys.modules[ext] is module
        finally:
            sys.modules.pop(ext, None)

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
                    import_root=str(package.root),
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
        assert text() is None
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
            with mm.fresh_modules(None):
                entered.set()
                release.wait(30)

        def wait():
            with mm.fresh_modules(None):
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

    def test_a_missing_module_keeps_the_environment_settled(
        self, package, tmp_path, monkeypatch
    ):
        # A require of a module that does not exist moves the epoch only once
        # its directory changes, so every other view and shadow stays cached.
        bad = "(require .nosuch *)\n(= zz (twice 3))"
        (package.pkg / "bad.lpy").write_text(bad)
        (package.pkg / "mod.lpy").write_text("(require .mac *)\n(= yy (twice 2))")
        cache = pv.ViewCache()
        kwargs = dict(package=package.name, import_root=str(package.root))

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
        compile_view = pv.compile_view
        monkeypatch.setattr(
            pv,
            "compile_view",
            lambda *a, **k: compiled.append(a) or compile_view(*a, **k),
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

    def test_a_second_instance_of_the_module_installs_nothing(
        self, package, monkeypatch
    ):
        # A second import of macro_modules (once a test or a reload dropped it
        # from sys.modules) must neither wrap path_stats again nor blind the
        # windows of the instance already in use.
        loader = importlib.machinery.SourceFileLoader
        stats, finders = loader.path_stats, list(sys.meta_path)
        monkeypatch.delitem(sys.modules, "lispy.lsp.macro_modules")
        second = importlib.import_module("lispy.lsp.macro_modules")
        assert second is not mm
        assert loader.path_stats is stats
        assert sys.meta_path == finders
        assert self.view(package).text.endswith("yy = 2 + 2")
        (package.pkg / "mac.lpy").write_text(THRICE)
        assert self.view(package).text.endswith("yy = 2 * 2 * 2")

    def test_lispython_own_modules_are_never_tracked(self, tmp_path):
        root = os.path.realpath(tmp_path)
        path = os.path.join(root, "m.py")
        assert mm._trackable_p("pvmmown.m", path, root)
        assert not mm._trackable_p("lispy", path, root)
        assert not mm._trackable_p("lispy.lsp.m", path, root)


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


def compile_unit_calls(path):
    """(nearest enclosing def's name or None, whether inside the body of a
    (with [(fresh-modules ...)] ...) form) of every (compile-unit ...) call
    in the lispython source at path."""
    stack = [(form, None, False) for form in parser.parse(path.read_text())]
    found = []
    while stack:
        node, def_name, fresh = stack.pop()
        if isinstance(node, nodes.Paren) and len(node):
            head = nodes.atom_text(node[0])
            if head == "def" and len(node) > 1:
                def_name = nodes.atom_text(node[1])
            elif head == "compile-unit":
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
