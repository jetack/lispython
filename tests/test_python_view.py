"""The LSP's Python view: a buffer's generated Python and its position map.

Expected positions are read off the source and generated text with
``locate``, so every assertion names the word it lands on.
"""

import hashlib
import importlib
import json
import ntpath
import os
import shutil
import subprocess
import sys
import threading
import time

import pytest

import lispy  # noqa: F401  (registers the .lpy importer)

pv = importlib.import_module("lispy.lsp.python_view")
server = importlib.import_module("lispy.lsp.server")


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
        assert (view.text, view.partial) == (line, partial)
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
        real = pv.compile_view

        def counting(source, **kw):
            calls.append(source)
            return real(source, **kw)

        monkeypatch.setattr(pv, "compile_view", counting)
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
        assert self.basedpyright_messages(tmp_path, project, root) == []

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

    def test_pyright_config(self, tmp_path, monkeypatch):
        project, root = tmp_path / "project", tmp_path / "view"
        site = project / ".venv" / "lib" / "python3.99" / "site-packages"
        site.mkdir(parents=True)
        monkeypatch.setattr(sys, "prefix", "/envs/my-venv")
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        path = pv.write_pyright_config(str(project), str(root))
        assert json.loads(open(path).read()) == {
            "typeCheckingMode": "off",
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


def test_moved_helpers_keep_one_home():
    for name in ["open_form_closers", "source_prefix", "py_name"]:
        assert getattr(server, name) is getattr(pv, name)
