"""The LSP's Python view: a buffer's generated Python and its position map.

Expected positions are read off the source and generated text with
``locate``, so every assertion names the word it lands on.
"""

import hashlib
import importlib
import json
import os
import sys

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

    def test_sentinel_already_in_the_buffer_is_none(self):
        src = f"(= {pv.CURSOR_SENTINEL} 1)\n(print my-x.fo)"
        assert pv.view_at_cursor(src, *plus(locate(src, "fo"), 2)) is None


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

    def test_module_names(self, tmp_path):
        root = str(tmp_path)
        names = [
            pv.module_names(root, os.path.join(root, *parts))
            for parts in [("top.lpy",), ("pkg", "__init__.lpy"), ("pkg", "a", "m.lpy")]
        ]
        assert names == [("top", ""), ("pkg", "pkg"), ("pkg.a.m", "pkg.a")]

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
