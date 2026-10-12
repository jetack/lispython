import os.path as osp
import sys
import types

from lispy.tools import l2py_s, macroexpand_then_compile, parse


class TestMetaFunctionsSync:
    """Test that core/meta_functions.py matches the transpiled result of core_meta_functions.lpy"""

    def test_meta_functions_in_sync(self):
        """core/meta_functions.py should match transpiled core_meta_functions.lpy"""
        base_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
        lpy_path = osp.join(base_dir, "src", "lispy", "core_meta_functions.lpy")
        py_path = osp.join(base_dir, "src", "lispy", "core", "meta_functions.py")

        with open(lpy_path, "r") as f:
            lpy_src = f.read()

        with open(py_path, "r") as f:
            py_contents = f.read()

        transpiled = l2py_s(lpy_src, no_lispy=True)

        assert transpiled == py_contents, (
            "core/meta_functions.py is out of sync with core_meta_functions.lpy. "
            "Run `lpy src/lispy/core_meta_functions.lpy` to regenerate."
        )


class TestRequireMacroList:
    """(require mod) registers all macros; the list selects, and [] selects none."""

    @staticmethod
    def _registered(monkeypatch, require_form):
        mod = types.ModuleType("require_list_fake")
        # setattr: a literal `mod.__macro_namespace` is name-mangled in a class body.
        setattr(mod, "__macro_namespace", {"a": lambda: 1, "b": lambda: 2})
        monkeypatch.setitem(sys.modules, "require_list_fake", mod)
        scope = {"__macro_namespace": {}}
        list(macroexpand_then_compile(parse(require_form), scope))
        return set(scope["__macro_namespace"])

    def test_absent_list_registers_all(self, monkeypatch):
        got = self._registered(monkeypatch, "(require require_list_fake)")
        assert got == {"require_list_fake.a", "require_list_fake.b"}

    def test_empty_list_registers_none(self, monkeypatch):
        got = self._registered(monkeypatch, "(require require_list_fake [])")
        assert got == set()

    def test_named_list_registers_those(self, monkeypatch):
        got = self._registered(monkeypatch, "(require require_list_fake [a])")
        assert got == {"a"}

    def test_star_registers_all_unqualified(self, monkeypatch):
        got = self._registered(monkeypatch, "(require require_list_fake *)")
        assert got == {"a", "b"}
