import os.path as osp
import sys
import types

import pytest

from lispy.core.errors import LispyCompileError
from lispy.core.macro import macroexpand_then_compile
from lispy.tools import l2py_s, parse


class TestMetaFunctionsSync:
    """core/meta_functions.py matches the transpiled core_meta_functions.lpy."""

    def test_meta_functions_in_sync(self):
        """core/meta_functions.py should match transpiled core_meta_functions.lpy"""
        base_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
        lpy_path = osp.join(base_dir, "src", "lispy", "core_meta_functions.lpy")
        py_path = osp.join(base_dir, "src", "lispy", "core", "meta_functions.py")

        with open(lpy_path) as f:
            lpy_src = f.read()

        with open(py_path) as f:
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


class TestRequireQualifier:
    """A qualified require registers every macro as qualifier.name, the
    qualifier being the name Python's import binds: an absolute module's
    dotted name, a relative module's last part, or the alias."""

    @staticmethod
    def _registered(monkeypatch, require_form, package, module):
        mod = types.ModuleType(module)
        setattr(mod, "__macro_namespace", {"a": lambda: 1, "b": lambda: 2})
        monkeypatch.setitem(sys.modules, module, mod)
        scope = {"__macro_namespace": {}, "__package__": package}
        list(macroexpand_then_compile(parse(require_form), scope))
        return set(scope["__macro_namespace"])

    @pytest.mark.parametrize(
        "form, package, module, keys",
        [
            ("(require reqq.mac)", None, "reqq.mac", {"reqq.mac.a", "reqq.mac.b"}),
            ("(require .mac)", "reqq", "reqq.mac", {"mac.a", "mac.b"}),
            ("(require ..sib.mac)", "reqq.sub", "reqq.sib.mac", {"mac.a", "mac.b"}),
            ("(require reqq.mac as m)", None, "reqq.mac", {"m.a", "m.b"}),
            ("(require .mac as m)", "reqq", "reqq.mac", {"m.a", "m.b"}),
            ("(require . as m)", "reqq", "reqq", {"m.a", "m.b"}),
            ("(require . *)", "reqq", "reqq", {"a", "b"}),
            ("(require .mac [a])", "reqq", "reqq.mac", {"a"}),
        ],
        ids=[
            "absolute",
            "relative",
            "nested relative",
            "absolute alias",
            "relative alias",
            "package alias",
            "package star",
            "relative list",
        ],
    )
    def test_qualifier(self, monkeypatch, form, package, module, keys):
        assert self._registered(monkeypatch, form, package, module) == keys

    @pytest.mark.parametrize(
        "form, fault",
        [
            ("(require)", "(require)"),
            ("(require 1)", "1"),
            ('(require "m")', '"m"'),
            ("(require .)", "."),
            ("(require ..)", ".."),
            ("(require m as)", "as"),
            ("(require m as a.b)", "a.b"),
            ("(require m as if)", "if"),
            ("(require m as [a])", "[a]"),
            ("(require m as c [a])", "[a]"),
            ("(require m as c *)", "*"),
            ("(require m * as c)", "as"),
            ("(require m [a] as c)", "as"),
            ("(require m [a] *)", "*"),
            ("(require m a)", "a"),
            ("(require m a b c)", "a"),
            ("(require m :as c)", ":as"),
            ("(require m [a 1])", "1"),
            ("(require mac.)", "mac."),
            ("(require a..b)", "a..b"),
            ("(require m.-)", "m.-"),
            ("(require a.if)", "a.if"),
            ("(require .mac.)", ".mac."),
            ("(require ..mac..x *)", "..mac..x"),
        ],
    )
    def test_a_malformed_require_is_a_compile_error_at_its_fault(self, form, fault):
        # Defect: a malformed require escapes as the ValueError or TypeError
        # of its own destructuring, with no position, or registers keys no
        # code can call (an alias kept as a selection).
        source = f"(= x 1)\n  {form}"
        with pytest.raises(LispyCompileError) as info:
            list(macroexpand_then_compile(parse(source), {"__macro_namespace": {}}))
        assert info.value.msg.startswith("(require ...) ")
        column = source.index(fault, source.index("(require")) - source.index("\n")
        assert (info.value.lineno, info.value.offset) == (2, column)

    @pytest.mark.parametrize("package", [None, ""], ids=["no package", "top level"])
    @pytest.mark.parametrize("form", ["(require .mac)", "(require .. *)"])
    def test_a_relative_require_outside_a_package_is_a_compile_error(
        self, package, form
    ):
        # Defect: importlib.import_module's bare TypeError ("the 'package'
        # argument is required") escapes with no position.
        scope = {"__macro_namespace": {}}
        if package is not None:
            scope["__package__"] = package
        with pytest.raises(LispyCompileError) as info:
            list(macroexpand_then_compile(parse(f"(= x 1)\n  {form}"), scope))
        assert "no known parent package" in info.value.msg
        assert (info.value.lineno, info.value.offset) == (2, 12)

    def test_a_malformed_relative_path_is_no_package_alone(self):
        # Defect: .mac. is read as a package alone, whose message suggests
        # (require .mac. as name), which fails again.
        scope = {"__macro_namespace": {}}
        with pytest.raises(LispyCompileError) as info:
            list(macroexpand_then_compile(parse("(require .mac.)"), scope))
        assert "expects a module name" in info.value.msg

    def test_a_package_alone_names_the_fix(self):
        # Python binds no name for from . import, so neither may a require:
        # the message must say what does.
        scope = {"__macro_namespace": {}}
        with pytest.raises(LispyCompileError) as info:
            list(macroexpand_then_compile(parse("(require .)"), scope))
        assert "as" in info.value.msg and "*" in info.value.msg
