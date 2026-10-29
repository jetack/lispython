"""Every AST node the compiler builds must be constructible on every CPython.

From 3.15 an AST constructor raises TypeError on a keyword that is neither a
field nor an attribute of the node (3.13/3.14 only warn), so position keywords
passed to a node without ``_attributes`` (``arguments``, ``comprehension``,
``withitem``, ``match_case``, ...) break the compiler outright.  The check
below is version-independent: a stray keyword lands in the node's ``__dict__``
on every release that still accepts it.
"""

import ast
import importlib.machinery
import importlib.util
import inspect
import pathlib

import pytest

from lispy.core.importer import _lpy_source_to_code, _org_source_to_code
from lispy.tools import macroexpand_then_compile, parse

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "lispy"
LPY_FILES = sorted(SRC_DIR.rglob("*.lpy"))

SNIPPET = """
(def f [a b / c :d 1 *args e :f 2 **kwargs] ^int (return a))
(lambda [x :y 2 *r :z 3 **k] x)
(async-def g [x]
  (async-with [(h) as y] (await y))
  {, x async-for x in xs})
(with [(open "f") as f (open "g")] (f.read))
[x for x in xs if x]
{k v for [k v] in items}
(match p
  (case [1 *rest] if rest ...)
  (case {"a" 1 **kw} ...)
  (case (Point :x 0) ...))
(try (pass) (except [ValueError as e] (pass)) (finally (pass)))
"""


def _stray_keywords(module):
    stray = []
    for node in ast.walk(module):
        allowed = set(node._fields) | set(node._attributes)
        extra = set(vars(node)) - allowed
        if extra:
            stray.append((type(node).__name__, sorted(extra)))
    return stray


def _compile_module(src):
    return ast.Module(macroexpand_then_compile(parse(src)), type_ignores=[])


def test_snippet_nodes_have_no_stray_keywords():
    module = _compile_module(SNIPPET)
    assert _stray_keywords(module) == []
    compile(module, "<snippet>", "exec")


@pytest.mark.parametrize("path", LPY_FILES, ids=lambda p: p.name)
def test_source_tree_nodes_have_no_stray_keywords(path):
    module = _compile_module(path.read_text(encoding="utf-8"))
    assert _stray_keywords(module) == []


def test_source_to_code_accepts_the_running_signature(tmp_path):
    """3.15 calls ``source_to_code(data, path, fullname)``: forward what CPython
    passes instead of hard-coding one release's parameter list."""
    path = tmp_path / "lpy_hygiene_mod.lpy"
    path.write_text("(= x (+ 1 2))\n", encoding="utf-8")
    loader = importlib.machinery.SourceFileLoader("lpy_hygiene_mod", str(path))
    params = inspect.signature(_org_source_to_code).parameters
    extra = ["lpy_hygiene_mod"] if "fullname" in params else []
    code = _lpy_source_to_code(loader, path.read_bytes(), str(path), *extra)
    scope = {}
    exec(code, scope)
    assert scope["x"] == 3
    # And through the real import machinery, whatever this release passes.
    module = _import(loader)
    assert module.x == 3


def _import(loader):
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module
