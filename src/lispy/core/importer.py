import ast
import importlib.machinery
import io
import os
import os.path as osp
import runpy
import sys

from lispy.core.macro import macroexpand_then_compile
from lispy.core.parser import parse


def _is_sy_file(filename):
    return osp.isfile(filename) and osp.splitext(filename)[1] == ".lpy"


# # importlib.machinery.SourceFileLoader.source_to_code injection
importlib.machinery.SOURCE_SUFFIXES.insert(0, ".lpy")
_org_source_to_code = importlib.machinery.SourceFileLoader.source_to_code


def _sy_source_to_code(self, data, path, *, _optimize=-1):
    if _is_sy_file(path):
        source = data.decode("utf-8")
        parsed = parse(source)
        data = ast.Module(macroexpand_then_compile(parsed), type_ignores=[])

    return _org_source_to_code(self, data, path, _optimize=_optimize)


importlib.machinery.SourceFileLoader.source_to_code = _sy_source_to_code  # type: ignore

# runpy._get_code_from_file injection
_org_get_code_from_file = runpy._get_code_from_file  # type: ignore


def _sy_get_code_from_file(run_name, fname):
    from pkgutil import read_code

    decoded_path = osp.abspath(os.fsdecode(fname))
    with io.open_code(decoded_path) as f:
        code = read_code(f)
    if code is None:
        if _is_sy_file(fname):
            with open(decoded_path, "rb") as f:
                src = f.read().decode("utf-8")
            parsed = parse(src)
            ast_module = ast.Module(macroexpand_then_compile(parsed), type_ignores=[])
            code = compile(ast_module, fname, "exec")
        else:
            code = _org_get_code_from_file(run_name, fname)[0]
    return [code, fname]


runpy._get_code_from_file = _sy_get_code_from_file  # type: ignore

sys.path_importer_cache.clear()
