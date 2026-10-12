import ast
import importlib.machinery
import io
import os
import os.path as osp
import runpy
import sys

from lispy.core.errors import compile_errors
from lispy.core.macro import macroexpand_then_compile
from lispy.core.parser import parse


def _is_lpy_file(filename):
    return osp.isfile(filename) and osp.splitext(filename)[1] == ".lpy"


def _lpy_code(source, path, compile_module):
    """Compile .lpy source to code through ``compile_module(ast.Module)``.

    Every error names this file and line as a LispyCompileError: the forms
    lispython refuses, and the SyntaxErrors CPython's compiler raises on the
    AST (e.g. a lazy import inside a def), whose positions are already ours.
    """
    with compile_errors(os.fsdecode(path), source):
        tree = ast.Module(macroexpand_then_compile(parse(source)), type_ignores=[])
        return compile_module(tree)


# # importlib.machinery.SourceFileLoader.source_to_code injection
importlib.machinery.SOURCE_SUFFIXES.insert(0, ".lpy")
_org_source_to_code = importlib.machinery.SourceFileLoader.source_to_code


def _lpy_source_to_code(self, data, path, *args, **kwargs):
    # Forward everything after (data, path) untouched: the signature grows across
    # CPython releases (3.15 added a positional ``fullname``), and only the
    # original implementation knows what each release accepts.
    if _is_lpy_file(path):
        return _lpy_code(
            data.decode("utf-8"),
            path,
            lambda tree: _org_source_to_code(self, tree, path, *args, **kwargs),
        )
    return _org_source_to_code(self, data, path, *args, **kwargs)


importlib.machinery.SourceFileLoader.source_to_code = _lpy_source_to_code  # type: ignore

# runpy._get_code_from_file injection
_org_get_code_from_file = runpy._get_code_from_file  # type: ignore


_PY312_PLUS = sys.version_info >= (3, 12)


def _lpy_get_code_from_file(*args):
    # Python 3.12+ changed signature from (run_name, fname) to (fname)
    # and now returns just code object instead of (code, fname) tuple
    from pkgutil import read_code

    if _PY312_PLUS:
        fname = args[0]
    else:
        fname = args[1]  # old signature: (run_name, fname)

    decoded_path = osp.abspath(os.fsdecode(fname))
    with io.open_code(decoded_path) as f:
        code = read_code(f)
    if code is None:
        if _is_lpy_file(fname):
            with open(decoded_path, "rb") as f:
                src = f.read().decode("utf-8")
            code = _lpy_code(src, fname, lambda tree: compile(tree, fname, "exec"))
        else:
            code = _org_get_code_from_file(*args)
            if not _PY312_PLUS:
                return code  # already a tuple (code, fname)
    if _PY312_PLUS:
        return code
    else:
        return (code, fname)


runpy._get_code_from_file = _lpy_get_code_from_file  # type: ignore

sys.path_importer_cache.clear()
