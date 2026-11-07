import ast
import builtins
import importlib.machinery
import importlib.util
import io
import marshal
import os
import os.path as osp
import runpy
import sys
import tempfile
import threading
import types

from lispy.core.compiler.utils import compile_nesting_errors
from lispy.core.errors import compile_errors
from lispy.core.macro import compile_unit
from lispy.core.parser import parse


def _is_lpy_file(filename):
    return osp.isfile(filename) and osp.splitext(filename)[1] == ".lpy"


def has_init_file(dir_path):
    """Whether dir_path is a package directory: it holds __init__.py or
    __init__.lpy."""
    return osp.exists(osp.join(dir_path, "__init__.py")) or osp.exists(
        osp.join(dir_path, "__init__.lpy")
    )


def package_root(dir_path):
    """(import root, package parts) of a module in dir_path: walking up while
    each directory is a package, the directory the walk stops at and the
    package names from the top one down.  A dir_path that is no package is
    its own import root, with no parts.  Pure (the CLI and the LSP both read
    it; the CLI adds the root to sys.path itself)."""
    parts = []
    current = dir_path
    while has_init_file(current):
        parts.insert(0, osp.basename(current))
        parent = osp.dirname(current)
        if parent == current:
            break
        current = parent
    return current, parts


def _lpy_code(source, path, compile_module, scope):
    """Compile .lpy source to code through ``compile_module(ast.Module)``.

    Every error names this file and line as a LispyCompileError: the forms
    lispython refuses, and the SyntaxErrors CPython's compiler raises on the
    AST (e.g. a lazy import inside a def), whose positions are already ours,
    and the AST nested past what compile() takes (compile_nesting_errors).

    compile() takes the module whole and names no line when it gives up, so
    the nesting error is placed by compiling the statements one by one: the
    first that fails alone is where compile() stopped (or, if an earlier
    statement raises its own error alone, which compiling the whole module
    never reached, that error).

    scope is the module's own compile-time scope (module_scope): macros a
    module defines or requires never reach another module.
    """
    with compile_errors(os.fsdecode(path), source):
        tree = ast.Module(compile_unit(parse(source), scope), type_ignores=[])
        try:
            return compile_module(tree)
        except RecursionError:
            for stmt in tree.body:
                with compile_nesting_errors([stmt]):
                    compile_module(ast.Module([stmt], type_ignores=[]))
            # Each fits alone: report the deepest path of the module.
            with compile_nesting_errors(tree.body):
                raise


def module_scope(name, package):
    """A fresh compile-time scope for the module name in package."""
    return {"__builtins__": builtins, "__name__": name, "__package__": package}


# # importlib.machinery.SourceFileLoader injection
importlib.machinery.SOURCE_SUFFIXES.insert(0, ".lpy")
_SOURCE_LOADER = importlib.machinery.SourceFileLoader
_org_source_to_code = _SOURCE_LOADER.source_to_code
_org_get_code = _SOURCE_LOADER.get_code
_org_path_stats = _SOURCE_LOADER.path_stats


def _lpy_compile(loader, data, path, *args, **kwargs):
    """(code, scope) of the .lpy source data at path: the one compile path of
    an imported .lpy (get_code and source_to_code both take it)."""
    name = loader.name
    package = name if loader.is_package(name) else name.rpartition(".")[0]
    scope = module_scope(name, package)
    code = _lpy_code(
        data.decode("utf-8"),
        path,
        lambda tree: _org_source_to_code(loader, tree, path, *args, **kwargs),
        scope,
    )
    return code, scope


def _lpy_source_to_code(self, data, path, *args, **kwargs):
    # Forward everything after (data, path) untouched: the signature grows across
    # CPython releases (3.15 added a positional ``fullname``), and only the
    # original implementation knows what each release accepts.
    if _is_lpy_file(path):
        return _lpy_compile(self, data, path, *args, **kwargs)[0]
    return _org_source_to_code(self, data, path, *args, **kwargs)


# ## The .lpy bytecode cache
#
# An .lpy module's code bakes in the expansions of every macro it used, and
# those come from other files: the modules it required (and what they
# required in turn), and whatever a macro imported while it expanded.
# CPython's pyc is validated against the module's own source alone (its
# mtime in whole seconds and its size), so an edit to a macro module would
# leave every module that used it stale for good.  lispython therefore keeps
# its own cache, `<pyc path>.lpyc` beside where the pyc would be (Python's
# own loader never reads it): a dependency record plus the code, as Racket's
# compiled/*.dep and GHC's .hi interface files do.  The record is a flat
# list of (real path, st_mtime_ns, st_size), the module's own source first;
# loading checks it with one stat per entry, and any change, missing file or
# unreadable cache compiles the module again.  Stamps, not content hashes:
# a hash would read every dependency on every import, and a stamp is what
# CPython's default pyc check trusts too (at nanosecond resolution here, so
# a same-size edit within one second is seen).
#
# A module's dependencies are
#  - its own source;
#  - the file of every macro in its macro table at the end of the compile
#    (its function's __module__), wherever it lives, with that module's own
#    record (_RECORDS, filled whenever an .lpy is compiled or loaded from its
#    cache in this process), so a require is followed transitively even when
#    the macro module was imported before this compile began;
#  - every .py/.lpy source read while it compiled (a thread-local window
#    around the compile, fed by path_stats, which SourceFileLoader.get_code
#    calls before it reads a source), outside the standard library,
#    site-packages and lispython itself (the version in the header covers
#    lispython): this catches a module a macro imports as it expands.
#    A nested .lpy import merges its whole record into every enclosing window.
#
# Accepted gap: a helper H that a macro module B imports at top level, where H
# was already in sys.modules before B (and the requiring module) compiled,
# through some unrelated import, is read in no window and is in no record,
# so an edit to H does not recompile the requiring module until something
# else does.  Closing it would take each macro module's full runtime import
# closure.

_CACHE_MAGIC = b"LPYC"
_LOCAL = threading.local()
_RECORDS = {}
# Real path to (st_mtime_ns, st_size) of every source path_stats observed, as
# of its latest load: a macro module with no record (a .py) is stamped as it
# was read, never as it is when a requiring module's compile ends.
_READ_STAMPS = {}
_cache_header = None
_excluded_dirs = None


def lpy_cache_path(path):
    """Where the cache of the .lpy source at path lives, or None when there
    is none (sys.implementation.cache_tag is None, or the path has no cache
    location): cache_from_source's pyc path, with .lpyc for .pyc, so
    sys.pycache_prefix and the optimization tag apply as to a pyc."""
    try:
        pyc = importlib.util.cache_from_source(path)
    except (NotImplementedError, ValueError):
        return None
    return osp.splitext(pyc)[0] + ".lpyc"


def lpy_record(path):
    """The dependency record of the .lpy at path as this process last
    compiled or loaded it from its cache: a tuple of (real path,
    st_mtime_ns, st_size), its own source first; None when it has none."""
    return _RECORDS.get(osp.realpath(path))


def _header():
    """What a cache must have been written under: the bytecode magic and the
    lispython version, read once."""
    global _cache_header
    if _cache_header is None:
        try:
            from importlib.metadata import version

            lispy_version = version("lispython")
        except Exception:
            lispy_version = "unknown"
        _cache_header = (bytes(importlib.util.MAGIC_NUMBER), lispy_version)
    return _cache_header


def _stamp(path):
    """(real path, st_mtime_ns, st_size) of path; OSError when it cannot be
    stat'ed."""
    st = os.stat(path)
    return osp.realpath(path), st.st_mtime_ns, st.st_size


def _excluded(real):
    """Whether the real path lies in the standard library, site-packages,
    dist-packages or lispython's own package: never a window dependency."""
    global _excluded_dirs
    if _excluded_dirs is None:
        # os.py's directory is the standard library's (sysconfig cannot be
        # asked here: its paths import modules, whose reads land here).
        _excluded_dirs = (
            osp.dirname(osp.realpath(os.__file__)),
            osp.dirname(osp.dirname(osp.realpath(__file__))),
        )
    parts = real.split(os.sep)
    if "site-packages" in parts or "dist-packages" in parts:
        return True
    for directory in _excluded_dirs:
        try:
            if osp.commonpath([real, directory]) == directory:
                return True
        except ValueError:
            pass
    return False


def _windows():
    """This thread's open compile windows, innermost last: each a dict of
    real path to (st_mtime_ns, st_size)."""
    windows = getattr(_LOCAL, "windows", None)
    if windows is None:
        windows = _LOCAL.windows = []
    return windows


def _record_into_windows(entries, *, filtered):
    windows = getattr(_LOCAL, "windows", None)
    if not windows:
        return
    for real, mtime, size in entries:
        if filtered and _excluded(real):
            continue
        for window in windows:
            window.setdefault(real, (mtime, size))


def _lpy_path_stats(self, path):
    # The stamp is taken here, before the loader reads the source, so it
    # predates the text compiled: an edit during the import makes the record
    # stale, never missed.
    try:
        stamp = _stamp(path)
    except OSError:
        pass
    else:
        _READ_STAMPS[stamp[0]] = stamp[1:]
        _record_into_windows([stamp], filtered=True)
    return _org_path_stats(self, path)


def _read_cache(cache_path, own):
    """The code in the cache at cache_path when its record still holds for
    the files as they are now (own is the source's stamp), with that record;
    None for a missing, unreadable, foreign or stale cache."""
    try:
        with open(cache_path, "rb") as f:
            data = f.read()
        if not data.startswith(_CACHE_MAGIC):
            return None
        start = len(_CACHE_MAGIC)
        length = int.from_bytes(data[start : start + 4], "little")
        header, deps = marshal.loads(data[start + 4 : start + 4 + length])
        if header != _header() or not deps or tuple(deps[0]) != own:
            return None
        for real, mtime, size in deps[1:]:
            if _stamp(real)[1:] != (mtime, size):
                return None
        code = marshal.loads(data[start + 4 + length :])
    except Exception:
        # A corrupt or crafted file can fail anywhere in the unpacking.
        return None
    if not isinstance(code, types.CodeType):
        return None
    return code, tuple(tuple(dep) for dep in deps)


def _write_cache(cache_path, deps, code, source_path):
    """Write the cache atomically (a temporary file in its directory, then a
    rename); any OSError leaves no cache, as CPython does for a pyc."""
    payload = marshal.dumps((_header(), deps))
    data = (
        _CACHE_MAGIC
        + len(payload).to_bytes(4, "little")
        + payload
        + marshal.dumps(code)
    )
    try:
        directory = osp.dirname(cache_path)
        os.makedirs(directory, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=directory, prefix=osp.basename(cache_path))
        try:
            with open(fd, "wb") as f:
                f.write(data)
            # The source's permission bits, writable by its owner, as
            # CPython gives a pyc (mkstemp creates the file owner-only).
            os.chmod(temp, (os.stat(source_path).st_mode & 0o666) | 0o200)
            os.replace(temp, cache_path)
        except OSError:
            try:
                os.unlink(temp)
            except OSError:
                pass
            raise
    except OSError:
        pass


def _dependencies(own, window, scope, fullname):
    """The flat record of a module just compiled: own (its source's stamp)
    first, then its macros' modules with their records, then the window's
    reads, each path once."""
    deps = {own[0]: own[1:]}
    namespace = scope.get("__macro_namespace") or {}
    for macro in list(namespace.values()):
        module_name = getattr(macro, "__module__", None)
        if not isinstance(module_name, str) or module_name == fullname:
            continue
        module_file = getattr(sys.modules.get(module_name), "__file__", None)
        if not isinstance(module_file, str):
            continue
        real = osp.realpath(module_file)
        if real in deps:
            continue
        record = _RECORDS.get(real)
        if record is None:
            stamp = _READ_STAMPS.get(real)
            if stamp is not None:
                record = ((real, *stamp),)
            else:
                try:
                    record = (_stamp(real),)
                except OSError:
                    continue
        for dep_real, mtime, size in record:
            deps.setdefault(dep_real, (mtime, size))
    for real, stamp in window.items():
        deps.setdefault(real, stamp)
    return tuple((real, *stamp) for real, stamp in deps.items())


def _lpy_get_code(self, fullname):
    source_path = self.get_filename(fullname)
    if not _is_lpy_file(source_path):
        return _org_get_code(self, fullname)
    try:
        # Called as SourceLoader.get_code calls it: path_stats is where a
        # source read is observed (the windows above, the LSP's tracking).
        self.path_stats(source_path)
    except OSError:
        pass
    cache_path = lpy_cache_path(source_path)
    try:
        own = _stamp(source_path)
    except OSError:
        own = None
    if own is not None and cache_path is not None:
        cached = _read_cache(cache_path, own)
        if cached is not None:
            code, deps = cached
            _RECORDS[own[0]] = deps
            _record_into_windows(deps, filtered=False)
            return code
    data = self.get_data(source_path)
    window = {}
    windows = _windows()
    windows.append(window)
    try:
        code, scope = _lpy_compile(self, data, source_path)
    finally:
        if windows and windows[-1] is window:
            windows.pop()
    if own is None:
        return code
    deps = _dependencies(own, window, scope, fullname)
    _RECORDS[own[0]] = deps
    _record_into_windows(deps, filtered=False)
    if cache_path is not None and not sys.dont_write_bytecode:
        _write_cache(cache_path, deps, code, source_path)
    return code


_SOURCE_LOADER.source_to_code = _lpy_source_to_code  # type: ignore
_SOURCE_LOADER.get_code = _lpy_get_code  # type: ignore
_SOURCE_LOADER.path_stats = _lpy_path_stats  # type: ignore

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
            code = _lpy_code(
                src,
                fname,
                lambda tree: compile(tree, fname, "exec"),
                module_scope("__main__", ""),
            )
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
