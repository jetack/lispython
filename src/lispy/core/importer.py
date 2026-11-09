import ast
import builtins
import hashlib
import importlib.machinery
import importlib.util
import io
import keyword
import marshal
import os
import os.path as osp
import runpy
import sys
import threading
import types
import weakref

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


def finder_suffixes():
    """The file suffixes CPython's FileFinder tries, in its order (extension
    modules, then source, then bytecode), read from importlib.machinery at
    each call, where this module adds .lpy to the source ones."""
    return (
        importlib.machinery.EXTENSION_SUFFIXES
        + importlib.machinery.SOURCE_SUFFIXES
        + importlib.machinery.BYTECODE_SUFFIXES
    )


def _same_dir_key(path):
    """The key two spellings of one directory share: its real path,
    case-normalised, so a root given as a workspace folder's real path
    matches the same directory reached through a symlinked spelling."""
    return osp.normcase(osp.realpath(path))


def _namespace_portion_p(dir_path):
    """Whether CPython's FileFinder, searching dir_path's parent, would
    import dir_path as a package: its name is an identifier that is no
    keyword, and it is a regular package (it holds an __init__) or no
    module file of its name with a finder suffix sits beside it, which
    FileFinder would pick over a namespace portion."""
    name = osp.basename(dir_path)
    if not name.isidentifier() or keyword.iskeyword(name):
        return False
    if has_init_file(dir_path):
        return True
    parent = osp.dirname(dir_path)
    return not any(
        osp.isfile(osp.join(parent, name + suffix)) for suffix in finder_suffixes()
    )


def package_root(dir_path, roots=()):
    """(import root, package parts) of a module in dir_path: the directory
    the module is imported from and the package names from the top one down.

    Walking up while each directory is a regular package (it holds
    __init__.py or __init__.lpy) gives the parts; with no roots, the
    directory the walk stops at is the import root, so a dir_path that is
    no package is its own import root, with no parts.

    roots are the search roots (sys.path entries) the caller imports from.
    A namespace package (PEP 420) exists only relative to one, so when the
    directory the walk stopped at is not itself a root and a root lies above
    it, the walk goes on through namespace-package directories up to the
    nearest (deepest) such root, which becomes the import root, every
    directory in between a part, as the importer names a module by how it
    was imported (loader.name).  If any directory on that way cannot be a
    namespace portion (_namespace_portion_p: no identifier, or a module file
    of its name beside it, which FileFinder imports instead), the regular
    walk's result stands unchanged.  Directories match a root by real path
    (_same_dir_key); the import root keeps dir_path's own spelling.

    Pure (the CLI, the nREPL and the LSP read it; the CLI adds the root to
    sys.path itself)."""
    parts = []
    current = dir_path
    while has_init_file(current):
        parts.insert(0, osp.basename(current))
        parent = osp.dirname(current)
        if parent == current:
            break
        current = parent
    root_keys = {_same_dir_key(root) for root in roots}
    if not root_keys or _same_dir_key(current) in root_keys:
        return current, parts
    between = []
    directory = current
    while True:
        between.insert(0, directory)
        parent = osp.dirname(directory)
        if parent == directory:
            return current, parts
        if _same_dir_key(parent) in root_keys:
            break
        directory = parent
    if not all(_namespace_portion_p(d) for d in between):
        return current, parts
    return parent, [osp.basename(d) for d in between] + parts


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
#  - for every macro in its macro table at the end of the compile, the
#    record of the code loaded into the module object its function's
#    __module__ names (_MODULE_RECORDS), wherever it lives: that is the code
#    whose expansions this module baked in, so a require is followed
#    transitively even when the macro module was imported before this
#    compile began, and a failed reload, which leaves the old module in
#    sys.modules, still names the old code.  An untagged module (exec'd
#    before it was put in sys.modules, or imported before this importer was
#    installed) falls back to its file: its record (_RECORDS), its read stamp
#    (_READ_STAMPS), or its stamp now;
#  - every .py/.lpy source read while it compiled (a thread-local window
#    around the compile, fed by path_stats, which SourceFileLoader.get_code
#    calls before it reads a source), outside the standard library,
#    site-packages and lispython itself (the header covers lispython: a
#    fingerprint of the stamps of every .py/.lpy in its package, so an edit
#    to the compiler, the prelude or the builtins in a checkout recompiles
#    every module, while an installed version string would not change):
#    this catches a module a macro imports as it expands.
#    A nested .lpy import merges its whole record into every enclosing window.
# Entries are distinct (real path, st_mtime_ns, st_size) triples, so one path
# may appear with two stamps; such a record never validates again.
#
# Accepted gap: a helper H that a macro module B imports at top level, where H
# was already in sys.modules before B (and the requiring module) compiled,
# through some unrelated import, is read in no window and is in no record,
# so an edit to H does not recompile the requiring module until something
# else does.  Closing it would take each macro module's full runtime import
# closure.

_CACHE_MAGIC = b"LPYC"
_LOCAL = threading.local()
# Real path of an .lpy to the record of the code this process loaded or last
# compiled from it; none while its latest compile failed.  Per file: read by
# lpy_record (the LSP) and the untagged fallback of _dependencies.
_RECORDS = {}
# Path as the loader stat'ed it (a module's __file__) to (st_mtime_ns,
# st_size), as of its latest read: how a .py's read stamp reaches its module's
# tag, and the untagged fallback.  Keyed by the path as given, so an import
# costs one stat, no realpath.
_READ_STAMPS = {}
# Module object to the record of the code its loader's get_code returned for
# it (_tag_module): what a requiring module's record takes for a macro module.
# A side table, never a key in the module's dict (its globals are the user's).
_MODULE_RECORDS = weakref.WeakKeyDictionary()
_cache_header = None
# Real path of lispython's package directory: excluded from every record,
# since the header's fingerprint covers it.
_LISPY_DIR = osp.dirname(osp.dirname(osp.realpath(__file__)))
_TEMP_ATTEMPTS = 3
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
    """The dependency record of the code this process loaded for the .lpy at
    path (compiled, or from its cache): a tuple of (real path, st_mtime_ns,
    st_size), its own source first.  None when it has none, which includes
    while its latest compile failed: a record only ever describes code that
    is loaded."""
    return _RECORDS.get(osp.realpath(path))


def _header():
    """What a cache must have been written under: the bytecode magic and the
    fingerprint of lispython's own sources (_compiler_fingerprint), computed
    once per process."""
    global _cache_header
    if _cache_header is None:
        _cache_header = (bytes(importlib.util.MAGIC_NUMBER), _compiler_fingerprint())
    return _cache_header


def _compiler_fingerprint():
    """A digest of the (path relative to _LISPY_DIR, st_mtime_ns, st_size)
    stamps of every .py and .lpy file in lispython's package, sorted: the
    compiler, the prelude and the builtins whose output a cache holds.  An
    unreadable file or directory folds in a marker instead of a stamp."""
    stamps = []

    def unreadable(error):
        stamps.append(("?", error.filename or "", 0, 0))

    for directory, dirs, files in os.walk(_LISPY_DIR, onerror=unreadable):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            if not name.endswith((".py", ".lpy")):
                continue
            path = osp.join(directory, name)
            relative = osp.relpath(path, _LISPY_DIR)
            try:
                st = os.stat(path)
            except OSError:
                stamps.append(("?", relative, 0, 0))
            else:
                stamps.append(("", relative, st.st_mtime_ns, st.st_size))
    stamps.sort()
    return hashlib.blake2b(repr(stamps).encode(), digest_size=16).hexdigest()


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
            _LISPY_DIR,
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
    """This thread's open compile windows, innermost last: each a dict whose
    keys, in insertion order, are (real path, st_mtime_ns, st_size)."""
    windows = getattr(_LOCAL, "windows", None)
    if windows is None:
        windows = _LOCAL.windows = []
    return windows


def _record_into_windows(entries, *, filtered):
    windows = getattr(_LOCAL, "windows", None)
    if not windows:
        return
    for entry in entries:
        if filtered and _excluded(entry[0]):
            continue
        for window in windows:
            window.setdefault(tuple(entry))


def _lpy_path_stats(self, path):
    # The stamp is taken here, before the loader reads the source, so it
    # predates the text compiled: an edit during the import makes the record
    # stale, never missed.
    try:
        st = os.stat(path)
    except OSError:
        pass
    else:
        _READ_STAMPS[path] = (st.st_mtime_ns, st.st_size)
        if getattr(_LOCAL, "windows", None):
            _record_into_windows(
                [(osp.realpath(path), st.st_mtime_ns, st.st_size)], filtered=True
            )
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
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        # The source's permission bits, writable by its owner, given at
        # creation so the umask applies, as CPython's pyc writer does.
        mode = (os.stat(source_path).st_mode & 0o666) | 0o200
        for attempt in range(_TEMP_ATTEMPTS):
            # A random name, so a temp file a crashed writer left blocks nothing.
            temp = f"{cache_path}.{os.urandom(8).hex()}.tmp"
            try:
                fd = os.open(temp, os.O_EXCL | os.O_CREAT | os.O_WRONLY, mode)
                break
            except FileExistsError:
                if attempt == _TEMP_ATTEMPTS - 1:
                    raise
        try:
            with open(fd, "wb") as f:
                f.write(data)
            os.replace(temp, cache_path)
        except OSError:
            try:
                os.unlink(temp)
            except OSError:
                pass
            raise
    except OSError:
        pass


def _module_record(module):
    """The record of the code loaded into module (its tag), else, for an
    untagged module, of its file; None when there is none."""
    try:
        record = _MODULE_RECORDS.get(module)
    except TypeError:  # not weak-referenceable
        record = None
    if record is not None:
        return record
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        return None
    real = osp.realpath(module_file)
    record = _RECORDS.get(real)
    if record is not None:
        return record
    stamp = _READ_STAMPS.get(module_file)
    if stamp is not None:
        return ((real, *stamp),)
    try:
        return (_stamp(real),)
    except OSError:
        return None


def _dependencies(own, window, scope, fullname):
    """The flat record of a module just compiled: own (its source's stamp)
    first, then its macros' modules with their records, then the window's
    reads, each triple once."""
    deps = {own: None}
    covered = set()
    namespace = scope.get("__macro_namespace") or {}
    for macro in list(namespace.values()):
        module_name = getattr(macro, "__module__", None)
        if not isinstance(module_name, str) or module_name == fullname:
            continue
        module = sys.modules.get(module_name)
        if module is None or id(module) in covered:
            continue
        covered.add(id(module))
        for dep in _module_record(module) or ():
            deps.setdefault(tuple(dep))
    for dep in window:
        deps.setdefault(dep)
    return tuple(deps)


def _tag_module(loader, fullname, record):
    """Tag the module that will run the code loader.get_code is returning
    with record: the module in sys.modules whose spec names this very
    loader, as importlib's load and reload run it.  A caller with another
    loader (a fresh find_spec, module_from_spec) tags nothing; runpy -m of
    an already imported module reuses its spec, so it tags that module with
    the stamps it already holds.

    The tag is the union of the new record and the module's previous tag:
    defmacro registers into the module's existing table (scope_table in
    core/macro_table.py), so a reload (a successful one, or one whose exec
    fails partway) leaves the earlier code's macros in the module, and a
    module requiring it may expand them.  A record holding two stamps of one
    path never validates again, which costs one recompile in the next
    process, never a stale load."""
    module = sys.modules.get(fullname)
    if getattr(getattr(module, "__spec__", None), "loader", None) is not loader:
        return
    try:
        previous = _MODULE_RECORDS.get(module, ())
        merged = dict.fromkeys(tuple(dep) for dep in record)
        merged.update((tuple(dep), None) for dep in previous if dep not in merged)
        _MODULE_RECORDS[module] = tuple(merged)
    except TypeError:  # not weak-referenceable
        pass


def _lpy_get_code(self, fullname):
    source_path = self.get_filename(fullname)
    if not _is_lpy_file(source_path):
        code = _org_get_code(self, fullname)
        # The original get_code ran path_stats on source_path (the read stamp).
        stamp = _READ_STAMPS.get(source_path)
        if stamp is not None:
            _tag_module(self, fullname, ((osp.realpath(source_path), *stamp),))
        return code
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
            _tag_module(self, fullname, deps)
            return code
    # The record of the code loaded before goes now: if this compile fails, no
    # record may describe code that is no longer what the file holds.
    _RECORDS.pop(osp.realpath(source_path), None)
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
    _tag_module(self, fullname, deps)
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
