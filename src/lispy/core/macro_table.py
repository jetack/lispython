"""A scope's macro table, keyed by name identity (name_key).

A macro is called by any spelling of its name, as every other name is:
``(defmacro my_mac ...)`` answers ``(my-mac ...)`` and ``(my_mac ...)``, and
``mod.my-mac`` is ``mod.my_mac`` (compiler/literal.py name_key, per dotted
part). Registering a name that is already there replaces its entry, macro
and spelling both: the last definition wins, as a Common Lisp or Clojure
redefinition does.

MacroTable is a MutableMapping over its own dict, never a dict subclass,
because dict's C methods (update, setdefault, pop, copy, the constructor)
bypass an overridden ``__setitem__``/``__getitem__`` and would store or miss
a key by its raw spelling.

The table a module's code registers into at runtime is a MacroTable too
(defmacro and require emit ``(.setdefault (globals) "__macro_namespace"
(MacroTable))``, core_meta_functions.lpy), not only the compile-time one:
with a plain dict, defining my_mac and then my-mac would leave two keys, and
a module requiring it, which copies the entries in order, could end with
either body depending on which spelling it is called by.

This module imports only name_key, so the generated code that imports it
costs nothing beyond what ``import lispy`` already loaded.
"""

import functools
from collections.abc import ItemsView, Mapping, MutableMapping, ValuesView

from lispy.core.compiler.literal import name_key

# Every macro lookup (every call head of every expansion pass) asks for a
# key; the mangling behind it is pure, so it is computed once per spelling.
_key = functools.lru_cache(maxsize=4096)(name_key)


class _Items(ItemsView):
    # A snapshot, as list(dict.items()) is: running code (an nREPL
    # connection's defmacro) may register into the table meanwhile.
    def __iter__(self):
        return iter(list(self._mapping._entries.values()))


class _Values(ValuesView):
    def __iter__(self):
        return iter([macro for _, macro in list(self._mapping._entries.values())])


class MacroTable(MutableMapping):
    """A scope's macro table (its ``__macro_namespace``), one entry per name
    (name_key), each under the spelling it was last registered by, which is
    what iteration yields. It carries the bookkeeping that describes it,
    kept on the table and never as a key in the scope, which is the user's
    globals (completion would offer it, ``(globals)`` would show it).
    Dropping the table drops its record with it, so a scope whose table was
    replaced is seeded again, and no side table outlives a scope.

    prelude: the built-in macros the scope was seeded with, what every later
    unit start copies from and what tells a built-in from a user macro.
    seeded: the keys a unit start (seed_prelude, MODULE, TOP) put there from
    the prelude and nothing has registered since: the entries a TOP start
    may take back. A require never adds to it. Its keys are spellings; look
    them up in the table, which compares them by name. Both are read
    through find_macro, since a caller may give a plain dict as prelude."""

    __slots__ = ("_entries", "prelude", "seeded")

    def __init__(self, table=(), prelude=None):
        self._entries = {}
        self.prelude = {} if prelude is None else prelude
        self.seeded = set()
        self.update(list(table.items()) if isinstance(table, Mapping) else table)

    def entry(self, name):
        """``(spelling, macro)`` of the macro name stands for, or None."""
        if not isinstance(name, str):
            return None
        return self._entries.get(_key(name))

    def __getitem__(self, name):
        found = self.entry(name)
        if found is None:
            raise KeyError(name)
        return found[1]

    def __setitem__(self, name, macro):
        self._entries[_key(name)] = (name, macro)

    def __delitem__(self, name):
        if not isinstance(name, str):
            raise KeyError(name)
        del self._entries[_key(name)]

    def __contains__(self, name):
        return self.entry(name) is not None

    def __iter__(self):
        return iter([spelling for spelling, _ in list(self._entries.values())])

    def __len__(self):
        return len(self._entries)

    def items(self):
        return _Items(self)

    def values(self):
        return _Values(self)

    def clear(self):
        self._entries.clear()

    def copy(self):
        table = MacroTable(prelude=self.prelude)
        table._entries = dict(self._entries)
        table.seeded = set(self.seeded)
        return table

    def __repr__(self):
        return f"MacroTable({dict(self.items())!r})"


def find_entry(table, name):
    """``(spelling, macro)`` of the macro name stands for in table, or None.
    table may be any mapping: a plain dict (a hand-written .py macro module's
    ``__macro_namespace``, a scope a caller built) is read by exact spelling
    first, then by name_key, the last matching key winning."""
    if not isinstance(name, str) or table is None:
        return None
    if isinstance(table, MacroTable):
        return table.entry(name)
    macro = table.get(name)
    if macro is not None:
        return (name, macro)
    key = _key(name)
    found = None
    for spelling, macro in list(table.items()):
        if isinstance(spelling, str) and _key(spelling) == key:
            found = (spelling, macro)
    return found


def find_macro(table, name):
    """The macro name stands for in table (any mapping, find_entry), or
    None."""
    found = find_entry(table, name)
    return None if found is None else found[1]


def required_macro(table, name):
    """The macro name stands for in a required module's table, for
    ``(require mod [name ...])``: a name the module lacks is a KeyError."""
    macro = find_macro(table, name)
    if macro is None:
        raise KeyError(name)
    return macro
