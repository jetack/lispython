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
(defmacro and require register through scope_table, core_meta_functions.lpy),
not only the compile-time one: with a plain dict, defining my_mac and then
my-mac would leave two keys, and a module requiring it could end with either
body depending on which spelling it is called by. Every scope lispython
builds starts with a MacroTable; scope_table converts a plain dict it finds
(a scope a caller built), so the rule holds in any scope. A plain dict is
left only in foreign tables (a hand-written .py macro module's), which are
read, never written, through find_entry.

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
    unit start copies from and what tells a built-in from a user macro; None
    until seed_prelude (core/macro.py) seeds the table, which it does once.
    seeded: the keys a unit start (seed_prelude, MODULE, TOP) put there from
    the prelude and nothing has registered or removed since: the entries a
    TOP start may take back. Registering or removing a name drops the entry
    it replaces from it (``__setitem__``/``__delitem__``), so a key a
    require or defmacro registers is theirs from then on, even when it is
    the very macro the start seeded; only a unit start adds to it, after
    setting the entry. Its keys are spellings (an entry's stored one); look
    them up in the table, which compares them by name.

    A unit start never reshapes the table in place: it builds the new state
    off to the side and swaps it in (``reset``), so a reader holding no lock
    (nREPL completion) iterates a whole old or a whole new table, never one
    half cleared."""

    __slots__ = ("_entries", "prelude", "seeded")

    def __init__(self, table=(), prelude=None):
        self._entries = {}
        self.prelude = prelude
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
        entries = self._entries
        key = _key(name)
        old = entries.get(key)
        entries[key] = (name, macro)
        if old is not None:
            self.seeded.discard(old[0])

    def __delitem__(self, name):
        if not isinstance(name, str):
            raise KeyError(name)
        old = self._entries.pop(_key(name))
        self.seeded.discard(old[0])

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
        self.reset(MacroTable(prelude=self.prelude))

    def reset(self, table):
        """Take table's state (entries, seeded, prelude) as this table's own,
        by attribute stores only, so an unlocked reader sees the old whole
        table or the new one. table is built off to the side and not used
        after. A macro other running code registers into the old entries
        while the new state is being built is lost, as it would be by a
        clear() in place, and one racing the swap may discard its replaced
        spelling from the new seeded set, un-seeding a built-in it never
        replaced there (a later TOP start then keeps that built-in)."""
        self.seeded = table.seeded
        self._entries = table._entries
        self.prelude = table.prelude

    def copy(self):
        table = MacroTable(prelude=self.prelude)
        table._entries = dict(self._entries)
        table.seeded = set(self.seeded)
        return table

    def __repr__(self):
        return f"MacroTable({dict(self.items())!r})"


def scope_table(scope):
    """scope's macro table for registering into, made a MacroTable first: a
    missing one is created, a plain dict replaced by a MacroTable of its
    entries (unseeded, so a later compile_unit still seeds it). The one way
    generated defmacro and require code reaches the table."""
    table = scope.get("__macro_namespace")
    if table is None:
        # setdefault: two threads creating the table at once share one.
        table = scope.setdefault("__macro_namespace", MacroTable())
    if not isinstance(table, MacroTable):
        # The swap is not atomic: a macro other running code registers in the
        # plain dict meanwhile is lost, which only a scope a caller built with
        # a plain dict (every scope lispython builds starts with a MacroTable)
        # can reach.
        table = scope["__macro_namespace"] = MacroTable(table)
    return table


def find_entry(table, name):
    """``(spelling, macro)`` of the macro name stands for in table, or None.
    table may be None (no macros) or any mapping: a plain dict (a
    hand-written .py macro module's ``__macro_namespace``) is read by
    name_key with the last matching key in insertion order winning, which
    is the entry copying it into a MacroTable (require's qualified and *
    forms) would keep."""
    if not isinstance(name, str) or table is None:
        return None
    if isinstance(table, MacroTable):
        return table.entry(name)
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


def register_required(scope, module, prefix="", names=None):
    """Register module's macros in scope's table, for the code a require
    generates (core_meta_functions.lpy): with names, exactly those
    (required_macro); else every one, its key prefixed (``"mod."`` for a
    qualified require, nothing for ``*``). The module's table is read from a
    snapshot, since running code may add to it. Living here, the work binds
    no global of the requiring module."""
    imported = getattr(module, "__macro_namespace", {})
    table = scope_table(scope)
    if names is None:
        for key, macro in list(imported.items()):
            table[prefix + key] = macro
    else:
        for name in names:
            table[name] = required_macro(imported, name)
