from lispy.core.nodes import (
    Annotation,
    Bracket,
    DoubleStarred,
    Expression,
    Keyword,
    Paren,
    QuasiQuote,
    Quote,
    Starred,
    Symbol,
    Unquote,
    Wrapper,
)

_gensym_counter = 0


def gensym(prefix="gensym"):
    global _gensym_counter
    if isinstance(prefix, Symbol):
        prefix = prefix.name
    elif not isinstance(prefix, str):
        raise TypeError(f"gensym prefix must be a Symbol or string, got {type(prefix).__name__}")
    sym = Symbol("__" + prefix + "_" + str(_gensym_counter))
    _gensym_counter += 1
    return sym


def reset_gensym_counter():
    global _gensym_counter
    _gensym_counter = 0


def replace_symbol(node, old, new):
    """Rename the evaluated occurrences of symbol ``old`` in ``node`` to ``new``.

    Only positions that would read the variable are renamed (as-> relies on
    this): a Keyword (``:x``) is always a name, never a reference; a quoted
    form is data, and inside a quasiquote only ``~``/``~@`` is evaluated; a
    ``lambda``/``fn`` whose params bind the same name shadows it in its body,
    while its defaults and annotations are evaluated in the enclosing scope.
    Mutates ``node`` in place and returns it (or ``new`` for a bare match).
    """
    return _replace_symbol(node, old.value, new, False)


_LAMBDA_HEADS = ("lambda", "fn")


def _replace_symbol(node, name, new, in_quasi):
    if isinstance(node, Symbol):
        return new if not in_quasi and node.value == name else node
    if isinstance(node, Keyword):
        return node
    if isinstance(node, QuasiQuote):
        node.value = _replace_symbol(node.value, name, new, True)
    elif isinstance(node, Quote):
        pass
    elif isinstance(node, Unquote):
        node.value = _replace_symbol(node.value, name, new, False)
    elif isinstance(node, Expression):
        lst = node.list
        if (
            not in_quasi
            and isinstance(node, Paren)
            and len(lst) >= 2
            and isinstance(lst[0], Symbol)
            and lst[0].value in _LAMBDA_HEADS
            and isinstance(lst[1], Bracket)
        ):
            if _replace_in_params(lst[1], name, new):
                return node
            node.list = lst[:2] + [
                _replace_symbol(child, name, new, False) for child in lst[2:]
            ]
        else:
            node.list = [_replace_symbol(child, name, new, in_quasi) for child in lst]
    elif isinstance(node, Wrapper):
        node.value = _replace_symbol(node.value, name, new, in_quasi)
    return node


def _replace_in_params(params, name, new):
    """Rename inside a lambda param list's defaults and annotations (see
    def_args_parse for the syntax); return whether a param binds ``name``."""
    binds = False
    expect_default = False
    lst = params.list
    for i, p in enumerate(lst):
        if isinstance(p, Annotation):
            lst[i] = _replace_symbol(p, name, new, False)
        elif expect_default:
            lst[i] = _replace_symbol(p, name, new, False)
            expect_default = False
        elif isinstance(p, Keyword):
            binds = binds or (isinstance(p.value, Symbol) and p.value.value == name)
            expect_default = True
        elif isinstance(p, (Starred, DoubleStarred)):
            binds = binds or (isinstance(p.value, Symbol) and p.value.value == name)
        elif isinstance(p, Symbol):
            binds = binds or p.value == name
        else:
            lst[i] = _replace_symbol(p, name, new, False)
    return binds
