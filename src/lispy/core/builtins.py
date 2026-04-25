from lispy.core.nodes import Expression, Symbol, Wrapper

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
    if isinstance(node, Symbol) and node.value == old.value:
        return new
    if isinstance(node, Expression):
        node.list = [replace_symbol(child, old, new) for child in node.list]
    elif isinstance(node, Wrapper):
        node.value = replace_symbol(node.value, old, new)
    return node
