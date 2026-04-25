from lispy.core.nodes import Symbol

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
