WARNING = "This script should be edited in .lpy file, not translated .py file."
from lispy.core.compiler.literal import identifier_p
from lispy.core.nodes import *


def macro_table():
    """The generated code's form for the module lispy.core.macro_table: fresh
    nodes per use, since compiling may change a node in place."""
    return Paren(Symbol("importlib.import-module"), String('"lispy.core.macro_table"'))


def defmacro_transform(sexp):
    [op, macroname, *body] = sexp.list
    macroname = str(macroname)
    newname = Symbol("___defmacro_temp___" + macroname)
    if not identifier_p(newname.name):
        newname = Symbol("___defmacro_temp___")
    return Paren(
        Symbol("do"),
        Paren(Symbol("from"), Symbol("lispy.core.nodes"), Symbol("*")),
        Paren(Symbol("from"), Symbol("lispy.core.builtins"), Symbol("*")),
        Paren(Symbol("import"), Symbol("importlib")),
        Paren(Symbol("def"), newname, *body),
        Paren(
            Symbol("="),
            Paren(
                Symbol("sub"),
                Paren(Symbol(".scope-table"), macro_table(), Paren(Symbol("globals"))),
                macroname,
            ),
            newname,
        ),
        Paren(Symbol("del"), newname),
    )


REQUIRE_QUALIFIED = "qualified"
REQUIRE_ALL = "all"
REQUIRE_NAMED = "named"


def require_selection(optional):
    """Which macros a (require module ...) form registers, from the forms after
    the module name, as (kind, names).  Absent: every macro, qualified by the
    module name.  *: every macro, unqualified.  A list: exactly those names,
    so an empty [] registers none.  The compiler and the LSP both read this."""
    if not optional:
        return (REQUIRE_QUALIFIED, None)
    [macro_names] = optional
    if str(macro_names) == "*":
        return (REQUIRE_ALL, None)
    return (REQUIRE_NAMED, [str(x) for x in macro_names])


def require_transform(sexp):
    [op, module_name, *optional] = sexp.list
    [kind, names] = require_selection(optional)
    module_name_str = str(module_name).replace("-", "_")
    is_relative = module_name_str.startswith(".")
    import_call = (
        Paren(
            Symbol("importlib.import-module"),
            module_name_str,
            Keyword(Symbol("package")),
            Symbol("__package__"),
        )
        if is_relative
        else Paren(Symbol("importlib.import-module"), module_name_str)
    )
    return Paren(
        Symbol("do"),
        Paren(Symbol("import"), Symbol("importlib")),
        Paren(
            Symbol("="),
            Symbol("___imported-macros"),
            Paren(
                Symbol("getattr"), import_call, String('"__macro_namespace"'), Brace()
            ),
        ),
        Paren(
            Symbol("for"),
            Bracket(Symbol("___k"), Symbol("___v")),
            Symbol("in"),
            Paren(
                Symbol("list"), Paren(Symbol(".items"), Symbol("___imported-macros"))
            ),
            Paren(
                Symbol("="),
                Paren(
                    Symbol("sub"),
                    Paren(
                        Symbol(".scope-table"), macro_table(), Paren(Symbol("globals"))
                    ),
                    Paren(Symbol("+"), str(module_name), String('"."'), Symbol("___k")),
                ),
                Symbol("___v"),
            ),
        )
        if kind == REQUIRE_QUALIFIED
        else Paren(
            Symbol(".update"),
            Paren(Symbol(".scope-table"), macro_table(), Paren(Symbol("globals"))),
            Paren(
                Symbol("list"), Paren(Symbol(".items"), Symbol("___imported-macros"))
            ),
        )
        if kind == REQUIRE_ALL
        else Paren(
            Symbol("for"),
            Symbol("___mac-name"),
            Symbol("in"),
            names,
            Paren(
                Symbol("="),
                Paren(
                    Symbol("sub"),
                    Paren(
                        Symbol(".scope-table"), macro_table(), Paren(Symbol("globals"))
                    ),
                    Symbol("___mac-name"),
                ),
                Paren(
                    Symbol(".required-macro"),
                    macro_table(),
                    Symbol("___imported-macros"),
                    Symbol("___mac-name"),
                ),
            ),
        ),
        Paren(Symbol("del"), Symbol("___imported-macros")),
    )


if __name__ == "__main__":
    import os.path as osp

    from lispy.tools import l2py_s

    path = osp.abspath(__file__)
    with open(path, "r") as f:
        org = f.read()
    translated = l2py_s(org, no_lispy=True)
    with open(osp.join(osp.dirname(path), "core", "meta_functions.py"), "w") as f:
        f.write(translated)