WARNING = "This script should be edited in .lpy file, not translated .py file."
from lispy.core.compiler.literal import identifier_p
from lispy.core.nodes import *


def macro_table():
    """The generated code's form for the module lispy.core.macro_table."""
    return Paren(
        Symbol("__import__"),
        String('"lispy.core.macro_table"'),
        Keyword(Symbol("fromlist")),
        Bracket(String('"MacroTable"')),
    )


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
            Symbol(".import-module"),
            Paren(Symbol("__import__"), String('"importlib"')),
            module_name_str,
            Keyword(Symbol("package")),
            Symbol("__package__"),
        )
        if is_relative
        else Paren(
            Symbol(".import-module"),
            Paren(Symbol("__import__"), String('"importlib"')),
            module_name_str,
        )
    )
    prefix = str(module_name) + "." if kind == REQUIRE_QUALIFIED else ""
    selected = names if kind == REQUIRE_NAMED else Constant("None")
    return Paren(
        Symbol(".register-required"),
        macro_table(),
        Paren(Symbol("globals")),
        import_call,
        prefix,
        selected,
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