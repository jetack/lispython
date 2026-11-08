WARNING = "This script should be edited in .lpy file, not translated .py file."
from lispy.core.builtins import __all__ as MACRO_SCOPE_NAMES
from lispy.core.compiler.literal import identifier_p, module_import_name, plain_name_p
from lispy.core.errors import LispyCompileError
from lispy.core.nodes import *
from lispy.core.nodes import atom_text


def builtin(name):
    """The generated code's form for the builtin NAME, read from a function's
    __builtins__, never by a global lookup: the user module can rebind a global
    name (or make it a macro), while a function's __builtins__ is the dict an
    import statement reads."""
    return Paren(
        Symbol("sub"),
        Paren(
            Symbol("."),
            Paren(Symbol("lambda"), Bracket(), Constant("0")),
            Symbol("__builtins__"),
        ),
        name,
    )


def macro_table():
    """The generated code's form for the module lispy.core.macro_table."""
    return Paren(
        builtin("__import__"),
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
    scopename = Symbol("___defmacro_scope___" + macroname)
    if not identifier_p(scopename.name):
        scopename = Symbol("___defmacro_scope___")
    tools = [Symbol(name) for name in MACRO_SCOPE_NAMES]
    return mark_compiler_forms(
        Paren(
            Symbol("do"),
            Paren(
                Symbol("def"),
                scopename,
                Bracket(),
                Paren(Symbol("from"), Symbol("lispy.core.builtins"), Bracket(*tools)),
                Paren(Symbol("def"), newname, *body),
                Paren(Symbol("return"), newname),
            ),
            Paren(
                Symbol("="),
                Paren(
                    Symbol("sub"),
                    Paren(
                        Symbol(".scope-table"), macro_table(), Paren(builtin("globals"))
                    ),
                    macroname,
                ),
                Paren(scopename),
            ),
            Paren(Symbol("del"), scopename),
        ),
        body,
    )


REQUIRE_QUALIFIED = "qualified"
REQUIRE_ALL = "all"
REQUIRE_NAMED = "named"


def require_error(node, problem):
    """The one shape of a malformed require's error, at node."""
    return LispyCompileError("(require ...) " + problem, node)


def symbol_text_p(node, text):
    """Whether node is the bare symbol spelled text."""
    return type(node) is Symbol and atom_text(node) == text


def require_qualifier(module):
    """The name a qualified (require module) registers every macro under (as
    qualifier.name): the name Python's import of module binds, its dotted
    name when absolute, its last part when relative (.mac and ..pkg.mac
    are mac, as from . import mac and from ..pkg import mac bind mac).  A
    relative module of dots alone binds no name: an error at it."""
    text = atom_text(module)
    if not text.startswith("."):
        return text
    last = text.lstrip(".").split(".")[-1]
    if not last:
        raise require_error(
            module,
            f"cannot qualify macros by {text}, which binds no name: use (require {text} as name), * or [names...]",
        )
    return last


def require_parse(sexp):
    """The one parse of (require module ...), read by the compiler and the LSP:
    (kind, qualifier, names, module).  REQUIRE-QUALIFIED for (require module) and
    (require module as alias): every macro, registered as qualifier.name,
    the qualifier being alias, else require-qualifier's.  REQUIRE-ALL for *:
    every macro, unqualified.  REQUIRE-NAMED for a list: exactly names, so
    an empty [] registers none.  The qualifier is None unless qualified,
    names None unless named.  module is the name the import takes: the
    relative level's dots, then the dotted path mangled as the import forms
    mangle theirs (module-import-name, which refuses a part that is no
    name: mac., a..b, m.-, ok?.if).  A malformed require is a LispyCompileError at
    the part at fault (require-error), as the import forms refuse theirs."""
    args = sexp.list[1:]
    if len(args) == 0:
        raise require_error(sexp, "expects a module name")
    module = args[0]
    if isinstance(module, GluedHead):
        raise module.misuse_error()
    if type(module) is not Symbol:
        raise require_error(module, f"expects a module name, got {module}")
    module_name = module_import_name(module)
    rest = args[1:]
    if len(rest) == 0:
        return (REQUIRE_QUALIFIED, require_qualifier(module), None, module_name)
    selection = rest[0]
    if symbol_text_p(selection, "as"):
        if len(rest) == 1:
            raise require_error(selection, "expects a name after as")
        alias = rest[1]
        if not plain_name_p(alias):
            raise require_error(alias, f"expects a name after as, got {alias}")
        if len(rest) > 2:
            raise require_error(
                rest[2],
                f"takes no selection after an alias, which qualifies every macro, got {rest[2]}",
            )
        return (REQUIRE_QUALIFIED, atom_text(alias), None, module_name)
    star = symbol_text_p(selection, "*")
    if not (star or isinstance(selection, Bracket)):
        raise require_error(
            selection,
            f"expects as alias, * or [names...] after the module, got {selection}",
        )
    if len(rest) > 1:
        raise require_error(
            rest[1],
            "takes no alias after a selection: an alias qualifies every macro"
            if symbol_text_p(rest[1], "as")
            else f"takes one selection after the module, got {rest[1]}",
        )
    if star:
        return (REQUIRE_ALL, None, None, module_name)
    for name in selection:
        if type(name) is not Symbol:
            raise require_error(name, f"expects a macro name, got {name}")
    return (REQUIRE_NAMED, None, [atom_text(name) for name in selection], module_name)


def require_transform(sexp):
    [kind, qualifier, names, module_name_str] = require_parse(sexp)
    is_relative = module_name_str.startswith(".")
    import_call = (
        Paren(
            Symbol(".import-module"),
            Paren(builtin("__import__"), String('"importlib"')),
            module_name_str,
            Keyword(Symbol("package")),
            Symbol("__package__"),
        )
        if is_relative
        else Paren(
            Symbol(".import-module"),
            Paren(builtin("__import__"), String('"importlib"')),
            module_name_str,
        )
    )
    prefix = qualifier + "." if kind == REQUIRE_QUALIFIED else ""
    selected = names if kind == REQUIRE_NAMED else Constant("None")
    return mark_compiler_forms(
        Paren(
            Symbol(".register-required"),
            macro_table(),
            Paren(builtin("globals")),
            import_call,
            prefix,
            selected,
        )
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