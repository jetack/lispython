from lispy.core.nodes import *


def defmacro_transform(sexp):
    [op, macroname, *body] = sexp.list
    macroname = str(macroname)
    newname = Symbol("___defmacro_temp___")
    return Paren(
        Symbol("do"),
        Paren(Symbol("from"), Symbol("lispy.core.nodes"), Symbol("*")),
        Paren(Symbol("def"), newname, *body),
        Paren(
            Symbol("="),
            Paren(
                Symbol("sub"),
                Paren(
                    Symbol(".setdefault"),
                    Paren(Symbol("globals")),
                    String('"__macro_namespace"'),
                    Brace(),
                ),
                macroname,
            ),
            newname,
        ),
        Paren(Symbol("del"), newname),
    )


def require_transform(sexp):
    [op, module_name, *optional] = sexp.list
    if optional:
        [macro_names] = optional
    else:
        macro_names = None
    return Paren(
        Symbol("do"),
        Paren(Symbol("import"), Symbol("importlib")),
        Paren(
            Symbol("="),
            Symbol("___imported-macros"),
            Paren(
                Symbol("getattr"),
                Paren(
                    Symbol("importlib.import-module"),
                    str(module_name).replace("-", "_"),
                ),
                String('"__macro_namespace"'),
                Brace(),
            ),
        ),
        Paren(
            Symbol("for"),
            Bracket(Symbol("k"), Symbol("v")),
            Symbol("in"),
            Paren(
                Symbol(".items"), Paren(Symbol(".copy"), Symbol("___imported-macros"))
            ),
            Paren(
                Symbol("="),
                Paren(
                    Symbol("sub"),
                    Paren(
                        Symbol(".setdefault"),
                        Paren(Symbol("globals")),
                        String('"__macro_namespace"'),
                        Brace(),
                    ),
                    Paren(Symbol("+"), str(module_name), String('"."'), Symbol("k")),
                ),
                Symbol("v"),
            ),
        )
        if not macro_names
        else Paren(
            Symbol(".update"),
            Paren(
                Symbol(".setdefault"),
                Paren(Symbol("globals")),
                String('"__macro_namespace"'),
                Brace(),
            ),
            Symbol("___imported-macros"),
        )
        if str(macro_names) == "*"
        else Paren(
            Symbol("for"),
            Symbol("mac-name"),
            Symbol("in"),
            [str(x) for x in macro_names],
            Paren(
                Symbol("="),
                Paren(
                    Symbol("sub"),
                    Paren(
                        Symbol(".setdefault"),
                        Paren(Symbol("globals")),
                        String('"__macro_namespace"'),
                        Brace(),
                    ),
                    Symbol("mac-name"),
                ),
                Paren(Symbol("sub"), Symbol("___imported-macros"), Symbol("mac-name")),
            ),
        ),
        Paren(Symbol("del"), Symbol("___imported-macros")),
    )
