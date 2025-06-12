import ast

unaryop_dict = {"+": ast.UAdd, "-": ast.USub, "not": ast.Not, "~": ast.Invert}
binop_dict = {
    "+": ast.Add,
    "-": ast.Sub,
    "*": ast.Mult,
    "/": ast.Div,
    "//": ast.FloorDiv,
    "%": ast.Mod,
    "**": ast.Pow,
    "<<": ast.LShift,
    ">>": ast.RShift,
    "|": ast.BitOr,
    "^": ast.BitXor,
    "&": ast.BitAnd,
    "@": ast.MatMult,
}
augassignop_dict = {k + "=": v for [k, v] in binop_dict.items()}
boolop_dict = {"and": ast.And, "or": ast.Or}
compare_dict = {
    "==": ast.Eq,
    "!=": ast.NotEq,
    "<": ast.Lt,
    "<=": ast.LtE,
    ">": ast.Gt,
    ">=": ast.GtE,
    "is": ast.Is,
    "is-not": ast.IsNot,
    "in": ast.In,
    "not-in": ast.NotIn,
}
