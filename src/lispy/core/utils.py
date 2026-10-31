import ast
from types import GeneratorType

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


def trampoline(steps):
    """Run ``steps``, a generator, to its return value on an explicit stack.

    A step asks for a sub-result with ``value = yield sub_steps``: a generator
    ``sub_steps`` (another such step) is run here and its return value sent
    back; anything else is already the result and is sent straight back, so a
    leaf (a name, a literal) costs no generator. Recursive in shape, iterative
    in Python: nesting depth costs no Python frame per level. An exception
    from a step is thrown into the step that asked for it, as a recursive call
    would raise it there (so a step's try/finally runs), and leaves here once
    no step catches it. The parser's f/t-string fields (parser.py), the
    compiler (compiler/expr.py), quote expansion and source rendering
    (nodes.py) and macro expansion (macro.py) run on it.
    """
    if not isinstance(steps, GeneratorType):
        return steps
    stack = [steps]
    value = None
    error = None
    while True:
        try:
            if error is None:
                sub = stack[-1].send(value)
            else:
                [thrown, error] = [error, None]
                sub = stack[-1].throw(thrown)
        except StopIteration as done:
            stack.pop()
            if not stack:
                return done.value
            value = done.value
            continue
        except BaseException as exc:
            stack.pop()
            if not stack:
                raise
            error = exc
            continue
        if isinstance(sub, GeneratorType):
            stack.append(sub)
            value = None
        else:
            value = sub
