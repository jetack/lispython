import ast
from collections import deque
from functools import reduce

from lispy.core.compiler import features
from lispy.core.compiler.features import require
from lispy.core.compiler.literal import *
from lispy.core.compiler.utils import *
from lispy.core.errors import LispyCompileError, nesting_error
from lispy.core.nodes import atom_text
from lispy.core.utils import *


def tuple_p(sexp):
    return atom_text(sexp.op) == ","


def tuple_compile(sexp, ctx):
    [op, *args] = sexp.list
    return ast.Tuple(
        elts=list(map(lambda x: expr_compile(x, ctx), args)),
        ctx=ctx(),
        **sexp.position_info
    )


def starred_compile(sexp, ctx):
    return ast.Starred(
        value=expr_compile(sexp.value, ctx), ctx=ctx(), **sexp.position_info
    )


def prefix_misuse(sexp):
    """A ``:name`` or ``**x`` reached expr_compile, so no call, def, class,
    dict or pattern consumed it: it is no value of its own."""
    raise LispyCompileError(
        f"{sexp} is only valid before a value in a call, def or class"
        if keyword_arg_p(sexp)
        else f"{sexp} is only valid in a call, a dict or a parameter list",
        sexp,
    )


def unaryop_p(sexp):
    return atom_text(sexp.op) in unaryop_dict and len(sexp) == 2


def unaryop_compile(sexp):
    [op, operand] = sexp.list
    return ast.UnaryOp(
        unaryop_dict[atom_text(op)](), expr_compile(operand), **sexp.position_info
    )


def binop_p(sexp):
    return atom_text(sexp.op) in binop_dict and len(sexp) > 2


def binop_compile(sexp):
    [op, *args] = sexp.list
    return reduce(
        lambda x, y: ast.BinOp(x, binop_dict[atom_text(op)](), y, **sexp.position_info),
        map(expr_compile, args),
    )


def boolop_p(sexp):
    return atom_text(sexp.op) in boolop_dict


def boolop_compile(sexp):
    [op, *args] = sexp.list
    if not args:
        raise form_error(sexp, "needs at least one value")
    return ast.BoolOp(
        boolop_dict[atom_text(op)](),
        list(map(expr_compile, args)),
        **sexp.position_info
    ) if len(args) >= 2 else expr_compile(args[0])


def compare_p(sexp):
    return atom_text(sexp.op) in compare_dict


def compare_compile(sexp):
    [op, *args] = sexp.list
    if len(args) < 2:
        raise form_error(sexp, "needs at least two operands")
    [left, *comparators] = map(expr_compile, args)
    return ast.Compare(
        left=left,
        ops=[compare_dict[atom_text(op)]() for i in range(len(comparators))],
        comparators=comparators,
        **sexp.position_info
    )


def keyword_value(keyword, q):
    """The value after ``keyword`` in ``q``, popped; a keyword with nothing
    after it is a LispyCompileError at the keyword."""
    if not q:
        raise LispyCompileError(f"{keyword} has no value after it", keyword)
    return q.popleft()


def call_args_parse(given):
    """``[args, keywords]`` of a call's (or a class's bases') operands:
    ``:name value`` and ``**mapping`` are keywords, the rest positional.
    A keyword name given twice is refused at its second ``:name``, with
    CPython's message, compared as the Python name (``:a-b`` is ``:a_b``)."""
    q = deque(given)
    args = []
    keywords = []
    seen = set()
    while q:
        arg = q.popleft()
        if keyword_arg_p(arg):
            name = symbol_name(arg.value, ast.Store)
            if name in seen:
                raise LispyCompileError(f"keyword argument repeated: {name}", arg)
            seen.add(name)
            keywords.append(
                ast.keyword(
                    arg=name,
                    value=expr_compile(keyword_value(arg, q)),
                    **arg.position_info
                )
            )
        elif doublestarred_p(arg):
            keywords.append(
                ast.keyword(value=expr_compile(arg.value), **arg.position_info)
            )
        else:
            args.append(expr_compile(arg))
    return [args, keywords]


# The callees CPython's compiler knows are not callable (its check_caller: a
# SyntaxWarning there, a TypeError at run time), by AST node class, named as
# its message names their type. A Constant is named by its value's type.
NOT_CALLABLE_TYPES = {
    "JoinedStr": "str",
    "TemplateStr": "Template",
    "Tuple": "tuple",
    "List": "list",
    "ListComp": "list",
    "Dict": "dict",
    "DictComp": "dict",
    "Set": "set",
    "SetComp": "set",
    "GeneratorExp": "generator",
}


def refuse_not_callable(func, head):
    """A call whose compiled callee ``func`` can never be called is refused at
    ``head``, in CPython's words."""
    kind = type(func).__name__
    name = (
        type(func.value).__name__
        if kind == "Constant"
        else NOT_CALLABLE_TYPES.get(kind)
    )
    if name is not None:
        raise LispyCompileError(f"'{name}' object is not callable", head)


def call_compile(sexp):
    [op, *operands] = sexp.list
    if isinstance(op, Symbol):
        head = atom_text(op)
        if head in STATEMENT_HEADS:
            hint = STATEMENT_HINTS.get(head)
            raise LispyCompileError(
                f"({head} ...) is a statement and cannot be used as an expression"
                + (f"; {hint}" if hint else ""),
                sexp,
            )
    func = expr_compile(op)
    refuse_not_callable(func, op)
    [args, keywords] = call_args_parse(operands)
    return ast.Call(func=func, args=args, keywords=keywords, **sexp.position_info)


def ifexp_p(sexp):
    return atom_text(sexp.op) == "ife"


def ifexp_compile(sexp):
    [[test, body], rest] = form_parts(sexp, ("test", "then"))
    if len(rest) > 1:
        raise form_error(sexp, "takes a test, a then and at most one else")
    orelse = rest[0] if len(rest) else Constant("None", **sexp.position_info)
    return ast.IfExp(
        test=expr_compile(test),
        body=expr_compile(body),
        orelse=expr_compile(orelse),
        **sexp.position_info
    )


def attribute_p(sexp):
    return atom_text(sexp.op) == "."


def attribute_compile(sexp, ctx):
    [[value], attrs] = form_parts(sexp, ("value",))
    if not attrs:
        # (. x) is x itself: its target context must reach the name home.
        return expr_compile(value, ctx)
    rst = expr_compile(value, ast.Load)
    position_info = {**sexp.position_info}
    for [i, attr] in enumerate(attrs):
        # One identifier per slot: (. a b.c) is refused, not read as a chain.
        name = symbol_name(attr, ctx if i == len(attrs) - 1 else ast.Load)
        position_info["end_lineno"] = attr.position_info["end_lineno"]
        position_info["end_col_offset"] = attr.position_info["end_col_offset"]
        rst = ast.Attribute(value=rst, attr=name, ctx=ast.Load(), **position_info)
    rst.ctx = ctx()
    return rst


def methodcall_p(sexp):
    return (atom_text(sexp.op) or "").startswith(".")


def methodcall_compile(sexp):
    method = sexp.op
    [[instance], operands] = form_parts(sexp, ("instance",), ".method")
    [args, keywords] = call_args_parse(operands)
    func = ast.Attribute(
        value=expr_compile(instance),
        attr=method_name(method),
        ctx=ast.Load(),
        **merge_position_infos(method.position_info, instance.position_info)
    )
    return ast.Call(func=func, args=args, keywords=keywords, **sexp.position_info)


# CPython's words for what cannot be a target ("cannot assign to literal").
TARGET_DESCRIPTIONS = {
    "Call": "function call",
    "Constant": "literal",
    "Compare": "comparison",
    "Lambda": "lambda",
    "IfExp": "conditional expression",
    "NamedExpr": "named expression",
    "Dict": "dict literal",
    "Set": "set display",
    "JoinedStr": "f-string expression",
    "ListComp": "list comprehension",
    "SetComp": "set comprehension",
    "DictComp": "dict comprehension",
    "GeneratorExp": "generator expression",
    "Await": "await expression",
    "Yield": "yield expression",
    "YieldFrom": "yield expression",
    "Tuple": "tuple",
    "List": "list",
    "Starred": "starred",
    "Attribute": "attribute",
    "Subscript": "subscript",
}


def target_error(node, kind):
    desc = TARGET_DESCRIPTIONS.get(type(node).__name__, "expression")
    msg = (
        f"cannot use assignment expressions with {desc}"
        if kind == "walrus"
        else f"cannot delete {desc}"
        if kind == "delete"
        else f"'{desc}' is an illegal expression for augmented assignment"
        if kind == "augassign"
        else f"only single target (not {desc}) can be annotated"
        if kind == "annotate" and desc in ("tuple", "list")
        else "illegal target for annotation"
        if kind == "annotate"
        else "starred assignment target must be in a list or tuple"
        if desc == "starred"
        else f"cannot assign to {desc}"
    )
    position = {k: getattr(node, k) for k in Node().position_info}
    return LispyCompileError(msg, Node(**position))


def check_target(node, kind, top=True):
    if kind == "walrus":
        if not isinstance(node, ast.Name):
            raise target_error(node, kind)
        return
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        return
    if kind in ("assign", "delete") and isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            check_target(elt, kind, top=False)
        starred = [elt for elt in node.elts if isinstance(elt, ast.Starred)]
        if len(starred) > 1:
            raise LispyCompileError(
                "multiple starred expressions in assignment",
                Node(**{k: getattr(starred[1], k) for k in Node().position_info}),
            )
        return
    if kind == "assign" and isinstance(node, ast.Starred) and not top:
        check_target(node.value, kind, top=False)
        return
    raise target_error(node, kind)


def target_compile(sexp, kind):
    """The one home of what may be bound or deleted, as CPython has it.

    ``kind`` is "assign" (=, for, with-as, a comprehension's for: names,
    attributes, subscripts and tuples/lists of them, one starred each),
    "annotate" and "augassign" (a single name, attribute or subscript),
    "delete" (no starred) or "walrus" (a single name). A bad target is
    refused at itself, with CPython's message, instead of an AST that
    compile() rejects. Each name in it comes from name_compile, which refuses
    binding __debug__.
    """
    if kind == "walrus":
        # Shape before name, as CPython: (:= a.__debug__ 1) is refused as an
        # attribute target, not as an assignment to __debug__.
        check_target(expr_compile(sexp), kind)
        return expr_compile(sexp, ast.Store)
    node = expr_compile(sexp, ast.Del if kind == "delete" else ast.Store)
    check_target(node, kind)
    return node


def namedexpr_p(sexp):
    return atom_text(sexp.op) == ":="


def namedexpr_compile(sexp):
    for x in sexp.operands:
        if isinstance(x, Annotation):
            raise form_error(x, "takes no annotation", ":=")
    [[target, value], rest] = form_parts(sexp, ("name", "value"))
    if len(rest):
        raise form_error(sexp, "takes one name and one value")
    return ast.NamedExpr(
        target=target_compile(target, "walrus"),
        value=expr_compile(value),
        **sexp.position_info
    )


def subscript_p(sexp):
    return atom_text(sexp.op) == "sub"


def subscript_slice_compile(slice):
    """A lone ``*x`` slice is a one-element Tuple, as ``t[*x]`` parses."""
    node = expr_compile(slice)
    if isinstance(node, ast.Starred):
        return ast.Tuple(elts=[node], ctx=ast.Load(), **slice.position_info)
    return node


def subscript_compile(sexp, ctx):
    op = sexp.op
    [[value], slices] = form_parts(sexp, ("value",))
    op_pos = op.position_info
    rst = reduce(
        lambda rst, slice: ast.Subscript(
            value=rst,
            slice=subscript_slice_compile(slice),
            ctx=ast.Load(),
            **merge_position_infos(op_pos, slice.position_info)
        ),
        slices,
        expr_compile(value),
    )
    rst.ctx = ctx()
    return rst


COMPREHENSION_FORS = ("for", "async-for")


def parse_comprehensions(generator_body):
    """The clauses after a comprehension's element: each ``for`` (or
    ``async-for``), a target, an optional ``in`` and an iterable, then any
    number of ``if test`` pairs. Anything else, or a clause cut short, is
    refused at the item or at the clause's keyword."""
    q = deque(generator_body)
    rst = []
    while q:
        keyword = q.popleft()
        head = atom_text(keyword)
        if head not in COMPREHENSION_FORS:
            raise form_error(
                keyword, f"expects a for clause, got {keyword}", "comprehension"
            )
        if not q:
            raise form_error(keyword, f"has {head} without a target", "comprehension")
        target = target_compile(q.popleft(), "assign")
        if len(q) and atom_text(q[0]) == "in":
            q.popleft()
        if not q:
            raise form_error(
                keyword, f"has {head} without an iterable", "comprehension"
            )
        iter = expr_compile(q.popleft())
        ifs = []
        while len(q) and atom_text(q[0]) not in COMPREHENSION_FORS:
            word = q.popleft()
            if atom_text(word) != "if":
                raise form_error(
                    word, f"expects if or a for clause, got {word}", "comprehension"
                )
            if not q or atom_text(q[0]) in COMPREHENSION_FORS:
                raise form_error(word, "has if without a test", "comprehension")
            ifs.append(expr_compile(q.popleft()))
        rst.append(
            ast.comprehension(
                target=target, iter=iter, ifs=ifs, is_async=int(head == "async-for")
            )
        )
    return rst


def comprehension_elt_compile(elt):
    """The element of a list/set/generator comprehension; ``*x`` is PEP 798."""
    if starred_p(elt):
        require("comprehension-unpacking", elt)
    return expr_compile(elt)


def gen_exp_compile(sexp):
    [elt, *generator_body] = sexp
    return ast.GeneratorExp(
        elt=comprehension_elt_compile(elt),
        generators=parse_comprehensions(generator_body),
        **sexp.position_info
    )


def param_name(arg):
    """The name a parameter binds: the symbol itself, or the one inside its
    ``:name`` (default), ``*args`` or ``**kwargs`` wrapper."""
    if isinstance(arg, (Keyword, Starred, DoubleStarred)):
        arg = arg.value
    return symbol_name(arg, ast.Store)


def def_args_parse(sexp):
    """The ast.arguments of a def/async-def/lambda ``[args]`` bracket.

    One pass in Python's order: positional parameters (those before a ``/``
    positional-only), one ``*`` or ``*args``, keyword-only parameters, one
    ``**kwargs`` last. A parameter may be followed by its ``^annotation``;
    ``:name value`` gives it a default. What CPython refuses in a parameter
    list (a duplicate name, a misplaced or repeated ``/``, ``*`` or ``**``, a
    bare ``*`` with no keyword-only parameter after it, a positional
    parameter without a default after one with a default) is refused at the
    offending item with CPython's message: read as it stands, such a list
    would compile to a different signature or an AST compile() rejects.
    """
    q = deque(sexp.list)
    posonlyargs = []
    args = []
    defaults = []
    kwonlyargs = []
    kw_defaults = []
    vararg = None
    kwarg = None
    slash = None
    star = None
    seen = set()

    def param(arg):
        name = param_name(arg)
        if name in seen:
            raise LispyCompileError(
                f"duplicate parameter '{name}' in function definition", arg
            )
        seen.add(name)
        node = ast.arg(arg=name, **arg.position_info)
        if len(q) and isinstance(q[0], Annotation):
            node.annotation = expr_compile(q.popleft())
        return node

    def refuse_bare_star():
        if star is not None and vararg is None and not kwonlyargs:
            raise LispyCompileError("named arguments must follow bare *", star)

    while q:
        arg = q.popleft()
        text = atom_text(arg)
        if kwarg is not None:
            raise LispyCompileError("arguments cannot follow var-keyword argument", arg)
        if text == "/":
            if slash is not None:
                raise LispyCompileError("/ may appear only once", arg)
            if star is not None:
                raise LispyCompileError("/ must be ahead of *", arg)
            if not args:
                raise LispyCompileError("at least one parameter must precede /", arg)
            slash = arg
            [posonlyargs, args] = [args, []]
        elif text == "*" or isinstance(arg, Starred):
            if star is not None:
                raise LispyCompileError("* argument may appear only once", arg)
            star = arg
            if isinstance(arg, Starred):
                vararg = param(arg)
        elif isinstance(arg, DoubleStarred):
            refuse_bare_star()
            kwarg = param(arg)
        elif star is None:
            node = param(arg)
            if keyword_arg_p(arg):
                defaults.append(expr_compile(keyword_value(arg, q)))
            elif len(defaults):
                # Positional-only and regular parameters share one defaults
                # list, the last ones': a gap would shift every default.
                raise LispyCompileError(
                    "parameter without a default follows parameter with a default",
                    arg,
                )
            args.append(node)
        else:
            kwonlyargs.append(param(arg))
            kw_defaults.append(
                expr_compile(keyword_value(arg, q)) if keyword_arg_p(arg) else None
            )
    refuse_bare_star()
    # ast.arguments has no _attributes: position keywords are a TypeError on 3.15+.
    return ast.arguments(
        posonlyargs=posonlyargs,
        args=args,
        vararg=vararg,
        kwonlyargs=kwonlyargs,
        kw_defaults=kw_defaults,
        kwarg=kwarg,
        defaults=defaults,
    )


def lambda_p(sexp):
    return atom_text(sexp.op) == "lambda"


def lambda_compile(sexp):
    [[args, body], rest] = form_parts(sexp, ("args", "body"))
    if not isinstance(args, Bracket):
        raise form_error(args, f"expects [args], got {args}", "lambda")
    if len(rest):
        raise form_error(sexp, "takes one body expression")
    return ast.Lambda(
        args=def_args_parse(args), body=expr_compile(body), **sexp.position_info
    )


def yield_compile(sexp):
    if len(sexp.operands) > 1:
        raise form_error(sexp, "takes at most one value")
    val_dict = {"value": expr_compile(sexp[1])} if len(sexp) > 1 else {}
    return ast.Yield(**val_dict, **sexp.position_info)


def yield_from_compile(sexp):
    [[value], rest] = form_parts(sexp, ("iterable",))
    if len(rest):
        raise form_error(sexp, "takes one iterable")
    return ast.YieldFrom(value=expr_compile(value), **sexp.position_info)


def await_compile(sexp):
    [[value], rest] = form_parts(sexp, ("value",))
    if len(rest):
        raise form_error(sexp, "takes one value")
    return ast.Await(value=expr_compile(value), **sexp.position_info)


def interpolated_values(sexp, field_compile):
    """Values of an f/t-string node: literals alternate with fields (FStrExpr).

    A ``{expr=}`` field echoes ``expr_text + debug`` at the end of the literal
    before it, built here so a field expanded from a quasiquote echoes its
    expanded text.  Empty literals are dropped, as in the AST CPython builds.
    """
    literals = [string_compile(v) for v in sexp.operands[0::2]]
    fields = sexp.operands[1::2]
    for [k, field] in enumerate(fields):
        if getattr(field, "debug", None) is not None:
            literals[k].value += field.expr_text + field.debug
    values = []
    for [k, literal] in enumerate(literals):
        if literal.value:
            values.append(literal)
        if k < len(fields):
            values.append(field_compile(fields[k]))
    return values


def format_spec_dict(sexp):
    return (
        {}
        if sexp.format_spec is None
        else {"format_spec": f_string_compile(sexp.format_spec)}
    )


def f_str_value_compile(sexp):
    return ast.FormattedValue(
        value=expr_compile(sexp.value),
        conversion=sexp.conversion,
        **format_spec_dict(sexp),
        **sexp.position_info
    )


def f_string_compile(sexp):
    return ast.JoinedStr(
        values=interpolated_values(sexp, f_str_value_compile), **sexp.position_info
    )


def interpolation_compile(sexp):
    text = sexp.expr_text
    return ast.Interpolation(
        value=expr_compile(sexp.value),
        str=(
            text
            if features.interpolation_keeps_trailing_whitespace()
            else text.rstrip()
        ),
        conversion=sexp.conversion,
        **format_spec_dict(sexp),
        **sexp.position_info
    )


def t_string_compile(sexp):
    require("t-string", sexp)
    return ast.TemplateStr(
        values=interpolated_values(sexp, interpolation_compile), **sexp.position_info
    )


def refuse_empty_form(sexp):
    """The one refusal of an empty (): an expression, a statement (through
    paren_compiler) or a match pattern (pattern_parse) reads its head."""
    if paren_p(sexp) and not len(sexp):
        raise LispyCompileError(
            "() is an empty form; use (,) for an empty tuple", sexp
        )


# Which compiler a non-empty () goes to: the first predicate that holds wins,
# and a () none claims is a call (call_compile). The one home of that dispatch:
# paren_compiler compiles through it, and pattern_parse asks call_form_p
# whether a () is call-shaped, i.e. a class pattern.
PAREN_FORMS = (
    (tuple_p, tuple_compile),
    (unaryop_p, unaryop_compile),
    (binop_p, binop_compile),
    (boolop_p, boolop_compile),
    (compare_p, compare_compile),
    (ifexp_p, ifexp_compile),
    (attribute_p, attribute_compile),
    (methodcall_p, methodcall_compile),
    (namedexpr_p, namedexpr_compile),
    (subscript_p, subscript_compile),
    (
        lambda sexp: len(sexp) > 1 and atom_text(sexp[1]) in ("for", "async-for"),
        gen_exp_compile,
    ),
    (lambda_p, lambda_compile),
    (lambda sexp: sexp.op == "yield", yield_compile),
    (lambda sexp: sexp.op == "yield-from", yield_from_compile),
    (lambda sexp: sexp.op == "await", await_compile),
    (lambda sexp: sexp.op == "f-string", f_string_compile),
    (lambda sexp: sexp.op == "t-string", t_string_compile),
)
# The PAREN_FORMS compilers that take the context (Load/Store/Del) too.
CTX_PAREN_FORMS = (tuple_compile, attribute_compile, subscript_compile)


def paren_form(sexp):
    """The compiler of a non-empty () (see PAREN_FORMS)."""
    return next((form for [p, form] in PAREN_FORMS if p(sexp)), call_compile)


def call_form_p(sexp):
    """Whether a non-empty () compiles as a call."""
    return paren_form(sexp) is call_compile


def paren_compiler(sexp, ctx):
    # Calls the compiler directly, adding no frame per nesting level: deeply
    # nested code is compiled by recursion.
    refuse_empty_form(sexp)
    form = paren_form(sexp)
    return form(sexp, ctx) if form in CTX_PAREN_FORMS else form(sexp)


def slice_compile(sexp):
    [_, *args] = sexp.list
    if not 1 <= len(args) <= 3:
        raise form_error(sexp, "takes one to three operands", "slice")
    [lower, upper, step] = args if len(args) == 3 else args + ["_"] if len(args) == 2 else ["_"] + args + ["_"]
    args_dict = {}
    if lower != "None" and lower != "_":
        args_dict["lower"] = expr_compile(lower)
    if upper != "None" and upper != "_":
        args_dict["upper"] = expr_compile(upper)
    if step != "None" and step != "_":
        args_dict["step"] = expr_compile(step)
    return ast.Slice(**args_dict, **sexp.position_info)


def list_comp_compile(sexp):
    [elt, *generator_body] = sexp
    return ast.ListComp(
        elt=comprehension_elt_compile(elt),
        generators=parse_comprehensions(generator_body),
        **sexp.position_info
    )


def list_compile(sexp, ctx):
    args = sexp.list
    return ast.List(
        elts=list(map(lambda x: expr_compile(x, ctx), args)),
        ctx=ctx(),
        **sexp.position_info
    )


def bracket_compiler(sexp, ctx):
    return (
        list_compile(sexp, ctx)
        if len(sexp) < 1
        else slice_compile(sexp)
        if atom_text(sexp.op) == ":"
        else list_comp_compile(sexp)
        if len(sexp) > 1 and atom_text(sexp[1]) in ("for", "async-for")
        else list_compile(sexp, ctx)
    )


def set_compile(sexp):
    [op, *args] = sexp.list
    return ast.Set(elts=list(map(expr_compile, args)), **sexp.position_info)


def dict_compile(sexp):
    """``{key value ... **mapping}``: a ``**mapping`` is a whole entry, so a
    key with no value before it or at the end is refused at the key."""
    q = deque(sexp)
    keys = []
    values = []
    while q:
        item = q.popleft()
        if doublestarred_p(item):
            keys.append(None)
            values.append(expr_compile(item.value))
            continue
        if not q or doublestarred_p(q[0]):
            raise form_error(item, "has a key without a value", "dict")
        keys.append(expr_compile(item))
        values.append(expr_compile(q.popleft()))
    return ast.Dict(keys=keys, values=values, **sexp.position_info)


def dict_comp_compile(sexp):
    [key, value, *generator_body] = sexp
    return ast.DictComp(
        key=expr_compile(key),
        value=expr_compile(value),
        generators=parse_comprehensions(generator_body),
        **sexp.position_info
    )


def dict_unpack_comp_compile(sexp):
    """``{**d for d in dicts}`` (PEP 798): a DictComp with no value."""
    [unpacked, *generator_body] = sexp
    require("comprehension-unpacking", unpacked)
    return ast.DictComp(
        key=expr_compile(unpacked.value),
        generators=parse_comprehensions(generator_body),
        **sexp.position_info
    )


def set_comp_compile(sexp):
    [elt, *generator_body] = sexp.operands
    return ast.SetComp(
        elt=comprehension_elt_compile(elt),
        generators=parse_comprehensions(generator_body),
        **sexp.position_info
    )


def brace_compiler(sexp):
    return (
        dict_compile(sexp)
        if len(sexp) < 1
        else (
            set_comp_compile if atom_text(sexp.op) == "," else dict_comp_compile
        )(sexp)
        if len(sexp) > 2 and atom_text(sexp[2]) in ("for", "async-for")
        else dict_unpack_comp_compile(sexp)
        if doublestarred_p(sexp[0]) and len(sexp) > 1
        and atom_text(sexp[1]) in ("for", "async-for")
        else set_compile(sexp)
        if atom_text(sexp.op) == ","
        else dict_compile(sexp)
    )


def metaindicator_p(sexp):
    return isinstance(sexp, MetaIndicator)


def metaindicator_compile(sexp):
    if isinstance(sexp, Quote):
        return expr_compile(
            sexp.value.generator_expression(isinstance(sexp, QuasiQuote))
        )
    form = (
        "~@ (unquote-splice)" if isinstance(sexp, UnquoteSplice) else "~ (unquote)"
    )
    raise LispyCompileError(f"{form} is only allowed inside a quasiquote", sexp)


def expr_compile(sexp, ctx=ast.Load):
    if isinstance(sexp, GluedHead):
        raise sexp.misuse_error()
    try:
        return (
            paren_compiler(sexp, ctx)
            if paren_p(sexp)
            else bracket_compiler(sexp, ctx)
            if bracket_p(sexp)
            else brace_compiler(sexp)
            if brace_p(sexp)
            else expr_compile(sexp.value)
            if isinstance(sexp, Annotation)
            else starred_compile(sexp, ctx)
            if starred_p(sexp)
            else constant_compile(sexp)
            if constant_p(sexp)
            else string_compile(sexp)
            if string_p(sexp)
            else metaindicator_compile(sexp)
            if metaindicator_p(sexp)
            else prefix_misuse(sexp)
            if keyword_arg_p(sexp) or doublestarred_p(sexp)
            else name_compile(sexp, ctx)
        )
    except RecursionError:
        # The compiler recurses, several frames per nesting level, so a form
        # nested past Python's recursion limit lands here. Building the error
        # this deep may hit the limit again; that RecursionError reaches the
        # next level's handler, which retries a level up, so the error is
        # positioned a few levels above the deepest form. A LispyCompileError
        # passes through every outer handler unchanged.
        raise nesting_error(sexp) from None
