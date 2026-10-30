import ast
import keyword
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
    return ast.BoolOp(
        boolop_dict[op.name](), list(map(expr_compile, args)), **sexp.position_info
    ) if len(args) >= 2 else expr_compile(args[0])


def compare_p(sexp):
    return atom_text(sexp.op) in compare_dict


def compare_compile(sexp):
    [op, *args] = sexp.list
    [left, *comparators] = map(expr_compile, args)
    return ast.Compare(
        left=left,
        ops=[compare_dict[atom_text(op)]() for i in range(len(comparators))],
        comparators=comparators,
        **sexp.position_info
    )


def call_args_parse(given):
    q = deque(given)
    args = []
    keywords = []
    while q:
        arg = q.popleft()
        keywords.append(
            ast.keyword(
                arg=arg.name, value=expr_compile(q.popleft()), **arg.position_info
            )
        ) if keyword_arg_p(arg) else keywords.append(
            ast.keyword(value=expr_compile(arg.value), **arg.position_info)
        ) if doublestarred_p(
            arg
        ) else args.append(
            expr_compile(arg)
        )
    return [args, keywords]


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
        if keyword.iskeyword(op.name):
            raise LispyCompileError(
                f"{head} is a Python keyword and cannot be called", sexp
            )
    op = expr_compile(op)
    [args, keywords] = call_args_parse(operands)
    return ast.Call(func=op, args=args, keywords=keywords, **sexp.position_info)


def ifexp_p(sexp):
    return atom_text(sexp.op) == "ife"


def ifexp_compile(sexp):
    [_, test, body, *rest] = sexp.list
    orelse = rest[0] if rest else Constant("None", **sexp.position_info)
    return ast.IfExp(
        test=expr_compile(test),
        body=expr_compile(body),
        orelse=expr_compile(orelse),
        **sexp.position_info
    )


def attribute_p(sexp):
    return atom_text(sexp.op) == "."


def attribute_compile(sexp, ctx):
    [_, value, *attrs] = sexp.list
    rst = expr_compile(value, ast.Load)
    position_info = {**sexp.position_info}
    for attr in attrs:
        position_info["end_lineno"] = attr.position_info["end_lineno"]
        position_info["end_col_offset"] = attr.position_info["end_col_offset"]
        rst = ast.Attribute(value=rst, attr=attr.name, ctx=ast.Load(), **position_info)
    rst.ctx = ctx()
    return rst


def methodcall_p(sexp):
    return (atom_text(sexp.op) or "").startswith(".")


def methodcall_compile(sexp):
    [method, instance, *operands] = sexp.list
    [args, keywords] = call_args_parse(operands)
    func = ast.Attribute(
        value=expr_compile(instance),
        attr=method.name[slice(1, None)],
        ctx=ast.Load(),
        **merge_position_infos(method.position_info, instance.position_info)
    )
    return ast.Call(func=func, args=args, keywords=keywords, **sexp.position_info)


def namedexpr_p(sexp):
    return atom_text(sexp.op) == ":="


def namedexpr_compile(sexp):
    [_, target, value] = sexp.list
    return ast.NamedExpr(
        target=expr_compile(target, ctx=ast.Store),
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
    [op, value, *slices] = sexp.list
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


def parse_comprehensions(generator_body):
    q = deque(generator_body)
    rst = []
    while q:
        is_async = 0 if q.popleft() == "for" else 1
        target = expr_compile(q.popleft(), ctx=ast.Store)
        if q[0] == "in":
            q.popleft()
        iter = expr_compile(q.popleft())
        comprehension = ast.comprehension(is_async=is_async, target=target, iter=iter)
        ifs = []
        while q and (atom_text(q[0]) not in ("for", "async-for")):
            ifs.append(q.popleft())
        comprehension.ifs = list(map(expr_compile, ifs[slice(1, None, 2)]))
        rst.append(comprehension)
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


def starred_head_p(arg):
    """Whether ``arg``'s text starts with ``*`` (a ``*args``/``**kwargs``
    parameter, or the bare ``*``), read without rendering it."""
    return isinstance(arg, (Starred, DoubleStarred)) or (
        atom_text(arg) or ""
    ).startswith("*")


def def_args_parse(sexp):
    q = deque(sexp.list)
    # ast.arguments has no _attributes: position keywords are a TypeError on 3.15+.
    rst = ast.arguments(posonlyargs=[])
    args = []
    defaults = []
    kwonlyargs = []
    kw_defaults = []

    # before starred
    while q and not starred_head_p(q[0]):
        arg = q.popleft()
        if arg == "/":
            rst.posonlyargs = args
            args = []
        else:
            ast_arg = ast.arg(arg=arg.name, **arg.position_info)
            if q and isinstance(q[0], Annotation):
                ast_arg.annotation = expr_compile(q.popleft())
            args.append(ast_arg)
            if keyword_arg_p(arg):
                defaults.append(expr_compile(q.popleft()))
    rst.args = args
    rst.defaults = defaults

    # starred
    if q and isinstance(q[0], Starred):
        arg = q.popleft()
        ast_arg = ast.arg(arg=arg.value.name, **arg.position_info)
        if q and isinstance(q[0], Annotation):
            ast_arg.annotation = expr_compile(q.popleft())
        rst.vararg = ast_arg
    if q and q[0] == "*":
        q.popleft()

    # before doublestarred
    while q and (not isinstance(q[0], DoubleStarred)):
        arg = q.popleft()
        ast_arg = ast.arg(arg=arg.name, **arg.position_info)
        if q and isinstance(q[0], Annotation):
            ast_arg.annotation = expr_compile(q.popleft())
        kwonlyargs.append(ast_arg)
        kw_defaults.append(expr_compile(q.popleft()) if keyword_arg_p(arg) else None)
    rst.kwonlyargs = kwonlyargs
    rst.kw_defaults = kw_defaults

    # doublestarred
    if q:
        arg = q.popleft()
        ast_arg = ast.arg(arg=arg.value.name, **arg.position_info)
        if q and isinstance(q[0], Annotation):
            ast_arg.annotation = expr_compile(q.popleft())
        rst.kwarg = ast_arg
    return rst


def lambda_p(sexp):
    return atom_text(sexp.op) == "lambda"


def lambda_compile(sexp):
    [_, args, body] = sexp.list
    return ast.Lambda(
        args=def_args_parse(args), body=expr_compile(body), **sexp.position_info
    )


def yield_compile(sexp):
    val_dict = {"value": expr_compile(sexp[1])} if len(sexp) > 1 else {}
    return ast.Yield(**val_dict, **sexp.position_info)


def yield_from_compile(sexp):
    value = sexp[1]
    return ast.YieldFrom(value=expr_compile(value), **sexp.position_info)


def await_compile(sexp):
    [_, value] = sexp.list
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
    assert 1 <= len(args) <= 3
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
    elts = reduce(
        lambda x, y: x
        + ([None, expr_compile(y.value)] if doublestarred_p(y) else [expr_compile(y)]),
        sexp,
        [],
    )
    keys = elts[slice(None, None, 2)]
    values = elts[slice(1, None, 2)]
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
    else:
        raise ValueError("'unquote' is not allowed here")


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
