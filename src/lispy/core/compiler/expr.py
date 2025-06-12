import ast
from collections import deque
from functools import reduce

from lispy.core.compiler.literal import *
from lispy.core.compiler.utils import *
from lispy.core.utils import *


def tuple_p(sexp):
    return str(sexp.op) == ","


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
    return str(sexp.op) in unaryop_dict and len(sexp) == 2


def unaryop_compile(sexp):
    [op, operand] = sexp.list
    return ast.UnaryOp(
        unaryop_dict[str(op)](), expr_compile(operand), **sexp.position_info
    )


def binop_p(sexp):
    return str(sexp.op) in binop_dict and len(sexp) > 2


def binop_compile(sexp):
    [op, *args] = sexp.list
    return reduce(
        lambda x, y: ast.BinOp(x, binop_dict[str(op)](), y, **sexp.position_info),
        map(expr_compile, args),
    )


def boolop_p(sexp):
    return str(sexp.op) in boolop_dict


def boolop_compile(sexp):
    [op, *args] = sexp.list
    return ast.BoolOp(
        boolop_dict[op.name](), list(map(expr_compile, args)), **sexp.position_info
    )


def compare_p(sexp):
    return str(sexp.op) in compare_dict


def compare_compile(sexp):
    [op, *args] = sexp.list
    [left, *comparators] = map(expr_compile, args)
    return ast.Compare(
        left=left,
        ops=[compare_dict[str(op)]() for i in range(len(comparators))],
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
    op = expr_compile(op)
    [args, keywords] = call_args_parse(operands)
    return ast.Call(func=op, args=args, keywords=keywords, **sexp.position_info)


def ifexp_p(sexp):
    return str(sexp.op) == "ife"


def ifexp_compile(sexp):
    [_, test, body, orelse] = sexp.list
    return ast.IfExp(
        test=expr_compile(test),
        body=expr_compile(body),
        orelse=expr_compile(orelse),
        **sexp.position_info
    )


def attribute_p(sexp):
    return str(sexp.op) == "."


def attribute_compile(sexp, ctx):
    [_, value, *attrs] = sexp.list
    rst = expr_compile(value, ast.Load)
    position_info = {**sexp.position_info}
    for attr in attrs:
        position_info["end_lineno"] = attr.position_info["end_lineno"]
        position_info["end_col_offset"] = attr.position_info["end_col_offset"]
        rst = ast.Attribute(value=rst, attr=str(attr), ctx=ast.Load(), **position_info)
    rst.ctx = ctx()
    return rst


def methodcall_p(sexp):
    return str(sexp.op).startswith(".")


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
    return str(sexp.op) == ":="


def namedexpr_compile(sexp):
    [_, target, value] = sexp.list
    return ast.NamedExpr(
        target=expr_compile(target, ctx=ast.Store),
        value=expr_compile(value),
        **sexp.position_info
    )


def subscript_p(sexp):
    return str(sexp.op) == "sub"


def subscript_compile(sexp, ctx):
    [op, value, *slices] = sexp.list
    op_pos = op.position_info
    rst = reduce(
        lambda rst, slice: ast.Subscript(
            value=rst,
            slice=expr_compile(slice),
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
        while q and (not str(q[0]) in ["for", "async-for"]):
            ifs.append(q.popleft())
        comprehension.ifs = list(map(expr_compile, ifs[slice(1, None, 2)]))
        rst.append(comprehension)
    return rst


def gen_exp_compile(sexp):
    [elt, *generator_body] = sexp
    return ast.GeneratorExp(
        elt=expr_compile(elt),
        generators=parse_comprehensions(generator_body),
        **sexp.position_info
    )


def def_args_parse(sexp):
    q = deque(sexp.list)
    rst = ast.arguments(posonlyargs=[], **sexp.position_info)
    args = []
    defaults = []
    kwonlyargs = []
    kw_defaults = []

    # before starred
    while q and (not str(q[0]).startswith("*")):
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
    return str(sexp.op) == "lambda"


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


def f_str_value_compile(sexp):
    format_spec_dict = (
        {}
        if sexp.format_spec is None
        else {
            "format_spec": ast.JoinedStr(
                values=[ast.Constant(value=sexp.format_spec, **sexp.position_info)],
                **sexp.position_info
            )
        }
    )
    return ast.FormattedValue(
        value=expr_compile(sexp.value),
        conversion=sexp.conversion,
        **format_spec_dict,
        **sexp.position_info
    )


def f_string_compile(sexp):
    values = sexp.operands
    compiled = []
    for [i, v] in enumerate(values):
        compiled.append(f_str_value_compile(v)) if i % 2 else compiled.append(
            string_compile(v)
        )
    return ast.JoinedStr(values=compiled, **sexp.position_info)


def paren_compiler(sexp, ctx):
    return (
        tuple_compile(sexp, ctx)
        if tuple_p(sexp)
        else unaryop_compile(sexp)
        if unaryop_p(sexp)
        else binop_compile(sexp)
        if binop_p(sexp)
        else boolop_compile(sexp)
        if boolop_p(sexp)
        else compare_compile(sexp)
        if compare_p(sexp)
        else ifexp_compile(sexp)
        if ifexp_p(sexp)
        else attribute_compile(sexp, ctx)
        if attribute_p(sexp)
        else methodcall_compile(sexp)
        if methodcall_p(sexp)
        else namedexpr_compile(sexp)
        if namedexpr_p(sexp)
        else subscript_compile(sexp, ctx)
        if subscript_p(sexp)
        else gen_exp_compile(sexp)
        if len(sexp) > 1 and str(sexp[1]) in ["for", "async-for"]
        else lambda_compile(sexp)
        if lambda_p(sexp)
        else yield_compile(sexp)
        if sexp.op == "yield"
        else yield_from_compile(sexp)
        if sexp.op == "yield-from"
        else await_compile(sexp)
        if sexp.op == "await"
        else f_string_compile(sexp)
        if sexp.op == "f-string"
        else call_compile(sexp)
    )


def slice_compile(sexp):
    [_, *args] = sexp.list
    args = deque(args)
    args_dict = {}
    if args:
        lower = args.popleft()
        if lower != "None" and lower != "_":
            args_dict["lower"] = expr_compile(lower)
    if args:
        upper = args.popleft()
        if upper != "None" and upper != "_":
            args_dict["upper"] = expr_compile(upper)
    if args:
        step = args.popleft()
        if step != "None" and step != "_":
            args_dict["step"] = expr_compile(step)
    return ast.Slice(**args_dict, **sexp.position_info)


def list_comp_compile(sexp):
    [elt, *generator_body] = sexp
    return ast.ListComp(
        elt=expr_compile(elt),
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
        if str(sexp.op) == ":"
        else list_comp_compile(sexp)
        if len(sexp) > 1 and str(sexp[1]) in ["for", "async-for"]
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


def set_comp_compile(sexp):
    [elt, *generator_body] = sexp.operands
    return ast.SetComp(
        elt=expr_compile(elt),
        generators=parse_comprehensions(generator_body),
        **sexp.position_info
    )


def brace_compiler(sexp):
    return (
        dict_compile(sexp)
        if len(sexp) < 1
        else (set_comp_compile if str(sexp.op) == "," else dict_comp_compile)(sexp)
        if len(sexp) > 2 and str(sexp[2]) in ["for", "async-for"]
        else set_compile(sexp)
        if str(sexp.op) == ","
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
