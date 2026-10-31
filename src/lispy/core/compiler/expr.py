"""Expression forms to Python AST.

The compiler recurses in shape but not in Python frames, so nesting depth is
bounded by memory, never by the recursion limit. Every function that compiles
a sub-form is a generator (a "step", run by ``trampoline`` in core/utils.py)
that asks for each sub-result with ``node = yield <steps>``: ``expr_steps`` of
the sub-form, or a generator helper (``call_args_parse``, ``target_compile``,
``def_args_parse``, ...). A step never calls ``expr_compile``/``stmt_compile``
(a second trampoline, so a Python frame per level again) and never uses
``yield from`` on a sub-form (a frame per level inside the generators). The
dispatchers (``expr_steps``, ``paren_compiler``, ...) are plain functions that
pick the form's compiler and return its steps; a leaf (name, literal) is its
own node, which the trampoline sends straight back. ``yield`` is illegal in a
comprehension, so sub-forms are compiled in loops, in the order the recursive
compiler used: it decides which error wins when several sub-forms are bad.
``expr_compile`` is the plain entry point. stmt.py follows the same rule.
"""

import ast
from collections import deque
from functools import reduce

from lispy.core.compiler import features
from lispy.core.compiler.features import require
from lispy.core.compiler.literal import *
from lispy.core.compiler.utils import *
from lispy.core.errors import LispyCompileError
from lispy.core.nodes import atom_text
from lispy.core.utils import *


def tuple_p(sexp):
    return atom_text(sexp.op) == ","


def tuple_compile(sexp, ctx):
    [op, *args] = sexp.list
    elts = yield expr_list_steps(args, ctx)
    return ast.Tuple(elts=elts, ctx=ctx(), **sexp.position_info)


def starred_compile(sexp, ctx):
    value = yield expr_steps(sexp.value, ctx)
    return ast.Starred(value=value, ctx=ctx(), **sexp.position_info)


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
        unaryop_dict[atom_text(op)](),
        (yield expr_steps(operand)),
        **sexp.position_info
    )


def binop_p(sexp):
    return atom_text(sexp.op) in binop_dict and len(sexp) > 2


def binop_compile(sexp):
    [op, *args] = sexp.list
    operands = yield expr_list_steps(args)
    return reduce(
        lambda x, y: ast.BinOp(x, binop_dict[atom_text(op)](), y, **sexp.position_info),
        operands,
    )


def boolop_p(sexp):
    return atom_text(sexp.op) in boolop_dict


def boolop_compile(sexp):
    [op, *args] = sexp.list
    if not args:
        raise form_error(sexp, "needs at least one value")
    if len(args) < 2:
        return (yield expr_steps(args[0]))
    return ast.BoolOp(
        boolop_dict[atom_text(op)](),
        (yield expr_list_steps(args)),
        **sexp.position_info
    )


def compare_p(sexp):
    return atom_text(sexp.op) in compare_dict


def compare_compile(sexp):
    [op, *args] = sexp.list
    if len(args) < 2:
        raise form_error(sexp, "needs at least two operands")
    [left, *comparators] = yield expr_list_steps(args)
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
            name = symbol_name(arg.value)
            if name in seen:
                raise LispyCompileError(f"keyword argument repeated: {name}", arg)
            seen.add(name)
            keywords.append(
                ast.keyword(
                    arg=name,
                    value=(yield expr_steps(keyword_value(arg, q))),
                    **arg.position_info
                )
            )
        elif doublestarred_p(arg):
            keywords.append(
                ast.keyword(value=(yield expr_steps(arg.value)), **arg.position_info)
            )
        else:
            args.append((yield expr_steps(arg)))
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
    func = yield expr_steps(op)
    [args, keywords] = yield call_args_parse(operands)
    return ast.Call(func=func, args=args, keywords=keywords, **sexp.position_info)


def ifexp_p(sexp):
    return atom_text(sexp.op) == "ife"


def ifexp_compile(sexp):
    [[test, body], rest] = form_parts(sexp, ("test", "then"))
    if len(rest) > 1:
        raise form_error(sexp, "takes a test, a then and at most one else")
    orelse = rest[0] if len(rest) else Constant("None", **sexp.position_info)
    return ast.IfExp(
        test=(yield expr_steps(test)),
        body=(yield expr_steps(body)),
        orelse=(yield expr_steps(orelse)),
        **sexp.position_info
    )


def attribute_p(sexp):
    return atom_text(sexp.op) == "."


def attribute_compile(sexp, ctx):
    # Python has no attribute access without an attribute: (. x) is refused.
    [[value, first_attr], more_attrs] = form_parts(sexp, ("value", "attr"))
    attrs = [first_attr, *more_attrs]
    rst = yield expr_steps(value, ast.Load)
    position_info = {**sexp.position_info}
    for [i, attr] in enumerate(attrs):
        # One identifier per slot: (. a b.c) is refused, not read as a chain.
        name = symbol_name(attr)
        position_info["end_lineno"] = attr.position_info["end_lineno"]
        position_info["end_col_offset"] = attr.position_info["end_col_offset"]
        rst = ast.Attribute(value=rst, attr=name, ctx=ast.Load(), **position_info)
    rst.ctx = ctx()
    return rst


def methodcall_p(sexp):
    # A method head is a dot followed by a name: the Ellipsis heads ``...``
    # and ``..`` are not one.
    text = atom_text(sexp.op) or ""
    return text.startswith(".") and not text[1:].startswith(".") and len(text) > 1


def methodcall_compile(sexp):
    method = sexp.op
    [[instance], operands] = form_parts(sexp, ("instance",), "method call")
    [args, keywords] = yield call_args_parse(operands)
    func = ast.Attribute(
        value=(yield expr_steps(instance)),
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


def check_target(node, kind):
    """Refuse a compiled target ``node`` CPython cannot bind as ``kind``.

    One iterative walk, so a target nested past the recursion limit is
    checked like any other. The refusal is the one recursion would raise
    first: the first bad node in a left-to-right pre-order walk, and a tuple
    or list with two starred elements once all its elements are checked.
    """
    if kind == "walrus":
        if not isinstance(node, ast.Name):
            raise target_error(node, kind)
        return
    # (node, top): top is whether node is the whole target; None marks a
    # tuple or list whose elements were all checked.
    pending = [(node, True)]
    while pending:
        [node, top] = pending.pop()
        if top is None:
            starred = [elt for elt in node.elts if isinstance(elt, ast.Starred)]
            if len(starred) > 1:
                raise LispyCompileError(
                    "multiple starred expressions in assignment",
                    Node(**{k: getattr(starred[1], k) for k in Node().position_info}),
                )
        elif isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
            pass
        elif kind in ("assign", "delete") and isinstance(node, (ast.Tuple, ast.List)):
            pending.append((node, None))
            # Reversed, so the first element is popped (checked) first.
            pending.extend((elt, False) for elt in reversed(node.elts))
        elif kind == "assign" and isinstance(node, ast.Starred) and not top:
            pending.append((node.value, False))
        else:
            raise target_error(node, kind)


def target_compile(sexp, kind):
    """The one home of what may be bound or deleted, as CPython has it.

    ``kind`` is "assign" (=, for, with-as, a comprehension's for: names,
    attributes, subscripts and tuples/lists of them, one starred each),
    "annotate" and "augassign" (a single name, attribute or subscript),
    "delete" (no starred) or "walrus" (a single name). A bad target is
    refused at itself, with CPython's message, instead of an AST that
    compile() rejects.
    """
    if kind == "walrus":
        check_target((yield expr_steps(sexp)), kind)
        return (yield expr_steps(sexp, ast.Store))
    node = yield expr_steps(sexp, ast.Del if kind == "delete" else ast.Store)
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
        target=(yield target_compile(target, "walrus")),
        value=(yield expr_steps(value)),
        **sexp.position_info
    )


def subscript_p(sexp):
    return atom_text(sexp.op) == "sub"


def subscript_slice_compile(slice):
    """A lone ``*x`` slice is a one-element Tuple, as ``t[*x]`` parses."""
    node = yield expr_steps(slice)
    if isinstance(node, ast.Starred):
        return ast.Tuple(elts=[node], ctx=ast.Load(), **slice.position_info)
    return node


def subscript_compile(sexp, ctx):
    op = sexp.op
    [[value], slices] = form_parts(sexp, ("value",))
    op_pos = op.position_info
    rst = yield expr_steps(value)
    for slice in slices:
        rst = ast.Subscript(
            value=rst,
            slice=(yield subscript_slice_compile(slice)),
            ctx=ast.Load(),
            **merge_position_infos(op_pos, slice.position_info)
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
        target = yield target_compile(q.popleft(), "assign")
        if len(q) and atom_text(q[0]) == "in":
            q.popleft()
        if not q:
            raise form_error(
                keyword, f"has {head} without an iterable", "comprehension"
            )
        iter = yield expr_steps(q.popleft())
        ifs = []
        while len(q) and atom_text(q[0]) not in COMPREHENSION_FORS:
            word = q.popleft()
            if atom_text(word) != "if":
                raise form_error(
                    word, f"expects if or a for clause, got {word}", "comprehension"
                )
            if not q or atom_text(q[0]) in COMPREHENSION_FORS:
                raise form_error(word, "has if without a test", "comprehension")
            ifs.append((yield expr_steps(q.popleft())))
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
    return (yield expr_steps(elt))


def gen_exp_compile(sexp):
    [elt, *generator_body] = sexp
    return ast.GeneratorExp(
        elt=(yield comprehension_elt_compile(elt)),
        generators=(yield parse_comprehensions(generator_body)),
        **sexp.position_info
    )


def param_name(arg):
    """The name a parameter binds: the symbol itself, or the one inside its
    ``:name`` (default), ``*args`` or ``**kwargs`` wrapper."""
    if isinstance(arg, (Keyword, Starred, DoubleStarred)):
        arg = arg.value
    return symbol_name(arg)


def def_args_parse(sexp, annotated=True):
    """The ast.arguments of a def/async-def/lambda ``[args]`` bracket.

    One pass in Python's order: positional parameters (those before a ``/``
    positional-only), one ``*`` or ``*args``, keyword-only parameters, one
    ``**kwargs`` last. A parameter may be followed by its ``^annotation``;
    ``:name value`` gives it a default. What CPython refuses in a parameter
    list (a misplaced or repeated ``/``, ``*`` or ``**``, a
    bare ``*`` with no keyword-only parameter after it, a positional
    parameter without a default after one with a default) is refused at the
    offending item with CPython's message: read as it stands, such a list
    would compile to a different signature or an AST compile() rejects.
    Without ``annotated`` (a lambda) a ``^annotation`` is refused at itself:
    compile() accepts an annotated lambda parameter no Python source spells.
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

    def param(arg):
        name = param_name(arg)
        node = ast.arg(arg=name, **arg.position_info)
        if len(q) and isinstance(q[0], Annotation):
            if not annotated:
                raise form_error(
                    q[0], "takes no annotations on its parameters", "lambda"
                )
            node.annotation = yield expr_steps(q.popleft())
        return node

    def refuse_bare_star():
        if star is not None and vararg is None and not kwonlyargs:
            raise LispyCompileError("named parameters must follow bare *", star)

    while q:
        arg = q.popleft()
        text = atom_text(arg)
        if kwarg is not None:
            raise LispyCompileError("parameters cannot follow var-keyword parameter", arg)
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
                raise LispyCompileError("* may appear only once", arg)
            star = arg
            if isinstance(arg, Starred):
                vararg = yield param(arg)
        elif isinstance(arg, DoubleStarred):
            refuse_bare_star()
            kwarg = yield param(arg)
        elif star is None:
            node = yield param(arg)
            if keyword_arg_p(arg):
                defaults.append((yield expr_steps(keyword_value(arg, q))))
            elif len(defaults):
                # Positional-only and regular parameters share one defaults
                # list, the last ones': a gap would shift every default.
                raise LispyCompileError(
                    "parameter without a default follows parameter with a default",
                    arg,
                )
            args.append(node)
        else:
            kwonlyargs.append((yield param(arg)))
            kw_defaults.append(
                (yield expr_steps(keyword_value(arg, q)))
                if keyword_arg_p(arg)
                else None
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
        args=(yield def_args_parse(args, annotated=False)),
        body=(yield expr_steps(body)),
        **sexp.position_info
    )


def yield_compile(sexp):
    if len(sexp.operands) > 1:
        raise form_error(sexp, "takes at most one value")
    val_dict = {"value": (yield expr_steps(sexp[1]))} if len(sexp) > 1 else {}
    return ast.Yield(**val_dict, **sexp.position_info)


def yield_from_compile(sexp):
    [[value], rest] = form_parts(sexp, ("iterable",))
    if len(rest):
        raise form_error(sexp, "takes one iterable")
    return ast.YieldFrom(value=(yield expr_steps(value)), **sexp.position_info)


def await_compile(sexp):
    [[value], rest] = form_parts(sexp, ("value",))
    if len(rest):
        raise form_error(sexp, "takes one value")
    return ast.Await(value=(yield expr_steps(value)), **sexp.position_info)


def interpolated_values(sexp, field_compile):
    """Values of an f/t-string node: literals alternate with fields (FStrExpr).

    A ``{expr=}`` field echoes ``expr_text + debug`` at the end of the literal
    before it, built here so a field expanded from a quasiquote echoes its
    expanded text.  Empty literals are dropped, as in the AST CPython builds.
    ``field_compile`` is the steps of one field.
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
            values.append((yield field_compile(fields[k])))
    return values


def format_spec_dict(sexp):
    if sexp.format_spec is None:
        return {}
    return {"format_spec": (yield f_string_compile(sexp.format_spec))}


def f_str_value_compile(sexp):
    return ast.FormattedValue(
        value=(yield expr_steps(sexp.value)),
        conversion=sexp.conversion,
        **(yield format_spec_dict(sexp)),
        **sexp.position_info
    )


def f_string_compile(sexp):
    return ast.JoinedStr(
        values=(yield interpolated_values(sexp, f_str_value_compile)),
        **sexp.position_info
    )


def interpolation_compile(sexp):
    text = sexp.expr_text
    return ast.Interpolation(
        value=(yield expr_steps(sexp.value)),
        str=(
            text
            if features.interpolation_keeps_trailing_whitespace()
            else text.rstrip()
        ),
        conversion=sexp.conversion,
        **(yield format_spec_dict(sexp)),
        **sexp.position_info
    )


def t_string_compile(sexp):
    require("t-string", sexp)
    return ast.TemplateStr(
        values=(yield interpolated_values(sexp, interpolation_compile)),
        **sexp.position_info
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
# whether a () is call-shaped, i.e. a class pattern. Every compiler here is a
# step (a generator, see the module docstring).
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
    """The steps of a () expression."""
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
        args_dict["lower"] = yield expr_steps(lower)
    if upper != "None" and upper != "_":
        args_dict["upper"] = yield expr_steps(upper)
    if step != "None" and step != "_":
        args_dict["step"] = yield expr_steps(step)
    return ast.Slice(**args_dict, **sexp.position_info)


def list_comp_compile(sexp):
    [elt, *generator_body] = sexp
    return ast.ListComp(
        elt=(yield comprehension_elt_compile(elt)),
        generators=(yield parse_comprehensions(generator_body)),
        **sexp.position_info
    )


def list_compile(sexp, ctx):
    args = sexp.list
    elts = yield expr_list_steps(args, ctx)
    return ast.List(elts=elts, ctx=ctx(), **sexp.position_info)


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
    return ast.Set(elts=(yield expr_list_steps(args)), **sexp.position_info)


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
            values.append((yield expr_steps(item.value)))
            continue
        if not q or doublestarred_p(q[0]):
            raise form_error(item, "has a key without a value", "dict")
        keys.append((yield expr_steps(item)))
        values.append((yield expr_steps(q.popleft())))
    return ast.Dict(keys=keys, values=values, **sexp.position_info)


def dict_comp_compile(sexp):
    [key, value, *generator_body] = sexp
    return ast.DictComp(
        key=(yield expr_steps(key)),
        value=(yield expr_steps(value)),
        generators=(yield parse_comprehensions(generator_body)),
        **sexp.position_info
    )


def dict_unpack_comp_compile(sexp):
    """``{**d for d in dicts}`` (PEP 798): a DictComp with no value."""
    [unpacked, *generator_body] = sexp
    require("comprehension-unpacking", unpacked)
    return ast.DictComp(
        key=(yield expr_steps(unpacked.value)),
        generators=(yield parse_comprehensions(generator_body)),
        **sexp.position_info
    )


def set_comp_compile(sexp):
    [elt, *generator_body] = sexp.operands
    return ast.SetComp(
        elt=(yield comprehension_elt_compile(elt)),
        generators=(yield parse_comprehensions(generator_body)),
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
        return (
            yield expr_steps(
                sexp.value.generator_expression(isinstance(sexp, QuasiQuote))
            )
        )
    form = (
        "~@ (unquote-splice)" if isinstance(sexp, UnquoteSplice) else "~ (unquote)"
    )
    raise LispyCompileError(f"{form} is only allowed inside a quasiquote", sexp)


def expr_steps(sexp, ctx=ast.Load):
    """The steps compiling one expression: its form's generator, or the node
    itself for a leaf (see the module docstring)."""
    # An annotation compiles as its value, loaded; a chain of them in a loop.
    while isinstance(sexp, Annotation):
        [sexp, ctx] = [sexp.value, ast.Load]
    if isinstance(sexp, GluedHead):
        raise sexp.misuse_error()
    return (
        paren_compiler(sexp, ctx)
        if paren_p(sexp)
        else bracket_compiler(sexp, ctx)
        if bracket_p(sexp)
        else brace_compiler(sexp)
        if brace_p(sexp)
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


def expr_list_steps(sexps, ctx=ast.Load):
    """The nodes of ``sexps``, compiled in order."""
    nodes = []
    for sexp in sexps:
        nodes.append((yield expr_steps(sexp, ctx)))
    return nodes


def expr_compile(sexp, ctx=ast.Load):
    return trampoline(expr_steps(sexp, ctx))
