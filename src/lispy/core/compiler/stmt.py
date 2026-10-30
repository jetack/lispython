import ast
from collections import deque
from functools import reduce

from lispy.core.compiler import features
from lispy.core.compiler.expr import (
    call_form_p,
    def_args_parse,
    expr_compile,
    refuse_empty_form,
)
from lispy.core.compiler.features import require
from lispy.core.compiler.literal import (
    SPECIAL_CONSTANTS,
    name_compile,
    number_parts,
    number_pattern_expr,
    reserved_name_p,
)
from lispy.core.compiler.utils import *
from lispy.core.errors import LispyCompileError, nesting_error
from lispy.core.nodes import GluedHead, Node, Paren, atom_text
from lispy.core.utils import *


def expr_wrapper(sexp):
    return ast.Expr(value=expr_compile(sexp), **sexp.position_info)


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
}


def target_error(node, kind):
    desc = TARGET_DESCRIPTIONS.get(type(node).__name__, "expression")
    msg = (
        f"cannot delete {desc}"
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

    ``kind`` is "assign" (=, for, with-as: names, attributes, subscripts and
    tuples/lists of them, one starred each), "annotate" and "augassign" (a
    single name, attribute or subscript) or "delete" (no starred). A bad
    target is refused at itself, with CPython's message, instead of an AST
    that compile() rejects.
    """
    node = expr_compile(sexp, ast.Del if kind == "delete" else ast.Store)
    check_target(node, kind)
    return node


def do_p(sexp):
    return atom_text(sexp.op) == "do"


def do_compile(sexp):
    [op, *sexps] = sexp.list
    return stmt_list_compile(sexps)


def assign_p(sexp):
    return atom_text(sexp.op) == "="


def assign_compile(sexp):
    [[target], body] = form_parts(sexp, ("target",))
    if not body:
        raise form_error(sexp, "is missing its value")
    for [i, x] in enumerate(sexp.operands):
        if isinstance(x, Annotation) and i != 1:
            raise form_error(
                x, "takes an annotation only right after one target", "="
            )
    if isinstance(body[0], Annotation):
        [annotation, *value] = body
        if len(value) > 1:
            raise form_error(sexp, "takes one value after an annotation")
        value_dict = {"value": expr_compile(value[0])} if value else {}
        return ast.AnnAssign(
            target=target_compile(target, "annotate"),
            annotation=expr_compile(annotation),
            simple=int(plain_name_p(target)),
            **value_dict,
            **sexp.position_info
        )
    else:
        [*targets, value] = [target, *body]
        return ast.Assign(
            targets=[target_compile(x, "assign") for x in targets],
            value=expr_compile(value),
            **sexp.position_info
        )


def augassign_p(sexp):
    return atom_text(sexp.op) in augassignop_dict


def augassign_compile(sexp):
    [[target, value], rest] = form_parts(sexp, ("target", "value"))
    args = [value, *rest]
    op = augassignop_dict[atom_text(sexp.op)]
    value = reduce(
        lambda x, y: ast.BinOp(x, op(), y, **sexp.position_info),
        map(expr_compile, args),
    )
    return ast.AugAssign(
        target=target_compile(target, "augassign"),
        op=op(),
        value=value,
        **sexp.position_info
    )


def del_compile(sexp):
    [op, *args] = sexp.list
    if not args:
        raise form_error(sexp, "needs at least one target")
    return ast.Delete(
        targets=[target_compile(x, "delete") for x in args],
        **sexp.position_info
    )


def import_name(sexp, form, dotted=True):
    """The one check on every name an import form reads: a bare symbol
    (dotted for a module path), as in Python, or a LispyCompileError at it."""
    if isinstance(sexp, GluedHead):
        raise sexp.misuse_error()
    if type(sexp) is not Symbol or (not dotted and "." in sexp.value):
        raise LispyCompileError(f"({form} ...) expects a name, got {sexp}", sexp)
    return sexp


def parse_names(names, form):
    rst = []
    q = deque([names] if names == "*" else names)
    while q:
        n = q.popleft()
        if n == "as" and rst and q:
            rst[-1].asname = import_name(q.popleft(), form, dotted=False).name
        else:
            n = import_name(n, form)
            rst.append(ast.alias(name=n.name, **n.position_info))
    return rst


def lazy_kwargs(sexp, lazy):
    if lazy:
        require(str(sexp.op), sexp)
        return {"is_lazy": 1}
    return {"is_lazy": 0} if features.import_has_is_lazy() else {}


def import_compile(sexp, lazy=False):
    [op, *names] = sexp.list
    is_lazy = lazy_kwargs(sexp, lazy)
    if not names:
        raise LispyCompileError(f"({op} ...) needs at least one module name", sexp)
    return ast.Import(
        names=parse_names(names, str(op)), **is_lazy, **sexp.position_info
    )


def importfrom_compile(sexp, lazy=False):
    [op, *args] = sexp.list
    form = str(op)
    is_lazy = lazy_kwargs(sexp, lazy)
    if not args or len(args) % 2:
        raise LispyCompileError(
            f"({form} ...) takes module and [names] pairs", sexp
        )
    modules = [import_name(module, form) for module in args[::2]]
    namess = args[1::2]
    for names in namess:
        if names != "*" and not isinstance(names, Bracket):
            raise LispyCompileError(
                f"({form} ...) expects [names] or * after a module, got {names}",
                names,
            )
    if lazy:
        # CPython's own static checks on `lazy from`, with its messages.
        for [module, names] in zip(modules, namess):
            if names == "*":
                raise LispyCompileError(
                    "lazy from ... import * is not allowed", sexp
                )
            if atom_text(module) == "__future__":
                raise LispyCompileError(
                    "lazy from __future__ import is not allowed", sexp
                )

    def helper(module):
        # Leading dots are the relative level; "." alone has no module name.
        x = module.name
        level = len(x) - len(x.lstrip("."))
        return [x[level:] or None, level, module.position_info]

    module_level = map(helper, modules)
    return [
        ast.ImportFrom(
            module=module,
            names=parse_names(names, form),
            level=level,
            **is_lazy,
            **merge_position_infos(module_pos_info, names.position_info)
        )
        for [[module, level, module_pos_info], names] in zip(module_level, namess)
    ]


def if_p(sexp):
    return atom_text(sexp.op) == "if"


def if_stmt_compile(sexp):
    [[test, then], orelse] = form_parts(sexp, ("test", "then"))
    return ast.If(
        test=expr_compile(test),
        body=stmt_list_compile([then]),
        orelse=stmt_list_compile(orelse),
        **sexp.position_info
    )


# How every statement form and clause is written: the one home of their arity,
# quoted by form_error in each refusal. Literal [ ] are lispython brackets.
FORM_USAGE = {
    "if": "(if test then else...)",
    "while": "(while test body...)",
    "for": "(for target in iterable body...)",
    "async-for": "(async-for target in iterable body...)",
    "def": "(def name [args] body...)",
    "async-def": "(async-def name [args] body...)",
    "class": "(class name [bases] body...)",
    "with": "(with [context as target ...] body...), each as target optional",
    "async-with": (
        "(async-with [context as target ...] body...), each as target optional"
    ),
    "try": "(try body... (except ...)... (else ...) (finally ...))",
    "except": (
        "(except [type ... as name] body...) or (except body...), as name optional"
    ),
    "except*": "(except* [type ... as name] body...), as name optional",
    "else": "(else body...)",
    "finally": "(finally body...)",
    "match": "(match subject (case ...)...)",
    "case": "(case pattern as name if guard body...), as name and if guard optional",
    "|": "(| pattern pattern...)",
    "pattern": (
        "a name, _, a literal, a dotted name, [sequence], {mapping}, "
        "(| ...) or (Class ...)"
    ),
    "sequence pattern": "[pattern... *rest pattern...], at most one *rest",
    "mapping pattern": (
        "{key pattern... **rest}, each key a literal or dotted name, **rest last"
    ),
    "class pattern": "(Class pattern... :attr pattern...), positional patterns first",
    "deco": "(deco decorator definition) or (deco [decorator ...] definition)",
    "raise": "(raise), (raise exception) or (raise exception from cause)",
    "assert": "(assert test) or (assert test message)",
    "=": (
        "(= target... value) or (= target ^annotation value), "
        "value optional after an annotation"
    ),
    **{op: f"({op} target value...)" for op in augassignop_dict},
    "del": "(del target...)",
    "global": "(global name...)",
    "nonlocal": "(nonlocal name...)",
}

# FORM_USAGE entries that are no head word: a message names them as they are
# ("class pattern has ..."), not as a form ("(| ...) needs ...").
PATTERN_KINDS = ("pattern", "sequence pattern", "mapping pattern", "class pattern")


def form_error(sexp, problem, form=None):
    """The one shape of a malformed-statement error: the problem, then usage.

    Positioned at ``sexp``; ``form`` (default: ``sexp``'s own head) names the
    statement or pattern kind whose usage is quoted, so an error at a clause
    or operand can still quote the enclosing form.
    """
    form = str(sexp.op) if form is None else form
    subject = form if form in PATTERN_KINDS else f"({form} ...)"
    return LispyCompileError(f"{subject} {problem}: {FORM_USAGE[form]}", sexp)


def form_parts(sexp, header, form=None):
    """``[parts, rest]``: the header operands of ``sexp`` in order, then the rest.

    ``header`` names the required operands; an entry ``(word, name)`` lets an
    optional ``word`` precede that operand (for's ``in``). A missing one is
    "is missing its {name}", at the form.
    """
    q = deque(sexp.operands)
    parts = []
    for entry in header:
        [word, name] = entry if isinstance(entry, tuple) else [None, entry]
        if word is not None and len(q) and q[0] == word:
            q.popleft()
        if not q:
            raise form_error(sexp, f"is missing its {name}", form)
        parts.append(q.popleft())
    return [parts, list(q)]


def clause_p(sexp, heads):
    """A Paren whose head is one of ``heads``; an empty () is no clause."""
    return isinstance(sexp, Paren) and len(sexp) > 0 and atom_text(sexp.op) in heads


def loop_parse(sexp, header):
    """Split a while/for form into its header operands, body and else body.

    ``header`` names the operands before the body, in order: ``("test",)``
    for while, ``("target", ("in", "iterable"))`` for for, whose optional
    ``in`` before the iterable is skipped. A trailing ``(else ...)`` holds the
    else body. Python has no bodiless loop, so an empty body is refused.
    """
    [parts, body] = form_parts(sexp, header)
    orelse = []
    if len(body) and clause_p(body[-1], ("else",)):
        orelse = body.pop().operands
    if not body:
        raise form_error(sexp, "needs a body")
    return parts, body, orelse


def while_p(sexp):
    return atom_text(sexp.op) == "while"


def while_compile(sexp):
    [[test], then, orelse] = loop_parse(sexp, ("test",))
    return ast.While(
        test=expr_compile(test),
        body=stmt_list_compile(then),
        orelse=stmt_list_compile(orelse),
        **sexp.position_info
    )


def for_p(sexp):
    return atom_text(sexp.op) == "for"


def for_compile(sexp, async_p=False):
    [[target, iterable], then, orelse] = loop_parse(
        sexp, ("target", ("in", "iterable"))
    )
    return (ast.AsyncFor if async_p else ast.For)(
        target=target_compile(target, "assign"),
        iter=expr_compile(iterable),
        body=stmt_list_compile(then),
        orelse=stmt_list_compile(orelse),
        **sexp.position_info
    )


def deco_p(sexp):
    return atom_text(sexp.op) == "deco"


def raise_compile(sexp):
    # Python's three shapes; a bare (raise) re-raises the active exception.
    body = sexp.operands
    if len(body) == 2 and body[1] == "from":
        raise form_error(sexp, "is missing its cause")
    if len(body) not in (0, 1, 3) or (len(body) == 3 and body[1] != "from"):
        raise form_error(sexp, "takes an exception and an optional from cause")
    kwargs = {"exc": expr_compile(body[0])} if len(body) else {}
    if len(body) == 3:
        kwargs["cause"] = expr_compile(body[2])
    return ast.Raise(**kwargs, **sexp.position_info)


def assert_compile(sexp):
    [[test], rest] = form_parts(sexp, ("test",))
    if len(rest) > 1:
        raise form_error(sexp, "takes a test and at most one message")
    kwargs = {"test": expr_compile(test)}
    if len(rest):
        kwargs["msg"] = expr_compile(rest[0])
    return ast.Assert(**kwargs, **sexp.position_info)


def except_types_parse(bracket, form):
    """``[types, name]`` of an except clause's ``[type ... as name]``.

    ``as`` may only come second to last, after at least one type, and be
    followed by a plain name; anything else is refused at the ``as``.
    """
    lst = bracket.list
    for [i, x] in enumerate(lst):
        if x != "as":
            continue
        if i == 0:
            raise form_error(x, "has as without a type before it", form)
        if i == len(lst) - 1:
            raise form_error(x, "has as without a name after it", form)
        if i != len(lst) - 2 or not plain_name_p(lst[-1]):
            raise form_error(x, "expects one name after as", form)
        return [lst[:i], lst[-1].name]
    if not len(lst):
        raise form_error(bracket, "has an empty [types] bracket", form)
    return [lst, None]


def parse_except(handler):
    """One except/except* clause: an optional [type ... as name], then a body."""
    form = str(handler.op)
    body = deque(handler.operands)
    kwargs = {}
    if len(body) and isinstance(body[0], Bracket):
        bracket = body.popleft()
        [types, name] = except_types_parse(bracket, form)
        kwargs["type"] = (
            ast.Tuple(
                elts=list(map(expr_compile, types)),
                ctx=ast.Load(),
                **bracket.position_info
            )
            if len(types) > 1
            else expr_compile(types[0])
        )
        if name is not None:
            kwargs["name"] = name
    elif form == "except*":
        raise form_error(handler, "needs a [types] bracket")
    if not body:
        raise form_error(handler, "needs a body")
    return ast.ExceptHandler(
        body=stmt_list_compile(body), **kwargs, **handler.position_info
    )


TRY_CLAUSES = ("except", "except*", "else", "finally")


def try_parse(sexp):
    """``[body, handlers, orelse, finalbody, star]`` of a try form.

    The one check of try's layout, as Python's grammar has it: a body, then
    except clauses or except* clauses (never both, a bare except last), an
    optional else (only after an except), an optional finally, and at least
    one except or a finally. A form out of that order is refused where it
    stands, never left in the body to compile as a call to a keyword.
    """
    operands = sexp.operands
    n = 0
    while n < len(operands) and not clause_p(operands[n], TRY_CLAUSES):
        n += 1
    body = operands[:n]
    if not body:
        raise form_error(sexp, "needs a body")
    [handlers, orelse, finalbody] = [[], [], None]
    [previous, star, bare] = [None, False, False]
    for clause in operands[n:]:
        if not clause_p(clause, TRY_CLAUSES):
            raise form_error(clause, f"has {clause} after its clauses", "try")
        kind = atom_text(clause.op)
        if kind in ("except", "except*"):
            if previous in ("else", "finally"):
                raise form_error(clause, f"has {kind} after {previous}", "try")
            if previous is not None and previous != kind:
                raise form_error(clause, "cannot mix except and except*", "try")
            if bare:
                raise form_error(clause, "has an except after a bare except", "try")
            star = kind == "except*"
            handlers.append(parse_except(clause))
            bare = handlers[-1].type is None
        else:
            if previous == kind:
                raise form_error(clause, f"has more than one {kind}", "try")
            if previous == "finally":
                raise form_error(clause, f"has {kind} after finally", "try")
            if kind == "else" and not handlers:
                raise form_error(clause, "has an else without an except", "try")
            if not len(clause.operands):
                raise form_error(clause, "needs a body")
            if kind == "else":
                orelse = clause.operands
            else:
                finalbody = clause.operands
        previous = kind
    if not handlers and finalbody is None:
        raise form_error(sexp, "needs an except or a finally clause")
    return [body, handlers, orelse, finalbody or [], star]


def try_compile(sexp):
    [body, handlers, orelse, finalbody, star] = try_parse(sexp)
    return (ast.TryStar if star else ast.Try)(
        body=stmt_list_compile(body),
        handlers=handlers,
        orelse=stmt_list_compile(orelse),
        finalbody=stmt_list_compile(finalbody),
        **sexp.position_info
    )


def with_items_parse(sexp, form):
    """The withitems of ``[context as target ...]``, ``as target`` optional."""
    rst = []
    targeted = True
    q = deque(sexp.list)
    while q:
        elt = q.popleft()
        if elt == "as":
            if targeted:
                raise form_error(elt, "has as without a context before it", form)
            if not q:
                raise form_error(elt, "has as without a target after it", form)
            rst[-1].optional_vars = target_compile(q.popleft(), "assign")
            targeted = True
        else:
            rst.append(ast.withitem(context_expr=expr_compile(elt)))
            targeted = False
    return rst


def with_compile(sexp, async_p=False):
    form = str(sexp.op)
    [[items], body] = form_parts(sexp, ("items",))
    if not isinstance(items, Bracket) or not len(items):
        raise form_error(
            items, f"expects a non-empty [context ...], got {items}", form
        )
    if not body:
        raise form_error(sexp, "needs a body")
    return (ast.AsyncWith if async_p else ast.With)(
        items=with_items_parse(items, form),
        body=stmt_list_compile(body),
        **sexp.position_info
    )


def signed_number_p(sexp):
    return (
        isinstance(sexp, Paren)
        and len(sexp) == 2
        and atom_text(sexp.op) in ("+", "-")
    )


def literal_pattern_expr(sexp):
    """The expression of a literal pattern (MatchValue value or mapping key).

    A signed number is the one Paren a literal pattern may be; numbers go
    through number_pattern_expr, which builds ast.parse's shapes. The one
    refusal of a sign before anything but a number (``(- a)``, ``(- 1+2j)``,
    ``--1``). ``+`` is a sign as in Python 3.15, gated by number_pattern_expr.
    """
    expr = None
    if signed_number_p(sexp):
        [sign, number] = sexp.list
        if (
            constant_p(number)
            and atom_text(number) not in SPECIAL_CONSTANTS
            and (number_parts(number.value) or [None] * 4)[2] is None
        ):
            expr = number_pattern_expr(str(sign) + number.value, sexp)
    elif constant_p(sexp) and atom_text(sexp) not in SPECIAL_CONSTANTS:
        expr = number_pattern_expr(sexp.value, sexp)
    else:
        return expr_compile(sexp)
    if expr is None:
        raise form_error(sexp, f"can sign only a number literal, got {sexp}", "pattern")
    return expr


def dotted_name_p(sexp):
    return type(sexp) is Symbol and "." in sexp.value


def name_or_attr_p(sexp):
    """Python's name_or_attr: every dot-part of the mangled name is an
    identifier, so ``.x``, ``a..b`` and ``x.`` are no value pattern, mapping
    key or class name. Expressions do not check this yet (TASK-757)."""
    return type(sexp) is Symbol and all(
        part.isidentifier() for part in sexp.name.split(".")
    )


def literal_pattern_p(sexp):
    """A number (signed or complex too), string, bytes, None, True or False."""
    return (
        signed_number_p(sexp)
        or string_p(sexp)
        or (constant_p(sexp) and atom_text(sexp) != "...")
    )


def value_pattern_expr(sexp, kind):
    """The expression of a value pattern or mapping key: a literal or a dotted
    name (``a.b``, which Python reads as a value, never a capture). Anything
    else is refused at ``sexp`` as no ``kind``."""
    if dotted_name_p(sexp) and name_or_attr_p(sexp):
        return name_compile(sexp, ast.Load)
    if literal_pattern_p(sexp):
        return literal_pattern_expr(sexp)
    if kind == "mapping pattern":
        raise form_error(
            sexp, f"expects a literal or dotted name as a key, got {sexp}", kind
        )
    raise form_error(sexp, f"expected, got {sexp}", kind)


def prefixed_name_p(sexp):
    """The name after a pattern's *, ** or : is an identifier that is no
    keyword once mangled.

    A stopgap for the reader bug TASK-756: parser.py's star_token_parse and
    keyword_token_parse pass the outer token type down, so the operand of
    ``*1``, ``**1a`` or ``:1a`` reads as a Symbol. Remove this check (back to
    plain_name_p) when the reader is fixed.
    """
    return (
        type(sexp) is Symbol
        and sexp.name.isidentifier()
        and not reserved_name_p(sexp.name)
    )


def match_sequence_parse(sexp):
    """``[pattern... *rest pattern...]``: the one place a *rest is a pattern."""
    patterns = []
    starred = False
    for elt in sexp.list:
        if not starred_p(elt):
            patterns.append(pattern_parse(elt))
            continue
        if starred:
            raise form_error(
                elt, f"has more than one *rest, got {elt}", "sequence pattern"
            )
        starred = True
        name = elt.value
        if atom_text(name) != "_" and not prefixed_name_p(name):
            raise form_error(
                elt, f"expects a name after *, got {name}", "sequence pattern"
            )
        patterns.append(
            ast.MatchStar(
                **{} if atom_text(name) == "_" else {"name": name.name}, **elt.position_info
            )
        )
    return {"patterns": patterns}


def match_mapping_parse(sexp):
    """``{key pattern... **rest}``: **rest once, last, a plain name."""
    items = list(sexp.list)
    rst = {}
    if items and doublestarred_p(items[-1]):
        rest = items.pop()
        if not prefixed_name_p(rest.value) or atom_text(rest.value) == "_":
            raise form_error(
                rest, f"expects a name after **, got {rest.value}", "mapping pattern"
            )
        rst["rest"] = rest.value.name
    for item in items:
        if doublestarred_p(item):
            raise form_error(item, f"has {item} before its end", "mapping pattern")
    if len(items) % 2:
        raise form_error(
            items[-1],
            f"has key {items[-1]} without a pattern after it",
            "mapping pattern",
        )
    rst["keys"] = [value_pattern_expr(key, "mapping pattern") for key in items[::2]]
    rst["patterns"] = list(map(pattern_parse, items[1::2]))
    return rst


def match_class_parse(sexp):
    """``(Class pattern... :attr pattern...)``, positional patterns first."""
    cls = sexp.op
    if not name_or_attr_p(cls):
        raise form_error(cls, f"needs a class name, got {cls}", "class pattern")
    q = deque(sexp.operands)
    patterns = []
    kwd_attrs = []
    kwd_patterns = []
    while q:
        arg = q.popleft()
        if keyword_arg_p(arg):
            if not q or keyword_arg_p(q[0]):
                raise form_error(
                    arg, f"has {arg} without a pattern after it", "class pattern"
                )
            if not prefixed_name_p(arg.value):
                raise form_error(
                    arg,
                    f"expects an attribute name after :, got {arg.value}",
                    "class pattern",
                )
            kwd_attrs.append(arg.name)
            kwd_patterns.append(pattern_parse(q.popleft()))
        elif kwd_attrs:
            raise form_error(arg, f"has {arg} after a keyword pattern", "class pattern")
        else:
            patterns.append(pattern_parse(arg))
    return {
        "cls": name_compile(cls, ast.Load),
        "patterns": patterns,
        "kwd_attrs": kwd_attrs,
        "kwd_patterns": kwd_patterns,
    }


def pattern_parse(sexp):
    """One case pattern, refused at the sub-form at fault when malformed.

    A ``()`` is a class pattern exactly when it would compile as a call
    (call_form_p), so (+ 1 2), (, a b) or f"x" is no pattern.
    """
    refuse_empty_form(sexp)
    position_info = sexp.position_info
    if starred_p(sexp):
        raise form_error(
            sexp, f"can be *rest only in a sequence pattern, got {sexp}", "pattern"
        )
    if doublestarred_p(sexp):
        raise form_error(
            sexp, f"can be **rest only last in a mapping pattern, got {sexp}", "pattern"
        )
    if type(sexp) is Symbol and atom_text(sexp) == "_":
        return ast.MatchAs(**position_info)
    if type(sexp) is Symbol and not dotted_name_p(sexp):
        name_compile(sexp, ast.Store)  # refuses a hard keyword (if) at itself
        return ast.MatchAs(name=sexp.name, **position_info)
    if atom_text(sexp) in ("True", "False", "None") and constant_p(sexp):
        return ast.MatchSingleton(value=eval(str(sexp)), **position_info)
    if bracket_p(sexp):
        return ast.MatchSequence(**match_sequence_parse(sexp), **position_info)
    if brace_p(sexp):
        return ast.MatchMapping(**match_mapping_parse(sexp), **position_info)
    if paren_p(sexp) and sexp.op == "|":
        if len(sexp.operands) < 2:
            raise form_error(sexp, "needs at least two patterns", "|")
        return ast.MatchOr(
            patterns=list(map(pattern_parse, sexp.operands)), **position_info
        )
    if paren_p(sexp) and not signed_number_p(sexp):
        if not call_form_p(sexp):
            raise form_error(sexp, f"expected, got {sexp}", "pattern")
        return ast.MatchClass(**match_class_parse(sexp), **position_info)
    return ast.MatchValue(value=value_pattern_expr(sexp, "pattern"), **position_info)


def case_parse(case):
    """``(case pattern [as name] [if guard] body...)``; the case is checked
    in match_compile, the parts here, each refusal at the case."""
    [[pattern_expr], body] = form_parts(case, ("pattern",))
    pattern = pattern_parse(pattern_expr)
    if len(body) and body[0] == "as":
        if len(body) < 2:
            raise form_error(case, "has as without a name after it")
        if not plain_name_p(body[1]):
            raise form_error(body[1], f"expects a name after as, got {body[1]}", "case")
        pattern = ast.MatchAs(
            pattern=pattern,
            name=body[1].name.replace("-", "_"),
            **merge_position_infos(pattern_expr.position_info, body[1].position_info)
        )
        body = body[2:]
    guard_dict = {}
    if len(body) and body[0] == "if":
        if len(body) < 2:
            raise form_error(case, "has if without a guard after it")
        guard_dict = {"guard": expr_compile(body[1])}
        body = body[2:]
        if len(body) and body[0] == "as":
            raise form_error(
                body[0], "has as after its guard; as name comes before if", "case"
            )
    if not body:
        raise form_error(case, "needs a body")
    return ast.match_case(pattern=pattern, **guard_dict, body=stmt_list_compile(body))


def match_compile(sexp):
    [[subject], cases] = form_parts(sexp, ("subject",))
    if not cases:
        raise form_error(sexp, "needs at least one case")
    for case in cases:
        if not clause_p(case, ("case",)):
            raise form_error(case, f"expects (case ...) clauses, got {case}", "match")
    cases = list(map(case_parse, cases))
    return ast.Match(subject=expr_compile(subject), cases=cases, **sexp.position_info)


def type_params_parse(glued):
    """The bracket of ``f[T ...]``, read with the def-argument markers.

    ``T`` TypeVar, ``T ^bound`` (a tuple bound is a constraint list),
    ``*Ts`` TypeVarTuple, ``**P`` ParamSpec; ``:T default`` (3.13) gives any
    of them a default, after its bound.
    """
    require("type-params", glued)
    q = deque(glued.params.list)
    if not q:
        raise LispyCompileError("Type parameter list cannot be empty", glued)
    rst = []
    while q:
        param = q.popleft()
        has_default = isinstance(param, Keyword)
        target = param.value if has_default else param
        last = target
        # Class names, resolved only after the gates: 3.11's ast has none.
        [kind, symbol] = (
            ["TypeVarTuple", target.value]
            if isinstance(target, Starred)
            else ["ParamSpec", target.value]
            if isinstance(target, DoubleStarred)
            else ["TypeVar", target]
        )
        if type(symbol) is not Symbol or "." in symbol.value:
            raise LispyCompileError(f"invalid type parameter {param}", param)
        fields = {"name": symbol.name}
        if q and isinstance(q[0], Annotation):
            bound = q.popleft()
            if kind != "TypeVar":
                raise LispyCompileError(f"cannot use bound with {kind}", bound)
            fields["bound"] = expr_compile(bound.value)
            last = bound
        if has_default:
            require("type-param-default", param)
            if not q:
                raise LispyCompileError(
                    f"type parameter {symbol.value} needs a default value", param
                )
            last = q.popleft()
            fields["default_value"] = expr_compile(last)
        # As in ast.parse, the node spans its name through any bound and default.
        span = {
            **target.position_info,
            "end_lineno": last.end_lineno,
            "end_col_offset": last.end_col_offset,
        }
        rst.append(getattr(ast, kind)(**fields, **span))
    return rst


def def_name_parse(name, form):
    """Name of a def/class and, for ``name[T ...]``, its ``type_params`` keyword.

    type_params is passed only when present, so 3.11's constructors never see it.
    """
    if isinstance(name, GluedHead):
        if not plain_name_p(name.head):
            raise LispyCompileError(
                f"{name.head} is not a valid name for a generic definition", name
            )
        return [name.head.name, {"type_params": type_params_parse(name)}]
    if not plain_name_p(name):
        raise form_error(name, f"expects a name, got {name}", form)
    return [name.name, {}]


def plain_name_p(sexp):
    """A bare, undotted symbol that is no hard keyword: what Python accepts
    as a def/class/type name."""
    return (
        type(sexp) is Symbol
        and "." not in sexp.value
        and not reserved_name_p(sexp.name)
    )


def type_alias_p(sexp):
    # `type` is a soft keyword, as in Python: only the statement
    # (type NAME value) / (type NAME[T ...] value) -- two operands, the first a
    # bare undotted symbol or a glued head -- is a type alias.  Every other shape
    # (one or three operands, any expression position, a non-symbol first
    # operand) is a call to the builtin type.  Code that rebinds `type` and calls
    # it as a bare two-argument statement must write (= _ (type a b)).
    if atom_text(sexp.op) != "type" or len(sexp) != 3:
        return False
    name = sexp[1]
    return plain_name_p(name.head if isinstance(name, GluedHead) else name)


def type_alias_compile(sexp):
    require("type-alias", sexp)
    [_, name, value] = sexp.list
    [head, type_params] = (
        [name.head, type_params_parse(name)]
        if isinstance(name, GluedHead)
        else [name, []]
    )
    return ast.TypeAlias(
        name=name_compile(head, ast.Store),
        type_params=type_params,
        value=expr_compile(value),
        **sexp.position_info
    )


DECORATABLE = ("def", "async-def", "class", "deco")


def deco_compile(sexp, decorator_list):
    [[decorator, def_statement], rest] = form_parts(
        sexp, ("decorator", "definition")
    )
    if rest:
        raise form_error(sexp, "takes one decorator and one definition")
    if not clause_p(def_statement, DECORATABLE):
        # Compiling anything else would drop the decorator without a word.
        raise form_error(
            def_statement,
            f"expects a def, async-def, class or deco form, got {def_statement}",
            "deco",
        )
    new_deco_list = decorator.list if isinstance(decorator, Bracket) else [decorator]
    return stmt_compile(
        def_statement, (decorator_list if decorator_list else []) + new_deco_list
    )


def functiondef_p(sexp):
    return atom_text(sexp.op) == "def"


def definition_parts(sexp, header):
    """``[name, bracket, body]`` of a def/async-def/class: the bracket (args
    or bases) is required, an empty body compiles to pass."""
    [[name, bracket], body] = form_parts(sexp, ("name", header))
    if not isinstance(bracket, Bracket):
        raise form_error(
            bracket, f"expects [{header}] after its name, got {bracket}", str(sexp.op)
        )
    return [name, bracket, body]


def functiondef_compile(sexp, decorator_list, async_p=False):
    [fnname, args, body] = definition_parts(sexp, "args")
    [name, type_params] = def_name_parse(fnname, str(sexp.op))
    if body and isinstance(body[0], Annotation):
        [ann, *body] = body
    else:
        ann = None
    return (ast.AsyncFunctionDef if async_p else ast.FunctionDef)(
        name=name,
        **type_params,
        args=def_args_parse(args),
        body=stmt_list_compile(body) if body else [ast.Pass(**sexp.position_info)],
        decorator_list=list(map(expr_compile, decorator_list))
        if decorator_list
        else [],
        returns=expr_compile(ann.value) if ann else None,
        **sexp.position_info
    )


def return_p(sexp):
    return atom_text(sexp.op) == "return"


def return_compile(sexp):
    # An empty [] or {} node is falsy (Expression.__len__): test presence by length.
    if len(sexp) > 2:
        raise LispyCompileError("(return ...) takes at most one value", sexp)
    value = expr_compile(sexp[1]) if len(sexp) > 1 else None
    return ast.Return(value=value, **sexp.position_info)


def scope_names(sexp):
    """The names of a global/nonlocal form: at least one, each a plain name."""
    names = sexp.operands
    if not names:
        raise form_error(sexp, "needs at least one name")
    for name in names:
        if isinstance(name, GluedHead):
            raise name.misuse_error()
        if not plain_name_p(name):
            raise form_error(name, f"expects a name, got {name}", str(sexp.op))
    return [name.name for name in names]


def global_compile(sexp):
    return ast.Global(names=scope_names(sexp), **sexp.position_info)


def nonlocal_compile(sexp):
    return ast.Nonlocal(names=scope_names(sexp), **sexp.position_info)


def classdef_p(sexp):
    return atom_text(sexp.op) == "class"


def classdef_args_parse(args):
    q = deque(args)
    bases = []
    keywords = []
    while q:
        arg = q.popleft()
        keywords.append(
            ast.keyword(
                arg=arg.name, value=expr_compile(q.popleft()), **arg.position_info
            )
        ) if keyword_arg_p(arg) else bases.append(expr_compile(arg))
    return [bases, keywords]


def classdef_compile(sexp, decorator_list):
    [clsname, args, body] = definition_parts(sexp, "bases")
    [name, type_params] = def_name_parse(clsname, str(sexp.op))
    [bases, keywords] = classdef_args_parse(args)
    return ast.ClassDef(
        name=name,
        **type_params,
        bases=bases,
        keywords=keywords,
        body=stmt_list_compile(body) if body else [ast.Pass(**sexp.position_info)],
        decorator_list=list(map(expr_compile, decorator_list))
        if decorator_list
        else [],
        **sexp.position_info
    )


def stmt_compile(sexp, decorator_list=None):
    try:
        return (
            expr_wrapper(sexp)
            if not paren_p(sexp) or not len(sexp)
            else do_compile(sexp)
            if do_p(sexp)
            else assign_compile(sexp)
            if assign_p(sexp)
            else augassign_compile(sexp)
            if augassign_p(sexp)
            else del_compile(sexp)
            if atom_text(sexp.op) == "del"
            else ast.Pass(**sexp.position_info)
            if atom_text(sexp.op) == "pass"
            else import_compile(sexp)
            if atom_text(sexp.op) == "import"
            else importfrom_compile(sexp)
            if atom_text(sexp.op) == "from"
            else import_compile(sexp, lazy=True)
            if atom_text(sexp.op) == "lazy-import"
            else importfrom_compile(sexp, lazy=True)
            if atom_text(sexp.op) == "lazy-from"
            else if_stmt_compile(sexp)
            if if_p(sexp)
            else while_compile(sexp)
            if while_p(sexp)
            else for_compile(sexp)
            if for_p(sexp)
            else ast.Break(**sexp.position_info)
            if atom_text(sexp.op) == "break"
            else ast.Continue(**sexp.position_info)
            if atom_text(sexp.op) == "continue"
            else raise_compile(sexp)
            if atom_text(sexp.op) == "raise"
            else assert_compile(sexp)
            if atom_text(sexp.op) == "assert"
            else try_compile(sexp)
            if atom_text(sexp.op) == "try"
            else with_compile(sexp)
            if atom_text(sexp.op) == "with"
            else match_compile(sexp)
            if atom_text(sexp.op) == "match"
            else deco_compile(sexp, decorator_list)
            if deco_p(sexp)
            else functiondef_compile(sexp, decorator_list)
            if functiondef_p(sexp)
            else return_compile(sexp)
            if return_p(sexp)
            else global_compile(sexp)
            if atom_text(sexp.op) == "global"
            else nonlocal_compile(sexp)
            if atom_text(sexp.op) == "nonlocal"
            else classdef_compile(sexp, decorator_list)
            if classdef_p(sexp)
            else functiondef_compile(sexp, decorator_list, async_p=True)
            if atom_text(sexp.op) == "async-def"
            else for_compile(sexp, async_p=True)
            if atom_text(sexp.op) == "async-for"
            else with_compile(sexp, async_p=True)
            if atom_text(sexp.op) == "async-with"
            else type_alias_compile(sexp)
            if type_alias_p(sexp)
            else expr_wrapper(sexp)
        )
    except RecursionError:
        # As in expr_compile: a form nested past Python's recursion limit.
        # A handler that hits the limit again building the error leaves it to
        # the next level's, which retries a level up.
        raise nesting_error(sexp) from None


def stmt_list_compile(sexp_list):
    return reduce(
        lambda x, y: x + (y if isinstance(y, list) else [y]),
        map(stmt_compile, sexp_list),
        [],
    )
