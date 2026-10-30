"""Statement forms and match patterns to Python AST.

Compiled on the trampoline as expressions are: every function that compiles
a sub-form is a step that asks for each sub-result with ``yield`` (the
convention is stated in expr.py's module docstring). ``stmt_compile`` and
``stmt_list_compile`` are the plain entry points.
"""

import ast
from collections import deque
from functools import reduce
from itertools import islice

from lispy.core.compiler import features
from lispy.core.compiler.expr import (
    call_args_parse,
    call_form_p,
    def_args_parse,
    expr_list_steps,
    expr_steps,
    refuse_empty_form,
    target_compile,
)
from lispy.core.compiler.features import require
from lispy.core.compiler.literal import (
    SPECIAL_CONSTANTS,
    module_path,
    name_compile,
    name_or_attr_p,
    number_parts,
    number_pattern_expr,
    plain_name_p,
    symbol_name,
)
from lispy.core.compiler.utils import *
from lispy.core.errors import LispyCompileError
from lispy.core.nodes import GluedHead, Paren, atom_text
from lispy.core.utils import *


def expr_wrapper(sexp):
    return ast.Expr(value=(yield expr_steps(sexp)), **sexp.position_info)


def do_p(sexp):
    return atom_text(sexp.op) == "do"


def do_compile(sexp):
    [op, *sexps] = sexp.list
    return (yield stmt_list_steps(sexps))


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
        value_dict = {"value": (yield expr_steps(value[0]))} if value else {}
        target_node = yield target_compile(target, "annotate")
        return ast.AnnAssign(
            target=target_node,
            annotation=(yield expr_steps(annotation)),
            # Decided on the compiled target: (sub a) compiles to the
            # name a, which Python records in __annotations__ like a bare a.
            simple=int(isinstance(target_node, ast.Name)),
            **value_dict,
            **sexp.position_info
        )
    else:
        [*targets, value] = [target, *body]
        target_nodes = []
        for x in targets:
            target_nodes.append((yield target_compile(x, "assign")))
        return ast.Assign(
            targets=target_nodes,
            value=(yield expr_steps(value)),
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
        (yield expr_list_steps(args)),
    )
    return ast.AugAssign(
        target=(yield target_compile(target, "augassign")),
        op=op(),
        value=value,
        **sexp.position_info
    )


def del_compile(sexp):
    [op, *args] = sexp.list
    if not args:
        raise form_error(sexp, "needs at least one target")
    targets = []
    for x in args:
        targets.append((yield target_compile(x, "delete")))
    return ast.Delete(targets=targets, **sexp.position_info)


def import_name(sexp, form, dotted=True):
    """The one check on every name an import form reads: a bare symbol
    (dotted for a module path), as in Python, or a LispyCompileError at it."""
    if isinstance(sexp, GluedHead):
        raise sexp.misuse_error()
    if type(sexp) is not Symbol or (not dotted and "." in sexp.value):
        raise LispyCompileError(f"({form} ...) expects a name, got {sexp}", sexp)
    return sexp


def parse_names(names, form, from_p=False):
    """The aliases of an import's names: ``name`` or ``name as alias``.

    Every part goes through the name model; the name the statement binds (the
    alias, else the first part of an import's module, else the imported
    name) as a binding, so ``import __debug__`` is refused and
    ``import __debug__ as x`` is not.
    """
    if names == "*":
        return [ast.alias(name="*", **names.position_info)]
    rst = []
    q = deque(names)
    while q:
        n = import_name(q.popleft(), form, dotted=not from_p)
        aliased = len(q) > 1 and q[0] == "as"
        name_ctx = ast.Load if aliased else ast.Store
        name = (
            symbol_name(n, name_ctx) if from_p else module_path(n, name_ctx)[0]
        )
        alias = ast.alias(name=name, **n.position_info)
        if aliased:
            q.popleft()
            alias.asname = symbol_name(
                import_name(q.popleft(), form, dotted=False), ast.Store
            )
        rst.append(alias)
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

    module_level = [
        [*module_path(module, relative=True), module.position_info]
        for module in modules
    ]
    return [
        ast.ImportFrom(
            module=module,
            names=parse_names(names, form, from_p=True),
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
        test=(yield expr_steps(test)),
        body=(yield stmt_list_steps([then])),
        orelse=(yield stmt_list_steps(orelse)),
        **sexp.position_info
    )


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
        test=(yield expr_steps(test)),
        body=(yield stmt_list_steps(then)),
        orelse=(yield stmt_list_steps(orelse)),
        **sexp.position_info
    )


def for_p(sexp):
    return atom_text(sexp.op) == "for"


def for_compile(sexp, async_p=False):
    [[target, iterable], then, orelse] = loop_parse(
        sexp, ("target", ("in", "iterable"))
    )
    return (ast.AsyncFor if async_p else ast.For)(
        target=(yield target_compile(target, "assign")),
        iter=(yield expr_steps(iterable)),
        body=(yield stmt_list_steps(then)),
        orelse=(yield stmt_list_steps(orelse)),
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
    kwargs = {"exc": (yield expr_steps(body[0]))} if len(body) else {}
    if len(body) == 3:
        kwargs["cause"] = yield expr_steps(body[2])
    return ast.Raise(**kwargs, **sexp.position_info)


def assert_compile(sexp):
    [[test], rest] = form_parts(sexp, ("test",))
    if len(rest) > 1:
        raise form_error(sexp, "takes a test and at most one message")
    kwargs = {"test": (yield expr_steps(test))}
    if len(rest):
        kwargs["msg"] = yield expr_steps(rest[0])
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
        return [lst[:i], symbol_name(lst[-1], ast.Store)]
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
                elts=(yield expr_list_steps(types)),
                ctx=ast.Load(),
                **bracket.position_info
            )
            if len(types) > 1
            else (yield expr_steps(types[0]))
        )
        if name is not None:
            kwargs["name"] = name
    elif form == "except*":
        raise form_error(handler, "needs a [types] bracket")
    if not body:
        raise form_error(handler, "needs a body")
    return ast.ExceptHandler(
        body=(yield stmt_list_steps(body)), **kwargs, **handler.position_info
    )


TRY_CLAUSES = ("except", "except*", "else", "finally")


def try_parse(sexp):
    """``[body, handlers, orelse, finalbody, star]`` of a try form.

    The one check of try's layout, as Python's grammar has it: a body, then
    except clauses or except* clauses (never both, a bare except last), an
    optional else (only after an except), an optional finally, and at least
    one except or a finally. A form out of that order is refused where it
    stands, never left in the body to compile as a call to a keyword. The
    handlers are compiled here, as each is read.
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
            handlers.append((yield parse_except(clause)))
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
    [body, handlers, orelse, finalbody, star] = yield try_parse(sexp)
    return (ast.TryStar if star else ast.Try)(
        body=(yield stmt_list_steps(body)),
        handlers=handlers,
        orelse=(yield stmt_list_steps(orelse)),
        finalbody=(yield stmt_list_steps(finalbody)),
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
            rst[-1].optional_vars = yield target_compile(q.popleft(), "assign")
            targeted = True
        else:
            rst.append(ast.withitem(context_expr=(yield expr_steps(elt))))
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
        items=(yield with_items_parse(items, form)),
        body=(yield stmt_list_steps(body)),
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
        return (yield expr_steps(sexp))
    if expr is None:
        raise form_error(sexp, f"can sign only a number literal, got {sexp}", "pattern")
    return expr


def dotted_name_p(sexp):
    return type(sexp) is Symbol and "." in sexp.value


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
        return (yield literal_pattern_expr(sexp))
    if kind == "mapping pattern":
        raise form_error(
            sexp, f"expects a literal or dotted name as a key, got {sexp}", kind
        )
    raise form_error(sexp, f"expected, got {sexp}", kind)


def match_sequence_parse(sexp):
    """``[pattern... *rest pattern...]``: the one place a *rest is a pattern."""
    patterns = []
    starred = False
    for elt in sexp.list:
        if not starred_p(elt):
            patterns.append((yield pattern_parse(elt)))
            continue
        if starred:
            raise form_error(
                elt, f"has more than one *rest, got {elt}", "sequence pattern"
            )
        starred = True
        name = elt.value
        if atom_text(name) != "_" and not plain_name_p(name):
            raise form_error(
                elt, f"expects a name after *, got {name}", "sequence pattern"
            )
        patterns.append(
            ast.MatchStar(
                **{}
                if atom_text(name) == "_"
                else {"name": symbol_name(name, ast.Store)},
                **elt.position_info
            )
        )
    return {"patterns": patterns}


def match_mapping_parse(sexp):
    """``{key pattern... **rest}``: **rest once, last, a plain name."""
    items = list(sexp.list)
    rst = {}
    if items and doublestarred_p(items[-1]):
        rest = items.pop()
        if not plain_name_p(rest.value) or atom_text(rest.value) == "_":
            raise form_error(
                rest, f"expects a name after **, got {rest.value}", "mapping pattern"
            )
        rst["rest"] = symbol_name(rest.value, ast.Store)
    for item in items:
        if doublestarred_p(item):
            raise form_error(item, f"has {item} before its end", "mapping pattern")
    if len(items) % 2:
        raise form_error(
            items[-1],
            f"has key {items[-1]} without a pattern after it",
            "mapping pattern",
        )
    rst["keys"] = []
    for key in items[::2]:
        rst["keys"].append((yield value_pattern_expr(key, "mapping pattern")))
    rst["patterns"] = []
    for pattern in items[1::2]:
        rst["patterns"].append((yield pattern_parse(pattern)))
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
            if not plain_name_p(arg.value):
                raise form_error(
                    arg,
                    f"expects an attribute name after :, got {arg.value}",
                    "class pattern",
                )
            kwd_attrs.append(symbol_name(arg.value, ast.Store))
            kwd_patterns.append((yield pattern_parse(q.popleft())))
        elif kwd_attrs:
            raise form_error(arg, f"has {arg} after a keyword pattern", "class pattern")
        else:
            patterns.append((yield pattern_parse(arg)))
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
        return ast.MatchAs(name=symbol_name(sexp, ast.Store), **position_info)
    if atom_text(sexp) in ("True", "False", "None") and constant_p(sexp):
        return ast.MatchSingleton(value=eval(str(sexp)), **position_info)
    if bracket_p(sexp):
        return ast.MatchSequence(
            **(yield match_sequence_parse(sexp)), **position_info
        )
    if brace_p(sexp):
        return ast.MatchMapping(**(yield match_mapping_parse(sexp)), **position_info)
    if paren_p(sexp) and sexp.op == "|":
        if len(sexp.operands) < 2:
            raise form_error(sexp, "needs at least two patterns", "|")
        patterns = []
        for pattern in sexp.operands:
            patterns.append((yield pattern_parse(pattern)))
        return ast.MatchOr(patterns=patterns, **position_info)
    if paren_p(sexp) and not signed_number_p(sexp):
        if not call_form_p(sexp):
            raise form_error(sexp, f"expected, got {sexp}", "pattern")
        return ast.MatchClass(**(yield match_class_parse(sexp)), **position_info)
    return ast.MatchValue(
        value=(yield value_pattern_expr(sexp, "pattern")), **position_info
    )


def case_parse(case):
    """``(case pattern [as name] [if guard] body...)``; the case is checked
    in match_compile, the parts here, each refusal at the case."""
    [[pattern_expr], body] = form_parts(case, ("pattern",))
    pattern = yield pattern_parse(pattern_expr)
    if len(body) and body[0] == "as":
        if len(body) < 2:
            raise form_error(case, "has as without a name after it")
        if not plain_name_p(body[1]):
            raise form_error(body[1], f"expects a name after as, got {body[1]}", "case")
        name = symbol_name(body[1], ast.Store)
        if name == "_":
            # A wildcard captures nothing: CPython refuses `case 1 as _`.
            raise LispyCompileError("cannot use '_' as a target", body[1])
        pattern = ast.MatchAs(
            pattern=pattern,
            name=name,
            **merge_position_infos(pattern_expr.position_info, body[1].position_info)
        )
        body = body[2:]
    guard_dict = {}
    if len(body) and body[0] == "if":
        if len(body) < 2:
            raise form_error(case, "has if without a guard after it")
        guard_dict = {"guard": (yield expr_steps(body[1]))}
        body = body[2:]
        if len(body) and body[0] == "as":
            raise form_error(
                body[0], "has as after its guard; as name comes before if", "case"
            )
    if not body:
        raise form_error(case, "needs a body")
    return ast.match_case(
        pattern=pattern, **guard_dict, body=(yield stmt_list_steps(body))
    )


def match_compile(sexp):
    [[subject], cases] = form_parts(sexp, ("subject",))
    if not cases:
        raise form_error(sexp, "needs at least one case")
    for case in cases:
        if not clause_p(case, ("case",)):
            raise form_error(case, f"expects (case ...) clauses, got {case}", "match")
    case_nodes = []
    for case in cases:
        case_nodes.append((yield case_parse(case)))
    return ast.Match(
        subject=(yield expr_steps(subject)), cases=case_nodes, **sexp.position_info
    )


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
        fields = {"name": symbol_name(symbol, ast.Store)}
        if q and isinstance(q[0], Annotation):
            bound = q.popleft()
            if kind != "TypeVar":
                raise LispyCompileError(f"cannot use bound with {kind}", bound)
            fields["bound"] = yield expr_steps(bound.value)
            last = bound
        if has_default:
            require("type-param-default", param)
            if not q:
                raise LispyCompileError(
                    f"type parameter {symbol.value} needs a default value", param
                )
            last = q.popleft()
            fields["default_value"] = yield expr_steps(last)
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
        return [
            symbol_name(name.head, ast.Store),
            {"type_params": (yield type_params_parse(name))},
        ]
    if not plain_name_p(name):
        raise form_error(name, f"expects a name, got {name}", form)
    return [symbol_name(name, ast.Store), {}]


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
        [name.head, (yield type_params_parse(name))]
        if isinstance(name, GluedHead)
        else [name, []]
    )
    return ast.TypeAlias(
        name=name_compile(head, ast.Store),
        type_params=type_params,
        value=(yield expr_steps(value)),
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
    return (
        yield stmt_steps(
            def_statement, (decorator_list if decorator_list else []) + new_deco_list
        )
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
    [name, type_params] = yield def_name_parse(fnname, str(sexp.op))
    if body and isinstance(body[0], Annotation):
        [ann, *body] = body
    else:
        ann = None
    return (ast.AsyncFunctionDef if async_p else ast.FunctionDef)(
        name=name,
        **type_params,
        args=(yield def_args_parse(args)),
        body=(
            (yield stmt_list_steps(body)) if body else [ast.Pass(**sexp.position_info)]
        ),
        decorator_list=(
            (yield expr_list_steps(decorator_list)) if decorator_list else []
        ),
        returns=(yield expr_steps(ann.value)) if ann else None,
        **sexp.position_info
    )


def return_p(sexp):
    return atom_text(sexp.op) == "return"


def return_compile(sexp):
    # An empty [] or {} node is falsy (Expression.__len__): test presence by length.
    if len(sexp) > 2:
        raise LispyCompileError("(return ...) takes at most one value", sexp)
    value = (yield expr_steps(sexp[1])) if len(sexp) > 1 else None
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
    # global/nonlocal declare names, binding none: __debug__ is accepted.
    return [symbol_name(name, ast.Load) for name in names]


def global_compile(sexp):
    return ast.Global(names=scope_names(sexp), **sexp.position_info)


def nonlocal_compile(sexp):
    return ast.Nonlocal(names=scope_names(sexp), **sexp.position_info)


def classdef_p(sexp):
    return atom_text(sexp.op) == "class"


def classdef_compile(sexp, decorator_list):
    [clsname, args, body] = definition_parts(sexp, "bases")
    [name, type_params] = yield def_name_parse(clsname, str(sexp.op))
    # Bases are a call's arguments, as in Python: class A(B, *bs, m=M, **kw).
    [bases, keywords] = yield call_args_parse(args)
    return ast.ClassDef(
        name=name,
        **type_params,
        bases=bases,
        keywords=keywords,
        body=(
            (yield stmt_list_steps(body)) if body else [ast.Pass(**sexp.position_info)]
        ),
        decorator_list=(
            (yield expr_list_steps(decorator_list)) if decorator_list else []
        ),
        **sexp.position_info
    )


def stmt_steps(sexp, decorator_list=None):
    """The steps compiling one statement: its form's generator, or the node
    (or list of nodes) itself for a statement with no sub-form (see expr.py's
    module docstring)."""
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


# The end of an operand iterator in stmt_list_steps.
_DONE = object()


def stmt_list_steps(sexp_list):
    """The statements of ``sexp_list``, compiled in order; a form compiling to
    several (do, from) contributes them all.

    A ``do`` that stmt_steps would give to do_compile is inlined from a
    worklist of operand iterators instead: a nest of them would otherwise copy
    each level's whole statement list into the level above, quadratic in the
    nesting.
    """
    stmts = []
    pending = [iter(sexp_list)]
    while pending:
        sexp = next(pending[-1], _DONE)
        if sexp is _DONE:
            pending.pop()
            continue
        if paren_p(sexp) and len(sexp) and do_p(sexp):
            pending.append(islice(sexp.list, 1, None))
            continue
        stmt = yield stmt_steps(sexp)
        if isinstance(stmt, list):
            stmts.extend(stmt)
        else:
            stmts.append(stmt)
    return stmts


def stmt_compile(sexp, decorator_list=None):
    return trampoline(stmt_steps(sexp, decorator_list))


def stmt_list_compile(sexp_list):
    return trampoline(stmt_list_steps(sexp_list))
