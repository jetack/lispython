"""Nodes a macro expansion builds take the macro call's source position.

Without one they would carry the Node default (line 0), so a traceback or a
compile error in macro-generated code would point nowhere."""

import ast
import signal
from collections import deque

import pytest

from lispy.core.errors import LispyCompileError
from lispy.core.macro import macroexpand_then_compile
from lispy.core.nodes import (
    Bracket,
    Constant,
    GluedHead,
    Keyword,
    Paren,
    Quote,
    String,
    Symbol,
    data_to_generator_expression,
    macro_expansion_nodes,
)
from lispy.core.parser import parse


def call_at(position):
    return Paren(Symbol("m"), **position)


@pytest.fixture(scope="module")
def scope():
    scope = {}
    macroexpand_then_compile(parse("(require lispy.macros *)"), scope)
    return scope


def compile_at_line_3(src, scope):
    return macroexpand_then_compile(parse("\n\n" + src), scope)


def nodes(stl):
    return [n for st in stl for n in ast.walk(st) if hasattr(n, "lineno")]


def calls_to(stl, name):
    return [
        n
        for n in nodes(stl)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == name
    ]


@pytest.mark.parametrize(
    "src",
    [
        "(-> 1 f)",
        "(->> 1 f print)",
        "(when True (f))",
        "(cond a (b) c (d))",
        "(as-> 1 x (f x))",
        "(fn [x] (f x))",
        "(inc (f))",
        "(++ x)",
        "(conde a (b) (c))",
        "(-> (read) solution print)",
        'f"{(-> 1 g):>{(inc w)}}"',
    ],
)
def test_macro_expansion_has_no_line_0(src, scope):
    stl = compile_at_line_3(src, scope)
    assert [type(n).__name__ for n in nodes(stl) if n.lineno == 0] == []


def test_generated_call_takes_the_macro_call_position(scope):
    stl = compile_at_line_3("  (-> 1 f)", scope)
    [call] = calls_to(stl, "f")
    assert (call.lineno, call.col_offset) == (3, 2)
    assert (call.end_lineno, call.end_col_offset) == (3, 10)


def test_generated_if_takes_the_macro_call_position(scope):
    [stmt] = compile_at_line_3("(when True (f))", scope)
    assert isinstance(stmt, ast.If)
    assert stmt.lineno == 3


def test_unquoted_user_arguments_keep_their_own_position(scope):
    stl = compile_at_line_3("(when True\n    (g))", scope)
    [stmt] = stl
    [call] = calls_to(stl, "g")
    assert stmt.lineno == 3
    assert (call.lineno, call.col_offset) == (4, 4)


def test_threaded_symbol_keeps_its_own_position(scope):
    stl = compile_at_line_3("(-> 1\n    f)", scope)
    [call] = calls_to(stl, "f")
    assert call.lineno == 3
    assert (call.func.lineno, call.func.col_offset) == (4, 4)


def define(src, scope):
    stl = macroexpand_then_compile(parse(src), scope)
    exec(compile(ast.Module(body=stl, type_ignores=[]), "<defs>", "exec"), scope)


def compile_at_line(line, src, scope):
    stl = macroexpand_then_compile(parse("\n" * (line - 1) + src), scope)
    compile(ast.Module(body=stl, type_ignores=[]), "<test>", "exec")
    return stl


def test_shared_template_takes_each_call_position(scope):
    # The macro returns one template object on every call; the first call
    # must not pin its position on the later ones.
    define("(= T `(sub [] 1))\n(defmacro shared-template [] (return T))", scope)
    for line in (10, 50):
        stl = compile_at_line(line, "(shared-template)", scope)
        assert [n.lineno for n in nodes(stl) if isinstance(n, ast.Subscript)] == [line]


def test_shared_node_beside_fresh_ones_keeps_a_valid_range(scope):
    # A fresh attribute form ending in a shared symbol: a stale symbol line
    # would end the attribute before it starts, which compile() rejects.
    define("(= A `y)\n(defmacro shared-attr [] (return `(. x ~A)))", scope)
    for line in (10, 50):
        stl = compile_at_line(line, "(shared-attr)", scope)
        [attr] = [n for n in nodes(stl) if isinstance(n, ast.Attribute)]
        assert (attr.lineno, attr.end_lineno) == (line, line)


# Filling positions copies the expansion. The copy must not spend Python frames
# per nesting level: a macro call nests as deeply as the same code without one
# (about 245 levels at the default recursion limit), where copy.deepcopy gave
# up near 160 with a raw RecursionError.
DEEP = 230


@pytest.mark.parametrize(
    "src",
    [
        "(when c " + "(f " * DEEP + "1" + ")" * DEEP + ")",
        "(-> x " + " ".join(f"f{i}" for i in range(DEEP)) + ")",
    ],
    ids=["nested-argument", "long-thread"],
)
def test_deeply_nested_macro_call_compiles(src, scope):
    compile_at_line(3, src, scope)


def test_position_fill_copies_and_keeps_sharing():
    shared = Symbol("x")
    placed = Symbol("y", lineno=7, col_offset=1, end_lineno=7, end_col_offset=2)
    tree = Paren(Symbol("f"), shared, shared, placed)
    tree.extra = (shared, deque([shared]))
    position = {"lineno": 3, "col_offset": 0, "end_lineno": 3, "end_col_offset": 9}
    copy = macro_expansion_nodes(tree, call_at(position))
    assert copy is not tree and shared.lineno == 0 and tree.lineno == 0
    [head, first, second, kept] = copy.list
    assert first is second is copy.extra[0] is copy.extra[1][0]
    assert first is not shared
    assert type(copy.extra) is tuple and type(copy.extra[1]) is deque
    assert [n.position_info for n in (copy, head, first)] == [position] * 3
    assert kept.position_info == placed.position_info and kept is not placed


def test_data_in_an_expression_list_becomes_a_positioned_node():
    inner = Paren(Symbol("g"))
    inner.list.append("s")
    tree = Paren(Symbol("f"), inner)
    tree.list.append(1)
    position = {"lineno": 3, "col_offset": 0, "end_lineno": 3, "end_col_offset": 9}
    copy = macro_expansion_nodes(tree, call_at(position))
    assert tree.list[2] == 1 and inner.list[1] == "s"
    [_, copied_inner, one] = copy.list
    assert type(one) is Constant and one.value == "1"
    assert type(copied_inner.list[1]) is String
    assert one.position_info == copied_inner.list[1].position_info == position


def test_a_data_expansion_becomes_a_positioned_node():
    position = {"lineno": 3, "col_offset": 0, "end_lineno": 3, "end_col_offset": 9}
    node = macro_expansion_nodes(7, call_at(position))
    assert type(node) is Constant and node.position_info == position


def nested_list(depth):
    data = [1]
    for _ in range(depth):
        data = [data]
    return data


def bracket_depth(node):
    depth = 0
    while isinstance(node, Bracket):
        [node] = node.list
        depth += 1
    assert type(node) is Constant and node.value == "1"
    return depth


DATA_DEPTH = 5000  # far past the recursion limit


def test_deeply_nested_data_converts_without_recursion():
    position = {"lineno": 3, "col_offset": 0, "end_lineno": 3, "end_col_offset": 9}
    tree = Paren(Symbol("f"))
    tree.list.append(nested_list(DATA_DEPTH))
    copy = macro_expansion_nodes(tree, call_at(position))
    assert bracket_depth(copy.list[1]) == DATA_DEPTH + 1
    assert copy.list[1].position_info == position
    assert bracket_depth(data_to_generator_expression(nested_list(DATA_DEPTH))) == (
        DATA_DEPTH + 1
    )
    assert bracket_depth(Paren(nested_list(DATA_DEPTH)).list[0]) == DATA_DEPTH + 1


@pytest.mark.parametrize(
    ("make", "type_name"),
    [("(dict)", "dict"), ("(set [1])", "set"), ("(object)", "object")],
)
def test_unrepresentable_data_is_refused_at_the_call(make, type_name, scope):
    define(
        f"(defmacro put-raw-{type_name} [] (= e `(f)) (e.list.append {make})"
        " (return e))\n"
        f"(defmacro ret-raw-{type_name} [] (return {make}))",
        scope,
    )
    for name in (f"put-raw-{type_name}", f"ret-raw-{type_name}"):
        with pytest.raises(LispyCompileError) as info:
            compile_at_line(3, f"  ({name})", scope)
        assert info.value.msg == (
            f"macro {name} returned a {type_name} in its expansion, "
            "which has no lispython form"
        )
        assert (info.value.lineno, info.value.offset) == (3, 3)
        assert (info.value.end_lineno, info.value.end_offset) == (3, 5 + len(name))


SELF_REFERENCE_BUDGET = 2.0  # seconds; a passing test takes milliseconds


@pytest.fixture
def time_bound():
    """Fail the test once it runs past SELF_REFERENCE_BUDGET.

    A self-referencing form has no finite copy: without the checks the
    conversion grows memory without bound (about 120 MB/s), fast enough to
    exhaust the machine before any memory error. The timer stops it first.
    Where setitimer is missing the test is skipped, never run unguarded.
    """
    if not hasattr(signal, "setitimer"):
        pytest.skip("needs signal.setitimer to bound a runaway self-reference")

    def expire(signum, frame):
        pytest.fail(f"ran past its {SELF_REFERENCE_BUDGET} s budget")

    old_handler = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, SELF_REFERENCE_BUDGET)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


# A form, list or tuple that contains itself has no finite form. Every test
# of one takes the time_bound fixture.
SELF_REFERENCING = {
    "ret-self": ("(defmacro ret-self [] (= x [1]) (x.append x) (return x))", "list"),
    "put-self": (
        "(defmacro put-self [] (= x [1]) (x.append x)"
        " (= e `(f)) (e.list.append x) (return e))",
        "list",
    ),
    "ret-self-tuple": (
        "(defmacro ret-self-tuple [] (= x [1]) (= t (, x)) (x.append t) (return t))",
        "tuple",
    ),
    "splice-self": (
        "(defmacro splice-self [] (= x [1]) (x.append x) (return `(f ~x)))",
        "list",
    ),
    "ret-self-paren": (
        "(defmacro ret-self-paren [] (= e `(f)) (e.list.append e) (return e))",
        "Paren",
    ),
    "ret-self-starred": (
        "(defmacro ret-self-starred [] (= x [1]) (= s `*~x) (x.append s)"
        " (return `(f ~s)))",
        "Starred",
    ),
    "ret-self-via-tuple": (
        "(defmacro ret-self-via-tuple [] (= e `(f))"
        ' (setattr e "extra" (, e)) (return e))',
        "Paren",
    ),
}


@pytest.mark.usefixtures("time_bound")
@pytest.mark.parametrize("name", sorted(SELF_REFERENCING))
def test_self_reference_is_refused_at_the_call(name, scope):
    [src, kind] = SELF_REFERENCING[name]
    define(src, scope)
    with pytest.raises(LispyCompileError) as info:
        compile_at_line(3, f"  ({name})", scope)
    assert info.value.msg == (
        f"macro {name} returned a self-referencing {kind} in its expansion"
    )
    assert (info.value.lineno, info.value.offset) == (3, 3)


@pytest.mark.usefixtures("time_bound")
def test_self_referencing_data_is_left_for_the_expander():
    # The quasiquote constructor converts what ~x unquotes; a list with no
    # finite form stays raw, so the expander refuses it at the macro call.
    x = [1]
    x.append(x)
    assert data_to_generator_expression(x) is x
    outer = [2, x]
    assert data_to_generator_expression(outer) is outer
    assert Paren(Symbol("f"), x).list[1] is x


POSITION = {"lineno": 3, "col_offset": 0, "end_lineno": 3, "end_col_offset": 9}


@pytest.mark.usefixtures("time_bound")
def test_a_slot_reaching_the_node_own_list_is_sharing():
    p = Paren(Symbol("f"), Symbol("a"))
    p.orig = p.list
    copy = macro_expansion_nodes(p, call_at(POSITION))
    assert copy.orig is copy.list and copy.list is not p.list
    p = Paren(Symbol("f"), Symbol("a"))
    q = Paren()
    q.list = p.list
    p.extra = q
    copy = macro_expansion_nodes(p, call_at(POSITION))
    assert copy.extra.list is copy.list and copy.list is not p.list


@pytest.mark.usefixtures("time_bound")
def test_a_cycle_through_a_shared_list_is_refused():
    p = Paren(Symbol("f"))
    q = Paren()
    q.list = p.list
    p.list.append(q)
    with pytest.raises(LispyCompileError, match="self-referencing list"):
        macro_expansion_nodes(p, call_at(POSITION))


class FormList(list):
    pass


@pytest.mark.usefixtures("time_bound")
def test_a_list_subclass_form_slot_is_walked():
    p = Paren(Symbol("f"))
    p.list = FormList([Symbol("f"), 1])
    copy = macro_expansion_nodes(p, call_at(POSITION))
    assert type(copy.list) is list and type(copy.list[1]) is Constant
    p.list.append(p)
    with pytest.raises(LispyCompileError, match="self-referencing Paren"):
        macro_expansion_nodes(p, call_at(POSITION))


@pytest.mark.usefixtures("time_bound")
def test_self_referencing_node_attribute_is_refused():
    call = call_at({"lineno": 3, "col_offset": 0, "end_lineno": 3, "end_col_offset": 9})
    tree = Paren(Symbol("f"))
    tree.extra = deque([tree])
    with pytest.raises(LispyCompileError, match="self-referencing Paren"):
        macro_expansion_nodes(tree, call)
    tree.extra = [Symbol("g")]
    tree.extra.append(tree.extra)
    with pytest.raises(LispyCompileError, match="self-referencing list"):
        macro_expansion_nodes(tree, call)


def test_shared_data_is_not_a_self_reference(scope):
    x = [1]
    assert str(data_to_generator_expression([x, [x]])) == "[[1] [[1]]]"
    define("(defmacro shared-data [] (= x [1]) (return [x x]))", scope)
    [stmt] = compile_at_line(3, "(shared-data)", scope)
    assert ast.unparse(stmt) == "[[1], [1]]"


@pytest.mark.parametrize(
    ("macro", "python"),
    [
        ("(= y 5) (return `(list *~y))", "list(*5)"),
        ("(= y [1 2]) (return `(f **~y))", "f(**[1, 2])"),
        ('(= y "int") (return `(= a ^~y 1))', "a: 'int' = 1"),
    ],
)
def test_data_in_a_wrapper_is_converted(macro, python, scope):
    define(f"(defmacro wrapped [] {macro})", scope)
    stl = compile_at_line(3, "(wrapped)", scope)
    assert ast.unparse(ast.Module(stl, [])) == python
    assert [n.lineno for n in nodes(stl)] == [3] * len(nodes(stl))


def test_unrepresentable_data_in_a_wrapper_is_refused(scope):
    define("(defmacro wrapped-dict [] (= y (dict)) (return `(list *~y)))", scope)
    with pytest.raises(LispyCompileError) as info:
        compile_at_line(3, "(wrapped-dict)", scope)
    assert info.value.msg.startswith("macro wrapped-dict returned a dict")
    assert info.value.lineno == 3


def test_data_in_every_child_form_slot_is_converted():
    position = {"lineno": 3, "col_offset": 0, "end_lineno": 3, "end_col_offset": 9}
    quote = Quote(None)
    quote.value = 5
    glued = GluedHead(Symbol("f"), None)
    glued.params = ["T"]
    keyword = Keyword(None)
    keyword.value = [1]
    tree = Paren(Symbol("g"), quote, glued, keyword)
    copy = macro_expansion_nodes(tree, call_at(position))
    [_, copied_quote, copied_glued, copied_keyword] = copy.list
    assert type(copied_quote.value) is Constant
    assert type(copied_glued.params) is Bracket
    assert type(copied_glued.params.list[0]) is String
    assert type(copied_keyword.value) is Bracket
    assert copied_keyword.value.list[0].position_info == position
    assert quote.value == 5 and glued.params == ["T"]
