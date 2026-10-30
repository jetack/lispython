import ast

import pytest

from lispy.core.errors import LispyCompileError
from lispy.tools import parse, src_to_python_org
from tests.utils import stmt_to_dump


class TestStarredSubscript:
    """``t[*Ts]`` is Subscript(slice=Tuple([Starred])) on every supported CPython."""

    @pytest.mark.parametrize(
        "lpy, py",
        [
            ("(sub tuple *Ts)", "tuple[*Ts]"),
            ("(= (sub x *a) 1)", "x[*a] = 1"),
            ("(+= (sub x *a) 1)", "x[*a] += 1"),
            ("(del (sub x *a))", "del x[*a]"),
            ("(sub x *a *b)", "x[*a][*b]"),
            ("(sub x (, *a 1))", "x[*a, 1]"),
        ],
    )
    def test_matches_python(self, lpy, py):
        assert stmt_to_dump(lpy) == ast.dump(ast.parse(py))


class TestUnaryOpMethods:
    def test_unaryop(self):
        assert src_to_python_org("+2") == "+2"


class TestBinOpMethods:
    def test_binop(self):
        assert src_to_python_org("(+ 2 3)") == "2 + 3"


class TestBoolOpMethods:
    def test_boolop(self):
        assert src_to_python_org("(and 1 2)") == "1 and 2"


class TestCompareMethods:
    def test_eq(self):
        assert src_to_python_org("(== 1 2 3)") == "1 == 2 == 3"

    def test_noteq(self):
        assert src_to_python_org("(!= 1 2 3)") == "1 != 2 != 3"

    def test_lt(self):
        assert src_to_python_org("(< 1 2 3)") == "1 < 2 < 3"

    def test_lte(self):
        assert src_to_python_org("(<= 1 2 3)") == "1 <= 2 <= 3"

    def test_gt(self):
        assert src_to_python_org("(> 1 2 3)") == "1 > 2 > 3"



class TestNumberAfterPrefix:
    """``*1``, ``**1`` and ``:1`` read a Constant, so every consumer of a
    prefixed operand either handles it or refuses at it: never a raw
    AttributeError/IndexError, nor a symbol named ``1``."""

    @pytest.mark.parametrize(
        ("src", "msg", "col"),
        [
            ("(f :1 x)", "expected a name, got 1", 5),
            ("(f :1.5 x)", "expected a name, got 1.5", 5),
            ("(f x :2j 3)", "expected a name, got 2j", 7),
            ("(def g [:1 2] 1)", "expected a name, got 1", 10),
            ("(def g [a :1] 1)", "expected a name, got 1", 12),
            ("(def g [*1] 1)", "expected a name, got 1", 10),
            ("(def g [**1] 1)", "expected a name, got 1", 11),
            ("(lambda [*1] 1)", "expected a name, got 1", 11),
            ("(class A [:1 2] 1)", "expected a name, got 1", 12),
            ("(= x :1)", ":1 is only valid before a value", 6),
            ("(with [x :1] 1)", ":1 is only valid before a value", 10),
        ],
    )
    def test_refused_at_the_operand_or_prefix(self, src, msg, col):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump(src)
        assert info.value.msg.startswith(msg)
        assert (info.value.lineno, info.value.offset) == (1, col)

    @pytest.mark.parametrize(
        ("src", "msg", "col"),
        [
            ("(= x :a)", ":a is only valid before a value", 6),
            ("(del :a)", ":a is only valid before a value", 6),
            ("(f :a :b 1)", ":b is only valid before a value", 7),
            ("[**1]", "**1 is only valid in a call, a dict", 2),
            ("(= x **a)", "**a is only valid in a call, a dict", 6),
        ],
    )
    def test_bare_prefixed_operand_refused(self, src, msg, col):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump(src)
        assert info.value.msg.startswith(msg)
        assert (info.value.lineno, info.value.offset) == (1, col)

    @pytest.mark.parametrize(
        ("src", "py"),
        [("(def g [a ^1] 1)", "def g(a: 1):\n    1"), ("(f *1)", "f(*1)"), ("(f **1)", "f(**1)"), ("[*1]", "[*1]")],
    )
    def test_star_of_a_number_compiles(self, src, py):
        assert src_to_python_org(src) == py


class TestKeywordWithoutValue:
    """A ``:name`` that ends a call, def or class argument list has no value
    to pop: a positioned LispyCompileError, not an IndexError."""

    @pytest.mark.parametrize(
        ("src", "col"),
        [
            ("(f :a)", 4),
            ("(f *a :b)", 7),
            ("(def g [a :b] 1)", 11),
            ("(def g [* a :b] 1)", 13),
            ("(class A [b :c] 1)", 13),
        ],
    )
    def test_refused(self, src, col):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump(src)
        assert info.value.msg.endswith("has no value after it")
        assert (info.value.lineno, info.value.offset) == (1, col)


def assert_refused_at(src, msg, at):
    """Compiling ``src`` (one line) is a LispyCompileError whose message
    starts with ``msg``, spanning ``at`` (``text@n``: its n-th occurrence in
    ``src``; None: all of ``src``)."""
    with pytest.raises(LispyCompileError) as info:
        stmt_to_dump(src)
    assert info.value.msg.startswith(msg)
    [text, _, nth] = (src if at is None else at).partition("@")
    index = -1
    for _ in range(int(nth or 1)):
        index = src.index(text, index + 1)
    assert (info.value.lineno, info.value.offset) == (1, index + 1)
    assert (info.value.end_lineno, info.value.end_offset) == (
        1,
        index + 1 + len(text),
    )
    return info.value


class TestAttributeNeedsAnAttribute:
    """Python has no attribute access without an attribute: (. x) is refused
    in every context, at the whole form, quoting the attribute form's usage."""

    @pytest.mark.parametrize(
        "src",
        [
            "(. a)",
            "(f (. a))",
            "(= (. a) 1)",
            "(= x (. a))",
            "(del (. a))",
            "(= (. a) ^int 1)",
            "(+= (. a) 1)",
            "(:= (. a) 1)",
            "(for (. a) in xs 1)",
            "(. (. a))",
            "(. (. a b))",
        ],
    )
    def test_one_operand_form_is_refused(self, src):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump(src)
        assert info.value.msg.startswith(
            "(. ...) is missing its attr: (. value attr...)"
        )

    @pytest.mark.parametrize(
        ("src", "at"),
        [
            ("(. a)", "(. a)"),
            ("(= (. a) 1)", "(. a)"),
            ("(del (. a))", "(. a)"),
            ("(= (. a) ^int 1)", "(. a)"),
            ("(+= (. a) 1)", "(. a)"),
        ],
    )
    def test_positioned_at_the_form(self, src, at):
        assert_refused_at(src, "(. ...) is missing its attr: (. value attr...)", at)

    def test_attribute_with_an_attr_still_compiles(self):
        assert stmt_to_dump("(= (. a b) ^int 1)") == ast.dump(
            ast.parse("a.b: int = 1")
        )


class TestExpressionFormArity:
    """A malformed expression form is a LispyCompileError naming the problem
    and quoting the form's FORM_USAGE line, positioned at the form or at the
    part at fault, never an IndexError/ValueError/AssertionError, an AST
    compile() rejects or a silently dropped operand."""

    @pytest.mark.parametrize(
        ("src", "msg", "at"),
        [
            ("(:=)", "(:= ...) is missing its name", None),
            ("(:= a)", "(:= ...) is missing its value", None),
            ("(:= a 1 2)", "(:= ...) takes one name and one value", None),
            ("(:= a ^int 1)", "(:= ...) takes no annotation", "^int"),
            ("(.)", "(. ...) is missing its value", None),
            ("(. a)", "(. ...) is missing its attr", None),
            ("(.foo)", "method call is missing its instance", None),
            ("(ife)", "(ife ...) is missing its test", None),
            ("(ife a)", "(ife ...) is missing its then", None),
            ("(ife a b c d)", "(ife ...) takes a test, a then and at most one", None),
            ("(sub)", "(sub ...) is missing its value", None),
            ("(lambda)", "(lambda ...) is missing its args", None),
            ("(lambda [a])", "(lambda ...) is missing its body", None),
            ("(lambda [a] 1 2)", "(lambda ...) takes one body expression", None),
            ("(lambda x 1)", "(lambda ...) expects [args], got x", "x"),
            (
                "(lambda [a ^int] 1)",
                "(lambda ...) takes no annotations on its parameters",
                "^int",
            ),
            (
                "(lambda [*a ^int] 1)",
                "(lambda ...) takes no annotations on its parameters",
                "^int",
            ),
            (
                "(lambda [:a ^int 1 **k ^str] 1)",
                "(lambda ...) takes no annotations on its parameters",
                "^int",
            ),
            ("(await)", "(await ...) is missing its value", None),
            ("(await a b)", "(await ...) takes one value", None),
            ("(yield a b)", "(yield ...) takes at most one value", None),
            ("(yield-from)", "(yield-from ...) is missing its iterable", None),
            ("(yield-from a b)", "(yield-from ...) takes one iterable", None),
            ("(and)", "(and ...) needs at least one value", None),
            ("(or)", "(or ...) needs at least one value", None),
            ("(<)", "(< ...) needs at least two operands", None),
            ("(< a)", "(< ...) needs at least two operands", None),
            ("(== a)", "(== ...) needs at least two operands", None),
            ("(is-not a)", "(is-not ...) needs at least two operands", None),
            ("(not-in a)", "(not-in ...) needs at least two operands", None),
            ("[:]", "slice takes one to three operands", None),
            ("[: 1 2 3 4]", "slice takes one to three operands", None),
            ("{1}", "dict has a key without a value", "1"),
            ("{1 2 3}", "dict has a key without a value", "3"),
            ("{a **m}", "dict has a key without a value", "a"),
            ("{**m a}", "dict has a key without a value", "a"),
            ("(x for)", "comprehension has for without a target", "for"),
            ("(x for y)", "comprehension has for without an iterable", "for"),
            ("(x for y in)", "comprehension has for without an iterable", "for"),
            (
                "(x async-for y)",
                "comprehension has async-for without an iterable",
                "async-for",
            ),
            ("{a b for}", "comprehension has for without a target", "for"),
            ("[x for y in z if]", "comprehension has if without a test", "if"),
            (
                "(x for y in z if for w in v)",
                "comprehension has if without a test",
                "if",
            ),
            (
                "(x for y in z if a b)",
                "comprehension expects if or a for clause, got b",
                "b",
            ),
            (
                "[x for y in z w for v in u]",
                "comprehension expects if or a for clause, got w",
                "w",
            ),
        ],
    )
    def test_refused_where_it_is_malformed(self, src, msg, at):
        err = assert_refused_at(src, msg, at)
        # The usage line follows the problem.
        assert ": " in err.msg[len(msg) :]

    @pytest.mark.parametrize(
        ("lispy", "python"),
        [
            ("(sub a)", "a"),
            ("(ife a b)", "b if a else None"),
            ("(ife a b c)", "b if a else c"),
            ("(or a)", "a"),
            ("(and a b)", "a and b"),
            ("(< a b)", "a < b"),
            ("(:= a 1)", "(a := 1)"),
            ("(lambda [a] 1)", "lambda a: 1"),
            ("(yield-from a)", "yield from a"),
            ("(sub x [: 1])", "x[:1]"),
            ("(sub x [: 1 2])", "x[1:2]"),
            ("(sub x [: 1 2 3])", "x[1:2:3]"),
            ("(sub x [: _ 2])", "x[:2]"),
            ("{1 2 **m 3 4}", "{1: 2, **m, 3: 4}"),
            ("(x for y in z if a if b)", "(x for y in z if a if b)"),
            ("(x for y z)", "(x for y in z)"),
            ("[x for y in z for w in y if w]", "[x for y in z for w in y if w]"),
            ("{k v for k in d if k}", "{k: v for k in d if k}"),
        ],
    )
    def test_well_formed_still_compiles(self, lispy, python):
        assert stmt_to_dump(lispy) == ast.dump(ast.parse(python))


class TestRepeatedKeywordArgument:
    """A keyword argument given twice is refused at the second, in CPython's
    words, compared as the Python name."""

    @pytest.mark.parametrize(
        ("src", "name", "at"),
        [
            ("(f :a 1 :a 2)", "a", ":a@2"),
            ("(f :a 1 **k :a 2)", "a", ":a@2"),
            ("(.m x :a 1 :a 2)", "a", ":a@2"),
            ("(f :a-b 1 :a_b 2)", "a_b", ":a_b"),
        ],
    )
    def test_refused_at_the_second(self, src, name, at):
        assert_refused_at(src, f"keyword argument repeated: {name}", at)

    def test_distinct_keywords_compile(self):
        assert stmt_to_dump("(f :a 1 **k :b 2)") == ast.dump(
            ast.parse("f(a=1, **k, b=2)")
        )


class TestNotCallableHead:
    """A call whose head compiles to a value CPython knows is not callable is
    refused at the head, as CPython's own compile-time check words it."""

    @pytest.mark.parametrize(
        ("src", "kind", "head"),
        [
            ("(True)", "bool", "True"),
            ("(None 1)", "NoneType", "None"),
            ("(1)", "int", "1"),
            ("(1.5)", "float", "1.5"),
            ("(1j)", "complex", "1j"),
            ('("s")', "str", '"s"'),
            ('(b"s")', "bytes", 'b"s"'),
            ('(f"s{x}")', "str", 'f"s{x}"'),
            ("([1] 2)", "list", "[1]"),
            ("([x for x in y])", "list", "[x for x in y]"),
            ("({} 1)", "dict", "{}"),
            ("({k v for k in d} 1)", "dict", "{k v for k in d}"),
            ("({, 1} 2)", "set", "{, 1}"),
            ("((, 1) 2)", "tuple", "(, 1)"),
            ("((x for x in y))", "generator", "(x for x in y)"),
            ("(...)", "ellipsis", "..."),
            ("(... x)", "ellipsis", "..."),
        ],
    )
    def test_refused_at_the_head(self, src, kind, head):
        assert_refused_at(src, f"'{kind}' object is not callable", head)

    @pytest.mark.parametrize(
        ("lispy", "python"),
        [
            ("(f 1)", "f(1)"),
            ("((. a b) 1)", "a.b(1)"),
            ("((sub fs 0) 1)", "fs[0](1)"),
            ("((lambda [x] x) 1)", "(lambda x: x)(1)"),
            ("((f) 1)", "f()(1)"),
        ],
    )
    def test_callable_heads_compile(self, lispy, python):
        assert stmt_to_dump(lispy) == ast.dump(ast.parse(python))
