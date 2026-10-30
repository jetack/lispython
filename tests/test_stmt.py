import ast
import enum

import pytest

from lispy.core.errors import LispyCompileError
from lispy.tools import macroexpand_then_compile, parse, src_to_python_org

from .utils import stmt_to_dump


class TestMatchMethods:
    def test_match_value(self):
        assert stmt_to_dump("""
(match x
  (case "Relevant" ...))""") == ast.dump(ast.parse("""
match x:
    case "Relevant":
        ..."""))

    def test_match_singleton(self):
        assert stmt_to_dump("""
(match x
  (case None
     ...))""") == ast.dump(ast.parse("""
match x:
    case None:
        ..."""))

    def test_match_sequence(self):
        assert stmt_to_dump("""
(match x
  (case [1 2]
     ...))""") == ast.dump(ast.parse("""
match x:
    case [1, 2]:
        ..."""))

    def test_match_star(self):
        assert stmt_to_dump("""
(match x
  (case [1 2 *rest]
     ...)
  (case [*_]
    ...))""") == ast.dump(ast.parse("""
match x:
    case [1, 2, *rest]:
        ...
    case [*_]:
        ..."""))

    def test_match_mapping(self):
        assert stmt_to_dump("""
(match x
  (case {1 _ 2 _}
    ...)
  (case {**rest}
    ...))""") == ast.dump(ast.parse("""
match x:
    case {1: _, 2: _}:
        ...
    case {**rest}:
        ..."""))

    def test_match_class(self):
        assert stmt_to_dump("""
(match x
  (case (Point2D 0 0)
    ...)
  (case (Point3D :x 0 :y 0 :z 0)
    ...))""") == ast.dump(ast.parse("""
match x:
    case Point2D(0, 0):
        ...
    case Point3D(x=0, y=0, z=0):
        ..."""))

    def test_match_as(self):
        assert stmt_to_dump("""
(match x
  (case [x] as y
    ...)
  (case _
    ...))""") == ast.dump(ast.parse("""
match x:
    case [x] as y:
        ...
    case _:
        ..."""))

    def test_match_or(self):
        assert stmt_to_dump("""
(match x
  (case (| [x] y)
    ...))""") == ast.dump(ast.parse("""
match x:
    case [x] | (y):
        ..."""))



class TestSignedLiteralPatterns:
    """Signed and complex literals build the AST ast.parse builds for a case."""

    @pytest.mark.parametrize(
        ["lpy", "py"],
        [
            ("-1", "-1"),
            ("-1.5", "-1.5"),
            ("-1j", "-1j"),
            ("1+2j", "1 + 2j"),
            ("-1-2j", "-1 - 2j"),
            ("1e3-.34j", "1e3 - .34j"),
            ("1e-3j", "1e-3j"),
            ("[-1 2 *rest]", "[-1, 2, *rest]"),
            ("{-1 _ 1+2j _}", "{-1: _, 1 + 2j: _}"),
            ("(Point -1 :y -2.5)", "Point(-1, y=-2.5)"),
            ("(| -1 0 1)", "-1 | 0 | 1"),
            ("(- 1)", "-1"),
            ("0x1e-2j", "0x1e - 2j"),
            ("-0x1e+2j", "-0x1e + 2j"),
            ("1E+5-2J", "1E+5 - 2J"),
            ("-0o7-1j", "-0o7 - 1j"),
            ("0b11", "0b11"),
            ("1_000j", "1_000j"),
            ("1_0e-2j", "1_0e-2j"),
        ],
    )
    def test_same_ast_as_python(self, lpy, py):
        assert stmt_to_dump(f"(match x (case {lpy} ...))") == ast.dump(
            ast.parse(f"match x:\n    case {py}:\n        ...")
        )

    def test_runtime(self):
        """Runs the AST lispython built, not a re-parse of its unparsed text."""
        src = """
(def classify [n]
  (match n
    (case -1 (return "minus one"))
    (case -1-2j (return "complex"))
    (case -0x1e+2j (return "hex complex"))
    (case [-2.5 *_] (return "list"))
    (case _ (return "other"))))
"""
        scope = {}
        module = ast.Module(macroexpand_then_compile(parse(src), {}), [])
        exec(compile(module, "<lpy>", "exec"), scope)
        classify = scope["classify"]
        assert classify(-1) == "minus one"
        assert classify(complex(-1, -2)) == "complex"
        assert classify(complex(-30, 2)) == "hex complex"
        assert classify(complex(30, 2)) == "other"
        assert classify([-2.5, 0]) == "list"
        assert classify(1) == "other"

    @pytest.mark.parametrize(
        ["pattern", "msg"],
        [
            ("1j+2j", "real number required in complex literal"),
            ("-1j-2j", "real number required in complex literal"),
            ("(- 1+2j)", "only a number literal can be signed in a match pattern"),
        ],
    )
    def test_complex_literal_rules(self, pattern, msg):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump(f"(match x\n  (case {pattern} ...))")
        assert info.value.msg == msg
        assert (info.value.lineno, info.value.offset) == (2, 9)


SEQ = "sequence pattern"
MAP = "mapping pattern"
CLS = "class pattern"


class TestMalformedPatterns:
    """A malformed case pattern is a LispyCompileError at the sub-form at
    fault, never a raw IndexError/ValueError/AttributeError from building the
    AST, nor an AST that compiles to something else."""

    @pytest.mark.parametrize(
        ("pattern", "msg", "at"),
        [
            ("(P :x)", f"{CLS} has :x without a pattern after it", ":x"),
            ("(P 1 :x)", f"{CLS} has :x without a pattern after it", ":x"),
            ("(P :x :y 1)", f"{CLS} has :x without a pattern after it", ":x"),
            ("(P :x 1 2)", f"{CLS} has 2 after a keyword pattern", "2"),
            ("(P :x.y 1)", f"{CLS} expects an attribute name after :, got x.y", ":x.y"),
            ("(P :if 1)", f"{CLS} expects an attribute name after :, got if", ":if"),
            ("(P :1 a)", f"{CLS} expects an attribute name after :, got 1", ":1"),
            ("(1)", f"{CLS} needs a class name, got 1", "1"),
            ("((f) 1)", f"{CLS} needs a class name, got (f)", "(f)"),
            ('("s" 1)', f'{CLS} needs a class name, got "s"', '"s"'),
            ("(:x 1)", f"{CLS} needs a class name, got :x", ":x"),
            ('{"a"}', f'{MAP} has key "a" without a pattern after it', '"a"'),
            ("{1 2 3}", f"{MAP} has key 3 without a pattern after it", "3"),
            ("{**r 1}", f"{MAP} has **r before its end", "**r"),
            ('{**r "a" 1}', f"{MAP} has **r before its end", "**r"),
            ("{**r **s}", f"{MAP} has **r before its end", "**r"),
            ("{**_}", f"{MAP} expects a name after **, got _", "**_"),
            ("{**1}", f"{MAP} expects a name after **, got 1", "**1"),
            ("{**x.y}", f"{MAP} expects a name after **, got x.y", "**x.y"),
            ("{a 1}", f"{MAP} expects a literal or dotted name as a key, got a", "a@3"),
            ("{(f) 1}", f"{MAP} expects a literal or dotted name as a key", "(f)"),
            ("{... 1}", f"{MAP} expects a literal or dotted name as a key", "..."),
            ("[*a *b]", f"{SEQ} has more than one *rest, got *b", "*b"),
            ("[*a b *c]", f"{SEQ} has more than one *rest, got *c", "*c"),
            ("[*(f)]", f"{SEQ} expects a name after *, got (f)", "*(f)"),
            ("[*x.y]", f"{SEQ} expects a name after *, got x.y", "*x.y"),
            ("[*if]", f"{SEQ} expects a name after *, got if", "*if"),
            ("[*1]", f"{SEQ} expects a name after *, got 1", "*1"),
            ("*a", "pattern can be *rest only in a sequence pattern", "*a"),
            ("*1", "pattern can be *rest only in a sequence pattern", "*1"),
            ("(| *a 1)", "pattern can be *rest only in a sequence pattern", "*a"),
            ("{1 *a}", "pattern can be *rest only in a sequence pattern", "*a"),
            ("(P *a)", "pattern can be *rest only in a sequence pattern", "*a"),
            ("(P :x *a)", "pattern can be *rest only in a sequence pattern", "*a"),
            ("(P **k)", "pattern can be **rest only last in a mapping", "**k"),
            ("[**k]", "pattern can be **rest only last in a mapping", "**k"),
            ("(|)", "(| ...) needs at least two patterns", "(|)"),
            ("(| 1)", "(| ...) needs at least two patterns", "(| 1)"),
            (":x", "pattern expected, got :x", ":x"),
            ("[:x]", "pattern expected, got :x", ":x"),
            ("...", "pattern expected, got ...", "..."),
            ("(+ 1 2)", "pattern expected, got (+ 1 2)", "(+ 1 2)"),
            ("(, a b)", "pattern expected, got (, a b)", "(, a b)"),
            ("(. a b)", "pattern expected, got (. a b)", "(. a b)"),
            ("(~ 1)", "pattern expected, got (~ 1)", "(~ 1)"),
            ('f"x"', "pattern expected, got (f-string", 'f"x"'),
            ("[(+ 1 2)]", "pattern expected, got (+ 1 2)", "(+ 1 2)"),
        ],
    )
    def test_refused_where_it_is_malformed(self, pattern, msg, at):
        form = f"(match x (case {pattern} 1))"
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump("(do\n  " + form + ")")
        assert info.value.msg.startswith(msg)
        assert ": " in info.value.msg[len(msg) :]
        assert_at(info.value, form, at)

    @pytest.mark.parametrize(
        ("pattern", "at"),
        [("if", "if"), ("[if]", "if"), ("a.if", "a.if"), ("(a.if 1)", "a.if")],
    )
    def test_keyword_name_refused_at_itself(self, pattern, at):
        form = f"(match x (case {pattern} 1))"
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump("(do\n  " + form + ")")
        assert info.value.msg == "if is a Python keyword and cannot be used as a name"
        assert_at(info.value, form, at)

    def test_message_quotes_the_pattern_usage(self):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump("(match x (case (P :x) 1))")
        assert info.value.msg == (
            "class pattern has :x without a pattern after it: "
            "(Class pattern... :attr pattern...), positional patterns first"
        )

    @pytest.mark.parametrize(
        ["lpy", "py"],
        [
            ("a.b", "a.b"),
            ("a.b.c", "a.b.c"),
            ("(P :x 1)", "P(x=1)"),
            ("(P a :x 1)", "P(a, x=1)"),
            ("(P a :x 1 :y [b])", "P(a, x=1, y=[b])"),
            ("(a.b 1)", "a.b(1)"),
            ("(P)", "P()"),
            ("[*a b]", "[*a, b]"),
            ("[*_]", "[*_]"),
            ('{"a" 1 **r}', '{"a": 1, **r}'),
            ("{x.y 1}", "{x.y: 1}"),
            ("{None 1 True 2 -1 3}", "{None: 1, True: 2, -1: 3}"),
            ("(| 1 2)", "1 | 2"),
            ("(| a.b None)", "a.b | None"),
            ('b"a"', 'b"a"'),
            ("[a [b *c]]", "[a, [b, *c]]"),
        ],
    )
    def test_same_ast_as_python(self, lpy, py):
        assert stmt_to_dump(f"(match x (case {lpy} ...))") == ast.dump(
            ast.parse(f"match x:\n    case {py}:\n        ...")
        )

    def test_dotted_name_is_a_value_not_a_capture(self):
        """``Color.RED`` compares against the member, as in Python; it is no
        capture pattern that matches anything."""
        src = """
(def name-of [c]
  (match c
    (case Color.RED (return "red"))
    (case _ (return "other"))))
"""
        scope = {"Color": enum.Enum("Color", ["RED", "GREEN"])}
        module = ast.Module(macroexpand_then_compile(parse(src), {}), [])
        exec(compile(module, "<lpy>", "exec"), scope)
        name_of = scope["name_of"]
        assert name_of(scope["Color"].RED) == "red"
        assert name_of(scope["Color"].GREEN) == "other"
        assert name_of(1) == "other"


class TestSignedNumberExpressions:
    """In expressions a signed complex token has the value Python gives it."""

    @pytest.mark.parametrize(
        ["lpy", "value"],
        [
            ("-1+2j", -1 + 2j),
            ("-0x1e+2j", -0x1E + 2j),
            ("--1+2j", --1 + 2j),
            ("1e3-.34j", 1e3 - 0.34j),
            ("0x1e-2j", 0x1E - 2j),
            ("-1", -1),
            ("0b11", 3),
            ("1_000", 1000),
        ],
    )
    def test_value(self, lpy, value):
        [stmt] = macroexpand_then_compile(parse(lpy), {})
        result = eval(compile(ast.Expression(stmt.value), "<lpy>", "eval"))
        assert result == value and type(result) is type(value)

    def test_signed_complex_prints_like_python(self):
        assert src_to_python_org("-1+2j") == "(-1+2j)"

    @pytest.mark.parametrize("pattern", ["(- x)", "(- True)", "(- (- 1))"])
    def test_only_numbers_are_signed(self, pattern):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump(f"(match x\n  (case {pattern} ...))")
        assert info.value.msg == (
            "only a number literal can be signed in a match pattern"
        )
        assert (info.value.lineno, info.value.offset) == (2, 9)


class TestImportNames:
    """Every import form checks its names in one place."""

    @pytest.mark.parametrize(
        ["lpy", "py"],
        [
            ("(from . [a])", "from . import a"),
            ("(from .. [a b as c])", "from .. import a, b as c"),
            ("(from .m [a])", "from .m import a"),
            ("(import os.path as p)", "import os.path as p"),
        ],
    )
    def test_same_ast_as_python(self, lpy, py):
        assert stmt_to_dump(lpy) == ast.dump(ast.parse(py))

    @pytest.mark.parametrize(
        ["lpy", "msg", "at"],
        [
            ('(import "x")', '(import ...) expects a name, got "x"', '"'),
            ("(import)", "(import ...) needs at least one module name", "("),
            ("(import os as a.b)", "(import ...) expects a name, got a.b", "a."),
            ("(from json)", "(from ...) takes module and [names] pairs", "("),
            (
                '(from json "dumps")',
                '(from ...) expects [names] or * after a module, got "dumps"',
                '"',
            ),
            ("(from json [1])", "(from ...) expects a name, got 1", "1"),
            ('(from "json" [a])', '(from ...) expects a name, got "json"', '"'),
            (
                "(import json[x])",
                "json[...] is only valid as a def/class/type name; "
                "use (sub json ...)",
                "j",
            ),
        ],
    )
    def test_refused(self, lpy, msg, at):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump(lpy)
        assert info.value.msg == msg
        assert (info.value.lineno, info.value.offset) == (1, lpy.index(at) + 1)


class TestEmptyLiteralOperands:
    """An empty [] or {} is an Expression of length 0, hence falsy: every form
    with an optional operand must test presence by length, not truthiness."""

    @pytest.mark.parametrize(
        ("lispy", "python"),
        [
            ("(return [])", "return []"),
            ("(return {})", "return {}"),
            ("(return)", "return"),
            ("(return (list))", "return list()"),
            ("(yield [])", "yield []"),
            ("(yield {})", "yield {}"),
            ("(yield)", "yield"),
            ("(raise [])", "raise []"),
            ("(assert x [])", "assert x, []"),
            ("(assert x {})", "assert x, {}"),
            ("(= x [])", "x = []"),
            ("(lambda [:a []] [])", "lambda a=[]: []"),
        ],
    )
    def test_matches_ast_parse(self, lispy, python):
        assert stmt_to_dump(lispy) == ast.dump(ast.parse(python))

    def test_return_takes_one_operand(self):
        with pytest.raises(LispyCompileError):
            stmt_to_dump("(return [] [])")

    @pytest.mark.parametrize(
        ("lispy", "python"),
        [
            ("(def f [:a [] *args :b {}] ...)", "def f(a=[], *args, b={}): ..."),
            ("(def f [:a []] (return []))", "def f(a=[]):\n    return []"),
        ],
    )
    def test_def_matches_ast_parse(self, lispy, python):
        # Compared piecewise: the FunctionDef's own fields (type_params)
        # differ by version in dumps.
        [got] = macroexpand_then_compile(parse(lispy))
        [want] = ast.parse(python).body
        assert ast.dump(got.args) == ast.dump(want.args)
        assert ast.dump(ast.Module(got.body, [])) == ast.dump(ast.Module(want.body, []))

    def test_empty_mapping_pattern(self):
        assert stmt_to_dump("(match x (case {} ...))") == ast.dump(
            ast.parse("match x:\n    case {}: ...")
        )


class TestLoopArity:
    """A loop with no target/iterable/test or no body is a LispyCompileError
    at the loop form, not an IndexError or an AST compile() rejects."""

    @pytest.mark.parametrize(
        ("loop", "msg"),
        [
            ("(for [x [1]] (print x))", "(for ...) needs a body"),
            ("(for x in xs (else 1))", "(for ...) needs a body"),
            ("(for x in xs)", "(for ...) needs a body"),
            ("(for x in)", "(for ...) is missing its iterable"),
            ("(for x)", "(for ...) is missing its iterable"),
            ("(for)", "(for ...) is missing its target"),
            ("(while x)", "(while ...) needs a body"),
            ("(while x (else 1))", "(while ...) needs a body"),
            ("(while)", "(while ...) is missing its test"),
            ("(async-for x in xs)", "(async-for ...) needs a body"),
        ],
    )
    def test_refused_at_the_loop(self, loop, msg):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump("(do\n  " + loop + ")")
        assert info.value.msg.startswith(msg + ": ")
        assert (info.value.lineno, info.value.offset) == (2, 3)
        assert (info.value.end_lineno, info.value.end_offset) == (2, 3 + len(loop))

    @pytest.mark.parametrize(
        ("lispy", "python"),
        [
            ("(for x in xs (f x))", "for x in xs:\n    f(x)"),
            ("(for x xs (f x) (g))", "for x in xs:\n    f(x)\n    g()"),
            (
                "(for x in xs (f x) (else (g)))",
                "for x in xs:\n    f(x)\nelse:\n    g()",
            ),
            ("(while x (f))", "while x:\n    f()"),
            ("(while x (f) (else (g)))", "while x:\n    f()\nelse:\n    g()"),
        ],
    )
    def test_well_formed_loops_compile(self, lispy, python):
        assert stmt_to_dump(lispy) == ast.dump(ast.parse(python))


def assert_at(err, form, at):
    """``err`` spans ``at`` (``text@n``: its n-th occurrence) in ``form``,
    written on line 2 at column 3."""
    [text, _, nth] = (form if at is None else at).partition("@")
    index = -1
    for _ in range(int(nth or 1)):
        index = form.index(text, index + 1)
    col = 3 + index
    assert (err.lineno, err.offset) == (2, col)
    assert (err.end_lineno, err.end_offset) == (2, col + len(text))


class TestFormArity:
    """A malformed statement form is a LispyCompileError naming the problem
    and the form's usage, never an IndexError, AssertionError or an AST that
    compile() rejects; it is positioned at the part at fault (``at``: the
    form, a clause or an operand)."""

    @pytest.mark.parametrize(
        ("form", "msg", "at"),
        [
            ("(if)", "(if ...) is missing its test", None),
            ("(if x)", "(if ...) is missing its then", None),
            ("(def)", "(def ...) is missing its name", None),
            ("(def f)", "(def ...) is missing its args", None),
            ("(async-def f)", "(async-def ...) is missing its args", None),
            ("(def f x 1)", "(def ...) expects [args] after its name, got x", "x"),
            ("(class)", "(class ...) is missing its name", None),
            ("(class C)", "(class ...) is missing its bases", None),
            ("(class C x)", "(class ...) expects [bases] after its name", "x"),
            ("(with)", "(with ...) is missing its items", None),
            ("(with [a])", "(with ...) needs a body", None),
            ("(async-with [a])", "(async-with ...) needs a body", None),
            ("(with x 1)", "(with ...) expects a non-empty [context ...]", "x"),
            ("(with [] 1)", "(with ...) expects a non-empty [context ...]", "[]"),
            ("(with [as b] 1)", "(with ...) has as without a context", "as"),
            ("(with [a as] 1)", "(with ...) has as without a target", "as"),
            ("(try)", "(try ...) needs a body", None),
            ("(try (except))", "(try ...) needs a body", None),
            ("(try (f))", "(try ...) needs an except or a finally clause", None),
            ("(try (f) (except [E]))", "(except ...) needs a body", "(except [E])"),
            ("(try (f) (finally))", "(finally ...) needs a body", "(finally)"),
            (
                "(try (f) (except [E] 1) (else))",
                "(else ...) needs a body",
                "(else)",
            ),
            (
                "(try (f) (except* [E] 1) (except [E] 2))",
                "(try ...) cannot mix except and except*",
                "(except [E] 2)",
            ),
            (
                "(try (f) (except [E] 2) (except* [E] 1))",
                "(try ...) cannot mix except and except*",
                "(except* [E] 1)",
            ),
            (
                "(try (f) (except 1) (except [E] 2))",
                "(try ...) has an except after a bare except",
                "(except [E] 2)",
            ),
            ("(try (f) (except* 1))", "(except* ...) needs a [types]", "(except* 1)"),
            ("(try (f) (except [] 1))", "(except ...) has an empty [types]", "[]"),
            ("(try (f) (except [as e] 1))", "(except ...) has as without a type", "as"),
            ("(try (f) (except [E as] 1))", "(except ...) has as without a name", "as"),
            ("(try (f) (except [E as e f] 1))", "(except ...) expects one name", "as"),
            (
                "(try (f) (else 1))",
                "(try ...) has an else without an except",
                "(else 1)",
            ),
            (
                "(try (f) (finally 1) (except [E] 1))",
                "(try ...) has except after finally",
                "(except [E] 1)",
            ),
            (
                "(try (f) (except [E] 1) (else 2) (except [F] 3))",
                "(try ...) has except after else",
                "(except [F] 3)",
            ),
            (
                "(try (f) (except [E] 1) (finally 2) (else 3))",
                "(try ...) has else after finally",
                "(else 3)",
            ),
            (
                "(try (f) (finally 1) (finally 2))",
                "(try ...) has more than one finally",
                "(finally 2)",
            ),
            (
                "(try (f) (except [E] 1) (g))",
                "(try ...) has (g) after its clauses",
                "(g)",
            ),
            ("(match)", "(match ...) is missing its subject", None),
            ("(match x)", "(match ...) needs at least one case", None),
            ("(match x (case))", "(case ...) is missing its pattern", "(case)"),
            ("(match x (case 1))", "(case ...) needs a body", "(case 1)"),
            ("(match x (case 1 as))", "(case ...) has as without a", "(case 1 as)"),
            ("(match x (case 1 as (f) 1))", "(case ...) expects a name", "(f)"),
            ("(match x (case 1 if))", "(case ...) has if without a", "(case 1 if)"),
            ("(match x (foo 1))", "(match ...) expects (case ...) clauses", "(foo 1)"),
            ("(deco)", "(deco ...) is missing its decorator", None),
            ("(deco d)", "(deco ...) is missing its definition", None),
            ("(deco d (f))", "(deco ...) expects a def, async-def, class or", "(f)"),
            (
                "(deco d (def f [] 1) 2)",
                "(deco ...) takes one decorator and one definition",
                None,
            ),
            ("(raise a b)", "(raise ...) takes an exception and an optional", None),
            ("(raise a from)", "(raise ...) is missing its cause", None),
            ("(raise a from b c)", "(raise ...) takes an exception and an", None),
            ("(assert)", "(assert ...) is missing its test", None),
            ("(assert a b c)", "(assert ...) takes a test and at most one", None),
            ("(=)", "(= ...) is missing its target", None),
            ("(= x)", "(= ...) is missing its value", None),
            ("(= x ^int 1 2)", "(= ...) takes one value after an annotation", None),
            ("(+=)", "(+= ...) is missing its target", None),
            ("(+= x)", "(+= ...) is missing its value", None),
            ("(//= x)", "(//= ...) is missing its value", None),
            ("(del)", "(del ...) needs at least one target", None),
            ("(global)", "(global ...) needs at least one name", None),
            ("(nonlocal)", "(nonlocal ...) needs at least one name", None),
            ("(global a.b)", "(global ...) expects a name, got a.b", "a.b"),
            ("(nonlocal (f))", "(nonlocal ...) expects a name, got (f)", "(f)"),
            ("(global if)", "(global ...) expects a name, got if", "if"),
            ("(def f (x) 1)", "(def ...) expects [args] after its name", "(x)"),
            ("(class C (B))", "(class ...) expects [bases] after its name", "(B)"),
            ("(def [] [] 1)", "(def ...) expects a name, got []", "[]"),
            ("(def if [] 1)", "(def ...) expects a name, got if", "if"),
            ("(class 1 [])", "(class ...) expects a name, got 1", "1"),
            ("(= ^int 1)", "(= ...) takes an annotation only right after", "^int"),
            ("(= a b ^int 1)", "(= ...) takes an annotation only right after", "^int"),
            ("(= x ^int ^str)", "(= ...) takes an annotation only right after", "^str"),
            (
                "(match x (case 1 if g as y 3))",
                "(case ...) has as after its guard",
                "as@2",
            ),
            ("(match x (case 1 as as 2))", "(case ...) expects a name", "as@3"),
            ("(try (f) (except [E as as] 1))", "(except ...) expects one name", "as"),
        ],
    )
    def test_refused_where_it_is_malformed(self, form, msg, at):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump("(do\n  " + form + ")")
        assert info.value.msg.startswith(msg)
        assert ": (" in info.value.msg[len(msg) :]
        assert_at(info.value, form, at)

    @pytest.mark.parametrize(
        ("form", "msg", "at"),
        [
            ("(= 1 2)", "cannot assign to literal", "1"),
            ("(= (f) 1)", "cannot assign to function call", "(f)"),
            ("(= (+ a b) 1)", "cannot assign to expression", "(+ a b)"),
            ("(= a (f) 1)", "cannot assign to function call", "(f)"),
            ("(= *a 1)", "starred assignment target must be in a list or tuple", "*a"),
            ("(= [*a *b] 1)", "multiple starred expressions in assignment", "*b"),
            ("(= [a 1] 2)", "cannot assign to literal", "1"),
            (
                "(= (, a b) ^int 1)",
                "only single target (not tuple) can be annotated",
                "(, a b)",
            ),
            ("(= (f) ^int 1)", "illegal target for annotation", "(f)"),
            ("(+= 1 2)", "'literal' is an illegal expression for augmented", "1"),
            ("(+= [] 1)", "'list' is an illegal expression for augmented", "[]"),
            ("(del 1)", "cannot delete literal", "1"),
            ("(del *a)", "cannot delete starred", "*a"),
            ("(for 1 in xs 2)", "cannot assign to literal", "1"),
            ("(with [a as 1] 2)", "cannot assign to literal", "1"),
            ("(raise from)", "from is a Python keyword and cannot be used as", "from"),
            ("(assert as)", "as is a Python keyword and cannot be used as", "as@2"),
            ("(= a.if 1)", "if is a Python keyword and cannot be used as", "a.if"),
            ("(match x (case 1 if if 2))", "if is a Python keyword", "if@2"),
            ("()", "() is an empty form; use (,) for an empty tuple", "()"),
            ("(if x ())", "() is an empty form", "()"),
            ("(if () 1)", "() is an empty form", "()"),
            ("(f ())", "() is an empty form", "()"),
            ("(try () (except 1))", "() is an empty form", "()"),
            ("(match x (case () 1))", "() is an empty form", "()"),
            ("(match x (case [()] 1))", "() is an empty form", "()"),
            ("(match x (case (| () 1) 2))", "() is an empty form", "()"),
            ("(match x (case {1 ()} 2))", "() is an empty form", "()"),
        ],
    )
    def test_bad_target_name_or_empty_form_refused_at_itself(self, form, msg, at):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump("(do\n  " + form + ")")
        assert info.value.msg.startswith(msg)
        assert_at(info.value, form, at)

    @pytest.mark.parametrize(
        ("lispy", "python"),
        [
            ("(raise)", "raise"),
            ("(raise e)", "raise e"),
            ("(raise e from c)", "raise e from c"),
            ('(assert a "m")', 'assert a, "m"'),
            ("(if x (f))", "if x:\n    f()"),
            ("(if x (f) (g) (h))", "if x:\n    f()\nelse:\n    g()\n    h()"),
            (
                "(try (f) (except* [E as e] (g)) (else (h)) (finally (k)))",
                "try:\n    f()\nexcept* E as e:\n    g()\n"
                "else:\n    h()\nfinally:\n    k()",
            ),
            (
                "(try (f) (except [A B as e] (g)) (except (h)))",
                "try:\n    f()\nexcept (A, B) as e:\n    g()\nexcept:\n    h()",
            ),
            ("(try (f) (finally (g)))", "try:\n    f()\nfinally:\n    g()"),
            (
                "(match x (case [a] as b if (> b 1) (f)) (case _ (g)))",
                "match x:\n    case [a] as b if b > 1:\n        f()\n"
                "    case _:\n        g()",
            ),
            ("(with [a as b c] (f))", "with a as b, c:\n    f()"),
            ("(= x ^int)", "x: int"),
            ("(= x ^int 1)", "x: int = 1"),
            ("(= a b 1)", "a = b = 1"),
            ("(+= x 1 2)", "x += 1 + 2"),
            ("(del a b)", "del a, b"),
            ("(global a b)", "global a, b"),
            ("(= x (,))", "x = ()"),
            ("(= (, a *b) 1)", "a, *b = 1"),
            ("(= [a (. b c) (sub d 0)] 1)", "[a, b.c, d[0]] = 1"),
            ("(del (, a b))", "del (a, b)"),
            ("(for (, a b) in xs (f))", "for a, b in xs:\n    f()"),
            ("(with [a as (, b c)] (f))", "with a as (b, c):\n    f()"),
            ("(+= (. a b) 1)", "a.b += 1"),
        ],
    )
    def test_well_formed_forms_compile(self, lispy, python):
        assert stmt_to_dump(lispy) == ast.dump(ast.parse(python))

    def test_deco_stacks(self):
        # Compared piecewise: FunctionDef dumps differ by version (type_params).
        [got] = macroexpand_then_compile(
            parse("(deco [a b] (deco c (def f [] (g))))")
        )
        [want] = ast.parse("@a\n@b\n@c\ndef f():\n    g()").body
        assert ast.dump(ast.Module(got.decorator_list, [])) == ast.dump(
            ast.Module(want.decorator_list, [])
        )
        assert ast.dump(ast.Module(got.body, [])) == ast.dump(ast.Module(want.body, []))
