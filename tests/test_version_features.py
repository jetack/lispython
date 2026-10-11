"""Version-gated forms: one feature table, one error type, every entry point.

``lispy.core.compiler.features.running_version`` is the only place the
compiler reads the interpreter version, so refusal is tested on every
interpreter by monkeypatching it below a feature's minimum.
"""

import ast
import builtins
import importlib
import re
import sys

import pytest

import lispy.tools as tools
from lispy.core.compiler import features
from lispy.core.errors import LispyCompileError, compile_errors, format_compile_error
from lispy.tools import macroexpand_then_compile, parse

OLD = (3, 11, 13)


@pytest.fixture
def old_python(monkeypatch):
    """Pretend the interpreter is 3.11.13."""
    monkeypatch.setattr(features, "running_version", lambda: OLD)


def compile_src(src):
    return macroexpand_then_compile(parse(src), {"__macro_namespace": {}})


def compile_expr(src):
    """Compile one lispython expression to an ``ast.Expression``."""
    [stmt] = compile_src(src)
    return ast.Expression(stmt.value)


def eval_lpy(src, scope):
    return eval(compile(compile_expr(src), "<lpy>", "eval"), dict(scope))


def refusal(src):
    with pytest.raises(LispyCompileError) as info:
        compile_src(src)
    return info.value


# A source whose second line holds a t-string, refused below 3.14.
GATED_SRC = '(= a 1)\n(print  t"hi {a}")\n'
GATED_MSG = 't-string t"..." needs Python 3.14+, running 3.11.13'
GATED_LINE, GATED_COL = 2, 9


# --------------------------------------------------------------------------
# The feature table and require()


class TestRequire:
    def test_table_entries(self):
        assert features.FEATURES["t-string"].min_version == (3, 14)
        assert features.FEATURES["comprehension-unpacking"].min_version == (3, 15)
        assert features.FEATURES["match-unary-plus"].min_version == (3, 15)

    def test_running_version_is_the_interpreter(self):
        assert features.running_version() == sys.version_info

    def test_new_enough_passes(self, monkeypatch):
        monkeypatch.setattr(features, "running_version", lambda: (3, 14, 0))
        sexp = parse("x")[0]
        assert features.require("t-string", sexp) is None

    def test_too_old_raises_at_the_sexp(self, old_python):
        sexp = parse("\n  (f x)")[0]
        with pytest.raises(LispyCompileError) as info:
            features.require("comprehension-unpacking", sexp)
        err = info.value
        assert isinstance(err, SyntaxError)
        assert err.msg == (
            "comprehension unpacking [*x for ...] needs Python 3.15+, running 3.11.13"
        )
        assert (err.lineno, err.offset) == (2, 3)
        assert (err.end_lineno, err.end_offset) == (2, 8)
        assert err.filename is None and err.text is None


class TestFormat:
    def test_path_line_col_source_caret(self):
        err = LispyCompileError("boom", parse("(a\n   (b c))")[0].list[1])
        err.filename = "x.lpy"
        err.text = "   (b c))"
        assert format_compile_error(err) == ("x.lpy:2:4: error: boom\n   (b c))\n   ^")

    def test_without_text(self):
        err = LispyCompileError("boom", parse("x")[0], filename="y.lpy")
        assert format_compile_error(err) == "y.lpy:1:1: error: boom"


# --------------------------------------------------------------------------
# Entry points: importer, lpy, l2py, REPL, nREPL, LSP

EXPECTED_REPORT = "{path}:{line}:{col}: error: {msg}\n{text}\n{caret}\n"


def expected_report(path):
    text = GATED_SRC.splitlines()[GATED_LINE - 1]
    return EXPECTED_REPORT.format(
        path=path,
        line=GATED_LINE,
        col=GATED_COL,
        msg=GATED_MSG,
        text=text,
        caret=" " * (GATED_COL - 1) + "^",
    )


@pytest.fixture
def lpy_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    path = tmp_path / "gated_mod.lpy"
    path.write_text(GATED_SRC, encoding="utf-8")
    return path


def run_cli(entry, argv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as info:
        entry()
    out, err = capsys.readouterr()
    return info.value.code, out, err


@pytest.mark.usefixtures("old_python")
class TestEntryPoints:
    def test_importer_fills_filename_and_text(self, lpy_file):
        sys.path.insert(0, str(lpy_file.parent))
        sys.modules.pop("gated_mod", None)
        with pytest.raises(LispyCompileError) as info:
            importlib.import_module("gated_mod")
        err = info.value
        assert err.filename == str(lpy_file)
        assert err.text == GATED_SRC.splitlines()[GATED_LINE - 1]
        assert (err.lineno, err.offset) == (GATED_LINE, GATED_COL)

    def test_lpy_file(self, lpy_file, monkeypatch, capsys):
        code, out, err = run_cli(tools.run, ["lpy", str(lpy_file)], monkeypatch, capsys)
        assert code == 1
        assert err == expected_report(str(lpy_file))
        assert "Traceback" not in out + err

    def test_lpy_module(self, lpy_file, monkeypatch, capsys):
        sys.path.insert(0, str(lpy_file.parent))
        sys.modules.pop("gated_mod", None)
        code, out, err = run_cli(
            tools.run, ["lpy", "-m", "gated-mod"], monkeypatch, capsys
        )
        assert code == 1
        assert err == expected_report(str(lpy_file))
        assert "Traceback" not in out + err

    def test_lpy_string(self, monkeypatch, capsys):
        code, out, err = run_cli(
            tools.run, ["lpy", "-c", GATED_SRC], monkeypatch, capsys
        )
        assert code == 1
        assert err == expected_report("<string>")
        assert "Traceback" not in out + err

    def test_l2py_file(self, lpy_file, monkeypatch, capsys):
        code, out, err = run_cli(
            tools.l2py, ["l2py", str(lpy_file)], monkeypatch, capsys
        )
        assert code == 1
        assert out == ""
        assert err == expected_report(str(lpy_file))
        assert "Traceback" not in err

    def test_l2py_string(self, monkeypatch, capsys):
        code, out, err = run_cli(
            tools.l2py, ["l2py", "-c", GATED_SRC], monkeypatch, capsys
        )
        assert code == 1
        assert out == ""
        assert err == expected_report("<string>")
        assert "Traceback" not in err

    def test_repl_reports_and_continues(self, monkeypatch, capsys):
        inputs = iter([GATED_SRC, "(print (+ 40 2))"])

        class Session:
            def prompt(self, _):
                try:
                    return next(inputs)
                except StopIteration:
                    raise EOFError from None

        monkeypatch.setattr(tools, "make_repl_session", Session)
        tools.repl()
        out, err = capsys.readouterr()
        assert err == expected_report("<repl>")
        assert "42" in out
        assert "Traceback" not in out + err

    def test_nrepl_eval(self):
        result = tools.nrepl_eval(GATED_SRC, {"__macro_namespace": {}})
        assert result["error"] == expected_report("<nrepl>").rstrip("\n")

    def test_lsp_reads_the_position(self):
        # lispy.lsp re-exports the server *object* as ``server``; take the module.
        server = importlib.import_module("lispy.lsp.server")
        diagnostics = server.compute_diagnostics(GATED_SRC, "file:///gated.lpy")
        [diag] = diagnostics
        assert diag.range.start.line == GATED_LINE - 1
        assert diag.range.start.character == GATED_COL - 1
        assert GATED_MSG in diag.message


# --------------------------------------------------------------------------
# t-strings (PEP 750, Python 3.14)

needs_314 = pytest.mark.skipif(sys.version_info < (3, 14), reason="t-strings: 3.14+")
before_314 = pytest.mark.skipif(sys.version_info >= (3, 14), reason="runs on <3.14")

# (lispython, Python) pairs that must build the same TemplateStr and Template.
T_STRING_CASES = [
    ('t"hi {x}"', 't"hi {x}"'),
    ('t"{ x }"', 't"{ x }"'),
    ('t"{x\t}"', 't"{x\t}"'),
    ('t"""{\n x \n}"""', 't"""{\n x \n}"""'),
    ('t"{x!r}"', 't"{x!r}"'),
    ('t"{ x !r}"', 't"{ x !r}"'),
    ('t"{x!s:}"', 't"{x!s:}"'),
    ('t"{x:>10}"', 't"{x:>10}"'),
    ('t"{x:}"', 't"{x:}"'),
    ('t"{ x !r : >10 }"', 't"{ x !r : >10 }"'),
    ('t"{x:{width}}"', 't"{x:{width}}"'),
    ('t"{x:{x}{x}}"', 't"{x:{x}{x}}"'),
    ('t"{x=}"', 't"{x=}"'),
    ('t"{ x = }"', 't"{ x = }"'),
    ('t"{x = !r}"', 't"{x = !r}"'),
    ('t"{x=!s:>5}"', 't"{x=!s:>5}"'),
    ('t"{x=:>5}"', 't"{x=:>5}"'),
    ('t"{{lit}} {x} }}"', 't"{{lit}} {x} }}"'),
    ('t""', 't""'),
    ('t"{x}{name}"', 't"{x}{name}"'),
    ('T"a\\nb{x}"', 'T"a\\nb{x}"'),
    ('t"\\N{BULLET} {x}"', 't"\\N{BULLET} {x}"'),
    ('rt"\\n{x}"', 'rt"\\n{x}"'),
    ('Rt"\\n{x}"', 'Rt"\\n{x}"'),
    ('tR"\\n{x}"', 'tR"\\n{x}"'),
    ('TR"\\{x}"', 'TR"\\{x}"'),
]
T_SCOPE = {"x": 42, "name": "bob", "width": 8}


def template_fields(template):
    return (
        template.strings,
        template.values,
        [
            (i.value, i.expression, i.conversion, i.format_spec)
            for i in template.interpolations
        ],
    )


@needs_314
class TestTStringSupported:
    @pytest.mark.parametrize(["lpy", "py"], T_STRING_CASES)
    def test_same_ast_as_python(self, lpy, py):
        assert ast.dump(compile_expr(lpy).body) == ast.dump(
            ast.parse(py, mode="eval").body
        )

    @pytest.mark.parametrize(["lpy", "py"], T_STRING_CASES)
    def test_same_template_as_python(self, lpy, py):
        from string.templatelib import Template

        ours = eval_lpy(lpy, T_SCOPE)
        assert isinstance(ours, Template)
        assert template_fields(ours) == template_fields(eval(py, dict(T_SCOPE)))

    def test_expression_is_the_lispython_text(self):
        template = eval_lpy('t"{(* 2 name)!r:>{width}} and { x = }"', T_SCOPE)
        assert template.strings == ("", " and  x = ", "")
        assert template.values == ("bobbob", 42)
        [first, second] = template.interpolations
        assert (first.expression, first.conversion, first.format_spec) == (
            "(* 2 name)",
            "r",
            ">8",
        )
        # Whitespace follows the running CPython (3.14 strips it, 3.15 keeps it).
        python = eval('t"{ x = }"', dict(T_SCOPE)).interpolations[0]
        assert (second.expression, second.conversion) == (python.expression, "r")

    def test_interpolations_are_not_evaluated_at_compile_time(self):
        # A plain-string path would eval() the literal inside the compiler.
        tree = compile_expr('t"{(undefined-at-compile-time)}"')
        assert isinstance(tree.body, ast.TemplateStr)

    @pytest.mark.parametrize("src", ['t"a{x!r:>{width}}b"', 't"{ x = }"'])
    def test_positions_match_cpython(self, src):
        [stmt] = compile_src(src)
        assert ast.dump(stmt, include_attributes=True) == ast.dump(
            ast.parse(src).body[0], include_attributes=True
        )

    def test_l2py_writes_python_text(self):
        """Python source gets the Python text; the compiled tree keeps lispython's."""
        [stmt] = compile_src('(= tpl t"Hi {(* 2 name)!r:>{width}} { {1 2} }")')
        pysrc = tools.ast_to_python(stmt)
        assert pysrc == "tpl = t'Hi {2 * name!r:>{width}} { {1: 2}}'"
        scope = dict(T_SCOPE)
        exec(pysrc, scope)
        assert [i.expression for i in scope["tpl"].interpolations] == [
            "2 * name",
            " {1: 2}",
        ]
        assert stmt.value.values[1].str == "(* 2 name)"


class TestTStringRefused:
    @pytest.mark.usefixtures("old_python")
    def test_refused_below_314(self):
        err = refusal('(= a 1)\n(print  t"hi {a}")')
        assert err.msg == 't-string t"..." needs Python 3.14+, running 3.11.13'
        assert (err.lineno, err.offset) == (2, 9)

    @before_314
    def test_refused_on_this_interpreter(self):
        err = refusal('(print  rt"hi {a}")')
        running = ".".join(map(str, sys.version_info[:3]))
        assert err.msg == f't-string t"..." needs Python 3.14+, running {running}'
        assert (err.lineno, err.offset) == (1, 9)

    @pytest.mark.parametrize(
        ["lpy", "msg", "at"],
        [
            ('ft"x"', "'f' and 't' prefixes are incompatible", "f"),
            ('tf"x"', "'f' and 't' prefixes are incompatible", "t"),
            ('bt"x"', "'b' and 't' prefixes are incompatible", "b"),
            ('Ut"x"', "'u' and 't' prefixes are incompatible", "U"),
            ('tt"x"', "invalid string prefix 'tt'", "t"),
            (
                't"{x!z}"',
                "t-string: invalid conversion character 'z': expected 's', 'r', or 'a'",
                "!",
            ),
            ('t"a{}"', "t-string: valid expression required before '}'", "{"),
            ('t"a}"', "t-string: single '}' is not allowed", "}"),
        ],
    )
    def test_malformed_refused_like_python(self, lpy, msg, at, monkeypatch):
        monkeypatch.setattr(features, "running_version", lambda: (3, 14, 0))
        err = refusal("\n   " + lpy)
        assert err.msg == msg
        assert (err.lineno, err.offset) == (2, 4 + lpy.index(at))

    @pytest.mark.usefixtures("old_python")
    @pytest.mark.parametrize("lpy", ['t"a{}"', 't"{x!z}"', 't"{x"', 'rt"a}"'])
    def test_version_reported_before_the_contents(self, lpy):
        err = refusal("\n   " + lpy)
        assert err.msg == 't-string t"..." needs Python 3.14+, running 3.11.13'
        assert (err.lineno, err.offset) == (2, 4)


# --------------------------------------------------------------------------
# Comprehension unpacking (PEP 798, Python 3.15)

needs_315 = pytest.mark.skipif(sys.version_info < (3, 15), reason="needs 3.15+")
before_315 = pytest.mark.skipif(sys.version_info >= (3, 15), reason="runs on <3.15")

UNPACK_MSG = "comprehension unpacking [*x for ...] needs Python 3.15+, running {}"
# (lispython, Python, value of the Python expression with UNPACK_SCOPE)
UNPACK_CASES = [
    ("[*L for L lists]", "[*L for L in lists]", [1, 2, 3]),
    ("[*L for L in lists if L]", "[*L for L in lists if L]", [1, 2, 3]),
    ("(list (*L for L lists))", "list((*L for L in lists))", [1, 2, 3]),
    ("{, *s for s sets}", "{*s for s in sets}", {1, 2, 3}),
    ("{**d for d dicts}", "{**d for d in dicts}", {"a": 1, "b": 3}),
    ("[*(f L) for L lists]", "[*f(L) for L in lists]", [1, 2, 3]),
]
UNPACK_SCOPE = {
    "lists": [[1, 2], [], [3]],
    "sets": [{1, 2}, {2, 3}],
    "dicts": [{"a": 1, "b": 2}, {"b": 3}],
    "f": list,
}
ASYNC_UNPACK_CASES = [
    ("[*L async-for L lists]", "[*L async for L in lists]"),
    ("(*L async-for L lists)", "(*L async for L in lists)"),
    ("{, *s async-for s sets}", "{*s async for s in sets}"),
    ("{**d async-for d dicts}", "{**d async for d in dicts}"),
]


@needs_315
class TestComprehensionUnpackingSupported:
    @pytest.mark.parametrize(["lpy", "py", "value"], UNPACK_CASES)
    def test_same_ast_and_value_as_python(self, lpy, py, value):
        assert ast.dump(compile_expr(lpy).body) == ast.dump(
            ast.parse(py, mode="eval").body
        )
        assert eval_lpy(lpy, UNPACK_SCOPE) == value == eval(py, dict(UNPACK_SCOPE))

    @pytest.mark.parametrize(["lpy", "py"], ASYNC_UNPACK_CASES)
    def test_async_same_ast_as_python(self, lpy, py):
        assert ast.dump(compile_expr(lpy).body) == ast.dump(
            ast.parse(py, mode="eval").body
        )

    def test_async_runs(self):
        import asyncio

        scope = {}
        src = """
(async-def agen [xs]
  (for x xs (yield x)))
(async-def flat [xs]
  (return [*L async-for L (agen xs)]))
"""
        module = ast.Module(compile_src(src), type_ignores=[])
        exec(compile(module, "<lpy>", "exec"), scope)
        assert asyncio.run(scope["flat"]([[1], [2, 3]])) == [1, 2, 3]


class TestComprehensionUnpackingRefused:
    FORMS = [
        "[*L for L lists]",
        "(*L for L lists)",
        "{, *s for s sets}",
        "{**d for d dicts}",
        "[*L async-for L lists]",
    ]

    @pytest.mark.usefixtures("old_python")
    @pytest.mark.parametrize("lpy", FORMS)
    def test_refused_below_315(self, lpy):
        # The error points at the unpacked element.
        err = refusal("(= x 1)\n" + lpy)
        assert err.msg == UNPACK_MSG.format("3.11.13")
        assert (err.lineno, err.offset) == (2, lpy.index("*") + 1)

    @before_315
    @pytest.mark.parametrize("lpy", FORMS)
    def test_refused_on_this_interpreter(self, lpy):
        err = refusal(lpy)
        assert err.msg == UNPACK_MSG.format(".".join(map(str, sys.version_info[:3])))
        assert (err.lineno, err.offset) == (1, lpy.index("*") + 1)

    @pytest.mark.usefixtures("old_python")
    def test_plain_comprehensions_still_compile(self):
        assert ast.unparse(compile_expr("[L for L lists]")) == "[L for L in lists]"
        assert ast.unparse(compile_expr("{k v for (, k v) items}")) == (
            "{k: v for k, v in items}"
        )
        assert ast.unparse(compile_expr("{, *s 1}")) == "{*s, 1}"


# --------------------------------------------------------------------------
# Unary plus in a match pattern (Python 3.15)

PLUS_MSG = "unary plus in a match pattern needs Python 3.15+, running {}"
PLUS_CASES = [
    ("+1", "+1"),
    ("+1.5", "+1.5"),
    ("+1j", "+1j"),
    ("+1+2j", "+1 + 2j"),
    ("[+1 -1]", "[+1, -1]"),
    ("{+1 _}", "{+1: _}"),
]


def match_src(pattern):
    return f"(match x\n  (case {pattern} ...))"


@needs_315
class TestMatchUnaryPlusSupported:
    @pytest.mark.parametrize(["lpy", "py"], PLUS_CASES)
    def test_same_ast_as_python(self, lpy, py):
        module = ast.Module(compile_src(match_src(lpy)), type_ignores=[])
        assert ast.dump(module) == ast.dump(
            ast.parse(f"match x:\n    case {py}:\n        ...")
        )

    def test_runtime(self):
        src = """
(def classify [n]
  (match n
    (case +1 (return "plus one"))
    (case +1+2j (return "complex"))
    (case _ (return "other"))))
"""
        scope = {}
        module = ast.Module(compile_src(src), type_ignores=[])
        exec(compile(module, "<lpy>", "exec"), scope)
        assert scope["classify"](1) == "plus one"
        assert scope["classify"](1 + 2j) == "complex"
        assert scope["classify"](-1) == "other"


class TestMatchUnaryPlusRefused:
    FORMS = ["+1", "+1.5", "+1+2j", "[0 +1]", "{+1 _}"]

    @pytest.mark.usefixtures("old_python")
    @pytest.mark.parametrize("pattern", FORMS)
    def test_refused_below_315(self, pattern):
        err = refusal(match_src(pattern))
        assert err.msg == PLUS_MSG.format("3.11.13")
        assert (err.lineno, err.offset) == (2, 9 + pattern.index("+"))

    @before_315
    @pytest.mark.parametrize("pattern", FORMS)
    def test_refused_on_this_interpreter(self, pattern):
        err = refusal(match_src(pattern))
        assert err.msg == PLUS_MSG.format(".".join(map(str, sys.version_info[:3])))
        assert (err.lineno, err.offset) == (2, 9 + pattern.index("+"))

    @pytest.mark.usefixtures("old_python")
    def test_minus_needs_no_new_python(self):
        module = ast.Module(compile_src(match_src("-1")), type_ignores=[])
        assert ast.unparse(module) == "match x:\n    case -1:\n        ..."


# --------------------------------------------------------------------------
# Type parameters on def/class (PEP 695: 3.12; defaults PEP 696: 3.13)

needs_312 = pytest.mark.skipif(sys.version_info < (3, 12), reason="needs 3.12+")
before_312 = pytest.mark.skipif(sys.version_info >= (3, 12), reason="runs on <3.12")
needs_313 = pytest.mark.skipif(sys.version_info < (3, 13), reason="needs 3.13+")
before_313 = pytest.mark.skipif(sys.version_info >= (3, 13), reason="runs on <3.13")

TYPE_PARAMS_MSG = "type parameters f[T] needs Python 3.12+, running {}"
DEFAULTS_MSG = "type parameter defaults f[:T int] needs Python 3.13+, running {}"
TYPE_PARAM_CASES = [
    ("(def f[T] [x ^T] ^T (return x))", "def f[T](x: T) -> T:\n    return x"),
    (
        "(def f[T ^int U ^(, int str) *Ts **P] [] (pass))",
        "def f[T: int, U: (int, str), *Ts, **P]():\n    pass",
    ),
    ("(async-def f[T] [x] (return x))", "async def f[T](x):\n    return x"),
    ("(class Box[T] [Base] (pass))", "class Box[T](Base):\n    pass"),
    ("(deco dec (def f[T] [x] (return x)))", "@dec\ndef f[T](x):\n    return x"),
    ("(def my-fn[my-T] [] (pass))", "def my_fn[my_T]():\n    pass"),
]
DEFAULT_CASES = [
    (
        "(def f[:T str :U ^int bool :*Ts *(sub tuple int) :**P [int str]] [] (pass))",
        "def f[T = str, U: int = bool, *Ts = *tuple[int], **P = [int, str]]():\n"
        "    pass",
    ),
    ("(class C[T :U int] [] (pass))", "class C[T, U = int]:\n    pass"),
]


def module_dump(src):
    return ast.dump(ast.Module(compile_src(src), type_ignores=[]))


def run_module(src, scope=None):
    scope = {} if scope is None else scope
    module = ast.Module(compile_src(src), type_ignores=[])
    exec(compile(module, "<lpy>", "exec"), scope)
    return scope


@needs_312
class TestTypeParamsSupported:
    @pytest.mark.parametrize(["lpy", "py"], TYPE_PARAM_CASES)
    def test_same_ast_as_python(self, lpy, py):
        assert module_dump(lpy) == ast.dump(ast.parse(py))

    def test_runtime(self):
        scope = run_module(
            "(def f[T ^int U ^(, int str) *Ts **P] [x ^T] ^T (return x))\n"
            "(class Box[T] [] (pass))"
        )
        [t, u, ts, p] = scope["f"].__type_params__
        assert (t.__name__, t.__bound__) == ("T", int)
        assert (u.__name__, u.__constraints__) == ("U", (int, str))
        assert [type(x).__name__ for x in (ts, p)] == ["TypeVarTuple", "ParamSpec"]
        assert scope["f"](3) == 3
        assert scope["f"].__annotations__["x"] is t
        assert [x.__name__ for x in scope["Box"].__type_params__] == ["T"]

    def test_l2py(self):
        assert tools.src_to_python_org("(class Box[T ^int] [] (pass))") == (
            "class Box[T: int]:\n    pass"
        )


@needs_313
class TestTypeParamDefaultsSupported:
    @pytest.mark.parametrize(["lpy", "py"], DEFAULT_CASES)
    def test_same_ast_as_python(self, lpy, py):
        assert module_dump(lpy) == ast.dump(ast.parse(py))

    def test_runtime(self):
        scope = run_module(DEFAULT_CASES[0][0])
        [t, u, ts, p] = scope["f"].__type_params__
        assert (t.__default__, u.__bound__, u.__default__) == (str, int, bool)
        assert ts.__default__ == next(iter(tuple[int]))
        assert p.__default__ == [int, str]


class TestTypeParamsRefused:
    FORMS = [
        "(def f[T] [x] x)",
        "(async-def f[T] [x] x)",
        "(class Box[T] [] (pass))",
        "(deco dec (def f[T] [x] x))",
    ]

    @pytest.mark.usefixtures("old_python")
    @pytest.mark.parametrize("lpy", FORMS)
    def test_refused_below_312(self, lpy):
        err = refusal("(= a 1)\n" + lpy)
        assert err.msg == TYPE_PARAMS_MSG.format("3.11.13")
        name_col = re.search(r"[\w-]+\[", lpy).start()
        assert (err.lineno, err.offset) == (2, name_col + 1)

    @before_312
    @pytest.mark.parametrize("lpy", FORMS)
    def test_refused_on_this_interpreter(self, lpy):
        err = refusal(lpy)
        running = ".".join(map(str, sys.version_info[:3]))
        assert err.msg == TYPE_PARAMS_MSG.format(running)

    def test_defaults_refused_below_313(self, monkeypatch):
        monkeypatch.setattr(features, "running_version", lambda: (3, 12, 9))
        err = refusal("(def f[:T str] [] (pass))")
        assert err.msg == DEFAULTS_MSG.format("3.12.9")
        assert (err.lineno, err.offset) == (1, 8)

    @before_313
    def test_defaults_refused_on_this_interpreter(self):
        err = refusal("(def f[:T str] [] (pass))")
        running = ".".join(map(str, sys.version_info[:3]))
        expected = (
            TYPE_PARAMS_MSG if sys.version_info < (3, 12) else DEFAULTS_MSG
        ).format(running)
        assert err.msg == expected

    @needs_312
    @pytest.mark.parametrize(
        ["lpy", "msg", "col"],
        [
            ("(def f[] [] (pass))", "Type parameter list cannot be empty", 6),
            ("(def f[*Ts ^int] [] (pass))", "cannot use bound with TypeVarTuple", 12),
            ("(def f[**P ^int] [] (pass))", "cannot use bound with ParamSpec", 12),
            ("(def f[a.b] [] (pass))", "invalid type parameter a.b", 8),
        ],
    )
    def test_malformed_like_python(self, lpy, msg, col):
        err = refusal(lpy)
        assert err.msg == msg
        assert (err.lineno, err.offset) == (1, col)


# --------------------------------------------------------------------------
# Type alias: `type` as a soft keyword (PEP 695, 3.12)

ALIAS_MSG = "type alias statement (type ...) needs Python 3.12+, running {}"
ALIAS_CASES = [
    ("(type X int)", "type X = int"),
    ("(type Vec[T] (sub list T))", "type Vec[T] = list[T]"),
    ("(type Num[T ^(, int float)] T)", "type Num[T: (int, float)] = T"),
    ("(type my-alias (| int None))", "type my_alias = int | None"),
    ("(if x (type X int))", "if x:\n    type X = int"),
]


@needs_312
class TestTypeAliasSupported:
    @pytest.mark.parametrize(["lpy", "py"], ALIAS_CASES)
    def test_same_ast_as_python(self, lpy, py):
        assert module_dump(lpy) == ast.dump(ast.parse(py))

    def test_runtime(self):
        scope = run_module("(type Vec[T] (sub list T))\n(type X (| int None))")
        vec = scope["Vec"]
        assert type(vec).__name__ == "TypeAliasType"
        assert [t.__name__ for t in vec.__type_params__] == ["T"]
        assert scope["X"].__value__ == int | None

    @needs_313
    def test_with_default(self):
        assert module_dump("(type Pair[:T int] (, T T))") == ast.dump(
            ast.parse("type Pair[T = int] = (T, T)")
        )


class TestTypeSoftKeyword:
    """Every other shape stays a call to the builtin, on every version."""

    @pytest.mark.parametrize(
        ["lpy", "py"],
        [
            ("(= t (type x))", "t = type(x)"),
            ("(type x)", "type(x)"),
            ('(type "N" (, object) {})', "type('N', (object,), {})"),
            ("(type name bases ns)", "type(name, bases, ns)"),
            (
                "(def f [a b] (return (type a b)))",
                "def f(a, b):\n    return type(a, b)",
            ),
            ("(print (type a b))", "print(type(a, b))"),
            ("(type x.y int)", "type(x.y, int)"),
            ('(type "N" int)', "type('N', int)"),
            ("(type (f) int)", "type(f(), int)"),
        ],
    )
    @pytest.mark.usefixtures("old_python")
    def test_call(self, lpy, py):
        assert tools.src_to_python_org(lpy) == py

    @pytest.mark.usefixtures("old_python")
    def test_call_runs(self):
        scope = run_module('(= N (type "N" (, object) {"a" 1}))\n(= t (type 1))')
        assert scope["N"].a == 1 and scope["t"] is int


class TestTypeAliasRefused:
    FORMS = ["(type X int)", "(type Vec[T] (sub list T))"]

    @pytest.mark.usefixtures("old_python")
    @pytest.mark.parametrize("lpy", FORMS)
    def test_refused_below_312(self, lpy):
        err = refusal("(= a 1)\n  " + lpy)
        assert err.msg == ALIAS_MSG.format("3.11.13")
        assert (err.lineno, err.offset) == (2, 3)

    @before_312
    @pytest.mark.parametrize("lpy", FORMS)
    def test_refused_on_this_interpreter(self, lpy):
        err = refusal(lpy)
        assert err.msg == ALIAS_MSG.format(".".join(map(str, sys.version_info[:3])))
        assert (err.lineno, err.offset) == (1, 1)


# --------------------------------------------------------------------------
# Lazy imports (PEP 810, 3.15)

LAZY_MSG = "lazy import (lazy-import ...) needs Python 3.15+, running {}"
LAZY_CASES = [
    ("(lazy-import json)", "lazy import json"),
    ("(lazy-import json as j os)", "lazy import json as j, os"),
    ("(lazy-import __future__)", "lazy import __future__"),
    ("(lazy-from json [dumps loads as l])", "lazy from json import dumps, loads as l"),
    (
        "(lazy-from json [dumps] os [path])",
        "lazy from json import dumps\nlazy from os import path",
    ),
    ("(lazy-from .mod [x])", "lazy from .mod import x"),
]


class TestPlainImportsMatchPython:
    """On 3.15 ast.parse marks plain imports is_lazy=0; earlier there is no field."""

    @pytest.mark.parametrize(
        ["lpy", "py"],
        [
            ("(import json)", "import json"),
            ("(import json as j os)", "import json as j, os"),
            ("(from json [dumps])", "from json import dumps"),
            ("(from json *)", "from json import *"),
        ],
    )
    def test_same_ast_as_python(self, lpy, py):
        assert module_dump(lpy) == ast.dump(ast.parse(py))


@needs_315
class TestLazyImportSupported:
    @pytest.mark.parametrize(["lpy", "py"], LAZY_CASES)
    def test_same_ast_as_python(self, lpy, py):
        assert module_dump(lpy) == ast.dump(ast.parse(py))

    def test_loads_on_first_use(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
        loaded = []
        monkeypatch.setattr(builtins, "LAZY_TEST_LOADED", loaded, raising=False)
        (tmp_path / "lazy_test_mod.py").write_text(
            "LAZY_TEST_LOADED.append(1)\nVALUE = 7\n"
        )
        monkeypatch.delitem(sys.modules, "lazy_test_mod", raising=False)
        scope = run_module(
            "(lazy-from lazy-test-mod [VALUE])\n"
            "(= before (len LAZY_TEST_LOADED))\n"
            "(= value VALUE)"
        )
        assert (scope["before"], scope["value"], loaded) == (0, 7, [1])


class TestLazyImportRefused:
    FORMS = ["(lazy-import json)", "(lazy-from json [dumps])"]

    @pytest.mark.usefixtures("old_python")
    @pytest.mark.parametrize("lpy", FORMS)
    def test_refused_below_315(self, lpy):
        err = refusal("(= a 1)\n  " + lpy)
        assert err.msg == LAZY_MSG.format("3.11.13")
        assert (err.lineno, err.offset) == (2, 3)

    @before_315
    @pytest.mark.parametrize("lpy", FORMS)
    def test_refused_on_this_interpreter(self, lpy):
        err = refusal(lpy)
        assert err.msg == LAZY_MSG.format(".".join(map(str, sys.version_info[:3])))
        assert (err.lineno, err.offset) == (1, 1)

    @pytest.mark.parametrize(
        ["lpy", "msg"],
        [
            ("(lazy-from json *)", "lazy from ... import * is not allowed"),
            (
                "(lazy-from os [path] json *)",
                "lazy from ... import * is not allowed",
            ),
            (
                "(lazy-from __future__ [annotations])",
                "lazy from __future__ import is not allowed",
            ),
        ],
    )
    def test_static_restrictions_like_python(self, lpy, msg, monkeypatch):
        monkeypatch.setattr(features, "running_version", lambda: (3, 15, 0))
        err = refusal("\n " + lpy)
        assert err.msg == msg
        assert (err.lineno, err.offset) == (2, 2)


# --------------------------------------------------------------------------
# SyntaxErrors from CPython's compiler on a .lpy module read like ours

RETURN_SRC = "(= a 1)\n  (return 1)\n"
SCOPED_LAZY_SRC = "(= a 1)\n(def f []\n  (lazy-import json))\n"


class TestCompilerSyntaxErrors:
    @pytest.fixture
    def write_lpy(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])

        def write(name, src):
            path = tmp_path / f"{name}.lpy"
            path.write_text(src, encoding="utf-8")
            monkeypatch.delitem(sys.modules, name, raising=False)
            return path

        return write

    def test_importer_converts_with_lpy_position(self, write_lpy):
        path = write_lpy("return_mod", RETURN_SRC)
        with pytest.raises(LispyCompileError) as info:
            importlib.import_module("return_mod")
        err = info.value
        assert err.msg == "'return' outside function"
        assert (err.filename, err.lineno, err.offset) == (str(path), 2, 3)
        assert err.text == "  (return 1)"

    def test_lpy_file(self, write_lpy, monkeypatch, capsys):
        path = write_lpy("return_script", RETURN_SRC)
        code, out, err = run_cli(tools.run, ["lpy", str(path)], monkeypatch, capsys)
        assert code == 1
        assert err == (
            f"{path}:2:3: error: 'return' outside function\n  (return 1)\n  ^\n"
        )
        assert "Traceback" not in out + err

    @needs_315
    def test_lazy_import_inside_def(self, write_lpy, monkeypatch, capsys):
        path = write_lpy("scoped_lazy", SCOPED_LAZY_SRC)
        code, out, err = run_cli(tools.run, ["lpy", str(path)], monkeypatch, capsys)
        assert code == 1
        assert err == (
            f"{path}:3:3: error: lazy import not allowed inside functions\n"
            "  (lazy-import json))\n  ^\n"
        )
        assert "Traceback" not in out + err

    def test_lpy_string(self, monkeypatch, capsys):
        code, out, err = run_cli(
            tools.run, ["lpy", "-c", RETURN_SRC], monkeypatch, capsys
        )
        assert code == 1
        assert err == (
            "<string>:2:3: error: 'return' outside function\n  (return 1)\n  ^\n"
        )
        assert "Traceback" not in out + err

    @needs_315
    def test_lpy_string_lazy_import_inside_def(self, monkeypatch, capsys):
        src = "(def f [] (lazy-import os))"
        code, out, err = run_cli(tools.run, ["lpy", "-c", src], monkeypatch, capsys)
        assert code == 1
        assert err == (
            "<string>:1:11: error: lazy import not allowed inside functions\n"
            f"{src}\n          ^\n"
        )

    def test_repl_reports_and_continues(self, monkeypatch, capsys):
        inputs = iter([RETURN_SRC, "(print (+ 40 2))"])

        class Session:
            def prompt(self, _):
                try:
                    return next(inputs)
                except StopIteration:
                    raise EOFError from None

        monkeypatch.setattr(sys, "path", list(sys.path))
        monkeypatch.setattr(tools, "make_repl_session", Session)
        tools.repl()
        out, err = capsys.readouterr()
        assert err == (
            "<repl>:2:3: error: 'return' outside function\n  (return 1)\n  ^\n"
        )
        assert "42" in out and "Traceback" not in out + err

    def test_nrepl_eval(self):
        result = tools.nrepl_eval(RETURN_SRC, {"__macro_namespace": {}})
        assert result["error"] == (
            "<nrepl>:2:3: error: 'return' outside function\n  (return 1)\n  ^"
        )

    def test_nothing_runs_before_a_compile_error(self):
        result = tools.nrepl_eval('(print "ran")\n' + RETURN_SRC, {})
        assert "'return' outside function" in result["error"]
        assert result["stdout"] == ""

    def test_runtime_syntax_error_is_the_programs_own(self, monkeypatch):
        src = '(compile "x = (" "<user>" "exec")'
        monkeypatch.setattr(sys, "argv", ["lpy", "-c", src])
        with pytest.raises(SyntaxError) as info:
            tools.run()
        assert type(info.value) is SyntaxError
        assert info.value.filename == "<user>"

    def test_source_line_only_for_the_same_file(self):
        src = (
            '(defmacro m [] (raise (SyntaxError "boom" (, "other.lpy" 1 1 None))))\n(m)'
        )
        with pytest.raises(LispyCompileError) as info:
            with compile_errors("<mine>", src):
                compile_src(src)
        assert (info.value.filename, info.value.text) == ("other.lpy", None)
