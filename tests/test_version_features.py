"""Version-gated forms: one feature table, one error type, every entry point.

``lispy.core.compiler.features.running_version`` is the only place the
compiler reads the interpreter version, so refusal is tested on every
interpreter by monkeypatching it below a feature's minimum.
"""

import ast
import importlib
import sys

import pytest

import lispy.tools as tools
from lispy.core.compiler import features
from lispy.core.errors import LispyCompileError, format_compile_error
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
        ["lpy", "msg"],
        [
            ('ft"x"', "'f' and 't' prefixes are incompatible"),
            ('tf"x"', "'f' and 't' prefixes are incompatible"),
            ('bt"x"', "'b' and 't' prefixes are incompatible"),
            ('Ut"x"', "'u' and 't' prefixes are incompatible"),
            ('tt"x"', "invalid string prefix 'tt'"),
            (
                't"{x!z}"',
                "t-string: invalid conversion character 'z': expected 's', 'r', or 'a'",
            ),
            ('t"{}"', "t-string: valid expression required before '}'"),
            ('t"a}"', "t-string: single '}' is not allowed"),
        ],
    )
    def test_malformed_refused_on_every_version(self, lpy, msg):
        err = refusal("\n   " + lpy)
        assert err.msg == msg
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
