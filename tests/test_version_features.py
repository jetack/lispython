"""Version-gated forms: one feature table, one error type, every entry point.

``lispy.core.compiler.features.running_version`` is the only place the
compiler reads the interpreter version, so refusal is tested on every
interpreter by monkeypatching it below a feature's minimum.
"""

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


# A statement the compiler gates as a t-string, until real gated forms exist.
GATED_SRC = "(= a 1)\n(print  (gated-form))\n"
GATED_MSG = 't-string t"..." needs Python 3.14+, running 3.11.13'
GATED_LINE, GATED_COL = 2, 9


@pytest.fixture
def gated_form(monkeypatch):
    from lispy.core.compiler import stmt

    original = stmt.stmt_compile

    def gated(sexp, decorator_list=None):
        for node in getattr(sexp, "list", []):
            if str(node) == "(gated-form)":
                features.require("t-string", node)
        return original(sexp, decorator_list)

    monkeypatch.setattr(stmt, "stmt_compile", gated)


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


@pytest.mark.usefixtures("old_python", "gated_form")
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
