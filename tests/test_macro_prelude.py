"""The built-in macro prelude and per-unit macro scopes (core/macro.py
``compile_unit``).

Every compilation unit starts with lispy.macros' macros at compile time only;
a unit whose first form is ``(require lispy.macros <selection>)`` keeps just
that selection; macros a module defines never reach another module.
"""

import importlib
import json
import linecache
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import lispy  # noqa: F401  (registers the .lpy importer)
from lispy import tools
from lispy.core.macro import UnitStart, compile_unit
from lispy.core.parser import parse

BIN = Path(sys.executable).parent
LPY = BIN / "lpy"
L2PY = BIN / "l2py"
SRC = Path(__file__).resolve().parent.parent / "src"

needs_scripts = pytest.mark.skipif(
    not (LPY.exists() and L2PY.exists()),
    reason="lpy/l2py console scripts are not installed",
)


def run(args, cwd, **env_extra):
    env = {
        k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSAFEPATH")
    }
    env.update(env_extra)
    return subprocess.run(
        [str(a) for a in args], cwd=cwd, env=env, capture_output=True, text=True
    )


def translate(src, scope=None):
    """src's Python, compiled as a unit in scope (a fresh one by default)."""
    return tools.src_to_python_org(src, scope={} if scope is None else scope)


# --- every entry point starts with the prelude ------------------------------


@needs_scripts
def test_prelude_in_lpy_file_import_and_m(tmp_path):
    # Before: the importer compiled in macro.py's one shared namespace, which
    # held the built-in macros only when tools.lpy had been compiled in the
    # same process, so (inc 1) was a NameError on a warm bytecode cache.
    (tmp_path / "lib.lpy").write_text("(= value (inc 1))\n")
    (tmp_path / "main.lpy").write_text("(import lib)\n(print lib.value (inc 10))\n")
    for args in ([LPY, "main.lpy"], [LPY, "-m", "main"]):
        r = run(args, tmp_path)
        assert (r.returncode, r.stdout) == (0, "2 11\n"), r.stderr


@needs_scripts
def test_prelude_in_lpy_c_and_l2py_cli(tmp_path):
    (tmp_path / "f.lpy").write_text("(inc 1)\n")
    r = run([LPY, "-c", "(print (inc 1))"], tmp_path)
    assert (r.returncode, r.stdout) == (0, "2\n"), r.stderr
    r = run([L2PY, "--no-lispy", "-c", "(inc 1)"], tmp_path)
    assert r.stdout.strip() == "1 + 1", r.stderr
    r = run([L2PY, "--no-lispy", "f.lpy"], tmp_path)
    assert r.stdout.strip() == "1 + 1", r.stderr


def test_prelude_in_repl_and_nrepl_evals(capsys):
    tools.eval_and_print(
        "(print (inc 1))", {}, filename="<repl>", start=UnitStart.CONTINUE
    )
    assert capsys.readouterr().out == "2\n"
    assert tools.nrepl_eval("(inc 1)", {})["value"] == "2"


def test_a_translation_is_its_own_unit(monkeypatch):
    # A translation naming no scope is a module of its own: it neither sees
    # the REPL session's macros nor resets them.
    session = tools.main_scope()
    monkeypatch.setattr(tools, "SCOPE", session)
    defmacro = "(defmacro sq [x] (return `(* ~x ~x)))"
    compile_unit(parse(defmacro), session, start=UnitStart.CONTINUE)
    assert tools.src_to_python_org("(sq 2)") == "sq(2)"
    assert tools.l2py_s("(sq 2)", no_lispy=True, format=False) == "sq(2)"
    assert "sq" in session["__macro_namespace"]


def test_prelude_in_l2py_s_and_l2py_f(tmp_path):
    assert tools.l2py_s("(inc 1)", fresh_scope=True, no_lispy=True, format=False) == (
        "1 + 1"
    )
    path = tmp_path / "f.lpy"
    path.write_text("(inc 1)\n")
    assert tools.l2py_f(str(path), no_lispy=True, format=False) == "1 + 1\n"


def test_prelude_in_the_lsp_view():
    view = importlib.import_module("lispy.lsp.python_view").compile_view(
        "(= x (inc 1))"
    )
    assert view.text == "x = 1 + 1"


# --- macros stay in the module that defines them -----------------------------


@needs_scripts
def test_a_module_macro_does_not_leak_into_a_later_module(tmp_path):
    # m's defmacro body uses the prelude's `when`, expanded in m's own scope;
    # c is compiled after m, and must not see m's `twice`.
    (tmp_path / "m.lpy").write_text(
        "(defmacro twice [x] (when True (return `(* 2 ~x))))\n"
        "(def f [] (return (twice (inc 1))))\n"
    )
    (tmp_path / "c.lpy").write_text(
        '(try (twice 3) (except [NameError] (print "no-leak")))\n'
    )
    (tmp_path / "main.lpy").write_text("(import m)\n(print (m.f))\n(import c)\n")
    for cache in ("cold", "warm"):
        r = run([LPY, "main.lpy"], tmp_path)
        assert (r.returncode, r.stdout) == (0, "4\nno-leak\n"), (cache, r.stderr)


@pytest.fixture
def import_dir(tmp_path, monkeypatch):
    """tmp_path on sys.path; modules imported from it are dropped after."""
    monkeypatch.syspath_prepend(str(tmp_path))
    before = set(sys.modules)
    yield tmp_path
    for name in set(sys.modules) - before:
        del sys.modules[name]


def runtime_macros(module):
    return set(getattr(module, "__macro_namespace", {}))


def test_runtime_namespace_holds_only_the_modules_own_macros(import_dir):
    # The prelude is compile-time only: a module that requires nothing has no
    # built-in macro at run time, so (require mod *) does not re-export them.
    (import_dir / "pm_own.lpy").write_text(
        "(defmacro twice [x] (return `(* 2 ~x)))\n(= v (inc 1))\n"
    )
    (import_dir / "pm_user.lpy").write_text("(require pm-own *)\n(= v (twice 3))\n")
    (import_dir / "pm_all.lpy").write_text("(require lispy.macros *)\n(= v (inc 1))\n")
    own = importlib.import_module("pm_own")
    user = importlib.import_module("pm_user")
    everything = importlib.import_module("pm_all")
    import lispy.macros

    assert (own.v, runtime_macros(own)) == (2, {"twice"})
    assert (user.v, runtime_macros(user)) == (6, {"twice"})
    expected = set(getattr(lispy.macros, "__macro_namespace"))
    assert (everything.v, runtime_macros(everything)) == (2, expected)


def test_the_prelude_emits_no_code():
    meta = tools.src_to_python_org("(inc 1)", include_meta=True, scope={})
    assert meta == "1 + 1"
    src = "(require lispy.macros *)\n(inc 1)"
    meta = tools.src_to_python_org(src, include_meta=True, scope={})
    assert meta.count("importlib.import_module('lispy.macros')") == 1
    assert meta.endswith("\n1 + 1")
    assert translate(src) == "1 + 1"


# --- opting out: (require lispy.macros <selection>) as the first form --------


@pytest.mark.parametrize(
    ["src", "expected"],
    [
        ["(require lispy.macros [])\n(inc 1)", "inc(1)"],
        ["(require lispy.macros [->])\n(-> 1 (f))\n(inc 1)", "f(1)\ninc(1)"],
        ["(require lispy.macros *)\n(inc 1)", "1 + 1"],
        ["(require lispy.macros)\n(lispy.macros.inc 1)\n(inc 1)", "1 + 1\ninc(1)"],
        ['"Doc."\n(require lispy.macros [])\n(inc 1)', "'Doc.'\ninc(1)"],
        # Only the first form opts out; anywhere else a require only adds.
        ["(= a 1)\n(require lispy.macros [])\n(inc 1)", "a = 1\n1 + 1"],
        # Only lispy.macros itself: a submodule's require only adds.
        ["(require lispy.macros.sugar [cond])\n(inc 1)", "1 + 1"],
    ],
)
def test_opt_out(src, expected):
    assert translate(src) == expected


def test_a_unit_start_forgets_an_earlier_units_macros():
    # A persistent scope compiling a new module or file unit starts from the
    # prelude alone, as a fresh module does: an earlier unit's macro of a
    # prelude name neither survives an opt-out nor shadows the built-in.
    scope = {}
    translate("(defmacro inc [x] (return `(+ ~x 1000)))\n(inc 1)", scope)
    assert translate("(require lispy.macros [])\n(inc 1)", scope) == "inc(1)"
    assert translate("(inc 1)", scope) == "1 + 1"


def test_repl_input_require_only_adds():
    scope = {}
    compile_unit(parse("(require lispy.macros [])"), scope, start=UnitStart.CONTINUE)
    assert translate("(dec 1)", scope) == "1 - 1"


# --- bootstrap ---------------------------------------------------------------


def test_lispy_macros_compiles_cold(tmp_path):
    # lispy.macros is the prelude's source: compiling it must not import it.
    r = run(
        [
            sys.executable,
            "-c",
            "import lispy.macros as m; print(len(m.__dict__['__macro_namespace']) > 0)",
        ],
        tmp_path,
        PYTHONPYCACHEPREFIX=str(tmp_path / "pycache"),
    )
    assert (r.returncode, r.stdout) == (0, "True\n"), r.stderr
    assert list((tmp_path / "pycache").rglob("sugar.*.lpyc"))


# --- nREPL: one scope per file ----------------------------------------------


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(linecache, "cache", dict(linecache.cache))
    scope = {"__builtins__": __builtins__, "__name__": "__main__"}
    tools.nrepl_eval("(from pprint [pprint])", scope)
    return scope


def request(scope, **msg):
    reply = tools._nrepl_dispatch(msg, scope)
    json.dumps(reply)  # every reply goes over the wire
    return reply


def test_nrepl_file_scopes(session, tmp_path):
    path = str(tmp_path / "f.lpy")
    Path(path).write_text(
        "(require lispy.macros [])\n(defmacro sq [x] (return `(* ~x ~x)))\n(= fval 7)\n"
    )
    assert request(session, op="load-file", path=path)["error"] is None

    # The session keeps every built-in macro and sees nothing of the file.
    assert request(session, op="eval", code="(inc 1)")["value"] == "2"
    error = request(session, op="eval", code="fval")["error"]
    assert "NameError: name 'fval' is not defined" in error

    # An eval naming the file runs in the file's scope, opt-out included.
    reply = request(session, op="eval", code="(sq fval)", file=path, line=4)
    assert reply["value"] == "49"
    error = request(session, op="eval", code="(inc 1)", file=path, line=4)["error"]
    assert "NameError: name 'inc' is not defined" in error

    file_completion = request(session, op="complete", prefix="sq", file=path)
    assert {"label": "sq", "kind": 3, "labelDetails": {"description": "macro"}} in (
        file_completion["items"]
    )
    assert "sq" not in request(session, op="complete", prefix="sq")["completions"]

    expansion = request(session, op="macroexpand", code="(sq 3)", file=path)
    assert expansion["expansion"] == "(* 3 3)"
    assert request(session, op="macroexpand", code="(sq 3)")["expansion"] == "(sq 3)"


def test_nrepl_region_eval_opts_out_only_at_the_file_start(session, tmp_path):
    # Editor clients send file, line and column with every region, so a
    # require on a later line is a later form of the file: it only adds.
    path = str(tmp_path / "f.lpy")
    Path(path).write_text("(= a 1)\n(require lispy.macros [->])\n(= b (inc 1))\n")
    assert request(session, op="load-file", path=path)["error"] is None
    region = "(require lispy.macros [->])"

    def file_expansion():
        # macroexpand reads the file scope as it stands, compiling no unit.
        return request(session, op="macroexpand", code="(inc 1)", file=path)

    reply = request(session, op="eval", code=region, file=path, line=2, column=1)
    assert reply["error"] is None
    assert file_expansion()["expansion"] == "(+ 1 1)"
    reply = request(session, op="eval", code="(inc 1)", file=path, line=3)
    assert reply["error"] is None
    # The same text at line 1, column 1 starts the file: it opts out.
    reply = request(session, op="eval", code=region, file=path, line=1, column=1)
    assert reply["error"] is None
    assert file_expansion()["expansion"] == "(inc 1)"
    error = request(session, op="eval", code="(inc 1)", file=path, line=3)["error"]
    assert "NameError: name 'inc' is not defined" in error


def expand_in(session, path, code):
    """code's expansion in path's file scope as it stands (no unit compiles)."""
    return request(session, op="macroexpand", code=code, file=str(path))["expansion"]


def eval_at(session, path, code, line):
    return request(session, op="eval", code=code, file=str(path), line=line, column=1)


def test_nrepl_top_region_keeps_the_files_macros(session, tmp_path):
    # Re-sending a file's top form (an import, its ns form) is not a reload:
    # the macros the file defined further down stay, as Clojure keeps a
    # namespace's vars when its ns form is evaluated again.
    path = tmp_path / "f.lpy"
    src = "(import math)\n(defmacro sq [x] (return `(* ~x ~x)))\n(print (sq 3))\n"
    assert load(session, path, src)["stdout"] == "9\n"
    assert eval_at(session, path, "(import math)", 1)["error"] is None
    assert eval_at(session, path, "(sq 4)", 3)["value"] == "16"


def test_nrepl_top_region_opt_out_drops_only_the_built_ins(session, tmp_path):
    path = tmp_path / "f.lpy"
    load(session, path, "(= a 1)\n(defmacro sq [x] (return `(* ~x ~x)))\n")
    assert eval_at(session, path, "(require lispy.macros [->])", 1)["error"] is None
    assert expand_in(session, path, "(inc 1)") == "(inc 1)"
    assert expand_in(session, path, "(-> 1 (f))") == "(f 1)"
    assert expand_in(session, path, "(sq 3)") == "(* 3 3)"
    # A region with no form beyond a docstring re-derives nothing.
    assert eval_at(session, path, '"Doc."', 1)["error"] is None
    assert expand_in(session, path, "(inc 1)") == "(inc 1)"
    # Editing the opt-out away and re-sending the top restores the built-ins.
    assert eval_at(session, path, "(= a 1)", 1)["error"] is None
    assert expand_in(session, path, "(inc 1)") == "(+ 1 1)"
    assert expand_in(session, path, "(sq 3)") == "(* 3 3)"


@pytest.mark.parametrize(
    "src",
    [
        "(require lispy.macros [->])\n(require lispy.macros [cond])\n(= a 1)\n",
        "(= a 1)\n(require lispy.macros [cond])\n",
    ],
)
def test_nrepl_top_region_keeps_a_built_in_the_file_required(session, tmp_path, src):
    # A built-in a later require registered is the require's, not the
    # unit start's: an opt-out at the top drops only what a start seeded.
    path = tmp_path / "f.lpy"
    assert load(session, path, src)["error"] is None
    assert eval_at(session, path, "(require lispy.macros [->])", 1)["error"] is None
    assert expand_in(session, path, "(cond a 1)") == "(if a 1)"
    assert expand_in(session, path, "(-> 1 (f))") == "(f 1)"
    assert expand_in(session, path, "(inc 1)") == "(inc 1)"


def test_nrepl_top_region_keeps_the_files_macro_of_a_built_in_name(session, tmp_path):
    path = tmp_path / "f.lpy"
    load(session, path, "(= a 1)\n(defmacro inc [x] (return `(+ ~x 1000)))\n")
    assert eval_at(session, path, "(= a 1)", 1)["error"] is None
    assert expand_in(session, path, "(inc 1)") == "(+ 1 1000)"


def test_nrepl_scope_lock_is_not_in_the_scope(session, tmp_path):
    # The lock lives beside the scope, never in the user's globals.
    path = tmp_path / "f.lpy"
    assert eval_at(session, path, "(= a 1)", 1)["error"] is None
    reply = request(session, op="complete", prefix="__nr", file=str(path))
    assert reply["completions"] == []
    assert tools.nrepl_scope_lock(session) is tools.nrepl_scope_lock(session)


def test_nrepl_macro_bookkeeping_is_not_a_global(session, tmp_path):
    # The prelude record and the seeded keys live on the macro table, so
    # neither completion nor (globals) shows them.
    path = tmp_path / "f.lpy"
    assert eval_at(session, path, "(= a 1)", 1)["error"] is None
    for scope in (session, tools.nrepl_file_scope(str(path))):
        assert [k for k in scope if k.startswith("__macro")] == ["__macro_namespace"]
    for prefix in ("__macro_p", "__macro_s"):
        assert request(session, op="complete", prefix=prefix)["completions"] == []


def test_a_scope_whose_table_is_dropped_is_seeded_again():
    scope = {}
    compile_unit(parse("(= a 1)"), scope)
    del scope["__macro_namespace"]
    compile_unit(parse("(= b 1)"), scope, start=UnitStart.CONTINUE)
    assert "inc" in scope["__macro_namespace"]


@pytest.mark.parametrize("opted_out_first", [True, False])
def test_nrepl_reload_after_the_first_form_changes(session, tmp_path, opted_out_first):
    # A file scope persists across loads; each load starts from the whole
    # prelude again, so editing the opt-out in or out takes effect.
    path = tmp_path / "f.lpy"
    opted_out = "(require lispy.macros [])\n(= x (inc 1))\n"
    plain = "(= x (inc 1))\n"
    loads = [opted_out, plain] if opted_out_first else [plain, opted_out]
    errors = []
    for src in loads:
        path.write_text(src)
        errors.append(request(session, op="load-file", path=str(path))["error"])
    if opted_out_first:
        assert "NameError: name 'inc' is not defined" in errors[0]
        assert errors[1] is None
        assert request(session, op="eval", code="x", file=str(path))["value"] == "2"
    else:
        assert errors[0] is None
        assert "NameError: name 'inc' is not defined" in errors[1]


def load(session, path, src):
    path.write_text(src)
    return request(session, op="load-file", path=str(path))


@pytest.mark.parametrize(
    ["before", "after"],
    [
        # A file turns its macro into a function.
        [
            "(defmacro helper [x] (return `(+ ~x 1000)))\n(print (helper 1))\n",
            "(def helper [x] (return (+ x 1)))\n(print (helper 1))\n",
        ],
        # A file's own macro of a prelude name, replaced after an opt-out.
        [
            "(defmacro inc [x] (return `(+ ~x 1000)))\n(print (inc 1))\n",
            "(require lispy.macros [])\n"
            "(def inc [x] (return (+ x 1)))\n(print (inc 1))\n",
        ],
    ],
)
def test_nrepl_reload_compiles_as_a_fresh_module(session, tmp_path, before, after):
    # A reload starts from the prelude alone, so a macro only the previous
    # load defined is not a macro to bind over, as under lpy.
    path = tmp_path / "f.lpy"
    first = load(session, path, before)
    assert (first["error"], first["stdout"]) == (None, "1001\n")
    second = load(session, path, after)
    assert (second["error"], second["stdout"]) == (None, "2\n")


def test_nrepl_reload_keeps_the_files_macros_and_the_prelude(session, tmp_path):
    path = tmp_path / "f.lpy"
    src = "(defmacro helper [x] (return `(inc ~x)))\n(print (helper 1))\n"
    for _ in range(2):
        reply = load(session, path, src)
        assert (reply["error"], reply["stdout"]) == (None, "2\n")


def test_nrepl_file_eval_without_a_line_is_not_the_file_start(session, tmp_path):
    # Only a line sent as 1 starts the file's unit: one with no line would
    # bring the prelude back mid-file, undoing the opt-out.
    path = tmp_path / "f.lpy"
    assert load(session, path, "(require lispy.macros [])\n(= a 1)\n")["error"] is None
    error = request(session, op="eval", code="(= b (inc 1))", file=str(path))["error"]
    assert "NameError: name 'inc' is not defined" in error
    reply = request(session, op="eval", code="(= c (inc 1))", file=str(path), line=3)
    assert "NameError: name 'inc' is not defined" in reply["error"]


# --- docs ---------------------------------------------------------------------


def test_the_built_in_macros_note_matches_behaviour():
    # The note feeds llms.txt and the docs: the built-ins need no require,
    # and every require of lispy.macros it shows, made a module's first
    # form, narrows them.
    from lispy.form_docs import BUILTIN_MACROS_NOTE

    assert translate("(inc 1)") == "1 + 1"
    spans = re.findall(r"`(\(require lispy\.macros[^`]*\))`", BUILTIN_MACROS_NOTE)
    assert spans
    for span in spans:
        assert translate(f"{span}\n(inc 1)").splitlines()[-1] == "inc(1)", span


# --- one home ----------------------------------------------------------------


def test_only_compile_unit_reaches_macroexpand_then_compile():
    # Policy: compile_unit is the one entry point of compilation, so every
    # unit gets the prelude and the opt-out. Allowlist: the module itself.
    allowed = {"lispy/core/macro.py"}
    pattern = re.compile(r"macroexpand[-_]then[-_]compile")
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in sorted(SRC.rglob("*"))
        if path.suffix in (".py", ".lpy")
        and path.relative_to(SRC).as_posix() not in allowed
        and pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
