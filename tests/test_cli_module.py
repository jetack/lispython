"""``lpy -m mod`` resolves ``mod`` from the cwd the way ``python -m`` does.

Runs the installed ``lpy`` console script: its sys.path[0] is the script's bin
directory, which every run mode replaces with the entry ``python`` gives it.
"""

import ast
import json
import os
import select
import socket
import subprocess
import sys
from pathlib import Path

import pytest

LPY = Path(sys.executable).parent / "lpy"

pytestmark = pytest.mark.skipif(
    not LPY.exists(), reason="lpy console script is not installed"
)


def run_m(
    cwd: Path, module: str = "cwdmod", **env_extra: str
) -> subprocess.CompletedProcess[str]:
    env = {
        k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSAFEPATH")
    }
    env.update(env_extra)
    return subprocess.run(
        [str(LPY), "-m", module],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def test_escaped_quote_in_a_field_warns_under_default_filters(tmp_path: Path):
    # A DeprecationWarning attributed to a .lpy module is hidden by Python's
    # default filters; a SyntaxWarning is shown.
    src = tmp_path / "esc.lpy"
    src.write_text('(print "start")\n(print f"{(.get {"k" 1} \\"k\\")}")\n')
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "PYTHONSAFEPATH", "PYTHONWARNINGS", "PYTHONDEVMODE")
    }
    r = subprocess.run([str(LPY), str(src)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert f"{src}:2: SyntaxWarning: f-string: write a plain" in r.stderr
    assert 'not \\"' in r.stderr


def test_m_finds_python_module_in_cwd(tmp_path: Path):
    (tmp_path / "cwdmod.py").write_text("print('ran-from-cwd')\n")
    r = run_m(tmp_path)
    assert r.returncode == 0, r.stderr
    assert "ran-from-cwd" in r.stdout


def test_m_finds_lpy_module_in_cwd(tmp_path: Path):
    (tmp_path / "cwdmod.lpy").write_text('(print "ran-lpy")\n')
    r = run_m(tmp_path)
    assert r.returncode == 0, r.stderr
    assert "ran-lpy" in r.stdout


def test_m_honours_safe_path(tmp_path: Path):
    (tmp_path / "cwdmod.py").write_text("print('ran-from-cwd')\n")
    r = run_m(tmp_path, PYTHONSAFEPATH="1")
    assert r.returncode != 0
    assert "ran-from-cwd" not in r.stdout
    assert "cwdmod" in r.stderr


@pytest.mark.parametrize(
    ("arg", "file"),
    [("ok?", "ok_p.lpy"), ("->foo", "_to_foo.lpy"), ("my-mod", "my_mod.lpy")],
)
def test_m_names_the_module_as_import_does(tmp_path: Path, arg: str, file: str):
    (tmp_path / file).write_text('(print "ran-mangled")\n')
    r = run_m(tmp_path, arg)
    assert r.returncode == 0, r.stderr
    assert "ran-mangled" in r.stdout


def test_m_refuses_a_part_that_is_no_name(tmp_path: Path):
    r = run_m(tmp_path, "class")
    assert r.returncode == 1
    assert r.stderr.startswith("<command line>:1:1: error:")
    assert r.stderr.splitlines()[1:] == ["class", "^"]
    assert "Traceback" not in r.stderr


@pytest.mark.parametrize(
    ("module", "message"),
    [
        ("no-such-mod", "No module named no_such_mod"),
        (".rel", "Relative module names not supported"),
        (
            "nopar.child",
            "Error while finding module specification for 'nopar.child' "
            "(ModuleNotFoundError: No module named 'nopar')",
        ),
    ],
)
def test_m_unfindable_module_is_one_line_like_python(
    tmp_path: Path, module: str, message: str
):
    r = run_m(tmp_path, module)
    assert r.returncode == 1
    assert r.stderr.strip().endswith(message)
    assert len(r.stderr.strip().splitlines()) == 1
    assert "Traceback" not in r.stderr


def test_m_import_error_inside_the_module_keeps_its_traceback(tmp_path: Path):
    (tmp_path / "cwdmod.py").write_text("import no_such_dependency\n")
    r = run_m(tmp_path)
    assert r.returncode == 1
    assert "Traceback" in r.stderr
    assert "No module named 'no_such_dependency'" in r.stderr


def test_m_import_error_from_a_macro_keeps_its_traceback(tmp_path: Path):
    (tmp_path / "cwdmod.lpy").write_text(
        "(defmacro m [] (import no_such_dep_q) 1)\n(print (m))\n"
    )
    r = run_m(tmp_path)
    assert r.returncode == 1
    assert "Traceback" in r.stderr
    assert "No module named 'no_such_dep_q'" in r.stderr


def test_m_compiles_an_lpy_module_once(tmp_path: Path):
    # With no bytecode cache written, a second compile would run the macro twice.
    (tmp_path / "cwdmod.lpy").write_text(
        '(defmacro m [] (print "MACRO-RAN") 1)\n(print (m))\n'
    )
    r = run_m(tmp_path, PYTHONDONTWRITEBYTECODE="1")
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("MACRO-RAN") == 1


def test_m_macro_import_error_in_a_package_main_names_the_real_cause(tmp_path: Path):
    pkg = tmp_path / "cwdmod"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.lpy").write_text(
        "(defmacro m [] (import no_such_dep_q) 1)\n(print (m))\n"
    )
    r = run_m(tmp_path)
    assert r.returncode == 1
    assert "Traceback" in r.stderr
    assert "No module named 'no_such_dep_q'" in r.stderr
    assert "cannot be directly executed" not in r.stderr


def test_m_compile_time_import_error_traceback_has_no_runpy_noise(tmp_path: Path):
    (tmp_path / "cwdmod.lpy").write_text(
        "(defmacro m [] (import no_such_dep_q) 1)\n(print (m))\n"
    )
    r = run_m(tmp_path)
    assert r.returncode == 1
    assert "Traceback" in r.stderr
    assert "During handling" not in r.stderr
    assert not [
        line for line in r.stderr.splitlines() if line.startswith("runpy._Error:")
    ]


def test_m_package_without_main_is_one_line(tmp_path: Path):
    pkg = tmp_path / "cwdmod"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    r = run_m(tmp_path)
    assert r.returncode == 1
    assert len(r.stderr.strip().splitlines()) == 1
    assert r.stderr.strip().endswith(
        "'cwdmod' is a package and cannot be directly executed"
    )


def test_c_imports_a_module_beside_the_cwd_unless_safe_path(tmp_path: Path):
    # Defects: `lpy -c` puts nothing on sys.path, so a module in the cwd,
    # which `python -c 'import helper'` imports, is ModuleNotFoundError; or
    # it puts the cwd there under -P / PYTHONSAFEPATH, where python does not.
    (tmp_path / "cwdhelper.lpy").write_text('(print "ran-helper")\n')
    env = {
        k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSAFEPATH")
    }

    def run_c(**extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(LPY), "-c", "(import cwdhelper)"],
            cwd=tmp_path,
            env={**env, **extra},
            capture_output=True,
            text=True,
        )

    r = run_c()
    assert r.returncode == 0, r.stderr
    assert "ran-helper" in r.stdout
    r = run_c(PYTHONSAFEPATH="1")
    assert "No module named 'cwdhelper'" in r.stderr


def clean_env(**extra: str) -> dict[str, str]:
    env = {
        k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSAFEPATH")
    }
    env.update(extra)
    return env


PRINT_PATH = "(import sys json) (print (json.dumps sys.path))"


def lpy_path(args: list[str], cwd: Path, **extra: str) -> list[str]:
    r = subprocess.run(
        [str(LPY), *args],
        cwd=cwd,
        env=clean_env(**extra),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.splitlines()[-1])


def test_c_puts_an_empty_entry_first_and_no_launcher_directory(tmp_path: Path):
    # Defect: `lpy -c` stacks the cwd in front of the console script's own
    # directory, so .venv/bin stays on sys.path (its activate_this.py is
    # importable) and an absolute cwd stands where python -c has ''.
    path = lpy_path(["-c", PRINT_PATH], tmp_path)
    assert path[0] == ""
    assert str(LPY.parent) not in path


def test_a_file_run_puts_its_directory_first_even_when_on_pythonpath(
    tmp_path: Path,
):
    # Defect: a file run skips its directory when PYTHONPATH already has it,
    # so a helper on an earlier PYTHONPATH entry wins over the script's
    # sibling, which `python sub/s.py` imports.
    sub, other = tmp_path / "sub", tmp_path / "other"
    for directory in (sub, other):
        directory.mkdir()
        (directory / "pathhelper.py").write_text(f"WHERE = {directory.name!r}\n")
    (sub / "s.lpy").write_text(
        "(import sys json pathhelper)\n"
        "(print (json.dumps (+ [pathhelper.WHERE] sys.path)))\n"
    )
    got = lpy_path([str(sub / "s.lpy")], tmp_path, PYTHONPATH=f"{other}:{sub}")
    assert got[:2] == ["sub", str(sub)]
    assert str(LPY.parent) not in got


def test_a_symlinked_script_runs_with_its_targets_directory(tmp_path: Path):
    # Defect: a file run puts the link's own directory first, where python
    # resolves the link and puts the target's directory (os.path.realpath).
    real, links = tmp_path / "real", tmp_path / "links"
    real.mkdir()
    links.mkdir()
    (real / "s.lpy").write_text(PRINT_PATH + "\n")
    (links / "s.lpy").symlink_to(real / "s.lpy")
    path = lpy_path([str(links / "s.lpy")], tmp_path)
    assert path[0] == str(real)
    assert str(links) not in path


def test_m_puts_the_absolute_cwd_first_and_no_launcher_directory(tmp_path: Path):
    # Defect: -m stacks its entry in front of the console script's
    # directory, or puts '' where python -m puts the absolute cwd.
    (tmp_path / "pathmod.lpy").write_text(PRINT_PATH + "\n")
    path = lpy_path(["-m", "pathmod"], tmp_path)
    assert path[0] == str(tmp_path)
    assert str(LPY.parent) not in path


@pytest.mark.parametrize("typed", ["app", "./app/"])
def test_a_directory_app_has_its_own_absolute_entry_alone(tmp_path: Path, typed: str):
    # Defects: `lpy app` (a directory with __main__.lpy) gets runpy's
    # relative "app" in front of the script-style absolute parent directory,
    # where `python app` has the absolute app directory alone, joined onto
    # the cwd as typed; or argv[0] becomes that absolute path, where
    # `python app` keeps it as typed.
    app = tmp_path / "app"
    app.mkdir()
    (app / "__main__.lpy").write_text(
        "(import sys json) (print (json.dumps (+ [(sub sys.argv 0)] sys.path)))\n"
    )
    [argv0, *path] = lpy_path([typed], tmp_path)
    assert argv0 == typed
    assert path[0] == os.path.join(str(tmp_path), typed)
    assert typed not in path
    assert str(tmp_path) not in path
    assert str(LPY.parent) not in path


def test_a_directory_app_runs_in_a_fresh_main(tmp_path: Path):
    # Defect: the app runs in the console script's __main__, so its globals
    # hold the launcher's `sys` and `run`, where `python app` gives it a
    # fresh __main__ and a bare `sys` is a NameError.
    app = tmp_path / "app"
    app.mkdir()
    (app / "__main__.lpy").write_text("(print (sorted (globals)))\n")
    r = subprocess.run(
        [str(LPY), "app"], cwd=tmp_path, env=clean_env(), capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    names = ast.literal_eval(r.stdout.splitlines()[-1])
    assert "__file__" in names
    assert "sys" not in names
    assert "run" not in names


def test_a_file_run_under_safe_path_adds_no_directory(tmp_path: Path):
    # Defect: a file run puts its directory first under PYTHONSAFEPATH,
    # where python adds none.
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "s.lpy").write_text(PRINT_PATH + "\n")
    path = lpy_path([str(sub / "s.lpy")], tmp_path, PYTHONSAFEPATH="1")
    assert str(sub) not in path
    assert "" not in path


@pytest.mark.parametrize("args", [[], ["-t"]])
def test_the_repl_puts_an_empty_entry_first(tmp_path: Path, args: list[str]):
    # Defect: the REPL (or -t, the translating REPL) puts the absolute cwd
    # first, in front of the console script's directory, where python's
    # REPL has '' (which follows a later os.chdir) and no launcher directory.
    # prompt_toolkit reads a piped \r as Enter; HOME keeps its history file
    # out of the user's.  The output is mixed with prompt redraws (and -t's
    # translation), so the path follows a marker the echoed input never holds.
    src = '(import sys json) (print (+ "PATH=" (json.dumps sys.path)))'
    r = subprocess.run(
        [str(LPY), *args],
        cwd=tmp_path,
        env=clean_env(HOME=str(tmp_path)),
        input=src + "\r",
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    [line] = [line for line in r.stdout.splitlines() if "PATH=[" in line]
    path = json.loads(line[line.index("PATH=[") + len("PATH=") :])
    assert path[0] == ""
    assert str(LPY.parent) not in path


@pytest.mark.parametrize("safe_path", [False, True])
def test_the_nrepl_server_puts_the_cwd_first_unless_safe_path(
    tmp_path: Path, safe_path: bool
):
    # Defects: the nREPL server stacks the cwd in front of the console
    # script's directory (or skips it when present elsewhere), or adds it
    # under PYTHONSAFEPATH, where python -m adds nothing.
    extra = {"PYTHONSAFEPATH": "1"} if safe_path else {}
    server = subprocess.Popen(
        [str(LPY), "--nrepl", "0"],
        cwd=tmp_path,
        env=clean_env(**extra),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert server.stdout is not None
        # Bounded: a server that never starts fails the test, never hangs it.
        ready, _, _ = select.select([server.stdout], [], [], 30)
        assert ready, "the nREPL server printed no port"
        port = int(server.stdout.readline().rsplit(":", 1)[1])
        with socket.create_connection(("127.0.0.1", port), timeout=30) as conn:
            request = {"op": "eval", "id": 1, "code": PRINT_PATH}
            conn.sendall(json.dumps(request).encode() + b"\n")
            reply = json.loads(conn.makefile("rb").readline())
    finally:
        server.kill()
        server.wait(timeout=30)
    path = json.loads(reply["stdout"])
    assert str(LPY.parent) not in path
    if safe_path:
        assert str(tmp_path) not in path
    else:
        assert path[0] == str(tmp_path)


def test_importing_tools_in_a_package_directory_leaves_sys_path_and_main(
    tmp_path: Path,
):
    # Defect: importing lispy.tools from a working directory inside a
    # package puts that package's root on sys.path and gives the REPL's
    # __main__ scope the package as __package__, which Python's own
    # __main__ never has: the REPL then resolves imports and relative
    # requires no `python` run there would.
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    code = (
        "import sys\n"
        "before = list(sys.path)\n"
        "import lispy.tools as t\n"
        "print(sys.path == before)\n"
        "print(t.main_scope()['__package__'] == __package__, repr(__package__))\n"
    )
    env = {
        k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSAFEPATH")
    }
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=pkg, env=env, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines()[:2] == ["True", "True None"]
