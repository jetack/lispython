"""``lpy -m mod`` resolves ``mod`` from the cwd the way ``python -m`` does.

Runs the installed ``lpy`` console script: its sys.path[0] is the script's bin
directory, so the cwd is on the path only if ``-m`` puts it there.
"""

import os
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
