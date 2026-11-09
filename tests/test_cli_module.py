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
