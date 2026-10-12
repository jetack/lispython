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


def run_m(cwd: Path, **env_extra: str) -> subprocess.CompletedProcess[str]:
    env = {
        k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSAFEPATH")
    }
    env.update(env_extra)
    return subprocess.run(
        [str(LPY), "-m", "cwdmod"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


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
