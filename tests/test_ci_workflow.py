"""The CI test job is noxfile.py's PYTHONS and EXTRAS on the locked set, so a new
interpreter or extra cannot be tested locally and missed (or the reverse) in CI."""

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _nox_constant(name):
    tree = ast.parse((ROOT / "noxfile.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"noxfile.py has no {name}")


def _nox_pythons():
    return _nox_constant("PYTHONS")


def _test_command():
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    runs = [s["run"] for s in workflow["jobs"]["test"]["steps"] if "run" in s]
    (command,) = [r.split() for r in runs if "pytest" in r]
    return command


def test_workflow_matrix_is_the_noxfile_matrix():
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]["python"]
    assert matrix == _nox_pythons()


def test_workflow_extras_are_the_noxfile_extras():
    command = _test_command()
    extras = [command[i + 1] for i, arg in enumerate(command) if arg == "--extra"]
    assert sorted(extras) == sorted(_nox_constant("EXTRAS"))


def test_nox_tests_session_syncs_the_lock_with_extras():
    # `tests` is the locked set; `tests_latest` is the only unlocked session.
    tree = ast.parse((ROOT / "noxfile.py").read_text())
    (session,) = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "tests"
    ]
    strings = {n.value for n in ast.walk(session) if isinstance(n, ast.Constant)}
    names = {n.id for n in ast.walk(session) if isinstance(n, ast.Name)}
    assert {"sync", "--frozen"} <= strings
    assert "EXTRAS" in names


def test_workflow_runs_the_locked_set():
    # nox's `tests` session syncs --frozen; CI must test the same set.
    assert "--frozen" in _test_command()
