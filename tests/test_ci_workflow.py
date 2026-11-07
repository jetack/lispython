"""The CI matrix is noxfile.py's PYTHONS, so a new interpreter cannot be tested
locally and missed (or the reverse) in CI."""

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _nox_pythons():
    tree = ast.parse((ROOT / "noxfile.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "PYTHONS":
                return ast.literal_eval(node.value)
    raise AssertionError("noxfile.py has no PYTHONS")


def test_workflow_matrix_is_the_noxfile_matrix():
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]["python"]
    assert matrix == _nox_pythons()
