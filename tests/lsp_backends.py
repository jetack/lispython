"""The real Python backends the LSP tests run on (conftest.real_backends).

A test whose defect does not depend on the backend runs on every installed
one; ``on(name, ...)`` narrows it: to basedpyright for a behaviour only it
has (its definition walk-up, incomplete lists, keyword items, its label
shapes), or to the default backend for a test that never asks a real one
(an off backend, a stubbed answer), so it runs once.
"""

import pytest

# pyrefly first: the default backend (lispy.lsp.backend.DEFAULT_BACKENDS).
REAL_BACKENDS = ("pyrefly", "basedpyright")


def on(*names):
    """Run the test on these real backends only (each skipped when not
    installed)."""
    return pytest.mark.parametrize("real_backends", names, indirect=True)


basedpyright_only = on("basedpyright")
# A test that never reaches the real backend: once, on the default.
once = on("pyrefly")
