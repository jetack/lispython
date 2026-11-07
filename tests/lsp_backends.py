"""The real Python backends the LSP tests run on (conftest.real_backends).

A test whose defect does not depend on the backend runs on every installed
one; ``on(name, ...)`` narrows it: to basedpyright for a behaviour only it
has (its definition walk-up, incomplete lists, keyword items, its label
shapes). ``once`` marks a test that never asks a real one (an off backend,
a stubbed answer): it runs on the backend the server picks by default, the
first installed of these (conftest skips it on the others).
"""

import pytest

# lispy.lsp.backend.DEFAULT_BACKENDS, in its order (a test pins it); not
# imported, so collecting the tests never loads the LSP.
REAL_BACKENDS = ("pyrefly", "basedpyright")

# The marker of a test that runs once, on the default backend.
ONCE_MARKER = "lsp_backend_once"


def on(*names):
    """Run the test on these real backends only (each skipped when not
    installed)."""
    return pytest.mark.parametrize("real_backends", names, indirect=True)


basedpyright_only = on("basedpyright")


def once(test):
    """Run the test on the default backend only (the first installed of
    REAL_BACKENDS)."""
    return getattr(pytest.mark, ONCE_MARKER)(on(*REAL_BACKENDS)(test))
