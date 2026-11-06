"""``->>`` threads last through bare-symbol steps; ``conde`` refuses a missing
else (macros/sugar.lpy)."""

import pytest

from lispy.core.errors import LispyCompileError
from lispy.tools import src_to_python_org

PREAMBLE = "(require lispy.macros *)\n"


def test_thread_last_keeps_threading_last_after_a_bare_symbol():
    # A bare-symbol step used to hand the rest to ``->``: g(f(1), 1).
    assert src_to_python_org(PREAMBLE + "(->> 1 f (g 2))") == "g(2, f(1))"


@pytest.mark.parametrize("operands", ["", "x 1", "x 1 y 2", "x 1 y 2 z 3"])
def test_conde_without_else_is_refused(operands):
    # Used to compile to ``1 if x else 2``: the last test was never evaluated.
    with pytest.raises(LispyCompileError) as info:
        src_to_python_org(PREAMBLE + f"(conde {operands})")
    assert "final else" in info.value.msg


def test_conde_with_else_compiles():
    assert (
        src_to_python_org(PREAMBLE + "(conde x 1 y 2 3)") == "1 if x else 2 if y else 3"
    )
