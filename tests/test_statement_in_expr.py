import inspect
import re

import pytest

import lispy.core.compiler.stmt as stmt_module
from lispy.core.compiler.utils import STATEMENT_HEADS
from lispy.core.errors import LispyCompileError
from lispy.core.utils import augassignop_dict
from lispy.tools import src_to_python_org


def compile_error(src):
    with pytest.raises(LispyCompileError) as info:
        src_to_python_org(src)
    return info.value


@pytest.mark.parametrize(
    ["src", "msg", "position"],
    [
        (
            "(def f [] (return (do (= a 1) a)))",
            "(do ...) is a statement and cannot be used as an expression; "
            "statement blocks, including what defmacro, require and statement macros "
            "such as when expand to, cannot be values; "
            "to sequence expressions use (sub (, a b c) -1)",
            (1, 19),
        ),
        (
            "(f (= a 1))",
            "(= ...) is a statement and cannot be used as an expression; "
            "use (:= name value)",
            (1, 4),
        ),
        (
            "(f (if a b c))",
            "(if ...) is a statement and cannot be used as an expression; "
            "use (ife test then else)",
            (1, 4),
        ),
        (
            "(f (+= a 1))",
            "(+= ...) is a statement and cannot be used as an expression",
            (1, 4),
        ),
        (
            "(f (return 1))",
            "(return ...) is a statement and cannot be used as an expression",
            (1, 4),
        ),
        (
            "(f (else 1))",
            "else is a Python keyword and cannot be called",
            (1, 4),
        ),
    ],
)
def test_statement_head_in_expression_is_a_compile_error(src, msg, position):
    err = compile_error(src)
    assert err.msg == msg
    assert (err.lineno, err.offset) == position


@pytest.mark.parametrize("head", ["type", "match"])
def test_soft_keywords_stay_callable(head):
    assert src_to_python_org(f"(f ({head} x))") == f"f({head}(x))"


def test_statement_position_is_untouched():
    assert src_to_python_org("(if a b c)").startswith("if a:")


def test_statement_heads_match_stmt_compile():
    """STATEMENT_HEADS lists every head stmt_compile dispatches as a statement."""
    source = inspect.getsource(stmt_module)
    dispatched = set(re.findall(r'atom_text\(sexp\.op\) == "([^"]+)"', source))
    assert {"do", "=", "return", "def", "if"} <= dispatched
    assert dispatched <= STATEMENT_HEADS | {"match", "type"}
    assert set(augassignop_dict) <= STATEMENT_HEADS
