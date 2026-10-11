import ast

import pytest

from lispy.core.errors import LispyCompileError
from lispy.tools import src_to_python_org

from .utils import stmt_to_dump


class TestNumberMethods:
    def test_integer(self):
        assert src_to_python_org("1") == "1"
        assert src_to_python_org("21234") == "21234"

    def test_float(self):
        assert src_to_python_org("12.34") == "12.34"
        assert src_to_python_org("12.") == "12.0"
        assert src_to_python_org(".34") == "0.34"

    def test_scientific_notation(self):
        assert src_to_python_org("1e3") == "1000.0"
        assert src_to_python_org("12.34e3") == "12340.0"

    def test_complex(self):
        assert src_to_python_org("3j") == "3j"
        assert src_to_python_org("1+2j") == "(1+2j)"
        assert src_to_python_org("1e3-.34j") == "(1000-0.34j)"


class TestStringMethods:
    def test_string(self):
        assert src_to_python_org('"asdfasdf"') == "'asdfasdf'"

    def test_raw_string(self):
        assert src_to_python_org('r"\\n"') == "'\\\\n'"
        assert src_to_python_org('"\\n"') == "'\\n'"

    def test_multi_line_string(self):
        assert src_to_python_org('''"a
b
c"''') == "'a\\nb\\nc'"
        assert src_to_python_org('''\"
\"''') == "'\\n'"

    def test_f_string(self):
        assert stmt_to_dump('f"sin({a}) is {(sin a):.3}"') == ast.dump(
            ast.parse('f"sin({a}) is {sin(a):.3}"')
        )

    def test_f_string_with_nested_quotes(self):
        # F-string with nested double-quoted string inside braces
        # Uses raw string to have literal backslash-quote as in .lpy files
        assert stmt_to_dump(
            r'f"Started at {(.strftime (datetime.now) \"%Y-%m-%d %H:%M:%S\")}"'
        ) == ast.dump(
            ast.parse('f"Started at {datetime.now().strftime(\'%Y-%m-%d %H:%M:%S\')}"')
        )

    def test_f_string_with_colon_in_nested_string(self):
        # Colon inside nested string should not be treated as format spec
        assert stmt_to_dump(r'f"time: {(.get d \"H:M:S\")}"') == ast.dump(
            ast.parse('f"time: {d.get(\'H:M:S\')}"')
        )


class TestInterpolatedStringScanner:
    """f-string bodies follow Python's rules (shared with t-strings)."""

    @pytest.mark.parametrize(
        ["lpy", "py"],
        [
            ('f"{{a}} {x} }}"', 'f"{{a}} {x} }}"'),
            ('f"{x=}"', 'f"{x=}"'),
            ('f"{ x = :>4}"', 'f"{ x = :>4}"'),
            ('f"{x = !s}"', 'f"{x = !s}"'),
            ('f"{x:>{w}.{p}}"', 'f"{x:>{w}.{p}}"'),
            # 3.11's own grammar has no whitespace after !r; the AST is the same.
            ('f"{ x !r : >10 }"', 'f"{ x !r: >10 }"'),
            ('f"{x:}"', 'f"{x:}"'),
            ('f""', 'f""'),
            ('F"{x}"', 'F"{x}"'),
            ('rf"\\n{x}"', 'rf"\\n{x}"'),
            ('fR"\\n{x}"', 'fR"\\n{x}"'),
            ('f"\\N{BULLET} {x}"', 'f"\\N{BULLET} {x}"'),
            ('f"{(!= a b)}"', 'f"{a != b}"'),
        ],
    )
    def test_matches_python(self, lpy, py):
        assert stmt_to_dump(lpy) == ast.dump(ast.parse(py))

    def test_self_documenting_echoes_lispython_text(self):
        scope = {"a": 2}
        exec(src_to_python_org('(= s f"{(+ a 1) = }")'), scope)
        assert scope["s"] == "(+ a 1) = 3"

    @pytest.mark.parametrize(
        ["lpy", "msg"],
        [
            ('f"{}"', "f-string: valid expression required before '}'"),
            ('f"{x!}"', "f-string: missing conversion character"),
            (
                'f"{x!z}"',
                "f-string: invalid conversion character 'z': expected 's', 'r', or 'a'",
            ),
            (
                'f"{x! r}"',
                "f-string: conversion type must come right after the exclamation mark",
            ),
            (
                'f"{a b}"',
                "f-string: expecting a single expression in a replacement field",
            ),
            ('f"a}b"', "f-string: single '}' is not allowed"),
            ('f"{x"', "f-string: expecting '}'"),
            ('bf"x"', "invalid string prefix 'bf'"),
        ],
    )
    def test_refused_like_python(self, lpy, msg):
        with pytest.raises(LispyCompileError) as info:
            src_to_python_org("\n  " + lpy)
        assert info.value.msg == msg
        assert (info.value.lineno, info.value.offset) == (2, 3)


class TestListMethods:
    def test_list(self):
        assert src_to_python_org("[1 2]") == "[1, 2]"

    def test_star(self):
        assert src_to_python_org("[*[1 2]]") == "[*[1, 2]]"
        assert src_to_python_org("[1 2 *[3 4]]") == "[1, 2, *[3, 4]]"


class TestTupleMethods:
    def test_tuple(self):
        assert src_to_python_org("(, 1 2)") == "(1, 2)"

    def test_star(self):
        assert src_to_python_org("(, *[1 2])") == "(*[1, 2],)"
        assert src_to_python_org("(, 1 2 *[3 4])") == "(1, 2, *[3, 4])"


class TestDictMethods:
    def test_dict(self):
        assert src_to_python_org("{1 2}") == "{1: 2}"

    def test_double_star(self):
        assert src_to_python_org("{1 2 **a **{3 4}}") == "{1: 2, **a, **{3: 4}}"


class TestSetMethods:
    def test_set(self):
        assert src_to_python_org("{, 1 2}") == "{1, 2}"

    def test_star(self):
        assert src_to_python_org("{, *[1 2 3] 2 3 4}") == "{*[1, 2, 3], 2, 3, 4}"
