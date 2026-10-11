import ast
import re
import sys

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
        assert (
            src_to_python_org('''"a
b
c"''')
            == "'a\\nb\\nc'"
        )
        assert (
            src_to_python_org("""\"
\"""")
            == "'\\n'"
        )

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
            ast.parse("f\"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\"")
        )

    def test_f_string_with_colon_in_nested_string(self):
        # Colon inside nested string should not be treated as format spec
        assert stmt_to_dump(r'f"time: {(.get d \"H:M:S\")}"') == ast.dump(
            ast.parse("f\"time: {d.get('H:M:S')}\"")
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
        ["lpy", "msg", "at"],
        [
            ('f"{}"', "f-string: valid expression required before '}'", "{"),
            ('f"{x!}"', "f-string: missing conversion character", "!"),
            (
                'f"{x!z}"',
                "f-string: invalid conversion character 'z': expected 's', 'r', or 'a'",
                "!",
            ),
            (
                'f"{x! r}"',
                "f-string: conversion type must come right after the exclamation mark",
                "!",
            ),
            (
                'f"a{a b}"',
                "f-string: expecting a single expression in a replacement field",
                "{",
            ),
            ('f"a}b"', "f-string: single '}' is not allowed", "}"),
            ('f"ab{x"', "f-string: expecting '}'", "{"),
            ('f"ab{x:{y}"', "f-string: expecting '}'", "{"),
            ('bf"x"', "invalid string prefix 'bf'", "b"),
        ],
    )
    def test_refused_like_python(self, lpy, msg, at):
        """Errors point at the offending field or character, as Python's do."""
        with pytest.raises(LispyCompileError) as info:
            src_to_python_org("\n  " + lpy)
        assert info.value.msg == msg
        assert (info.value.lineno, info.value.offset) == (2, 3 + lpy.index(at))

    @pytest.mark.skipif(
        sys.version_info < (3, 12), reason="3.11 positions f-string nodes differently"
    )
    @pytest.mark.parametrize(
        "src", ['f"a{x!r:>{width}}b"', 'f"{x=}"', 'f"ab{ x = !s:<{w}.{p}}"']
    )
    def test_positions_match_cpython(self, src):
        [stmt] = compile_lpy(src)
        assert ast.dump(stmt, include_attributes=True) == ast.dump(
            ast.parse(src).body[0], include_attributes=True
        )

    def test_keyword_argument_in_a_field(self):
        assert src_to_python_org('f"{(f :k 1)}"') == "f'{f(k=1)}'"

    def test_unlimited_nesting(self):
        lpy = 'f"{ {"a" {1 {2 "}"}}} } {(g {1 {2 "x"}})}"'
        py = "f\"{ {'a': {1: {2: '}'}}} } {g({1: {2: 'x'}})}\""
        assert stmt_to_dump(lpy) == ast.dump(ast.parse(py))

    def test_commas_inside_nested_strings_are_kept(self):
        assert src_to_python_org('f"{(f "a,b" 1,2)}"') == "f\"{f('a,b', 1, 2)}\""

    def test_field_positions(self):
        [stmt] = compile_lpy('(print\n  f"ab{(+ x 1)!r:>{w}}")')
        joined = stmt.value.args[0]
        [literal, field] = joined.values
        assert (literal.lineno, literal.col_offset, literal.end_col_offset) == (2, 4, 6)
        assert (field.lineno, field.col_offset, field.end_col_offset) == (2, 6, 22)
        assert (field.value.col_offset, field.value.end_col_offset) == (7, 14)
        assert field.value.left.col_offset == 10
        nested = field.format_spec.values[1]
        assert (nested.col_offset, nested.value.col_offset) == (18, 19)


class TestStringLineEndings:
    """Every string literal reads line endings as CPython reads a source file."""

    @pytest.mark.parametrize(
        "literal",
        [
            '"a\r\nb"',
            'r"a\r\nb"',
            '"a\rb"',
            'r"a\rb"',
            'b"a\r\nb"',
            'rb"a\r\nb"',
            '"a\\\r\nb"',
            'f"a\r\nb{x}"',
            'rf"a\r\n\\{x}"',
        ],
    )
    def test_value_matches_cpython(self, literal):
        lpy = f"(= x 1)\r\n(= s {literal})\r\n"
        scope = {}
        exec(compile(ast.Module(compile_lpy(lpy), []), "<lpy>", "exec"), scope)
        # Python needs triple quotes for a multi-line literal; the value rule is
        # the same.
        prefix, body = re.fullmatch(r'([A-Za-z]*)"(.*)"', literal, re.S).groups()
        python = {}
        source = f'x = 1\r\ns = {prefix}"""{body}"""\r\n'.encode()
        exec(compile(source, "<py>", "exec"), python)
        assert scope["s"] == python["s"]

    def test_macro_arguments_use_the_same_rules(self):
        src = '(defmacro width [s] (return (len s)))\r\n(= n (width "a\r\nb"))'
        scope = {}
        module = ast.Module(compile_lpy(src, scope), [])
        exec(compile(module, "<lpy>", "exec"), scope)
        assert scope["n"] == 3

    @pytest.mark.parametrize(
        ["literal", "msg"],
        [
            ('xyz"a"', "invalid string prefix 'xyz'"),
            ('b"\u00e9"', "bytes can only contain ASCII literal characters"),
        ],
    )
    def test_refused(self, literal, msg):
        with pytest.raises(LispyCompileError) as info:
            compile_lpy("\n  " + literal)
        assert info.value.msg.startswith(msg)
        assert (info.value.lineno, info.value.offset) == (2, 3)


def compile_lpy(src, scope=None):
    from lispy.tools import macroexpand_then_compile, parse

    return macroexpand_then_compile(
        parse(src), {"__macro_namespace": {}} if scope is None else scope
    )


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


class TestInterpolatedQuasiquote:
    """An f-string inside a quasiquote keeps conversion, spec and text."""

    def test_round_trip_through_a_macro(self):
        src = (
            "(= x 42)\n"
            '(defmacro show [] (return `f"[{x!r:>6}] {x=} {(+ x 1):{x}}"))\n'
            "(= s (show))"
        )
        scope = {}
        module = ast.Module(compile_lpy(src, scope), [])
        exec(compile(module, "<lpy>", "exec"), scope)
        assert scope["s"] == f"[{42!r:>6}] x=42 {43:42}"
