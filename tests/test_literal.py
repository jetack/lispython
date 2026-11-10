import ast
import re
import sys
import warnings

import pytest

from lispy.core.compiler.literal import (
    completion_prefix,
    hyphen_spelling,
    hyphenated_name_p,
    identifier_p,
    name_key,
    name_spellings,
)
from lispy.core.errors import LispyCompileError, compile_errors
from lispy.core.macro import macroexpand_then_compile
from lispy.core.nodes import Symbol
from lispy.tools import parse, src_to_python_org

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
        # A string in a field uses the f-string's own quote, as in Python 3.12+.
        assert stmt_to_dump(
            'f"Started at {(.strftime (datetime.now) "%Y-%m-%d %H:%M:%S")}"'
        ) == ast.dump(
            ast.parse("f\"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\"")
        )

    def test_f_string_with_colon_in_nested_string(self):
        # Colon inside nested string should not be treated as format spec
        assert stmt_to_dump('f"time: {(.get d "H:M:S")}"') == ast.dump(
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
            ('f"{}"', "f-string: valid expression required before '}'", "}"),
            ('f"{x!}"', "f-string: missing conversion character", "}"),
            (
                'f"{x!z}"',
                "f-string: invalid conversion character 'z': expected 's', 'r', or 'a'",
                "z",
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
    from lispy.core.macro import macroexpand_then_compile
    from lispy.tools import parse

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


class TestFormatSpecMacros:
    """A macro call inside a replacement field's format spec expands like one
    in the field's value."""

    def test_spec_field_expands(self):
        src = '(defmacro m [] (return 5))\n(print f"{1:{(m)}}")'
        assert src_to_python_org(src) == "print(f'{1:{5}}')"

    def test_spec_field_runs(self):
        src = '(defmacro w [] (return 4))\n(= r f"{1:>{(w)}}|{2:{(w)}d}")'
        assert run_lpy(src) == f"{1:>4}|{2:4d}"

    def test_quasiquote_built_spec_expands(self):
        src = (
            "(defmacro w [] (return 3))\n"
            '(defmacro show [x] (return `f"{~x:>{(w)}}"))\n'
            "(= r (show 7))"
        )
        assert run_lpy(src) == f"{7:>3}"

    def test_quasiquoted_spec_stays_data(self):
        # Inside a quasiquote the spec is a template like the value: (m) stays
        # a form, while an unquoted (m) expands.
        from lispy.core import builtins
        from lispy.core.nodes import atom_text

        scope = {name: getattr(builtins, name) for name in builtins.__all__}
        scope["__macro_namespace"] = {}
        src = '(defmacro m [] (return 5))\n(= r `f"{1:{(m)}}{2:{~(m)}}")'
        module = ast.Module(compile_lpy(src, scope), [])
        exec(compile(module, "<lpy>", "exec"), scope)
        quoted, unquoted = (
            scope["r"].list[i].format_spec.list[2].value for i in (2, 4)
        )
        assert [atom_text(x) for x in quoted.list] == ["m"]
        assert atom_text(unquoted) == "5"

    @pytest.mark.skipif(sys.version_info < (3, 14), reason="t-strings need 3.14")
    def test_t_string_spec_field_expands(self):
        template = run_lpy('(defmacro w [] (return 4))\n(= r t"{1:>{(w)}}")')
        assert template.interpolations[0].format_spec == ">4"


class TestNameSpellings:
    def test_spelling_in_a_format_spec_field(self):
        spellings = name_spellings(parse('(print f"{1:{ok?}}")'))
        assert spellings["ok_p"] == "ok?"


class TestMalformedLiterals:
    """A malformed literal is reported at its token, with CPython's message."""

    @pytest.mark.parametrize(
        ["lpy", "msg", "col"],
        [
            (
                "(print 09)",
                "leading zeros in decimal integer literals are not permitted; "
                "use an 0o prefix for octal integers",
                10,
            ),
            ("(print 0x)", "invalid hexadecimal literal", 10),
            (
                "(match x (case 09 ...))",
                "leading zeros in decimal integer literals are not permitted; "
                "use an 0o prefix for octal integers",
                18,
            ),
            (
                '(print f"bad \\x4 {1}")',
                "(unicode error) 'unicodeescape' codec can't decode bytes in "
                "position 4-6: truncated \\xXX escape",
                12,
            ),
            (
                '(print "bad \\x4")',
                "(unicode error) 'unicodeescape' codec can't decode bytes in "
                "position 4-6: truncated \\xXX escape",
                10,
            ),
            ('(print rb"é")', "bytes can only contain ASCII literal characters", 10),
            (
                "(defmacro m [x] (return x))\n  (print (m 09))",
                "leading zeros in decimal integer literals are not permitted; "
                "use an 0o prefix for octal integers",
                13,
            ),
        ],
    )
    def test_positioned(self, lpy, msg, col):
        with pytest.raises(LispyCompileError) as info:
            compile_lpy("\n  " + lpy)
        assert info.value.msg == msg
        assert (info.value.lineno, info.value.offset) == (
            lpy.count("\n") + 2,
            col,
        )


class TestUnquotedFields:
    """An unquoted f-string field holds the expanded expression, and its text is
    what the user would have written after expansion."""

    def run(self, src):
        scope = {}
        module = ast.Module(compile_lpy(src, scope), [])
        exec(compile(module, "<lpy>", "exec"), scope)
        return scope

    def test_literal_argument_as_the_whole_field(self):
        scope = self.run(
            '(defmacro show-val [x] (return `f"<{~x}>"))\n(= s (show-val 41))'
        )
        assert scope["s"] == "<41>"

    @pytest.mark.parametrize(
        ["field", "expected"],
        [
            ("{~x=}", "(+ 40 1)=41"),
            ("{ ~x = }", " (+ 40 1) = 41"),
            ("{~x=!s:>4}", "(+ 40 1)=  41"),
        ],
    )
    def test_self_documenting_echo_uses_the_expanded_text(self, field, expected):
        scope = self.run(
            f'(defmacro dbg [x] (return `f"{field}"))\n(= s (dbg (+ 40 1)))'
        )
        assert scope["s"] == expected

    def test_written_field_keeps_its_text(self):
        scope = self.run('(= y 2)\n(defmacro dbg [] (return `f"{y=}"))\n(= s (dbg))')
        assert scope["s"] == "y=2"


# 3.11 warns an invalid escape with DeprecationWarning, 3.12+ SyntaxWarning.
ESCAPE_WARNING = DeprecationWarning if sys.version_info < (3, 12) else SyntaxWarning


class TestInvalidEscapeWarning:
    """An invalid escape warns at the .lpy string token, never at
    ``<unknown>`` line 1, the text literal_eval was handed."""

    @pytest.mark.parametrize(
        "src", ['(= a 1)\n(print "\\d")', '(= a 1)\n(print f"\\d{a}")']
    )
    def test_warning_names_the_lpy_line(self, src):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with compile_errors("m.lpy", src):
                macroexpand_then_compile(parse(src), {})
        [warning] = caught
        assert warning.category is ESCAPE_WARNING
        # Python's own wording, which changes between versions, passes through.
        assert "invalid escape sequence" in str(warning.message)
        assert "\\d" in str(warning.message)
        assert (warning.filename, warning.lineno) == ("m.lpy", 2)

    @pytest.mark.parametrize(
        ["src", "msg", "col"],
        [
            ("(= a 1if)", "invalid decimal literal", 6),
            ("(= a 1else)", "invalid decimal literal", 6),
            ("(= a 1e5if)", "invalid decimal literal", 6),
            ("(= a 1in2)", "invalid decimal literal", 6),
            ("(= a 0x1for)", "invalid hexadecimal literal", 6),
            ("(= a 1jor)", "invalid imaginary literal", 6),
            ("(= a 0b1and)", "invalid binary literal", 6),
            ("(= a 0o7in)", "invalid octal literal", 6),
            ("(match x (case 1if 2))", "invalid decimal literal", 16),
            ("(match x (case 0x1for 2))", "invalid hexadecimal literal", 16),
            # Digit-led symbols, refused as names: worded the same.
            ("(= a 1.else)", "invalid decimal literal", 6),
            ("(= a 1.e5if)", "invalid decimal literal", 6),
            ("(= a 1__0)", "invalid decimal literal", 6),
            ("(= a 1_)", "invalid decimal literal", 6),
            ("(= a 1+2jif)", "invalid imaginary literal", 6),
            ("(match x (case 1__0 2))", "invalid decimal literal", 16),
            ("(match x (case 1.5if 2))", "invalid decimal literal", 16),
            ("(match x (case 1.else 2))", "invalid decimal literal", 16),
            ("(match x (case 1.e5if 2))", "invalid decimal literal", 16),
            ("(match x (case {1.5if 2} 2))", "invalid decimal literal", 17),
            ("(match x (case (1.5if) 2))", "invalid decimal literal", 17),
            ("(match x (case (1else :a b) 2))", "invalid decimal literal", 17),
            ("(match x (case (a.b :1.5if k) 2))", "invalid decimal literal", 22),
            # The head is refused before the operands are read.
            ("(match x (case (1.5if :a) 2))", "invalid decimal literal", 17),
            # A number token CPython reads as more than one token.
            ("(= a 406E-9.8J)", "invalid number literal 406E-9.8J", 6),
            ("(= a 1e+7.J)", "invalid number literal 1e+7.J", 6),
            # No SyntaxError from literal_eval: still no valid name.
            ("(= a 1-x)", "1-x is not a valid Python name", 6),
        ],
    )
    def test_a_malformed_number_is_one_positioned_error_and_no_warning(
        self, src, msg, col
    ):
        # literal_eval warns "invalid decimal literal" (naming <unknown>
        # line 1) before raising "invalid syntax": the error at the token
        # takes the warning's word, as compile() raises it under -W error.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(LispyCompileError) as info:
                stmt_to_dump(src)
        assert [str(w.message) for w in caught] == []
        assert (info.value.msg, info.value.lineno, info.value.offset) == (msg, 1, col)

    @pytest.mark.parametrize(
        ["pattern", "prefix"],
        [
            ("a..b", "pattern expected, got"),
            ("1.0.0", "pattern expected, got"),
            ("(a..b)", "class pattern needs a class name, got"),
            ("(1)", "class pattern needs a class name, got"),
            ("(a.b :a..c k)", "class pattern expects an attribute name after :"),
        ],
    )
    def test_a_dotted_pattern_without_a_number_shape_keeps_the_pattern_error(
        self, pattern, prefix
    ):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump(f"(match x (case {pattern} 2))")
        assert info.value.msg.startswith(prefix)

    @pytest.mark.usefixtures("time_bound")
    @pytest.mark.parametrize(
        "name",
        [
            # No number shape: literal_eval would overflow its parser on
            # either (RecursionError or MemoryError, by version).
            "1" + "-" * 7000 + "x",
            "+".join(["1"] * 3001),
            # Nor an attribute chain, each "." one more level.
            "1e5" + ".real" * 3000,
            "1e5" + ".real" * 100000,
            "1j" + ".imag" * 3000,
            "1-1e5" + ".real" * 3000,
            # Number-shaped, but CPython's word is no word for a token.
            "1.0.0",
            "1..2",
            # Parsed as Python, "invalid syntax" or "'(' was never closed".
            "1=x",
            "1:x",
            "1(",
        ],
        ids=lambda name: name if len(name) < 20 else f"{name[:12]}...({len(name)})",
    )
    def test_a_digit_led_name_not_worded_as_a_token_is_no_valid_name(self, name):
        with pytest.raises(LispyCompileError) as info:
            stmt_to_dump(f"(= a {name})")
        # Compared apart, so a failure does not print a 500 kB name.
        worded = info.value.msg == f"{name} is not a valid Python name"
        assert worded, info.value.msg[:120]
        assert (info.value.lineno, info.value.offset) == (1, 6)

    def test_a_literal_without_a_backslash_keeps_the_warning_registry(self):
        # Swapping the warning filters resets every "warn once" registry, so a
        # REPL session would repeat warnings it already showed after each
        # string it compiles.
        registry = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            warnings.warn_explicit("once", UserWarning, "f", 1, registry=registry)
            stmt_to_dump('(= s "plain") (= n 1.5)')
            warnings.warn_explicit("once", UserWarning, "f", 1, registry=registry)
        assert len(caught) == 1


class TestPlainStringExtent:
    """Plain strings end where CPython's do: the f/t-string scanner finds them."""

    @pytest.mark.parametrize(
        "literal",
        [
            '"a\\\nb"',
            '"""a"b"""',
            '"""say "hi" """',
            "'''it's'''",
            'r"a\\"b"',
            'b"""a"b"""',
            '"x{y}"',
            'f"""a"{x}"""',
        ],
    )
    def test_value_matches_cpython(self, literal):
        scope = {}
        module = ast.Module(compile_lpy(f"(= x 1)\n(= s {literal})"), [])
        exec(compile(module, "<lpy>", "exec"), scope)
        python = {}
        exec(f"x = 1\ns = {literal}", python)
        assert scope["s"] == python["s"]

    @pytest.mark.parametrize(
        ["lpy", "kind"],
        [
            ('(= s "abc', "string"),
            ('(= s """abc"', "triple-quoted string"),
            ('(= s f"abc', "f-string"),
        ],
    )
    def test_unterminated(self, lpy, kind):
        # At the string's prefix (column 6 here).
        with pytest.raises(LispyCompileError) as info:
            compile_lpy(lpy)
        assert info.value.msg == f"unterminated {kind} literal (detected at line 1)"
        assert (info.value.lineno, info.value.offset) == (1, 6)


class TestHyphenSpelling:
    """hyphen_spelling, the spelling completion offers for a Python name."""

    EDGE_NAMES = [
        "__init__",
        "_x",
        "class_",
        "x_",
        "a__b",
        "_private_helper",
        "__a_b__",
        "MAX_SIZE",
        "x_1",
        "é_x",
    ]

    @staticmethod
    def real_names():
        import builtins
        import collections
        import heapq
        import os

        names = set(dir(builtins)) | set(dir(heapq)) | set(dir(collections))
        names |= set(dir(os))
        return sorted(
            n for n in names | set(TestHyphenSpelling.EDGE_NAMES) if identifier_p(n)
        )

    def test_the_spelling_compiles_back_to_the_name(self):
        # Defect: an offered spelling that is not the name it was offered
        # for (a__b offered as a--b, which name_key, the LSP's and the macro
        # clash check's name identity, reads as another name), so accepting
        # a completion binds or calls something else.
        names = self.real_names()
        spellings = [hyphen_spelling(n) for n in names]
        for name, spelled in zip(names, spellings):
            assert name_key(spelled) == name, spelled
            assert Symbol(spelled).name == name, spelled
            assert hyphenated_name_p(spelled), spelled
        source = "\n".join(f"(= {s} 1)" for s in spellings)
        assert src_to_python_org(source) == "\n".join(f"{n} = 1" for n in names)

    @pytest.mark.parametrize(
        "text", ["__init__", "_x", "class_", "x_", "a__b", "sep=", "->", "my-x", "_"]
    )
    def test_never_converted(self, text):
        # Defects: a dunder offered as --init--, an edge or doubled
        # underscore turned into a hyphen (no hyphenated name, so another
        # name), or an operator or already hyphenated text rewritten.
        assert hyphen_spelling(text) == text

    @pytest.mark.parametrize(
        "name, spelled",
        [
            ("move_to_end", "move-to-end"),
            ("_private_helper", "_private-helper"),
            ("__a_b__", "__a-b__"),
            ("mod.my_mac", "mod.my-mac"),
            ("mod.->", "mod.->"),
        ],
    )
    def test_interior_underscores_become_hyphens(self, name, spelled):
        # Defect: a name offered in its Python spelling (move_to_end), or a
        # dotted name converted as one part.
        assert hyphen_spelling(name) == spelled


class TestTypedInterpolatedHeads:
    """A typed ``(f-string ...)`` head is not reader-built: parts out of place
    are refused at the part, never an AttributeError."""

    @pytest.mark.parametrize(
        ["src", "msg", "col"],
        [
            (
                '(f-string "a" 1 "b")',
                "f-string: part 2 must be a replacement field",
                17,
            ),
            ('(f-string "a" "b")', "f-string: part 2 must be a replacement field", 17),
            ("(f-string 5)", "f-string: part 1 must be a string literal", 13),
            ('(f-string b"a")', "f-string: part 1 must be a string literal", 13),
        ],
    )
    def test_refused(self, src, msg, col):
        with pytest.raises(LispyCompileError) as info:
            compile_lpy("\n  " + src)
        assert info.value.msg.startswith(msg)
        assert (info.value.lineno, info.value.offset) == (2, col)

    def test_macro_built_bytes_part_refused(self):
        src = '(defmacro mk [s] (return `(f-string ~s)))\n(= a (mk b"x"))'
        with pytest.raises(LispyCompileError) as info:
            compile_lpy(src)
        assert info.value.msg.startswith("f-string: part 1 must be a string literal")
        assert (info.value.lineno, info.value.offset) == (2, 6)

    @pytest.mark.skipif(sys.version_info < (3, 14), reason="t-strings need 3.14")
    def test_t_string_refused(self):
        with pytest.raises(LispyCompileError) as info:
            compile_lpy('(t-string "a" 1 "b")')
        assert info.value.msg.startswith("t-string: part 2 must be a replacement")

    @pytest.mark.parametrize("src", ["(f-string)", '(f-string "a")'])
    def test_valid_heads_compile(self, src):
        assert compile_lpy(src)


def macro_field_src(head, field_args="", tail='(String "\\"\\"")'):
    """A module whose macro ``m`` (called on line 3) returns a hand-built
    ``(head "a" {x} tail)`` node, its field given ``field_args``."""
    return (
        f'(defmacro m [] (return (Paren (Symbol "{head}") (String "\\"a\\"")'
        f' (FStrExpr (Symbol "x") {field_args}) {tail})))\n(= x 7)\n(= r (m))'
    )


def run_lpy(src):
    namespace = {}
    module = ast.Module(compile_lpy(src), type_ignores=[])
    exec(compile(module, "<test>", "exec"), namespace)
    return namespace["r"]


class TestMacroBuiltFields:
    """A macro builds an FStrExpr with any slot values: the defaults compile
    as the reader's field would, and a slot the reader never fills that way is
    a LispyCompileError at the macro call, never an AttributeError/TypeError."""

    def test_default_slots_compile(self):
        assert run_lpy(macro_field_src("f-string")) == "a7"

    def test_given_slots_compile(self):
        spec = '(Paren (Symbol "f-string") (String "\\">3\\""))'
        args = f':conversion 114 :format-spec {spec} :debug "="'
        assert run_lpy(macro_field_src("f-string", args)) == "ax=  7"

    @pytest.mark.skipif(sys.version_info < (3, 14), reason="t-strings need 3.14")
    def test_default_slots_compile_in_t_string(self):
        template = run_lpy(macro_field_src("t-string"))
        assert (template.strings, template.values) == (("a", ""), (7,))

    @pytest.mark.parametrize(
        ["args", "msg"],
        [
            (":conversion 5", "f-string: replacement field conversion"),
            (':conversion "r"', "f-string: replacement field conversion"),
            (":conversion 115.0", "f-string: replacement field conversion"),
            (":debug 5", "f-string: replacement field debug text"),
            (
                ':format-spec (String "\\"x\\"")',
                "f-string: replacement field format spec",
            ),
            (
                ':format-spec (Paren (Symbol "t-string") (String "\\"x\\""))',
                "f-string: replacement field format spec",
            ),
            # Data in the spec is never converted to a node (_child_form_keys).
            (':format-spec ">3"', "f-string: replacement field format spec"),
            (":format-spec [1]", "f-string: replacement field format spec"),
        ],
    )
    def test_bad_slot_refused(self, args, msg):
        with pytest.raises(LispyCompileError) as info:
            compile_lpy(macro_field_src("f-string", args))
        assert info.value.msg.startswith(msg)
        assert info.value.lineno == 3

    @pytest.mark.parametrize(
        "src",
        [
            macro_field_src("f-string", tail="(String 3)"),
            # Slot 1 is read by the form dispatch (atom_text) before any
            # f-string code runs.
            '\n(defmacro m [] (return (Paren (Symbol "f-string") (String 3))))'
            "\n(= r (m))",
            '\n(defmacro m [] (return (Paren (Symbol "print") (Symbol 3))))\n(= r (m))',
            "\n(defmacro m [] (return (Paren (Symbol 3))))\n(= r (m))",
            '\n(defmacro m [] (return (Paren (Symbol "print") (Constant None))))'
            "\n(= r (m))",
        ],
    )
    def test_non_str_atom_refused(self, src):
        # A macro may build an atom around any value: refused at the call,
        # never a TypeError wherever the compiler first reads its text.
        with pytest.raises(LispyCompileError) as info:
            compile_lpy(src)
        assert "an atom's text must be a str" in info.value.msg
        assert info.value.lineno == 3

    @pytest.mark.parametrize(
        "use",
        ["(= r (m))", "(m)", "(= (m) 1)", "(del (m))", "(for (m) in [1] (pass))"],
    )
    def test_field_outside_interpolation_refused(self, use):
        # Expression, statement and target positions all reach expr_steps.
        src = f'(defmacro m [] (return (FStrExpr (Symbol "x"))))\n\n{use}'
        with pytest.raises(LispyCompileError) as info:
            compile_lpy(src)
        assert info.value.msg.startswith("a replacement field (FStrExpr) is valid")
        assert info.value.lineno == 3

    @pytest.mark.parametrize("head", ["f-string", "t-string"])
    def test_spec_part_names_its_owner(self, head):
        if head == "t-string" and sys.version_info < (3, 14):
            pytest.skip("t-strings need 3.14")
        spec = '(Paren (Symbol "f-string") (String "\\"x\\"") (Symbol "y"))'
        with pytest.raises(LispyCompileError) as info:
            compile_lpy(macro_field_src(head, f":format-spec {spec}"))
        assert info.value.msg.startswith(
            f"{head} format spec: part 2 must be a replacement field"
        )
        assert info.value.lineno == 3


class TestCompletionPrefix:
    """completion_prefix, the rule the LSP completion filters items by."""

    @pytest.mark.parametrize(
        "typed, label",
        [
            ("my-", "my-var"),
            ("my_", "my-var"),
            ("my_val-x", "my-val-x"),
            ("my-val_x", "my-val-x"),
            ("my-v", "my_var"),
            ("MY-V", "my-var"),
            ("Ok", "ok?"),
            ("->f", "->foo"),
            ("_", "_x"),
            ("", "anything"),
            ("my-var", "my-var"),
        ],
    )
    def test_matching_labels_keep_the_typed_word(self, typed, label):
        # Defects: a separator spelt the other way (`my_v` against `my-var`)
        # or a case difference filtering the item out, or a filterText that
        # does not start with the word as typed, which a client's own prefix
        # match then drops.
        assert completion_prefix(typed, label) == typed + label[len(typed) :]

    @pytest.mark.parametrize(
        "typed, label",
        [
            ("-", "_x"),
            ("_", "-x"),
            ("-f", "_foo"),
            ("my-var", "my-va"),
            ("my--", "my-_x"),
            ("myv", "my-var"),
            ("x", "ax"),
        ],
    )
    def test_other_labels_do_not_match(self, typed, label):
        # Defects: a leading separator matched loosely (`-` offering every
        # `_private`), a separator after a separator treated as a word
        # break, a shorter label compared past its end, a dropped separator
        # or an infix match.
        assert completion_prefix(typed, label) is None
