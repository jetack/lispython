import ast
import io
import re
import tokenize

from lispy.core.compiler.features import require
from lispy.core.errors import LispyCompileError

SPECIAL_CONSTANTS = ("True", "False", "None", "...")


def number_parts(text):
    """``[signs, real, op, imag]`` of a numeric token, or None if it is not one.

    The one place a signed or complex number token is split, read by Python's
    own tokenizer so hex/octal/binary, exponents and underscores split as
    Python splits them: ``-0x1e+2j`` is ``["-", "0x1e", "+", "2j"]``, and
    ``1e-5j`` is a single imaginary number.  ``op``/``imag`` are None for a
    plain number.
    """
    try:
        tokens = [
            t
            for t in tokenize.generate_tokens(io.StringIO(text).readline)
            if t.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER)
        ]
    except (tokenize.TokenError, SyntaxError):
        return None
    signs = ""
    while tokens and tokens[0].type == tokenize.OP and tokens[0].string in "+-":
        signs += tokens.pop(0).string
    kinds = [t.type for t in tokens]
    if kinds == [tokenize.NUMBER]:
        return [signs, tokens[0].string, None, None]
    complex_kinds = [tokenize.NUMBER, tokenize.OP, tokenize.NUMBER]
    if kinds == complex_kinds and tokens[1].string in ("+", "-"):
        return [signs, tokens[0].string, tokens[1].string, tokens[2].string]
    return None


def _apply_signs(value, signs):
    for sign in reversed(signs):
        value = -value if sign == "-" else +value
    return value


def constant_compile(constant):
    """A Constant node in an expression, with the value Python computes.

    ``-1+2j`` is ``(-1) + 2j``: the sign binds to the real part, as in Python.
    """
    text = constant.value
    if text in SPECIAL_CONSTANTS:
        return ast.Constant(value=ast.literal_eval(text), **constant.position_info)
    parts = number_parts(text)
    if parts is None:
        # Numbers a macro returned, written by str(): "(1+2j)", "-1.5".
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = None
        if type(value) not in (int, float, complex):
            raise LispyCompileError(f"invalid number literal {text}", constant)
        return ast.Constant(value=value, **constant.position_info)
    [signs, real, op, imag] = parts
    value = _apply_signs(ast.literal_eval(real), signs)
    if op:
        imag_value = ast.literal_eval(imag)
        value = value + imag_value if op == "+" else value - imag_value
    return ast.Constant(value=value, **constant.position_info)


def number_pattern_expr(text, sexp):
    """A number in a match pattern, shaped as ast.parse shapes it.

    ``-1`` is UnaryOp(USub, 1) on every version and ``+1`` UnaryOp(UAdd, 1)
    from 3.15; ``-1+2j`` is BinOp(UnaryOp(USub, 1), Add, 2j).  Python's
    complex-literal rules apply: a real part, then an imaginary one.
    """
    parts = number_parts(text)
    if parts is None or len(parts[0]) > 1:
        raise LispyCompileError(
            "only a number literal can be signed in a match pattern", sexp
        )
    [signs, real, op, imag] = parts
    position_info = sexp.position_info
    if signs == "+":
        require("match-unary-plus", sexp)
    real_value = ast.literal_eval(real)
    if op and isinstance(real_value, complex):
        raise LispyCompileError("real number required in complex literal", sexp)
    expr = ast.Constant(value=real_value, **position_info)
    if signs:
        unary = ast.USub if signs == "-" else ast.UAdd
        expr = ast.UnaryOp(unary(), expr, **position_info)
    if op:
        imag_value = ast.literal_eval(imag)
        if not isinstance(imag_value, complex):
            raise LispyCompileError(
                "imaginary number required in complex literal", sexp
            )
        binop = ast.Add if op == "+" else ast.Sub
        imag_expr = ast.Constant(value=imag_value, **position_info)
        expr = ast.BinOp(expr, binop(), imag_expr, **position_info)
    return expr


# Prefixes of a plain (non-interpolated) string literal, lower-cased.
_PLAIN_STRING_PREFIXES = ("", "r", "u", "b", "br", "rb")
_QUOTES = ('"""', "'''", '"', "'")


def string_literal_value(prefix, body):
    """The value of a string literal's body, as CPython computes it from source.

    The one home for string evaluation: plain strings, the literal stretches of
    f/t-strings and macro arguments all come here.  Like CPython reading a
    source file, every line ending (\\r\\n or a lone \\r) becomes \\n, raw
    strings included; a raw body is its own value; otherwise escapes are
    decoded by ast.literal_eval, which never runs code.
    """
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    lowered = prefix.lower()
    if "r" in lowered:
        return body.encode("ascii") if "b" in lowered else body
    src = []
    i = 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            src.append(body[i : i + 2])
            i += 2
        else:
            src.append({'"': '\\"', "\n": "\\n"}.get(body[i], body[i]))
            i += 1
    return ast.literal_eval(("b" if "b" in lowered else "") + '"' + "".join(src) + '"')


def string_node_value(string):
    """Value of a String node (its token text), or a LispyCompileError at it."""
    token = string.value
    prefix = re.match("[A-Za-z]*", token).group()
    rest = token[len(prefix) :]
    quote = next((q for q in _QUOTES if rest.startswith(q) and rest.endswith(q)), None)
    if quote is None or len(rest) < 2 * len(quote):
        raise LispyCompileError(f"invalid string literal {token}", string)
    if prefix.lower() not in _PLAIN_STRING_PREFIXES:
        raise LispyCompileError(f"invalid string prefix {prefix!r}", string)
    try:
        return string_literal_value(prefix, rest[len(quote) : -len(quote)])
    except (SyntaxError, ValueError, UnicodeEncodeError) as err:
        msg = err.msg if isinstance(err, SyntaxError) else str(err)
        raise LispyCompileError(msg, string) from err


def string_compile(string):
    return ast.Constant(value=string_node_value(string), **string.position_info)


def name_compile(symbol, ctx):
    [name, *attrs] = symbol.name.replace("-", "_").split(".")
    position_info = {**symbol.position_info}
    position_info["end_col_offset"] = position_info["col_offset"] + len(name)
    rst = ast.Name(id=name, ctx=ast.Load(), **position_info)
    for attr in attrs:
        position_info["end_col_offset"] += 1 + len(attr)
        rst = ast.Attribute(value=rst, attr=attr, ctx=ast.Load(), **position_info)
    rst.ctx = ctx()
    return rst
