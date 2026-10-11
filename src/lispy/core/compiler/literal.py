import ast
import re

from lispy.core.errors import LispyCompileError


def constant_compile(constant):
    return ast.Constant(value=eval(constant.value), **constant.position_info)


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
