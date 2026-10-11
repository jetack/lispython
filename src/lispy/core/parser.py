import ast
import re
from collections import deque
from functools import reduce

from lispy.core.errors import LispyCompileError
from lispy.core.nodes import *
from lispy.core.nodes import FStrExpr, Paren, String, Symbol
from lispy.core.utils import augassignop_dict


def tokenize(src):
    # Remove commas except those following { or ( with optional whitespace, but preserve commas in strings
    src = re.sub(
        r'(\w*"""(?:[^\\"]|\\.)*"""|\w*\'\'\'(?:[^\\\']|\\.)*\'\'\'|\w*"(?:[^\\"]|\\.)*"|[{(]\s*,)|,',
        lambda m: m.group(1) if m.group(1) else " ",
        src,
    )
    lines = src.split("\n")
    tokens = deque([])
    int_simple = r"\d+[\dA-Za-z]*(?:_[\dA-Za-z]+)*"
    float_simples = [
        "\\d+(?:_\\d+)*\\.\\d+(?:_\\d+)*",
        "\\d+(?:_\\d+)*\\.\\d*",
        "\\d*\\.\\d+(?:_\\d+)*",
    ]
    scientific_simples = list(
        map(lambda x: x + "e[\\-\\+]?\\d+(?:_\\d+)*", float_simples + [int_simple])
    )
    number_simple = "|".join(scientific_simples + float_simples + [int_simple])
    complex_simple = f"(?:(?:{number_simple})[+-])?(?:{number_simple})j"
    # F-string pattern with nested quotes in braces: f"...{expr "str"}..."
    inner_str = r'\"(?:[^\"\\]|\\.)*\"'
    brace_content = r'(?:[^{}\"\\]|\\.|\{[^{}]*\}|' + inner_str + r')*'
    brace_expr = r'\{' + brace_content + r'\}'
    f_string_content = r'(?:[^\"{}\\]|\\.|' + brace_expr + r')*'
    # (?!"") leaves triple-quoted f/t-strings to the triple-quote pattern below.
    f_string_pattern = (
        r'(?:[rR]?[fFtT]|[fFtT][rR])\"(?!\"\")' + f_string_content + r'\"'
    )
    pattern_labels = [
        ["\\^?\\*{0,2}[\\(\\{\\[]", "opening"],
        ["[\\)\\}\\]]", "closing"],
        # F-string pattern must come before regular string patterns
        [f_string_pattern, '"'],
        *list(
            zip(
                map(
                    lambda x: "\\w*" + x,
                    [
                        "\\'\\'\\'(?:[^\\\\']|\\\\.)*\\'\\'\\'",
                        '\\"\\"\\"(?:[^\\\\\\"]|\\\\.)*\\"\\"\\"',
                        '\\"(?:[^\\\\\\"]|\\\\.)*\\"',
                    ],
                ),
                ["'''", '"""', '"'],
            )
        ),
        ["\\^?\\*{0,2}(?:'|`|~@|~)(?!\\s|$)", "meta-indicator"],
        [";[^\\n]*", "comment"],
        [
            "[+-]*(?:"
            + "|".join([complex_simple, number_simple])
            + ")(?=\\s|$|\\)|\\}|\\])",
            "number",
        ],
        ['[^\\s\\)\\}\\]\\"]+', "symbol"],
        ["\\n", "new-line"],
        [" +", "spaces"],
    ]
    [patterns, labels] = zip(*pattern_labels)
    combined_pattern = "|".join(map(lambda p: f"({p})", patterns))
    lineno = 1
    col_offset = 0
    re_applied = reduce(
        lambda x, y: x + y,
        map(
            lambda x: list(filter(lambda y: y[0], zip(x, labels))),
            re.findall(combined_pattern, src),
        ),
        [],
    )
    for [tk, tktype] in re_applied:
        splitted = tk.split("\n")
        [num_newline, col_shift] = [len(splitted) - 1, len(splitted[-1])]
        end_lineno = lineno + num_newline
        end_col_offset = col_shift + (col_offset if num_newline == 0 else 0)
        tokens.append(
            [
                tk,
                tktype,
                {
                    "lineno": lineno,
                    "col_offset": col_offset,
                    "end_lineno": end_lineno,
                    "end_col_offset": end_col_offset,
                },
            ]
        ) if not tktype in ["new-line", "spaces", "comment"] else None
        lineno = end_lineno
        col_offset = end_col_offset
    return tokens


def position_info_into_list(position_info):
    return list(
        map(
            lambda x: position_info[x],
            ["lineno", "col_offset", "end_lineno", "end_col_offset"],
        )
    )


wrapper_dict = {
    "": lambda value, **kwargs: value,
    "*": Starred,
    "**": DoubleStarred,
    "^": Annotation,
}
opening_dict = {"(": Paren, "{": Brace, "[": Bracket}
meta_indicator_dict = {"'": Quote, "`": QuasiQuote, "~": Unquote, "~@": UnquoteSplice}


def parse(src):
    tokens = tokenize(src)
    stack = []
    rst = []
    while tokens:
        [t, tktype, position_info] = tokens.popleft()
        [lineno, col_offset, end_lineno, end_col_offset] = position_info_into_list(
            position_info
        )
        if tktype == "opening":
            wrapper = t[:-1]
            opening = t[-1]
            if wrapper:
                stack.append(
                    wrapper_dict[wrapper](
                        value=None, lineno=lineno, col_offset=col_offset
                    )
                )
            stack.append(
                opening_dict[opening](
                    lineno=lineno, col_offset=col_offset + len(wrapper)
                )
            )
        elif tktype == "meta-indicator":
            wrapper = (
                "**"
                if t.startswith("**")
                else "*"
                if t.startswith("*")
                else "^"
                if t.startswith("^")
                else ""
            )
            meta_ind = t[len(wrapper) :]
            if wrapper:
                stack.append(
                    wrapper_dict[wrapper](
                        value=None, lineno=lineno, col_offset=col_offset
                    )
                )
            stack.append(
                meta_indicator_dict[meta_ind](
                    value=None, lineno=lineno, col_offset=col_offset + len(wrapper)
                )
            )
        else:
            if tktype == "closing":
                e = stack.pop()
                e.update_dict("end_lineno", end_lineno)
                e.update_dict("end_col_offset", end_col_offset)
            else:
                e = token_parse(t, tktype, position_info)
            while stack and (
                isinstance(stack[-1], MetaIndicator) or isinstance(stack[-1], Wrapper)
            ):
                popped = stack.pop()
                popped.value = e
                popped.update_dict("end_lineno", end_lineno)
                popped.update_dict("end_col_offset", end_col_offset)
                e = popped
            (stack[-1] if stack else rst).append(e)
    return rst


special_literals = ("True", "False", "None", "...")


def token_parse(token, tktype, position_info):
    return (
        Symbol(token, **position_info)
        if token in augassignop_dict
        else Constant(token, **position_info)
        if token in special_literals
        else Constant(token, **position_info)
        if tktype == "number" and (not token[0] in "+-")
        else string_parse(token, tktype, position_info)
        if tktype in ["'''", '"""', '"']
        else Symbol(token, **position_info)
        if len(token) < 2
        else annotation_token_parse(token, tktype, position_info)
        if token[0] == "^"
        else keyword_token_parse(token, tktype, position_info)
        if token[0] == ":"
        else star_token_parse(token, tktype, position_info)
        if token[0] == "*"
        else unary_op_parse(token, tktype, position_info)
        if token[0] in "+-" and (lambda rest: rest and (rest[0].isalnum() or rest[0] in "_."))(token.lstrip("+-"))
        else Symbol(token, **position_info)
    )


def annotation_token_parse(token, tktype, position_info):
    inner_position = {**position_info}
    inner_position["col_offset"] += 1
    return Annotation(token_parse(token[1:], tktype, inner_position), **position_info)


def keyword_token_parse(token, tktype, position_info):
    inner_position = {**position_info}
    inner_position["col_offset"] += 1
    return Keyword(token_parse(token[1:], tktype, inner_position), **position_info)


def star_token_parse(token, tktype, position_info):
    num_star = 2 if token[1] == "*" else 1
    inner_position = {**position_info}
    inner_position["col_offset"] += num_star
    return wrapper_dict["*" * num_star](
        token_parse(token[slice(num_star, None)], tktype, inner_position),
        **position_info,
    )


def unary_op_parse(token, tktype, position_info):
    stack = []
    idx = 0
    lineno = position_info["lineno"]
    col_offset = position_info["col_offset"]
    while token[idx] in "+-":
        stack.append(
            Symbol(
                token[idx],
                **{
                    "lineno": lineno,
                    "end_lineno": lineno,
                    "col_offset": col_offset + idx,
                    "end_col_offset": col_offset + idx + 1,
                },
            )
        )
        idx += 1
    position_info["col_offset"] += idx
    rst = token_parse(token[slice(idx, None)], tktype, {**position_info})
    while stack:
        position_info["col_offset"] -= 1
        rst = Paren(stack.pop(), rst, **position_info)
    return rst


def string_parse(token, tktype, position_info):
    [prefix, *contents, _] = token.split(tktype)
    node = String(token, **position_info)
    kind = interpolated_kind(prefix, node)
    return (
        interpolated_parse(
            kind, "r" in prefix.lower(), tktype.join(contents), position_info
        )
        if kind
        else node
    )


def interpolated_kind(prefix, node):
    """``"f-string"``/``"t-string"`` for an f/t-prefixed literal, else None.

    Python's rules: f/F or t/T, alone or with r/R in either order and case;
    f, t, b and u never combine.  Any other prefix containing f or t is
    refused here, so it can never reach the eval() of a plain string literal
    (which would evaluate a t-string's interpolations inside the compiler).
    """
    lowered = prefix.lower()
    if "t" in lowered:
        if lowered in ("t", "rt", "tr"):
            return "t-string"
        for other in "fbu":
            if other in lowered:
                raise LispyCompileError(
                    f"'{other}' and 't' prefixes are incompatible", node
                )
    elif "f" not in lowered:
        return None
    elif lowered in ("f", "rf", "fr"):
        return "f-string"
    raise LispyCompileError(f"invalid string prefix {prefix!r}", node)


conversion_dict = {"s": 115, "r": 114, "a": 97}


def _interpolation_error(kind, msg, position_info):
    return LispyCompileError(f"{kind}: {msg}", String('""', **position_info))


def scan_interpolated(content, raw, kind, position_info):
    """Split an f/t-string body into ``[literal, field, literal, ..., literal]``.

    Literals are still source text (escapes unprocessed) with ``{{``/``}}``
    collapsed to one brace; fields are the text between a replacement field's
    braces, nested braces (format specs) included.
    """
    pieces = []
    buf = []
    i = 0
    n = len(content)
    while i < n:
        c = content[i]
        if c == "\\" and i + 1 < n:
            nxt = content[i + 1]
            if not raw and content.startswith("N{", i + 1):
                end = content.find("}", i + 3)
                end = n - 1 if end < 0 else end
                buf.append(content[i : end + 1])
                i = end + 1
            elif nxt in "{}":
                # A backslash never escapes a brace: it is a literal backslash.
                buf.append("\\" if raw else "\\\\")
                i += 1
            else:
                buf.append(content[i : i + 2])
                i += 2
        elif c == "{" and content.startswith("{", i + 1):
            buf.append("{")
            i += 2
        elif c == "}" and content.startswith("}", i + 1):
            buf.append("}")
            i += 2
        elif c == "{":
            end = _field_end(content, i + 1, kind, position_info)
            pieces.append("".join(buf))
            pieces.append(content[i + 1 : end])
            buf = []
            i = end + 1
        elif c == "}":
            raise _interpolation_error(kind, "single '}' is not allowed", position_info)
        else:
            buf.append(c)
            i += 1
    pieces.append("".join(buf))
    return pieces


def _field_end(content, start, kind, position_info):
    """Index of the ``}`` closing the replacement field that starts at ``start``."""
    depth = 0
    quote = None
    i = start
    n = len(content)
    while i < n:
        if quote:
            if content.startswith(quote, i):
                i += len(quote)
                quote = None
            else:
                i += 2 if content[i] == "\\" else 1
            continue
        if content.startswith('\\"', i):
            quote = '\\"'
            i += 2
            continue
        c = content[i]
        if c == '"':
            quote = '"'
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0 and c == "}":
                return i
            depth -= 1
        i += 1
    raise _interpolation_error(kind, "expecting '}'", position_info)


def _literal_value(piece, raw):
    """The str value of a literal stretch of an f/t-string body."""
    if raw:
        return piece
    src = []
    i = 0
    while i < len(piece):
        if piece[i] == "\\" and i + 1 < len(piece):
            src.append(piece[i : i + 2])
            i += 2
        else:
            src.append({'"': '\\"', "\n": "\\n", "\r": "\\r"}.get(piece[i], piece[i]))
            i += 1
    return ast.literal_eval('"' + "".join(src) + '"')


# A trailing "=" after one of these is part of an operator symbol, not a
# self-documenting "=" (``{x=}``).
_DEBUG_BLOCKERS = "=!<>:+-*/%&|^@~"


def field_parse(field, kind, raw, position_info):
    """Parse one replacement field into ``(FStrExpr, debug_text)``.

    Follows Python: ``expr[=][!conv][:spec]``; whitespace around ``expr`` is
    part of its text; ``=`` echoes the text before it (and implies ``!r``
    unless a conversion or spec is given); the spec is itself an f-string body.
    """
    # \" is the escape for a quote inside braces (see the tokenizer).
    text = field.replace('\\"', '"')
    depth = 0
    in_string = False
    expr_end = len(text)
    delimiter = "}"
    conversion = None
    spec = None
    i = 0
    while i < len(text):
        c = text[i]
        if in_string:
            if c == "\\":
                i += 1
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0 and c == "!" and not text.startswith("!=", i):
            expr_end = i
            delimiter = "!"
            j = i + 1
            while j < len(text) and (text[j].isalnum() or text[j] == "_"):
                j += 1
            conversion = text[i + 1 : j]
            if not conversion:
                if j < len(text) and text[j].isspace():
                    msg = "conversion type must come right after the exclamation mark"
                else:
                    msg = "missing conversion character"
                raise _interpolation_error(kind, msg, position_info)
            if conversion not in conversion_dict:
                raise _interpolation_error(
                    kind,
                    f"invalid conversion character {conversion!r}: "
                    "expected 's', 'r', or 'a'",
                    position_info,
                )
            rest = text[j:].lstrip()
            if rest.startswith(":"):
                spec = rest[1:]
            elif rest:
                raise _interpolation_error(kind, "expecting '}'", position_info)
            break
        elif depth == 0 and c == ":":
            expr_end = i
            delimiter = ":"
            spec = text[i + 1 :]
            break
        i += 1

    expr_text = text[:expr_end]
    debug_text = ""
    stripped = expr_text.rstrip()
    if stripped.endswith("=") and stripped[-2:-1] not in _DEBUG_BLOCKERS:
        debug_text = expr_text
        expr_text = stripped[:-1]
        delimiter = "="
        if conversion is None and spec is None:
            conversion = "r"
    forms = parse(expr_text) if expr_text.strip() else []
    if len(forms) != 1:
        raise _interpolation_error(
            kind,
            f"valid expression required before {delimiter!r}"
            if not forms
            else "expecting a single expression in a replacement field",
            position_info,
        )
    format_spec = (
        None
        if spec is None
        else interpolated_parse("f-string", raw, spec, position_info, kind)
    )
    return (
        FStrExpr(
            forms[0],
            conversion=conversion_dict.get(conversion, -1),
            format_spec=format_spec,
            expr_text=expr_text,
            **position_info,
        ),
        debug_text,
    )


def interpolated_parse(symbol, raw, content, position_info, kind=None):
    """``(f-string lit field lit ... lit)``: literals alternate with FStrExprs.

    ``kind`` names the literal in error messages; a format spec is always an
    f-string body, even inside a t-string.
    """
    kind = kind or symbol
    pieces = scan_interpolated(content, raw, kind, position_info)
    literals = [_literal_value(piece, raw) for piece in pieces[0::2]]
    fields = []
    for [k, field] in enumerate(pieces[1::2]):
        [fstr_expr, debug_text] = field_parse(field, kind, raw, position_info)
        literals[k] += debug_text
        fields.append(fstr_expr)
    processed = [String(repr(literals[0]), **position_info)]
    for [fstr_expr, literal] in zip(fields, literals[1:]):
        processed += [fstr_expr, String(repr(literal), **position_info)]
    return Paren(Symbol(symbol, **position_info), *processed, **position_info)
