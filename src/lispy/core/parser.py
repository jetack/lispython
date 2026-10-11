import contextvars
import re
from collections import deque

from lispy.core.compiler.features import require
from lispy.core.compiler.literal import number_parts, string_literal_value
from lispy.core.errors import LispyCompileError
from lispy.core.nodes import *
from lispy.core.nodes import (
    Brace,
    Bracket,
    Expression,
    FStrExpr,
    GluedHead,
    MetaIndicator,
    Node,
    Paren,
    String,
    Symbol,
    Unquote,
    UnquoteSplice,
    Wrapper,
)
from lispy.core.utils import augassignop_dict

_INT = r"\d+[\dA-Za-z]*(?:_[\dA-Za-z]+)*"
_FLOATS = [
    "\\d+(?:_\\d+)*\\.\\d+(?:_\\d+)*",
    "\\d+(?:_\\d+)*\\.\\d*",
    "\\d*\\.\\d+(?:_\\d+)*",
]
_SCIENTIFIC = [x + "[eE][\\-\\+]?\\d+(?:_\\d+)*" for x in _FLOATS + [_INT]]
_NUMBER = "|".join(_SCIENTIFIC + _FLOATS + [_INT])
_COMPLEX = f"(?:(?:{_NUMBER})[+-])?(?:{_NUMBER})[jJ]"
_TOKEN_PATTERNS = [
    ["\\^?\\*{0,2}[\\(\\{\\[]", "opening"],
    ["[\\)\\}\\]]", "closing"],
    # Only the opening of an f/t-string: scan_interpolated finds its end, so
    # nesting inside replacement fields is unlimited.
    ["(?:[rR]?[fFtT]|[fFtT][rR])(?:\"\"\"|'''|\")", "interpolated"],
    ["\\w*'''(?:[^\\\\']|\\\\.)*'''", "'''"],
    ['\\w*"""(?:[^\\\\"]|\\\\.)*"""', '"""'],
    ['\\w*"(?:[^\\\\"]|\\\\.)*"', '"'],
    ["\\^?\\*{0,2}(?:'|`|~@|~)(?!\\s|$)", "meta-indicator"],
    [";[^\\n]*", "comment"],
    [
        "[+-]*(?:" + "|".join([_COMPLEX, _NUMBER]) + ")(?=\\s|$|\\)|\\}|\\]|,)",
        "number",
    ],
    [",", "comma"],
    # "[" ends a symbol so that f[T] is a symbol glued to a bracket.
    ['[^\\s\\)\\}\\]\\"\\[,]+', "symbol"],
    ["\\n", "new-line"],
    [" +", "spaces"],
]
_TOKEN_RE = re.compile("|".join(f"({p})" for [p, _] in _TOKEN_PATTERNS))
_TOKEN_LABELS = [label for [_, label] in _TOKEN_PATTERNS]


def _advance(text, lineno, col_offset):
    """Position after ``text`` when it starts at (lineno, col_offset)."""
    newlines = text.count("\n")
    if newlines:
        return [lineno + newlines, len(text) - text.rfind("\n") - 1]
    return [lineno, col_offset + len(text)]


def _source_locator(src):
    """locate(i, j) -> position_info of src[i:j]."""

    def position(i):
        return [src.count("\n", 0, i) + 1, i - src.rfind("\n", 0, i) - 1]

    def locate(i, j=None):
        [lineno, col_offset] = position(i)
        [end_lineno, end_col_offset] = position(i if j is None else j)
        return {
            "lineno": lineno,
            "col_offset": col_offset,
            "end_lineno": end_lineno,
            "end_col_offset": end_col_offset,
        }

    return locate


def _interpolated_token(src, start, opening):
    """Extent of the f/t-string opening at ``start``: [token, quote, error].

    A malformed literal is not reported here: it runs to the end of the
    source and carries its error, which string_parse raises after the t-string
    version gate, so an old interpreter hears about the version first.
    """
    quote = opening.lstrip("rRfFtT")
    raw = "r" in opening[: -len(quote)].lower()
    kind = "t-string" if "t" in opening.lower() else "f-string"
    try:
        [_, end] = scan_interpolated(
            src, start + len(opening), None, quote, raw, kind, _source_locator(src)
        )
    except LispyCompileError as err:
        return [src[start:], quote, err]
    return [src[start:end], quote, None]


def tokenize(src):
    """Tokens as [text, type, position_info, error]; error is set only on a
    malformed f/t-string (see _interpolated_token)."""
    tokens = deque([])
    [lineno, col_offset] = [1, 0]
    pos = 0
    previous = None
    while True:
        m = _TOKEN_RE.search(src, pos)
        if m is None:
            break
        [lineno, col_offset] = _advance(src[pos : m.start()], lineno, col_offset)
        tk = m.group()
        tktype = _TOKEN_LABELS[m.lastindex - 1]
        error = None
        if tktype == "interpolated":
            [tk, tktype, error] = _interpolated_token(src, m.start(), tk)
        elif tktype == "comma":
            # A comma is whitespace, except right after "(" or "{": (, a b) is a
            # tuple and {, a b} a set.
            kept = previous is not None and previous[1] == "opening"
            tktype = "symbol" if kept and previous[0][-1] in "({" else "spaces"
        [end_lineno, end_col_offset] = _advance(tk, lineno, col_offset)
        if tktype not in ("new-line", "spaces"):
            previous = [tk, tktype]
        if tktype not in ("new-line", "spaces", "comment"):
            position_info = {
                "lineno": lineno,
                "col_offset": col_offset,
                "end_lineno": end_lineno,
                "end_col_offset": end_col_offset,
            }
            tokens.append([tk, tktype, position_info, error])
        [lineno, col_offset] = [end_lineno, end_col_offset]
        pos = m.start() + len(tk)
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


def _glue_target(container, lineno, col_offset):
    """``[prefixes, head]`` when a ``[`` at (lineno, col) is glued to the last form.

    The head is the bare Symbol (or an unquoted ``~symbol``, so a quasiquote
    can splice the name of a generic def) that ends exactly where the bracket
    starts, found under any prefixes of the last form: ``^list[int]``,
    ``*args[0]``, ``:k[0]``, ``'f[T]``.  The prefixes then wrap the GluedHead.
    """
    items = container if isinstance(container, list) else container.list
    node = items[-1] if items else None
    prefixes = []
    while isinstance(node, (Wrapper, MetaIndicator)) and not isinstance(
        node, FStrExpr
    ):
        if _unquoted_symbol_p(node):
            break
        prefixes.append(node)
        node = node.value
    glued = (
        (type(node) is Symbol or _unquoted_symbol_p(node))
        and (node.end_lineno, node.end_col_offset) == (lineno, col_offset)
    )
    return [prefixes, node] if glued else None


def _unquoted_symbol_p(node):
    return (
        isinstance(node, Unquote)
        and not isinstance(node, UnquoteSplice)
        and type(node.value) is Symbol
    )


_OPENERS = {Paren: "(", Bracket: "[", Brace: "{"}
_CLOSERS = {")": "(", "]": "[", "}": "{"}


def _check_closing(stack, closing, position_info):
    """CPython's errors for a closing bracket with nothing (or the wrong
    kind) open, instead of popping the wrong node or an empty stack."""
    top = stack[-1] if stack else None
    if type(top) not in _OPENERS:
        raise LispyCompileError(
            f"unmatched '{closing}'", Symbol(closing, **position_info)
        )
    if _OPENERS[type(top)] != _CLOSERS[closing]:
        raise LispyCompileError(
            f"closing parenthesis '{closing}' does not match opening "
            f"parenthesis '{_OPENERS[type(top)]}'",
            Symbol(closing, **position_info),
        )


def _check_all_closed(stack):
    unclosed = [node for node in stack if type(node) in _OPENERS]
    if unclosed:
        opener = unclosed[-1]
        position_info = {
            "lineno": opener.lineno,
            "col_offset": opener.col_offset,
            "end_lineno": opener.lineno,
            "end_col_offset": opener.col_offset + 1,
        }
        raise LispyCompileError(
            f"'{_OPENERS[type(opener)]}' was never closed",
            Symbol(_OPENERS[type(opener)], **position_info),
        )


# Set while parse(..., partial=True) runs: version gates are skipped.
_PARTIAL = contextvars.ContextVar("lispy_partial_parse", default=False)


def parse(src, partial=False):
    """The top-level forms of ``src``.

    Strict by default: any malformed input raises LispyCompileError.  With
    ``partial=True`` (for the LSP, whose buffer is mid-edit) it never raises:
    it returns the complete top-level forms before the first error and skips
    version gates, so symbols, hover and definitions survive an unfinished
    form.  The LSP's diagnostics still use the strict parse to report it.
    """
    if not partial:
        return _parse(src, [])
    rst = []
    token = _PARTIAL.set(True)
    try:
        _parse(src, rst)
    except Exception:
        pass
    finally:
        _PARTIAL.reset(token)
    return rst


def _parse(src, rst):
    tokens = tokenize(src)
    stack = []
    while tokens:
        [t, tktype, position_info, error] = tokens.popleft()
        [lineno, col_offset, end_lineno, end_col_offset] = position_info_into_list(
            position_info
        )
        if tktype == "opening":
            wrapper = t[:-1]
            opening = t[-1]
            container = stack[-1] if stack else rst
            glueable = isinstance(container, (list, Expression))
            target = (
                _glue_target(container, lineno, col_offset)
                if opening == "[" and not wrapper and glueable
                else None
            )
            if target is not None:
                [prefixes, head] = target
                (container if isinstance(container, list) else container.list).pop()
                # Reopen the prefixes around a pending head: when the bracket
                # closes, the GluedHead takes it and the prefixes close over it.
                for prefix in prefixes:
                    prefix.value = None
                    stack.append(prefix)
                position = {"lineno": head.lineno, "col_offset": head.col_offset}
                stack.append(GluedHead(head, None, **position))
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
                _check_closing(stack, t, position_info)
                e = stack.pop()
                e.update_dict("end_lineno", end_lineno)
                e.update_dict("end_col_offset", end_col_offset)
            elif error is not None:
                e = string_parse(t, tktype, position_info, error)
            else:
                e = token_parse(t, tktype, position_info)
            while stack and isinstance(stack[-1], (MetaIndicator, Wrapper, GluedHead)):
                popped = stack.pop()
                if isinstance(popped, GluedHead):
                    popped.params = e
                else:
                    popped.value = e
                popped.update_dict("end_lineno", end_lineno)
                popped.update_dict("end_col_offset", end_col_offset)
                e = popped
            (stack[-1] if stack else rst).append(e)
    _check_all_closed(stack)
    return rst


special_literals = ("True", "False", "None", "...")


def token_parse(token, tktype, position_info):
    return (
        Symbol(token, **position_info)
        if token in augassignop_dict
        else Constant(token, **position_info)
        if token in special_literals
        else Constant(token, **position_info)
        if tktype == "number" and (token[0] not in "+-" or _signed_complex_p(token))
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


def _signed_complex_p(token):
    # -1+2j is one Constant, so the sign binds to the real part as in Python;
    # a plain signed number stays (- 1), a unary op.
    parts = number_parts(token)
    return parts is not None and parts[2] is not None


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


def string_parse(token, tktype, position_info, error=None):
    [prefix, *contents, _] = token.split(tktype)
    node = String(token, **position_info)
    kind = interpolated_kind(prefix, node)
    if kind == "t-string":
        # Gate where the prefix is seen, so an old interpreter reports the
        # version before any complaint about the literal's contents.
        if not _PARTIAL.get():
            require("t-string", node)
    if error is not None:
        raise error
    if not kind:
        return node
    locate = _token_locator(token, position_info)
    raw = "r" in prefix.lower()
    start = len(prefix) + len(tktype)
    [pieces, _] = scan_interpolated(token, start, None, tktype, raw, kind, locate)
    return interpolated_node(kind, kind, token, pieces, raw, locate, position_info)


def interpolated_kind(prefix, node):
    """``"f-string"``/``"t-string"`` for an f/t-prefixed literal, else None.

    Python's rules: f/F or t/T, alone or with r/R in either order and case;
    f, t, b and u never combine.  Any other prefix containing f or t is
    refused here, so it can never be evaluated as a plain string literal.
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


def _token_locator(token, position_info):
    """locate(i, j) -> position_info of token[i:j], from the token's position."""

    def position(i):
        newline = token.rfind("\n", 0, i)
        if newline < 0:
            return [position_info["lineno"], position_info["col_offset"] + i]
        return [position_info["lineno"] + token.count("\n", 0, i), i - newline - 1]

    def locate(i, j=None):
        [lineno, col_offset] = position(i)
        [end_lineno, end_col_offset] = position(i if j is None else j)
        return {
            "lineno": lineno,
            "col_offset": col_offset,
            "end_lineno": end_lineno,
            "end_col_offset": end_col_offset,
        }

    return locate


def _interpolation_error(kind, msg, locate, i, j=None):
    return LispyCompileError(f"{kind}: {msg}", String('""', **locate(i, j)))


def scan_interpolated(text, start, end, quote, raw, kind, locate):
    """The one scanner of f/t-string bodies: the tokenizer uses it for the
    literal's extent, the parser for its pieces.

    Scans ``text`` from ``start`` to the closing ``quote`` (or to ``end`` for a
    format spec, which has no quote) and returns ``[pieces, after]``: pieces
    alternate ``[literal, start, end]`` and ``[field, start, end]``, beginning
    and ending with a literal.  Literals are source text (escapes unprocessed)
    with ``{{``/``}}`` collapsed; a field spans its braces and ``field`` is the
    text between them.  ``after`` is the index past the closing quote.
    """
    end = len(text) if end is None else end
    pieces = []
    buf = []
    literal_start = i = start
    while i < end:
        if quote is not None and text.startswith(quote, i):
            pieces.append(["".join(buf), literal_start, i])
            return [pieces, i + len(quote)]
        c = text[i]
        if c == "\\" and i + 1 < end:
            nxt = text[i + 1]
            if not raw and text.startswith("N{", i + 1):
                close = text.find("}", i + 3, end)
                close = end - 1 if close < 0 else close
                buf.append(text[i : close + 1])
                i = close + 1
            elif nxt in "{}":
                # A backslash never escapes a brace: it is a literal backslash.
                buf.append("\\" if raw else "\\\\")
                i += 1
            else:
                buf.append(text[i : i + 2])
                i += 2
        elif c in "{}" and text.startswith(c, i + 1):
            buf.append(c)
            i += 2
        elif c == "{":
            close = _field_end(text, i + 1, end, kind, locate)
            pieces.append(["".join(buf), literal_start, i])
            pieces.append([text[i + 1 : close], i, close + 1])
            buf = []
            literal_start = i = close + 1
        elif c == "}":
            raise _interpolation_error(
                kind, "single '}' is not allowed", locate, i, i + 1
            )
        else:
            buf.append(c)
            i += 1
    if quote is not None:
        raise LispyCompileError(
            f"unterminated {kind} literal", String('""', **locate(start, end))
        )
    pieces.append(["".join(buf), literal_start, end])
    return [pieces, end]


def _field_chars(text, i, end):
    """(index, char, depth) for a replacement field's characters outside
    nested strings; depth counts the brackets open around the char."""
    depth = 0
    quote = None
    while i < end:
        if quote:
            if text.startswith(quote, i):
                i += len(quote)
                quote = None
            else:
                i += 2 if text[i] == "\\" else 1
            continue
        # \" is the escape for a quote inside braces; a bare " also opens one.
        opener = '\\"' if text.startswith('\\"', i) else '"' if text[i] == '"' else None
        if opener:
            quote = opener
            i += len(opener)
            continue
        c = text[i]
        if c in ")]}":
            depth -= 1
        yield [i, c, depth]
        if c in "([{":
            depth += 1
        i += 1


def _field_end(text, start, end, kind, locate):
    """Index of the ``}`` closing the replacement field that starts at ``start``."""
    for [i, c, depth] in _field_chars(text, start, end):
        if c == "}" and depth < 0:
            return i
    raise _interpolation_error(kind, "expecting '}'", locate, start - 1, start)


# A trailing "=" after one of these is part of an operator symbol, not a
# self-documenting "=" (``{x=}``).
_DEBUG_BLOCKERS = "=!<>:+-*/%&|^@~"


def _relocate(node, locate, origin):
    """Move a node parsed from a field's text to where that text sits."""
    if isinstance(node, list):
        for item in node:
            _relocate(item, locate, origin)
        return
    if not isinstance(node, Node):
        return
    if node.lineno:
        [line, col] = [node.lineno, node.col_offset]
        [end_line, end_col] = [node.end_lineno, node.end_col_offset]
        start = locate(origin)
        # Set this node only: a wrapper's update_dict would also overwrite the
        # position of the node it wraps, which is relocated on its own below.
        node.__dict__.update(
            lineno=start["lineno"] + line - 1,
            col_offset=col + (start["col_offset"] if line == 1 else 0),
            end_lineno=start["lineno"] + end_line - 1,
            end_col_offset=end_col + (start["col_offset"] if end_line == 1 else 0),
        )
    for value in list(vars(node).values()):
        if isinstance(value, (Node, list)):
            _relocate(value, locate, origin)


def _relocate_error(err, start):
    """Move an error raised parsing a field's text to where that text sits."""
    for [line_attr, col_attr] in [["lineno", "offset"], ["end_lineno", "end_offset"]]:
        line = getattr(err, line_attr)
        if line:
            if line == 1 and getattr(err, col_attr):
                setattr(err, col_attr, getattr(err, col_attr) + start["col_offset"])
            setattr(err, line_attr, line + start["lineno"] - 1)


def field_parse(text, field, kind, raw, locate):
    """Parse one replacement field into ``[FStrExpr, debug_text]``.

    ``field`` is ``[_, open_brace, after_close_brace]`` from scan_interpolated.
    Follows Python: ``expr[=][!conv][:spec]``; whitespace around ``expr`` is
    part of its text; ``=`` echoes the text before it (and implies ``!r``
    unless a conversion or spec is given); the spec is itself an f-string body.
    """
    [_, open_brace, after] = field
    [start, close] = [open_brace + 1, after - 1]
    expr_end = close
    delimiter = "}"
    conversion = None
    spec_start = None
    for [i, c, depth] in _field_chars(text, start, close):
        if depth != 0:
            continue
        if c == "!" and not text.startswith("!=", i):
            [expr_end, delimiter] = [i, "!"]
            j = i + 1
            while j < close and (text[j].isalnum() or text[j] == "_"):
                j += 1
            conversion = text[i + 1 : j]
            if not conversion:
                msg = (
                    "conversion type must come right after the exclamation mark"
                    if j < close and text[j].isspace()
                    else "missing conversion character"
                )
                raise _interpolation_error(kind, msg, locate, i, i + 1)
            if conversion not in conversion_dict:
                raise _interpolation_error(
                    kind,
                    f"invalid conversion character {conversion!r}: "
                    "expected 's', 'r', or 'a'",
                    locate,
                    i,
                    j,
                )
            rest = j + len(text[j:close]) - len(text[j:close].lstrip())
            if rest < close and text[rest] == ":":
                spec_start = rest + 1
            elif rest < close:
                raise _interpolation_error(
                    kind, "expecting '}'", locate, rest, rest + 1
                )
            break
        if c == ":":
            [expr_end, delimiter, spec_start] = [i, ":", i + 1]
            break

    expr_text = text[start:expr_end].replace('\\"', '"')
    debug_text = ""
    stripped = expr_text.rstrip()
    if stripped.endswith("=") and stripped[-2:-1] not in _DEBUG_BLOCKERS:
        debug_text = expr_text
        expr_text = stripped[:-1]
        delimiter = "="
        if conversion is None and spec_start is None:
            conversion = "r"
    try:
        forms = parse(expr_text) if expr_text.strip() else []
    except LispyCompileError as err:
        _relocate_error(err, locate(start))
        raise
    if len(forms) != 1:
        raise _interpolation_error(
            kind,
            f"valid expression required before {delimiter!r}"
            if not forms
            else "expecting a single expression in a replacement field",
            locate,
            open_brace,
            after,
        )
    _relocate(forms[0], locate, start)
    format_spec = None
    if spec_start is not None:
        [pieces, _] = scan_interpolated(
            text, spec_start, close, None, raw, kind, locate
        )
        # As in CPython, the spec's JoinedStr starts at the ":".
        spec_position = locate(spec_start - 1, close)
        format_spec = interpolated_node(
            "f-string", kind, text, pieces, raw, locate, spec_position
        )
    return [
        FStrExpr(
            forms[0],
            conversion=conversion_dict.get(conversion, -1),
            format_spec=format_spec,
            expr_text=expr_text,
            **locate(open_brace, after),
        ),
        debug_text,
    ]


def interpolated_node(symbol, kind, text, pieces, raw, locate, position_info):
    """``(f-string lit field lit ... lit)`` (or ``t-string``) from scanned pieces.

    ``kind`` names the literal in error messages; a format spec is always an
    f-string body, even inside a t-string.
    """
    literals = [
        [
            string_literal_value(
                "r" if raw else "", p[0], String('""', **locate(p[1], p[2]))
            ),
            p[1],
            p[2],
        ]
        for p in pieces[0::2]
    ]
    fields = []
    for [k, field] in enumerate(pieces[1::2]):
        [fstr_expr, debug_text] = field_parse(text, field, kind, raw, locate)
        if debug_text:
            # The echoed {expr=} text joins the literal before it; CPython
            # places that constant over the echoed text.
            echo_start = field[1] + 1
            if not literals[k][0]:
                literals[k][1] = echo_start
            literals[k][2] = echo_start + len(debug_text)
            literals[k][0] += debug_text
        fields.append(fstr_expr)
    processed = []
    for [k, [value, start, end]] in enumerate(literals):
        processed.append(String(repr(value), **locate(start, end)))
        if k < len(fields):
            processed.append(fields[k])
    return Paren(Symbol(symbol, **position_info), *processed, **position_info)
