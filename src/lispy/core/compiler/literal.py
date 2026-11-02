import ast
import io
import keyword
import re
import tokenize
import warnings

from lispy.core.compiler.features import require
from lispy.core.errors import LispyCompileError, compile_warning
from lispy.core.nodes import GluedHead, Node, Symbol, atom_text
from lispy.core.utils import binop_dict, boolop_dict, compare_dict, unaryop_dict

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


def literal_value(text, node, malformed=False):
    """``ast.literal_eval(text)``, the only evaluation of literal text.

    A malformed literal (``09``, a truncated ``\\x`` escape, non-ASCII bytes) is
    a LispyCompileError at ``node``, so every caller reports it at the token.
    A warning it raises (an invalid escape) is re-issued by compile_warning
    at ``node``, naming the .lpy file and line where literal_eval would name
    ``<unknown>`` line 1, the text it was handed; one raised before an error
    (``1if``: "invalid decimal literal") becomes the error's message
    (_literal_eval). ``malformed`` says ``text`` is a number token
    number_parts rejected: its error keeps only wording of the token
    (_number_wording), else is "invalid number literal".
    """
    try:
        # Only an escape or a malformed number makes literal_eval warn.
        # catch_warnings swaps the process-wide filters (not thread-safe, and
        # it resets the warning registry a REPL, nREPL or LSP session keeps),
        # so it is entered only for text that can warn.
        [value, caught] = _literal_eval(text, malformed or "\\" in text)
    except (SyntaxError, ValueError) as err:
        msg = err.msg if isinstance(err, SyntaxError) else str(err)
        if malformed:
            # 406E-9.8J is "invalid syntax", 1e+7145.J "malformed node":
            # how CPython parses the text, no word for the token.
            msg = _number_wording(msg) or f"invalid number literal {text}"
        raise LispyCompileError(msg, node) from err
    for warning in caught:
        compile_warning(str(warning.message), warning.category, node.position_info)
    return value


def _literal_eval(text, record):
    """``[ast.literal_eval(text), warnings]``, the warnings recorded only with
    ``record``.

    A SyntaxError raised after a SyntaxWarning takes the first such warning's
    message: CPython's word for the malformed token (``1if``, ``0x1for``:
    "invalid decimal literal", "invalid hexadecimal literal"), what
    compile() raises for it under ``-W error``, where the error itself says
    "invalid syntax". The one home of that wording, for number tokens
    (literal_value) and digit-led names (python_name).
    """
    caught = []
    try:
        if not record:
            return [ast.literal_eval(text), caught]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            return [ast.literal_eval(text), caught]
    except SyntaxError as err:
        warned = [w for w in caught if issubclass(w.category, SyntaxWarning)]
        if warned:
            err.msg = str(warned[0].message)
        raise


def _number_wording(msg):
    """``msg`` where it words the number token itself ("invalid decimal
    literal", "invalid digit '9' in octal literal", "leading zeros in decimal
    integer literals ...", "invalid character ... (U+0661)" for a non-ASCII
    digit), else None.

    ⚠️ Any other message ("invalid syntax" for 1.0.0, "'{' was never
    closed") says how CPython parses the text as Python source, which a
    lispython token never is worded by.
    """
    return msg if "literal" in msg or msg.startswith("invalid character") else None


# A text shaped as one number or a complex's two parts, ASCII: the only
# digit-led names literal_eval is asked about. Nothing in it nests, so
# literal_eval's cost is linear in it, where 1-...-x or 1+1+...+1 overflow
# its parser (RecursionError, MemoryError).
_NUMBER_SHAPE_RE = re.compile("[0-9][0-9A-Za-z_.]*(?:[+-][0-9][0-9A-Za-z_.]*)?")


def _malformed_number_message(text):
    """CPython's word for a number-shaped ``text`` that is no name
    (``1.else``, ``1__0``: "invalid decimal literal"), or None where the text
    has no number shape (``1-x``, ``1=x``) or literal_eval does not word the
    token (``1.0.0``)."""
    if not _NUMBER_SHAPE_RE.fullmatch(text):
        return None
    try:
        _literal_eval(text, True)
    except SyntaxError as err:
        return _number_wording(err.msg)
    except (ValueError, TypeError):
        return None
    return None


def exact_p(node):
    """Whether ``node`` is a Constant carrying its value as ``exact``."""
    return "exact" in getattr(node, "__dict__", ())


def constant_compile(constant):
    """A Constant node in an expression, with the value Python computes.

    ``-1+2j`` is ``(-1) + 2j``: the sign binds to the real part, as in Python.
    A Constant made from macro data with no literal spelling (nan, a complex
    with a signed zero, inf or nan part) carries the value as ``exact``; its
    text is display only (``_exact_constant`` in nodes.py).
    """
    if exact_p(constant):
        return ast.Constant(value=constant.exact, **constant.position_info)
    text = constant.value
    if text in SPECIAL_CONSTANTS:
        return ast.Constant(
            value=literal_value(text, constant), **constant.position_info
        )
    parts = number_parts(text)
    if parts is None:
        # Not a signed number token (number_parts): a malformed literal.
        # CPython's own message for a malformed number ("leading zeros ...").
        value = literal_value(text, constant, malformed=True)
        if type(value) not in (int, float, complex):
            raise LispyCompileError(f"invalid number literal {text}", constant)
        return ast.Constant(value=value, **constant.position_info)
    [signs, real, op, imag] = parts
    value = _apply_signs(literal_value(real, constant), signs)
    if op:
        imag_value = literal_value(imag, constant)
        value = value + imag_value if op == "+" else value - imag_value
    return ast.Constant(value=value, **constant.position_info)


def number_pattern_expr(text, sexp):
    """A number in a match pattern, shaped as ast.parse shapes it.

    ``-1`` is UnaryOp(USub, 1) on every version and ``+1`` UnaryOp(UAdd, 1)
    from 3.15; ``-1+2j`` is BinOp(UnaryOp(USub, 1), Add, 2j).  Python's
    complex-literal rules apply: a real part, then an imaginary one.  A
    Constant carrying an ``exact`` value (see ``constant_compile``) is that
    value, which CPython accepts as a MatchValue; signed, it is no literal
    (its text is a call, for display only).

    None when ``text`` is no number a pattern may sign (``--1``, a signed
    exact value): the caller, stmt.literal_pattern_expr, owns that refusal.
    """
    if exact_p(sexp):
        return ast.Constant(value=sexp.exact, **sexp.position_info)
    if any(exact_p(operand) for operand in getattr(sexp, "list", ())):
        return None
    parts = number_parts(text)
    if parts is None:
        # CPython's own message for a malformed number ("leading zeros ...").
        literal_value(text.lstrip("+-"), sexp, malformed=True)
    if parts is None or len(parts[0]) > 1:
        return None
    [signs, real, op, imag] = parts
    position_info = sexp.position_info
    if signs == "+":
        require("match-unary-plus", sexp)
    real_value = literal_value(real, sexp)
    if op and isinstance(real_value, complex):
        raise LispyCompileError("real number required in complex literal", sexp)
    expr = ast.Constant(value=real_value, **position_info)
    if signs:
        unary = ast.USub if signs == "-" else ast.UAdd
        expr = ast.UnaryOp(unary(), expr, **position_info)
    if op:
        imag_value = literal_value(imag, sexp)
        if not isinstance(imag_value, complex):
            raise LispyCompileError(
                "imaginary number required in complex literal", sexp
            )
        binop = ast.Add if op == "+" else ast.Sub
        imag_expr = ast.Constant(value=imag_value, **position_info)
        expr = ast.BinOp(expr, binop(), imag_expr, **position_info)
    return expr


# Prefixes of a plain (non-interpolated) string literal, lower-cased.
PLAIN_STRING_PREFIXES = ("", "r", "u", "b", "br", "rb")
_QUOTES = ('"""', "'''", '"', "'")


def check_plain_string_prefix(prefix, node):
    """Refuse, at ``node``, a prefix no plain string literal may carry."""
    if prefix.lower() not in PLAIN_STRING_PREFIXES:
        raise LispyCompileError(f"invalid string prefix {prefix!r}", node)


def string_literal_value(prefix, body, node):
    """The value of a string literal's body, as CPython computes it from source.

    The one home for string evaluation: plain strings, the literal stretches of
    f/t-strings and macro arguments all come here.  Like CPython reading a
    source file, every line ending (\\r\\n or a lone \\r) becomes \\n, raw
    strings included; a raw body is its own value; otherwise escapes are
    decoded by literal_value.  Errors are LispyCompileErrors at ``node``.
    """
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    lowered = prefix.lower()
    if "r" in lowered:
        if "b" not in lowered:
            return body
        if not body.isascii():
            raise LispyCompileError(
                "bytes can only contain ASCII literal characters", node
            )
        return body.encode("ascii")
    src = []
    i = 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            src.append(body[i : i + 2])
            i += 2
        else:
            src.append({'"': '\\"', "\n": "\\n"}.get(body[i], body[i]))
            i += 1
    source = ("b" if "b" in lowered else "") + '"' + "".join(src) + '"'
    return literal_value(source, node)


def string_node_value(string):
    """Value of a String node (its token text), or a LispyCompileError at it."""
    token = string.value
    prefix = re.match("[A-Za-z]*", token).group()
    rest = token[len(prefix) :]
    quote = next((q for q in _QUOTES if rest.startswith(q) and rest.endswith(q)), None)
    if quote is None or len(rest) < 2 * len(quote):
        raise LispyCompileError(f"invalid string literal {token}", string)
    check_plain_string_prefix(prefix, string)
    return string_literal_value(prefix, rest[len(quote) : -len(quote)], string)


def string_compile(string):
    return ast.Constant(value=string_node_value(string), **string.position_info)


# The name model. Every identifier the compiler emits (a Name, an attribute,
# a parameter, a keyword argument, an import or pattern name) comes from
# python_name. The home is here, not Symbol.name (nodes.py): that property is
# the data-level mangling macros read at expansion time and the compiler
# reads for non-name purposes (operator heads, gensym prefixes, macro
# registry keys, import paths); whether a text is a valid name depends on
# the slot (dotted or not, after a method's dot), which a
# node does not know; and a refusal is a compiler concern, a
# LispyCompileError positioned at the symbol. A pin test keeps expr.py,
# stmt.py, compiler/utils.py and features.py from reading a symbol's mangled
# name around these functions.

# Operator heads, named as such when one is refused as a name. Only the
# wording: an operator whose mangled text is a name (``-`` is ``_``,
# ``is-not`` is ``is_not``) stays a legal name, as it always was.
OPERATOR_TEXTS = frozenset([*unaryop_dict, *binop_dict, *compare_dict, *boolop_dict])


def identifier_p(name):
    """Whether the mangled text ``name`` is a Python name: an identifier that
    is no hard keyword. Soft keywords (match, case, type, _) are names."""
    return name.isidentifier() and not keyword.iskeyword(name)


def python_name(name, node):
    """``name`` (mangled text) as a Python name, else a LispyCompileError at
    ``node``: the one home every emitted identifier passes through.
    """
    text = atom_text(node) if isinstance(node, Node) else name
    if keyword.iskeyword(name):
        raise LispyCompileError(
            f"{name} is a Python keyword and cannot be used as a name", node
        )
    if not name.isidentifier():
        msg = (
            f"{text} is an operator and cannot be used as a name"
            if text in OPERATOR_TEXTS
            else _malformed_number_message(text) or f"{text} is not a valid Python name"
        )
        raise LispyCompileError(msg, node)
    return name


def symbol_name(sexp):
    """The Python name an undotted symbol stands for (python_name); anything
    else is a LispyCompileError at itself."""
    if isinstance(sexp, GluedHead):
        raise sexp.misuse_error()
    if type(sexp) is not Symbol:
        raise LispyCompileError(f"expected a name, got {sexp}", sexp)
    return python_name(sexp.name, sexp)


def method_name(method):
    """The attribute of a ``(.method obj ...)`` head: the text after its dot,
    mangled as a symbol and passed through python_name at the head."""
    return python_name(Symbol(atom_text(method)[1:]).name, method)


def module_path(sexp, relative=False):
    """``[path, level]`` of a dotted module symbol, every part through
    python_name. With ``relative``, leading dots are the level and the path may be empty
    (None), as in ``from . import x``."""
    name = sexp.name
    level = len(name) - len(name.lstrip(".")) if relative else 0
    path = name[level:]
    if level and not path:
        return [None, level]
    [first, *rest] = path.split(".")
    python_name(first, sexp)
    for part in rest:
        python_name(part, sexp)
    return [path, level]


def plain_name_p(sexp):
    """Whether ``sexp`` is shaped as a name: a symbol whose mangled text
    passes identifier_p (so no dot). A shape test for sites with their own
    form error; the name itself is still read through symbol_name, which also
    refuses an operator."""
    return type(sexp) is Symbol and identifier_p(sexp.name)


def name_or_attr_p(sexp):
    """Python's name_or_attr shape: every dot-part of the mangled name is an
    identifier, so ``.x``, ``a..b`` and ``x.`` are no value pattern, mapping
    key or class name. A hard keyword part (``a.if``) passes the shape and is
    refused by name_compile, with the keyword message, at the symbol."""
    return type(sexp) is Symbol and all(
        part.isidentifier() for part in sexp.name.split(".")
    )


def name_compile(symbol, ctx):
    """A Name, or an Attribute chain for a dotted symbol, every part through
    python_name. ``ctx`` is the undotted name's, else the last attribute's;
    other parts load."""
    [name, *attrs] = symbol.name.split(".")
    python_name(name, symbol)
    for attr in attrs:
        python_name(attr, symbol)
    position_info = {**symbol.position_info}
    position_info["end_col_offset"] = position_info["col_offset"] + len(name)
    rst = ast.Name(id=name, ctx=ast.Load(), **position_info)
    for attr in attrs:
        position_info["end_col_offset"] += 1 + len(attr)
        rst = ast.Attribute(value=rst, attr=attr, ctx=ast.Load(), **position_info)
    rst.ctx = ctx()
    return rst
