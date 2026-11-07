import ast
import contextvars
import io
import keyword
import re
import tokenize

from lispy.core.compiler.features import require
from lispy.core.errors import LispyCompileError, compile_warning, filtered_warnings
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
        # catch_warnings swaps the process-wide filters (under
        # filtered_warnings' lock, which a concurrent compile may hold) and
        # resets the warning registry a REPL, nREPL or LSP session keeps,
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
        with filtered_warnings("always", record=True) as caught:
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


# A text shaped as one number or a complex's two parts, ASCII, each part with
# at most one ".": the only digit-led names literal_eval is asked about.
# ⚠️ No part may repeat an operator or a ".", because Python parses each
# repeat as one more nesting level (1-...-x, 1+...+1, 1e5.real.real...),
# and literal_eval then overflows (RecursionError, MemoryError).
_NUMBER_PART = "[0-9][0-9A-Za-z_]*(?:\\.[0-9A-Za-z_]*)?"
_NUMBER_SHAPE_RE = re.compile(f"{_NUMBER_PART}(?:[+-]{_NUMBER_PART})?")


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

# Operator heads. An operator is never a name, even where its mangled text is
# one (``-`` mangles to ``_``), because operators have no function value: only
# a hyphenated name (``is-not``, ``not-in``, the operator module's ``is_not``)
# stays a name (operator_spelling_p).
OPERATOR_TEXTS = frozenset([*unaryop_dict, *binop_dict, *compare_dict, *boolop_dict])


def identifier_p(name):
    """Whether the mangled text ``name`` is a Python name: an identifier that
    is no hard keyword. Soft keywords (match, case, type, _) are names."""
    return name.isidentifier() and not keyword.iskeyword(name)


def operator_spelling_p(text):
    """Whether the symbol part spelled ``text`` is an operator, which is never
    a name: an OPERATOR_TEXTS spelling that is no hyphenated name
    (hyphenated_name_p), so ``-`` is the operator and ``is-not`` the name
    ``is_not``."""
    return text in OPERATOR_TEXTS and not hyphenated_name_p(text)


def python_name_p(text):
    """Whether the undotted symbol spelled ``text`` is a name python_name
    accepts: its mangled text passes identifier_p and it is no operator
    (operator_spelling_p). Not the LSP's name-spelling-p (python_view.lpy),
    which also asks for a letter or digit, so ``_`` and ``->`` are names here
    and none there."""
    return identifier_p(Symbol(text).name) and not operator_spelling_p(text)


def python_name(spelling, node):
    """The Python name of the symbol part spelled ``spelling`` (mangled as
    Symbol.name), else a LispyCompileError at ``node``: the one home every
    emitted identifier passes through. A dotted symbol's caller passes each
    part in source order, so the first part at fault is the one reported, by
    the first rule it breaks: a hard keyword, an operator
    (operator_spelling_p: ``a.-``, the method head ``.-``), no identifier.
    """
    name = Symbol(spelling).name
    text = atom_text(node) if isinstance(node, Node) else spelling
    if keyword.iskeyword(name):
        raise LispyCompileError(
            f"{name} is a Python keyword and cannot be used as a name", node
        )
    if operator_spelling_p(spelling):
        raise LispyCompileError(
            f"{spelling} is an operator and cannot be used as a name", node
        )
    if not name.isidentifier():
        msg = _malformed_number_message(text) or f"{text} is not a valid Python name"
        raise LispyCompileError(msg, node)
    return name


def symbol_name(sexp):
    """The Python name an undotted symbol stands for (python_name); anything
    else is a LispyCompileError at itself."""
    if isinstance(sexp, GluedHead):
        raise sexp.misuse_error()
    if type(sexp) is not Symbol:
        raise LispyCompileError(f"expected a name, got {sexp}", sexp)
    return python_name(atom_text(sexp), sexp)


# Bindings that shadow a macro. Macro lookup is by a call's head name with no
# lexical scope (core/macro.py), so a name bound while a macro of that Python
# name is active stays a macro call: (def inc [x] ...) then (inc 1) silently
# expands the built-in inc. Every site that binds a name refuses that here:
# name_compile for a stored undotted name, binding_name for every other name
# slot (parameters, def/class names, imports, except, patterns, type
# parameters). core/macro.py sets the value around compiling a unit's
# forms: a callable from a binding's spelling to None, or to the kind of the
# active macro that is one name with it (name_key) and that macro's key:
# ("built-in", key) for a prelude macro, ("macro", key) for any other. None outside a
# unit, and while require compiles its generated import, means no check;
# defmacro's generated def is checked, since its parameters and locals are
# bindings.
ACTIVE_MACROS = contextvars.ContextVar("lispy_active_macros", default=None)


def check_binding(node, spelling=None):
    """Refuse, at ``node``, binding the name spelled ``spelling`` (default:
    the symbol's text) where an active macro is one name with it
    (name_key). The message names the macro as its key spells it, which
    may differ from the binding (``my_x`` against the macro ``my-x``)."""
    active = ACTIVE_MACROS.get()
    if active is None:
        return
    spelling = atom_text(node) if spelling is None else spelling
    found = active(spelling)
    if found is None:
        return
    [kind, key] = found
    if kind == "built-in":
        raise LispyCompileError(
            f"cannot bind '{spelling}': it is the built-in macro '{key}' here, so "
            f"({key} ...) would expand the macro instead of using this name; "
            "rename it, or make (require lispy.macros [...]) the module's first "
            "form listing only the macros it uses",
            node,
        )
    raise LispyCompileError(
        f"cannot bind '{spelling}': it is the macro '{key}' here, so ({key} ...) "
        "would expand the macro instead of using this name; rename it",
        node,
    )


def binding_name(sexp):
    """symbol_name for a symbol the construct binds, refused where it shadows
    an active macro (check_binding)."""
    name = symbol_name(sexp)
    check_binding(sexp)
    return name


def method_name(method):
    """The attribute of a ``(.method obj ...)`` head: the text after its dot,
    mangled as a symbol and passed through python_name at the head."""
    return python_name(atom_text(method)[1:], method)


def module_path(sexp, relative=False):
    """``[path, level]`` of a dotted module symbol, every part through
    python_name. With ``relative``, leading dots are the level and the path may be empty
    (None), as in ``from . import x``."""
    name = sexp.name
    level = len(name) - len(name.lstrip(".")) if relative else 0
    path = name[level:]
    if level and not path:
        return [None, level]
    for part in atom_text(sexp)[level:].split("."):
        python_name(part, sexp)
    return [path, level]


def hyphenated_name_p(text):
    """Whether ``text`` is a name written with hyphens as word separators
    (``my-x``, ``inc``): its Python name is ``text`` with each ``-`` as ``_``
    and no hyphen leads, trails or doubles. Anything else (``->``, ``--``,
    ``ok?``) is mangled into its Python name rather than spelled as it."""
    name = Symbol(text).name
    return (
        identifier_p(name)
        and name == text.replace("-", "_")
        and "" not in text.split("-")
    )


def name_key(text):
    """The name ``text`` stands for when two spellings are compared: each
    ``.``-separated part its Python name when it is a hyphenated name
    (hyphenated_name_p: ``my-x`` and ``my_x`` are one name), else the part as
    written (``->``, ``ok?``, ``x-`` are only themselves). The one home of
    name identity: the compiler's macro clash check (core/macro.py) and the
    LSP (python_view's name-key) both compare through it."""
    return ".".join(
        Symbol(part).name if hyphenated_name_p(part) else part
        for part in text.split(".")
    )


# An underscore with a letter or digit on each side: the only ones a hyphen
# spells, since a leading, trailing or doubled hyphen is no hyphenated name.
_WORD_UNDERSCORE = re.compile(r"(?<=[^\W_])_(?=[^\W_])")


def hyphen_spelling(text):
    """How lispython spells the name ``text``: each ``.``-separated part that
    is a Python name (identifier_p) with every ``_`` between two letters or
    digits written ``-`` (``move_to_end`` is ``move-to-end``,
    ``_private_helper`` is ``_private-helper``); any other part, and every
    leading, trailing or doubled ``_`` (``__init__``, ``class_``, ``a__b``),
    as written.  The one home of the default spelling completion offers (the
    LSP and the nREPL).  For every identifier part ``n`` the spelling is
    ``hyphenated_name_p``, ``name_key(hyphen_spelling(n)) == n`` and
    ``Symbol(hyphen_spelling(n)).name == n``: it compiles back to ``n``."""
    return ".".join(
        _WORD_UNDERSCORE.sub("-", part) if identifier_p(part) else part
        for part in text.split(".")
    )


def plain_name_p(sexp):
    """Whether ``sexp`` is shaped as a name: a symbol python_name_p accepts
    (so no dot and no operator: ``-`` fails the shape as ``+`` does). A shape
    test for sites with their own form error; the name itself is still read
    through symbol_name."""
    return type(sexp) is Symbol and python_name_p(atom_text(sexp))


def name_or_attr_p(sexp):
    """Python's name_or_attr shape: every dot-part's mangled text is an
    identifier and no part is an operator (operator_spelling_p), so ``.x``,
    ``a..b``, ``x.`` and ``a.-`` are no value pattern, mapping key or class
    name. A hard keyword part (``a.if``) passes the shape and is refused by
    name_compile, with the keyword message, at the symbol."""
    return type(sexp) is Symbol and all(
        Symbol(part).name.isidentifier() and not operator_spelling_p(part)
        for part in atom_text(sexp).split(".")
    )


def malformed_number_p(sexp):
    """A symbol python_name words as a malformed number (``1.5if``)."""
    return (
        type(sexp) is Symbol and _malformed_number_message(atom_text(sexp)) is not None
    )


def name_compile(symbol, ctx):
    """A Name, or an Attribute chain for a dotted symbol, every part through
    python_name. ``ctx`` is the undotted name's, else the last attribute's;
    other parts load."""
    [name, *attrs] = [
        python_name(part, symbol) for part in atom_text(symbol).split(".")
    ]
    if ctx is ast.Store and not attrs:
        # A stored undotted name is a binding; a dotted one an attribute.
        check_binding(symbol)
    position_info = {**symbol.position_info}
    # Positions are .lpy columns: measure the spelling's components (``a.x->y``),
    # never the mangled Python names, which differ in length. Measure only when
    # the spelling fits the symbol's own one-line span, so a measured end never
    # leaves it: a symbol that borrowed another node's position (a macro
    # template placed at its call, quote expansion's classname symbol) is
    # measured when its spelling fits that span, and otherwise keeps the whole
    # span on every part.
    [spelled, *spelled_attrs] = symbol.value.split(".")
    measured = (
        position_info["lineno"] == position_info["end_lineno"]
        and position_info["col_offset"] + len(symbol.value)
        <= position_info["end_col_offset"]
    )
    if measured:
        position_info["end_col_offset"] = position_info["col_offset"] + len(spelled)
    rst = ast.Name(id=name, ctx=ast.Load(), **position_info)
    for attr, spelled_attr in zip(attrs, spelled_attrs, strict=True):
        if measured:
            position_info["end_col_offset"] += 1 + len(spelled_attr)
        rst = ast.Attribute(value=rst, attr=attr, ctx=ast.Load(), **position_info)
    rst.ctx = ctx()
    return rst
