"""A Pygments lexer for lispython, registered as the ``lispython`` (``lpy``)
language through the ``pygments.lexers`` entry point (pyproject.toml).

Neither Pygments' Python lexer nor a Lisp one reads lispython: Python's colours
a ``;`` comment as code and gives a ``:kw`` keyword no colour, Clojure's and
Hy's miss Python's keywords and builtins. The docs' code fences and any tool
that highlights through Pygments find this lexer by its alias.

Nothing here restates the language. Token boundaries are the reader's own
patterns (``_token_patterns`` in ``lispy.core.parser``, numbers and string
prefixes included); the form names come from ``lispy.form_docs`` (special
forms, the ``type`` alias, built-in macros), and the Python names from
``keyword`` and ``builtins``. A lettered special form that is no Python keyword
(``ife``, ``f-string``, ``match``) colours as a form only at a form head, where
the compiler dispatches it; elsewhere it is the name it binds.

Only Pygments (through the entry point) and the tests import this module, so
Pygments is not a runtime dependency. Loading it imports ``lispy``, which
installs the ``.lpy`` import hook process-wide, and Pygments loads every plugin
lexer on a filename lookup or ``guess_lexer``.
"""

import builtins
import keyword
import re

from pygments.lexer import RegexLexer, bygroups
from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Text,
    Whitespace,
)

import lispy  # noqa: F401  (the .lpy import hook, for lispy.form_docs)
from lispy import form_docs
from lispy.core.compiler.literal import PLAIN_STRING_PREFIXES
from lispy.core.parser import (
    _F_STRING_PREFIXES,
    _FIELD_BRACES,
    _STRING_TYPES,
    _T_STRING_PREFIXES,
    _token_patterns,
    special_literals,
)

_MACROS = frozenset(form_docs.BUILTIN_MACROS)
# The soft keywords lispython reads (match, case, type) are its forms;
# keyword.softkwlist would add one per Python release that lispython lacks.
_KEYWORDS = frozenset(form_docs.ALL_FORMS) - _MACROS | frozenset(keyword.kwlist)
# The lettered forms a Python name can spell: a form only right after "(".
_HEAD_ONLY = frozenset(
    name
    for name in _KEYWORDS - frozenset(keyword.kwlist)
    if any(c.isalpha() for c in name)
)
_CONSTANTS = frozenset(special_literals)
_BUILTINS = {
    name: Name.Exception
    if isinstance(value, type) and issubclass(value, BaseException)
    else Name.Builtin
    for name, value in vars(builtins).items()
    if not name.startswith("_")
}


def _symbol_tokens(text, start, after_paren):
    """(index, token type, text) of a symbol token, classified as the reader
    reads it (``token_parse``): a ^, *, ** or sign prefix wraps the rest of
    the token, and a ``:`` prefix makes a keyword."""
    while True:
        if text in _CONSTANTS:
            yield start, Keyword.Constant, text
            return
        if text in _KEYWORDS and (after_paren or text not in _HEAD_ONLY):
            kind = Keyword if any(c.isalpha() for c in text) else Operator
            yield start, kind, text
            return
        if text in _MACROS:
            yield start, Keyword.Pseudo, text
            return
        if len(text) >= 2 and text[0] in "^*":
            width = 2 if text.startswith("**") else 1
        else:
            rest = text.lstrip("+-")
            width = (
                len(text) - len(rest)
                if text[0] in "+-" and rest and (rest[0].isalnum() or rest[0] in "_.")
                else 0
            )
        if width:
            yield start, Operator, text[:width]
            [text, start, after_paren] = [text[width:], start + width, False]
            continue
        if len(text) >= 2 and text[0] == ":":
            kind = String.Symbol
        elif text in _BUILTINS:
            kind = _BUILTINS[text]
        else:
            kind = Name.Function if after_paren else Name
        yield start, kind, text
        return


def _symbol(lexer, match):
    start = match.start()
    after_paren = start > 0 and match.string[start - 1] == "("
    yield from _symbol_tokens(match.group(), start, after_paren)


def _number(lexer, match):
    text = match.group()
    digits = text.lstrip("+-").lower()
    if digits.endswith("j"):
        kind = Number
    elif digits.startswith("0x"):
        kind = Number.Hex
    elif digits.startswith("0o"):
        kind = Number.Oct
    elif digits.startswith("0b"):
        kind = Number.Bin
    elif "." in digits or "e" in digits:
        kind = Number.Float
    else:
        kind = Number.Integer
    yield match.start(), kind, text


def _prefixed(lexer, match):
    """A token with a wrapper (^, *, **) before an opening or a quote."""
    text = match.group()
    body = 1 if text[-1] in "([{" else len(text.lstrip("^*"))
    if len(text) > body:
        yield match.start(), Operator, text[:-body]
    kind = Punctuation if text[-1] in "([{" else Operator
    yield match.start() + len(text) - body, kind, text[-body:]


# The quotes that open a string, longest first; in a field also each double
# quote escaped (\" and \"\"\"), which opens a deprecated string there.
_QUOTES = sorted(_STRING_TYPES, key=len, reverse=True)
_ESCAPED_QUOTES = ["".join("\\" + c for c in q) for q in _QUOTES if q[0] == '"']
_STRING_ESCAPE = (
    r"\\(?:N\{[^}\n]*\}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|x[0-9a-fA-F]{2}|[0-7]{1,3}"
    r"|[\\abfnrtv\"'\n])"
)
_BYTES_ESCAPE = r"\\(?:x[0-9a-fA-F]{2}|[0-7]{1,3}|[\\abfnrtv\"'\n])"


def _prefix_kinds():
    """[prefix regex, body kind] for the string prefixes the reader knows, by
    kind: s(tring), b(ytes) or i(nterpolated, f/t), with r if raw."""
    groups = {}
    for prefix in PLAIN_STRING_PREFIXES + _F_STRING_PREFIXES + _T_STRING_PREFIXES:
        interpolated = prefix in _F_STRING_PREFIXES + _T_STRING_PREFIXES
        kind = ("i" if interpolated else "b" if "b" in prefix else "s") + (
            "r" if "r" in prefix else ""
        )
        groups.setdefault(kind, []).append(prefix)
    return [
        ["(?i:" + "|".join(prefixes) + ")", kind] for kind, prefixes in groups.items()
    ]


def _string_openers(in_field):
    """The rules opening a string (each body is the state named by its kind
    and quote); the reader's \\w* before a quote is the prefix."""
    rules = []
    for quote in _QUOTES:
        for prefix, kind in _prefix_kinds():
            rules.append(
                (
                    f"({prefix})({re.escape(quote)})",
                    bygroups(String.Affix, String),
                    f"{kind}{quote}",
                )
            )
        # Any other prefix is the reader's error; the body reads as a string's.
        rules.append(
            (rf"(\w+)({re.escape(quote)})", bygroups(String.Affix, String), f"s{quote}")
        )
    if in_field:
        for quote in _ESCAPED_QUOTES:
            rules.append(
                (rf"(\w*)({re.escape(quote)})", bygroups(String.Affix, String), quote)
            )
    return rules


def _string_body(kind, quote):
    """The body of a string closed by ``quote`` of ``kind`` (``_prefix_kinds``):
    escapes unless raw, and in an f/t-string ``{{``/``}}`` and fields."""
    rules = [(re.escape(quote), String, "#pop")]
    if "r" not in kind:
        rules.append((_BYTES_ESCAPE if kind == "b" else _STRING_ESCAPE, String.Escape))
    # A backslash and the character after it are one pair, raw too, so \"
    # never closes the string; but a backslash never escapes a brace
    # (scan_string_body), so before one it is a lone backslash.
    interpolated = kind.startswith("i")
    braces = re.escape(_FIELD_BRACES) if interpolated else ""
    rules.append((rf"\\[^{braces}]" if braces else r"\\[\s\S]", String))
    stops = re.escape("\\" + quote[0]) + braces
    if interpolated:
        rules += [
            (r"\{\{|\}\}", String.Escape),
            (r"\{", String.Interpol, "field"),
            (r"\}", String),
        ]
    rules += [(f"[^{stops}]+", String), (f"[{stops}]", String)]
    return rules


def _escaped_string_body(quote):
    """A field's deprecated \\"-opened string: only its own quote closes it."""
    return [
        (re.escape(quote), String, "#pop"),
        (r'\\\\"', String.Escape),
        (r"\\[\s\S]", String.Escape),
        (r"[^\\]+", String),
    ]


# What each reader token becomes; "string" is replaced by the openers above.
_LABEL_ACTIONS = {
    "opening": _prefixed,
    "closing": Punctuation,
    "dangling-quote": _prefixed,
    "meta-indicator": _prefixed,
    "bad-prefix": Operator,
    "comment": Comment.Single,
    "number": _number,
    "comma": Operator,
    "symbol": _symbol,
    "new-line": Whitespace,
    "spaces": Whitespace,
}


def _code_rules(state):
    """The rules of lispython code at the top level (``root``), at an f/t-string
    field's own level (``field``) or inside a field's brackets
    (``field-nested``), from the reader's token patterns for that place."""
    in_field = state != "root"
    rules = []
    if state == "field":
        # The field's delimiters: its end, a conversion, a format spec.
        rules += [
            (r"\}", String.Interpol, "#pop"),
            (r"!(?!=)\w*", String.Interpol),
            (":", String.Interpol, "spec"),
        ]
    for pattern, label in _token_patterns(in_field, field_level=state == "field"):
        if label == "string":
            rules += _string_openers(in_field)
            continue
        if label not in _LABEL_ACTIONS:
            raise RuntimeError(f"lispython reader token with no highlighting: {label}")
        action = _LABEL_ACTIONS[label]
        if state == "field-nested" and label in ("opening", "closing"):
            rules.append(
                (pattern, action, "field-nested" if label == "opening" else "#pop")
            )
        elif state == "field" and label == "opening":
            rules.append((pattern, action, "field-nested"))
        else:
            rules.append((pattern, action))
    # Blanks the reader skips between tokens (tab, CR, form feed), and a lone
    # backslash a field cannot otherwise take.
    rules.append((r"\s+", Whitespace))
    if in_field:
        rules.append((r"\\", Text))
    return rules


def _tokens():
    tokens = {state: _code_rules(state) for state in ("root", "field", "field-nested")}
    # A format spec: text and nested fields, up to the field's closing brace.
    tokens["spec"] = [
        (r"\{", String.Interpol, "field"),
        (r"\}", String.Interpol, "#pop:2"),
        (r"[^{}]+", String),
    ]
    kinds = {kind for _, kind in _prefix_kinds()} | {"s"}
    for quote in _QUOTES:
        for kind in kinds:
            tokens[f"{kind}{quote}"] = _string_body(kind, quote)
    for quote in _ESCAPED_QUOTES:
        tokens[quote] = _escaped_string_body(quote)
    return tokens


class LispythonLexer(RegexLexer):
    """Lexer for lispython (``.lpy``) source."""

    name = "LisPython"
    url = "https://jetack.github.io/lispython"
    aliases = ["lispython", "lpy"]
    filenames = ["*.lpy"]
    mimetypes = ["text/x-lispython"]
    # The reader's patterns use $ for the end of the source, as it compiles them.
    flags = 0

    tokens = _tokens()

    def analyse_text(text):
        return 0.8 if "(require lispy.macros" in text else 0.0
