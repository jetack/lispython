import contextvars
import re
from bisect import bisect_left, bisect_right
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
    Paren,
    String,
    Symbol,
    Unquote,
    UnquoteSplice,
    Wrapper,
)
from lispy.core.utils import augassignop_dict

# ⚠️ One leading digit, not \d+: \d+ and the [\dA-Za-z]* after it match the
# same digits, so a long digit run before a char that fails the number's stop
# lookahead was tried at every split of the two (quadratic). The token set is
# unchanged. The repeat cannot be possessive: a complex's [jJ] and a
# scientific's [eE] need the letters given back.
_INT = r"\d[\dA-Za-z]*(?:_[\dA-Za-z]+)*"
_FLOATS = [
    "\\d+(?:_\\d+)*\\.\\d+(?:_\\d+)*",
    "\\d+(?:_\\d+)*\\.\\d*",
    "\\d*\\.\\d+(?:_\\d+)*",
]
_SCIENTIFIC = [x + "[eE][\\-\\+]?\\d+(?:_\\d+)*" for x in _FLOATS + [_INT]]
_NUMBER = "|".join(_SCIENTIFIC + _FLOATS + [_INT])
_COMPLEX = f"(?:(?:{_NUMBER})[+-])?(?:{_NUMBER})[jJ]"
# The wrapper an opening or a meta-indicator may carry: one of ^, *, **
# (wrapper_dict). ^ followed by stars has no meaning before an opening or a
# quote/unquote (whatever follows it), nor alone (^* ending the token); that is
# the "bad-prefix" token, which _parse refuses. It must match before "symbol",
# which would otherwise swallow ^*( or ^*' whole (giving an unbalanced paren
# or the call f(*', a)); ^*Ts stays a symbol.
_WRAPPER = "(?:\\^|\\*{1,2})?"
_OPENING = "[\\(\\{\\[]"
_META_CHARS = "(?:'|`|~@|~)"
# What can follow a form but never start one: it ends a number or a symbol
# (so a comment needs no space before it: a;c is the symbol a), a quote
# followed by it quotes nothing, and ~ followed by it is the symbol ~.
_FORM_END = "\\s\\)\\}\\],;"
# What ends a symbol token.
_SYMBOL_STOP = _FORM_END + '\\"\\['


def _token_patterns(in_field, field_level=False):
    """The token patterns, outside replacement fields or inside one.

    Inside a field \\" opens an "escaped" string (it is how a "..." literal
    holds a string in a field: f"{(.get d \\"H:M:S\\")}"; a token starting
    with \\" is never valid Python, so the spelling is lispython's own) and
    ends a symbol. A bare quote opens a string there too, scanned as at the
    top level (scan_string_body). An escaped triple quote is three \\", never
    a mix.

    At the field's own level (``field_level``, outside its brackets) the
    expression ends at a delimiter (``:``, ``}``, or ``!`` not before ``=``):
    no token runs past one, and a lookahead for the end of the source takes it
    for that end (so x:>4 is the symbol x, 1!r the number 1). A comment there
    ends at one too; inside brackets it runs to the end of the line, as
    anywhere else.

    ⚠️ Never let a token run past the delimiter and cut it back there: a
    symbol runs to the next "}", which in a spec chain f"{x:{x:{x:...}}}" is
    past every field, so each field would rescan the rest of the source.
    """
    quote = '(?:\\\\"){3}|\\\\"|' if in_field else ""
    end = "$|:|!(?!=)" if field_level else "$"
    stop_ahead = (
        f'(?=[{_SYMBOL_STOP}]|\\\\"|{end})' if in_field else f"(?=[{_SYMBOL_STOP}]|$)"
    )
    # A symbol character (in a field), and a comment's. ⚠️ Their repeats are
    # possessive (++, *+): a plain + makes re keep a backtracking point per
    # character, hundreds of MB for a long symbol or comment.
    if field_level:
        char = f"[^{_SYMBOL_STOP}\\\\:!]|!(?==)"
        comment_char = "[^\\n}:!]|!(?==)"
    else:
        [char, comment_char] = [f"[^{_SYMBOL_STOP}\\\\]", "[^\\n]"]
    return [
        [_WRAPPER + _OPENING, "opening"],
        ["[\\)\\}\\]]", "closing"],
        # Only the opening of a string (prefix and quote): scan_string_body,
        # the one scanner of string bodies, reads the rest.
        [f'\\w*(?:{quote}"""|\'\'\'|")', "string"],
        # ' ` ~@ (any wrapper) with no form right after: refused by _parse. ~
        # alone is the invert operator, a symbol.
        [_WRAPPER + f"(?:'|`|~@)(?=[{_FORM_END}]|{end})", "dangling-quote"],
        [_WRAPPER + _META_CHARS + f"(?![{_FORM_END}]|{end})", "meta-indicator"],
        [f"\\^\\*+(?:{_OPENING}|{_META_CHARS}|{stop_ahead})", "bad-prefix"],
        [f";(?:{comment_char})*+", "comment"],
        [
            "[+-]*(?:" + "|".join([_COMPLEX, _NUMBER]) + f")(?=[{_FORM_END}]|{end})",
            "number",
        ],
        [",", "comma"],
        # "[" ends a symbol so that f[T] is a symbol glued to a bracket. At a
        # field's own level a lone ":" or "!" matches too, so the lexer sees
        # the delimiter where a token starts.
        [
            f'(?:{char}|\\\\(?!"))++' + ("|[:!]" if field_level else "")
            if in_field
            else f"[^{_SYMBOL_STOP}]+",
            "symbol",
        ],
        ["\\n", "new-line"],
        [" +", "spaces"],
    ]


_TOKEN_RE = re.compile("|".join(f"({p})" for [p, _] in _token_patterns(False)))
_FIELD_TOKEN_RE = re.compile(
    "|".join(f"({p})" for [p, _] in _token_patterns(True, field_level=True))
)
_BRACKETED_FIELD_TOKEN_RE = re.compile(
    "|".join(f"({p})" for [p, _] in _token_patterns(True))
)
_TOKEN_LABELS = [label for [_, label] in _token_patterns(False)]
_NUMBER_TOKEN_RE = re.compile("[+-]*(?:" + "|".join([_COMPLEX, _NUMBER]) + ")$")
_STRING_PREFIX_RE = re.compile("\\w*")
# Where a string body scan has to look: an escape, a quote, and (in an f/t
# string) a brace. Everything between is copied as one slice.
_INTERPOLATED_BODY_STOP = re.compile("[\\\\{}\"']")
_PLAIN_BODY_STOP = re.compile("[\\\\\"']")


def _source_locator(src):
    """locate(i, j) -> position_info of src[i:j], in any order of calls.

    The line starts are found once, so a position costs at most a bisection,
    never a count from the start of the source (or of a token): counting made
    a literal with many fields quadratic.
    """
    line_starts = [0]
    line_starts.extend(m.end() for m in re.finditer("\n", src))
    line_starts.append(len(src) + 1)
    # [line, its start, the next line's start] of the last position: tokens
    # come in order, so most lookups stay on that line.
    last = [1, 0, line_starts[1]]

    def position(i):
        if last[1] <= i < last[2]:
            return [last[0], i - last[1]]
        line = bisect_right(line_starts, i)
        last[:] = [line, line_starts[line - 1], line_starts[line]]
        return [line, i - last[1]]

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


# The field-quote escape: \" written for " inside a string opened with \" in a
# replacement field (an "escaped" string, see _token_patterns). ⚠️ It belongs
# to that string alone, never to everything lexed inside a field: since PEP 701
# a field also holds strings opened with a bare quote, and in those CPython's
# own escapes apply (f"{g("say \"hi\"")}", f"{g("a\\")}").
_FIELD_QUOTE_ESCAPE = re.compile('\\\\(?=")')


def _unescape_field_quotes(text):
    # An escaped string's text with each \" read as " (see above).
    return text.replace('\\"', '"')


def _quote_length(src, i, quote, escaped):
    """Length of the source text at ``i`` that closes a ``quote`` string, or 0.

    In an ``escaped`` string each " of the quote may be written \\".
    """
    if not escaped:
        return len(quote) if src.startswith(quote, i) else 0
    j = i
    for _ in quote:
        if src.startswith('"', j):
            j += 1
        elif src.startswith('\\"', j):
            j += 2
        else:
            return 0
    return j - i


def scan_string_body(src, i, quote, raw, kind, escaped, spec=False):
    """The one scanner of string bodies: a plain string's whole body, and an
    f/t-string's literal text up to its next field, its end, or the end of a
    format spec.

    Scans ``src`` from ``i`` and returns ``[piece, stop, at]``: the
    text read (source text, escapes unprocessed, ``{{``/``}}`` collapsed, and
    in an ``escaped`` string \\" read as "), why it stopped, and where:
    ``"quote"`` at the closing quote, ``"field"`` at a single ``{``,
    ``"close"`` at a single ``}``, ``"eof"`` at the end of the source. For a
    plain string (``kind`` "string") braces are ordinary characters. In a
    format spec (``spec``) no brace is doubled, as in CPython: every ``{``
    opens a field and the first ``}`` ends the spec. A string not ``escaped``
    reads as Python reads it, wherever it stands: a backslash and the
    character after it are one pair (raw too), so \\" never closes it.
    """
    interpolated = kind != "string"
    stops = _INTERPOLATED_BODY_STOP if interpolated else _PLAIN_BODY_STOP
    n = len(src)
    buf = []
    while True:
        m = stops.search(src, i)
        if m is None:
            buf.append(src[i:])
            return ["".join(buf), "eof", n]
        if m.start() > i:
            buf.append(src[i : m.start()])
        i = m.start()
        c = src[i]
        if c == "\\" and escaped and src.startswith('"', i + 1):
            if _quote_length(src, i, quote, escaped):
                return ["".join(buf), "quote", i]
            buf.append('"')
            i += 2
        elif c == "\\":
            if i + 1 >= n:
                buf.append(c)
                i += 1
            elif not raw and src.startswith("N{", i + 1):
                close = src.find("}", i + 3)
                close = n - 1 if close < 0 else close
                escape = src[i : close + 1]
                buf.append(_unescape_field_quotes(escape) if escaped else escape)
                i = close + 1
            elif interpolated and src[i + 1] in "{}":
                # A backslash never escapes a brace: it is a literal backslash.
                buf.append("\\" if raw else "\\\\")
                i += 1
            elif escaped and src.startswith('\\"', i + 1):
                buf.append('\\"')
                i += 3
            else:
                buf.append(src[i : i + 2])
                i += 2
        elif c in "\"'":
            if _quote_length(src, i, quote, escaped):
                return ["".join(buf), "quote", i]
            buf.append(c)
            i += 1
        elif src.startswith(c, i + 1):
            if spec:
                return ["".join(buf), "field" if c == "{" else "close", i]
            buf.append(c)
            i += 2
        else:
            return ["".join(buf), "field" if c == "{" else "close", i]


class _LiteralState:
    """An f/t-string literal the lexer is inside."""

    __slots__ = (
        "kind",
        "raw",
        "quote",
        "escaped",
        "quote_start",
        "body",
    )

    def __init__(self, kind, raw, quote, escaped, quote_start, body):
        self.kind = kind
        self.raw = raw
        self.quote = quote
        self.escaped = escaped  # opened with \" in a field (_FIELD_QUOTE_ESCAPE)
        self.quote_start = quote_start
        self.body = body


class _FieldState:
    """A replacement field the lexer is inside: in its expression, or (once
    ``spec`` is set) in its format spec."""

    __slots__ = (
        "literal",
        "open",
        "depth",
        "spec",
        "previous",
        "tail",
        "record",
    )

    def __init__(self, literal, open_brace, record):
        self.literal = literal
        self.open = open_brace
        self.depth = 0  # brackets open in the expression
        self.spec = False
        self.previous = None  # for the comma rule, as at the top level
        # [first token index, start, end, type] of the last token at depth 0,
        # with the quotes glued before it ('x): what {x=} takes the = off.
        self.tail = None
        self.record = record  # shared with the parser through the token


class _Lexer:
    """The tokens of a source, in one pass over it.

    Outside f/t-strings a token is what ``_TOKEN_RE`` matches. An f/t-string
    is lexed in place, CPython-style, over a mode stack of the literals and
    fields open (``_LiteralState``/``_FieldState``): its opening, each piece of
    literal text, each field's ``{``, the tokens of its expression (lexed like
    any others, with bracket depth tracked so ``!``, ``:`` and ``}`` delimit it
    only outside brackets), its conversion and format spec (itself literal text
    and fields), its ``}``, and the closing quote. So no character is read more
    than a bounded number of times however deep literals and fields nest, and
    every position is already absolute. A malformed literal is an "error"
    token where it is found, and lexing stops there: the parser raises it when
    it gets there, so errors come in source order.
    """

    def __init__(self, src):
        self.src = src
        self.locate = _source_locator(src)
        self.tokens = []
        self.stack = []  # _LiteralState / _FieldState, innermost last
        self.pos = 0
        self.previous = None  # the last token outside any field
        self.done = False
        # Where each field-quote escape's backslash is, in source order: what
        # a field's text (_FieldText) drops to read as the field means.
        self.escapes = []

    def run(self):
        while not self.done:
            top = self.stack[-1] if self.stack else None
            if top is None or (isinstance(top, _FieldState) and not top.spec):
                self._token(top)
            else:
                self._body(top)
        return self.tokens

    def _emit(self, text, tktype, i, j, extra=None):
        self.tokens.append([text, tktype, self.locate(i, j), extra])

    def _note_escapes(self, i, j):
        """Record the field-quote escapes in ``src[i:j]``, a span of an escaped
        string (its quotes or a piece of its body). Spans come in source order,
        so ``escapes`` stays sorted."""
        self.escapes.extend(
            m.start() for m in _FIELD_QUOTE_ESCAPE.finditer(self.src, i, j)
        )

    def _error(self, msg, i, j):
        return LispyCompileError(msg, String('""', **self.locate(i, j)))

    def _fail(self, error):
        position_info = {
            "lineno": error.lineno,
            "col_offset": error.offset - 1,
            "end_lineno": error.end_lineno,
            "end_col_offset": error.end_offset - 1,
        }
        self.tokens.append([error.msg, "error", position_info, error])
        self.done = True

    def _token(self, field):
        """One token at the top level or in a field's expression."""
        src = self.src
        regex = (
            _TOKEN_RE
            if field is None
            else _FIELD_TOKEN_RE
            if field.depth == 0
            else _BRACKETED_FIELD_TOKEN_RE
        )
        m = regex.search(src, self.pos)
        if m is None:
            if field is not None:
                self._fail(self._eof_error())
            self.done = True
            return
        start = m.start()
        if field is not None and field.depth == 0:
            c = src[start]
            if c in ":}" or (c == "!" and not src.startswith("=", start + 1)):
                self._delimiter(field, start)
                return
        tk = m.group()
        tktype = _TOKEN_LABELS[m.lastindex - 1]
        end = m.end()
        if tktype == "string":
            self._string(start, tk)
            return
        context = field if field is not None else self
        if tktype == "comma":
            # A comma is whitespace, except right after "(" or "{": (, a b) is a
            # tuple and {, a b} a set.
            previous = context.previous
            kept = previous is not None and previous[1] == "opening"
            tktype = "symbol" if kept and previous[0][-1] in "({" else "spaces"
        if field is not None:
            if tktype == "opening" or (tktype == "bad-prefix" and tk[-1] in "([{"):
                field.depth += 1
            elif tktype == "closing" and field.depth:
                field.depth -= 1
        if tktype not in ("new-line", "spaces"):
            context.previous = [tk, tktype]
        if tktype not in ("new-line", "spaces", "comment"):
            self._emit(tk, tktype, start, end)
            if field is not None and field.depth == 0:
                tail = field.tail
                glued = tail is not None and tail[2] == start
                glued = glued and tail[3] == "meta-indicator"
                first = tail[:2] if glued else [len(self.tokens) - 1, start]
                field.tail = [*first, end, tktype]
        self.pos = end

    def _string(self, start, opening):
        """A string opening at ``start``: a plain string is one token; an f/t
        string opens a literal, whose body is lexed next."""
        src = self.src
        prefix = _STRING_PREFIX_RE.match(opening).group()
        quote_text = opening[len(prefix) :]
        quote = _unescape_field_quotes(quote_text)
        lowered = prefix.lower()
        raw = "r" in lowered
        kind = (
            "t-string" if "t" in lowered else "f-string" if "f" in lowered else "string"
        )
        # Only a field's token pattern lets a quote start with \.
        escaped = quote_text[0] == "\\"
        body = start + len(opening)
        if kind == "string":
            [_, stop, at] = scan_string_body(src, body, quote, raw, kind, escaped)
            error = None
            if stop == "quote":
                end = at + _quote_length(src, at, quote, escaped)
            else:
                end = len(src)
                error = self._eof_error(kind, quote, body - len(quote_text), body)
            text = src[start:end]
            if escaped:
                text = _unescape_field_quotes(text)
                self._note_escapes(start, end)
            # Plain strings are a single token, carrying any error: the parser
            # raises it after the prefix check.
            self._emit(text, quote, start, end, error)
            if error is not None:
                self.done = True
                return
            (self.stack[-1] if self.stack else self).previous = [text, quote]
            self.pos = end
            return
        self._emit(opening, "interpolated-start", start, body, prefix)
        if escaped:
            self._note_escapes(start, body)
        self.stack.append(
            _LiteralState(kind, raw, quote, escaped, body - len(quote_text), body)
        )
        self.pos = body

    def _eof_error(self, kind=None, quote=None, quote_start=None, body=None):
        """The source ends inside a field (``expecting '}'`` at the innermost
        one's ``{``) or else inside a literal (unterminated, at its quote)."""
        for entry in reversed(self.stack):
            if isinstance(entry, _FieldState):
                return self._error(
                    f"{entry.literal.kind}: expecting '}}'", entry.open, entry.open + 1
                )
        if kind is None:
            literal = self.stack[-1]
            [kind, quote] = [literal.kind, literal.quote]
            [quote_start, body] = [literal.quote_start, literal.body]
        triple = "triple-quoted " if len(quote) == 3 else ""
        # At the opening quote, as CPython reports it.
        return self._error(f"unterminated {triple}{kind} literal", quote_start, body)

    def _body(self, top):
        """Literal text of an f/t-string, or of a field's format spec, up to
        what ends it."""
        [literal, spec] = (
            [top, None] if isinstance(top, _LiteralState) else [top.literal, top]
        )
        src = self.src
        [piece, stop, at] = scan_string_body(
            src,
            self.pos,
            literal.quote,
            literal.raw,
            literal.kind,
            literal.escaped,
            spec is not None,
        )
        if stop == "eof":
            self._fail(self._eof_error())
            return
        self._emit(piece, "interpolated-middle", self.pos, at)
        if literal.escaped:
            self._note_escapes(self.pos, at)
        kind = literal.kind
        if stop == "quote":
            if spec is not None:
                # The literal ends inside a format spec.
                self._fail(
                    self._error(f"{kind}: expecting '}}'", spec.open, spec.open + 1)
                )
                return
            end = at + _quote_length(src, at, literal.quote, literal.escaped)
            self._emit(src[at:end], "interpolated-end", at, end)
            if literal.escaped:
                self._note_escapes(at, end)
            self.stack.pop()
            context = self.stack[-1] if self.stack else self
            context.previous = [src[at:end], "interpolated-end"]
            self.pos = end
        elif stop == "field":
            record = {}
            self._emit("{", "field-start", at, at + 1, record)
            self.stack.append(_FieldState(literal, at, record))
            self.pos = at + 1
        elif spec is None:
            self._fail(self._error(f"{kind}: single '}}' is not allowed", at, at + 1))
        else:
            self._close_field(spec, at)

    def _delimiter(self, field, at):
        """The end of a field's expression, at a ``}``, ``:`` or ``!`` outside
        brackets: the expression's text (with a trailing ``=`` read as Python
        reads ``{x=}``), then the conversion or format spec that follows.

        The token carries the expression's extent, never its text: an outer
        field's text holds every field nested in it, so a copy per field is
        memory quadratic in the nesting. The FStrExpr keeps the extent too
        (_FieldText).
        """
        src = self.src
        expr_start = field.open + 1
        last = at  # past the expression's last non-space character
        while last > expr_start and src[last - 1].isspace():
            last -= 1
        # Python's {x=}: a final "=" not ending an operator (x==, x<=).
        debug = (
            last - 1 > expr_start
            and src[last - 1] == "="
            and src[last - 2] not in _DEBUG_BLOCKERS
        )
        if debug:
            self._drop_debug_equals(field, last - 1)
        self._emit(
            src[at],
            "field-expression-end",
            at,
            at,
            {
                "start": expr_start,
                "end": at,
                "debug": last - 1 if debug else None,  # the "=" of {x=}
                "echo": self.locate(expr_start, at),
                "escapes": self.escapes,  # shared, never copied (_FieldText)
            },
        )
        if src[at] == "}":
            self._close_field(field, at)
            return
        if src[at] == "!":
            at = self._conversion(field, at)
            if at is None:
                return
            if src[at] == "}":
                self._close_field(field, at)
                return
        self._emit(":", "field-spec-start", at, at + 1)
        field.spec = True
        self.pos = at + 1

    def _conversion(self, field, bang):
        """The ``!conv`` at ``bang``: emits its token and returns the index of
        the ``}`` or ``:`` after it, or fails and returns None.

        As CPython 3.12 reads it: a literal or source ending before the ``}``
        or ``:`` is "expecting '}'" whatever came between, and the name is
        checked only once one of them follows it.
        """
        src = self.src
        kind = field.literal.kind
        name_start = bang + 1
        while name_start < len(src) and src[name_start].isspace():
            name_start += 1
        name_end = name_start
        while name_end < len(src) and (src[name_end].isalnum() or src[name_end] == "_"):
            name_end += 1
        rest = name_end
        while rest < len(src) and src[rest].isspace():
            rest += 1
        conversion = src[name_start:name_end]
        if rest == len(src):
            self._fail(self._eof_error())
            return None
        if _quote_length(src, rest, field.literal.quote, field.literal.escaped):
            # CPython's column: what follows the space after "!" (x! r"), else
            # the quote.
            at = name_start if name_start > bang + 1 else rest
            self._fail(self._error(f"{kind}: expecting '}}'", at, at + 1))
            return None
        if conversion and conversion not in conversion_dict:
            msg = (
                f"invalid conversion character {conversion!r}: "
                "expected 's', 'r', or 'a'"
            )
            self._fail(self._error(f"{kind}: {msg}", bang, name_end))
            return None
        if src[rest] not in "}:":
            # CPython: a conversion that is no name at all (f"""{x!" """).
            self._fail(
                self._error(
                    f"{kind}: expecting '}}'"
                    if conversion
                    else f"{kind}: invalid conversion character",
                    rest,
                    rest + 1,
                )
            )
            return None
        if not conversion or name_start > bang + 1:
            msg = (
                "conversion type must come right after the exclamation mark"
                if conversion
                else "missing conversion character"
            )
            self._fail(self._error(f"{kind}: {msg}", bang, bang + 1))
            return None
        self._emit(src[bang:name_end], "field-conversion", bang, name_end, conversion)
        return rest

    def _drop_debug_equals(self, field, equals):
        """Take the ``=`` of ``{x=}`` at ``equals`` off the expression's last
        token (a symbol ending there), lexing the text before it again with
        any quotes glued to it, as the expression's own text would lex."""
        tail = field.tail
        if tail is None or tail[2] != equals + 1 or tail[3] != "symbol":
            return
        [index, start, _, _] = tail
        del self.tokens[index:]
        pos = start
        while pos < equals:
            m = _FIELD_TOKEN_RE.search(self.src, pos, equals)
            if m is None:
                break
            tktype = _TOKEN_LABELS[m.lastindex - 1]
            if tktype not in ("new-line", "spaces", "comment"):
                self._emit(m.group(), tktype, m.start(), m.end())
            pos = m.end()

    def _close_field(self, field, at):
        self._emit("}", "field-end", at, at + 1)
        field.record["end"] = self.locate(at + 1)
        self.stack.pop()
        self.pos = at + 1


def tokenize(src):
    """Tokens as [text, type, position_info, extra], in one pass (_Lexer).

    ``extra`` is the error of a malformed plain string, the data of an f/t
    string's own tokens, else None. An "error" token (placed at the error,
    which is its ``extra``) ends the tokens of a source the lexer cannot read
    on; otherwise they run to the end of the source, however deep f/t-strings
    and their fields nest (the lexer has no limit of its own), in time and
    memory linear in its length.
    """
    return deque(_Lexer(src).run())


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
    while isinstance(node, (Wrapper, MetaIndicator)) and not isinstance(node, FStrExpr):
        if _unquoted_symbol_p(node):
            break
        prefixes.append(node)
        node = node.value
    glued = (type(node) is Symbol or _unquoted_symbol_p(node)) and (
        node.end_lineno,
        node.end_col_offset,
    ) == (lineno, col_offset)
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
        return _parse_into(src, [])
    rst = []
    token = _PARTIAL.set(True)
    try:
        _parse_into(src, rst)
    except Exception:
        pass
    finally:
        _PARTIAL.reset(token)
    return rst


def _parse_into(src, rst):
    """Parse ``src`` into ``rst``, one token at a time on an explicit stack.

    Every open form is a stack entry, never a Python frame: a bracket's node,
    a pending prefix, and an f/t-string's literal, fields and format specs
    (``_Interpolated``/``_Field``), which the lexer's tokens build in place,
    so any nesting parses in one pass with no recursion.
    """
    tokens = tokenize(src)
    stack = []
    while tokens:
        [t, tktype, position_info, extra] = tokens.popleft()
        if tktype in _LEXER_ONLY_TOKENS:
            if tktype == "error":
                raise extra
            if tktype == "interpolated-start":
                stack.append(string_parse(t, tktype, position_info, extra))
            else:
                _interpolated_step(src, stack, t, tktype, position_info, extra)
            continue
        [lineno, col_offset, end_lineno, end_col_offset] = position_info_into_list(
            position_info
        )
        if tktype == "opening":
            wrapper = t[:-1]
            opening = t[-1]
            container = stack[-1] if stack else rst
            glueable = isinstance(container, (list, Expression, _Field))
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
        elif tktype == "bad-prefix":
            [prefix, indicator] = _split_bad_prefix(t)
            raise LispyCompileError(
                f"{prefix} prefix has no meaning before {indicator!r}"
                if indicator
                else f"{prefix} has no meaning on its own",
                Symbol(t, **position_info),
            )
        elif tktype == "dangling-quote":
            raise LispyCompileError(
                f"{t} must be directly followed by a form", Symbol(t, **position_info)
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
            elif tktype == "interpolated-end":
                e = stack.pop().close(end_lineno, end_col_offset)
            elif tktype in _STRING_TYPES:
                e = string_parse(t, tktype, position_info, extra)
            else:
                e = token_parse(t, tktype, position_info)
            closed = 0
            while stack and isinstance(stack[-1], (MetaIndicator, Wrapper, GluedHead)):
                popped = stack.pop()
                if isinstance(popped, GluedHead):
                    popped.params = e
                else:
                    popped.value = e
                if closed:
                    # e already carries these ends down its whole chain (set
                    # when it was popped), so a wrapper's update_dict would
                    # only set them again: setting this node alone keeps a
                    # chain of N prefixes linear rather than quadratic.
                    popped.__dict__["end_lineno"] = end_lineno
                    popped.__dict__["end_col_offset"] = end_col_offset
                else:
                    popped.update_dict("end_lineno", end_lineno)
                    popped.update_dict("end_col_offset", end_col_offset)
                closed += 1
                e = popped
            (stack[-1] if stack else rst).append(e)
    _check_all_closed(stack)
    return rst


def _split_bad_prefix(token):
    """A bad-prefix token as [prefix, opening or meta-indicator or ""]."""
    end = len(token) - len(token.lstrip("^*"))
    return [token[:end], token[end:]]


special_literals = ("True", "False", "None", "...")

_STRING_TYPES = ("'''", '"""', '"')


def token_parse(token, tktype, position_info):
    """The node of a symbol or number token (strings go to string_parse).

    A prefix (``^``, ``:``, ``*``, ``**``, or a run of signs before a name or
    number) wraps the rest of the token, which may carry another. One loop
    peels them outside in, then the leaf is wrapped inside out, so a chain of
    prefixes costs no Python frame per prefix.
    """
    # Outermost first: [wrapper class, position] or ["signs", position, signs].
    prefixes = []
    while True:
        if token in augassignop_dict:
            node = Symbol(token, **position_info)
            break
        if token in special_literals:
            node = Constant(token, **position_info)
            break
        if tktype == "number" and (token[0] not in "+-" or _signed_complex_p(token)):
            node = Constant(token, **position_info)
            break
        if len(token) < 2:
            node = Symbol(token, **position_info)
            break
        if token[0] in "^:*":
            width = 2 if token.startswith("**") else 1
            wrapper = (
                Annotation
                if token[0] == "^"
                else Keyword
                if token[0] == ":"
                else wrapper_dict["*" * width]
            )
            prefixes.append([wrapper, position_info])
            position_info = {**position_info}
            position_info["col_offset"] += width
            token = token[width:]
            tktype = _operand_type(token, tktype)
            continue
        rest = token.lstrip("+-")
        if token[0] in "+-" and rest and (rest[0].isalnum() or rest[0] in "_."):
            signs = token[: len(token) - len(rest)]
            prefixes.append(["signs", position_info, signs])
            position_info = {**position_info}
            position_info["col_offset"] += len(signs)
            token = rest
            continue
        node = Symbol(token, **position_info)
        break
    for [wrapper, outer, *signs] in reversed(prefixes):
        node = (
            _unary_op_wrap(signs[0], node, outer) if signs else wrapper(node, **outer)
        )
    return node


def _signed_complex_p(token):
    # -1+2j is one Constant, so the sign binds to the real part as in Python;
    # a plain signed number stays (- 1), a unary op.
    parts = number_parts(token)
    return parts is not None and parts[2] is not None


def _operand_type(operand, tktype):
    """Token type of the text after a ^, * or ** or : prefix: the prefixed
    token is one lexeme, so it is the operand's own text that says whether
    it is a number (*1 is a starred 1, not a starred symbol named 1)."""
    return "number" if _NUMBER_TOKEN_RE.match(operand) else tktype


def _unary_op_wrap(signs, operand, position_info):
    """``(- (+ operand))`` for the signs ``-+`` starting at ``position_info``:
    each sign's Paren starts at that sign and ends where the token ends."""
    lineno = position_info["lineno"]
    col_offset = position_info["col_offset"]
    node = operand
    for idx in reversed(range(len(signs))):
        sign = Symbol(
            signs[idx],
            **{
                "lineno": lineno,
                "end_lineno": lineno,
                "col_offset": col_offset + idx,
                "end_col_offset": col_offset + idx + 1,
            },
        )
        node = Paren(sign, node, **{**position_info, "col_offset": col_offset + idx})
    return node


def string_parse(token, tktype, position_info, extra):
    """A plain string token's node, or the stack entry an f/t-string opening
    (``tktype`` "interpolated-start", ``extra`` its prefix) starts."""
    if tktype != "interpolated-start":
        [prefix, *_] = token.split(tktype)
        node = String(token, **position_info)
        interpolated_kind(prefix, node)
        if extra is not None:
            raise extra
        return node
    prefix = extra
    node = String(token, **position_info)
    kind = interpolated_kind(prefix, node)
    if kind == "t-string":
        # Gate where the prefix is seen, so an old interpreter reports the
        # version before any complaint about the literal's contents.
        if not _PARTIAL.get():
            require("t-string", node)
    return _Interpolated(kind, kind, "r" in prefix.lower(), position_info)


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

# A trailing "=" after one of these is part of an operator symbol, not a
# self-documenting "=" (``{x=}``).
_DEBUG_BLOCKERS = "=!<>:+-*/%&|^@~"


def _interpolation_error(kind, msg, position_info):
    return LispyCompileError(f"{kind}: {msg}", String('""', **position_info))


class _Interpolated:
    """``(f-string lit field lit ... lit)`` (or ``t-string``) being built on
    the parser's stack from the lexer's tokens: an f/t-string literal, or a
    field's format spec (always an ``f-string`` body; ``kind`` names the
    literal in error messages)."""

    def __init__(self, symbol, kind, raw, position_info):
        self.symbol = symbol
        self.kind = kind
        self.raw = raw
        self.start = [position_info["lineno"], position_info["col_offset"]]
        self.parts = []
        self.pending = None  # [value, position_info] of the literal text last read

    def literal(self, text, position_info):
        value = string_literal_value(
            "r" if self.raw else "", text, String('""', **position_info)
        )
        self.pending = [value, position_info]

    def field(self, fstr_expr, echo):
        [value, position_info] = self.pending
        if echo is not None:
            # The echoed {expr=} text joins the literal before it; CPython
            # places that constant over the echoed text.
            position_info = {
                **position_info,
                "end_lineno": echo["end_lineno"],
                "end_col_offset": echo["end_col_offset"],
            }
            if not value:
                position_info["lineno"] = echo["lineno"]
                position_info["col_offset"] = echo["col_offset"]
        self.parts.append(String(repr(value), **position_info))
        self.parts.append(fstr_expr)
        self.pending = None

    def close(self, end_lineno, end_col_offset):
        [value, position_info] = self.pending
        self.parts.append(String(repr(value), **position_info))
        position_info = {
            "lineno": self.start[0],
            "col_offset": self.start[1],
            "end_lineno": end_lineno,
            "end_col_offset": end_col_offset,
        }
        return Paren(Symbol(self.symbol, **position_info), *self.parts, **position_info)


class _FieldText:
    """A field expression's source text, read only when asked for (str()).

    ⚠️ Never sliced up front: an outer field's text holds every field nested
    in it, so a copy per field is memory quadratic in the nesting. The node
    keeps the source (shared) and the span; the few readers (the {x=} echo,
    Interpolation.str, a quasiquote's rebuild) slice what they need.

    The text drops the backslash of each field-quote escape in it (``escapes``,
    the lexer's sorted list of their indices), so (.get d \\"k\\") reads
    (.get d "k"); a string opened with a bare quote keeps its \\" as written.
    """

    __slots__ = ("src", "start", "end", "escapes")

    def __init__(self, src, start, end, escapes):
        [self.src, self.start, self.end] = [src, start, end]
        self.escapes = escapes

    def __str__(self):
        [src, start, end, escapes] = [self.src, self.start, self.end, self.escapes]
        lo = bisect_left(escapes, start)
        hi = bisect_left(escapes, end, lo)
        parts = []
        for k in escapes[lo:hi]:
            parts.append(src[start:k])
            start = k + 1
        parts.append(src[start:end])
        return "".join(parts)


class _Field:
    """A replacement field being built on the parser's stack.

    Follows Python: ``expr[=][!conv][:spec]``; whitespace around ``expr`` is
    part of its text; ``=`` echoes the text before it (and implies ``!r``
    unless a conversion or spec is given); the spec is itself an f-string
    body. ``list`` takes the expression's forms as a Paren's list takes its
    elements (so ``x[0]`` glues there too); there must be exactly one.
    """

    def __init__(self, owner, position_info, record):
        self.owner = owner
        self.position_info = position_info  # of the "{"
        self.record = record  # the lexer's: where the field ends
        self.list = []
        self.value = None
        self.expr_text = None
        self.debug = None  # {x=}: the "=" and the spaces after it
        self.echo = None
        self.conversion = None
        self.format_spec = None

    def append(self, node):
        self.list.append(node)

    def end_expression(self, src, data):
        if len(self.list) != 1:
            position_info = {**self.position_info}
            end = self.record.get("end")
            if end is not None:
                position_info["end_lineno"] = end["lineno"]
                position_info["end_col_offset"] = end["col_offset"]
            delimiter = "=" if data["debug"] is not None else src[data["end"]]
            raise _interpolation_error(
                self.owner.kind,
                f"valid expression required before {delimiter!r}"
                if not self.list
                else "expecting a single expression in a replacement field",
                position_info,
            )
        [self.value] = self.list
        end = data["end"]
        if data["debug"] is not None:
            [end, self.debug] = [data["debug"], src[data["debug"] : end]]
            self.echo = data["echo"]
        self.expr_text = _FieldText(src, data["start"], end, data["escapes"])

    def node(self, end):
        conversion = self.conversion
        if self.debug and conversion is None and self.format_spec is None:
            conversion = "r"
        return FStrExpr(
            self.value,
            conversion=conversion_dict.get(conversion, -1),
            format_spec=self.format_spec,
            expr_text=self.expr_text,
            debug=self.debug,
            lineno=self.position_info["lineno"],
            col_offset=self.position_info["col_offset"],
            end_lineno=end["end_lineno"],
            end_col_offset=end["end_col_offset"],
        )


def _field_on_top(stack):
    """The field whose expression just ended, refusing forms left open in it.

    The lexer ends an expression only outside its brackets, and a quote
    glued to nothing there is a dangling-quote token, so this is a guard.
    """
    if isinstance(stack[-1], _Field):
        return stack[-1]
    at = len(stack) - 1
    while not isinstance(stack[at], _Field):
        at -= 1
    _check_all_closed(stack[at + 1 :])
    raise _interpolation_error(
        stack[at].owner.kind,
        "expecting a single expression in a replacement field",
        stack[at].position_info,
    )


def _interpolated_step(src, stack, t, tktype, position_info, extra):
    """Apply one of an f/t-string's own tokens (other than its opening and
    closing quote) to the entries on the parser's stack."""
    if tktype == "interpolated-middle":
        stack[-1].literal(t, position_info)
    elif tktype == "field-start":
        stack.append(_Field(stack[-1], position_info, extra))
    elif tktype == "field-expression-end":
        _field_on_top(stack).end_expression(src, extra)
    elif tktype == "field-conversion":
        stack[-1].conversion = extra
    elif tktype == "field-spec-start":
        owner = stack[-1].owner
        # As in CPython, the spec's JoinedStr starts at the ":".
        stack.append(_Interpolated("f-string", owner.kind, owner.raw, position_info))
    else:  # field-end
        if isinstance(stack[-1], _Interpolated):
            spec = stack.pop()
            # ... and ends at the "}".
            stack[-1].format_spec = spec.close(
                position_info["lineno"], position_info["col_offset"]
            )
        field = stack.pop()
        stack[-1].field(field.node(position_info), field.echo)


# Token types the parse loop hands off before reading positions: the lexer's
# error, and an f/t-string's own tokens but its closing quote (which ends a form
# as a closing bracket does).
_LEXER_ONLY_TOKENS = frozenset(
    [
        "error",
        "interpolated-start",
        "interpolated-middle",
        "field-start",
        "field-expression-end",
        "field-conversion",
        "field-spec-start",
        "field-end",
    ]
)
