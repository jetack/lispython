"""The completion word boundary (completion_word_span, the reader's word) and
its published spec (python -m lispy.word_boundary), which lpy-mode and
vsc-lispy pin against."""

import itertools
import json
import re
import subprocess
import sys

import pytest

import lispy.macros
from lispy.core import parser
from lispy.core.compiler.literal import number_parts
from lispy.core.nodes import Constant, Paren, Symbol
from lispy.core.parser import (
    _signed_complex_p,
    caret_context,
    caret_symbol_p,
    completion_word_span,
    parse,
    token_prefixes,
    tokenize,
)
from lispy.word_boundary import CASE_INPUTS, case_input

# (text with | at the caret, (start, replace_start, end) on the caret's line
# or None for no word), checked by hand.
SPANS = [
    ("foo|", (0, 0, 3)),
    ("fo|o", (0, 0, 3)),
    ("|x", (0, 0, 1)),
    ("|", (0, 0, 0)),
    ("x1|", (0, 0, 2)),
    # Every character the reader lets into a symbol is a word character: a
    # macro name is no Python name.
    ("(->>|", (1, 1, 4)),
    ("(++|", (1, 1, 3)),
    ("(swap!|", (1, 1, 6)),
    ("ok?|", (0, 0, 3)),
    ("a~b|", (0, 0, 3)),
    ("a(b|", (0, 0, 3)),
    ("a'b|", (0, 0, 3)),
    ("except*|", (0, 0, 7)),
    ("!=|", (0, 0, 2)),
    ("a-|>b", (0, 0, 4)),
    # An augmented assignment is never peeled.
    ("**=|", (0, 0, 3)),
    ("+=|", (0, 0, 2)),
    ("-=|", (0, 0, 2)),
    ("^**=|", (1, 1, 4)),
    # A lone sign run is the word, the prefix of -> or --.
    ("-|", (0, 0, 1)),
    ("(-|", (1, 1, 2)),
    ("--|", (0, 0, 2)),
    ("-->x|", (0, 0, 4)),
    # Leading signs before a letter, digit, _ or . are the reader's unary
    # operators, never part of the word.
    ("-fo|", (1, 1, 3)),
    ("--x|", (2, 2, 3)),
    ("(- -fo|", (4, 4, 6)),
    ("-.x|", (1, 2, 3)),
    ("-_a|", (1, 1, 3)),
    ("-1|", (1, 1, 2)),
    # A combining mark is no letter or digit: no peel. A superscript two is
    # one (str.isalnum), so the sign before it is peeled.
    ("-́x|", (0, 0, 3)),
    ("-²|", (1, 1, 2)),
    ("-|foo", (1, 1, 4)),
    # The caret inside the peeled front: the whole token is the word.
    ("-|-x", (0, 0, 3)),
    ("*|*x", (0, 0, 3)),
    # token_parse's prefixes, and the reader's openings and quote marks.
    (":kw|", (1, 1, 3)),
    ("::x|", (2, 2, 3)),
    ("^x|", (1, 1, 2)),
    ("*x|", (1, 1, 2)),
    ("**x|", (2, 2, 3)),
    ("^*x|", (2, 2, 3)),
    ("'x|", (1, 1, 2)),
    ("`x|", (1, 1, 2)),
    ("~x|", (1, 1, 2)),
    ("~@x|", (2, 2, 3)),
    ("^'x|", (2, 2, 3)),
    ("'|x", (1, 1, 2)),
    ("(|", (1, 1, 1)),
    ("~|@x", (1, 1, 1)),
    # ~ with nothing after it is the invert symbol.
    ("~|", (0, 0, 1)),
    ("{x|", (1, 1, 2)),
    # A { ends a symbol: in an f-string the word is the field's own name,
    # never glued to the text or format spec before the field's brace.
    ("a{b|", (2, 2, 3)),
    ("a|{b", (0, 0, 1)),
    ('f"{1:{wi|}}"', (6, 6, 8)),
    ('f"{x:>{wi|}}"', (7, 7, 9)),
    ('f"ab{wi|}"', (5, 5, 7)),
    ('f"{a}b{c|}"', (7, 7, 8)),
    # A signed complex before a { is a number, never a symbol whose sign is
    # peeled.
    ("-1+2j|{x", (0, 0, 5)),
    ("[x|", (1, 1, 2)),
    (",x|", (1, 1, 2)),
    ("#x|", (0, 0, 2)),
    # The reader lexes the line, not the run: an opening or a string opening
    # may pass the run's end, and the caret in it is at no word.
    ("*|[1 2]", (1, 1, 1)),
    ('a|"', (1, 1, 1)),
    ('f|"x"', (1, 1, 1)),
    # A string opening inside the run: the caret after it is in its body.
    ("x'''|", None),
    # \w is Unicode: 한" is a string opening, so the caret before its quote is
    # at no word.
    ('한|"', (1, 1, 1)),
    # A signed complex is one number, never peeled; after a : prefix the rest
    # is re-typed as a number.
    ("-1+2j|", (0, 0, 5)),
    (":-1+2j|", (1, 1, 6)),
    # A non-breaking space and a file separator are whitespace to the reader.
    ("a b|", (2, 2, 3)),
    ("a\u001cb|", (2, 2, 3)),
    # The dotted word: the replaced part follows the last dot; the end stops
    # at the next one.
    ("heapq.nl|", (0, 6, 8)),
    ("heapq.|", (0, 6, 6)),
    ("x.|", (0, 2, 2)),
    ("a.b|.c", (0, 2, 3)),
    ("a..b|", (0, 3, 4)),
    (".method|", (0, 1, 7)),
    ("(.c|", (1, 2, 3)),
    ("x .c|", (2, 3, 4)),
    # Unicode identifier characters, astral ones included (code points).
    ("한국|", (0, 0, 2)),
    ("x한|", (0, 0, 2)),
    ("𝑥y|", (0, 0, 2)),
    # In a field the word is the reader's token there: it stops at the
    # field's delimiters, so the replaced range never eats a conversion or a
    # format spec, and a debug = is no part of it.
    ('f"{x|!r}"', (3, 3, 4)),
    ('f"{x|:>10}"', (3, 3, 4)),
    ('t"{x|!r}"', (3, 3, 4)),
    ('f"""{x|:>4}"""', (5, 5, 6)),
    ('f"{x|=}"', (3, 3, 4)),
    # Past the expression: a conversion and a spec are text; after a debug
    # = the word is empty.
    ('f"{x!|r}"', None),
    ('f"{x:>1|0}"', None),
    ('f"{x=|}"', (5, 5, 5)),
    ('f"{x =|}"', (6, 6, 6)),
    # != and == are operators, never a delimiter or a debug =.
    ('f"{x|!=y}"', (3, 3, 7)),
    ('f"{x|==y}"', (3, 3, 7)),
    # Inside a field's brackets, and in a top-level dict, : is no delimiter.
    ('f"{(a|:b)}"', (4, 4, 7)),
    ("{x|:y}", (1, 1, 4)),
    # A field nested in a format spec, and a field after a string in one.
    ('f"{x:{wi|}}"', (6, 6, 8)),
    ('f"{"a"}{b|}"', (8, 8, 9)),
    ('f"{\\"a\\" b|}"', (9, 9, 10)),
    # A string's body, a literal's text and a comment hold no word: the
    # caret there is no name being typed.
    ('"ab|"', None),
    ('f"{{ab|"', None),
    ("; ab|", None),
    ("a;b|", None),
    ("'''ab|'''", None),
    ('(print "ab|', None),
    ("'''a'''b|", (7, 7, 8)),
    # \N{...} is one escape in a non-raw literal; a raw one opens a field.
    ('rf"\\N{ab|}"', (6, 6, 8)),
    ('f"\\N{ab|}"', None),
    # A string closed in the run: the word starts after it at the earliest,
    # never with the string's text.
    ("(f '''a-b'''|)", (12, 12, 12)),
    ("'''x:'''|", (8, 8, 8)),
    ("f\"{'''a-b'''|}\"", (12, 12, 12)),
    # The context is read from the text's start: the later lines of a
    # multi-line f-string are its literal, a multi-line string's end closes
    # it, a comment ends at its line's end.
    ('f"""\n  Name: {name|!r}\n"""', (9, 9, 13)),
    ('f"""\n  Width: {w|:>10}\n"""', (10, 10, 11)),
    ('f"""\n  {x|=}\n"""', (3, 3, 4)),
    ('f"""\n  te|xt {x}\n"""', None),
    ('(print "doc\nend." (fo|o))', (7, 7, 10)),
    ('"a\nb|"', None),
    ("; c\nfo|", (0, 0, 2)),
    # An unclosed literal is text, but a field before its end is a field.
    ('f"{x|!r}', (3, 3, 4)),
    # The debug = is lexed off as the reader does: a\= is the symbol a then
    # a continuation, \= no symbol at all.
    ('f"{a\\|=}"', (5, 5, 5)),
    ('f"{\\|=}"', (4, 4, 4)),
    # Texts the reader refuses before the caret, so no published case: past
    # each lexer error the reader resumes just after it, in the literal open
    # there, so the text after an error keeps its reading.
    ('f"{x!z}" (fo|', (10, 10, 12)),
    ('f"}"\nf"{x|!r}"', (3, 3, 4)),
    ('f"{x!}"\n(print "ab|c")', None),
    ('x = f"{a:\n}"\n"abc|"', None),
    # A later line continuing a multi-line literal stays in it: a string's
    # text, a field's word up to its conversion.
    ('f"{x!}"\n"""doc\nprose|"""', None),
    ('f"{x!}"\nf"""{a}\n{b|!r}"""', (1, 1, 2)),
    # On the error's own line, after it, in the literal it stood in.
    ('f"{z!} {y|!r}"', (8, 8, 9)),
    # A string after a conversion's ! is refused at its first quote,
    # whatever holds the caret; the reader resumes after that quote, in the
    # f-literal, which the next " closes, and ":" is a closed string.
    ('f"{!""":"|', (9, 9, 9)),
    # A debug = followed only by blanks to the line's end: the field's
    # delimiter comes on a later line.
    ('f"""{x|=\n}"""', (5, 5, 6)),
    ('f"""{x|= \n}"""', (5, 5, 6)),
    ('f"{x|=\\\n}"', (3, 3, 4)),
]


def span_at(marked):
    """The span at the caret of ``marked``, a text with ``|`` at the caret:
    its line's columns, the text before its line passed as before."""
    before, line, caret = case_input(marked)
    return completion_word_span(line, caret, before)


@pytest.mark.parametrize("marked, expected", SPANS, ids=[m for m, _ in SPANS])
def test_completion_word_span(marked, expected):
    # Defects: a word cut at a character no Python name holds (->> to an
    # empty word, so a macro name is never completed); an augmented
    # assignment cut down to =; a peeled prefix or sign kept in the word (-fo,
    # which matches no name); a lone - dropped (-> accepted from (- gives
    # (-->); the dotted word's head left in the replaced word; a field's word
    # run past its delimiter (x!r, x:>10, x=), so a completion there eats the
    # conversion, the spec or the debug =; a word found in a string's text or
    # a comment; after a lexer error, the caret's line read alone, so a later
    # line of a multi-line literal (a docstring's prose) reads as code.
    assert span_at(marked) == expected


@pytest.mark.parametrize("text", ['f"{!""":"', "f\"{x!r'''}\""])
def test_a_string_after_a_conversion_bang_has_no_caret_context(text):
    # Defect: a string lexed after ! that reaches the caret reports its text
    # context, though the reader refuses the conversion whatever the string
    # holds, so a client would read the line as text the reader never lexes.
    assert caret_context(text, len(text)) is None


def test_the_symbol_after_a_lexer_error_is_read_in_the_resumed_context():
    # Defect: caret_symbol_p (the LSP's qualified-base code check) answers
    # False wherever the text before the caret holds a lexer error, so a
    # mod.| on a line after one gets no qualified macros; or reads the line
    # alone, so a docstring's prose after the error passes for code.
    assert caret_symbol_p("mod.", 4, 'f"{x!}"\n')
    assert not caret_symbol_p("mod.", 4, 'f"{x!}"\n"""doc\n')


def test_resuming_past_an_error_before_the_resume_point_ends(time_bound):
    # Defect: an escaped literal refused at its own opening (a bare " in
    # f\"a"), which lies before the point the reader resumed from, is
    # resumed in and refused again forever, so the completion never answers.
    line = 'f"{f\\"a"} fo'
    assert completion_word_span(line, len(line)) == (10, 10, 12)


def reader_leaf(line):
    """The leaf node the reader makes of the one token in "(" + token +
    ")"."""
    [form] = parse(line + ")")
    [node] = form.list
    while not isinstance(node, (Symbol, Constant)):
        if isinstance(node, Paren):
            node = node.list[1]  # a peeled sign: (- operand)
        else:
            node = node.value  # a prefix wrapper or quote mark
    return node


PRELUDE_MACROS = sorted(getattr(lispy.macros, "__macro_namespace"))
TOKENS = [
    *PRELUDE_MACROS,
    "swap!",
    "ok?",
    "a~b",
    "a'b",
    "except*",
    "!=",
    "**=",
    "^**=",
    "-fo",
    "--x",
    "-_a",
    "-1",
    "-́x",
    "-²",
    ":kw",
    "^x",
    "**x",
    "^*x",
    "'x",
    "`x",
    "~x",
    "~@x",
    "-",
    "~",
]


@pytest.mark.parametrize("token", TOKENS)
def test_the_word_is_the_readers_leaf(token):
    # Defect: the word rule and the reader disagree (a character set or a
    # peel of the word rule's own), so the word typed is not the name, macro
    # or operator the code holds: ->>, ++ and swap! get no word at all.
    line = "(" + token
    leaf = reader_leaf(line)
    start, _, end = completion_word_span(line, len(line))
    assert (start, end) == (leaf.col_offset, leaf.end_col_offset)


# Lines the lexer reads: each template's {} filled with every string over the
# alphabet up to the length, at the top level, in a field, in a spec, and
# right after a string that closed in the run.
READER_TEMPLATES = [
    "{}",
    'f"{{{}}}"',
    'rf"\\N{{{}}}"',
    't"a{{x:{}}}"',
    "'''a:'''{}",
]
READER_ALPHABET = "x=:!r(\"; {}'"


def test_every_word_is_a_token_the_reader_lexes():
    # Defect: the word scan and the lexer part in a string or field (a word
    # run past a field's delimiter, a word in a literal's or spec's text, a
    # field's token missed), so a completion replaces text the reader never
    # read as that name, or a string closed in the run read again as code
    # ('''a-b'''). Only lines the lexer reads whole: one it stops on (a
    # string never closed, an error token or a string's error) has a reading
    # of its own (caret_context).
    for template in READER_TEMPLATES:
        for size in range(5):
            for chars in itertools.product(READER_ALPHABET, repeat=size):
                line = template.format("".join(chars))
                tokens = list(tokenize(line))
                if any(
                    tktype == "error" or isinstance(extra, Exception)
                    for _, tktype, _, extra in tokens
                ):
                    continue
                words = [
                    (place["col_offset"], place["end_col_offset"], text, tktype)
                    for text, tktype, place, _ in tokens
                    if tktype in ("symbol", "number")
                ]
                for start, end, text, tktype in words:
                    leaf = token_prefixes(text, tktype)[1]
                    span = completion_word_span(line, end)
                    assert span is not None, (line, end)
                    assert (span[0], span[2]) == (end - len(leaf), end), (line, end)
                for caret in range(len(line) + 1):
                    span = completion_word_span(line, caret)
                    [start, _, end] = span or (caret, caret, caret)
                    assert start == end or any(
                        tk_start <= start and end <= tk_end and caret <= tk_end
                        for tk_start, tk_end, _, _ in words
                    ), (line, caret)


def test_the_spec_command_prints_the_canonical_spans():
    # Defects: the module does not run as a command, prints something that is
    # no JSON, or publishes spans other than completion_word_span's.
    result = subprocess.run(
        [sys.executable, "-m", "lispy.word_boundary"],
        capture_output=True,
        check=True,
    )
    spec = json.loads(result.stdout.decode("utf-8"))
    assert spec["version"] == 7 and spec["columns"] == "code points"
    assert len(spec["cases"]) == len(CASE_INPUTS)
    for case in spec["cases"]:
        expected = completion_word_span(case["text"], case["caret"], case["before"])
        got = (case["start"], case["replaceStart"], case["end"])
        assert got == (expected or (None, None, None))


def test_the_specs_stop_characters_end_the_readers_symbol():
    # Defect: the published stop set is not what ends a symbol in the reader
    # (\s read as ASCII whitespace, a stop left out), so a client built from
    # it joins words the reader splits, or splits words it keeps whole.
    result = subprocess.run(
        [sys.executable, "-m", "lispy.word_boundary"],
        capture_output=True,
        check=True,
    )
    stops = set(json.loads(result.stdout.decode("utf-8"))["rule"]["stopChars"])
    probes = {chr(c) for c in range(32, 127)} | {
        chr(c) for c in range(sys.maxunicode + 1) if chr(c).isspace()
    }
    for ch in sorted(probes - {"\n", "\r"}):
        tokens = [(tk, tktype) for tk, tktype, *_ in tokenize(f"a{ch}b")]
        whole = tokens == [(f"a{ch}b", "symbol")]
        assert whole == (ch not in stops), repr(ch)


def one_of(entries):
    """A regex alternation of literal entries, longest first."""
    return "|".join(re.escape(e) for e in sorted(entries, key=len, reverse=True))


def chars(entries):
    """A regex class of the code points ``entries``."""
    return "[" + "".join(re.escape(c) for c in entries) + "]"


def _recover_past_errors(errors):
    """Read the word on a line after ``errors`` lexer errors, every caret
    cache emptied first, so the recovery runs in full."""
    parser.caret_context.cache_clear()
    parser._recovered_context.cache_clear()
    parser._caret_lexer.cache_clear()
    before = 'f"{x!}" (a b)\n' * errors
    assert completion_word_span("(fo", 3, before) == (1, 1, 3)


def test_resuming_past_more_errors_finds_no_more_line_starts(monkeypatch):
    # Defect: each resumed pass finds the line starts of the whole text again
    # (a fresh locator), so a buffer with k lexer errors before the caret
    # costs k passes over all of it per keystroke. The count is compared
    # across two error counts, so other one-off reads of the line starts do
    # not matter.
    calls = []
    line_starts = parser._line_starts

    def counted(src):
        calls.append(len(src))
        return line_starts(src)

    monkeypatch.setattr(parser, "_line_starts", counted)
    counts = []
    for errors in (20, 40):
        calls.clear()
        _recover_past_errors(errors)
        counts.append(len(calls))
    assert counts[0] == counts[1], counts


def test_resuming_past_many_errors_copies_the_tokens_once(monkeypatch):
    # Defect: each resumed pass copies the whole token list so far, so k
    # lexer errors before the caret copy k lists as long as the buffer's
    # tokens. Only the cached first pass's list is copied (it must stay
    # unchanged for caret_tokens); every later pass takes over the list of
    # the pass it resumes, so all resumed passes share one list.
    lists = []
    run = parser._Lexer.run

    def recorded(lexer):
        lists.append(lexer.tokens)
        return run(lexer)

    monkeypatch.setattr(parser._Lexer, "run", recorded)
    _recover_past_errors(20)
    [first, *resumed] = lists
    assert len(resumed) >= 20
    assert len({id(tokens) for tokens in resumed}) == 1
    assert resumed[0] is not first


# spec_word_span's answer where the reader refuses the text before the caret.
REFUSED = "refused"


def spec_word_span(rule):
    """A function (line, caret, before) -> (start, replaceStart, end), or
    None for no word (REFUSED where the reader refuses the text), as an
    editor client reproduces it from the published rule alone: no parser
    internal is read here. The rule's regexes are compiled once."""
    stop = chars(rule["stopChars"])
    wrapper = f"(?:{one_of(rule['wrappers'])})"
    opening = f"(?:{one_of(rule['openings'])})"
    meta = f"(?:{one_of(rule['metaChars'])})"
    quotes = f"(?:{one_of(rule['danglingQuotes'])})"
    delimiter = rule["fieldDelimiter"]
    continuation = rule["fieldContinuation"]
    escaped = one_of(rule["escapedStringOpenings"])
    openings = one_of(rule["stringOpenings"])
    line_break = chars(rule["lineBreakChars"])
    backslash = rule["backslash"]
    [named_open, named_close] = rule["namedEscape"]

    def token_regex(mode):
        """The token patterns of code, field or bracket mode."""
        in_field = mode != "code"
        end = f"$|{delimiter}" if mode == "field" else "$"
        end = f"{continuation}|{end}" if in_field else end
        form_end = f"(?:{chars(rule['formEndChars'])}|{end})"
        number_end = f"(?:{chars(rule['numberEndChars'])}|{end})"
        comment = f";(?:(?!{line_break}).)*"
        if in_field:
            string = f"{rule['stringPrefix']}(?:{escaped}|{openings})"
            bad_end = f"(?={stop}|{escaped}|{end})"
            guard = f"(?!{delimiter})" if mode == "field" else ""
            char = f"{guard}(?!{stop})(?!{re.escape(backslash)})(?s:.)"
            symbol = f"(?:{char}|{rule['fieldSymbolBackslash']})+"
            if mode == "field":
                symbol += f"|{delimiter}"
                comment = f";(?:(?!{delimiter}|{line_break}|}})(?s:.))*"
            spaces = f"(?: |{continuation})+"
        else:
            string = f"{rule['stringPrefix']}(?:{openings})"
            bad_end = f"(?={stop}|{end})"
            [symbol, spaces] = [f"(?:(?!{stop})(?s:.))+", " +"]
        return re.compile(
            "|".join(
                f"(?P<{kind}>{pattern})"
                for kind, pattern in [
                    ("opening", wrapper + opening),
                    ("closing", chars(rule["closings"])),
                    ("string", string),
                    ("dangling", f"{wrapper}{quotes}(?={form_end})"),
                    ("meta", f"{wrapper}{meta}(?!{form_end})"),
                    ("bad", f"\\^\\*+(?:{opening}|{meta}|{bad_end})"),
                    ("comment", comment),
                    ("number", f"(?:{rule['numberToken']})(?={number_end})"),
                    ("comma", ","),
                    ("symbol", symbol),
                    ("spaces", spaces),
                ]
            )
        )

    token_res = {mode: token_regex(mode) for mode in ("code", "field", "bracket")}
    delimiter_re = re.compile(delimiter)
    prefix_re = re.compile(rule["stringPrefix"])
    stop_re = re.compile(stop)
    number_re = re.compile(rule["numberToken"])
    signed_complex_re = re.compile(rule["signedComplex"])
    signs = "".join(rule["signs"])
    braces = rule["fieldBraces"]
    whitespace = set(rule["whitespace"])
    continuation_re = re.compile(rule["fieldContinuation"])

    def body(text, i, quote, raw, kind, doubled):
        """(stop, at): the body step from i."""
        escaped_quote = quote in rule["escapedStringOpenings"]
        stops = rule["plainBodyStops" if kind == "string" else "interpolatedBodyStops"]
        n = len(text)
        while True:
            j = next((k for k in range(i, n) if text[k] in stops), None)
            if j is None:
                return "end", n
            c = text[j]
            if c == backslash:
                if escaped_quote and text.startswith(quote, j):
                    return "quote", j
                if j + 1 >= n:
                    i = j + 1
                elif escaped_quote and text.startswith(rule["escapedQuote"], j + 1):
                    i = j + 1 + len(rule["escapedQuote"])
                elif not raw and text.startswith(named_open, j + 1):
                    close = text.find(named_close, j + 1 + len(named_open))
                    i = n if close < 0 else close + 1
                elif kind != "string" and text[j + 1] in braces:
                    i = j + 1
                else:
                    i = j + 2
            elif c in rule["quoteChars"]:
                if escaped_quote and c == rule["bareQuote"]:
                    return "refused", j
                if text.startswith(quote, j):
                    return "quote", j
                i = j + 1
            elif text.startswith(c, j + 1) and c in doubled:
                i = j + 2
            else:
                return ("field" if c == braces[0] else "close"), j

    def string_kind(prefix):
        lowered = prefix.lower()
        kinds = [kind for letter, kind in rule["kindLetters"] if letter in lowered]
        return rule["rawLetter"] in lowered, (kinds + ["string"])[0]

    def context(text, stop):
        """(mode at stop, resume), the context and resume steps, or None
        where the reader refuses the text before stop (no agreement asked)."""
        # Literals: ["literal", quote, raw, kind]; fields: ["field", dict].
        stack, pos, resume = [], 0, 0
        while True:
            top = stack[-1] if stack else None
            if top is not None and (top[0] == "literal" or top[1]["spec"]):
                [literal, spec] = (
                    [top, None] if top[0] == "literal" else [top[1]["literal"], top[1]]
                )
                if pos >= stop:
                    return "text", resume
                doubled = (
                    braces if spec is None else braces[0] if spec["nested"] else ""
                )
                stop_kind, at = body(
                    text, pos, literal[1], literal[2], literal[3], doubled
                )
                if stop_kind == "quote":
                    if at + len(literal[1]) > stop:
                        return "text", resume
                    if spec is not None:
                        return None
                    pos = resume = at + len(literal[1])
                    stack.pop()
                elif stop_kind == "refused" and at < stop:
                    return None
                elif stop_kind in ("end", "refused") or at >= stop:
                    return "text", resume
                elif stop_kind == "field":
                    if spec is not None:
                        spec["nested"] = True
                    field = {
                        "literal": literal,
                        "brackets": 0,
                        "spec": False,
                        "nested": False,
                    }
                    stack.append(["field", field])
                    pos = at + 1
                elif spec is not None:
                    stack.pop()
                    pos = at + 1
                else:
                    return None
                continue
            field = None if top is None else top[1]
            mode = (
                "code" if top is None else "bracket" if field["brackets"] else "field"
            )
            m = token_res[mode].search(text, pos)
            if m is None or m.start() >= stop:
                return mode, resume
            start, end, kind = m.start(), m.end(), m.lastgroup
            if mode == "field":
                if text[start] == braces[1]:
                    stack.pop()
                    pos = start + 1
                    continue
                if delimiter_re.match(text, start):
                    if text[start] == ":":
                        field["spec"] = True
                        pos = start + 1
                        continue
                    ends = [
                        k
                        for k in range(start + 1, stop)
                        if text[k] in rule["conversionEnds"]
                    ]
                    if not ends:
                        return "text", resume
                    name_end = start + 1
                    while text[name_end].isalnum() or text[name_end] == "_":
                        name_end += 1
                    if text[start + 1 : name_end] not in rule["conversionNames"]:
                        return None
                    while name_end < ends[0] and (
                        text[name_end] in whitespace
                        or continuation_re.match(text, name_end)
                    ):
                        name_end += 1
                    if name_end < ends[0]:
                        return None
                    pos = ends[0]
                    continue
            if kind == "comment" and end >= stop:
                return "text", resume
            if end > stop:
                return mode, resume
            pos = end
            if kind == "string":
                prefix = prefix_re.match(text, start).group()
                quote = text[start + len(prefix) : end]
                raw, string = string_kind(prefix)
                if string != "string":
                    stack.append(["literal", quote, raw, string])
                    continue
                stop_kind, at = body(text, end, quote, raw, string, "")
                if stop_kind == "refused" and at < stop:
                    return None
                if stop_kind != "quote" or at + len(quote) > stop:
                    return "text", resume
                pos = resume = at + len(quote)
            elif field is not None:
                if kind == "opening" or (
                    kind == "bad" and text[end - 1] in rule["openings"]
                ):
                    field["brackets"] += 1
                elif kind == "closing" and field["brackets"]:
                    field["brackets"] -= 1

    def debug_equals(line, end):
        """The debug = step at a symbol ending at end."""
        if end < 2 or line[end - 1] != "=" or line[end - 2] in rule["debugBlockers"]:
            return False
        at = end
        while at < len(line) and (
            line[at] in whitespace or (line[at] == backslash and at + 1 == len(line))
        ):
            at += 1
        return at == len(line) or (
            line[at] == braces[1] or bool(delimiter_re.match(line, at))
        )

    def word_token(line, run_start, caret, mode, endpos):
        """The word token step over the line up to endpos, or None."""
        run_end = run_start
        while run_end < endpos and not stop_re.match(line[run_end]):
            run_end += 1
        pos = run_start
        while pos < run_end:
            m = token_res[mode].match(line, pos, endpos)
            start, end, kind = pos, m.end(), m.lastgroup
            if mode == "field" and delimiter_re.match(line, start):
                return None
            if end > caret or (end == caret and kind in ("symbol", "number")):
                return start, end, kind
            pos = end
        return None

    def word_span(line, caret, before=""):
        found = context(before + line, len(before) + caret)
        if found is None:
            return REFUSED  # the rule has no answer where the reader refuses
        [mode, resume] = [found[0], max(found[1] - len(before), 0)]
        if mode == "text":
            return None
        run_start = caret
        while run_start > 0 and not stop_re.match(line[run_start - 1]):
            run_start -= 1
        run_start = max(run_start, resume)
        token = word_token(line, run_start, caret, mode, len(line))
        if token is None or token[2] not in ("symbol", "number"):
            return caret, caret, caret
        [start, end, kind] = token
        if mode == "field" and kind == "symbol" and debug_equals(line, end):
            token = word_token(line, start, caret, mode, end - 1)
            if token is None or token[2] not in ("symbol", "number"):
                return caret, caret, caret
            [start, end, kind] = token
            if end < caret:
                return caret, caret, caret
        leaf = line[start:end]
        while leaf not in rule["neverPeeled"] and len(leaf) >= 2:
            if kind == "number" and (
                leaf[0] not in signs or signed_complex_re.fullmatch(leaf)
            ):
                break
            prefix = next((p for p in rule["tokenPrefixes"] if leaf.startswith(p)), "")
            rest = leaf.lstrip(signs)
            if prefix:
                leaf = leaf[len(prefix) :]
                kind = "number" if number_re.fullmatch(leaf) else kind
            elif (
                rest != leaf
                and rest
                and (rest[0].isalnum() or rest[0] in rule["signPeelFollowers"])
            ):
                leaf = rest
            else:
                break
        if caret >= end - len(leaf):
            start = end - len(leaf)
        dot = line.find(".", caret, end)
        return start, start + line[start:caret].rfind(".") + 1, end if dot < 0 else dot

    return word_span


@pytest.fixture(scope="module")
def published():
    """The spec as ``python -m lispy.word_boundary`` prints it."""
    result = subprocess.run(
        [sys.executable, "-m", "lispy.word_boundary"],
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout.decode("utf-8"))


def test_the_published_rule_alone_reproduces_every_case(published):
    # Defect: the published rule is not the reader's (it lexes the run alone,
    # leaves out a string opening, a quote mark or the signed complex, or
    # misorders the peel), so a client implementing it to the letter fails
    # the cases it is pinned against.
    word_span = spec_word_span(published["rule"])
    for case in published["cases"]:
        # No case is a text the reader refuses, where the rule has no answer.
        text = case["before"] + case["text"]
        assert caret_context(text, len(case["before"]) + case["caret"]) is not None
        expected = (case["start"], case["replaceStart"], case["end"])
        got = word_span(case["text"], case["caret"], case["before"])
        assert (got or (None, None, None)) == expected, (case["text"], case["caret"])


# The entry's prefix followed by every text up to the length over its
# alphabet, at every caret; with refusals set, the rule must also refuse
# exactly the texts the reader refuses.
DIFFERENTIAL = [
    ("-+*^:'`~@([{\" .1jx;#", 4, "", False),
    ("-+*:1jx.e{", 4, "", False),
    # Strings and fields: delimiters, debug =, conversions, specs.
    ("f\"{}:!x='", 5, "", False),
    # Raw and \N{ escapes, escaped quotes, brackets and comments in a field.
    ('rf\\N{}"(;x ', 4, "", False),
    # Strings, fields and comments across lines.
    ('f"{}x;\n', 5, "", False),
    # A debug = and the line break or continuation after it.
    ('f"{}x=\n\\', 5, "", False),
    # A conversion past its !: valid and invalid names, the blanks (Unicode
    # whitespace, a line break, a continuation) up to its end, anything else.
    # No case may be a refused text, so the conversion check (conversionNames,
    # the blanks) is pinned by the rule refusing where the reader does.
    ("rsaz \x85:}\n\\(", 4, 'f"{x!', True),
]


@pytest.mark.parametrize("alphabet, length, prefix, refusals", DIFFERENTIAL)
def test_the_published_rule_agrees_with_the_reader_everywhere(
    published, alphabet, length, prefix, refusals
):
    # Defect: the rule and completion_word_span part on a line no case
    # names (an ordering of the token patterns, a lookahead, the peel's
    # re-typing, the blanks a conversion takes), so a client pinned on the
    # cases still differs from the reader as the user types.
    word_span = spec_word_span(published["rule"])
    ends = published["rule"]["conversionEnds"]
    for size in range(length + 1):
        for chars in itertools.product(alphabet, repeat=size):
            text = prefix + "".join(chars)
            # Past the conversion's end the reader refuses texts the rule
            # leaves to it (a line break in a single-quoted spec).
            end = next(
                (k for k in range(len(prefix), len(text)) if text[k] in ends),
                len(text),
            )
            for at in range(len(text) + 1):
                before, line, caret = case_input(text[:at] + "|" + text[at:])
                got = word_span(line, caret, before)
                refused = caret_context(before + line, len(before) + caret) is None
                if refusals and at <= end + 1:
                    assert (got == REFUSED) == refused, (text, at)
                if refused:
                    continue  # the reader refuses the text: no span asked
                assert got == completion_word_span(line, caret, before), (text, at)


def number_token_corpus():
    """Number token texts the reader lexes: signs, every integer base,
    exponents, underscores and junk the reader still takes as a number."""
    reals = ["1", "0", "00", "1_000", "1__0", "0x1e", "0X_f", "0o17", "0o8"]
    reals += ["0b101", "0b12", "1.", ".5", "1.5", "1e5", "1e+5", "1E-5"]
    reals += ["1.e5", "1_0.0_1e1_0", "1abc", "0x1j", "1j", "1e", "١"]
    imags = ["2j", "2J", "2", ".5j", "5.j", "1e5j", "1e+5j", "2_0j", "0x2j"]
    imags += ["02j", "2jj", "2e", "1_j", "١j"]
    texts = set()
    for sign, real in itertools.product(["", "-", "+", "--", "-+"], reals):
        texts |= {sign + real, sign + real + "j"}
        texts |= {sign + real + op + imag for op in "+-" for imag in imags}
    return [
        text
        for text in sorted(texts)
        if [(tk, tktype) for tk, tktype, *_ in tokenize(text)] == [(text, "number")]
    ]


def test_the_signed_complex_pattern_is_number_parts_reading():
    # Defect: the published pattern and number_parts (Python's tokenizer)
    # disagree on which signed number token is one Constant (1e+5j is one
    # imaginary, 0x1e+2 a pair), so the reader peels a sign the compile kept
    # whole, or a client does.
    corpus = number_token_corpus()
    assert len(corpus) > 1000
    for text in corpus:
        parts = number_parts(text)
        assert _signed_complex_p(text) == (
            parts is not None and parts[2] is not None
        ), text


@pytest.mark.parametrize("token", ["-01+2j", "-0x1e+02", "-01-1e5j"])
def test_a_leading_zero_decimal_is_one_number_on_every_interpreter(token):
    # Defect: 3.11's pure-Python tokenizer splits 01, so there alone the sign
    # of -01+2j was peeled and its malformed literal placed a column later.
    # On the old code this fails only on 3.11 (the nox matrix runs it there).
    assert token_prefixes(token, "number")[0] == []
