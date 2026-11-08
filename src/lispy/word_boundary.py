"""The completion word boundary as a conformance spec for editor clients.

``python -m lispy.word_boundary`` prints, as JSON, the rule of
``completion_word_span`` (``core/parser.py``, its one home) and a set of
cases whose expected spans that function computes. lpy-mode and vsc-lispy
compute the boundary natively on every keystroke, so they pin their copies
against these cases. Columns are code points (Python str indices); a client
counting UTF-16 units converts (the astral case is there for it).

Every character set and pattern in the rule is read off the reader's own
patterns and tables, never written here; a pattern is Python ``re`` syntax
over str (``\\d``, ``\\w`` and ``\\s`` are Unicode). No pygls or lsprotocol
import: this runs wherever lispython does.
"""

import itertools
import json
import re
import string
import sys
import unicodedata

from lispy.core import parser
from lispy.core.parser import (
    SIGN_PEEL_FOLLOWERS,
    SIGNS,
    TOKEN_PREFIXES,
    completion_word_span,
)
from lispy.core.utils import augassignop_dict

VERSION = 3

# The case inputs, each a line with ``|`` at the caret. Only the inputs are
# written here: the expected spans are completion_word_span's.
CASE_INPUTS = (
    "foo|",
    "fo|o",
    "|foo",
    "|",
    "|x",
    "x1|",
    "ok?|",
    "swap!|",
    "(swap!|",
    "->f|",
    "(->>|",
    "(++|",
    "(--|",
    "as->|",
    "a~b|",
    "a@b|",
    "a<b|",
    "a(b|",
    "a'b|",
    "except*|",
    "!=|",
    "-fo|",
    "--x|",
    "(- -fo|",
    "-.x|",
    "-_a|",
    "-́x|",
    "-²|",
    "-1|",
    "-|foo",
    "-|-x",
    "*|*x",
    "-|",
    "--|",
    "(-|",
    "(-|)",
    "-->x|",
    "-?|",
    ":kw|",
    "::x|",
    "^x|",
    "*x|",
    "**x|",
    "^*x|",
    "**=|",
    "^**=|",
    "+=|",
    "-=|",
    "'x|",
    "`x|",
    "~x|",
    "~@x|",
    "^'x|",
    "'|x",
    "~|",
    "~|@x",
    "(|",
    "{x|",
    "[x|",
    ",x|",
    "#x|",
    "x'''|",
    "*|[1 2]",
    'a|"',
    'f|"x"',
    "-1+2j|",
    ":-1+2j|",
    '한|"',
    "heapq.nl|",
    "heapq.|",
    "a.b|.c",
    "a..b|",
    "x.|",
    ".method|",
    "(.c|",
    "x .c|",
    "1.5|",
    "(foo bar|)",
    '"ab|"',
    "a b|",
    "a\u001cb|",
    "한국|",
    "x한|",
    "𝑥y|",
)


def case(marked):
    """The case of ``marked`` (a line with ``|`` at the caret)."""
    caret = marked.index("|")
    text = marked[:caret] + marked[caret + 1 :]
    start, replace_start, end = completion_word_span(text, caret)
    return {
        "text": text,
        "caret": caret,
        "start": start,
        "replaceStart": replace_start,
        "end": end,
    }


def _matching(pattern, length):
    """The ASCII punctuation strings up to ``length`` characters that the
    reader's regex ``pattern`` matches whole: a pattern's alternatives as
    data."""
    regex = re.compile(pattern)
    found = [""] if regex.fullmatch("") else []
    for size in range(1, length + 1):
        for chars in itertools.product(string.punctuation, repeat=size):
            if regex.fullmatch("".join(chars)):
                found.append("".join(chars))
    return found


def _class_members(char_class):
    """Every code point the reader's character class ``char_class`` holds."""
    regex = re.compile(f"[{char_class}]")
    return [chr(c) for c in range(sys.maxunicode + 1) if regex.match(chr(c))]


def rule():
    """The rule as data an implementer reproduces, read off the reader."""
    return {
        "unicode": unicodedata.unidata_version,
        # Python's re \s for str: the code points str.isspace holds.
        "stopChars": _class_members(parser._SYMBOL_STOP),
        "formEndChars": _class_members(parser._FORM_END),
        "wrappers": _matching(parser._WRAPPER, 2),
        "openings": _matching(parser._OPENING, 1),
        "metaChars": _matching(parser._META_CHARS, 2),
        "tokenPrefixes": list(TOKEN_PREFIXES),
        "signs": list(SIGNS),
        "signPeelFollowers": list(SIGN_PEEL_FOLLOWERS),
        "neverPeeled": sorted(augassignop_dict),
        "stringPrefix": parser._STRING_PREFIX,
        "stringOpenings": sorted(parser._STRING_TYPES, key=len, reverse=True),
        "danglingQuotes": _matching(parser._DANGLING_QUOTES, 2),
        "numberToken": parser._NUMBER_TOKEN,
        "signedComplex": parser._SIGNED_COMPLEX,
        "steps": [
            "run: the stretch of the line around the caret holding no stopChars",
            "lex the line, not the run alone, from the run's start, token "
            "after token, until a token reaches or passes the run's end; at "
            "each token start try, in this order: an opening token (a wrappers "
            "entry + an openings entry); a string opening (stringPrefix + a "
            "stringOpenings entry); a dangling quote (a wrappers entry + a "
            "danglingQuotes entry, followed by a formEndChars character or the "
            "line's end); a meta-indicator (a wrappers entry + a metaChars "
            "entry, followed by anything but a formEndChars character or the "
            "line's end); a bad prefix (^ + one or more *, then an openings "
            "entry, a metaChars entry, or before a stopChars character or the "
            "line's end); a number (numberToken, followed by a formEndChars "
            "character or the line's end); else a symbol (the rest of the "
            "run). Entries are tried longest first. ~ is a metaChars entry and "
            "no danglingQuotes one: a ~ followed by a formEndChars character "
            "or the line's end matches none of these patterns, so it is a "
            "symbol",
            "the word token: the first token that ends after the caret, or "
            "ends at it and is a symbol or number; when there is none, or it "
            "is neither a symbol nor a number, the word is empty at the caret",
            "peel the word token's front, repeatedly: never a neverPeeled "
            "token; never a number token that does not start with a signs "
            "character or that matches signedComplex whole; never a token "
            "under 2 characters; else the first tokenPrefixes entry it starts "
            "with, after which the rest is a number when it matches "
            "numberToken whole and keeps its type otherwise; else a run of "
            "signs followed by a str.isalnum character or a signPeelFollowers "
            "one, the rest keeping its type; else nothing (the rest is the "
            "leaf)",
            "start: the leaf's start, or the token's when the caret is in the "
            "peeled front",
            "replaceStart: after the last . between start and the caret",
            "end: the first . after the caret in the token, else its end",
        ],
    }


def spec():
    return {
        "version": VERSION,
        "columns": "code points",
        "patterns": "Python re syntax over str: \\d, \\w and \\s are Unicode "
        "(\\w holds 한, \\d holds ١)",
        "rule": rule(),
        "cases": [case(marked) for marked in CASE_INPUTS],
    }


def main():
    # UTF-8 whatever the console's encoding, so every copy diffs clean.
    text = json.dumps(spec(), ensure_ascii=False, indent=2) + "\n"
    sys.stdout.buffer.write(text.encode("utf-8"))


if __name__ == "__main__":
    main()
