"""The completion word boundary as a conformance spec for editor clients.

``python -m lispy.word_boundary`` prints, as JSON, the rule of
``completion_word_span`` (``core/parser.py``, its one home) and a set of
cases whose expected spans that function computes. lpy-mode and vsc-lispy
compute the boundary natively on every keystroke, so they pin their copies
against these cases. Columns are code points (Python str indices); a client
counting UTF-16 units converts (the astral case is there for it).

Every character set in the rule is read off the reader's own patterns and
tables, never written here. No pygls or lsprotocol import: this runs
wherever lispython does.
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

VERSION = 2

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
        "steps": [
            "run: the stretch of the line around the caret holding no stopChars",
            "lex the run from its start, trying at each token start in order: "
            "a wrapper + an opening (prefix token); \\w* + ''' (a string "
            "opening: no word); a wrapper + one of ' ` ~@ before a "
            "formEndChars character or the line's end (dangling: no word); a "
            "wrapper + a metaChar before anything else (prefix token); ^ + "
            "one or more * before an opening, a metaChar, a stopChars "
            "character or the line's end (prefix token); else the rest of the "
            "run is one symbol token",
            "the word token is the symbol token ending at or holding the "
            "caret; with none (the caret in or right after a prefix token "
            "with no symbol after it) the word is empty at the caret",
            "peel the token's front, repeatedly: a tokenPrefixes entry, else a "
            "run of signs followed by a str.isalnum character or a "
            "signPeelFollowers one; never a neverPeeled token, a token under "
            "2 characters, an unsigned number or a signed complex number",
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
        "rule": rule(),
        "cases": [case(marked) for marked in CASE_INPUTS],
    }


def main():
    # UTF-8 whatever the console's encoding, so every copy diffs clean.
    text = json.dumps(spec(), ensure_ascii=False, indent=2) + "\n"
    sys.stdout.buffer.write(text.encode("utf-8"))


if __name__ == "__main__":
    main()
