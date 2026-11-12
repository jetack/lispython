"""The completion word boundary as a conformance spec for editor clients.

``python -m lispy.word_boundary`` prints, as JSON, the rule of
``completion_word_span`` (``core/parser.py``, its one home) and a set of
cases whose expected spans that function computes. The rule's input is the
text before the caret's line, the caret's line and the caret's column: the
caret's context (code, an f/t-string field, or a string's or comment's text)
is read from the start of the text. lpy-mode and vsc-lispy
compute the boundary natively on every keystroke, so they pin their copies
against these cases. Columns are code points (Python str indices); a client
counting UTF-16 units converts (the astral case is there for it).

Every character set and pattern in the rule is read off the reader's own
patterns and tables, never written here; a pattern is Python ``re`` syntax
over str (``\\d``, ``\\w`` and ``\\s`` are Unicode as the running Python's
own tables define them; the Unicode version varies by CPython release, so the
spec names none, and a client approximates ``\\w`` and ``\\d`` with its own
engine's tables). No pygls or lsprotocol import: this runs wherever
lispython does.
"""

import itertools
import json
import re
import string
import sys

from lispy.core import parser
from lispy.core.parser import (
    SIGN_PEEL_FOLLOWERS,
    SIGNS,
    TOKEN_PREFIXES,
    completion_word_span,
    split_lines,
)
from lispy.core.utils import augassignop_dict

VERSION = 7

# The case inputs, each a text with ``|`` at the caret, the caret's line its
# last line holding the caret. Only the inputs are written here: the
# expected spans are completion_word_span's.
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
    "a\u001cb|",
    # A string's body and a comment hold no word.
    "; ab|",
    'f"{{ab|"',
    # In a field the word stops at the field's delimiters and before a debug
    # =; a conversion and a format spec are text.
    'f"{x|!r}"',
    'f"{x|:>10}"',
    'f"{x!|r}"',
    # A valid conversion name, blanks and a continuation after it included,
    # ends at its } (an invalid one is refused, and no case is refused text).
    'f"{x!r} (fo|',
    'f"{x!r \\\n} (fo|',
    'f"{x:>1|0}"',
    'f"{x|=}"',
    'f"{x =|}"',
    'f"{x=|}"',
    'f"{x|==y}"',
    'f"{x|!=y}"',
    't"{x|!r}"',
    'f"""{x|:>4}"""',
    # Inside a field's brackets, and in a top-level dict, : is no delimiter.
    'f"{(a|:b)}"',
    "{x|:y}",
    'f"{x:{wi|}}"',
    # \N{...} is one escape in a non-raw literal only.
    'rf"\\N{ab|}"',
    'f"\\N{ab|}"',
    # A string in a field, a string opened by a symbol in the run, an escaped
    # string in a field.
    'f"{"a"}{b|}"',
    "'''ab|'''",
    "'''a'''b|",
    'f"{\\"a\\" b|}"',
    # A string closed in the run: a word starts after it at the earliest.
    "(f '''a-b'''|)",
    "'''x:'''|",
    "f\"{'''a-b'''|}\"",
    # The context is read from the text's start: a multi-line f-string's
    # later lines are in its literal, a multi-line string's end closes it.
    'f"""\n  Name: {name|!r}\n"""',
    'f"""\n  Width: {w|:>10}\n"""',
    'f"""\n  {x|=}\n"""',
    'f"""\n  te|xt {x}\n"""',
    '(print "doc\nend." (fo|o))',
    '"a\nb|"',
    "; c\nfo|",
    # A string never closed is text to the end.
    '(print "ab|',
    'f"{x|!r}',
    # The debug = is lexed off as the reader does: a\= is a then a
    # continuation, \= no symbol at all.
    'f"{a\\|=}"',
    # A debug = followed only by blanks to the line's end.
    'f"""{x|=\n}"""',
    'f"""{x|= \n}"""',
    'f"{x|=\\\n}"',
    # A continuation's backslash before a debug =.
    'f"{\\|=}"',
    'f"{\\=|}"',
    # A field's { ends the word before it, in a format spec and after text:
    # the line is lexed as code, so the word is the field's own name.
    'f"{1:{wi|',
    'f"{x:>{wi|}}"',
    'f"ab{wi|',
    'f"{a}b{c|}"',
    # Each form-end stop char, the opening { (and two non-ASCII spaces) beside
    # a word, on both sides of it: a client word set that drops one fails.
    "a{b|",
    "a|{b",
    # A number ends at { too (numberEndChars), never left to the symbol rule,
    # which would peel the signed complex's sign.
    "1|{x",
    "-1+2j|{x",
    "1{x|",
    "a;b|",
    "a|;b",
    "a]b|",
    "a|]b",
    "a}b|",
    "a|}b",
    "a)b|",
    "a|)b",
    "a,b|",
    "a|,b",
    "a\u00a0b|",
    "a|\u00a0b",
    "a\u3000b|",
    "a|\u3000b",
    "한국|",
    "x한|",
    "𝑥y|",
)


def case_input(marked):
    """``(before, line, caret)`` of ``marked`` (a text with ``|`` at the
    caret): the text before the caret's line, that line and the caret's
    column in it."""
    at = marked.index("|")
    source = marked[:at] + marked[at + 1 :]
    before = "".join(split_lines(source[:at], keepends=True)[:-1])
    line = split_lines(source[len(before) :])[0]
    return before, line, at - len(before)


def case(marked):
    """The case of ``marked``: no word (null bounds) in a string's or
    comment's text."""
    before, line, caret = case_input(marked)
    span = completion_word_span(line, caret, before)
    [start, replace_start, end] = (None, None, None) if span is None else span
    return {
        "before": before,
        "text": line,
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


def _members(regex):
    """Every code point the reader's one-character ``regex`` matches."""
    return [chr(c) for c in range(sys.maxunicode + 1) if regex.match(chr(c))]


def _class_members(char_class):
    """Every code point the reader's character class ``char_class`` holds."""
    return _members(re.compile(f"[{char_class}]"))


def rule():
    """The rule as data an implementer reproduces, read off the reader."""
    return {
        # Python's re \s for str: the code points str.isspace holds.
        "stopChars": _class_members(parser._SYMBOL_STOP),
        "formEndChars": _class_members(parser._FORM_END),
        "numberEndChars": _class_members(parser._NUMBER_END),
        "wrappers": _matching(parser._WRAPPER, 2),
        "openings": _matching(parser._OPENING, 1),
        "closings": _matching(parser._CLOSING, 1),
        "metaChars": _matching(parser._META_CHARS, 2),
        "tokenPrefixes": list(TOKEN_PREFIXES),
        "signs": list(SIGNS),
        "signPeelFollowers": list(SIGN_PEEL_FOLLOWERS),
        "neverPeeled": sorted(augassignop_dict),
        "stringPrefix": parser._STRING_PREFIX,
        "stringOpenings": sorted(parser._STRING_TYPES, key=len, reverse=True),
        "escapedStringOpenings": list(parser._ESCAPED_STRING_TYPES),
        "rawLetter": parser._RAW_LETTER,
        "kindLetters": [list(entry) for entry in parser._KIND_LETTERS],
        "plainBodyStops": _members(parser._PLAIN_BODY_STOP),
        "interpolatedBodyStops": _members(parser._INTERPOLATED_BODY_STOP),
        "fieldBraces": list(parser._FIELD_BRACES),
        "backslash": parser._BACKSLASH,
        "namedEscape": list(parser._NAMED_ESCAPE),
        "quoteChars": list(parser._QUOTE_CHARS),
        "escapedQuote": parser._ESCAPED_QUOTE,
        "bareQuote": parser._BARE_QUOTE,
        "lineBreakChars": _class_members(parser._LINE_BREAK_CHARS),
        # What _skip_blanks skips: str.isspace.
        "whitespace": [chr(c) for c in range(sys.maxunicode + 1) if chr(c).isspace()],
        "fieldDelimiter": parser._FIELD_DELIMITER,
        "conversionNames": sorted(parser.conversion_dict),
        "conversionEnds": list(parser._CONVERSION_ENDS),
        "fieldContinuation": parser._FIELD_CONTINUATION,
        "fieldSymbolBackslash": parser._FIELD_SYMBOL_BACKSLASH,
        "debugBlockers": list(parser._DEBUG_BLOCKERS),
        "danglingQuotes": _matching(parser._DANGLING_QUOTES, 2),
        "numberToken": parser._NUMBER_TOKEN,
        "signedComplex": parser._SIGNED_COMPLEX,
        "steps": [
            "input: the text before the caret's line (its line breaks "
            "included), the caret's line and the caret's column; the scan "
            "reads the text before the line followed by the line, from its "
            "start, and stops at the caret: where it would read a character "
            "at or past the caret",
            "context: lex that text over a stack of the f/t literals and "
            "fields open, in one of four modes: code (top level), field (a "
            "field's own level), bracket (inside a field's brackets), text (a "
            "string's body, a literal's or format spec's text, a conversion, a "
            "comment); the context is the mode the caret lies in. In text "
            "there is no word: start, replaceStart and end are null (an editor "
            "offers no completion there). The reader's scan is its lexer: "
            "where that lexer refuses the text before the caret (a malformed "
            "literal, field or conversion) the reader resumes just after the "
            "error, in the innermost f/t literal open there (top level when "
            "none, resume then the error's end at the earliest), repeating on "
            "each further error until it reaches the caret (an error not ending "
            "past where its scan resumed also drops the innermost literal "
            "kept); the cases hold no "
            "such text, and a client need not agree on a text the reader "
            "refuses",
            "code, field and bracket mode: find the next token with the mode's "
            "patterns (a character no pattern matches is skipped); none, or "
            "one starting at or after the caret, stops the scan in the mode. "
            "A comment ending at or after the caret, or any other token ending "
            "after it, holds the caret: text for a comment, else the mode. In "
            "field mode a token starting with } closes the field (back to its "
            "literal's text); one starting at a fieldDelimiter match ends the "
            "expression: at : the format spec's text follows; at ! the "
            "conversion, which ends at the first conversionEnds character "
            "after the !, where field mode resumes: with none before the "
            "caret, the caret is in its text; else the conversion is the name "
            "right after the ! (characters str.isalnum holds, or _), then only "
            "blanks (whitespace characters and fieldContinuation matches) up "
            "to that character: a name not in conversionNames (an empty one "
            "included), or anything else there, is a refusal of the text, "
            "so the check decides where the reading goes on: a valid "
            "conversion resumes field mode at that character, while past a "
            "refusal the reader recovers (context step) in the field's "
            "literal, never in the field. "
            "Any other token: a string "
            "opening reads its prefix (stringPrefix) lower-cased: the first "
            "kindLetters entry whose letter it holds makes it an f/t-string, "
            "rawLetter raw; an f/t-string opening opens a literal (its quote "
            "as written) whose text is read next; a plain string reads its "
            "body (body step): text when it does not close before the caret "
            "(its closing quote ending after it, or none), else the scan goes "
            "on after its closing quote; in field or bracket mode an opening "
            "token, or a bad prefix ending in an openings entry, opens a "
            "bracket and a closing token closes one (if any is open)",
            "text of a literal or a format spec: at or past the caret it is "
            "text; read the body (body step): a closing quote ending after the "
            "caret, or any other stop at or after it, is text; the closing "
            "quote closes the literal; a { opens a field (in a format spec, "
            "marking the spec nested); a } in a format spec closes its field",
            "resume: where the last string or literal the scan closed ends "
            "(its closing quote's end)",
            "body: read from the position to the next plainBodyStops "
            "character (a plain string) or interpolatedBodyStops character (an "
            "f/t-string). A backslash pairs with the character after it (raw "
            "too), except: one ending the text stands alone; in a non-raw "
            "string a backslash followed by namedEscape's opening runs through "
            "the next namedEscape close (to the text's end when none); in an "
            "f/t-string one before a fieldBraces character stands alone, the "
            "brace read next. A quoteChars character is the closing quote "
            "where the string's quote starts, else text. In an escaped string "
            "(its quote an escapedStringOpenings entry) its quote closes it, a "
            "backslash before escapedQuote is one piece with it, and a "
            "bareQuote ends the scan (the reader refuses it). A doubled brace "
            "is text where it is in the doubled set: both braces in a "
            "literal's own text, none in a format spec before its first "
            "nested field, only { after one; else it, as a single brace, ends "
            "the read",
            "run: the stretch of the line around the caret holding no "
            "stopChars; it starts at resume instead when that lies later on "
            "the caret's line",
            "lex the line, not the run alone, from the run's start, token "
            "after token with the context's patterns, until a token reaches or "
            "passes the run's end; at each token start try, in this order: an "
            "opening token (a wrappers entry + an openings entry); a closing "
            "token (a closings entry); a string opening (stringPrefix + a "
            "stringOpenings entry); a dangling quote (a wrappers entry + a "
            "danglingQuotes entry, followed by a formEndChars character or the "
            "line's end); a meta-indicator (a wrappers entry + a metaChars "
            "entry, followed by anything but a formEndChars character or the "
            "line's end); a bad prefix (^ + one or more *, then an openings "
            "entry, a metaChars entry, or before a stopChars character or the "
            "line's end); a comment (; up to a lineBreakChars character); a "
            "number (numberToken, followed by a numberEndChars character or "
            "the line's end); a "
            "comma; else a symbol (the rest of the run); spaces "
            "(a run of spaces). Entries are tried longest first. ~ is a "
            "metaChars entry and no danglingQuotes one: a ~ followed by a "
            "formEndChars character or the line's end matches none of these "
            "patterns, so it is a symbol",
            "field and bracket patterns differ from code's: the line's end in "
            "every lookahead above is also a fieldContinuation match, and in "
            "field mode a fieldDelimiter match; a string opening also takes an "
            "escapedStringOpenings entry after stringPrefix, tried first; a "
            "bad prefix also stands before an escapedStringOpenings entry; a "
            "symbol is a run of characters, each either no stopChars character "
            "and no backslash, or a backslash matching fieldSymbolBackslash, "
            "and in field mode never one where fieldDelimiter matches, a "
            "fieldDelimiter match alone being a symbol too; in field mode a "
            "comment stops before a } or a fieldDelimiter match; spaces also "
            "take fieldContinuation matches",
            "the word token: the first token that ends after the caret, or "
            "ends at it and is a symbol or number; when there is none, or it "
            "is neither a symbol nor a number, the word is empty at the caret. "
            "In field mode a token met there that starts at a fieldDelimiter "
            "match makes the word empty",
            "debug =: in field mode a symbol word token ending in = whose "
            "character before the = is no debugBlockers entry, followed by "
            "only whitespace characters and backslashes ending the line, then "
            "a } or a fieldDelimiter match or the line's end, loses its =: "
            "lex again from the token's start with the line ending at the = "
            "(for every lookahead "
            "too) and take the word token of that lex; none, or one ending "
            "before the caret, is the empty word at the caret",
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
        "as the running Python's tables define them (\\w holds 한, \\d holds ١); "
        "a client approximates \\w and \\d with its own engine's tables",
        "rule": rule(),
        "cases": [case(marked) for marked in CASE_INPUTS],
    }


def main():
    # UTF-8 whatever the console's encoding, so every copy diffs clean.
    text = json.dumps(spec(), ensure_ascii=False, indent=2) + "\n"
    sys.stdout.buffer.write(text.encode("utf-8"))


if __name__ == "__main__":
    main()
