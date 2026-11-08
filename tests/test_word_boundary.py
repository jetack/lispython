"""The completion word boundary (completion_word_span, the reader's word) and
its published spec (python -m lispy.word_boundary), which lpy-mode and
vsc-lispy pin against."""

import json
import subprocess
import sys

import pytest

import lispy.macros
from lispy.core.nodes import Constant, Paren, Symbol
from lispy.core.parser import completion_word_span, parse, tokenize
from lispy.word_boundary import CASE_INPUTS

# (line with | at the caret, (start, replace_start, end)), checked by hand.
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
    ("[x|", (1, 1, 2)),
    (",x|", (1, 1, 2)),
    ("#x|", (0, 0, 2)),
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
]


def span_at(marked):
    caret = marked.index("|")
    return completion_word_span(marked[:caret] + marked[caret + 1 :], caret)


@pytest.mark.parametrize("marked, expected", SPANS, ids=[m for m, _ in SPANS])
def test_completion_word_span(marked, expected):
    # Defects: a word cut at a character no Python name holds (->> to an
    # empty word, so a macro name is never completed); an augmented
    # assignment cut down to =; a peeled prefix or sign kept in the word (-fo,
    # which matches no name); a lone - dropped (-> accepted from (- gives
    # (-->); the dotted word's head left in the replaced word.
    assert span_at(marked) == expected


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


def test_the_spec_command_prints_the_canonical_spans():
    # Defects: the module does not run as a command, prints something that is
    # no JSON, or publishes spans other than completion_word_span's.
    result = subprocess.run(
        [sys.executable, "-m", "lispy.word_boundary"],
        capture_output=True,
        check=True,
    )
    spec = json.loads(result.stdout.decode("utf-8"))
    assert spec["version"] == 2 and spec["columns"] == "code points"
    assert len(spec["cases"]) == len(CASE_INPUTS)
    for case in spec["cases"]:
        expected = completion_word_span(case["text"], case["caret"])
        assert (case["start"], case["replaceStart"], case["end"]) == expected


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
