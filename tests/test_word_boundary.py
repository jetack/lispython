"""The completion word boundary (completion_word_span) and its published spec
(python -m lispy.word_boundary), which lpy-mode and vsc-lispy pin against."""

import json
import string
import subprocess
import sys

import pytest

from lispy.core.compiler.literal import completion_word_span, name_char_p
from lispy.core.nodes import NAME_MANGLE, Paren, Symbol
from lispy.core.parser import parse
from lispy.word_boundary import CASE_INPUTS

# (line with | at the caret, (start, replace_start, end)), checked by hand.
SPANS = [
    ("foo|", (0, 0, 3)),
    ("fo|o", (0, 0, 3)),
    ("|x", (0, 0, 1)),
    ("|", (0, 0, 0)),
    ("x1|", (0, 0, 2)),
    # The mangle's punctuation is part of a name.
    ("ok?|", (0, 0, 3)),
    ("my-x|", (0, 0, 4)),
    ("x-|", (0, 0, 2)),
    ("->f|", (0, 0, 3)),
    ("-|>f", (0, 0, 3)),
    # A > is a name character only as the > of ->.
    ("a>b|", (2, 2, 3)),
    ("-->x|", (0, 0, 4)),
    ("-?|", (0, 0, 2)),
    # Leading signs before a letter, digit, _ or . are the reader's unary
    # operators, never part of the name.
    ("-fo|", (1, 1, 3)),
    ("--x|", (2, 2, 3)),
    ("(- -fo|", (4, 4, 6)),
    ("-|foo", (1, 1, 4)),
    ("-.x|", (1, 2, 3)),
    # A sign run alone is the operator; a caret inside a peeled run is
    # outside the name.
    ("-|", (1, 1, 1)),
    ("--|", (2, 2, 2)),
    ("-|-x", (1, 1, 1)),
    # Whatever the reader lets into a symbol beyond the name rule ends the
    # word.
    ("a~b|", (2, 2, 3)),
    ("a@b|", (2, 2, 3)),
    ("ok!|", (3, 3, 3)),
    ("a<b|", (2, 2, 3)),
    ("a=b|", (2, 2, 3)),
    ("a/b|", (2, 2, 3)),
    ("#x|", (1, 1, 2)),
    ("+x|", (1, 1, 2)),
    # Prefixes are no name characters.
    (":kw|", (1, 1, 3)),
    ("^x|", (1, 1, 2)),
    ("*x|", (1, 1, 2)),
    ("**x|", (2, 2, 3)),
    ("'x|", (1, 1, 2)),
    ("`x|", (1, 1, 2)),
    ("~x|", (1, 1, 2)),
    ("~@x|", (2, 2, 3)),
    (",x|", (1, 1, 2)),
    # An augmented assignment has no name characters: never cut to =.
    ("**=|", (3, 3, 3)),
    ("+=|", (2, 2, 2)),
    # The dotted word starts at its first part, the replaced word after the
    # last dot; the end never crosses a dot.
    ("heapq.nl|", (0, 6, 8)),
    ("heapq.|", (0, 6, 6)),
    ("a.b|.c", (0, 2, 3)),
    (".method|", (0, 1, 7)),
    ("(.c|", (1, 2, 3)),
    ("a.-|", (0, 2, 3)),
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
    # Defects: the reader's symbol characters taken as name characters (a~b
    # one word, **= cut to =); a peeled sign kept in the word (-fo, which
    # matches no name); a > counted outside -> (a>b one word); the dotted
    # word's head left in the replaced word.
    assert span_at(marked) == expected


def test_the_mangle_is_the_only_name_punctuation():
    # Defect: a mangle entry added in nodes.NAME_MANGLE while the word rule
    # keeps a punctuation set of its own, so completion cuts the new name
    # (or, the other way, the rule accepts punctuation no name can hold).
    keys = [spelling for spelling, _ in NAME_MANGLE]
    for key in keys:
        line = f"x{key}y"
        assert span_at(line + "|") == (0, 0, len(line)), key
    mangled = {ch for key in keys for ch in key}
    for ch in sorted(set(string.punctuation) - mangled - {"_", "."}):
        assert completion_word_span(f"a{ch}b", 3) == (2, 2, 3), ch
        assert not name_char_p(ch, 0), ch


def reader_name_start(token):
    """The column where the reader's name starts in token, past the unary
    operators it peels off the front."""
    [node] = parse(token)
    while isinstance(node, Paren) and node.list[0].value in "+-":
        node = node.list[1]
    assert type(node) is Symbol
    return node.col_offset


@pytest.mark.parametrize(
    "token", ["-foo", "--x", "-_a", "-한", "-.x", "->f", "-->x", "-?", "--?", "x-"]
)
def test_the_word_peels_the_signs_the_reader_peels(token):
    # Defect: the word rule and the reader (token_parse) peel different sign
    # runs, so the word typed is not the name the code holds.
    assert completion_word_span(token, len(token))[0] == reader_name_start(token)


def test_the_spec_command_prints_the_canonical_spans():
    # Defects: the module does not run as a command, prints something that is
    # no JSON, or publishes spans other than completion_word_span's.
    result = subprocess.run(
        [sys.executable, "-m", "lispy.word_boundary"],
        capture_output=True,
        check=True,
    )
    spec = json.loads(result.stdout.decode("utf-8"))
    assert spec["version"] == 1 and spec["columns"] == "code points"
    assert spec["rule"]["mangle"] == [list(pair) for pair in NAME_MANGLE]
    assert len(spec["cases"]) == len(CASE_INPUTS)
    for case in spec["cases"]:
        expected = completion_word_span(case["text"], case["caret"])
        assert (case["start"], case["replaceStart"], case["end"]) == expected
