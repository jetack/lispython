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
from lispy.core.compiler.literal import number_parts
from lispy.core.nodes import Constant, Paren, Symbol
from lispy.core.parser import (
    _signed_complex_p,
    completion_word_span,
    parse,
    token_prefixes,
    tokenize,
)
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
    # The reader lexes the line, not the run: an opening or a string opening
    # may pass the run's end, and the caret in it is at no word.
    ("*|[1 2]", (1, 1, 1)),
    ('a|"', (1, 1, 1)),
    ('f|"x"', (1, 1, 1)),
    # A string opening inside the run, the caret after it.
    ("x'''|", (4, 4, 4)),
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
    assert spec["version"] == 3 and spec["columns"] == "code points"
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


def one_of(entries):
    """A regex alternation of literal entries, longest first."""
    return "|".join(re.escape(e) for e in sorted(entries, key=len, reverse=True))


def spec_word_span(rule):
    """A function (line, caret) -> (start, replaceStart, end), as an editor
    client reproduces it from the published rule alone: no parser internal is
    read here. The rule's regexes are compiled once."""
    stop = "[" + "".join(re.escape(c) for c in rule["stopChars"]) + "]"
    form_end = "(?:[" + "".join(re.escape(c) for c in rule["formEndChars"]) + "]|$)"
    wrapper = f"(?:{one_of(rule['wrappers'])})"
    opening = f"(?:{one_of(rule['openings'])})"
    meta = f"(?:{one_of(rule['metaChars'])})"
    quotes = f"(?:{one_of(rule['danglingQuotes'])})"
    token_re = re.compile(
        "|".join(
            f"(?P<{kind}>{pattern})"
            for kind, pattern in [
                ("opening", wrapper + opening),
                (
                    "string",
                    f"{rule['stringPrefix']}(?:{one_of(rule['stringOpenings'])})",
                ),
                ("dangling", f"{wrapper}{quotes}(?={form_end})"),
                ("meta", f"{wrapper}{meta}(?!{form_end})"),
                ("bad", f"\\^\\*+(?:{opening}|{meta}|(?={stop}|$))"),
                ("number", f"(?:{rule['numberToken']})(?={form_end})"),
                ("symbol", f"(?:(?!{stop}).)+"),
            ]
        )
    )
    stop_re = re.compile(stop)
    number_re = re.compile(rule["numberToken"])
    signed_complex_re = re.compile(rule["signedComplex"])
    signs = "".join(rule["signs"])

    def word_span(line, caret):
        run_start, run_end = caret, caret
        while run_start > 0 and not stop_re.match(line[run_start - 1]):
            run_start -= 1
        while run_end < len(line) and not stop_re.match(line[run_end]):
            run_end += 1
        # Lex the line from the run's start until a token reaches the run's end.
        pos, kind = run_start, None
        while pos < run_end:
            m = token_re.match(line, pos)
            start, end, kind = pos, m.end(), m.lastgroup
            if end > caret or (end == caret and kind in ("symbol", "number")):
                break
            pos, kind = end, None
        if kind not in ("symbol", "number"):
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
        expected = (case["start"], case["replaceStart"], case["end"])
        got = word_span(case["text"], case["caret"])
        assert got == expected, (case["text"], case["caret"])


# Every line up to the length over each alphabet, at every caret.
DIFFERENTIAL = [("-+*^:'`~@([\" .1jx;#", 4), ("-+*:1jx.e", 5)]


@pytest.mark.parametrize("alphabet, length", DIFFERENTIAL)
def test_the_published_rule_agrees_with_the_reader_everywhere(
    published, alphabet, length
):
    # Defect: the rule and completion_word_span part on a line no case
    # names (an ordering of the token patterns, a lookahead, the peel's
    # re-typing), so a client pinned on the cases still differs from the
    # reader as the user types.
    word_span = spec_word_span(published["rule"])
    for size in range(length + 1):
        for chars in itertools.product(alphabet, repeat=size):
            line = "".join(chars)
            for caret in range(size + 1):
                got = word_span(line, caret)
                assert got == completion_word_span(line, caret), (line, caret)


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
