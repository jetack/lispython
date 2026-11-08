"""The completion word boundary as a conformance spec for editor clients.

``python -m lispy.word_boundary`` prints, as JSON, the rule of
``completion_word_span`` (``core/compiler/literal.py``, its one home) and a
set of cases whose expected spans that function computes. lpy-mode and
vsc-lispy compute the boundary natively on every keystroke, so they pin their
copies against these cases. Columns are code points (Python str indices); a
client counting UTF-16 units converts (the astral case is there for it).

No pygls or lsprotocol import: this runs wherever lispython does.
"""

import json
import sys

from lispy.core.compiler.literal import SIGNS, completion_word_span, name_char_p
from lispy.core.nodes import NAME_MANGLE

VERSION = 1

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
    "a?b|",
    "->f|",
    "-|>f",
    "-->x|",
    "-?|",
    "-fo|",
    "--x|",
    "-|foo",
    "-|-x",
    "-|",
    "--|",
    "(- -fo|",
    "(- -fo|)",
    "-.x|",
    "a.-|",
    ":kw|",
    "^x|",
    "*x|",
    "**x|",
    "**=|",
    "+=|",
    "+x|",
    "'x|",
    "`x|",
    "~x|",
    "~@x|",
    ",x|",
    "#x|",
    "a~b|",
    "a@b|",
    "ok!|",
    "a<b|",
    "a=b|",
    "a>b|",
    "a/b|",
    "my-x|",
    "my-|x",
    "x-|",
    "heapq.nl|",
    "heapq.|",
    "a.b|.c",
    "a..c|",
    ".method|",
    "(.c|",
    "x .c|",
    "(foo bar|)",
    '"ab|"',
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


def rule():
    """The rule's data, read from the homes the compiler uses: the mangle,
    the punctuation it makes name characters, and where each counts."""
    keys = [spelling for spelling, _ in NAME_MANGLE]
    standalone = sorted({ch for key in keys for ch in key if name_char_p(ch, 0)})
    in_context = {
        ch: key
        for key in keys
        for ch in key
        if not ("a" + ch).isidentifier() and ch not in standalone
    }
    return {
        "nameChars": 'c where ("a" + c).isidentifier(), or name punctuation',
        "mangle": [list(pair) for pair in NAME_MANGLE],
        "namePunctuation": standalone,
        "punctuationOnlyInside": in_context,
        "dot": ".",
        "signs": [ch for ch in SIGNS if name_char_p(ch, 0)],
        "signRunPeeledBefore": 'c where c.isalnum() or c in "_."',
        "allSignsWord": "empty at the caret (the operator)",
        "caretInsidePeeledSigns": "empty at the caret",
        "start": "back over name chars and dots, less peeled leading signs",
        "replaceStart": "after the last dot between start and caret",
        "end": "forward over name chars from the caret (no dot)",
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
