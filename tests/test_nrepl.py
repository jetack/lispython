"""nREPL connection handler: framing, UTF-8 decoding and error replies.

The handler is driven with a fake connection whose recv returns fixed byte
chunks, so where a read boundary falls is deterministic."""

import gc
import json
import linecache
import re
import sys
import tracemalloc

import pytest

from lispy import tools


class FakeConn:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.sent = []
        self.closed = False

    def recv(self, n):
        return self.chunks.pop(0) if self.chunks else b""

    def sendall(self, data):
        self.sent.append(data)

    def close(self):
        self.closed = True


def line(**msg):
    return json.dumps(msg, ensure_ascii=False).encode("utf-8") + b"\n"


def serve(chunks):
    conn = FakeConn(chunks)
    scope = {"__name__": "__nrepl_test__", "__macro_namespace": {}}
    tools.nrepl_eval("(require lispy.macros *)", scope)
    tools._nrepl_handle_client(conn, ("127.0.0.1", 0), scope)
    assert conn.closed
    replies = []
    for data in conn.sent:
        assert data.endswith(b"\n")
        replies.append(json.loads(data.decode("utf-8")))
    return replies


def test_character_split_across_reads_is_decoded_whole():
    first = line(op="eval", id=1, code='(print "한국어")')
    second = line(op="ping", id=2)
    cut = first.index("국".encode()) + 1  # inside the 3-byte character
    replies = serve([first[:cut], first[cut:] + second])
    assert [r["id"] for r in replies] == [1, 2]
    assert replies[0]["op"] == "eval-result"
    assert replies[0]["error"] is None
    assert replies[0]["stdout"] == "한국어\n"
    assert replies[1] == {"op": "pong", "id": 2}


def test_unknown_op_replies_with_error_and_keeps_serving():
    replies = serve([line(op="bogus", id=7) + line(op="ping", id=8)])
    assert replies == [
        {"op": "error", "error": "unknown op: bogus", "id": 7},
        {"op": "pong", "id": 8},
    ]


def test_handler_exception_replies_with_error_and_keeps_serving():
    replies = serve([line(op="eval", id=3) + line(op="ping", id=4)])
    assert replies == [
        {"op": "error", "error": "KeyError: 'code'", "id": 3},
        {"op": "pong", "id": 4},
    ]


def test_invalid_utf8_line_replies_without_id_and_keeps_serving():
    replies = serve([b"\xff\n" + line(op="ping", id=5)])
    assert replies[0]["op"] == "error"
    assert "utf-8" in replies[0]["error"]
    assert "id" not in replies[0]
    assert replies[1] == {"op": "pong", "id": 5}


def test_json_parse_error_replies_without_id_and_keeps_serving():
    replies = serve([b"{not json\n" + line(op="ping", id=6)])
    assert replies[0]["op"] == "error"
    assert "id" not in replies[0]
    assert replies[1] == {"op": "pong", "id": 6}


def test_connection_closes_when_recv_returns_empty():
    conn = FakeConn([])
    tools._nrepl_handle_client(conn, ("127.0.0.1", 0), {})
    assert conn.closed
    assert conn.sent == []


def test_exit_in_eval_is_an_eval_error_and_keeps_serving():
    replies = serve([line(op="eval", id=1, code="(exit)") + line(op="ping", id=2)])
    assert replies[0]["op"] == "eval-result"
    assert replies[0]["id"] == 1
    assert "SystemExit" in replies[0]["error"]
    assert replies[1] == {"op": "pong", "id": 2}


def test_unserialisable_result_replies_with_error_and_keeps_serving(monkeypatch):
    monkeypatch.setattr(tools, "nrepl_complete", lambda prefix, scope: {"x": object()})
    replies = serve([line(op="complete", id=1, prefix="a") + line(op="ping", id=2)])
    assert replies[0]["op"] == "error"
    assert replies[0]["id"] == 1
    assert "not JSON serializable" in replies[0]["error"]
    assert replies[1] == {"op": "pong", "id": 2}


def test_crlf_terminated_lines_are_served():
    replies = serve([b'{"op": "ping", "id": 1}\r\n{"op": "ping", "id": 2}\r\n'])
    assert [r["id"] for r in replies] == [1, 2]


def test_blank_lines_get_no_reply():
    assert serve([b"\n  \n\r\n\t\n"]) == []


def test_partial_trailing_line_at_eof_gets_no_reply():
    assert serve([line(op="ping", id=1) + b'{"op": "ping", "id": 2}']) == [
        {"op": "pong", "id": 1}
    ]


def test_non_object_json_replies_without_id():
    replies = serve([b"[1]\n" + line(op="ping", id=3)])
    assert replies[0]["op"] == "error"
    assert "id" not in replies[0]
    assert replies[1] == {"op": "pong", "id": 3}


def test_deeply_nested_json_replies_without_id_and_keeps_serving():
    replies = serve([b"[" * 200000 + b"\n" + line(op="ping", id=9)])
    assert replies[0]["op"] == "error"
    assert "id" not in replies[0]
    assert replies[1] == {"op": "pong", "id": 9}


def test_overlong_line_replies_without_id_and_keeps_serving(monkeypatch):
    monkeypatch.setattr(tools, "NREPL_MAX_LINE", 100)
    # The oversized line spans several reads and ends mid-chunk; the line after
    # its newline is still served, and only one error is sent for the line.
    replies = serve([b"x" * 60, b"x" * 60, b"x" * 60, b"x\n" + line(op="ping", id=4)])
    assert [r["op"] for r in replies] == ["error", "pong"]
    assert "id" not in replies[0]
    assert replies[1] == {"op": "pong", "id": 4}


def test_line_that_never_ends_is_rejected_before_its_newline_arrives(monkeypatch):
    monkeypatch.setattr(tools, "NREPL_MAX_LINE", 100)
    # No newline ever comes: the error must be sent once while the buffer would
    # otherwise still be growing, and the discarded tail must not add replies.
    replies = serve([b"x" * 60] * 10)
    assert [r["op"] for r in replies] == ["error"]
    assert "id" not in replies[0]


def test_overlong_complete_line_in_one_read_replies_once(monkeypatch):
    monkeypatch.setattr(tools, "NREPL_MAX_LINE", 100)
    replies = serve([b"x" * 500 + b"\n" + line(op="ping", id=5)])
    assert [r["op"] for r in replies] == ["error", "pong"]


def test_line_at_the_cap_is_served(monkeypatch):
    ping = line(op="ping", id=6)
    monkeypatch.setattr(tools, "NREPL_MAX_LINE", len(ping) - 1)
    assert serve([ping[:-1], ping[-1:]]) == [{"op": "pong", "id": 6}]


def test_overlong_valid_json_line_is_rejected_for_its_length(monkeypatch):
    monkeypatch.setattr(tools, "NREPL_MAX_LINE", 100)
    replies = serve([line(op="ping", id=1, pad="x" * 200) + line(op="ping", id=2)])
    assert [r["op"] for r in replies] == ["error", "pong"]
    assert replies[0]["error"] == "request line too long"


@pytest.mark.parametrize(
    "error", [ConnectionResetError, ConnectionAbortedError, TimeoutError]
)
def test_recv_socket_error_ends_quietly(error):
    class Failing(FakeConn):
        def recv(self, n):
            raise error

    conn = Failing([])
    tools._nrepl_handle_client(conn, ("127.0.0.1", 0), {})
    assert conn.closed


@pytest.mark.parametrize("error", [BrokenPipeError, OSError])
def test_sendall_socket_error_ends_quietly(error):
    class Failing(FakeConn):
        def sendall(self, data):
            raise error

    conn = Failing([line(op="ping", id=1) + line(op="ping", id=2)])
    tools._nrepl_handle_client(conn, ("127.0.0.1", 0), {})
    assert conn.closed
    assert conn.sent == []


# --- eval errors point at the client's buffer --------------------------------
#
# An eval request carries the buffer's file and the line/column of the code's
# first character; tracebacks and compile errors must name those, show the
# source line, and leave out the server's own frames.


@pytest.fixture(autouse=True)
def isolated_linecache(monkeypatch):
    """Every eval registers its source in linecache; drop it after the test."""
    monkeypatch.setattr(linecache, "cache", dict(linecache.cache))


@pytest.fixture
def isolated(monkeypatch):
    """A fresh scope; sys.path changes (load-file) end with the test."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    scope = {"__name__": "__nrepl_test__", "__macro_namespace": {}}
    tools.nrepl_eval("(require lispy.macros *)", scope)
    return scope


def op(scope, name, **msg):
    return tools._nrepl_dispatch({"op": name, **msg}, scope)


def dispatch(scope, **msg):
    return op(scope, "eval", **msg)


UPDATE = "(def update [xs]\n  (return (sub xs [0])))"
SOLUTION = "(def solution [xs]\n  (return (update xs)))"


def assert_clean_traceback(error):
    assert error.startswith("Traceback (most recent call last):")
    assert "tools.lpy" not in error
    assert "line 0," not in error
    assert error.endswith(
        "TypeError: list indices must be integers or slices, not list"
    )


def frame(path, lineno, name, source):
    return f'File "{path}", line {lineno}, in {name}\n    {source}\n'


def test_eval_traceback_names_buffer_file_lines_and_source(isolated, tmp_path):
    path = str(tmp_path / "unsaved.lpy")  # never written: an unsaved buffer
    code = f"{UPDATE}\n{SOLUTION}\n(-> [1 2] solution print)"
    error = dispatch(isolated, code=code, file=path, line=90)["error"]
    assert_clean_traceback(error)
    assert frame(path, 94, "<module>", "(-> [1 2] solution print)") in error
    assert frame(path, 93, "solution", "(return (update xs)))") in error
    assert frame(path, 91, "update", "(return (sub xs [0])))") in error


def test_separate_evals_of_one_file_all_show_their_source(isolated, tmp_path):
    path = str(tmp_path / "unsaved.lpy")
    assert dispatch(isolated, code=UPDATE, file=path, line=90)["error"] is None
    assert dispatch(isolated, code=SOLUTION, file=path, line=140)["error"] is None
    error = dispatch(isolated, code="(solution [1 2])", file=path, line=200)["error"]
    assert_clean_traceback(error)
    assert frame(path, 200, "<module>", "(solution [1 2])") in error
    assert frame(path, 141, "solution", "(return (update xs)))") in error
    assert frame(path, 91, "update", "(return (sub xs [0])))") in error


def test_evals_without_file_get_distinct_names_and_show_source(isolated):
    assert dispatch(isolated, code=UPDATE)["error"] is None
    error = dispatch(isolated, code="(update [1 2])")["error"]
    assert_clean_traceback(error)
    frames = re.findall(r'File "(<nrepl-\d+>)", line (\d+), in \S+\n    (.*)', error)
    [[module_file, *module_frame], [def_file, *def_frame]] = frames
    assert module_file != def_file
    assert module_frame == ["1", "(update [1 2])"]
    assert def_frame == ["2", "(return (sub xs [0])))"]


def test_compile_error_reports_buffer_line_and_column(isolated, tmp_path):
    path = str(tmp_path / "unsaved.lpy")
    error = dispatch(isolated, code="(return 1)", file=path, line=7, column=5)
    assert error["error"].startswith(f"{path}:7:5: error: ")
    assert error["error"].endswith("\n    (return 1)\n    ^")
    code = "(+ 1 2)\n  (return 1)"
    error = dispatch(isolated, code=code, file=path, line=10, column=5)
    assert error["error"].startswith(f"{path}:11:3: error: ")


def test_load_file_traceback_and_replaces_earlier_evals(isolated, tmp_path):
    path = tmp_path / "prog.lpy"
    path.write_text(f"{UPDATE}\n(update [1 2])\n")
    stale = "(= a 1)\n(= b 2)\n(= c 3)\n(= d 4)\n;; stale"
    assert dispatch(isolated, code=stale, file=str(path), line=1)["error"] is None
    reply = tools._nrepl_dispatch({"op": "load-file", "path": str(path)}, isolated)
    error = reply["error"]
    assert_clean_traceback(error)
    assert frame(path, 3, "<module>", "(update [1 2])") in error
    assert frame(path, 2, "update", "(return (sub xs [0])))") in error
    assert linecache.getlines(str(path)) == path.read_text().splitlines(True)


def test_region_eval_overlays_the_file_on_disk(isolated, tmp_path):
    path = tmp_path / "prog.lpy"
    path.write_text("(= a 1)\n(= b 2)\n(= c 3)\n")
    assert dispatch(isolated, code="(= b 20)", file=str(path), line=2)["error"] is None
    assert linecache.getlines(str(path))[:] == ["(= a 1)\n", "(= b 20)\n", "(= c 3)\n"]


def test_whole_file_eval_must_start_the_file(isolated):
    with pytest.raises(ValueError, match="line 1, column 1"):
        tools.nrepl_eval("(= a 1)", isolated, filename="f.lpy", line=2, whole_file=True)


# linecache keeps an eval's entry for the whole session and every file-less
# eval adds one, so an entry must hold the code sent, not a slot per buffer
# line: a padded list at the line cap pinned about 8 MB per request.
MAX_LINE = tools.NREPL_MAX_POSITION_LINE


def test_eval_at_the_line_cap_retains_only_its_own_code(isolated, tmp_path):
    path = str(tmp_path / "unsaved.lpy")
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        reply = dispatch(isolated, code="(= z 1)", file=path, line=MAX_LINE, column=3)
        gc.collect()
        retained = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    assert reply["error"] is None
    assert retained < 256 * 1024
    assert len(linecache.getlines(path)) == MAX_LINE
    assert linecache.getline(path, MAX_LINE) == "  (= z 1)\n"
    assert linecache.getline(path, 1) == "\n"
    assert linecache.getline(path, MAX_LINE - 1) == "\n"
    assert linecache.getline(path, MAX_LINE + 1) == ""


def test_traceback_at_the_line_cap_shows_every_region(isolated, tmp_path):
    path = str(tmp_path / "unsaved.lpy")
    update_line = MAX_LINE - 1
    assert dispatch(isolated, code=UPDATE, file=path, line=update_line)["error"] is None
    assert dispatch(isolated, code=SOLUTION, file=path, line=3)["error"] is None
    error = dispatch(isolated, code="(solution [1 2])", file=path, line=1)["error"]
    assert_clean_traceback(error)
    assert frame(path, 1, "<module>", "(solution [1 2])") in error
    assert frame(path, 4, "solution", "(return (update xs)))") in error
    assert frame(path, MAX_LINE, "update", "(return (sub xs [0])))") in error


LIMITS = {
    "line": tools.NREPL_MAX_POSITION_LINE,
    "column": tools.NREPL_MAX_POSITION_COLUMN,
}


@pytest.mark.parametrize(
    ["field", "value"],
    [
        ["line", 0],
        ["line", "x"],
        ["line", True],
        ["column", 0],
        ["column", 1.0],
        # Past the cap: the source would be padded to this position, so a
        # huge one would make the server allocate without bound.
        ["line", tools.NREPL_MAX_POSITION_LINE + 1],
        ["line", 10**9],
        ["column", tools.NREPL_MAX_POSITION_COLUMN + 1],
        ["column", 10**9],
    ],
)
def test_invalid_position_replies_with_error_and_keeps_serving(field, value):
    request = line(op="eval", id=9, code="1", **{field: value})
    replies = serve([request + line(op="ping", id=10)])
    limit = LIMITS[field]
    assert replies == [
        {
            "op": "error",
            "error": f"ValueError: {field} must be an integer from 1 to {limit}, "
            f"got {value!r}",
            "id": 9,
        },
        {"op": "pong", "id": 10},
    ]


def test_position_at_the_caps_is_evaluated():
    request = line(
        op="eval",
        id=11,
        code="(+ 1 2)",
        line=tools.NREPL_MAX_POSITION_LINE,
        column=tools.NREPL_MAX_POSITION_COLUMN,
    )
    [reply] = serve([request])
    assert (reply["id"], reply["value"], reply["error"]) == (11, "3", None)


def test_null_position_fields_take_the_defaults():
    request = line(op="eval", id=1, code="(+ 1 2)", file=None, line=None, column=None)
    [reply] = serve([request])
    assert (reply["value"], reply["error"]) == ("3", None)


# --- ns picks the scope only -------------------------------------------------


def test_eval_ns_runs_in_that_files_scope_without_naming_it(isolated, tmp_path):
    path = tmp_path / "prog.lpy"
    path.write_text("(= on-disk 1)\n")
    ns = str(path)
    define = "(def nsfun [] (return (sub [] 0)))"
    assert dispatch(isolated, code=define, ns=ns)["error"] is None
    # The definition is in the file's scope, not the session's.
    assert "NameError" in dispatch(isolated, code="(nsfun)")["error"]
    error = dispatch(isolated, code="(nsfun)", ns=ns)["error"]
    assert "IndexError" in error
    assert "<nrepl-" in error
    assert ns not in error
    assert linecache.getlines(ns) == ["(= on-disk 1)\n"]
    assert "IndexError" in dispatch(isolated, code="(nsfun)", file=ns, line=1)["error"]


def test_eval_file_wins_over_ns(isolated, tmp_path):
    file_path, ns_path = str(tmp_path / "a.lpy"), str(tmp_path / "b.lpy")
    code = "(def only-in-a [] (return (sub [] 0)))\n(only-in-a)"
    reply = dispatch(isolated, code=code, file=file_path, ns=ns_path, line=1)
    assert f'File "{file_path}"' in reply["error"]
    assert ns_path not in reply["error"]
    call = dispatch(isolated, code="(only-in-a)", file=file_path)
    assert "IndexError" in call["error"]
    assert "NameError" in dispatch(isolated, code="(only-in-a)", ns=ns_path)["error"]


def test_complete_ns_completes_from_that_files_scope(isolated, tmp_path):
    ns = str(tmp_path / "prog.lpy")
    dispatch(isolated, code="(= nscomplete-me 1)", ns=ns)
    session = op(isolated, "complete", prefix="nscomplete")
    assert "nscomplete-me" not in session["completions"]
    assert (
        "nscomplete-me"
        in op(isolated, "complete", prefix="nscomplete", ns=ns)["completions"]
    )


def test_macroexpand_ns_expands_a_macro_of_that_files_scope(isolated, tmp_path):
    ns = str(tmp_path / "prog.lpy")
    dispatch(isolated, code="(defmacro nsmac [x] `(+ ~x 1))", ns=ns)
    reply = op(isolated, "macroexpand", code="(nsmac 5)", ns=ns)
    assert reply["error"] is None
    assert "nsmac" not in reply["expansion"]
    assert op(isolated, "macroexpand", code="(nsmac 5)")["expansion"] == "(nsmac 5)"


@pytest.mark.parametrize("name", ["eval", "complete", "macroexpand"])
@pytest.mark.parametrize("ns", [5, [], "", "user"])
@pytest.mark.parametrize("file", [None, "/tmp/some-file.lpy"])
def test_invalid_ns_replies_with_error_and_keeps_serving(name, ns, file):
    msg = {"code": "1", "prefix": "a", "ns": ns}
    if file:
        msg["file"] = file  # file wins the scope, but a bad ns is still refused
    replies = serve([line(op=name, id=7, **msg) + line(op="ping", id=8)])
    assert replies == [
        {
            "op": "error",
            "error": f"ValueError: ns must be an absolute file path, got {ns!r}",
            "id": 7,
        },
        {"op": "pong", "id": 8},
    ]


def test_ns_eval_at_line_one_never_re_derives_the_prelude(isolated, tmp_path):
    ns = str(tmp_path / "prog.lpy")
    first = dispatch(isolated, code="(require lispy.macros [])", file=ns, line=1)
    assert first["error"] is None
    probe = "(when True 1)"
    assert op(isolated, "macroexpand", code=probe, ns=ns)["expansion"] == probe
    assert dispatch(isolated, code="(+ 1 2)", ns=ns, line=1, column=1)["error"] is None
    # still absent: the ns eval at line 1 was not a TOP start
    assert op(isolated, "macroexpand", code=probe, ns=ns)["expansion"] == probe


@pytest.mark.parametrize("name", ["eval", "complete", "macroexpand"])
def test_null_ns_counts_as_absent(name):
    [reply] = serve([line(op=name, id=1, code="(+ 1 2)", prefix="pri", ns=None)])
    assert reply["op"] != "error"
    assert reply["id"] == 1


def test_ns_field_has_one_spelling_in_tools():
    # "ns" is spelled once, as NREPL_NS_FIELD; editor clients mirror that name,
    # and a second inline spelling could drift from it.
    from pathlib import Path

    src = Path(tools.__file__).with_name("tools.lpy").read_text()
    assert tools.NREPL_NS_FIELD == "ns"
    assert re.findall(r'"ns"', src) == ['"ns"']


@pytest.fixture
def spelled_scope(tmp_path, monkeypatch):
    (tmp_path / "mac_mod.lpy").write_text("(defmacro my-mac [] 1)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "mac_mod", raising=False)
    scope = {"__name__": "__nrepl_spelling__", "__macro_namespace": {}}
    result = tools.nrepl_eval(
        "(import collections)\n"
        "(= my_val 1)\n"
        "(= a__b 1)\n"
        "(= _priv_x 1)\n"
        "(defmacro my_mac [x] x)\n"
        "(defmacro top-mac [] 3)\n"
        "(require mac-mod)",
        scope,
    )
    assert result["error"] is None
    yield scope
    sys.modules.pop("mac_mod", None)


def complete_labels(prefix, scope):
    return [item["label"] for item in tools.nrepl_complete(prefix, scope)["items"]]


@pytest.mark.parametrize(
    "prefix, label, absent",
    [
        ("my", "my-val", "my_val"),
        ("a", "a__b", "a--b"),
        ("_pri", "_priv-x", "_priv_x"),
        ("collections.OrderedDict.move", "move-to-end", None),
        ("my-v", "my-val", "my_val"),
        # A typed _ matches a -: the label stays the LSP's, whatever the
        # spelling typed.
        ("my_v", "my-val", "my_val"),
        ("my_", "my-val", "my_val"),
        ("collections.OrderedDict.move_to", "move-to-end", "move_to_end"),
        # A macro takes the default spelling too, however it is typed.
        ("my", "my-mac", "my_mac"),
        ("my_m", "my-mac", "my_mac"),
        ("top_m", "top-mac", "top_mac"),
        ("mac_m", "mac-mod.my-mac", "mac_mod.my_mac"),
    ],
)
def test_complete_labels_names_by_the_hyphen_spelling(
    spelled_scope, prefix, label, absent
):
    # Defects: the nREPL drifting from the LSP's rule (hyphen_spelling), as
    # its old unmangling did (a__b offered as a--b, another name); a label
    # respelled after the typed word, which the LSP does not, so an editor
    # merging both answers shows the name twice; one name twice.
    labels = complete_labels(prefix, spelled_scope)
    assert labels.count(label) == 1
    assert absent not in labels


def test_a_dotted_macro_prefix_keeps_its_typed_head(spelled_scope):
    # Defect: the whole key respelled under a dotted prefix (mac-mod.my-mac
    # after mac_mod.), so the legacy full name no longer starts with what the
    # client typed and it shows nothing.
    reply = tools.nrepl_complete("mac_mod.my", spelled_scope)
    assert [item["label"] for item in reply["items"]] == ["my-mac"]
    assert reply["completions"] == ["mac_mod.my-mac"]


def test_a_dotted_prefix_labels_the_name_after_its_last_dot(isolated):
    # Defect: labels carrying the typed head (heapq.nlargest), which every
    # client then strips itself, unlike the LSP's labels; and no word to tell
    # a client how much typed text a label replaces.
    assert dispatch(isolated, code="(import heapq)")["error"] is None
    reply = op(isolated, "complete", prefix="heapq.nl")
    assert reply["word"] == "nl"
    assert [item["label"] for item in reply["items"]] == ["nlargest"]
    assert reply["completions"] == ["heapq.nlargest"]
    assert op(isolated, "complete", prefix="heapq")["word"] == "heapq"


def test_complete_labels_a_name_as_the_evaluated_code_spells_it(isolated):
    # Defects (TASK-1141): ok_p offered as ok-p beside the LSP's ok?, so an
    # editor merging both shows the name twice; _to_foo offered as _to-foo
    # at the typed ->f, a label the client's prefix filter drops.
    code = "(def ok? [] 1)\n(def ->foo [] 2)"
    assert dispatch(isolated, code=code)["error"] is None
    assert complete_labels("ok", isolated) == ["ok?"]
    assert complete_labels("->f", isolated) == ["->foo"]


def test_complete_labels_a_symbol_only_name_as_the_code_spells_it(isolated):
    # Defect: the record keeping only spellings with a letter or digit, so
    # ? (the name _p) is offered as _p beside the LSP's ?, and an editor
    # merging both shows the name twice.
    assert dispatch(isolated, code="(= ? 1)")["error"] is None
    assert complete_labels("?", isolated) == ["?"]


def test_a_later_hyphen_spelling_keeps_the_code_spelling(isolated):
    # Defect: a later eval's ok-p overwriting the recorded ok?, so ok_p is
    # labelled ok-p while the LSP (the buffer's first spelling) keeps ok?,
    # and a client merging both shows the name twice.
    assert dispatch(isolated, code="(def ok? [] 1)")["error"] is None
    assert dispatch(isolated, code="(ok-p)")["error"] is None
    assert complete_labels("ok", isolated) == ["ok?"]


def test_load_file_replaces_the_recorded_spellings(isolated, tmp_path):
    # Defect: a reload merged into the record, so the ok? the file no longer
    # spells outlives the edit that respelled it ok-p.
    path = tmp_path / "prog.lpy"
    path.write_text("(def ok? [] 1)\n")
    assert op(isolated, "load-file", path=str(path))["error"] is None
    path.write_text("(def ok-p [] 1)\n")
    assert op(isolated, "load-file", path=str(path))["error"] is None
    assert complete_labels("ok", tools.nrepl_file_scope(str(path))) == ["ok-p"]


@pytest.mark.parametrize("prefix", ["ok-", "_to"])
def test_a_code_spelled_name_matches_only_its_label(isolated, prefix):
    # Defect: a second spelling (ok-p, _to-foo) offered for a word typed in
    # another spelling, a label the LSP never gives the name, so an editor
    # merging both answers shows the name twice.
    code = "(def ok? [] 1)\n(def ->foo [] 2)"
    assert dispatch(isolated, code=code)["error"] is None
    assert complete_labels(prefix, isolated) == []


def test_a_failed_load_file_keeps_the_recorded_spellings(isolated, tmp_path):
    # Defect: the record cleared or updated before the compile succeeded, so
    # a load-file that bound nothing respells the name the scope still holds.
    path = tmp_path / "prog.lpy"
    path.write_text("(def ok? [] 1)\n")
    assert op(isolated, "load-file", path=str(path))["error"] is None
    path.write_text("(def ok-p [] (return 1 2))\n")
    assert op(isolated, "load-file", path=str(path))["error"] is not None
    assert complete_labels("ok", tools.nrepl_file_scope(str(path))) == ["ok?"]


# --- a file scope's package ---------------------------------------------------


@pytest.fixture
def namespace(tmp_path, monkeypatch):
    """A namespace package nreplnspkg (no __init__) holding mac.lpy and a
    sibling b.lpy, under tmp_path, the server's working directory (its
    sys.path snapshot); what an eval imports is dropped after the test."""
    names = ["nreplnspkg", "nreplnspkg.mac"]
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(tools, "_NREPL_IMPORT_ROOTS", [str(tmp_path)])
    pkg = tmp_path / "nreplnspkg"
    pkg.mkdir()
    (pkg / "mac.lpy").write_text("(defmacro twice [x] (return `(+ ~x ~x)))\n")
    (pkg / "b.lpy").write_text("(= b 1)\n")
    yield pkg
    for name in names:
        sys.modules.pop(name, None)


def test_a_file_eval_requires_relatively_in_a_namespace_package(isolated, namespace):
    # Defects: the file scope gets no package, so the require fails with
    # "attempted relative import with no known parent package"; or the
    # package is decided against the live sys.path, where loading the
    # sibling b.lpy put the package directory itself, so a.lpy, first used
    # after it, is again a top-level module.
    load = op(isolated, "load-file", path=str(namespace / "b.lpy"))
    assert load["error"] is None
    code = "(require .mac *)\n(= y (twice 2))\n(print y)"
    reply = dispatch(isolated, code=code, file=str(namespace / "a.lpy"), line=1)
    assert reply["error"] is None
    assert reply["stdout"] == "4\n"


# --- stream evals, concurrent requests and interrupt -------------------------
#
# Driven over real sockets: the behaviour is the reader, the eval worker and a
# stream's flusher running at once.  Every wait is bounded (socket timeout,
# join timeout), and the endless loop a test starts is stopped by closing its
# connection in the fixture's teardown, which interrupts a running eval.

WAIT = 10  # seconds any one reply may take

ENDLESS = '(import time)\n(while True (print "tick") (time.sleep 0.005))'

WIRE_NAMES = {
    "NREPL_EVAL_OP": "eval",
    "NREPL_EVAL_RESULT_OP": "eval-result",
    "NREPL_LOAD_FILE_OP": "load-file",
    "NREPL_LOAD_FILE_RESULT_OP": "load-file-result",
    "NREPL_STREAM_FIELD": "stream",
    "NREPL_OUT_OP": "out",
    "NREPL_STDOUT_FIELD": "stdout",
    "NREPL_STDERR_FIELD": "stderr",
    "NREPL_INTERRUPT_OP": "interrupt",
    "NREPL_INTERRUPT_RESULT_OP": "interrupt-result",
    "NREPL_INTERRUPT_ID_FIELD": "interrupt-id",
    "NREPL_INTERRUPTED_ERROR": "KeyboardInterrupt",
}


def test_nrepl_wire_names():
    # The names lpy-mode and vsc-lispy send and read: a rename on one side
    # silently stops the feature on the other.
    assert {name: getattr(tools, name) for name in WIRE_NAMES} == WIRE_NAMES


class Client:
    def __init__(self, sock):
        self.sock = sock
        self.buf = b""

    def send(self, **msg):
        self.sock.sendall(line(**msg))

    def recv(self):
        while b"\n" not in self.buf:
            data = self.sock.recv(65536)  # bounded by the socket timeout
            assert data, "server closed the connection"
            self.buf += data
        raw, self.buf = self.buf.split(b"\n", 1)
        return json.loads(raw)

    def until(self, done):
        """Every message up to and including the first that done accepts,
        within WAIT seconds even while out messages keep arriving."""
        import time

        deadline = time.monotonic() + WAIT
        messages = []
        while True:
            assert time.monotonic() < deadline, f"no such reply in {messages[-3:]}"
            messages.append(self.recv())
            if done(messages[-1]):
                return messages


def is_out(msg, req_id):
    return msg.get("op") == "out" and msg.get("id") == req_id


def without_out(messages, req_id):
    return [m for m in messages if not is_out(m, req_id)]


@pytest.fixture
def connect(isolated):
    import socket
    import threading

    served = []

    def connect():
        client_sock, server_sock = socket.socketpair()
        client_sock.settimeout(WAIT)
        thread = threading.Thread(
            target=tools._nrepl_handle_client,
            args=(server_sock, ("test", 0), isolated),
            daemon=True,
        )
        thread.start()
        served.append((client_sock, thread))
        return Client(client_sock)

    yield connect
    # EOF on each connection interrupts its running eval and ends its worker.
    for client_sock, _ in served:
        client_sock.close()
    for _, thread in served:
        thread.join(WAIT)
    assert not any(thread.is_alive() for _, thread in served)


def test_stream_eval_streams_output_before_its_result(connect):
    client = connect()
    client.send(op="eval", id=1, stream=True, code=ENDLESS)
    first = client.recv()
    assert is_out(first, 1)
    assert "tick" in first["stdout"]


def test_partial_line_is_sent_before_the_eval_ends(connect, isolated):
    import threading

    isolated["gate"] = gate = threading.Event()
    client = connect()
    code = '(print "part" :end "")\n(gate.wait 10)'
    client.send(op="eval", id=1, stream=True, code=code)
    first = client.recv()  # no newline was written: only the flush delay sends it
    gate.set()
    assert first == {"op": "out", "stdout": "part", "id": 1}
    result = client.recv()
    assert result["op"] == "eval-result"
    assert result["error"] is None
    assert result["stdout"] == ""
    assert "stderr" not in result


def test_stream_output_keeps_write_order_across_stdout_and_stderr(connect):
    client = connect()
    code = '(import sys)\n(print "a")\n(print "b" :file sys.stderr)\n(print "c")'
    client.send(op="eval", id=1, stream=True, code=code)
    messages = client.until(lambda m: m.get("op") == "eval-result")
    chunks = []
    for msg in messages[:-1]:
        assert is_out(msg, 1)
        [(key, text)] = [(k, v) for k, v in msg.items() if k in ("stdout", "stderr")]
        if chunks and chunks[-1][0] == key:
            chunks[-1] = (key, chunks[-1][1] + text)
        else:
            chunks.append((key, text))
    assert chunks == [("stdout", "a\n"), ("stderr", "b\n"), ("stdout", "c\n")]
    assert messages[-1]["stdout"] == ""


def test_requests_are_answered_while_a_stream_eval_runs(connect):
    client = connect()
    client.send(op="eval", id=1, stream=True, code=ENDLESS)
    assert is_out(client.recv(), 1)
    client.send(op="complete", id=2, prefix="pri")
    before = client.until(lambda m: m.get("id") == 2)
    assert all(is_out(m, 1) for m in before[:-1])
    assert before[-1]["op"] == "complete-result"
    assert "print" in before[-1]["completions"]
    assert is_out(client.recv(), 1)  # the eval is still running


def test_interrupt_stops_a_running_stream_eval(connect):
    client = connect()
    client.send(op="eval", id=1, stream=True, code=ENDLESS)
    assert is_out(client.recv(), 1)
    client.send(op="interrupt", id=2, **{"interrupt-id": 1})
    messages = without_out(client.until(lambda m: m.get("op") == "eval-result"), 1)
    assert messages[0] == {"op": "interrupt-result", "interrupt-id": 1, "id": 2}
    [result] = messages[1:]
    assert result["id"] == 1
    assert result["error"].startswith("Traceback (most recent call last):")
    assert result["error"].endswith("KeyboardInterrupt")
    client.send(op="eval", id=3, code="(+ 1 2)")
    assert client.recv() == {
        "op": "eval-result",
        "value": "3",
        "stdout": "",
        "error": None,
        "id": 3,
    }


def test_every_interrupt_ends_in_a_result_and_the_worker_survives(connect):
    # The interrupt lands wherever the eval happens to be: in the user's
    # loop, in print's path through the output router and stream, or in
    # the worker's bookkeeping.  None of it may lose the final result or
    # kill the connection's worker.
    client = connect()
    busy = '(while True (print "t"))'
    for round_id in range(2, 202, 2):
        client.send(op="eval", id=round_id, stream=True, code=busy)
        assert is_out(client.recv(), round_id)
        client.send(op="interrupt", id=round_id + 1, **{"interrupt-id": round_id})
        messages = without_out(
            client.until(lambda m: m.get("op") == "eval-result"), round_id
        )
        assert [m["op"] for m in messages] == ["interrupt-result", "eval-result"]
        assert messages[1]["error"].endswith("KeyboardInterrupt")
    client.send(op="eval", id=1, code="(+ 1 2)")
    assert client.recv()["value"] == "3"


def test_interrupt_of_an_unknown_id_is_an_error_reply(connect):
    client = connect()
    client.send(op="interrupt", id=5, **{"interrupt-id": 99})
    reply = client.recv()
    assert reply["op"] == "error"
    assert reply["id"] == 5
    client.send(op="ping", id=6)
    assert client.recv() == {"op": "pong", "id": 6}


def test_interrupt_cancels_a_queued_stream_eval(connect):
    client = connect()
    client.send(op="eval", id=1, stream=True, code=ENDLESS)
    assert is_out(client.recv(), 1)
    client.send(op="eval", id=2, stream=True, code='(print "never")')
    client.send(op="interrupt", id=3, **{"interrupt-id": 2})
    client.send(op="interrupt", id=4, **{"interrupt-id": 1})
    messages = without_out(client.until(lambda m: m.get("id") == 2), 1)
    assert [m["id"] for m in messages] == [3, 4, 1, 2]
    assert messages[0]["op"] == messages[1]["op"] == "interrupt-result"
    assert messages[2]["error"].endswith("KeyboardInterrupt")
    assert messages[3] == {
        "op": "eval-result",
        "value": None,
        "stdout": "",
        "error": "KeyboardInterrupt",
        "id": 2,
    }


def test_a_stream_eval_reusing_a_running_id_is_refused(connect):
    client = connect()
    client.send(op="eval", id=1, stream=True, code=ENDLESS)
    assert is_out(client.recv(), 1)
    client.send(op="eval", id=1, stream=True, code='(print "twice")')
    refusal = without_out(client.until(lambda m: m.get("op") == "error"), 1)
    error = "an eval with id 1 is already queued or running"
    assert refusal == [{"op": "error", "error": error, "id": 1}]


def test_concurrent_stream_evals_keep_their_own_output(connect, isolated):
    import threading

    # Each eval waits for the other, so their prints surely overlap in time.
    isolated["barrier"] = threading.Barrier(2)
    clients = {"alpha": connect(), "beta": connect()}
    for marker, client in clients.items():
        code = (
            "(import time)\n(barrier.wait 10)\n"
            f'(for i (range 40) (print "{marker}") (time.sleep 0.002))'
        )
        client.send(op="eval", id=1, stream=True, code=code)
    for marker, client in clients.items():
        messages = client.until(lambda m: m.get("op") == "eval-result")
        assert messages[-1]["error"] is None
        assert "".join(m["stdout"] for m in messages[:-1]) == f"{marker}\n" * 40


def test_a_plain_eval_behind_a_stream_eval_keeps_the_reader_reading(connect):
    # Waiting for the plain eval would wait for the endless one ahead of
    # it: no interrupt would ever be read.
    client = connect()
    client.send(op="eval", id=1, stream=True, code=ENDLESS)
    assert is_out(client.recv(), 1)
    client.send(op="eval", id=2, code='(print "two") 2')
    client.send(op="interrupt", id=3, **{"interrupt-id": 1})
    messages = without_out(client.until(lambda m: m.get("id") == 2), 1)
    assert [(m["id"], m["op"]) for m in messages] == [
        (3, "interrupt-result"),
        (1, "eval-result"),
        (2, "eval-result"),
    ]
    assert messages[1]["error"].endswith("KeyboardInterrupt")
    assert messages[2]["value"] == "2"
    assert messages[2]["stdout"] == "two\n"


BASE_EXCEPTION = "(class Stop [BaseException] (pass))\n(raise (Stop))"


@pytest.mark.parametrize("stream", [False, True])
def test_a_base_exception_is_the_evals_error(connect, stream):
    client = connect()
    client.send(op="eval", id=1, stream=stream, code=BASE_EXCEPTION)
    result = without_out(
        client.until(lambda m: m.get("id") == 1 and not is_out(m, 1)), 1
    )
    assert result[-1]["op"] == "eval-result"
    assert result[-1]["error"].startswith("Traceback (most recent call last):")
    assert result[-1]["error"].endswith("Stop")
    client.send(op="eval", id=2, stream=stream, code="(+ 1 2)")
    after = without_out(client.until(lambda m: m.get("op") == "eval-result"), 2)
    assert after[-1]["value"] == "3"
    assert not isinstance(sys.stdout, tools._NreplRouter)


@pytest.mark.parametrize("req_id", [None, 0, ""])
def test_a_stream_eval_without_an_id_is_refused(connect, req_id):
    client = connect()
    msg = {"op": "eval", "stream": True, "code": '(print "never")'}
    if req_id is not None:
        msg["id"] = req_id
    client.sock.sendall(json.dumps(msg).encode() + b"\n")
    reply = client.recv()
    assert reply["op"] == "error"
    assert "needs an id" in reply["error"]
    client.send(op="ping", id=9)
    assert client.recv() == {"op": "pong", "id": 9}


def test_a_lock_an_interrupt_left_held_is_released(connect, isolated):
    # An interrupt can land right after a `with` acquired a lock, before the
    # block that releases it.  The eval here holds the scope lock on purpose,
    # as such a leak would; once interrupted, another connection's complete,
    # which takes that lock, must be answered.
    isolated["held"] = tools.nrepl_scope_lock(isolated)
    first, second = connect(), connect()
    code = '(import time)\n(.acquire held)\n(while True (print "t") (time.sleep 0.005))'
    first.send(op="eval", id=1, stream=True, code=code)
    assert is_out(first.recv(), 1)
    first.send(op="interrupt", id=2, **{"interrupt-id": 1})
    first.until(lambda m: m.get("op") == "eval-result")
    second.sock.settimeout(3)
    second.send(op="complete", id=3, prefix="pri")
    assert second.recv()["op"] == "complete-result"


CATCH_ONCE = (
    "(import time)\n"
    '(try (while True (print "a") (time.sleep 0.005))\n'
    '  (except [KeyboardInterrupt] (print "caught")))\n'
    '(while True (print "b") (time.sleep 0.005))'
)


def test_an_interrupt_that_raised_nothing_is_never_acknowledged(connect):
    # The eval survives its first interrupt.  A second one inside
    # NREPL_INTERRUPT_REPEAT raises nothing, so it must not claim success;
    # one sent after that window stops the eval.
    import time

    client = connect()
    client.send(op="eval", id=1, stream=True, code=CATCH_ONCE)
    assert is_out(client.recv(), 1)
    client.send(op="interrupt", id=2, **{"interrupt-id": 1})
    assert client.until(lambda m: m.get("id") == 2)[-1] == {
        "op": "interrupt-result",
        "interrupt-id": 1,
        "id": 2,
    }
    time.sleep(0.1)
    client.send(op="interrupt", id=3, **{"interrupt-id": 1})
    refused = client.until(lambda m: m.get("id") == 3)[-1]
    assert refused["op"] == "error"
    assert "was interrupted less than" in refused["error"]
    time.sleep(tools.NREPL_INTERRUPT_REPEAT)
    client.send(op="interrupt", id=4, **{"interrupt-id": 1})
    messages = client.until(lambda m: m.get("op") == "eval-result")
    replies = without_out(messages, 1)
    assert replies[0] == {"op": "interrupt-result", "interrupt-id": 1, "id": 4}
    assert replies[1]["error"].endswith("KeyboardInterrupt")
    # it was the second loop that the last interrupt stopped
    assert "b" in "".join(m["stdout"] for m in messages if is_out(m, 1))


def test_a_plain_eval_queued_behind_a_stream_eval_can_be_cancelled(connect):
    client = connect()
    client.send(op="eval", id=1, stream=True, code=ENDLESS)
    assert is_out(client.recv(), 1)
    client.send(op="eval", id=2, code='(print "never")')
    client.send(op="interrupt", id=3, **{"interrupt-id": 2})
    client.send(op="interrupt", id=4, **{"interrupt-id": 1})
    messages = without_out(client.until(lambda m: m.get("id") == 2), 1)
    assert [(m["id"], m["op"]) for m in messages] == [
        (3, "interrupt-result"),
        (4, "interrupt-result"),
        (1, "eval-result"),
        (2, "eval-result"),
    ]
    assert messages[3]["error"] == "KeyboardInterrupt"
    assert messages[3]["stdout"] == ""


def test_a_plain_eval_running_behind_a_stream_eval_can_be_interrupted(
    connect, isolated
):
    import threading

    isolated["gate"] = gate = threading.Event()
    client = connect()
    client.send(op="eval", id=1, stream=True, code="(gate.wait 10)")
    client.send(op="eval", id=2, code=ENDLESS)  # not waited for: 1 is ahead
    gate.set()
    assert client.recv()["id"] == 1  # the stream eval's result: 2 now runs
    client.send(op="interrupt", id=3, **{"interrupt-id": 2})
    messages = client.until(lambda m: m.get("id") == 2)
    assert [(m["id"], m["op"]) for m in messages] == [
        (3, "interrupt-result"),
        (2, "eval-result"),
    ]
    assert messages[1]["error"].endswith("KeyboardInterrupt")
    assert messages[1]["stdout"].startswith("tick\n")


@pytest.mark.parametrize("method", ["_deliver", "_close_window"])
def test_a_worker_fault_still_sends_one_reply_and_frees_the_id(
    connect, monkeypatch, method
):
    # _deliver failing before it sends, and a fault in the window close
    # (which must not be retried forever), each end the job with one
    # internal-error reply; the id is free again and plain evals are again
    # waited for in order.
    original = getattr(tools._NreplConnection, method)
    failed = []

    def faulty(self, job, *args):
        if not failed:
            failed.append(job)
            raise RuntimeError("boom")
        return original(self, job, *args)

    monkeypatch.setattr(tools._NreplConnection, method, faulty)
    client = connect()
    client.send(op="eval", id=1, stream=True, code="(+ 1 1)")
    [reply] = without_out(client.until(lambda m: m.get("id") == 1), 1)
    assert reply == {
        "op": "error",
        "error": "internal error: RuntimeError: boom",
        "id": 1,
    }
    client.send(op="eval", id=1, stream=True, code="(+ 2 2)")
    assert without_out(client.until(lambda m: m.get("id") == 1), 1)[-1]["value"] == "4"
    client.send(op="eval", id=5, code="(+ 3 3)")
    client.send(op="ping", id=6)
    assert [client.recv()["id"], client.recv()["id"]] == [5, 6]


def test_closing_gives_up_on_an_eval_that_never_stops(isolated, monkeypatch):
    # An eval that swallows the interrupt keeps its worker: the connection
    # still ends after NREPL_CLOSE_TIMEOUT, leaving only that daemon thread.
    import socket
    import threading

    monkeypatch.setattr(tools, "NREPL_CLOSE_TIMEOUT", 0.5)
    isolated["gate"] = gate = threading.Event()
    client_sock, server_sock = socket.socketpair()
    client_sock.settimeout(WAIT)
    handler = threading.Thread(
        target=tools._nrepl_handle_client,
        args=(server_sock, ("test", 0), isolated),
        daemon=True,
    )
    handler.start()
    client = Client(client_sock)
    code = (
        '(print "in")\n'
        "(while (not (gate.is-set))\n"
        "  (try (gate.wait 0.01) (except [KeyboardInterrupt] (pass))))"
    )
    try:
        client.send(op="eval", id=1, stream=True, code=code)
        assert is_out(client.recv(), 1)
        client_sock.close()  # EOF: the server interrupts, the eval ignores it
        handler.join(WAIT)
        assert not handler.is_alive()
        assert server_sock.fileno() == -1  # closed
    finally:
        gate.set()  # lets the eval, and its worker, end


def test_an_eval_replacing_stdout_gets_the_original_back(isolated):
    before = (sys.stdout, sys.stderr)
    code = "(import io)\n(= sys.stdout (io.StringIO))\n(= sys.stderr (io.StringIO))"
    assert tools.nrepl_eval("(import sys)\n" + code, isolated)["error"] is None
    assert (sys.stdout, sys.stderr) == before


def test_a_writer_over_the_cap_waits_for_the_flusher(monkeypatch):
    # Bounded memory: an eval printing faster than the client reads blocks
    # once NREPL_OUT_CAP characters are buffered or being sent.
    import threading

    monkeypatch.setattr(tools, "NREPL_OUT_CAP", 10)
    release = threading.Event()
    sent = []

    def send(data):
        release.wait(WAIT)  # the client is not reading
        sent.append(json.loads(data))

    stream = tools._NreplStream(send, {"id": 1})
    stream.start()
    writer = threading.Thread(
        target=lambda: [stream.write("stdout", "x" * 6 + "\n") for _ in range(3)]
    )
    writer.start()
    writer.join(0.5)
    assert writer.is_alive()  # blocked behind the unsent output
    release.set()
    writer.join(WAIT)
    stream.stop()
    assert not writer.is_alive()
    assert "".join(m["stdout"] for m in sent) == ("x" * 6 + "\n") * 3


def test_a_dead_stream_drops_output_and_never_blocks(monkeypatch):
    # The flusher stops draining once a send failed, so a writer that kept
    # counting toward the cap would wait for it forever.
    import threading

    monkeypatch.setattr(tools, "NREPL_OUT_CAP", 10)

    def send(data):
        raise BrokenPipeError

    stream = tools._NreplStream(send, {"id": 1})
    stream.start()
    writer = threading.Thread(
        target=lambda: [stream.write("stdout", "line\n") for _ in range(100)],
        daemon=True,
    )
    writer.start()
    writer.join(2)
    stream.stop()
    assert not writer.is_alive()
    assert stream.dead


def test_single_character_writes_count_toward_the_cap_as_memory(monkeypatch):
    # 1 MiB of one-character writes would hold ~75 MB of queued objects if
    # the cap counted characters only.
    import threading

    monkeypatch.setattr(tools, "NREPL_OUT_CAP", 1000)
    release = threading.Event()
    stream = tools._NreplStream(lambda data: release.wait(WAIT), {"id": 1})
    stream.start()
    writer = threading.Thread(
        target=lambda: [stream.write("stdout", "x") for _ in range(100)], daemon=True
    )
    writer.start()
    writer.join(0.5)
    blocked = writer.is_alive()  # 100 characters: under the cap by count
    release.set()
    writer.join(WAIT)
    stream.stop()
    assert blocked
    assert not writer.is_alive()


def test_flush_with_nothing_written_queues_nothing():
    stream = tools._NreplStream(lambda data: None, {"id": 1})
    for _ in range(1000):
        stream.flush()
    stream.write("stdout", "x")
    stream.flush()
    stream.flush()
    # one write and one flush marker, never a marker per call
    assert stream._items.qsize() == 2


def test_an_interrupt_traceback_ends_at_the_users_code():
    # An interrupt usually lands inside print's path through the server's
    # output router and stream; those frames say nothing to the user.
    own = {}
    exec(
        compile(
            "def own_frame():\n    raise KeyboardInterrupt\n",
            tools._NREPL_OWN_FILE,
            "exec",
        ),
        own,
    )
    try:
        own["own_frame"]()  # the user's line
    except KeyboardInterrupt as e:
        error = tools._nrepl_format_exception(e)
    assert error.endswith("KeyboardInterrupt")
    assert "test_nrepl.py" in error
    assert tools._NREPL_OWN_FILE not in error


def test_long_output_is_split_into_capped_out_messages(monkeypatch):
    monkeypatch.setattr(tools, "NREPL_OUT_MAX_MESSAGE", 64)
    sent = []
    stream = tools._NreplStream(lambda data: sent.append(json.loads(data)), {"id": 1})
    stream.start()
    stream.write("stdout", "y" * 150)
    stream.stop()
    assert [len(m["stdout"]) for m in sent] == [64, 64, 22]
    assert all(m == {"op": "out", "stdout": m["stdout"], "id": 1} for m in sent)
