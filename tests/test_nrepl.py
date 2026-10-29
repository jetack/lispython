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
    replies = serve(
        [b"x" * 60, b"x" * 60, b"x" * 60, b"x\n" + line(op="ping", id=4)]
    )
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


def dispatch(scope, **msg):
    return tools._nrepl_dispatch({"op": "eval", **msg}, scope)


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
