"""nREPL connection handler: framing, UTF-8 decoding and error replies.

The handler is driven with a fake connection whose recv returns fixed byte
chunks, so where a read boundary falls is deterministic."""

import json

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
