"""Timeout fail-fast tests for fullagent.client.

Proves a hung model endpoint fails fast with a clear APITimeoutError
instead of hanging forever:

* a black-hole endpoint (accepts the TCP connection, then never responds)
  fails within ~`timeout` seconds on the NON-streaming path — the old code
  retried with a fresh full budget per attempt (MAX_RETRIES x 300s = 900s+
  of silence, matching the reported 512s+ hangs);
* a stalled SSE stream (headers sent, then silence) is aborted by the
  60s stall watchdog on the streaming path;
* a healthy stream still completes (regression guard for the watchdog).

Run:  python3 -m pytest tests/test_client_timeout.py -x -q
   or: python3 -m unittest tests.test_client_timeout -v
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)


import socket
import threading
import time
import unittest

from fullagent import client
# NOTE (weird CPython 3.12.3 quirk, verified repeatedly): a multi-name
# from-import with APITimeoutError FIRST —
#   from fullagent.client import (APItimeoutError, APIError)
# raises "ImportError: cannot import name 'APItimeoutError'" even though
# the name exists on the module. Any other order, a single name, an
# alias, or attribute access all work. So APITimeoutError is not first.
from fullagent.client import (APIError, APITimeoutError, chat_blocking,
                              chat_stream)
from fullagent.config import Effort, Model, Provider


def _serve(handler):
    """Run a raw-socket HTTP server on 127.0.0.1:0; return (socket, port).

    The handler gets the accepted connection after the request headers
    have been read. Daemon threads: nothing to join on teardown.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]

    def _read_headers(conn):
        conn.settimeout(10)
        data = b""
        try:
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            if b"\r\n\r\n" in data:
                head, rest = data.split(b"\r\n\r\n", 1)
                cl = 0
                for line in head.split(b"\r\n")[1:]:
                    if line.lower().startswith(b"content-length:"):
                        try:
                            cl = int(line.split(b":", 1)[1].strip())
                        except ValueError:
                            cl = 0
                while len(rest) < cl:
                    chunk = conn.recv(min(65536, cl - len(rest)))
                    if not chunk:
                        break
                    rest += chunk
        except OSError:
            pass

    def _loop():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            _read_headers(conn)
            t = threading.Thread(target=handler, args=(conn,), daemon=True)
            t.start()

    threading.Thread(target=_loop, daemon=True).start()
    return srv, port


def _hang_forever(conn):
    """Black hole: accept the request, then never respond at all."""
    try:
        time.sleep(3600)
    except OSError:
        pass
    finally:
        conn.close()


def _stalled_sse(conn):
    """Send SSE headers, then never send another byte."""
    try:
        conn.sendall(b"HTTP/1.1 200 OK\r\n"
                     b"Content-Type: text/event-stream\r\n"
                     b"Cache-Control: no-cache\r\n"
                     b"Connection: keep-alive\r\n\r\n")
        time.sleep(3600)
    except OSError:
        pass
    finally:
        conn.close()


def _healthy_sse(conn):
    """One keepalive comment, one event, then [DONE] — must complete."""
    try:
        conn.sendall(b"HTTP/1.1 200 OK\r\n"
                     b"Content-Type: text/event-stream\r\n"
                     b"Connection: close\r\n\r\n"
                     b": ping\n\n"
                     b'data: {"model":"t","choices":[{"delta":{"content":"hi"},'
                     b'"finish_reason":"stop"}]}\n\n'
                     b"data: [DONE]\n\n")
    except OSError:
        pass
    finally:
        conn.close()


def _mk(base_url):
    prov = Provider(key="t", name="t", base_url=base_url, api_key="dummy",
                    color="red")
    mod = Model(id="t", provider="t", label="t")
    eff = Effort(key="t", label="T", color="red", max_tokens=None,
                 temperature=0.0, reasoning_effort=None, description="t")
    return prov, mod, eff


MSGS = [{"role": "user", "content": "hi"}]


class TimeoutTests(unittest.TestCase):
    def test_black_hole_blocking_fails_fast(self):
        """Hung endpoint, non-streaming: APITimeoutError in ~timeout,
        not MAX_RETRIES x DEFAULT_TIMEOUT (900s+) of silence."""
        srv, port = _serve(_hang_forever)
        self.addCleanup(srv.close)
        prov, mod, eff = _mk(f"http://127.0.0.1:{port}")
        t0 = time.monotonic()
        with self.assertRaises(APITimeoutError) as cm:
            chat_blocking(prov, mod, eff, MSGS, None, timeout=6)
        dt = time.monotonic() - t0
        print(f"\nblack-hole blocking: APITimeoutError after {dt:.1f}s "
              f"(old worst case: {client.MAX_RETRIES} x "
              f"{__import__('fullagent.config', fromlist=['DEFAULT_TIMEOUT']).DEFAULT_TIMEOUT:g}s+)"
              f" \u2014 {cm.exception}")
        # ~6s read budget + epsilon; old code needed 900s+ to surface
        self.assertLess(dt, 60, f"took too long: {dt:.1f}s")

    def test_stalled_sse_stream_aborts(self):
        """SSE headers then silence: the stall watchdog aborts the stream
        instead of waiting out the full read budget."""
        srv, port = _serve(_stalled_sse)
        self.addCleanup(srv.close)
        prov, mod, eff = _mk(f"http://127.0.0.1:{port}")
        old = client.STREAM_STALL_TIMEOUT
        client.STREAM_STALL_TIMEOUT = 2.0
        self.addCleanup(setattr, client, "STREAM_STALL_TIMEOUT", old)
        t0 = time.monotonic()
        with self.assertRaises(APITimeoutError) as cm:
            chat_stream(prov, mod, eff, MSGS, None, timeout=30)
        dt = time.monotonic() - t0
        print(f"\nstalled SSE: APITimeoutError after {dt:.1f}s \u2014 {cm.exception}")
        self.assertLess(dt, 20, f"watchdog did not fire: {dt:.1f}s")

    def test_healthy_stream_still_completes(self):
        """Regression: a normal stream (with a keepalive comment) still
        completes through the watchdog path."""
        srv, port = _serve(_healthy_sse)
        self.addCleanup(srv.close)
        prov, mod, eff = _mk(f"http://127.0.0.1:{port}")
        t0 = time.monotonic()
        res = chat_stream(prov, mod, eff, MSGS, None, timeout=30)
        dt = time.monotonic() - t0
        self.assertEqual(res.content, "hi")
        self.assertLess(dt, 30)
        print(f"\nhealthy SSE: content={res.content!r} in {dt:.1f}s")

    def test_timeout_error_is_api_error(self):
        """APITimeoutError subclasses APIError: existing handlers keep
        working; only the type/message got clearer."""
        self.assertTrue(issubclass(APITimeoutError, APIError))


if __name__ == "__main__":
    unittest.main(verbosity=2)
