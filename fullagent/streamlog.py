"""Streaming logs — append-mode log writer + tail/list/follow tools.

Infrastructure for long-running commands (jobs.py / bgsh.py) that produce
huge output. The design goal: never hold full output in RAM.

Public API::

    from .streamlog import StreamLogger

* :class:`StreamLogger` -- opens ``~/.fullagent/logs/{stream_id}.log`` in
  append mode; :meth:`write` flushes immediately. Rotates to ``.1`` when the
  log exceeds ``max_bytes`` (default 50MB, one backup kept).
* :func:`tail_lines(stream_id, n)` -- last N lines via seek-from-end, no
  full read for large files.
* :func:`list_logs()` -- all log files with size + mtime.
* :func:`follow(stream_id, timeout)` -- blocks up to ``timeout`` seconds,
  returns lines appended since the previous call (per-stream read offsets
  kept in memory).
* :func:`register` -- attaches ``LogTail`` / ``LogList`` / ``LogFollow``
  tools to an agent.

``python3 -m fullagent.streamlog`` runs the built-in self-test.
"""

from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .tools import Tool

LOG_DIR = os.path.join(os.environ.get("FULLAGENT_HOME") or os.path.join(os.path.expanduser("~"), ".fullagent"), "logs")
DEFAULT_MAX_BYTES = 50 * 1024 * 1024  # 50MB rotation threshold

_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_stream_id(stream_id: Optional[str]) -> str:
    sid = (stream_id or "default").strip() or "default"
    return _SAFE_RE.sub("_", sid)[:64]


def _log_path(stream_id: str) -> str:
    """Absolute path of a stream's log file (sanitized)."""
    return os.path.join(LOG_DIR, _safe_stream_id(stream_id) + ".log")


# ---------------------------------------------------------------------------
# StreamLogger
# ---------------------------------------------------------------------------
class StreamLogger:
    """Append-mode streaming log writer.

    Opens ``~/.fullagent/logs/{stream_id}.log`` once and keeps the handle
    open; :meth:`write` appends and flushes immediately so a crashed reader
    never loses buffered output. Full output is never held in RAM.
    """

    def __init__(self, stream_id: str,
                 max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.stream_id = _safe_stream_id(stream_id)
        self.max_bytes = max_bytes
        self.path = _log_path(self.stream_id)
        os.makedirs(LOG_DIR, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8", errors="replace")

    def write(self, line: str) -> None:
        """Append one line (newline added if missing) and flush immediately."""
        if not line.endswith("\n"):
            line += "\n"
        data = line.encode("utf-8", errors="replace")
        with self._lock:
            try:
                if (os.path.getsize(self.path) + len(data)
                        >= self.max_bytes):
                    self._rotate_locked()
            except OSError:
                pass
            self._fh.write(line)
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                pass

    def _rotate_locked(self) -> None:
        """Rotate current log to ``.1``, keeping one backup."""
        self._fh.close()
        backup = self.path + ".1"
        try:
            if os.path.exists(backup):
                os.remove(backup)
            os.replace(self.path, backup)
        except OSError:
            pass
        self._fh = open(self.path, "a", encoding="utf-8", errors="replace")

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except OSError:
                pass

    def __enter__(self) -> "StreamLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Read helpers (module-level, agent-free — reusable from jobs.py / bgsh.py)
# ---------------------------------------------------------------------------
def tail_lines(stream_id: str, n: int = 50) -> List[str]:
    """Return the last ``n`` lines of a log via seek-from-end.

    Never reads the whole file: walks backwards in blocks until ``n``
    newlines are found. Fast even for multi-hundred-MB logs.
    """
    n = max(1, int(n))
    path = _log_path(stream_id)
    if not os.path.exists(path):
        return []
    lines: List[str] = []
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        buf = b""
        block = 64 * 1024
        pos = size
        newlines = 0
        while pos > 0 and newlines <= n:
            step = min(block, pos)
            pos -= step
            fh.seek(pos)
            chunk = fh.read(step)
            buf = chunk + buf
            newlines = buf.count(b"\n")
        lines = buf.split(b"\n")
    text = [ln.decode("utf-8", errors="replace") for ln in lines]
    if text and text[-1] == "":
        text.pop()  # trailing newline of last line
    return text[-n:]


def list_logs() -> List[Dict[str, Any]]:
    """All ``*.log`` files in LOG_DIR with size + mtime, newest first."""
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(LOG_DIR):
        return out
    for name in os.listdir(LOG_DIR):
        if not name.endswith(".log"):
            continue
        p = os.path.join(LOG_DIR, name)
        try:
            st = os.stat(p)
        except OSError:
            continue
        out.append({
            "stream_id": name[:-4],
            "path": p,
            "size_bytes": st.st_size,
            "mtime": datetime.fromtimestamp(
                st.st_mtime, tz=timezone.utc).isoformat(),
        })
    out.sort(key=lambda d: d["mtime"], reverse=True)
    return out


# Per-stream read offsets for follow(): stream_id -> byte offset.
_follow_offsets: Dict[str, int] = {}
_follow_lock = threading.Lock()


def follow(stream_id: str, timeout: float = 5.0) -> List[str]:
    """Block up to ``timeout`` seconds, returning lines appended to the log
    since the previous ``follow`` call for this stream.

    First call for a stream starts at EOF (``tail -f`` semantics). If the
    log shrank (rotation), the offset resets to 0.
    """
    sid = _safe_stream_id(stream_id)
    path = _log_path(sid)
    if not os.path.exists(path):
        return []
    with _follow_lock:
        start = _follow_offsets.get(sid)
        if start is None:
            try:
                start = os.path.getsize(path)
            except OSError:
                start = 0
    deadline = time.monotonic() + max(0.0, float(timeout))
    result: List[str] = []
    while True:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        if size < start:
            start = 0  # rotated / truncated
        if size > start:
            with open(path, "rb") as fh:
                fh.seek(start)
                chunk = fh.read(size - start)
            result.extend(chunk.decode("utf-8",
                                       errors="replace").splitlines())
            start = size
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    with _follow_lock:
        _follow_offsets[sid] = start
    return result


def reset_follow_offset(stream_id: str, offset: int = 0) -> None:
    """Reset a stream's follow offset (useful for tests / re-follow)."""
    with _follow_lock:
        _follow_offsets[_safe_stream_id(stream_id)] = offset


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n:.1f}GB"


def make_log_tail_tool() -> Tool:
    def handler(stream_id: str, lines: int = 50) -> str:
        got = tail_lines(stream_id, lines)
        if not got:
            return f"No log found for stream '{_safe_stream_id(stream_id)}'."
        return "\n".join(got)

    return Tool(
        name="LogTail",
        description=(
            "Read the last N lines of a streaming log (seek-based, fast "
            "even for huge logs). Use for checking long-running command "
            "output without loading the whole file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "stream_id": {"type": "string",
                             "description": "Log stream id"},
                "lines": {"type": "integer", "default": 50,
                          "description": "How many trailing lines to return"},
            },
            "required": ["stream_id"],
        },
        handler=handler,
    )


def make_log_list_tool() -> Tool:
    def handler() -> str:
        logs = list_logs()
        if not logs:
            return "No streaming logs yet."
        rows = [f"{d['stream_id']}: {_fmt_size(d['size_bytes'])}, "
                f"modified {d['mtime']}" for d in logs]
        return "\n".join(rows)

    return Tool(
        name="LogList",
        description="List all streaming log files with size and mtime.",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )


def make_log_follow_tool() -> Tool:
    def handler(stream_id: str, timeout: float = 5.0) -> str:
        got = follow(stream_id, timeout)
        if not got:
            return (f"No new lines in stream '{_safe_stream_id(stream_id)}' "
                    f"within {timeout}s.")
        return "\n".join(got)

    return Tool(
        name="LogFollow",
        description=(
            "Block up to `timeout` seconds and return new log lines "
            "appended since the last LogFollow call for this stream "
            "(like tail -f)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "stream_id": {"type": "string",
                             "description": "Log stream id"},
                "timeout": {"type": "number", "default": 5.0,
                            "description": "Max seconds to wait"},
            },
            "required": ["stream_id"],
        },
        handler=handler,
    )


def register(agent: Any) -> None:
    """Attach LogTail / LogList / LogFollow tools to an agent (duck-typed)."""
    agent.tools["LogTail"] = make_log_tail_tool()
    agent.tools["LogList"] = make_log_list_tool()
    agent.tools["LogFollow"] = make_log_follow_tool()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    # Isolate the log dir so the self-test never touches ~/.fullagent.
    tmp = tempfile.mkdtemp(prefix="streamlog_selftest_")
    LOG_DIR = tmp

    class FakeAgent:
        def __init__(self):
            self.session_id = "selftest-session"
            self.tools = {}

    a = FakeAgent()
    register(a)
    assert set(["LogTail", "LogList", "LogFollow"]) <= set(a.tools), a.tools

    # 1) Write 100k lines via StreamLogger; verify file on disk.
    N = 100_000
    with StreamLogger("big") as lg:
        for i in range(N):
            lg.write(f"line-{i:06d}")
    path = _log_path("big")
    assert os.path.exists(path), path
    size = os.path.getsize(path)
    print(f"wrote {N} lines -> {path} ({_fmt_size(size)})")
    assert size > 1_000_000, size

    # 2) LogTail returns correct last N lines WITHOUT loading whole file.
    t0 = time.monotonic()
    last = tail_lines("big", 10)
    dt = time.monotonic() - t0
    assert last == [f"line-{i:06d}" for i in range(N - 10, N)], last[:3]
    print(f"tail_lines(10) on {N}-line file took {dt * 1000:.1f}ms "
          f"(fast seek-based tail)")
    assert dt < 2.0, f"tail too slow: {dt}s"
    tool_out = a.tools["LogTail"].handler(stream_id="big", lines=3)
    assert tool_out.endswith(f"line-{N - 1:06d}"), tool_out

    # 3) LogList shows the log with size + mtime.
    lst = a.tools["LogList"].handler()
    assert "big" in lst and "MB" in lst, lst
    print("LogList:", lst.splitlines()[0])

    # 4) LogFollow picks up new lines written after the offset.
    reset_follow_offset("big", os.path.getsize(path))  # at EOF
    with StreamLogger("big") as lg:
        lg.write("follow-one")
        lg.write("follow-two")
    got = follow("big", timeout=2.0)
    assert got == ["follow-one", "follow-two"], got
    tool_out = a.tools["LogFollow"].handler(stream_id="big", timeout=0.5)
    assert "No new lines" in tool_out, tool_out  # offset now at EOF
    print("LogFollow picked up 2 new lines after offset")

    # 5) Rotation triggers at the size limit (small limit via parameter).
    with StreamLogger("rot", max_bytes=1024) as lg:
        for i in range(500):
            lg.write(f"rot-line-{i:04d}-xxxxxxxxxxxxxxxxxxxxxxxx")
    rp, rb = _log_path("rot"), _log_path("rot") + ".1"
    assert os.path.exists(rb), "backup .1 missing"
    assert os.path.getsize(rp) < 1024 * 4, os.path.getsize(rp)
    assert not os.path.exists(rb + ".1"), "more than one backup kept"
    print(f"rotation ok: {rp} ({_fmt_size(os.path.getsize(rp))}), "
          f"backup {rb} ({_fmt_size(os.path.getsize(rb))})")

    # 6) follow() survives rotation (file shrinks -> offset resets).
    reset_follow_offset("rot", 10 ** 9)
    got = follow("rot", timeout=0.5)
    assert got, "follow after rotation returned nothing"
    print("follow after rotation ok")

    print("PASS: streamlog self-test (write/tail/list/follow/rotation)")
