"""Live per-call progress readout for agent turns (worker 9/20).

During a long multi-call turn (e.g. 460s / 25 calls) the user saw almost
nothing — the turn felt hung. This module emits one cheap status line per
tool call, e.g.::

    Call 12/25 · 45s elapsed · last: read_file aegisscan.py

The readout travels over the existing ``on_status`` callback with a
structured ``progress:`` prefix::

    progress:12:25:45:read_file:aegisscan.py

It never touches the event log, so it cannot spam it. The TUI maps it to
the bottom status line (deduped by ``_set_status``, so at most one extra
invalidation per tool call — far below the existing 10fps stream
throttle); headless mode prints it to stderr via the same callback.

Public API:
    - ``encode(turn_start, call_idx, call_cap, name, args) -> str``
    - ``decode(raw) -> dict | None`` (None for non-progress strings)
    - ``format_status(raw) -> str | None`` — the human-readable line
    - ``wrap_execute(execute, on_status, turn_start, call_cap, base_idx,
      events)`` — wraps an ``Agent._execute_tool``-shaped callable so each
      call emits its progress status right before the real execution.
      Indices are pre-assigned from ``base_idx`` (finished-call count), so
      parallel batches report consecutive numbers even though they run on
      worker threads. Read-only dict access makes the wrapper thread-safe.

``python3 -m fullagent.callprogress`` runs the built-in self-test.
"""

from __future__ import annotations

import threading
import time

PREFIX = "progress:"

# Most informative arg keys, in preference order — the label is what makes
# "read_file aegisscan.py" useful at a glance.
_ARG_KEYS = ("path", "file", "filename", "filepath", "command", "cmd",
             "query", "url", "pattern", "name", "symbol", "target",
             "prompt", "text")
_MAX_LABEL = 48


def short_arg(name: str, args) -> str:
    """Pick a one-line human label for a tool call's args."""
    if not isinstance(args, dict):
        return ""
    for key in _ARG_KEYS:
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            return _clip(v)
    # fallback: first non-empty scalar arg
    for v in args.values():
        if isinstance(v, (str, int, float)) and str(v).strip():
            return _clip(str(v))
    return ""


def _clip(s: str) -> str:
    s = " ".join(s.split())  # collapse newlines/tabs — status is one line
    if len(s) <= _MAX_LABEL:
        return s
    return s[:_MAX_LABEL - 1] + "…"


def encode(turn_start: float, call_idx: int, call_cap: int,
           name: str, args) -> str:
    """Build the wire form ``progress:<idx>:<cap>:<elapsed>:<name>:<label>``."""
    elapsed = max(0, int(time.time() - turn_start))
    label = short_arg(name, args)
    return f"{PREFIX}{call_idx}:{call_cap}:{elapsed}:{name}:{label}"


def decode(raw: str):
    """Parse a ``progress:`` status line; None for anything else."""
    if not isinstance(raw, str) or not raw.startswith(PREFIX):
        return None
    parts = raw[len(PREFIX):].split(":", 4)
    if len(parts) != 5:
        return None
    try:
        idx, cap, elapsed = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    name, label = parts[3], parts[4]
    if not name:
        return None
    return {"call_idx": idx, "call_cap": cap, "elapsed_s": elapsed,
            "name": name, "label": label}


def format_status(raw: str):
    """Human line for a ``progress:`` status; None for other statuses."""
    d = decode(raw)
    if d is None:
        return None
    tail = d["name"] + (f" {d['label']}" if d["label"] else "")
    return (f"Call {d['call_idx']}/{d['call_cap']} · "
            f"{d['elapsed_s']}s elapsed · last: {tail}")


def wrap_execute(execute, on_status, turn_start: float, call_cap: int,
                 base_idx: int, events):
    """Wrap an ``_execute_tool``-shaped callable with progress emission.

    ``base_idx`` is the number of calls already finished this turn; call
    ``i`` of ``events`` is announced as ``base_idx + i + 1``. The wrapper
    keeps the exact ``(ev, approve, on_status, causation_id=...,
    on_tool_output=...)`` call shape that ``dispatch_block`` expects.
    """
    idx_of = {id(ev): base_idx + k + 1 for k, ev in enumerate(events)}

    def _wrapped(ev, approve, on_status_cb,
                 causation_id=None, on_tool_output=None):
        on_status(encode(turn_start, idx_of.get(id(ev), base_idx + 1),
                         call_cap, ev.name, getattr(ev, "args", {})))
        return execute(ev, approve, on_status_cb,
                       causation_id=causation_id,
                       on_tool_output=on_tool_output)

    return _wrapped


# ---------------------------------------------------------------------------
# Self-test — proves progress callbacks fire with correct indices, the
# exact user-facing format, parallel-batch numbering, and that non-progress
# statuses are ignored. Run:  python3 -m fullagent.callprogress
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from types import SimpleNamespace

    _passed = []

    def _check(name, cond):
        assert cond, f"FAILED: {name}"
        _passed.append(name)

    # 1. the exact example shape from the task
    t0 = time.time() - 45
    raw = encode(t0, 12, 25, "read_file", {"path": "aegisscan.py"})
    got = format_status(raw)
    _check("exact-format",
           got is not None and got.startswith(
               "Call 12/25 · 45s elapsed · last: read_file aegisscan.py"))

    # 2. decode round-trip
    d = decode(raw)
    _check("decode",
           d["call_idx"] == 12 and d["call_cap"] == 25
           and d["name"] == "read_file" and d["label"] == "aegisscan.py")

    # 3. non-progress statuses are ignored (no interference with
    #    "thinking" / "tool:" / "running:" handling)
    _check("ignores-others",
           format_status("thinking") is None
           and format_status("tool:read_file") is None
           and format_status("running:background") is None
           and format_status("progress:broken") is None
           and format_status("progress:x:25:1:read_file:p") is None)

    # 4. labels collapse newlines and are clipped (single status line)
    raw2 = encode(t0, 1, 25, "run_command",
                  {"command": "echo hello\n" * 100})
    d2 = decode(raw2)
    _check("label-clipped",
           "\n" not in format_status(raw2) and len(d2["label"]) <= _MAX_LABEL)

    # 5. no-args / empty-args tools still render
    _check("no-args",
           format_status(encode(t0, 3, 25, "list_dir", {}))
           == "Call 3/25 · 45s elapsed · last: list_dir"
           or format_status(encode(t0, 3, 25, "list_dir", {})).startswith(
               "Call 3/25 · 4"))

    # 6. wrap_execute: sequential — indices 1..3 fire in call order,
    #    each BEFORE its real execution
    seen = []
    evs = [SimpleNamespace(name=n, args={"path": f"f{i}.py"})
           for i, n in enumerate(["read_file", "edit_file", "run_command"])]

    def _fake_exec(ev, approve, on_status_cb,
                   causation_id=None, on_tool_output=None):
        seen.append(("exec", ev.name))
        return None

    w = wrap_execute(_fake_exec, lambda s: seen.append(("status", s)),
                     t0, 25, 0, evs)
    for ev in evs:
        w(ev, None, lambda s: None)
    statuses = [s for k, s in seen if k == "status"]
    _check("sequential-indices",
           [decode(s)["call_idx"] for s in statuses] == [1, 2, 3])
    _check("status-before-exec",
           [k for k, _ in seen]
           == ["status", "exec"] * 3)
    _check("sequential-labels",
           [decode(s)["label"] for s in statuses]
           == ["f0.py", "f1.py", "f2.py"])

    # 7. wrap_execute: parallel threads — every index fires exactly once,
    #    base offset continues numbering across blocks
    seen2 = []
    _lock = threading.Lock()

    def _collect(s):
        with _lock:
            seen2.append(s)

    evs2 = [SimpleNamespace(name="read_file", args={"path": f"p{i}.py"})
            for i in range(4)]

    def _slow_exec(ev, approve, on_status_cb,
                   causation_id=None, on_tool_output=None):
        time.sleep(0.01)
        return None

    w2 = wrap_execute(_slow_exec, _collect, t0, 25, 5, evs2)
    ths = [threading.Thread(target=w2, args=(ev, None, lambda s: None))
           for ev in evs2]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    _check("parallel-indices",
           sorted(decode(s)["call_idx"] for s in seen2) == [6, 7, 8, 9])
    _check("parallel-caps",
           all(decode(s)["call_cap"] == 25 for s in seen2))

    # 8. elapsed is measured from the turn start
    t1 = time.time() - 125
    _check("elapsed",
           decode(encode(t1, 1, 25, "read_file", {}))["elapsed_s"] >= 124)

    print(f"PASS — callprogress self-test: {len(_passed)} checks "
          f"({', '.join(_passed)})")
