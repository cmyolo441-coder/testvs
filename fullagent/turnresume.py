"""Long-turn resume — REAL crash recovery for in-flight turns.

A turn can die mid-flight: the process is killed (OOM, SIGKILL, machine
reboot, ``kill -9``) while tools are still running. This module persists
turn state incrementally so a fresh process can see what was in flight
and offer to pick it back up.

How it works:

* ``~/.fullagent/turns/<session_id>.json`` holds the state of the current
  turn. Every write is **atomic** (write to a temp file in the same
  directory, ``fsync``, then ``os.replace``), so a crash mid-write can
  never leave a half-written file behind.
* The state records ``pid``. On startup, ``in_progress`` files whose pid
  is **dead** are reported as incomplete (crashed) turns; files whose pid
  is still **alive** belong to a running process and are left alone.
  ``agent.incomplete_turns`` exposes the dead ones.
* ``register()`` wraps ``agent.run_turn`` (original saved, restored-safe)
  and ``agent._execute_tool`` so state is written at turn start, after
  every completed tool call, and at turn end ("completed").
* The ``TurnResume`` tool lists incomplete turns; the model must ask the
  user to confirm before resuming, because resuming re-injects saved
  messages/tool history into the live conversation.

Public API:

* :func:`register` — hook into an agent (duck-typed).
* :func:`load_incomplete_turns` — scan the turns dir for dead
  in-progress turns.
* :func:`is_pid_alive` — crash-vs-active discriminator.
* :func:`atomic_write_json` — temp+rename crash-safe write.

``python3 -m fullagent.turnresume`` runs the built-in self-test.
"""

from __future__ import annotations

import functools
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

TURNS_DIRNAME = "turns"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_INTERRUPTED = "interrupted"   # Esc / TurnCancelled — clean stop
STATUS_RESUMED = "resumed"           # a later session picked it back up

# any of these, from a DEAD process, count as "incomplete"
INCOMPLETE_STATUSES = frozenset({
    STATUS_IN_PROGRESS, STATUS_INTERRUPTED})

_MSG_CONTENT_CAP = 4000   # per-message content cap in the snapshot
# SOVEREIGN ENFORCEMENT: the sealed system prompt (49k+ master document)
# must never be clipped in the canonical snapshot — a 4k fragment would
# corrupt every subsequent wire build.
_SYSTEM_CONTENT_CAP = 300_000
_MAX_MESSAGES = 200       # keep system prompt (idx 0) + tail
_RESULT_CAP = 2000        # per tool-call result cap
_ARGS_CAP = 4000          # per tool-call args JSON cap


def turns_dir() -> Path:
    """~/.fullagent/turns — created on demand."""
    base = Path(os.environ["FULLAGENT_HOME"]).expanduser() if os.environ.get("FULLAGENT_HOME") else (Path.home() / ".fullagent")
    d = base / TURNS_DIRNAME
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return d


def turn_path(session_id: str) -> Path:
    safe = "".join(c for c in str(session_id) if c.isalnum() or c in "-_")
    return turns_dir() / f"{safe or 'unknown'}.json"


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically: temp file + fsync + os.replace.

    A crash at ANY point leaves either the old file or the new file
    intact — never a partial write.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def is_pid_alive(pid: Any) -> bool:
    """True iff a process with *pid* is currently running.

    ``os.kill(pid, 0)`` raises ProcessLookupError when the pid is dead and
    PermissionError when it exists but belongs to another user — both are
    authoritative. A pid of 0/negative/None is never "alive".

    Caveat: OS pid reuse can theoretically make a recycled pid look alive;
    the stored ``started_at`` + ``session_id`` let a human disambiguate.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# state model
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clip(text: Any, limit: int) -> str:
    s = text if isinstance(text, str) else json.dumps(text, default=str,
                                                      ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + f"…[{len(s) - limit} truncated]"


def _snapshot_messages(messages: list) -> list:
    """Compact, JSON-safe snapshot of the message list.

    Keeps the system prompt (index 0 — the sealed prompt must survive a
    resume) plus the most recent tail; caps long contents so the state
    file stays small even in huge conversations.
    """
    snap = []
    kept = messages if len(messages) <= _MAX_MESSAGES else (
        [messages[0]] + messages[-(_MAX_MESSAGES - 1):])
    for m in kept:
        try:
            _is_system = m.get("role") == "system"
            snap.append({
                "role": m.get("role", "?"),
                "content": _clip(
                    m.get("content", ""),
                    _SYSTEM_CONTENT_CAP if _is_system else _MSG_CONTENT_CAP),
                "tool_calls": m.get("tool_calls", []),
                "tool_call_id": m.get("tool_call_id"),
            })
        except Exception:
            snap.append({"role": "?", "content": "[unsnapshotable message]"})
    return snap


def _snapshot_tool_call(ev: Any) -> dict:
    try:
        return {
            "name": getattr(ev, "name", "?"),
            "args": _clip(getattr(ev, "args", {}), _ARGS_CAP),
            "result": _clip(getattr(ev, "result", ""), _RESULT_CAP),
            "status": getattr(ev, "status", "?"),
            "duration": round(float(getattr(ev, "duration", 0.0) or 0.0), 3),
        }
    except Exception:
        return {"name": "?", "status": "?"}


def load_incomplete_turns(exclude_session: str | None = None) -> list[dict]:
    """Scan the turns dir for turns a dead process left behind.

    Returns one dict per dead in-progress turn: session_id, pid,
    started_at, user_text preview, n_tool_calls. Files that are corrupt,
    already completed, or owned by a live process are skipped.
    """
    out: list[dict] = []
    d = turns_dir()
    if not d.is_dir():
        return out
    for path in sorted(d.glob("*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue  # corrupt file — never trust it, never crash on it
        if not isinstance(state, dict):
            continue
        if state.get("status") not in INCOMPLETE_STATUSES:
            continue
        sid = state.get("session_id") or path.stem
        if exclude_session and sid == exclude_session:
            continue
        if is_pid_alive(state.get("pid")):
            continue  # still running — belongs to a live process
        out.append({
            "session_id": sid,
            "pid": state.get("pid"),
            "started_at": state.get("started_at", "?"),
            "user_text": str(state.get("user_text", ""))[:200],
            "n_tool_calls": len(state.get("tool_calls_done", []) or []),
            "path": str(path),
        })
    return out


def _read_state(session_id: str) -> dict | None:
    p = turn_path(session_id)
    try:
        state = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def mark_status(session_id: str, status: str) -> None:
    """Best-effort status flip on the existing state file (keeps payload)."""
    state = _read_state(session_id) or {}
    state["status"] = status
    state["ended_at"] = _now_iso()
    try:
        atomic_write_json(turn_path(session_id), state)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# per-turn tracker (drives the run_turn / _execute_tool wrappers)
# ---------------------------------------------------------------------------

class _TurnTracker:
    """Holds the live state of the current turn; writes it incrementally."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.active = False
        self.state: dict = {}

    def begin(self, user_text: str) -> None:
        sid = str(getattr(self.agent, "session_id", "unknown"))
        self.state = {
            "schema": 1,
            "session_id": sid,
            "pid": os.getpid(),
            "status": STATUS_IN_PROGRESS,
            "started_at": _now_iso(),
            "user_text": user_text,
            "messages_so_far": _snapshot_messages(
                list(getattr(self.agent, "messages", []) or [])),
            "tool_calls_done": [],
        }
        self.active = True
        self._write()

    def record_tool(self, ev: Any) -> None:
        if not self.active:
            return
        try:
            self.state["tool_calls_done"].append(_snapshot_tool_call(ev))
            self.state["messages_so_far"] = _snapshot_messages(
                list(getattr(self.agent, "messages", []) or []))
            self._write()
        except Exception:
            pass  # persistence must never break a turn

    def end(self, status: str) -> None:
        if not self.active:
            return
        self.active = False
        try:
            self.state["status"] = status
            self.state["ended_at"] = _now_iso()
            self.state["messages_so_far"] = _snapshot_messages(
                list(getattr(self.agent, "messages", []) or []))
            self._write()
        except Exception:
            pass

    def _write(self) -> None:
        try:
            atomic_write_json(turn_path(str(self.state.get("session_id"))),
                              self.state)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# resume (TurnResume tool + shared implementation)
# ---------------------------------------------------------------------------

def _find_incomplete(agent: Any, session_id: str) -> dict | None:
    for t in (getattr(agent, "incomplete_turns", None) or []):
        if t.get("session_id") == session_id:
            return t
    return None


def list_incomplete(agent: Any) -> str:
    turns = getattr(agent, "incomplete_turns", None) or []
    if not turns:
        return "no incomplete turns — nothing crashed mid-turn"
    lines = [f"{len(turns)} incomplete turn(s) from crashed/interrupted "
             "sessions:"]
    for t in turns:
        lines.append(
            f"  • session {t['session_id']} (pid {t['pid']}, "
            f"started {t['started_at']}, {t['n_tool_calls']} tool call(s))")
        if t["user_text"]:
            lines.append(f"    user: {t['user_text'][:120]}")
    lines.append("To resume one, confirm with the user first, then call "
                 "TurnResume(action=\"resume\", session_id=\"<id>\").")
    return "\n".join(lines)


def resume_turn(agent: Any, session_id: str) -> str:
    """Restore a dead turn's messages + tool history into this agent.

    The caller (TurnResume tool handler) is responsible for getting the
    user's confirmation first — this function does the mechanical restore.
    """
    meta = _find_incomplete(agent, session_id)
    if meta is None:
        return (f"ERROR: no incomplete turn for session '{session_id}'. "
                "Run TurnResume(action=\"list\") to see candidates.")
    state = _read_state(session_id)
    if state is None or not isinstance(state.get("messages_so_far"), list):
        return (f"ERROR: turn state for '{session_id}' is missing or "
                "corrupt — cannot resume.")
    if is_pid_alive(state.get("pid")):
        return (f"ERROR: turn for '{session_id}' belongs to a live process "
                f"(pid {state.get('pid')}) — refusing to resume.")

    restored_msgs = [
        m for m in state["messages_so_far"] if isinstance(m, dict)]
    tool_hist = state.get("tool_calls_done", []) or []

    # rebuild a minimal Turn record so history views stay coherent
    try:
        from .agent import Turn  # local import: avoid module cycle
    except Exception:
        Turn = None  # type: ignore[assignment]
    if Turn is not None:
        try:
            t = Turn(user_text=state.get("user_text", ""),
                     model_id=getattr(getattr(agent, "cfg", None),
                                      "model_id", ""))
            for tc in tool_hist:
                ev = type("ResumedToolCall", (), {})()
                ev.name = tc.get("name", "?")
                ev.args = tc.get("args", "")
                ev.result = tc.get("result", "")
                ev.status = tc.get("status", "done")
                ev.duration = tc.get("duration", 0.0)
                t.tools.append(ev)
            agent.turns.append(t)
        except Exception:
            pass

    agent.messages = restored_msgs
    agent.incomplete_turns = [t for t in (agent.incomplete_turns or [])
                              if t.get("session_id") != session_id]
    mark_status(session_id, STATUS_RESUMED)
    try:
        agent.log.append("turn.resumed",
                         {"from_session": session_id,
                          "messages_restored": len(restored_msgs),
                          "tool_calls_restored": len(tool_hist)},
                         actor="system")
    except Exception:
        pass
    return (f"✓ resumed turn from session {session_id}: restored "
            f"{len(restored_msgs)} message(s) and {len(tool_hist)} "
            f"tool call(s). Continue the conversation from here.")


def _tool_handler(action: str = "list", session_id: str = "",
                  _agent: Any = None) -> str:
    if action == "resume":
        if not session_id:
            return "ERROR: resume needs session_id"
        return resume_turn(_agent, session_id)
    return list_incomplete(_agent)


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------

def register(agent: Any) -> None:
    """Hook long-turn resume into an agent (duck-typed).

    * sets ``agent.incomplete_turns`` from dead in-progress turn files
    * wraps ``agent.run_turn`` to persist turn state incrementally
    * wraps ``agent._execute_tool`` to snapshot after each tool call
    * registers the ``TurnResume`` tool
    """
    # 1. crash detection at startup — only dead processes count
    try:
        agent.incomplete_turns = load_incomplete_turns(
            exclude_session=str(getattr(agent, "session_id", "")))
    except Exception:
        agent.incomplete_turns = []

    tracker = _TurnTracker(agent)
    agent.turnresume_tracker = tracker

    # 2. hook the turn lifecycle — wrap run_turn (original preserved)
    orig_run_turn = agent.run_turn

    @functools.wraps(orig_run_turn)
    def run_turn_with_persistence(user_text: str, *args: Any,
                                  **kwargs: Any) -> Any:
        tracker.begin(user_text)
        try:
            result = orig_run_turn(user_text, *args, **kwargs)
        except KeyboardInterrupt:
            tracker.end(STATUS_INTERRUPTED)
            raise
        except Exception as e:
            # TurnCancelled (Esc) is a clean stop, not a crash
            if type(e).__name__ == "TurnCancelled":
                tracker.end(STATUS_INTERRUPTED)
            else:
                # unexpected error: leave in_progress ONLY if the process
                # itself is dying is impossible to know here — mark it
                # interrupted so it stays resumable but is honest
                tracker.end(STATUS_INTERRUPTED)
            raise
        tracker.end(STATUS_COMPLETED)
        return result

    # only wrap once (register is idempotent)
    if not getattr(orig_run_turn, "_turnresume_wrapped", False):
        run_turn_with_persistence._turnresume_wrapped = True  # type: ignore[attr-defined]
        agent.run_turn = run_turn_with_persistence

    # 3. snapshot after every completed tool call
    orig_execute = getattr(agent, "_execute_tool", None)
    if orig_execute is not None and not getattr(
            orig_execute, "_turnresume_wrapped", False):

        @functools.wraps(orig_execute)
        def execute_with_snapshot(ev: Any, *args: Any,
                                  **kwargs: Any) -> Any:
            try:
                return orig_execute(ev, *args, **kwargs)
            finally:
                try:
                    tracker.record_tool(ev)
                except Exception:
                    pass

        execute_with_snapshot._turnresume_wrapped = True  # type: ignore[attr-defined]
        agent._execute_tool = execute_with_snapshot

    # 4. the TurnResume tool (model must confirm with the user first)
    try:
        from .tools import Tool
        agent.tools["TurnResume"] = Tool(
            "TurnResume",
            "List and resume turns that crashed mid-flight in a previous "
            "session. Actions: list (default) shows incomplete turns with "
            "session ids; resume restores a turn's messages and tool "
            "history so it can continue — ALWAYS ask the user to confirm "
            "before resuming. Args: action ('list'|'resume'), "
            "session_id (for resume).",
            {"type": "object", "properties": {
                "action": {"type": "string",
                           "enum": ["list", "resume"]},
                "session_id": {"type": "string"}},
             "additionalProperties": False},
            lambda action="list", session_id="", **_: _tool_handler(
                action, session_id, agent))
    except Exception:
        pass

    try:
        agent.turnresume_supported = True
        agent.log.append("feature.registered", {"module": "turnresume"},
                         actor="system")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# self-test: `python3 -m fullagent.turnresume` → PASS
# ---------------------------------------------------------------------------

def _selftest() -> None:
    import tempfile

    print("turnresume self-test")
    tmp = Path(tempfile.mkdtemp(prefix="turnresume_selftest_")).resolve()
    global turns_dir
    real_turns_dir = turns_dir
    turns_dir = lambda: tmp  # noqa: E731 — redirect storage for the test
    try:
        # 1. atomic write: file is valid JSON at every instant
        p = tmp / "a.json"
        atomic_write_json(p, {"status": "in_progress", "n": 1})
        atomic_write_json(p, {"status": "completed", "n": 2})
        assert json.loads(p.read_text())["n"] == 2
        assert not list(tmp.glob(".*.tmp")), "temp file leaked"
        print("  [ok] atomic write (no partial file, no tmp leak)")

        # 2. pid liveness discriminator
        assert is_pid_alive(os.getpid()) is True
        assert is_pid_alive(1) in (True, False)  # init — either is fine
        assert is_pid_alive(999999999) is False
        assert is_pid_alive(None) is False and is_pid_alive(-3) is False
        print("  [ok] pid liveness (self alive, bogus pid dead)")

        # 3. detection: fake a crashed turn with a dead pid
        dead_pid = 999999999
        atomic_write_json(tmp / "deadbeef.json", {
            "session_id": "deadbeef", "pid": dead_pid,
            "status": "in_progress", "started_at": _now_iso(),
            "user_text": "do the thing", "messages_so_far": [],
            "tool_calls_done": [{"name": "read_file", "status": "done"}]})
        # an active turn from THIS process must NOT be flagged
        atomic_write_json(tmp / "live1234.json", {
            "session_id": "live1234", "pid": os.getpid(),
            "status": "in_progress", "started_at": _now_iso(),
            "user_text": "still running", "messages_so_far": [],
            "tool_calls_done": []})
        # a completed turn must NOT be flagged
        atomic_write_json(tmp / "done5678.json", {
            "session_id": "done5678", "pid": dead_pid,
            "status": "completed", "started_at": _now_iso(),
            "user_text": "finished", "messages_so_far": [],
            "tool_calls_done": []})
        # corrupt file must not crash the scan
        (tmp / "corrupt.json").write_text("{not json")
        found = load_incomplete_turns()
        sids = [t["session_id"] for t in found]
        assert sids == ["deadbeef"], sids
        assert found[0]["n_tool_calls"] == 1
        print("  [ok] detection (dead in_progress found; live, completed, "
              "corrupt skipped)")

        # 4. register() on a fake agent: wrappers + TurnResume tool
        class FakeAgent:
            pass

        ag = FakeAgent()
        ag.session_id = "testsession"
        ag.messages = [{"role": "system", "content": "sys"},
                       {"role": "user", "content": "hi"}]
        ag.turns = []
        ag.tools = {}
        calls = []

        def fake_run_turn(user_text, *a, **k):
            # simulate one tool call inside the turn
            ev = type("E", (), {})()
            ev.name, ev.args, ev.result = "read_file", {"p": "x"}, "ok"
            ev.status, ev.duration = "done", 0.1
            ag._execute_tool(ev)
            ag.messages.append({"role": "assistant", "content": "done"})
            return "TURN-RESULT"

        ag.run_turn = fake_run_turn

        def fake_execute(ev, *a, **k):
            calls.append(ev.name)

        ag._execute_tool = fake_execute

        class FakeLog:
            def append(self, *a, **k):
                pass

        ag.log = FakeLog()
        register(ag)

        assert len(ag.incomplete_turns) == 1
        assert ag.incomplete_turns[0]["session_id"] == "deadbeef"
        assert "TurnResume" in ag.tools
        print("  [ok] register: incomplete_turns exposed, tool registered")

        # 5. full turn lifecycle writes through the wrapper
        out = ag.run_turn("test turn")
        assert out == "TURN-RESULT"
        state = json.loads((tmp / "testsession.json").read_text())
        assert state["status"] == "completed", state["status"]
        assert state["pid"] == os.getpid()
        assert len(state["tool_calls_done"]) == 1
        assert state["tool_calls_done"][0]["name"] == "read_file"
        assert len(state["messages_so_far"]) == 3  # sys+user+assistant
        assert calls == ["read_file"]
        print("  [ok] run_turn wrapper: start → tool snapshot → completed")

        # 6. resume restores messages + history, marks resumed
        msg = ag.tools["TurnResume"].handler(action="list")
        assert "deadbeef" in msg
        res = ag.tools["TurnResume"].handler(action="resume",
                                            session_id="deadbeef")
        assert "resumed" in res.lower(), res
        state = json.loads((tmp / "deadbeef.json").read_text())
        assert state["status"] == "resumed"
        assert ag.incomplete_turns == []
        assert len(ag.turns) == 1 and len(ag.turns[0].tools) == 1
        print("  [ok] TurnResume: list + resume restores state, marks resumed")

        # 7. register is idempotent (no double wrap)
        before = ag.run_turn
        register(ag)
        assert ag.run_turn is before
        print("  [ok] register idempotent")
    finally:
        turns_dir = real_turns_dir  # noqa: E731
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS")


if __name__ == "__main__":
    _selftest()
