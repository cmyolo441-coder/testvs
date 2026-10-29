"""Multi-session support: several concurrent, isolated conversations.

Each session is ``{id, name, created_at, updated_at, todos_file}``.
Session *metadata* lives in ``~/.fullagent/sessions_meta.json``; the
*messages* themselves already persist per session in the shared event log
(``user.message`` / ``assistant.message`` events carry the session id),
so switching sessions is a pure event-log fold — message text is never
duplicated into the metadata file.

Isolation contract:
  * ``agent.session_id`` + ``agent.log.session`` always name the active
    session, so new events are tagged to it.
  * On switch, the current session is snapshotted first (a session-file
    backup via ``agent.save_session()``; the event log already holds
    every message), then ``agent.messages`` is *replaced* with the
    target session's messages folded from the event log.

Public API (attached to the agent by :func:`register`):
  * ``agent.create_session(name="")`` -> new session id (and switches to it)
  * ``agent.list_sessions()``          -> [{id, name, created_at,
    updated_at, msg_count, preview}]
  * ``agent.switch_session(sid)``      -> full id switched to

Tools: ``SessionNew``, ``SessionList``, ``SessionSwitch``.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .tools import Tool

# ------------------------------------------------------------------ paths --

_META_PATH_OVERRIDE: Optional[Path] = None
_meta_lock = threading.Lock()


def set_meta_path(path: str | Path | None) -> None:
    """Override the metadata file location (self-tests). None resets."""
    global _META_PATH_OVERRIDE
    _META_PATH_OVERRIDE = Path(path) if path else None


def meta_path() -> Path:
    if _META_PATH_OVERRIDE is not None:
        return _META_PATH_OVERRIDE
    from . import config  # local import: config has no heavy deps
    return config.APP_DIR / "sessions_meta.json"


# ------------------------------------------------------------- metadata ---

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_meta() -> dict[str, Any]:
    """Read the metadata file; a missing/corrupt file yields an empty one."""
    try:
        with open(meta_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(
                data.get("sessions"), list):
            return {"sessions": [s for s in data["sessions"]
                                 if isinstance(s, dict)]}
    except (OSError, ValueError):
        pass
    return {"sessions": []}


def save_meta(meta: dict[str, Any]) -> bool:
    """Atomic best-effort write of the metadata file."""
    try:
        dest = meta_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=str(dest.parent), suffix=".tmp", delete=False,
            encoding="utf-8",
        ) as fh:
            json.dump(meta, fh, indent=1, ensure_ascii=False)
            tmp = fh.name
        os.replace(tmp, dest)
        return True
    except (OSError, ValueError, TypeError):
        return False


def _entry(meta: dict, sid: str) -> Optional[dict]:
    for s in meta.get("sessions", []):
        if isinstance(s, dict) and s.get("id") == sid:
            return s
    return None


def _todos_file(sid: str) -> str:
    from . import todos as _todos  # local import: no cycle (todos is standalone)
    return _todos._persist_path(sid)


def _record(meta: dict, sid: str, name: str = "",
            msg_count: Optional[int] = None) -> dict:
    """Upsert a metadata entry for a session. Returns the entry."""
    ent = _entry(meta, sid)
    if ent is None:
        ent = {
            "id": sid,
            "name": name or f"session {sid}",
            "created_at": _now(),
            "updated_at": _now(),
            "todos_file": _todos_file(sid),
            "msg_count": msg_count or 0,
        }
        meta.setdefault("sessions", []).append(ent)
    else:
        if name:
            ent["name"] = name
        ent["updated_at"] = _now()
        try:
            ent["todos_file"] = _todos_file(sid)
        except Exception:
            pass
        if msg_count is not None:
            ent["msg_count"] = msg_count
    return ent


def _new_id(meta: dict) -> str:
    for _ in range(64):
        sid = uuid.uuid4().hex[:8]
        if _entry(meta, sid) is None:
            return sid
    raise RuntimeError("could not mint a unique session id")


# ------------------------------------------------- message reconstruction --

_MESSAGE_TYPES = ("user.message", "assistant.message")


def messages_for_session(log: Any, sid: str) -> list[dict[str, Any]]:
    """Fold a session's messages out of the event log.

    Returns fresh ``{"role", "content"}`` dicts in causal order. The
    event log is the persistence layer: ``user.message`` /
    ``assistant.message`` events are appended with the session id while
    a turn runs, so no in-memory copy needs to be saved first.
    """
    msgs: list[dict[str, Any]] = []
    try:
        events = log.events()  # shared cached chain — read-only, never mutate
    except Exception:
        return msgs
    for ev in events:
        try:
            if getattr(ev, "session", "") != sid:
                continue
            etype = getattr(ev, "type", "")
            if etype not in _MESSAGE_TYPES:
                continue
            data = getattr(ev, "data", None) or {}
            text = data.get("text", "")
            text = text if isinstance(text, str) else str(text)
        except Exception:
            continue
        msgs.append({"role": ("user" if etype == "user.message"
                              else "assistant"),
                     "content": text})
    return msgs


def _last_user_preview(log: Any, sid: str, limit: int = 90) -> str:
    for msg in reversed(messages_for_session(log, sid)):
        if msg.get("role") == "user":
            text = str(msg.get("content", "")).strip().replace("\n", " ")
            return text[:limit] + ("…" if len(text) > limit else "")
    return ""


# ------------------------------------------------------------- primitives --

def _reseat(agent: Any) -> None:
    """Best-effort system-prompt reseat after replacing agent.messages."""
    fn = getattr(agent, "_reseat_system_prompt", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass


def _reseat_todos(agent: Any, sid: str) -> None:
    """Point the agent's todo state at the new session's todo file."""
    try:
        from . import todos as _todos
        mgr = _todos.TodoManager(sid)
        agent.todo_manager = mgr
        # keep the module-level helpers (todo panel) in sync too
        with _todos._manager_lock:
            _todos._manager = mgr
    except Exception:
        pass


def _snapshot_current(agent: Any) -> None:
    """Persist the outgoing session before leaving it.

    The event log already holds every message; this additionally writes
    the per-session JSON backup (``Agent.save_session``) and refreshes
    the metadata entry's timestamp/message count. Best-effort: a failed
    snapshot must never block a switch.
    """
    sid = getattr(agent, "session_id", "") or ""
    if not sid:
        return
    save_session = getattr(agent, "save_session", None)
    if callable(save_session):
        try:
            save_session()
        except Exception:
            pass
    try:
        msgs = getattr(agent, "messages", None) or []
        with _meta_lock:
            meta = load_meta()
            _record(meta, sid, msg_count=sum(
                1 for m in msgs if isinstance(m, dict)
                and m.get("role") in ("user", "assistant")))
            save_meta(meta)
    except Exception:
        pass


def _reset_subagents(agent: Any, next_session_id: str) -> None:
    """Stop outgoing-session workers before changing the event-log session."""
    manager = getattr(agent, "subagents", None)
    reset = getattr(manager, "reset_session", None)
    if callable(reset):
        reset(next_session_id)


def create_session(agent: Any, name: str = "") -> str:
    """Create a new session, snapshot the current one, switch to the new."""
    with _meta_lock:
        meta = load_meta()
        old = getattr(agent, "session_id", "") or ""
        if old:
            _record(meta, old)
        sid = _new_id(meta)
        _record(meta, sid, name=name)
        save_meta(meta)
    _reset_subagents(agent, sid)
    # snapshot the outgoing session BEFORE reseating state
    _snapshot_current(agent)

    agent.session_id = sid
    try:
        agent.log.session = sid
    except Exception:
        pass
    agent.messages = []
    if hasattr(agent, "turns"):
        try:
            agent.turns = []
        except Exception:
            pass
    _reseat(agent)
    try:
        agent.log.append("session.start",
                         {"session_id": sid,
                          "name": name or f"session {sid}"},
                         actor="system")
    except Exception:
        pass
    _reseat_todos(agent, sid)
    return sid


def _resolve_id(meta: dict, sid: str) -> str:
    """Exact id, else an unambiguous id prefix. Raises ValueError."""
    sid = (sid or "").strip()
    if not sid:
        raise ValueError("no session id given — use /switch <id>")
    if _entry(meta, sid):
        return sid
    cands = [s["id"] for s in meta.get("sessions", [])
             if isinstance(s, dict) and str(s.get("id", "")).startswith(sid)]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        known = ", ".join(str(s.get("id")) for s in meta.get("sessions", [])
                          if isinstance(s, dict)) or "(none)"
        raise ValueError(f"unknown session '{sid}' — known: {known}")
    raise ValueError(f"ambiguous prefix '{sid}' — matches: "
                     + ", ".join(cands))


def switch_session(agent: Any, sid: str) -> str:
    """Switch to another session, snapshotting the current one first."""
    with _meta_lock:
        meta = load_meta()
        full = _resolve_id(meta, sid)
    current = getattr(agent, "session_id", "") or ""
    if full == current:
        return full

    _reset_subagents(agent, full)
    # NEVER corrupt the outgoing session: snapshot it before touching it.
    _snapshot_current(agent)

    msgs = messages_for_session(agent.log, full)

    agent.session_id = full
    try:
        agent.log.session = full
    except Exception:
        pass
    agent.messages = []
    _reseat(agent)          # sealed system prompt back at position 0
    agent.messages.extend(msgs)
    if hasattr(agent, "turns"):
        try:
            agent.turns = []
        except Exception:
            pass
    try:
        agent.log.append("session.resumed", {"session_id": full},
                         actor="system")
    except Exception:
        pass
    _reseat_todos(agent, full)
    with _meta_lock:
        meta = load_meta()
        _record(meta, full)
        save_meta(meta)
    return full


def list_sessions(agent: Any) -> list[dict[str, Any]]:
    """All known sessions: id, name, timestamps, message count, preview."""
    with _meta_lock:
        meta = load_meta()
    log = getattr(agent, "log", None)
    out: list[dict[str, Any]] = []
    for s in meta.get("sessions", []):
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id", ""))
        count = (len(messages_for_session(log, sid))
                 if log is not None else int(s.get("msg_count", 0) or 0))
        out.append({
            "id": sid,
            "name": s.get("name", f"session {sid}"),
            "created_at": s.get("created_at", ""),
            "updated_at": s.get("updated_at", ""),
            "msg_count": count,
            "preview": _last_user_preview(log, sid) if log is not None else "",
            "current": sid == (getattr(agent, "session_id", "") or ""),
        })
    out.sort(key=lambda e: e.get("updated_at", ""), reverse=True)
    return out


def format_session_list(entries: list[dict[str, Any]]) -> str:
    """Human-readable session list for the TUI / tool output."""
    if not entries:
        return "no sessions yet."
    lines = []
    for e in entries:
        mark = "●" if e.get("current") else "○"
        name = e.get("name", "")
        prev = e.get("preview", "")
        line = (f"{mark} {e.get('id')}  {name}  "
                f"({e.get('msg_count', 0)} msgs)")
        if prev:
            line += f"\n    last: {prev}"
        lines.append(line)
    return "\n".join(lines)


# ------------------------------------------------------------------ tools --

def make_session_new_tool(agent: Any) -> Tool:
    def _handle(name: str = "") -> str:
        sid = create_session(agent, (name or "").strip())
        return (f"created session '{name or sid}' (id: {sid}) — "
                f"switched to it. Previous session snapshotted.")

    return Tool(
        name="SessionNew",
        description=("Create a new conversation session and switch to it. "
                     "The current session is snapshotted first so nothing "
                     "is lost."),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string",
                         "description": "Human-friendly session name."},
            },
        },
        handler=_handle,
    )


def make_session_list_tool(agent: Any) -> Tool:
    def _handle() -> str:
        return format_session_list(list_sessions(agent))

    return Tool(
        name="SessionList",
        description=("List all conversation sessions with message counts "
                     "and a preview of the last user message."),
        parameters={"type": "object", "properties": {}},
        handler=_handle,
    )


def make_session_switch_tool(agent: Any) -> Tool:
    def _handle(session_id: str) -> str:
        full = switch_session(agent, session_id)
        n = len(getattr(agent, "messages", []) or [])
        return (f"switched to session {full} — {n} message(s) restored. "
                f"Previous session snapshotted.")

    return Tool(
        name="SessionSwitch",
        description=("Switch to another conversation session by id (a "
                     "unique id prefix also works). The current session is "
                     "snapshotted first; its messages are restored from the "
                     "event log on return."),
        parameters={
            "type": "object",
            "properties": {
                "session_id": {"type": "string",
                               "description": "Session id (or unique prefix)."},
            },
            "required": ["session_id"],
        },
        handler=_handle,
    )


# --------------------------------------------------------------- register --

def register(agent: Any) -> None:
    """Wire multi-session support into an agent (duck-typed).

    Attaches ``create_session`` / ``list_sessions`` / ``switch_session``
    bound helpers, registers the SessionNew/SessionList/SessionSwitch
    tools, and records the agent's current session in the metadata file
    so it shows up in listings.
    """
    agent.create_session = lambda name="": create_session(agent, name)
    agent.list_sessions = lambda: list_sessions(agent)
    agent.switch_session = lambda sid: switch_session(agent, sid)

    tools = getattr(agent, "tools", None)
    if isinstance(tools, dict):
        tools["SessionNew"] = make_session_new_tool(agent)
        tools["SessionList"] = make_session_list_tool(agent)
        tools["SessionSwitch"] = make_session_switch_tool(agent)

    # the pre-existing conversation is session #1 — record it
    try:
        sid = getattr(agent, "session_id", "") or ""
        if sid:
            with _meta_lock:
                meta = load_meta()
                _record(meta, sid)
                save_meta(meta)
    except Exception:
        pass


# -------------------------------------------------------------- self-test --

if __name__ == "__main__":
    from types import SimpleNamespace

    from .kernel import EventLog
    from . import config as _config

    tmp = Path(tempfile.mkdtemp(prefix="multisess_selftest_"))
    set_meta_path(tmp / "sessions_meta.json")

    # isolate the per-session JSON snapshot dir (Agent.save_session target)
    _real_sessions_dir = _config.SESSIONS_DIR
    _config.SESSIONS_DIR = tmp / "sessions"

    class FakeAgent:
        """Duck-typed agent: real EventLog, real message flow."""

        def __init__(self):
            self.session_id = "aaaa1111"
            self.messages: list = []
            self.tools: dict = {}
            self.turns: list = []
            self.cfg = SimpleNamespace(model_id="selftest")
            self.log = EventLog(tmp / "eventlog.jsonl",
                                session=self.session_id)

        def _reseat_system_prompt(self, sections=None):
            if not self.messages or self.messages[0].get("role") != "system":
                self.messages.insert(0, {"role": "system",
                                         "content": "SYSTEM PROMPT"})

        def save_session(self):
            # mirrors Agent.save_session (snapshot backup)
            _config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            p = _config.SESSIONS_DIR / f"{self.session_id}.json"
            p.write_text(json.dumps(
                {"session_id": self.session_id,
                 "messages": list(self.messages)}), encoding="utf-8")
            return p

    def say(agent, text, reply):
        """Simulate one real turn: messages + tagged event-log entries."""
        agent.messages.append({"role": "user", "content": text})
        agent.log.append("user.message",
                         {"text": text, "session": agent.session_id},
                         actor="human")
        agent.messages.append({"role": "assistant", "content": reply})
        agent.log.append("assistant.message",
                         {"text": reply, "session": agent.session_id},
                         actor="sovereign")

    try:
        a = FakeAgent()
        register(a)
        assert "SessionNew" in a.tools
        assert "SessionList" in a.tools
        assert "SessionSwitch" in a.tools
        assert callable(a.create_session) and callable(a.switch_session)

        # --- session A: two distinct messages ---------------------------
        sid_a = a.session_id
        say(a, "alpha: how do I reverse a list in python?",
            "alpha-reply: use list[::-1]")
        say(a, "alpha: and sort it?",
            "alpha-reply: sorted()")

        # --- create session B, verify clean slate -----------------------
        sid_b = a.create_session("beta-work")
        assert sid_b != sid_a and a.session_id == sid_b, "new id not active"
        assert a.log.session == sid_b, "log not retagged"
        # only the reseated system prompt; A's messages must NOT leak
        assert all("alpha" not in str(m.get("content", ""))
                   for m in a.messages), "A leaked into B"
        say(a, "beta: write a haiku about rust",
            "beta-reply: orange flakes fall")
        assert len([m for m in a.messages
                    if m.get("role") in ("user", "assistant")]) == 2

        # --- metadata persisted -----------------------------------------
        mp = tmp / "sessions_meta.json"
        assert mp.exists(), "sessions_meta.json not written"
        meta = json.loads(mp.read_text(encoding="utf-8"))
        ids = {s["id"] for s in meta["sessions"]}
        assert {sid_a, sid_b} <= ids, f"meta missing sessions: {ids}"
        names = {s["id"]: s["name"] for s in meta["sessions"]}
        assert names[sid_b] == "beta-work", names
        assert all("todos_file" in s and s["todos_file"].endswith(
            f"{s['id']}.json") for s in meta["sessions"]), "todos ref missing"

        # --- SessionList: counts + previews ------------------------------
        entries = a.list_sessions()
        by_id = {e["id"]: e for e in entries}
        assert by_id[sid_a]["msg_count"] == 4, by_id[sid_a]
        assert by_id[sid_b]["msg_count"] == 2, by_id[sid_b]
        assert "alpha: and sort it?" in by_id[sid_a]["preview"], by_id[sid_a]
        assert "beta: write a haiku" in by_id[sid_b]["preview"], by_id[sid_b]
        assert by_id[sid_b]["current"] is True
        listed = a.tools["SessionList"].handler()
        assert sid_a in listed and sid_b in listed and "beta-work" in listed

        # --- switch back to A: full isolation -----------------------------
        a.switch_session(sid_a)
        assert a.session_id == sid_a and a.log.session == sid_a
        body = [m for m in a.messages
                if m.get("role") in ("user", "assistant")]
        assert len(body) == 4, f"A lost messages: {len(body)}"
        texts = [m["content"] for m in body]
        assert texts[0] == "alpha: how do I reverse a list in python?"
        assert texts[1] == "alpha-reply: use list[::-1]"
        assert texts[2] == "alpha: and sort it?"
        assert texts[3] == "alpha-reply: sorted()"
        assert not any("beta" in t for t in texts), "B leaked into A"
        # outgoing session B was snapshotted first
        assert (tmp / "sessions" / f"{sid_b}.json").exists(), \
            "B not snapshotted on switch"

        # --- and back to B: nothing lost ---------------------------------
        a.switch_session(sid_b)
        body = [m for m in a.messages
                if m.get("role") in ("user", "assistant")]
        assert len(body) == 2, f"B lost messages: {len(body)}"
        assert body[0]["content"] == "beta: write a haiku about rust"
        assert body[1]["content"] == "beta-reply: orange flakes fall"

        # --- tools: SessionNew + SessionSwitch (prefix) --------------------
        out = a.tools["SessionNew"].handler(name="gamma")
        sid_c = a.session_id
        assert sid_c not in (sid_a, sid_b) and "gamma" in out, out
        say(a, "gamma: ping", "gamma-reply: pong")
        out = a.tools["SessionSwitch"].handler(session_id=sid_a[:4])
        assert a.session_id == sid_a, out
        assert f"switched to session {sid_a}" in out, out
        body = [m for m in a.messages
                if m.get("role") in ("user", "assistant")]
        assert len(body) == 4 and "alpha" in body[0]["content"]

        # --- error paths ----------------------------------------------------
        try:
            a.switch_session("nope-nope")
            raise AssertionError("unknown id should raise")
        except ValueError:
            pass

        # --- persistence survives a fresh read ------------------------------
        meta2 = load_meta()
        assert len(meta2["sessions"]) == 3, meta2
        a.log.close()
        log2 = EventLog(tmp / "eventlog.jsonl")
        assert len(messages_for_session(log2, sid_a)) == 4
        assert len(messages_for_session(log2, sid_b)) == 2
        assert len(messages_for_session(log2, sid_c)) == 2
        log2.close()

        print("multisess self-test: PASS — 3 sessions, full isolation, "
              "meta persisted, prefix switch works")
    finally:
        _config.SESSIONS_DIR = _real_sessions_dir
        set_meta_path(None)
