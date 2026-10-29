"""Session persistence helpers: save / list / load / resume chat sessions.

A session is a JSON file in ``~/.fullagent/sessions/`` named
``{session_id}.json`` — the same schema and location as
``Agent.save_session()`` in agent.py, so files written by either side are
interchangeable. Messages are capped at the last 200 to bound file size.

This module never imports ``.agent``: it works on any object with
``session_id``, ``cfg.model_id`` and ``messages`` attributes.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Max messages kept in a saved session (messages may be large).
MAX_MESSAGES = 200

_STORE_DIR_OVERRIDE: Optional[Path] = None


def set_store_dir(path: str | Path | None) -> None:
    """Override the session store dir (tests). Pass None to reset."""
    global _STORE_DIR_OVERRIDE
    _STORE_DIR_OVERRIDE = Path(path) if path else None


def store_dir() -> Path:
    if _STORE_DIR_OVERRIDE is not None:
        return _STORE_DIR_OVERRIDE
    from . import config  # local import: config has no heavy deps
    return config.SESSIONS_DIR


def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def _message_text(msg: Any) -> str:
    """Best-effort plain-text preview of one message dict."""
    if not isinstance(msg, dict):
        return str(msg)[:120]
    content = msg.get("content", "")
    if isinstance(content, str):
        return content.strip()[:120]
    if isinstance(content, list):  # content blocks
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return str(block.get("text", "")).strip()[:120]
    return ""


def _last_user_preview(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return _message_text(msg)
    return _message_text(messages[-1]) if messages else ""


# ---------------------------------------------------------------- save ------

def save(agent: Any, session_id: Optional[str] = None,
         path: Optional[Path] = None) -> Optional[Path]:
    """Persist an agent's conversation to the session store.

    Uses the same schema as ``Agent.save_session()``; keeps the last
    ``MAX_MESSAGES`` messages only. Returns the file path, or None on error.
    """
    try:
        sid = session_id or getattr(agent, "session_id", None) or "session"
        sid = Path(str(sid)).name or "session"
        cfg = getattr(agent, "cfg", None)
        model_id = getattr(cfg, "model_id", "") if cfg is not None else ""
        messages = list(getattr(agent, "messages", []) or [])[-MAX_MESSAGES:]
        payload: dict[str, Any] = {
            "session_id": sid,
            "model_id": model_id,
            "saved_at": datetime.now().isoformat(),
            "messages": messages,
        }
        todos = getattr(agent, "todos", None)
        if todos:
            try:
                json.dumps(todos)  # only if serializable
                payload["todos"] = todos
            except (TypeError, ValueError):
                pass

        dest = Path(path) if path else store_dir() / f"{sid}.json"
        _ensure_dir(dest.parent)
        # atomic write: never leave a torn/truncated session file
        with tempfile.NamedTemporaryFile(
            "w", dir=str(dest.parent), suffix=".tmp", delete=False,
            encoding="utf-8",
        ) as fh:
            json.dump(payload, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
            tmp = fh.name
        os.replace(tmp, dest)
        return dest
    except (OSError, ValueError, TypeError, RuntimeError):
        return None


# ---------------------------------------------------------------- list ------

def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """List saved sessions, newest first.

    Each entry: {id, saved_at, model_id, msg_count, preview}.
    """
    d = store_dir()
    if not d.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for fp in d.glob("*.json"):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            # valid JSON but not a session object ("[]", "42", ...):
            # data.get() below would raise AttributeError and kill the
            # whole listing for one stray file
            continue
        messages = data.get("messages") or []
        entries.append({
            "id": data.get("session_id") or fp.stem,
            "saved_at": data.get("saved_at") or "",
            "model_id": data.get("model_id") or "",
            "msg_count": len(messages),
            "preview": _last_user_preview(messages),
        })
    entries.sort(key=lambda e: e["saved_at"], reverse=True)
    return entries[:limit]


def latest_session() -> Optional[dict[str, Any]]:
    """Newest saved session entry, or None."""
    sessions = list_sessions(limit=1)
    return sessions[0] if sessions else None


# ---------------------------------------------------------------- load ------

def load(session_id: str) -> Optional[dict[str, Any]]:
    """Load a saved session dict by id. ``"latest"`` resolves newest first."""
    if session_id == "latest":
        entry = latest_session()
        if entry is None:
            return None
        session_id = entry["id"]
    session_id = Path(str(session_id)).name
    fp = store_dir() / f"{session_id}.json"
    if not fp.is_file():
        return None
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    # a valid-JSON non-object file is not a session; callers expect a dict
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------- resume -----

def resume_into(agent: Any, data: dict[str, Any]) -> str:
    """Restore a loaded session dict into an agent.

    Replaces ``agent.messages`` and ``agent.session_id`` (plus ``todos`` when
    present). Returns a one-line human-readable summary.
    """
    messages = list(data.get("messages") or [])
    sid = str(data.get("session_id") or getattr(agent, "session_id", ""))
    agent.messages = messages
    if sid:
        agent.session_id = sid
        log = getattr(agent, "log", None)
        if log is not None:
            try:
                log.session = sid
            except AttributeError:
                pass
    todos = data.get("todos")
    if todos is not None:
        try:
            agent.todos = todos
        except AttributeError:
            pass

    n_user = sum(1 for m in messages
                 if isinstance(m, dict) and m.get("role") == "user")
    n_asst = sum(1 for m in messages
                 if isinstance(m, dict) and m.get("role") == "assistant")
    preview = _last_user_preview(messages)
    summary = (f"resumed session {sid} — {len(messages)} messages "
               f"({n_user} user / {n_asst} assistant)")
    if preview:
        summary += f"\n  last prompt: {preview[:100]}"
    return summary


# --------------------------------------------------------------- self-test --

def _selftest() -> int:
    """tmp store dir: save fake agent → list → load → resume_into asserts."""
    import tempfile
    from types import SimpleNamespace

    tmp = Path(tempfile.mkdtemp(prefix="fullagent-sessions-test-"))
    set_store_dir(tmp)
    try:
        agent = SimpleNamespace(
            session_id="abc12345",
            cfg=SimpleNamespace(model_id="test-model"),
            messages=[
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "second question"},
            ],
        )

        p = save(agent)
        assert p is not None and p.is_file(), "save failed"
        assert p.name == "abc12345.json", f"unexpected name {p.name}"

        listed = list_sessions()
        assert len(listed) == 1, f"expected 1 session, got {len(listed)}"
        e = listed[0]
        assert e["id"] == "abc12345", e
        assert e["msg_count"] == 3, e
        assert e["model_id"] == "test-model", e
        assert "second question" in e["preview"], e

        data = load("abc12345")
        assert data is not None, "load failed"
        assert data["messages"][0]["content"] == "hello world", data
        assert load("nonexistent") is None, "load should return None"

        # "latest" alias
        data2 = load("latest")
        assert data2 is not None and data2["session_id"] == "abc12345"

        # resume into a fresh agent
        fresh = SimpleNamespace(
            session_id="fresh000",
            cfg=SimpleNamespace(model_id="test-model"),
            messages=[],
        )
        summary = resume_into(fresh, data)
        assert fresh.messages == agent.messages, "messages not restored"
        assert fresh.session_id == "abc12345", "session_id not restored"
        assert "3 messages" in summary and "abc12345" in summary, summary

        # message cap: 250 messages → saved file keeps last 200
        big = SimpleNamespace(
            session_id="big999",
            cfg=SimpleNamespace(model_id="m"),
            messages=[{"role": "user", "content": f"m{i}"} for i in range(250)],
        )
        save(big)
        assert len(load("big999")["messages"]) == 200, "cap not applied"
        listed = list_sessions()
        assert listed[0]["id"] == "big999", "newest-first ordering broken"

        print("PASS: sessions save/list/load/resume_into all OK")
        return 0
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        return 1
    finally:
        set_store_dir(None)


if __name__ == "__main__":
    raise SystemExit(_selftest())
