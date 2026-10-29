"""Checkpoints & rewind — snapshot conversation + file state, restore on demand.

Claude Code-style ``/rewind`` support:

- :class:`CheckpointManager` snapshots ``agent.messages`` (deep copy) plus the
  *before-content* of files the session touched, and can restore both.
- :func:`maybe_auto_checkpoint` snapshots automatically before risky tools.
- :func:`register` attaches a manager to an agent instance.
- :func:`rewind_command_data` / :func:`do_rewind` are the TUI glue for the
  ``/rewind`` slash-command picker.

Duck-typing only: this module never imports ``.agent`` or ``.tui``. The agent
only needs ``messages`` (a list), ``session_id`` (a str) and ``tools`` (a
name -> Tool mapping with a ``risk`` attribute) attributes.

Intended wiring (done by the caller — this module does not patch anything):

    # agent.py — right after the Agent is built / session starts:
    from fullagent.checkpoints import register
    register(agent)                       # attaches agent.checkpoints

    # agent.py — in _execute_tool, after the risk gate, before the tool runs:
    from fullagent.checkpoints import maybe_auto_checkpoint
    maybe_auto_checkpoint(self, ev.name)  # snapshots before risky tools only

    # tui.py — inside the slash-command dispatcher:
    elif cmd == "/rewind":
        from fullagent.checkpoints import rewind_command_data, do_rewind
        items = rewind_command_data(ui)   # -> [{id, label, time, msg_count}]
        picked = ui.pick("Rewind to…", items)   # whatever picker the TUI has
        if picked:
            do_rewind(ui, picked["id"])

File-state tracking: the tool handlers that mutate the filesystem should call
``agent.checkpoints.note_file_touched(path, content_before)`` *before* they
mutate. The first note per path wins (we keep the earliest before-content, so
a rewind returns the file to its pre-session state). The store is capped at
~50 files / 2 MB of before-content total.
"""

from __future__ import annotations

import copy
import json
import os
import logging
import time
import uuid
from pathlib import Path

_log = logging.getLogger(__name__)

# Caps for the tracked before-content store.
MAX_TRACKED_FILES = 50
MAX_TRACKED_BYTES = 2 * 1024 * 1024  # 2 MB total

# Auto-checkpoint ring.
AUTO_LABEL_PREFIX = "auto: "
MAX_AUTO_CHECKPOINTS = 10

# MEMORY BOUND (worker 19/20): manual checkpoints are full deep-copies of
# the message history — cap them so /checkpoint spam in a long session
# cannot grow RAM without limit. Oldest manual goes first; auto ring is
# enforced separately and is untouched by this cap.
MAX_MANUAL_CHECKPOINTS = 20

RISK_SAFE = "safe"


def _checkpoints_dir(session_id: str) -> Path:
    # Use config.APP_DIR (which already handles read-only-home tmp
    # fallback) instead of re-deriving ~/.fullagent — a divergent base
    # could raise here and crash startup via register().
    from .config import APP_DIR
    d = APP_DIR / "checkpoints" / str(session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


class CheckpointManager:
    """Snapshots conversation + file state for one agent session."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.dir = _checkpoints_dir(self.session_id)
        # id -> {"messages": [...], "files": {path: bytes|None}}
        self._snapshots: dict[str, dict] = {}
        # id -> metadata dict
        self._meta: dict[str, dict] = {}
        # First-note-per-path before-content: {path: bytes | None}
        # None content means "did not exist at note time".
        self._before: dict[str, bytes | None] = {}
        self._before_bytes = 0
        # Ordered ids of auto checkpoints (oldest first) for the ring.
        self._auto_ids: list[str] = []
        # Ordered ids of manual checkpoints (oldest first) — bounded by
        # MAX_MANUAL_CHECKPOINTS so snapshots can't grow RAM unboundedly.
        self._manual_ids: list[str] = []

    # ------------------------------------------------------------------ files

    def note_file_touched(self, path: str | Path, content_before) -> bool:
        """Record a file's before-content before a tool mutates it.

        The first note per path wins. Returns False if the note was dropped
        because the tracked store is at capacity.
        """
        key = str(path)
        if key in self._before:
            return True  # already tracked; keep the earliest before-content
        if len(self._before) >= MAX_TRACKED_FILES:
            return False
        if content_before is None:
            content = None
        elif isinstance(content_before, bytes):
            content = content_before
        else:
            content = str(content_before).encode("utf-8", "replace")
        if content is not None and self._before_bytes + len(content) > MAX_TRACKED_BYTES:
            return False
        self._before[key] = content
        if content is not None:
            self._before_bytes += len(content)
        return True

    # ------------------------------------------------------------------ create

    def create(self, agent, label: str, auto: bool = False) -> str:
        """Snapshot agent.messages + tracked file state. Returns checkpoint id."""
        cid = uuid.uuid4().hex[:8]
        messages = getattr(agent, "messages", []) or []
        snapshot = {
            "messages": copy.deepcopy(messages),
            "files": dict(self._before),  # before-contents, immutable from here
        }
        ts = time.time()
        meta = {
            "id": cid,
            "label": label,
            "time": ts,
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
            "msg_count": len(messages),
            "auto": auto,
            "files": sorted(self._before.keys()),
        }
        self._snapshots[cid] = snapshot
        self._meta[cid] = meta
        if auto:
            self._auto_ids.append(cid)
            self._persist_meta(meta)
            self._enforce_auto_ring()
        else:
            self._manual_ids.append(cid)
            self._persist_meta(meta)
            # bound memory: each manual snapshot is a full deepcopy of
            # the message history — evict the oldest manual first
            while len(self._manual_ids) > MAX_MANUAL_CHECKPOINTS:
                self.discard(self._manual_ids.pop(0))
        return cid

    def _persist_meta(self, meta: dict) -> None:
        # Metadata only — file *contents* stay in memory.
        try:
            (self.dir / f"{meta['id']}.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
        except OSError as e:  # metadata loss must never break the session
            _log.warning("checkpoint metadata persist failed: %s", e)

    def _enforce_auto_ring(self) -> None:
        while len(self._auto_ids) > MAX_AUTO_CHECKPOINTS:
            oldest = self._auto_ids.pop(0)
            self.discard(oldest)

    # ------------------------------------------------------------------ query

    def list(self) -> list[dict]:
        """Newest first: [{id, label, time, time_str, msg_count, auto}]."""
        return sorted(
            (dict(m) for m in self._meta.values()),
            key=lambda m: m["time"],
            reverse=True,
        )

    def get(self, checkpoint_id: str) -> dict | None:
        return self._meta.get(checkpoint_id)

    def discard(self, checkpoint_id: str) -> bool:
        self._snapshots.pop(checkpoint_id, None)
        self._meta.pop(checkpoint_id, None)
        if checkpoint_id in self._auto_ids:
            self._auto_ids.remove(checkpoint_id)
        if checkpoint_id in self._manual_ids:
            self._manual_ids.remove(checkpoint_id)
        try:
            (self.dir / f"{checkpoint_id}.json").unlink(missing_ok=True)
        except OSError:
            pass
        return True

    # ------------------------------------------------------------------ restore

    def restore(self, agent, checkpoint_id: str) -> dict:
        """Put back messages + rewrite tracked files. Returns result summary."""
        snap = self._snapshots.get(checkpoint_id)
        if snap is None:
            raise KeyError(
                f"checkpoint {checkpoint_id!r} not available "
                "(full state is in-memory only; metadata is persisted)"
            )
        agent.messages = copy.deepcopy(snap["messages"])
        restored, errors = [], []
        for path, before in snap["files"].items():
            try:
                p = Path(path)
                if before is None:
                    # File did not exist at note time — remove what was created.
                    p.unlink(missing_ok=True)
                else:
                    if p.parent != p and not p.parent.exists():
                        p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(before)
                restored.append(path)
            except OSError as e:  # per-file failure; keep going
                errors.append(f"{path}: {e}")
        # Best-effort: persist the rewound conversation if the agent supports it.
        saver = getattr(agent, "save_session", None)
        if callable(saver):
            try:
                saver()
            except Exception as e:  # noqa: BLE001
                _log.warning("save_session after rewind failed: %s", e)
        return {
            "checkpoint_id": checkpoint_id,
            "messages_restored": len(agent.messages),
            "files_restored": restored,
            "file_errors": errors,
        }


# ------------------------------------------------------------------ wiring API


def register(agent) -> CheckpointManager:
    """Attach a CheckpointManager as ``agent.checkpoints``. Idempotent."""
    mgr = getattr(agent, "checkpoints", None)
    if isinstance(mgr, CheckpointManager):
        return mgr
    mgr = CheckpointManager(session_id=getattr(agent, "session_id", None))
    agent.checkpoints = mgr
    return mgr


def maybe_auto_checkpoint(agent, tool_name: str) -> str | None:
    """Snapshot before risky tools (risk != "safe"). Ring-capped at 10 auto cps.

    Never raises — an auto-checkpoint failure must not break the turn.
    """
    try:
        mgr = getattr(agent, "checkpoints", None)
        if not isinstance(mgr, CheckpointManager):
            return None
        tools = getattr(agent, "tools", {}) or {}
        tool = tools.get(tool_name) if isinstance(tools, dict) else None
        risk = getattr(tool, "risk", RISK_SAFE) if tool is not None else RISK_SAFE
        if risk == RISK_SAFE:
            return None
        label = f"{AUTO_LABEL_PREFIX}{tool_name}"
        return mgr.create(agent, label, auto=True)
    except Exception as e:  # noqa: BLE001
        _log.warning("auto checkpoint before %s failed: %s", tool_name, e)
        return None


# ------------------------------------------------------------------ TUI glue


def rewind_command_data(ui) -> list[dict]:
    """Checkpoints for the /rewind picker. `ui` duck-typed (needs .agent)."""
    agent = getattr(ui, "agent", None)
    mgr = getattr(agent, "checkpoints", None)
    if not isinstance(mgr, CheckpointManager):
        return []
    return mgr.list()


def do_rewind(ui, checkpoint_id: str) -> tuple[bool, str]:
    """Perform the rewind and report through `ui` (best-effort print)."""
    agent = getattr(ui, "agent", None)
    mgr = getattr(agent, "checkpoints", None)
    if not isinstance(mgr, CheckpointManager):
        return False, "no checkpoints registered for this session"

    def say(msg: str) -> None:
        for meth in ("print", "echo", "show", "info"):
            fn = getattr(ui, meth, None)
            if callable(fn):
                try:
                    fn(msg)
                    return
                except Exception:  # noqa: BLE001
                    continue
        print(msg)

    try:
        res = mgr.restore(agent, checkpoint_id)
    except KeyError as e:
        say(f"rewind failed: {e}")
        return False, str(e)
    files = len(res["files_restored"])
    msg = (f"rewound to '{checkpoint_id}': "
           f"{res['messages_restored']} messages, {files} file(s) restored")
    if res["file_errors"]:
        msg += f"; {len(res['file_errors'])} file error(s)"
    say(msg)
    return True, msg


# ------------------------------------------------------------------ self-test

if __name__ == "__main__":
    import tempfile
    from types import SimpleNamespace

    failures = []

    def check(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            failures.append(name)

    tmp = Path(tempfile.mkdtemp(prefix="cp_selftest_"))
    f1 = tmp / "a.txt"
    f2 = tmp / "new.txt"
    f1.write_text("before-content", encoding="utf-8")

    # Fake agent: messages list + tools dict with risk attributes.
    fake = SimpleNamespace(
        session_id="selftest-session",
        messages=[{"role": "user", "content": "hello"}],
        tools={
            "read_file": SimpleNamespace(risk="safe"),
            "write_file": SimpleNamespace(risk="confirm"),
            "run_command": SimpleNamespace(risk="confirm"),
        },
    )
    mgr = register(fake)
    check("register attaches checkpoints",
          fake.checkpoints is mgr and isinstance(mgr, CheckpointManager))
    check("register idempotent", register(fake) is mgr)

    # Track f1 (existed) and f2 (did not exist), then mutate both.
    mgr.note_file_touched(f1, f1.read_bytes())
    mgr.note_file_touched(f2, None)
    cid = mgr.create(fake, "before risky edits")
    check("create returns id", isinstance(cid, str) and cid)
    check("list has 1 checkpoint", len(mgr.list()) == 1)
    check("list msg_count", mgr.list()[0]["msg_count"] == 1)

    fake.messages.append({"role": "assistant", "content": "edited"})
    f1.write_text("AFTER", encoding="utf-8")
    f2.write_text("created", encoding="utf-8")

    res = mgr.restore(fake, cid)
    check("messages restored", fake.messages == [{"role": "user", "content": "hello"}])
    check("file content back", f1.read_text(encoding="utf-8") == "before-content")
    check("created file removed", not f2.exists())
    check("restore summary", res["checkpoint_id"] == cid and res["messages_restored"] == 1)

    # Cap enforcement: 2MB total.
    big = b"x" * (MAX_TRACKED_BYTES + 1)
    kept = mgr.note_file_touched(tmp / "big.bin", big)
    check("oversize note dropped", kept is False)
    for i in range(MAX_TRACKED_FILES + 5):
        mgr.note_file_touched(tmp / f"f{i}.bin", b"z" * 8)
    check("file cap respected", len(mgr._before) <= MAX_TRACKED_FILES)

    # Auto-checkpoint ring: safe tool skipped, risky tools ring-capped at 10.
    check("safe tool no checkpoint", maybe_auto_checkpoint(fake, "read_file") is None)
    ids = set()
    for i in range(15):
        got = maybe_auto_checkpoint(fake, "write_file")
        if got:
            ids.add(got)
    auto_cps = [m for m in mgr.list() if m["auto"]]
    check("auto ring capped at 10", len(auto_cps) == MAX_AUTO_CHECKPOINTS)
    check("auto ids unique", len(ids) == 15 and len(set(ids)) == 15)
    check("ring keeps newest 10 only", len(mgr._auto_ids) == MAX_AUTO_CHECKPOINTS)
    check("auto label prefix", all(m["label"].startswith(AUTO_LABEL_PREFIX) for m in auto_cps))

    # Metadata persisted on disk (contents stay in-memory).
    meta_files = list(mgr.dir.glob("*.json"))
    check("metadata json persisted", len(meta_files) >= 1)
    meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
    check("metadata has fields", {"id", "label", "time", "msg_count"} <= set(meta))

    # Unknown checkpoint raises cleanly.
    try:
        mgr.restore(fake, "nope")
        check("unknown checkpoint raises", False)
    except KeyError:
        check("unknown checkpoint raises", True)

    # TUI glue with a fake ui.
    ui = SimpleNamespace(agent=fake, lines=[], print=lambda s: ui.lines.append(s))
    data = rewind_command_data(ui)
    check("rewind_command_data", isinstance(data, list) and len(data) >= 1)
    ok, msg = do_rewind(ui, cid)
    check("do_rewind ok", ok and "rewound" in msg)
    check("do_rewind reports via ui", len(ui.lines) >= 1)
    ok2, _ = do_rewind(SimpleNamespace(), cid)
    check("do_rewind no manager fails clean", ok2 is False)

    # Selftest dir should not leak into real ~/.fullagent (check we're using home).
    check("dir under ~/.fullagent", ".fullagent" in str(mgr.dir) or (os.environ.get("FULLAGENT_HOME", "") and os.environ["FULLAGENT_HOME"] in str(mgr.dir)))

    # MEMORY BOUND (worker 19/20): manual checkpoints are full deep-copies
    # of the message history — creating far more than the cap must evict
    # the oldest manual first, never the newest
    big_msgs = [{"role": "user", "content": "x" * 2000}] * 50
    fake.messages = big_msgs
    newest = None
    for i in range(MAX_MANUAL_CHECKPOINTS + 5):
        newest = mgr.create(fake, f"spam {i}")
    check("manual snapshots capped",
          sum(1 for m in mgr._meta.values() if not m["auto"])
          <= MAX_MANUAL_CHECKPOINTS)
    check("manual ring keeps exactly the cap",
          len(mgr._manual_ids) == MAX_MANUAL_CHECKPOINTS)
    check("newest manual still restorable",
          mgr.restore(fake, newest)["checkpoint_id"] == newest)
    check("auto ring untouched by manual cap",
          len(mgr._auto_ids) <= MAX_AUTO_CHECKPOINTS)

    # Cleanup.
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(mgr.dir, ignore_errors=True)

    print()
    if failures:
        print(f"RESULT: FAIL — {failures}")
        raise SystemExit(1)
    print("RESULT: PASS")
