"""Checkpoint-before-edit snapshots + EditRollback tool.

Incident motivation: during a long (>500s) turn loop the agent repeatedly
edits a large file; one bad edit mid-loop can corrupt the user's file with
no way back. This module wraps the mutating edit tools (``write_file``,
``edit_file``, ``MultiEdit``) so that BEFORE an edit is applied to an
EXISTING file, the original bytes are copied to

    ``~/.fullagent/snapshots/<session>/<relpath>.<timestamp>.bak``

New files skip snapshotting (there is nothing to restore). Retention is
bounded: only the last :data:`KEEP_SNAPSHOTS` (20) snapshots per file are
kept; older ones are pruned automatically.

Also registers an ``EditRollback`` tool that restores the most recent
snapshot for a path (or a specific timestamp), returning
``"rolled back <path> to <timestamp>"``.

Public API:
    - :func:`register` -- wrap the edit tools on an agent (duck-typed,
      same pattern as :mod:`fullagent.todos`) and add ``EditRollback``.
    - :func:`snapshot_file` -- snapshot one existing file now.
    - :func:`list_snapshots` -- newest-first list of ``(timestamp, path)``.
    - :func:`rollback` -- restore a snapshot; returns a status string.
"""

from __future__ import annotations

import dataclasses
import os
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:  # pragma: no cover - exercised through the package path
    from .tools import Tool, _resolve
except ImportError:  # running the file directly (dev convenience)
    from fullagent.tools import Tool, _resolve

SNAPSHOT_BASE = os.path.join(os.environ.get("FULLAGENT_HOME") or os.path.join(os.path.expanduser("~"), ".fullagent"), "snapshots")

#: Bounded retention: how many snapshots to keep per file.
KEEP_SNAPSHOTS = 20

_WRAPPED_ATTR = "_editbackup_wrapped"
_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]")

_lock = threading.Lock()


def _safe_session_id(session_id: Optional[str]) -> str:
    sid = (session_id or "default").strip() or "default"
    return _FILENAME_RE.sub("_", sid)[:64]


def _safe_relpath(p: Path) -> str:
    """Turn an absolute path into a safe relative dir/file structure."""
    p = p.resolve()
    try:
        rel = p.relative_to(Path.cwd().resolve())
        parts = rel.parts
    except ValueError:
        parts = tuple(x for x in p.parts if x not in ("/", os.sep))
    safe = [
        _FILENAME_RE.sub("_", part)
        for part in parts
        if part and part != ".."
    ]
    return os.path.join(*safe) if safe else "_root"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S%f")


def _session_dir(session_id: Optional[str]) -> Path:
    d = Path(SNAPSHOT_BASE) / _safe_session_id(session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_ts(name: str, rel: str) -> str:
    """Extract the timestamp from a snapshot file name."""
    stem = name[len(rel) + 1:]  # strip "<rel>."
    if stem.endswith(".bak"):
        stem = stem[: -len(".bak")]
    return stem


def _iter_snapshots(session_id: Optional[str], rel: str,
                    search_all_sessions: bool = False) -> List[Tuple[str, Path]]:
    """All (timestamp, path) snapshots for a relpath, oldest-first."""
    found: List[Tuple[str, Path]] = []
    base = Path(SNAPSHOT_BASE)
    if not base.is_dir():
        return found
    if search_all_sessions:
        dirs = [d for d in base.iterdir() if d.is_dir()]
    else:
        d = base / _safe_session_id(session_id)
        dirs = [d] if d.is_dir() else []
    for d in dirs:
        parent = d / os.path.dirname(rel)
        if not parent.is_dir():
            continue
        for f in parent.glob(os.path.basename(rel) + ".*.bak"):
            found.append((_parse_ts(f.name, os.path.basename(rel)), f))
    found.sort(key=lambda t: t[0])
    return found


def snapshot_file(path: str, session_id: Optional[str] = None) -> Optional[str]:
    """Copy the current bytes of an EXISTING file into the snapshot store.

    Returns the snapshot file path, or ``None`` when there is nothing to
    back up (new file, directory, unreadable path). Never raises.
    """
    try:
        p = _resolve(path)
    except Exception:
        return None
    try:
        if not p.is_file():
            return None  # new files / dirs: nothing to restore
        rel = _safe_relpath(p)
        ts = _timestamp()
        dest = _session_dir(session_id) / (rel + "." + ts + ".bak")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        with _lock:
            _prune_locked(session_id, rel)
        return str(dest)
    except Exception:
        return None


def _prune_locked(session_id: Optional[str], rel: str) -> None:
    snaps = _iter_snapshots(session_id, rel)
    for _, old in snaps[: max(0, len(snaps) - KEEP_SNAPSHOTS)]:
        try:
            old.unlink()
        except OSError:
            pass


def list_snapshots(path: str,
                   session_id: Optional[str] = None) -> List[Tuple[str, str]]:
    """Newest-first ``(timestamp, snapshot_path)`` list for a file."""
    try:
        p = _resolve(path)
    except Exception:
        return []
    rel = _safe_relpath(p)
    snaps = _iter_snapshots(session_id, rel, search_all_sessions=True)
    return [(ts, str(f)) for ts, f in reversed(snaps)]


def rollback(path: str, session_id: Optional[str] = None,
             timestamp: str = "") -> str:
    """Restore a snapshot over the live file.

    ``timestamp`` empty → most recent snapshot. Returns a status string,
    either ``"rolled back <path> to <timestamp>"`` or an ``"ERROR: ..."``.
    """
    try:
        p = _resolve(path)
    except Exception as e:
        return f"ERROR: bad path: {e}"
    rel = _safe_relpath(p)
    snaps = _iter_snapshots(session_id, rel, search_all_sessions=True)
    if not snaps:
        return f"ERROR: no snapshots found for {p}"
    if timestamp:
        match = [s for s in snaps if s[0] == timestamp or s[0].startswith(timestamp)]
        if not match:
            return (f"ERROR: no snapshot {timestamp!r} for {p}; "
                    f"available: {', '.join(t for t, _ in snaps)}")
        ts, src = match[-1]
    else:
        ts, src = snaps[-1]
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, p)
    except OSError as e:
        return f"ERROR: restore failed: {e}"
    return f"rolled back {p} to {ts}"


# -- wrapping ------------------------------------------------------------
_EDIT_TOOL_NAMES = ("write_file", "edit_file", "MultiEdit")


def _wrap_edit_tool(tool: Any, session_id: Optional[str]) -> Any:
    """Wrap a mutating file tool's handler to snapshot before editing.

    ``Tool`` is a frozen dataclass, so this returns a replacement ``Tool``
    with a wrapped handler (same name/description/parameters/risk) — the
    caller swaps it into ``agent.tools``. Idempotent: a tool is never
    wrapped twice. A failed snapshot never blocks the edit — backing up
    is best-effort, applying the edit is not.
    """
    if getattr(tool, _WRAPPED_ATTR, False):
        return tool
    orig: Callable[..., str] = tool.handler

    def handler(*args: Any, **kwargs: Any) -> str:
        path = args[0] if args else None
        if not isinstance(path, str):
            for key in ("path", "file_path", "filepath", "file", "target"):
                v = kwargs.get(key)
                if isinstance(v, str):
                    path = v
                    break
        if isinstance(path, str) and path:
            snapshot_file(path, session_id)
        return orig(*args, **kwargs)

    # Preserve the dataclass repr/name for debugging.
    handler.__name__ = getattr(orig, "__name__", "handler")  # type: ignore[attr-defined]
    new_tool = dataclasses.replace(tool, handler=handler)
    object.__setattr__(new_tool, _WRAPPED_ATTR, True)
    return new_tool


def make_edit_rollback_tool(session_id: Optional[str] = None) -> "Tool":
    def edit_rollback(path: str, timestamp: str = "") -> str:
        """Restore a file from its edit-backup snapshot."""
        return rollback(path, session_id, timestamp)

    return Tool(
        "EditRollback",
        "Restore a file to a pre-edit snapshot taken by the edit-backup "
        "checkpoint. Empty timestamp restores the most recent snapshot; "
        "list_snapshots via file_info if unsure.",
        {"type": "object",
         "properties": {
             "path": {"type": "string",
                      "description": "file to restore"},
             "timestamp": {"type": "string",
                           "description": "snapshot timestamp prefix "
                                          "(default: most recent)"}},
         "required": ["path"]},
        edit_rollback,
    )


def register(agent: Any) -> None:
    """Wrap write_file/edit_file/MultiEdit + add the EditRollback tool.

    Duck-typed like the other feature modules: needs ``agent.tools``
    (dict) and optionally ``agent.session_id``. The wrapper is
    idempotent, and snapshot failures never block an edit.
    """
    session_id = getattr(agent, "session_id", None)
    tools: Dict[str, Any] = getattr(agent, "tools", {})
    for name in _EDIT_TOOL_NAMES:
        tool = tools.get(name)
        if tool is not None and callable(getattr(tool, "handler", None)):
            tools[name] = _wrap_edit_tool(tool, session_id)
    tools["EditRollback"] = make_edit_rollback_tool(session_id)
    agent.edit_backup_session = _safe_session_id(session_id)


if __name__ == "__main__":
    import tempfile

    def check(name: str, cond: bool, extra: str = "") -> None:
        print(("PASS" if cond else "FAIL"), "-", name, extra)
        if not cond:
            raise SystemExit(f"self-test failed: {name}")

    # Isolate the snapshot store for the self-test.
    tmp = tempfile.mkdtemp(prefix="editbackup_selftest_")
    SNAPSHOT_BASE = os.path.join(tmp, "snapshots")

    CONTENT_A = "line one\nline two\nline three\n"
    CONTENT_B = "CHANGED line one\nline two\nline three\nextra line\n"

    class FakeAgent:
        def __init__(self):
            self.session_id = "selftest-session"
            self.tools = {}

    def fake_write_file(path: str, content: str) -> str:
        Path(path).write_text(content, encoding="utf-8")
        return "wrote"

    def fake_edit_file(path: str, old_string: str, new_string: str,
                       replace_all: bool = False) -> str:
        p = Path(path)
        p.write_text(p.read_text(encoding="utf-8").replace(
            old_string, new_string, 1 if not replace_all else -1),
            encoding="utf-8")
        return "edited"

    def fake_multi_edit(path: str, edits) -> str:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        for e in edits:
            text = text.replace(e["old_string"], e["new_string"], 1)
        p.write_text(text, encoding="utf-8")
        return f"Applied {len(edits)} edits"

    from fullagent.tools import Tool as _T, RISK_CONFIRM

    a = FakeAgent()
    a.tools["write_file"] = _T("write_file", "d", {}, fake_write_file,
                               risk=RISK_CONFIRM)
    a.tools["edit_file"] = _T("edit_file", "d", {}, fake_edit_file,
                              risk=RISK_CONFIRM)
    a.tools["MultiEdit"] = _T("MultiEdit", "d", {}, fake_multi_edit,
                              risk=RISK_CONFIRM)
    register(a)

    check("EditRollback tool registered", "EditRollback" in a.tools)

    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "victim.txt")
        Path(f).write_text(CONTENT_A, encoding="utf-8")

        # 1. write_file through the wrapper snapshots first, then edits
        out = a.tools["write_file"].handler(f, CONTENT_B)
        check("wrapped write applies", out == "wrote"
              and Path(f).read_text(encoding="utf-8") == CONTENT_B)
        snaps = list_snapshots(f, "selftest-session")
        check("write_file snapshotted the original", len(snaps) == 1, snaps)
        check("snapshot bytes equal content A",
              Path(snaps[0][1]).read_bytes() == CONTENT_A.encode("utf-8"))

        # 2. EditRollback restores byte-identical content A
        out = a.tools["EditRollback"].handler(f)
        check("rollback message format",
              out.startswith("rolled back ") and " to " in out, out)
        check("file byte-identical to A after rollback",
              Path(f).read_bytes() == CONTENT_A.encode("utf-8"))

        # 3. edit_file wrapper snapshots too
        a.tools["edit_file"].handler(f, "line two", "LINE TWO")
        check("edit_file applied",
              "LINE TWO" in Path(f).read_text(encoding="utf-8"))
        check("edit_file also snapshotted",
              len(list_snapshots(f, "selftest-session")) == 2)

        # 4. rollback with explicit timestamp
        ts = list_snapshots(f, "selftest-session")[1][0]
        out = a.tools["EditRollback"].handler(f, ts)
        check("explicit-timestamp rollback",
              out == f"rolled back {Path(f).resolve()} to {ts}", out)
        check("explicit rollback restores A",
              Path(f).read_bytes() == CONTENT_A.encode("utf-8"))

        # 5. MultiEdit wrapper snapshots
        n0 = len(list_snapshots(f, "selftest-session"))
        a.tools["MultiEdit"].handler(
            f, [{"old_string": "line one", "new_string": "ONE"}])
        check("MultiEdit wrapped and snapshotted",
              len(list_snapshots(f, "selftest-session")) == n0 + 1)

        # 6. new files skip snapshotting
        g = os.path.join(d, "brand_new.txt")
        a.tools["write_file"].handler(g, "fresh content")
        check("new file skips snapshot",
              list_snapshots(g, "selftest-session") == [])

        # 7. rollback with no snapshots is a clean error, never raises
        out = a.tools["EditRollback"].handler(g)
        check("rollback with no snapshots errors cleanly",
              out.startswith("ERROR:"), out)

    # 8. retention pruning: 25 snapshots → ≤20 kept
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "prune.txt")
        Path(f).write_text("v0", encoding="utf-8")
        for i in range(25):
            Path(f).write_text(f"v{i}", encoding="utf-8")
            snapshot_file(f, "selftest-session")
        snaps = list_snapshots(f, "selftest-session")
        check("retention keeps <=20", len(snaps) == 20,
              f"got {len(snaps)}")
        # newest survive: the last snapshot's bytes are the final content
        check("newest snapshot survives",
              Path(snaps[0][1]).read_bytes() == b"v24")

    # 9. register() is idempotent: no double-wrapping
    h0 = a.tools["write_file"].handler
    register(a)
    check("double register does not re-wrap",
          a.tools["write_file"].handler is h0)

    # 10. session isolation: different session ids go to different dirs
    b = FakeAgent()
    b.session_id = "other-session"
    b.tools["write_file"] = _T("write_file", "d", {}, fake_write_file)
    register(b)
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "iso.txt")
        Path(f).write_text("x", encoding="utf-8")
        b.tools["write_file"].handler(f, "y")
        other_dir = Path(tmp) / "snapshots" / "other-session"
        self_dir = Path(tmp) / "snapshots" / "selftest-session"
        b_snaps = [p for p in other_dir.rglob("*.bak")]
        # the other session's snapshot does not leak into this session's dir
        check("snapshot lands in its own session dir",
              len(b_snaps) == 1
              and not any(p.name.startswith("iso.txt.")
                          for p in self_dir.rglob("*.bak")))
        # but rollback still finds it (cross-session safety net)
        check("rollback finds cross-session snapshot",
              a.tools["EditRollback"].handler(f).startswith("rolled back "))

    print("editbackup self-test PASSED")
