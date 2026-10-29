"""File change detection: stat + content fingerprint tracking between read/edit.

Incident (512s turn): the agent's ``old_string`` didn't match the file at
edit time — likely the file changed between the agent's read and its edit
(external edit, another tool, a rerun). The failure surfaced as a cryptic
"old_string not found" instead of pointing at the real cause.

This module keeps a tiny per-agent store:
``{realpath: (mtime_ns, size, content_digest)}``. ``read_file`` records the
baseline when the agent reads; ``edit_file`` / ``multi_edit`` re-check before
applying and, on mismatch, return a clear warning telling the agent to
re-read and retry instead of failing cryptically. The content digest matters
because some filesystems only expose coarse mtimes: a same-size rewrite within
the timestamp tick would otherwise look unchanged. After the agent's own
write/edit, the new baseline is recorded so its own writes never trip the
check.
"""

from __future__ import annotations

import hashlib
import os
import threading
import weakref
from datetime import datetime

_lock = threading.Lock()
# agent -> {realpath: (mtime_ns, size, content_digest)}. WeakKeyDictionary so
# dead agents don't pin their stores. Agents that can't be weak-referenced (or
# a None agent with no current agent set) fall back to _GLOBAL_STORE.
_agent_stores: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_id_stores: dict[int, dict[str, tuple[int, int, str]]] = {}
_GLOBAL_STORE: dict[str, tuple[int, int, str]] = {}
_current: "weakref.ref | None" = None


def register(agent) -> None:
    """Give an agent its own file-stat store. Called once per agent.

    Never raises: this must never break agent startup.
    """
    try:
        with _lock:
            try:
                store = _agent_stores.get(agent)
                if store is None:
                    store = {}
                    _agent_stores[agent] = store
            except TypeError:
                # not weak-referenceable: fall back to id-keyed store
                store = _id_stores.setdefault(id(agent), {})
            try:
                agent._file_stats = store
            except Exception:
                pass
            global _current
            try:
                _current = weakref.ref(agent)
            except TypeError:
                _current = None
    except Exception:
        pass


def clear(agent) -> None:
    """Drop the agent's file-stat baselines (e.g. on ``/reset``).

    Never raises: a failed cache clear must never kill a turn.
    """
    try:
        with _lock:
            store = None
            try:
                store = _agent_stores.get(agent)
            except TypeError:
                store = _id_stores.get(id(agent))
            if store is not None:
                store.clear()
            try:
                # keep the attribute pointing at the same (now empty) store
                if store is not None:
                    agent._file_stats = store
            except Exception:
                pass
    except Exception:
        pass


def _store_for(agent) -> dict[str, tuple[int, int, str]]:
    """Resolve the stat store for an agent (or the current/global one)."""
    with _lock:
        if agent is not None:
            try:
                store = _agent_stores.get(agent)
                if store is None:
                    store = {}
                    _agent_stores[agent] = store
                return store
            except TypeError:
                return _id_stores.setdefault(id(agent), {})
        a = _current() if _current is not None else None
        if a is not None:
            try:
                store = _agent_stores.get(a)
                if store is None:
                    store = {}
                    _agent_stores[a] = store
                return store
            except TypeError:
                return _id_stores.setdefault(id(a), {})
        return _GLOBAL_STORE


def _key(path: str) -> str:
    try:
        return os.path.realpath(path)
    except Exception:
        return os.path.abspath(path)


def _stat(path: str) -> "tuple[int, int, str] | None":
    """Return an identity for the current file content.

    mtime/size alone are not enough on filesystems with coarse timestamp
    granularity: a same-length rewrite can land in the same timestamp tick.
    A bounded-memory content digest closes that false-negative window.
    """
    try:
        st = os.stat(path)
        if not os.path.isfile(path):
            return None
        digest = hashlib.blake2b(digest_size=16)
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return (st.st_mtime_ns, st.st_size, digest.hexdigest())
    except OSError:
        return None


def note_read(agent, path: str) -> None:
    """Record the current stat of a file the agent just read."""
    note_written(agent, path)


def note_written(agent, path: str) -> None:
    """Record the current stat after the agent's own write/edit.

    Same as note_read: the agent knows this exact file state, so its own
    changes must never look like an external modification.
    """
    cur = _stat(path)
    if cur is None:
        return
    try:
        _store_for(agent)[_key(path)] = cur
    except Exception:
        pass


def _fmt_mtime(mtime_ns: int) -> str:
    try:
        return datetime.fromtimestamp(mtime_ns / 1e9).strftime(
            "%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(mtime_ns)


def check_stale(agent, path: str) -> str:
    """Return a warning if the file changed since the agent last read it.

    Returns "" when there's nothing to warn about: the agent never read
    this file (no baseline), the file can't be stat'ed (let the caller's
    own error handle it), or the stat is unchanged.
    """
    try:
        store = _store_for(agent)
        key = _key(path)
        old = store.get(key)
        if old is None:
            return ""
        cur = _stat(path)
        if cur is None:
            return ""
        if cur == old:
            return ""
        # Tolerate two-field baselines created by an older process/version.
        old_mtime, old_size = old[0], old[1]
        new_mtime, new_size = cur[0], cur[1]
        # Refresh the baseline to the newest observed state so a second
        # edit attempt doesn't re-warn about the same external change.
        store[key] = cur
        return (
            "WARNING: File changed since you read it "
            f"(mtime {_fmt_mtime(old_mtime)} → {_fmt_mtime(new_mtime)}, "
            f"size {old_size} → {new_size}). "
            "Re-read the region and retry — do NOT guess old_string from "
            "stale content."
        )
    except Exception:
        return ""


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    def check(name: str, cond: bool) -> None:
        print(("PASS" if cond else "FAIL"), "-", name)
        if not cond:
            raise SystemExit(f"self-test failed: {name}")

    # import through the package so __main__ works from repo root
    from fullagent import tools as _tools

    class FakeAgent:
        def __init__(self):
            self.tools = {}

    with tempfile.TemporaryDirectory() as d:
        # --- 1. edit_file: unchanged file -> no warning, edit applies ---
        f = Path(d) / "f.txt"
        f.write_text("hello world\nsecond line\n", encoding="utf-8")
        out = _tools.read_file(str(f))
        check("read_file ok", out.startswith("["))
        out = _tools.edit_file(str(f), "hello world", "HELLO WORLD")
        check("unchanged file: no stale warning",
              "changed since you read it" not in out)
        check("unchanged file: edit applied",
              out.startswith("Updated") and "HELLO WORLD" in
              f.read_text(encoding="utf-8"))

        # --- 2. edit_file: external change -> clear warning, not cryptic ---
        g = Path(d) / "g.txt"
        g.write_text("alpha one\nbeta two\n", encoding="utf-8")
        _tools.read_file(str(g))                      # agent reads
        g.write_text("alpha one\nEXTERNAL EDIT\nbeta two\n",  # changed
                     encoding="utf-8")                 # externally
        out = _tools.edit_file(str(g), "alpha one\nbeta two",
                               "should not apply")
        check("stale file: warning mentions the change",
              "changed since you read it" in out)
        check("stale file: not the cryptic error",
              "old_string not found" not in out)
        check("stale file: no write happened",
              "EXTERNAL EDIT" in g.read_text(encoding="utf-8") and
              "should not apply" not in g.read_text(encoding="utf-8"))

        # --- 3. multi_edit: external change -> warning, atomic no-op ---
        h = Path(d) / "h.txt"
        h.write_text("one\ntwo\nthree\n", encoding="utf-8")
        from fullagent.multiedit import multi_edit
        _tools.read_file(str(h))
        h.write_text("one\nTWO CHANGED\nthree\n", encoding="utf-8")
        out = multi_edit(str(h), [{"old_string": "two",
                                   "new_string": "2"}])
        check("multi_edit stale: warning mentions the change",
              "changed since you read it" in out)
        check("multi_edit stale: file untouched",
              h.read_text(encoding="utf-8") == "one\nTWO CHANGED\nthree\n")

        # --- 4. multi_edit: unchanged -> applies, no warning ---
        out = multi_edit(str(h), [{"old_string": "TWO CHANGED",
                                   "new_string": "two!"}])
        check("multi_edit unchanged: no warning",
              "changed since you read it" not in out)
        check("multi_edit unchanged: applied",
              "two!" in h.read_text(encoding="utf-8"))

        # --- 5. never-read file -> edit_file behaves as before ---
        n = Path(d) / "never.txt"
        n.write_text("aaa\n", encoding="utf-8")
        out = _tools.edit_file(str(n), "aaa", "bbb")
        check("never-read file: normal edit, no warning",
              out.startswith("Updated") and
              "changed since you read it" not in out)

        # --- 6. agent-scoped store via register() ---
        a = FakeAgent()
        register(a)
        check("register sets agent._file_stats",
              isinstance(getattr(a, "_file_stats", None), dict))
        note_read(a, str(n))
        check("note_read populates agent store",
              len(a._file_stats) == 1)
        n.write_text("aaa\nmore\n", encoding="utf-8")
        warn = check_stale(a, str(n))
        check("agent-scoped stale warning", "changed since you read it"
              in warn and "mtime" in warn)
        check("stale warning names mtimes",
              "→" in warn and "size" in warn)

        # --- 7. size-only change still warns (content same length trick) ---
        s = Path(d) / "s.txt"
        s.write_text("xy\n", encoding="utf-8")
        note_read(None, str(s))
        s.write_text("zz\n", encoding="utf-8")  # same size, new mtime
        check("mtime-only change warns",
              "changed since you read it" in check_stale(None, str(s)))

    print("ALL SELF-TESTS PASSED")
