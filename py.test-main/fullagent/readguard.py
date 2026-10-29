"""Read-before-edit enforcement: ground every edit in a fresh read.

Incident it prevents: the agent issues ``edit_file`` / ``MultiEdit`` with a
stale or guessed ``old_string`` (often without having read the file this
turn). The edit fails with "old_string not found in file", the model retries
blindly, and the loop burns minutes (observed: 512 s of retries).

This module wraps the file tools on an agent:

* ``read_file`` — after a successful read, records the file's
  ``(mtime_ns, size)`` in ``agent.readguard_seen`` (per-turn map).
* ``write_file`` / ``edit_file`` / ``MultiEdit`` — before editing an
  *existing* file, check the map. If the file was not read this turn, or it
  changed on disk since the read, auto-read it (bounded) first, record it,
  and proceed with the edit. Never fails: the guard only *grounds*, it never
  blocks. New files (not on disk yet) skip the guard.
* ``agent.run_turn`` is wrapped so the map resets at the start of each turn.

The wrapped ``Tool`` objects keep their name, description, parameters and
risk — only the handler is swapped, so the model-visible surface is
unchanged.
"""

from __future__ import annotations

import functools
import os
from dataclasses import replace as _replace_tool
from typing import Any, Callable

SEEN_ATTR = "readguard_seen"
# read_file limit bounds: read_file itself caps limit at 5000 lines.
_AUTO_READ_LIMIT = 5000
# Marker set on wrapped handlers so register() stays idempotent.
_WRAPPED_MARK = "__readguard_wrapped__"


def _norm(path: str) -> str:
    """Canonical absolute path for the seen-map key."""
    try:
        return os.path.realpath(os.path.expanduser(path))
    except (OSError, TypeError, ValueError):
        return str(path)


def _fingerprint(path: str):
    """Return (mtime_ns, size) for an existing file, else None."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _seen(agent: Any) -> dict:
    """The per-turn read map; created lazily."""
    seen = getattr(agent, SEEN_ATTR, None)
    if seen is None:
        seen = {}
        setattr(agent, SEEN_ATTR, seen)
    return seen


def _record(agent: Any, path: str) -> None:
    fp = _fingerprint(path)
    if fp is not None:
        _seen(agent)[_norm(path)] = fp


def _is_fresh(agent: Any, path: str) -> bool:
    """True iff the file was read this turn and is unchanged since."""
    key = _norm(path)
    fp = _seen(agent).get(key)
    if fp is None:
        return False
    current = _fingerprint(path)
    return current is not None and current == fp


def _extract_path(args: tuple, kwargs: dict) -> str | None:
    """Pull the target file path from write/edit/MultiEdit call args.

    ``write_file``/``edit_file`` use ``path``; ``MultiEdit`` uses
    ``file_path``.
    """
    if args:
        return args[0]
    for name in ("path", "file_path"):
        if name in kwargs:
            return kwargs[name]
    return None


def _wrap_reader(agent: Any, tool):
    """Return a new Tool with the read handler recording reads."""
    orig: Callable = tool.handler
    if getattr(orig, _WRAPPED_MARK, False):
        return tool

    @functools.wraps(orig)
    def read_guarded(*args, **kwargs):
        result = orig(*args, **kwargs)
        path = args[0] if args else kwargs.get("path")
        if (isinstance(result, str) and not result.startswith("ERROR")
                and isinstance(path, str) and path):
            _record(agent, _norm(path))
        return result

    read_guarded.__dict__[_WRAPPED_MARK] = True
    # Tool is a frozen dataclass: build a new instance (name, description,
    # parameters, risk unchanged).
    return _replace_tool(tool, handler=read_guarded)


def _auto_read(agent: Any, orig_reader: Callable | None, path: str) -> None:
    """Bounded grounding read before an edit. Never raises, never fails."""
    if orig_reader is None:
        return
    try:
        orig_reader(path, 1, _AUTO_READ_LIMIT)
    except Exception:
        pass


def _wrap_editor(agent: Any, tool, orig_reader: Callable | None):
    """Return a new Tool whose edit handler auto-reads ungrounded files."""
    orig: Callable = tool.handler
    if getattr(orig, _WRAPPED_MARK, False):
        return tool

    @functools.wraps(orig)
    def edit_guarded(*args, **kwargs):
        path = _extract_path(args, kwargs)
        normed = _norm(path) if isinstance(path, str) and path else None
        existed_before = (normed is not None and os.path.exists(normed))
        if existed_before:
            # Existing file: must be grounded in a fresh read first.
            if not _is_fresh(agent, normed):
                # Auto-read (bounded) so the *model* sees fresh content
                # this turn; never fails the edit if the read errors.
                _auto_read(agent, orig_reader, path)
                _record(agent, normed)
        result = orig(*args, **kwargs)
        # Our own successful write keeps the file fresh, so consecutive
        # edits by the agent don't trigger a redundant re-read.
        if (normed is not None and existed_before
                and isinstance(result, str)
                and not result.startswith("ERROR")):
            _record(agent, normed)
        return result

    edit_guarded.__dict__[_WRAPPED_MARK] = True
    # Tool is a frozen dataclass: build a new instance (name, description,
    # parameters, risk unchanged).
    return _replace_tool(tool, handler=edit_guarded)


def register(agent: Any) -> None:
    """Install read-before-edit guards on an agent (duck-typed, no imports).

    Wraps ``read_file``, ``write_file``, ``edit_file`` and ``MultiEdit``
    (where present) in ``agent.tools``, initialises
    ``agent.readguard_seen``, and wraps ``agent.run_turn`` so the map
    resets each turn. Idempotent.

    Ordering: run this *after* feature modules that add edit tools
    (e.g. ``multiedit``), so their tools get wrapped too.
    """
    if getattr(agent, "__readguard_registered__", False):
        return
    tools = getattr(agent, "tools", {}) or {}

    reader_tool = tools.get("read_file")
    orig_reader = None
    if reader_tool is not None:
        orig_reader = reader_tool.handler
        tools["read_file"] = _wrap_reader(agent, reader_tool)

    for name in ("write_file", "edit_file", "MultiEdit"):
        tool = tools.get(name)
        if tool is not None:
            tools[name] = _wrap_editor(agent, tool, orig_reader)

    # Reset the seen-map at the start of every turn.
    run_turn = getattr(agent, "run_turn", None)
    if callable(run_turn) and not getattr(run_turn, _WRAPPED_MARK, False):
        @functools.wraps(run_turn)
        def run_turn_guarded(*args, **kwargs):
            try:
                getattr(agent, SEEN_ATTR, {}).clear()
            except Exception:
                pass
            return run_turn(*args, **kwargs)

        run_turn_guarded.__dict__[_WRAPPED_MARK] = True
        agent.run_turn = run_turn_guarded

    agent.__readguard_registered__ = True


if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fullagent.tools import build_registry
    from fullagent.multiedit import register as multiedit_register

    failures = []

    def check(name: str, cond: bool) -> None:
        print(("PASS" if cond else "FAIL"), "-", name)
        if not cond:
            failures.append(name)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        class FakeAgent:
            pass

        agent = FakeAgent()
        agent.tools = build_registry()
        multiedit_register(agent)   # adds "MultiEdit"
        register(agent)

        # Spy on the guard's auto-read hook (module-global rebinding).
        auto_reads = {"n": 0}
        _real_auto_read = _auto_read

        def _spy_auto_read(a, reader, path):
            auto_reads["n"] += 1
            return _real_auto_read(a, reader, path)

        globals()["_auto_read"] = _spy_auto_read
        try:
            f = d / "target.txt"
            f.write_text("line one\nline two\nline three\n", encoding="utf-8")

            # (i) edit WITHOUT a prior read: guard must auto-read, then the
            # edit succeeds.
            check("no prior read recorded", not _is_fresh(agent, str(f)))
            out = agent.tools["edit_file"].handler(
                str(f), "line two", "LINE TWO")
            check("edit after auto-read succeeds",
                  out.startswith("OK") or out.startswith("Updated"))
            check("content actually changed",
                  "LINE TWO" in f.read_text(encoding="utf-8"))
            check("auto-read happened exactly once", auto_reads["n"] == 1)
            check("file now recorded as fresh", _is_fresh(agent, str(f)))

            # (iii) already-read file is NOT re-read.
            auto_reads["n"] = 0
            out = agent.tools["edit_file"].handler(
                str(f), "LINE TWO", "line 2")
            check("second edit succeeds",
                  out.startswith("OK") or out.startswith("Updated"))
            check("fresh file not re-read", auto_reads["n"] == 0)

            # Stale read: file changed on disk since the read -> re-read.
            auto_reads["n"] = 0
            f.write_text("totally new content\n", encoding="utf-8")
            out = agent.tools["edit_file"].handler(
                str(f), "totally new content", "replaced")
            check("stale file re-read then edited", auto_reads["n"] == 1)
            check("stale edit succeeded",
                  f.read_text(encoding="utf-8") == "replaced\n")

            # (ii) new-file write is unaffected by the guard.
            auto_reads["n"] = 0
            newf = d / "brand_new.txt"
            out = agent.tools["write_file"].handler(str(newf), "hello\n")
            check("new-file write succeeds",
                  out.startswith("OK: created") and newf.exists())
            check("new-file write triggers no read", auto_reads["n"] == 0)

            # write_file OVERWRITING an existing unread file also auto-reads.
            auto_reads["n"] = 0
            existing = d / "overwrite.txt"
            existing.write_text("old stuff\n", encoding="utf-8")
            out = agent.tools["write_file"].handler(str(existing),
                                                    "new stuff\n")
            check("overwrite of unread file succeeds",
                  out.startswith("Updated") and
                  existing.read_text(encoding="utf-8") == "new stuff\n")
            check("overwrite auto-read the file", auto_reads["n"] == 1)

            # MultiEdit on an unread file: auto-read then atomic edit.
            auto_reads["n"] = 0
            m = d / "multi.txt"
            m.write_text("aaa\nbbb\n", encoding="utf-8")
            out = agent.tools["MultiEdit"].handler(
                str(m), [{"old_string": "aaa", "new_string": "AAA"}])
            check("MultiEdit succeeds after auto-read",
                  out.startswith("Applied 1 edit"))
            check("MultiEdit auto-read happened", auto_reads["n"] == 1)

            # read_file still records; a normal explicit read marks fresh.
            agent.tools["read_file"].handler(str(m))
            check("explicit read recorded", _is_fresh(agent, str(m)))

            # Turn reset: wrapping run_turn clears the map. Needs a truly
            # fresh agent + fresh (unwrapped) registry, because the read
            # guard closures bind to the agent they were registered on.
            class TurnAgent(FakeAgent):
                def run_turn(self, user_text):
                    return f"turn:{user_text}"

            import fullagent.tools as _toolsmod
            _toolsmod._REGISTRY = None
            ta = TurnAgent()
            ta.tools = _toolsmod.build_registry()
            register(ta)
            g = d / "turn.txt"
            g.write_text("x\n", encoding="utf-8")
            ta.tools["read_file"].handler(str(g))
            check("read recorded before turn", _is_fresh(ta, str(g)))
            check("run_turn still works",
                  ta.run_turn("hi") == "turn:hi")
            check("turn reset clears seen-map", not _is_fresh(ta, str(g)))

            # register() is idempotent.
            register(agent)
            register(agent)
            check("idempotent register", True)
        finally:
            globals()["_auto_read"] = _real_auto_read

    if failures:
        raise SystemExit(f"self-test failed: {failures}")
    print("ALL SELF-TESTS PASSED")
