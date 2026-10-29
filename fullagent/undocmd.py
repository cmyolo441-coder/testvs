"""TUI ``/undo`` command — undo the last file write/edit.

Claude Code CLI-style: restores the files a mutating tool touched to
their pre-tool state, using the ``snapshot.taken`` events the agent
logs before every mutating tool call (Axiom A2: every mutation is
reversible — see :mod:`fullagent.snapshots`).

Undo is READ-ONLY on history: snapshots are never deleted, so a later
redo-style feature can still reach them. Re-running ``/undo``
immediately is a harmless no-op — it re-applies the same newest
snapshot.

Usage from the TUI::

    /undo    restore the files from the last file-mutating tool call

``python3 -m fullagent.undocmd`` runs the built-in self-test.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

# Tool names (as logged in snapshot.taken's ``before_tool``) whose
# snapshots cover file mutations. Matches agent._MUTATING_TOOLS minus
# the command runners (run_command / live_shell), whose snapshots sweep
# the whole cwd instead of targeting the files a tool touched.
# Compared case-insensitively — MultiEdit registers with a capital M.
_FILE_MUTATION_TOOLS = frozenset({
    "write_file", "edit_file", "apply_patch", "multiedit",
    "create_directory", "delete_path", "copy_path", "move_path",
})

_NOTHING_TO_UNDO = "nothing to undo — no file writes or edits yet"


def _display(path: Path, cwd: Path) -> str:
    """Short display name: relative to the cwd when possible."""
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def _under_cwd(path: Path, cwd: Path) -> bool:
    """Resolved path stays inside the working directory (follows
    symlinks, so a symlink escaping the cwd is refused)."""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == cwd or cwd in resolved.parents


def _restore_tree(store: Any, manifest: dict, cwd: Path) -> dict:
    """Materialise a tree manifest restricted to paths under ``cwd``.

    Same blob-restore semantics as SnapshotStore.materialise, but any
    path resolving outside the working directory is refused (reported,
    never touched). Returns a summary dict.
    """
    restored: list[str] = []
    removed: list[str] = []
    skipped: list[str] = []
    missing = 0
    for key, blob in manifest.items():
        p = Path(key)
        if not _under_cwd(p, cwd):
            skipped.append(key)
            continue
        try:
            rp = p.resolve()
        except OSError:
            missing += 1
            continue
        if blob is None:
            # The file did not exist at snapshot time: it was created
            # by the undone mutation, so remove it if still present.
            try:
                if rp.is_symlink() or rp.is_file():
                    rp.unlink()
                    removed.append(_display(rp, cwd))
                elif rp.is_dir():
                    shutil.rmtree(rp, ignore_errors=True)
                    removed.append(_display(rp, cwd))
            except OSError:
                missing += 1
            continue
        data = store.get_blob(blob)
        if data is None:
            missing += 1
            continue
        try:
            current = rp.read_bytes() if rp.is_file() else None
        except OSError:
            current = None
        if current == data:
            continue  # already matches the snapshot — nothing to do
        try:
            rp.parent.mkdir(parents=True, exist_ok=True)
            # Append, don't replace, the suffix (see snapshots.py:
            # with_suffix("") is a no-op and would clobber the target).
            tmp = rp.with_name(rp.name + ".undo-tmp")
            tmp.write_bytes(data)
            os.replace(tmp, rp)
            restored.append(_display(rp, cwd))
        except OSError:
            missing += 1
    return {"restored": restored, "removed": removed,
            "skipped": skipped, "missing_blobs": missing}


def _summarise(result: dict) -> str:
    restored = result["restored"]
    removed = result["removed"]
    skipped = result["skipped"]
    missing = result["missing_blobs"]
    parts: list[str] = []
    if restored:
        parts.append(
            f"Restored {len(restored)} file"
            f"{'s' if len(restored) != 1 else ''}: "
            + ", ".join(restored))
    if removed:
        parts.append(
            f"Removed {len(removed)} created file"
            f"{'s' if len(removed) != 1 else ''}: "
            + ", ".join(removed))
    if missing:
        parts.append(
            f"({missing} file{'s' if missing != 1 else ''} could not be "
            "restored — snapshot data missing, left as-is)")
    if skipped:
        parts.append(
            f"(refused {len(skipped)} path{'s' if len(skipped) != 1 else ''} "
            "outside the working directory)")
    if not parts:
        return "nothing changed — files already match the last snapshot"
    return "\n".join(parts)


def undo_last(agent: Any) -> str:
    """Restore the files from the most recent file-mutating tool call.

    Finds the newest ``snapshot.taken`` event whose ``before_tool`` is a
    file mutation (write_file/edit_file/apply_patch/multiedit, …),
    restores that tree — files under the current working directory
    only — and returns a human-readable summary like
    ``"Restored 2 files: a.py, b.py"``.

    Snapshots are never deleted (redo stays possible). Paths resolving
    outside the cwd are refused, never restored.
    """
    store = getattr(agent, "store", None)
    log = getattr(agent, "log", None)
    if store is None or log is None:
        return "undo unavailable: this agent has no snapshot store"

    try:
        events = log.events()
    except Exception:  # noqa: BLE001 — a broken log must not crash /undo
        events = []
    target = None
    for ev in reversed(events):
        if getattr(ev, "type", None) != "snapshot.taken":
            continue
        data = getattr(ev, "data", None) or {}
        before = str(data.get("before_tool", "") or "").strip().lower()
        if before in _FILE_MUTATION_TOOLS:
            target = ev
            break
    if target is None:
        return _NOTHING_TO_UNDO

    tree_hash = str((target.data or {}).get("tree", "") or "")
    manifest = None
    try:
        manifest = store.load_tree(tree_hash)
    except Exception:  # noqa: BLE001
        manifest = None
    if not manifest:
        return ("nothing to undo — the last snapshot "
                f"({tree_hash[:10]}…) no longer resolves")

    cwd = Path.cwd().resolve()
    return _summarise(_restore_tree(store, manifest, cwd))


def handle_undo(ui: Any, arg: str) -> None:
    """TUI ``/undo`` handler: undo the last file write/edit.

    Prints what was restored via ``ui.print_info`` (or the
    nothing-to-undo message). Never raises — one filesystem hiccup must
    not kill the TUI session.
    """
    agent = getattr(ui, "agent", None)
    if agent is None:
        ui.print_error("/undo needs the TUI host agent")
        return
    try:
        ui.print_info(undo_last(agent))
    except Exception as e:  # noqa: BLE001 — the UI must survive
        ui.print_error(f"/undo failed: {e}")


def register(agent: Any) -> None:
    """Wire ``/undo`` support into an agent (duck-typed).

    Undo needs no agent-side state — everything is read on demand from
    ``agent.store`` (SnapshotStore) and ``agent.log`` (snapshot.taken
    events). This exists so the feature-module loop in
    ``Agent._register_feature_modules`` picks the module up like every
    other feature module.
    """
    agent.undo_supported = True


# ---------------------------------------------------------------------------
# self-test: `python3 -m fullagent.undocmd`  →  PASS
# ---------------------------------------------------------------------------

def _selftest() -> None:
    import tempfile

    from .kernel import EventLog
    from .snapshots import SnapshotStore

    tmp = Path(tempfile.mkdtemp(prefix="undo_selftest_")).resolve()
    store = SnapshotStore(tmp / "store")
    log = EventLog(tmp / "log.jsonl")

    class FakeAgent:
        pass

    agent = FakeAgent()
    agent.store = store
    agent.log = log

    class FakeUI:
        def __init__(self):
            self.agent = agent
            self.infos: list[str] = []
            self.errors: list[str] = []

        def print_info(self, text: str, *a: Any) -> None:
            self.infos.append(text)

        def print_error(self, text: str, *a: Any) -> None:
            self.errors.append(text)

    # 1. nothing to undo yet
    assert "nothing to undo" in undo_last(agent), undo_last(agent)

    old_cwd = os.getcwd()
    work = tmp / "work"
    work.mkdir()
    os.chdir(work)
    try:
        a = work / "a.py"
        b = work / "b.py"
        c = work / "c.py"  # does not exist yet
        a.write_bytes(b"original a\n")
        b.write_bytes(b"original b\n")

        # 2. snapshot BEFORE the mutation (what _execute_tool does);
        #    c.py is absent, so its creation is part of the mutation.
        snap = store.take([str(a), str(b), str(c)])
        log.append("snapshot.taken",
                   {"tree": snap["tree"], "paths": list(snap["paths"]),
                    "before_tool": "write_file"})

        # 3. the mutation: edit both files + create c.py
        a.write_bytes(b"MUTATED a\n" * 100)
        b.write_bytes(b"MUTATED b\n")
        c.write_bytes(b"created by mutation\n")

        # 4. undo restores byte-for-byte, removes the created file
        out = undo_last(agent)
        assert "Restored 2 files: a.py, b.py" in out, out
        assert "Removed 1 created file: c.py" in out, out
        assert a.read_bytes() == b"original a\n"
        assert b.read_bytes() == b"original b\n"
        assert not c.exists()

        # 5. idempotent: undo again changes nothing
        out = undo_last(agent)
        assert "nothing changed" in out, out
        assert a.read_bytes() == b"original a\n"

        # 6. snapshots are never deleted by undo — blobs still resolve
        assert store.get_blob(snap["paths"][str(a)]) == b"original a\n"
        assert store.load_tree(snap["tree"]) is not None

        # 7. non-file-mutation snapshots are skipped: a run_command
        #    snapshot taken after the write must NOT become the undo
        #    target — the write_file snapshot stays newest-file-mutation.
        snap_cmd = store.take([str(a)])
        log.append("snapshot.taken",
                   {"tree": snap_cmd["tree"],
                    "paths": list(snap_cmd["paths"]),
                    "before_tool": "run_command"})
        a.write_bytes(b"changed again\n")
        out = undo_last(agent)
        assert "Restored 1 file: a.py" in out, out
        assert a.read_bytes() == b"original a\n"

        # 8. safety: absolute paths outside the cwd are refused.
        #    Revert the run_command snapshot first so the outside-cwd
        #    snapshot becomes the newest file-mutation target.
        outside = tmp / "outside.txt"
        outside.write_bytes(b"do not touch\n")
        snap_out = store.take([str(outside)])
        log.append("snapshot.taken",
                   {"tree": snap_out["tree"],
                    "paths": list(snap_out["paths"]),
                    "before_tool": "edit_file"})
        # undo of an older (in-cwd) mutation is shadowed by the newer
        # outside-cwd one, which must be refused rather than restored.
        outside.write_bytes(b"mutated outside\n")
        out = undo_last(agent)
        assert outside.read_bytes() == b"mutated outside\n", out
        assert "refused 1 path" in out and "outside the working directory" \
            in out, out

        # 9. handle_undo prints via ui.print_info
        ui = FakeUI()
        handle_undo(ui, "")
        assert ui.infos and "refused 1 path" in ui.infos[-1], ui.infos
        assert not ui.errors

        # 10. handle_undo with no agent -> print_error, no raise
        ui2 = FakeUI()
        ui2.agent = None
        handle_undo(ui2, "")
        assert ui2.errors and "host agent" in ui2.errors[0], ui2.errors

        # 11. register() exposes the module to the feature loop
        register(agent)
        assert agent.undo_supported is True
    finally:
        os.chdir(old_cwd)

    print("PASS")


if __name__ == "__main__":
    _selftest()
