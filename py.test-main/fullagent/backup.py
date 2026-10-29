"""Backup / restore — real tar.gz snapshots of the ``~/.fullagent`` home dir.

Uses stdlib ``tarfile`` only. A backup captures everything under the app
dir (``$FULLAGENT_HOME`` or ``~/.fullagent``): config files, sessions,
``jobs.db``, todos, skills, commands, ``hooks.json`` — excluding the
``backups/`` and ``logs/`` directories themselves. A ``MANIFEST.json``
with version + file list + timestamp is written as the first entry of
every archive.

Tools (registered by :func:`register`):
    - ``BackupCreate``  — create a backup, returns the real path + size.
    - ``BackupList``    — list backups with size + date.
    - ``BackupRestore`` — restore from a backup. Safety: extracts into a
      temp dir first, verifies the manifest and path safety, then copies
      into place. Refuses to overwrite existing files without
      ``confirm=True``.

TUI: ``/backup [create|list]``, ``/restore <file> [--yes]``.

``python3 -m fullagent.backup`` runs the built-in self-test, which builds
a fake ``~/.fullagent`` tree in a temp dir, runs a real backup, lists it
via ``tarfile``, corrupts a file, restores, and verifies the original
content is restored byte-identically.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .tools import RISK_CONFIRM, RISK_SAFE, Tool

MANIFEST_NAME = "MANIFEST.json"
BACKUP_VERSION = 1

# Directories at the top of the app dir that are never backed up:
# backups/ (the archives themselves) and logs/ (volatile, huge).
EXCLUDE_DIRS = {"backups", "logs"}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _app_dir() -> Path:
    """Resolve the app dir at call time (not import time).

    This lets ``$FULLAGENT_HOME`` be changed between calls (the self-test
    points it at a temp dir) and always reflects the current environment.
    """
    from . import config
    return config._pick_app_dir()


def _backups_dir(app_dir: Optional[Path] = None) -> Path:
    d = (app_dir or _app_dir()) / "backups"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def _iter_backup_files(app_dir: Path) -> List[Tuple[Path, str]]:
    """Return ``[(absolute path, archive name)]`` for everything under
    *app_dir* except the excluded top-level dirs."""
    out: List[Tuple[Path, str]] = []
    for root, dirs, names in os.walk(app_dir):
        root_p = Path(root)
        rel_root = root_p.relative_to(app_dir).as_posix()
        # Prune excluded top-level directories.
        dirs[:] = [d for d in dirs
                   if not (rel_root == "." and d in EXCLUDE_DIRS)]
        for name in sorted(names):
            full = root_p / name
            arcname = (root_p / name).relative_to(app_dir).as_posix()
            if full.is_file() or full.is_symlink():
                out.append((full, arcname))
    return out


# ---------------------------------------------------------------------------
# Core: create / list / restore
# ---------------------------------------------------------------------------

def create_backup(app_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Create a real ``backup-<timestamp>.tar.gz`` and return metadata.

    Returns ``{"path", "size_bytes", "file_count", "timestamp", "name"}``.
    """
    app_dir = app_dir or _app_dir()
    bdir = _backups_dir(app_dir)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = bdir / f"backup-{ts}.tar.gz"
    # Avoid same-second collisions.
    i = 2
    while dest.exists():
        dest = bdir / f"backup-{ts}-{i}.tar.gz"
        i += 1

    files = _iter_backup_files(app_dir)
    manifest = {
        "version": BACKUP_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "app_dir": str(app_dir),
        "files": [arc for _, arc in files],
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

    # Write to a temp file and rename on success: a crash or full disk
    # mid-archive used to leave a truncated backup-*.tar.gz that
    # list_backups() then advertised as a real backup.
    fd, tmp_name = tempfile.mkstemp(dir=str(bdir),
                                    prefix=dest.name + ".", suffix=".tmp")
    os.close(fd)
    tmp_dest = Path(tmp_name)
    try:
        with tarfile.open(tmp_dest, "w:gz") as tar:
            # MANIFEST.json first, so it can be read without scanning the
            # whole archive on restore.
            info = tarfile.TarInfo(MANIFEST_NAME)
            info.size = len(manifest_bytes)
            info.mtime = int(time.time())
            with tempfile.SpooledTemporaryFile() as buf:
                buf.write(manifest_bytes)
                buf.seek(0)
                tar.addfile(info, buf)  # type: ignore[arg-type]
            for full, arcname in files:
                tar.add(full, arcname=arcname, recursive=False)
        os.replace(tmp_dest, dest)
    except BaseException:
        try:
            tmp_dest.unlink()
        except OSError:
            pass
        raise

    return {
        "name": dest.name,
        "path": str(dest),
        "size_bytes": dest.stat().st_size,
        "file_count": len(files),
        "timestamp": ts,
        "manifest": manifest,
    }


def list_backups(app_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """List backups with size + date, newest first."""
    bdir = _backups_dir(app_dir)
    out: List[Dict[str, Any]] = []
    for p in sorted(bdir.glob("backup-*.tar.gz"),
                    key=lambda q: q.stat().st_mtime, reverse=True):
        st = p.stat()
        out.append({
            "name": p.name,
            "path": str(p),
            "size_bytes": st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime)
                               .isoformat(timespec="seconds"),
        })
    return out


def _resolve_backup_ref(ref: str, app_dir: Path) -> Path:
    """Resolve a backup ref: absolute path, relative path, or bare name in
    the backups dir."""
    p = Path(os.path.expanduser(ref))
    if p.is_absolute() or (os.sep in ref or "/" in ref):
        return p
    candidate = app_dir / "backups" / ref
    if candidate.exists():
        return candidate
    return p  # let the caller report "not found"


def _safe_members(tar: tarfile.TarFile) -> Tuple[List[tarfile.TarInfo],
                                                 List[str]]:
    """Check every member for path-traversal hazards.

    Returns (members, problems). A problem list of any length means the
    archive is rejected.
    """
    members: List[tarfile.TarInfo] = []
    problems: List[str] = []
    for m in tar.getmembers():
        if m.name == MANIFEST_NAME:
            members.append(m)
            continue
        p = Path(m.name)
        if p.is_absolute() or ".." in p.parts:
            problems.append(f"unsafe member path: {m.name!r}")
            continue
        if m.islnk() or m.issym():
            # Symlinks are only accepted when their target is relative and
            # stays inside the archive (no ".."), so restoring can never
            # write outside the app dir through a link.
            target = Path(m.linkname)
            if target.is_absolute() or ".." in target.parts:
                problems.append(f"unsafe link target: {m.name!r} -> "
                                f"{m.linkname!r}")
                continue
            members.append(m)
            continue
        members.append(m)
    return members, problems


def restore_backup(ref: str, confirm: bool = False,
                   app_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Restore a backup into the app dir.

    Safety:
      1. Extract into a temp dir first — nothing touches the real home.
      2. Verify ``MANIFEST.json`` exists and parses (version + file list).
      3. Reject path-traversal / link members.
      4. Refuse to overwrite existing files unless ``confirm=True``.

    Returns ``{"ok", "restored_files", "skipped_existing"|..., "message"}``.
    """
    app_dir = app_dir or _app_dir()
    path = _resolve_backup_ref(ref, app_dir)
    if not path.exists():
        return {"ok": False,
                "message": f"backup not found: {ref} "
                           f"(looked in {app_dir / 'backups'})"}

    tmp = Path(tempfile.mkdtemp(prefix="fullagent-restore-"))
    try:
        try:
            with tarfile.open(path, "r:gz") as tar:
                members, problems = _safe_members(tar)
                if problems:
                    return {"ok": False,
                            "message": "refusing to restore — unsafe "
                                       f"archive: {'; '.join(problems)}"}
                # Verify the manifest before touching anything real.
                try:
                    raw = tar.extractfile(MANIFEST_NAME)
                except KeyError:
                    raw = None
                if raw is None:
                    return {"ok": False,
                            "message": "refusing to restore — archive has "
                                       "no MANIFEST.json"}
                try:
                    manifest = json.loads(raw.read().decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    return {"ok": False,
                            "message": "refusing to restore — MANIFEST.json "
                                       "is corrupt"}
                archived = {m.name for m in members
                            if m.name != MANIFEST_NAME
                            and (m.isfile() or m.issym())}
                listed = set(manifest.get("files", []))
                if archived != listed:
                    return {"ok": False,
                            "message": "refusing to restore — manifest file "
                                       "list does not match archive contents"}
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(tmp, members=members, filter="data")
                else:
                    tar.extractall(tmp, members=members)
        except (tarfile.TarError, OSError, EOFError, TypeError) as e:
            return {"ok": False,
                    "message": f"refusing to restore — cannot read archive: {e}"}

        # Confirm check: refuse to overwrite existing files without consent.
        targets = [(m, tmp / m.name, app_dir / m.name)
                   for m in members if m.isfile() or m.issym()]
        existing = [str(dst) for _, _, dst in targets
                    if dst.exists() or dst.is_symlink()]
        if existing and not confirm:
            return {"ok": False,
                    "message": "refusing to overwrite existing files "
                               f"({len(existing)} exist, e.g. {existing[0]}) — "
                               "pass confirm=True to restore anyway"}

        restored = 0
        for member, src, dst in targets:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if member.issym():
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(member.linkname)
            else:
                shutil.copy2(src, dst)
            restored += 1
        return {"ok": True,
                "restored_files": restored,
                "message": f"restored {restored} file(s) from {path.name} "
                           f"into {app_dir}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _handle_backup_create() -> str:
    info = create_backup()
    return (f"Backup created: {info['path']}\n"
            f"size: {_fmt_size(info['size_bytes'])} "
            f"({info['size_bytes']} bytes), "
            f"{info['file_count']} files")


def _handle_backup_list() -> str:
    items = list_backups()
    if not items:
        return "No backups yet — run /backup or the BackupCreate tool."
    lines = ["Backups:"]
    for b in items:
        lines.append(f"  {b['name']}  {_fmt_size(b['size_bytes'])}  "
                     f"{b['modified']}\n    {b['path']}")
    return "\n".join(lines)


def _handle_backup_restore(path: str, confirm: bool = False) -> str:
    res = restore_backup(path, confirm=confirm)
    status = "✓" if res.get("ok") else "✗"
    return f"{status} {res.get('message')}"


def make_backup_create_tool() -> Tool:
    return Tool(
        name="BackupCreate",
        description=("Create a real tar.gz backup of the ~/.fullagent home "
                     "directory (config files, sessions, jobs.db, todos, "
                     "skills, commands, hooks.json — everything except "
                     "backups/ and logs/). Returns the backup path and size."),
        parameters={"type": "object", "properties": {}},
        handler=_handle_backup_create,
        risk=RISK_SAFE,
    )


def make_backup_list_tool() -> Tool:
    return Tool(
        name="BackupList",
        description=("List existing backups with size and date, newest "
                     "first."),
        parameters={"type": "object", "properties": {}},
        handler=_handle_backup_list,
        risk=RISK_SAFE,
    )


def make_backup_restore_tool() -> Tool:
    return Tool(
        name="BackupRestore",
        description=("Restore a backup (backup file path or bare name from "
                     "the backups dir) into the ~/.fullagent home directory. "
                     "Refuses to overwrite existing files unless "
                     "confirm=true."),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "Backup file path or bare backup "
                                        "name (e.g. backup-20261009-120000."
                                        "tar.gz)."},
                "confirm": {"type": "boolean",
                            "description": "Set true to allow overwriting "
                                           "existing files."},
            },
            "required": ["path"],
        },
        handler=_handle_backup_restore,
        risk=RISK_CONFIRM,
    )


def register(agent: Any) -> None:
    """Wire BackupCreate / BackupList / BackupRestore into an agent."""
    agent.tools["BackupCreate"] = make_backup_create_tool()
    agent.tools["BackupList"] = make_backup_list_tool()
    agent.tools["BackupRestore"] = make_backup_restore_tool()


# ---------------------------------------------------------------------------
# TUI entry points
# ---------------------------------------------------------------------------

BACKUP_USAGE = (
    "usage:\n"
    "  /backup            create a backup of ~/.fullagent\n"
    "  /backup list       list existing backups\n"
    "  /restore <file> [--yes]   restore a backup "
    "(--yes allows overwriting)"
)


def handle_backup(ui: Any, arg: str) -> None:
    """Handle ``/backup [create|list]``; prints via the TUI."""
    agent = getattr(ui, "agent", None)
    if agent is None:
        ui.print_error("/backup needs the TUI host agent")
        return
    tokens = (arg or "").strip().split()
    action = tokens[0].lower() if tokens else "create"
    if action == "list":
        out = _handle_backup_list()
        ui.print_info(out)
        return
    if action not in ("create", ""):
        ui.print_error(BACKUP_USAGE)
        return
    try:
        out = _handle_backup_create()
    except OSError as e:
        ui.print_error(f"backup failed: {e}")
        return
    ui.print_info(f"✓ {out}")


def handle_restore(ui: Any, arg: str) -> None:
    """Handle ``/restore <file> [--yes]``; prints via the TUI."""
    agent = getattr(ui, "agent", None)
    if agent is None:
        ui.print_error("/restore needs the TUI host agent")
        return
    tokens = (arg or "").strip().split()
    yes = any(t in ("--yes", "-y") for t in tokens)
    refs = [t for t in tokens if t not in ("--yes", "-y")]
    if not refs:
        ui.print_error(BACKUP_USAGE)
        return
    ui.print_info(_handle_backup_restore(refs[0], confirm=yes))


# ---------------------------------------------------------------------------
# self-test: `python3 -m fullagent.backup` → PASS
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sqlite3

    # Isolate: point $FULLAGENT_HOME at a fake ~/.fullagent tree.
    tmp = tempfile.mkdtemp(prefix="backup_selftest_")
    os.environ["FULLAGENT_HOME"] = tmp
    app = Path(tmp)

    # -- fake home structure ------------------------------------------------
    (app / "config.json").write_text(json.dumps({"model": "m1"}))
    (app / "sessions").mkdir()
    (app / "sessions" / "sess-1.json").write_text(
        json.dumps({"messages": ["hi"]}))
    conn = sqlite3.connect(app / "jobs.db")
    conn.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO jobs VALUES (1, 'nightly')")
    conn.commit()
    conn.close()
    (app / "todos").mkdir()
    (app / "todos" / "t1.json").write_text(json.dumps({"done": []}))
    (app / "skills").mkdir()
    (app / "skills" / "mine.md").write_text("# my skill")
    (app / "commands").mkdir()
    (app / "commands" / "foo.md").write_text("/foo cmd")
    (app / "hooks.json").write_text(json.dumps({"pre": "echo x"}))
    # A relative symlink inside the tree must round-trip as a symlink.
    os.symlink("mine.md", app / "skills" / "alias.md")
    # Excluded content (must NOT land in the tar).
    (app / "logs").mkdir()
    (app / "logs" / "debug.log").write_text("noise\n" * 1000)
    (app / "backups").mkdir()
    (app / "backups" / "stale.tar.gz").write_bytes(b"not-a-real-backup")

    # Snapshot originals for byte-identity checks after restore.
    originals = {p: p.read_bytes()
                 for p in app.rglob("*") if p.is_file()
                 and "backups" not in p.parts and "logs" not in p.parts}

    # -- BackupCreate: real tar.gz ------------------------------------------
    info = create_backup()
    bkp = Path(info["path"])
    assert bkp.exists() and bkp.suffixes == [".tar", ".gz"], bkp
    assert info["size_bytes"] > 0, info
    assert info["file_count"] == len(originals), (info, len(originals))
    print(f"BackupCreate → {bkp} ({_fmt_size(info['size_bytes'])}, "
          f"{info['file_count']} files)")

    # -- the archive really is a tar.gz with the expected members -----------
    with tarfile.open(bkp, "r:gz") as tar:
        names = tar.getnames()
    assert MANIFEST_NAME in names, names
    expected = {p.relative_to(app).as_posix() for p in originals}
    archived = {n for n in names if n != MANIFEST_NAME}
    assert archived == expected, (archived ^ expected)
    for bad in ("logs/debug.log", "backups/stale.tar.gz",
                "backups/" + bkp.name):
        assert bad not in names, f"excluded file leaked into backup: {bad}"
    print(f"tar members OK ({len(names)} entries); "
          f"backups/ + logs/ correctly excluded")

    # -- MANIFEST.json ------------------------------------------------------
    with tarfile.open(bkp, "r:gz") as tar:
        raw = tar.extractfile(MANIFEST_NAME)
        assert raw is not None
        manifest = json.loads(raw.read().decode("utf-8"))
    assert manifest["version"] == BACKUP_VERSION, manifest
    assert "created" in manifest and manifest["created"], manifest
    assert set(manifest["files"]) == expected, "manifest file list mismatch"
    print(f"MANIFEST.json OK (version={manifest['version']}, "
          f"{len(manifest['files'])} files listed)")

    # -- BackupList ----------------------------------------------------------
    items = list_backups()
    assert any(i["path"] == str(bkp) for i in items), items
    first = items[0]
    assert first["size_bytes"] == info["size_bytes"], first
    assert first["modified"], first
    print(f"BackupList → {len(items)} backup(s), newest {first['name']} "
          f"{_fmt_size(first['size_bytes'])}")

    # -- register() wiring ----------------------------------------------------
    class FakeAgent:
        def __init__(self):
            self.tools = {}

    fa = FakeAgent()
    register(fa)
    for tool_name in ("BackupCreate", "BackupList", "BackupRestore"):
        assert tool_name in fa.tools, tool_name
    print("register(agent) wires BackupCreate/BackupList/BackupRestore")

    # -- corrupt a file, restore WITHOUT confirm → must refuse ---------------
    corrupted = app / "config.json"
    corrupted.write_bytes(b"CORRUPTED-DATA")
    res = restore_backup(bkp.name, confirm=False)
    assert not res["ok"], res
    assert "confirm" in res["message"].lower(), res["message"]
    assert corrupted.read_bytes() == b"CORRUPTED-DATA", "must not overwrite"
    print(f"refusal without confirm OK: {res['message'][:80]}…")

    # -- restore WITH confirm → byte-identical --------------------------------
    res = restore_backup(bkp.name, confirm=True)
    assert res["ok"], res
    for p, original in originals.items():
        assert p.read_bytes() == original, f"mismatch after restore: {p}"
    print(f"restore OK ({res['restored_files']} files), all byte-identical")

    # -- corrupt the sqlite db too, restore again -----------------------------
    db = app / "jobs.db"
    db.write_bytes(b"JUNK")
    res = restore_backup(str(bkp), confirm=True)
    assert res["ok"], res
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT name FROM jobs").fetchall()
    conn.close()
    assert rows == [("nightly",)], rows
    print("sqlite jobs.db restored and queryable again")

    # -- symlink round-trip -------------------------------------------------
    alias = app / "skills" / "alias.md"
    assert alias.is_symlink(), "symlink must round-trip as a symlink"
    assert os.readlink(alias) == "mine.md", os.readlink(alias)
    print("relative symlink restored as symlink")

    # -- restore of a non-existent backup -------------------------------------
    res = restore_backup("no-such-backup.tar.gz", confirm=True)
    assert not res["ok"] and "not found" in res["message"], res
    print("missing-backup error OK")

    # -- restore rejects a poisoned archive ----------------------------------
    poison = app / "backups" / "poison.tar.gz"
    with tarfile.open(poison, "w:gz") as tar:
        mi = tarfile.TarInfo(MANIFEST_NAME)
        body = json.dumps({"version": BACKUP_VERSION,
                           "created": "x", "files": ["../evil.txt"]}
                          ).encode()
        mi.size = len(body)
        tar.addfile(mi, io.BytesIO(body))
        ei = tarfile.TarInfo("../evil.txt")
        eb = b"evil"
        ei.size = len(eb)
        tar.addfile(ei, io.BytesIO(eb))
    res = restore_backup(str(poison), confirm=True)
    assert not res["ok"] and "unsafe" in res["message"], res["message"]
    assert not (app / "evil.txt").exists(), "traversal must not write"
    print("poisoned-archive rejection OK")

    shutil.rmtree(tmp, ignore_errors=True)
    print("backup self-test: PASS")
