"""Snapshot Store — content-addressed, deduplicated file snapshots (§8.3).

Axiom A2: every mutation is reversible. No byte is written by a mutating
tool without a committed recovery path. This module is that recovery path.

Design (pure Python, stdlib only):
  * Blobs are zlib-compressed and addressed by sha256(bytes) — identical
    content collapses automatically, so a 100-step session editing the same
    files stores each unique version once.
  * A snapshot is a TREE: a manifest mapping absolute path -> blob hash.
    Trees are stored as JSON objects in the same CAS.
  * The log records a 'snapshot.taken' event (tree hash + paths + the seq
    of the action it precedes), so rewind/revert are folds: find the newest
    snapshot at/below the target seq and materialise its tree.
  * diff(tree_a, tree_b) computes added/removed/modified — the confirmation
    delta shown before a rewind (§9.2 step 4).

Layout under <root>/store/:
    objects/xx/<hash>      compressed blobs (file contents, tree manifests)
"""

from __future__ import annotations

import hashlib
import json
import os
import zlib
from pathlib import Path

from .kernel import EventLog
from ._foundation import get_logger

_log = get_logger("snapshots")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# a valid object id is always a 64-char lowercase hex digest — anything
# else is corruption or an attack, never a real blob
_VALID_HASH = frozenset("0123456789abcdef")


def _valid_hash(h: str) -> bool:
    return len(h) == 64 and all(c in _VALID_HASH for c in h)


class SnapshotStore:
    """Content-addressed blob + tree storage. Stateless beyond the disk."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)

    # -- blobs ---------------------------------------------------------------

    def _obj_path(self, h: str) -> Path:
        if not _valid_hash(h):
            # corrupt manifests / hostile input must not turn into
            # filesystem path traversal (e.g. h="../../etc/x")
            raise ValueError(f"invalid blob hash: {h!r}")
        return self.objects / h[:2] / h[2:]

    def put_blob(self, data: bytes) -> str:
        """Store bytes, deduplicated by content hash. Returns the hash."""
        h = _hash_bytes(data)
        p = self._obj_path(h)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            import threading
            tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_bytes(zlib.compress(data, 6))
            os.replace(tmp, p)  # atomic: a crash never leaves a torn blob
            # fsync the DIRECTORY entry: without it a crash can lose the
            # rename itself, silently dropping a snapshot this store
            # exists to guarantee (Axiom A2: every mutation reversible)
            try:
                fd = os.open(p.parent, os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError:
                pass
        return h

    def get_blob(self, h: str) -> bytes | None:
        if not _valid_hash(h):
            return None
        p = self._obj_path(h)
        if not p.exists():
            return None
        try:
            return zlib.decompress(p.read_bytes())
        except (OSError, zlib.error):
            return None

    # -- trees ---------------------------------------------------------------

    def take(self, paths: list[str]) -> dict:
        """Snapshot the given file paths NOW. Returns the tree manifest:
        {"tree": <hash>, "paths": {abs_path: blob_hash}}.

        Missing files are recorded as None (their absence is part of the
        state — restoring the tree removes files created later).
        Directories are not snapshotted as content; only regular files.
        """
        manifest: dict[str, str | None] = {}
        for raw in paths:
            p = Path(raw).expanduser().resolve()
            key = str(p)
            if p.is_file():
                try:
                    manifest[key] = self.put_blob(p.read_bytes())
                except OSError:
                    manifest[key] = None
            else:
                manifest[key] = None
        tree_bytes = json.dumps(manifest, sort_keys=True,
                                ensure_ascii=False).encode("utf-8")
        tree_hash = self.put_blob(tree_bytes)
        return {"tree": tree_hash, "paths": manifest}

    def load_tree(self, tree_hash: str) -> dict[str, str | None] | None:
        """Load a tree manifest by its hash. None if it does not resolve."""
        data = self.get_blob(tree_hash)
        if data is None:
            return None
        try:
            manifest = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        return manifest if isinstance(manifest, dict) else None

    def materialise(self, tree_hash: str) -> dict:
        """Restore the filesystem to a tree's state.

        Files present in the tree are restored from blobs; paths recorded
        as absent (None) are deleted if they now exist (they were created
        after the snapshot). Returns a summary dict.
        """
        manifest = self.load_tree(tree_hash)
        if manifest is None:
            return {"restored": 0, "removed": 0, "missing_blobs": 0,
                    "error": f"tree {tree_hash[:10]} does not resolve"}
        restored = removed = missing = 0
        for key, blob in manifest.items():
            p = Path(key)
            if blob is None:
                # file did not exist at snapshot time. is_symlink() is
                # required: a DANGLING symlink created after the snapshot
                # makes exists() false, so it used to survive the restore.
                if p.exists() or p.is_symlink():
                    try:
                        if p.is_dir() and not p.is_symlink():
                            import shutil
                            shutil.rmtree(p, ignore_errors=True)
                        else:
                            p.unlink()
                        removed += 1
                    except OSError:
                        pass
                continue
            data = self.get_blob(blob)
            if data is None:
                missing += 1
                continue
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                # Append, don't replace, the suffix: paths with no existing
                # extension (Makefile, LICENSE, ...) used to collide because
                # with_suffix("") is a no-op, so tmp == p and the atomic
                # write silently clobbered the destination before replace.
                import threading
                tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.snap-tmp")
                tmp.write_bytes(data)
                os.replace(tmp, p)
                restored += 1
            except OSError:
                missing += 1
        return {"restored": restored, "removed": removed,
                "missing_blobs": missing}

    def diff(self, tree_a: str, tree_b: str) -> dict:
        """Paths added / removed / modified going from tree A to tree B.
        A None entry means the path did not exist at snapshot time."""
        a = self.load_tree(tree_a) or {}
        b = self.load_tree(tree_b) or {}
        added = sorted(p for p in b
                       if b[p] is not None and a.get(p) is None)
        removed = sorted(p for p in a
                         if a[p] is not None and b.get(p) is None)
        modified = sorted(p for p in a if p in b
                          and a[p] is not None and b[p] is not None
                          and a[p] != b[p])
        return {"added": added, "removed": removed, "modified": modified}

    # -- GC --------------------------------------------------------------------

    def reachable_hashes(self, log: EventLog) -> set[str]:
        """Every blob/tree hash referenced by any snapshot event (I9)."""
        from .kernel import fold
        snaps = [s for s in fold(log).snapshots if isinstance(s, dict)]
        # fold() only sees the in-RAM retention window of the checked-out
        # branch. Snapshot events deeper in the archive (or taken on
        # another branch) are still reachable: collecting their blobs
        # would make those snapshots unrestorable. The JSONL archive is
        # append-only and complete, so scan it and merge the references.
        try:
            with log.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if '"snapshot.taken"' not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(rec, dict) or \
                            rec.get("type") != "snapshot.taken":
                        continue
                    data = rec.get("data")
                    if isinstance(data, dict):
                        snaps.append(data)
        except (OSError, UnicodeDecodeError):
            pass
        keep: set[str] = set()
        for snap in snaps:
            tree = snap.get("tree")
            if tree:
                keep.add(tree)
                manifest = self.load_tree(tree) or {}
                keep.update(h for h in manifest.values() if h)
        return keep

    def gc(self, log: EventLog) -> int:
        """Mark-and-sweep: delete blobs no reachable snapshot references.
        Never collects a blob referenced by a reachable event (I9).

        Blobs created AFTER the sweep started are never deleted, even if
        the keep-set (computed at sweep start) does not list them yet —
        otherwise a concurrent take() racing the gc could lose a blob
        whose snapshot event had not been sealed when reachable_hashes()
        ran.
        """
        import time
        # nanosecond clock BEFORE the keep-set is computed: a blob born
        # after this instant is concurrent with the sweep and must survive
        # (see docstring). Starting the clock after reachable_hashes()
        # left a take() sealed in that gap deletable.
        started_ns = time.time_ns()
        keep = self.reachable_hashes(log)
        deleted = 0
        if not self.objects.exists():
            return 0
        for sub in self.objects.iterdir():
            if not sub.is_dir():
                continue
            for f in sub.iterdir():
                h = sub.name + f.name
                if f.name.endswith(".tmp"):
                    # abandoned atomic-write leftovers — sweep them too,
                    # otherwise they accumulate forever. A NEW tmp file is
                    # a concurrent put_blob() mid-write: unlinking it makes
                    # that os.replace fail and take() silently records the
                    # file as absent, so only sweep old ones.
                    try:
                        if f.stat().st_mtime_ns >= started_ns:
                            continue
                        f.unlink()
                        deleted += 1
                    except OSError:
                        pass
                elif h not in keep:
                    try:
                        # newborn blob racing this gc: leave it alone
                        if f.stat().st_mtime_ns >= started_ns:
                            continue
                    except OSError:
                        continue
                    try:
                        f.unlink()
                        deleted += 1
                    except OSError:
                        pass
        return deleted


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = SnapshotStore(root / "store")
        work = root / "work"
        work.mkdir()
        f1 = work / "a.txt"
        f2 = work / "b.txt"
        f1.write_text("version one")

        # take: existing file + missing file
        snap1 = store.take([str(f1), str(f2)])
        assert snap1["paths"][str(f1)] is not None
        assert snap1["paths"][str(f2)] is None

        # dedup: identical content -> identical blob hash
        h1 = store.put_blob(b"version one")
        assert h1 == snap1["paths"][str(f1)]

        # mutate both files, take a second snapshot
        f1.write_text("version two")
        f2.write_text("created later")
        snap2 = store.take([str(f1), str(f2)])

        # diff sees the modification and the addition
        d = store.diff(snap1["tree"], snap2["tree"])
        assert str(f1) in d["modified"], d
        assert str(f2) in d["added"], d

        # materialise snap1: f1 reverts, f2 (absent in snap1) is removed
        res = store.materialise(snap1["tree"])
        assert f1.read_text() == "version one"
        assert not f2.exists()
        assert res["restored"] == 1 and res["removed"] == 1, res

        # materialise snap2 again: forward travel works too
        store.materialise(snap2["tree"])
        assert f1.read_text() == "version two"
        assert f2.read_text() == "created later"

        # GC keeps referenced blobs, sweeps orphans
        log = EventLog(root / "log.jsonl")
        log.append("snapshot.taken", {"tree": snap2["tree"],
                                      "paths": list(snap2["paths"])})
        orphan = store.put_blob(b"unreferenced garbage")
        deleted = store.gc(log)
        assert deleted >= 1
        assert store.get_blob(orphan) is None
        assert store.get_blob(snap2["paths"][str(f1)]) is not None
        # the referenced tree still materialises after GC
        f1.unlink()
        store.materialise(snap2["tree"])
        assert f1.read_text() == "version two"

    print("SNAPSHOT SELF-TEST PASS")
