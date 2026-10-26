"""Side-effect-free identity for final Git-visible working-tree content."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final, Protocol

from .working_tree_fingerprint_content import ContentRace, overlay

WORKING_TREE_CONTENT_FINGERPRINT_SCHEMA: Final = "working_tree_content_fingerprint/v1"
GIT_TIMEOUT_SECONDS: Final = 15
MAX_GIT_CALLS: Final = 10


class WorkingTreeFingerprintState(StrEnum):
    CLEAN = "clean"
    DIRTY = "dirty"
    UNSUPPORTED = "unsupported"
    TIMED_OUT = "timed_out"
    NOT_A_REPOSITORY = "not_a_repository"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class WorkingTreeFingerprint:
    """A complete workspace identity or an explicit non-authoritative outcome."""

    state: WorkingTreeFingerprintState
    fingerprint: str | None
    head_tree: str | None
    git_calls: int

    @property
    def authoritative(self) -> bool:
        """Whether this value is safe to use as a local freshness authority."""
        return self.state in {WorkingTreeFingerprintState.CLEAN, WorkingTreeFingerprintState.DIRTY}


class GitCompletedProcess(Protocol):
    """The read-only subset of ``subprocess.CompletedProcess`` this collector uses."""

    returncode: int
    stdout: bytes | str | None


class GitRunner(Protocol):
    """Injectable bounded Git runner for deterministic collector tests."""

    def __call__(
        self,
        args: Sequence[str | bytes],
        *,
        cwd: str,
        text: bool,
        capture_output: bool,
        timeout: float,
        env: dict[str, str],
        input: bytes | None = None,
    ) -> GitCompletedProcess: ...


def working_tree_content_fingerprint(
    repo_root: str | Path | None = None,
    *,
    runner: GitRunner | None = None,
    timeout_seconds: int = GIT_TIMEOUT_SECONDS,
) -> WorkingTreeFingerprint:
    """Collect one final-content identity without touching the checkout's Git state.

    A temporary copied index and temporary object directory isolate all plumbing.
    Clean worktrees hash the committed tree only after status proves cleanliness;
    dirty worktrees stream only changed/untracked paths into a canonical overlay.
    Unsupported Git features fail closed instead of silently falling back to HEAD.
    """
    root = Path(repo_root).expanduser() if repo_root is not None else Path.cwd()
    calls = 0
    active_runner = runner or _run_git

    def git(
        args: list[str | bytes], *, env: dict[str, str] | None = None, input: bytes | None = None
    ) -> bytes | None:
        nonlocal calls
        calls += 1
        try:
            completed = active_runner(
                ["git", "-c", "core.fsmonitor=false", "-c", "core.filemode=true", "--no-optional-locks", *args],
                cwd=str(root),
                text=False,
                capture_output=True,
                timeout=timeout_seconds,
                env=env or _isolated_environment(),
                input=input,
            )
        except subprocess.TimeoutExpired:
            raise
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        value = completed.stdout or b""
        return value.encode() if isinstance(value, str) else bytes(value)

    try:
        top_level = git(["rev-parse", "--show-toplevel"])
        if top_level is None:
            return _result(WorkingTreeFingerprintState.NOT_A_REPOSITORY, calls=calls)
        resolved_root = Path(os.fsdecode(_line_output(top_level)))
        if resolved_root.resolve() != root.resolve():
            root = resolved_root
        index_name = git(["rev-parse", "--git-path", "index"])
        objects_name = git(["rev-parse", "--git-path", "objects"])
        content_config = git(["config", "--null", "--get-regexp", r"^core\.(sparsecheckout|autocrlf)$"])
        split = git(["config", "--bool", "--get", "core.splitIndex"])
        if index_name is None or objects_name is None:
            return _result(WorkingTreeFingerprintState.UNSUPPORTED, calls=calls)
        if _unsupported_content_config(content_config) or _enabled(split):
            return _result(WorkingTreeFingerprintState.UNSUPPORTED, calls=calls)
        index_path = _git_path(root, index_name)
        objects_path = _git_path(root, objects_name)
        if not index_path.is_file() or not objects_path.is_dir():
            return _result(WorkingTreeFingerprintState.UNSUPPORTED, calls=calls)
        with TemporaryDirectory(prefix="omh-working-tree-") as temporary:
            temporary_root = Path(temporary)
            temporary_index = temporary_root / "index"
            temporary_objects = temporary_root / "objects"
            _ = temporary_objects.mkdir()
            # Git's racy-clean check needs the index's original timestamp, not
            # the copy time. Observe it first so a concurrent index replacement
            # cannot lend a newer timestamp to older copied stat-cache entries.
            index_metadata = index_path.stat()
            _ = shutil.copyfile(index_path, temporary_index)
            os.utime(temporary_index, ns=(index_metadata.st_atime_ns, index_metadata.st_mtime_ns))
            environment = _isolated_environment(
                index=temporary_index,
                objects=temporary_objects,
                alternate_objects=objects_path,
            )
            index_entries = git(["ls-files", "-v", "-s", "-z"], env=environment)
            head_tree = git(["rev-parse", "HEAD^{tree}"], env=environment)
            if index_entries is None or head_tree is None:
                return _result(WorkingTreeFingerprintState.UNSUPPORTED, calls=calls)
            index_paths = _index_paths(index_entries)
            head = _object_id(head_tree)
            attributes = git(
                ["check-attr", "-a", "-z", "--stdin"],
                env=environment,
                input=_nul_terminated(index_paths),
            )
            if attributes is None or _has_filter(attributes):
                return _result(WorkingTreeFingerprintState.UNSUPPORTED, head, calls)
            status = git(["status", "--porcelain=v1", "-z", "--untracked-files=all"], env=environment)
            if status is None:
                return _result(WorkingTreeFingerprintState.UNSUPPORTED, head, calls)
            paths = _status_paths(status)
            if not paths:
                return _identity(head, head, dirty=False, calls=calls)
            tree = git(["ls-tree", "-r", "-z", "HEAD"], env=environment)
            if tree is None:
                return _result(WorkingTreeFingerprintState.UNSUPPORTED, head, calls)
            head_entries = _tree_entries(tree)
            try:
                changed_overlay = overlay(root, paths, head_entries)
            except ContentRace:
                return _result(WorkingTreeFingerprintState.UNAVAILABLE, head, calls)
            if changed_overlay is None:
                return _result(WorkingTreeFingerprintState.UNSUPPORTED, head, calls)
            final_tree = _canonical_tree(head_entries, changed_overlay)
            return _identity(final_tree, head, dirty=True, calls=calls)
    except subprocess.TimeoutExpired:
        return _result(WorkingTreeFingerprintState.TIMED_OUT, calls=calls)
    except (OSError, UnicodeError, ValueError):
        return _result(WorkingTreeFingerprintState.UNAVAILABLE, calls=calls)


def _run_git(
    args: Sequence[str | bytes],
    *,
    cwd: str,
    text: bool,
    capture_output: bool,
    timeout: float,
    env: dict[str, str],
    input: bytes | None = None,
) -> GitCompletedProcess:
    return subprocess.run(
        list(args), cwd=cwd, text=text, capture_output=capture_output, timeout=timeout, env=env, input=input
    )


def _isolated_environment(
    *, index: Path | None = None,
    objects: Path | None = None,
    alternate_objects: Path | None = None,
) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    if index is not None and objects is not None and alternate_objects is not None:
        environment["GIT_INDEX_FILE"] = str(index)
        environment["GIT_OBJECT_DIRECTORY"] = str(objects)
        environment["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = str(alternate_objects)
    return environment


def _git_path(root: Path, output: bytes) -> Path:
    value = Path(os.fsdecode(_line_output(output)))
    return value if value.is_absolute() else root / value


def _line_output(output: bytes) -> bytes:
    """Remove Git's one record terminator without changing a newline path byte."""
    return output[:-1] if output.endswith(b"\n") else output


def _object_id(output: bytes) -> str:
    value = _line_output(output)
    if len(value) != 40:
        raise ValueError("unsupported Git object format")
    decoded = value.decode("ascii", "strict")
    _ = bytes.fromhex(decoded)
    return decoded


def _enabled(value: bytes | None) -> bool:
    return value is not None and value.strip().lower() in {b"true", b"1", b"yes", b"on"}


def _unsupported_content_config(value: bytes | None) -> bool:
    """Reject index modes that cannot prove final unnormalized working bytes."""
    if value is None:
        return False
    effective: dict[bytes, bytes] = {}
    for record in value.split(b"\0"):
        if not record:
            continue
        key, separator, setting = record.partition(b"\n")
        if not separator or key.lower() not in {b"core.sparsecheckout", b"core.autocrlf"}:
            raise ValueError("invalid content configuration record")
        # --get-regexp includes overridden system/global/include values too;
        # Git's last value for each key is the policy used by status.
        effective[key.lower()] = setting.strip().lower()
    return any(setting not in {b"", b"false", b"0", b"no", b"off"} for setting in effective.values())


def _index_paths(entries: bytes) -> list[bytes]:
    paths: list[bytes] = []
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        fields = metadata.split(b" ")
        if (
            len(fields) != 4
            or fields[0] != b"H"
            or fields[1] not in {b"100644", b"100755", b"120000"}
            or fields[3] != b"0"
            or not _is_repository_path(path)
        ):
            raise ValueError("unsupported index entry")
        _ = bytes.fromhex(fields[2].decode("ascii", "strict"))
        if len(fields[2]) != 40:
            raise ValueError("unsupported Git object format")
        paths.append(path)
    return sorted(set(paths))


def _status_paths(status: bytes) -> list[bytes]:
    paths: list[bytes] = []
    records = iter(status.split(b"\0"))
    for record in records:
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise ValueError("invalid porcelain record")
        path = record[3:]
        if record[:2] == b"??" and path.endswith(b"/"):
            path = path[:-1]
        if not _is_repository_path(path):
            raise ValueError("invalid porcelain path")
        paths.append(path)
        if record[:1] in {b"R", b"C"} or record[1:2] in {b"R", b"C"}:
            source = next(records)
            if not _is_repository_path(source):
                raise ValueError("invalid porcelain path")
            paths.append(source)
    return sorted(set(paths))


def _is_repository_path(path: bytes) -> bool:
    return bool(path) and not path.startswith(b"/") and all(part not in {b"", b".", b".."} for part in path.split(b"/"))


def _nul_terminated(paths: list[bytes]) -> bytes:
    return b"".join(path + b"\0" for path in paths)


def _has_filter(attributes: bytes) -> bool:
    fields = attributes.split(b"\0")
    if fields[-1:] == [b""]:
        fields.pop()
    if len(fields) % 3:
        raise ValueError("invalid attribute record")
    transforms = {b"filter", b"text", b"eol", b"ident", b"crlf", b"working-tree-encoding"}
    return any(
        attribute in transforms and value not in {b"unspecified", b"unset"}
        for attribute, value in zip(fields[1::3], fields[2::3])
    )


def _tree_entries(tree: bytes) -> dict[bytes, tuple[bytes, bytes]]:
    entries: dict[bytes, tuple[bytes, bytes]] = {}
    for record in tree.split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode, kind, blob = metadata.split(b" ", 2)
        if mode not in {b"100644", b"100755", b"120000"} or kind != b"blob" or not _is_repository_path(path):
            raise ValueError("unsupported tree entry")
        if len(blob) != 40:
            raise ValueError("unsupported Git object format")
        _ = bytes.fromhex(blob.decode("ascii", "strict"))
        if path in entries:
            raise ValueError("duplicate tree entry")
        entries[path] = (mode, blob)
    return entries


def _canonical_tree(
    head_entries: dict[bytes, tuple[bytes, bytes]], overlay: list[tuple[bytes, bytes, bytes]]
) -> str:
    entries = dict(head_entries)
    for path, mode, blob in overlay:
        if mode == b"deleted":
            entries.pop(path, None)
        else:
            entries[path] = (mode, blob)
    root: dict[bytes, object] = {}
    for path, entry in entries.items():
        directory = root
        parts = path.split(b"/")
        for part in parts[:-1]:
            child = directory.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError("tree path collision")
            directory = child
        if parts[-1] in directory:
            raise ValueError("tree path collision")
        directory[parts[-1]] = entry

    def tree_object_id(directory: dict[bytes, object], *, is_root: bool = False) -> bytes | None:
        records: list[tuple[bytes, bytes, bytes, bytes]] = []
        for name, value in directory.items():
            if isinstance(value, dict):
                object_id = tree_object_id(value)
                if object_id is None:
                    continue
                mode = b"40000"
            else:
                if not isinstance(value, tuple) or len(value) != 2:
                    raise ValueError("invalid tree entry")
                mode, object_id = value
                if not isinstance(mode, bytes) or not isinstance(object_id, bytes):
                    raise ValueError("invalid tree entry")
            records.append((name + (b"/" if mode == b"40000" else b""), mode, name, object_id))
        if not records and not is_root:
            return None
        payload = b"".join(
            mode + b" " + name + b"\0" + bytes.fromhex(object_id.decode("ascii", "strict"))
            for _sort_key, mode, name, object_id in sorted(records)
        )
        return hashlib.sha1(b"tree " + str(len(payload)).encode() + b"\0" + payload).hexdigest().encode()

    final_tree = tree_object_id(root, is_root=True)
    if final_tree is None:
        raise ValueError("missing final tree")
    return final_tree.decode("ascii")


def _identity(final_tree: str, head: str, *, dirty: bool, calls: int) -> WorkingTreeFingerprint:
    digest = hashlib.sha256()
    digest.update(WORKING_TREE_CONTENT_FINGERPRINT_SCHEMA.encode())
    digest.update(b"\0")
    digest.update(final_tree.encode())
    state = WorkingTreeFingerprintState.DIRTY if dirty else WorkingTreeFingerprintState.CLEAN
    return WorkingTreeFingerprint(state, digest.hexdigest(), head, calls)


def _result(state: WorkingTreeFingerprintState, head_tree: str | None = None, calls: int = 0) -> WorkingTreeFingerprint:
    return WorkingTreeFingerprint(state, None, head_tree, calls)
