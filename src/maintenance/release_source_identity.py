"""Read-only source identity and deterministic release input manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping

from ..system.hashutil import sha256_file
from .release_notes import NotesMetadata

RELEASE_EVIDENCE_BUNDLE_SCHEMA_V2 = "omh_release_evidence_bundle/v2"
RELEASE_SOURCE_IDENTITY_SCHEMA = "omh_release_source_identity/v1"
RELEASE_INPUT_MANIFEST_SCHEMA = "omh_release_input_manifest/v1"

SOURCE_ORIGINS = ("git_checkout", "source_archive", "installed_package", "unknown")

_GIT_TIMEOUT_SECONDS = 15
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DEFAULT_GIT_RUNNER = subprocess.run


def probe_source_identity(
    repo_root: str | Path | None,
    *,
    runner: Callable[..., Any] = _DEFAULT_GIT_RUNNER,
    archive_digest: str = "",
    artifact_digest: str = "",
) -> dict[str, object]:
    """Probe the immutable identity of the source under `repo_root`.

    Read-only local git plumbing (``rev-parse`` and ``status --porcelain``)
    runs with repository-config isolation: no network call, nothing written,
    no ref moved. Full hashes are recorded
    because a binding contract cannot use truncated ones. Any probe failure --
    no git binary, no repository, a timeout, a malformed hash -- fails soft
    to ``identity_status: unavailable``; it never raises and never passes
    implicitly. Without a git checkout, identity exists only when an explicit
    digest is supplied: ``archive_digest`` for a source archive,
    ``artifact_digest`` for an installed package.
    """
    root = Path(repo_root).expanduser() if repo_root is not None else None
    commit_sha: str | None = None
    tree_sha: str | None = None
    dirty: bool | None = None
    dirty_file_count = 0
    if root is not None:
        commit_raw = _git_output(runner, root, ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"])
        tree_raw = _git_output(
            runner, root, ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD^{tree}"]
        )
        status_raw = _git_output(
            runner,
            root,
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "--no-optional-locks",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
        )
        if commit_raw is not None and _FULL_SHA_RE.match(commit_raw.strip()):
            commit_sha = commit_raw.strip()
        if tree_raw is not None and _FULL_SHA_RE.match(tree_raw.strip()):
            tree_sha = tree_raw.strip()
        if status_raw is not None:
            changed = [line for line in status_raw.splitlines() if line.strip()]
            dirty = bool(changed)
            dirty_file_count = len(changed)
    git_available = commit_sha is not None and tree_sha is not None and dirty is not None
    if git_available:
        origin = "git_checkout"
        identity_status = "available"
        recorded_archive_digest = None
        recorded_artifact_digest = None
    elif _DIGEST_RE.match(archive_digest or ""):
        origin = "source_archive"
        identity_status = "available"
        commit_sha = tree_sha = dirty = None
        dirty_file_count = 0
        recorded_archive_digest = archive_digest
        recorded_artifact_digest = None
    elif _DIGEST_RE.match(artifact_digest or ""):
        origin = "installed_package"
        identity_status = "available"
        commit_sha = tree_sha = dirty = None
        dirty_file_count = 0
        recorded_archive_digest = None
        recorded_artifact_digest = artifact_digest
    else:
        origin = "unknown"
        identity_status = "unavailable"
        commit_sha = tree_sha = dirty = None
        dirty_file_count = 0
        recorded_archive_digest = None
        recorded_artifact_digest = None
    return {
        "schema_version": RELEASE_SOURCE_IDENTITY_SCHEMA,
        "origin": origin,
        "commit_sha": commit_sha,
        "tree_sha": tree_sha,
        "dirty": dirty,
        "dirty_file_count": dirty_file_count,
        "identity_status": identity_status,
        "archive_digest": recorded_archive_digest,
        "artifact_digest": recorded_artifact_digest,
    }


def build_input_manifest(
    *,
    source_identity: Mapping[str, object],
    paths: Any = None,
    artifact: str | Path | None = None,
    release_notes: NotesMetadata | None = None,
) -> dict[str, object]:
    """Build the deterministic manifest of the inputs the evidence depends on.

    The git tree hash covers every tracked source file and tracked generated
    projection in one platform-independent digest. The local artifact store
    (the maintainer-machine files that feed `use_case_readiness`) is hashed
    file by file with store-relative POSIX paths. The release artifact is
    hashed when one is supplied. Entries carry relative paths and ``sha256:``
    digests only -- never absolute paths, URLs, environment data, or file
    contents -- and the manifest digest is canonical-JSON SHA-256, so two
    builds over the same inputs are byte-identical.
    """
    store_entries: list[dict[str, str]] = []
    store_status = "not_checked"
    store_digest = None
    if paths is not None:
        store_root = Path(paths.use_cases_dir)
        if store_root.is_dir():
            files = sorted(path for path in store_root.rglob("*") if path.is_file())
            store_entries = [
                {
                    "path": path.relative_to(store_root).as_posix(),
                    "sha256": f"sha256:{sha256_file(path)}",
                }
                for path in files
            ]
            store_status = "recorded"
            store_digest = _canonical_digest({"entries": store_entries})
        else:
            store_status = "not_written"
    artifact_entry: dict[str, object] = {"binding": "not_recorded", "name": None, "sha256": None}
    if artifact is not None:
        artifact_path = Path(artifact)
        if not artifact_path.is_file():
            raise ValueError(f"release evidence artifact is not a readable file: {artifact_path.name}")
        artifact_entry = {
            "binding": "recorded",
            "name": artifact_path.name,
            "sha256": f"sha256:{sha256_file(artifact_path)}",
        }
    manifest: dict[str, object] = {
        "schema_version": RELEASE_INPUT_MANIFEST_SCHEMA,
        "source_tree_sha": source_identity.get("tree_sha"),
        "local_artifact_store": {
            "status": store_status,
            "digest": store_digest,
            "entries": store_entries,
        },
        "artifact": artifact_entry,
    }
    if release_notes is not None:
        manifest["release_notes"] = release_notes
    manifest["digest"] = _canonical_digest(manifest)
    return manifest


def _git_output(runner: Callable[..., Any], root: Path, command: list[str]) -> str | None:
    try:
        completed = runner(
            command,
            cwd=str(root),
            text=True,
            capture_output=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(completed, "returncode", 1) != 0:
        return None
    return str(getattr(completed, "stdout", "") or "")


def _canonical_digest(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
