"""Symlink read-only-ish build artifacts into fresh fanout unit worktrees.

Each `omh coding fanout dispatch` unit gets its own `git worktree add`, which
starts cold: dependency directories and build caches the parent checkout
already paid for (`node_modules`, `.venv`, `target`, ...) have to be rebuilt
from scratch inside every unit worktree. This module absorbs the precedent
documented in oh-my-pi's `.omp/commands/fix-issues.md:64-81`: symlink those
directories in from the parent checkout instead of rebuilding them, so a unit
starts warm.

The MUST NOT is the interesting half of that precedent, and it drives every
guard here: never symlink a directory that holds tracked sources, because a
folder-level link would shadow real files a unit is supposed to see and edit.
Four independent guards enforce that, checked in this order for each
allowlist entry, and any one of them failing falls back silently-but-recorded
to the cold behavior that already exists:

1. The name must be on `SHAREABLE_ARTIFACT_ALLOWLIST` -- a small, named set,
   not an open pattern.
2. The parent checkout must itself report the path `git check-ignore`d.
3. The platform must support symlinks (`os.name != "nt"`). This check is
   deliberately LAST, immediately before the `os.symlink` call, rather than a
   single up-front skip for every entry: an entry that would have refused
   anyway (missing source, not gitignored, or a tracked path already in the
   worktree) reports THAT reason on every platform, including Windows, so the
   guard order stays exercised and testable without a real symlink syscall.
   Only an entry that has cleared every other guard reports
   `platform_unsupported`.
4. AFTER the symlink is created, the unit worktree must ALSO report the same
   path ignored. This second gitignore check exists because a directory-only
   gitignore pattern (`node_modules/`, the common shape) matches a real
   directory but NOT a symlink of the same name -- git's dir-only match uses
   the on-disk type, and a symlink is never a directory to it. Skipping this
   re-check would let a shared artifact masquerade as a unit-authored change
   and corrupt `fanout_retry`'s worktree-side-effect detection for a unit
   that never touched it.

Nothing here raises. A unit worktree that cannot be warmed still gets created
cold, and every allowlist entry's outcome -- linked, or skipped with a named
reason -- rides into the caller's returned record so a run stays explainable.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

SHARED_ARTIFACT_SCHEMA_VERSION = "fanout_shared_artifacts/v1"
SHARED_ARTIFACT_CLAIM_BOUNDARY = (
    "A shared_artifacts record states which parent-checkout directories a fanout unit worktree "
    "started with a symlink instead of a cold rebuild, and why any allowlisted entry was skipped. "
    "It is not a claim that the linked content is correct, complete, fresh, or safe to trust blindly."
)

# Opt-out only: `OMH_FANOUT_SHARE_ARTIFACTS=0` disables linking entirely and a
# unit worktree starts exactly as it did before this feature. Same shape as
# the existing `OMH_MENUBAR` / `OMH_PROGRESS` toggles -- default on, explicit
# `0` to turn off.
OPT_OUT_ENV_VAR = "OMH_FANOUT_SHARE_ARTIFACTS"

# Every name this feature MAY symlink from the parent checkout into a fanout
# unit worktree, with the rationale for including it. This list is not a
# guarantee any entry gets linked in a given repo: `_link_one` still requires
# the parent checkout to report the path gitignored before it ever touches a
# unit worktree, so a repo that tracks a directory with one of these names
# never has it linked.
SHAREABLE_ARTIFACT_ALLOWLIST: tuple[dict[str, str], ...] = (
    {
        "name": "node_modules",
        "rationale": (
            "JS/TS dependency install, reproducible from lockfiles; one of the two directories "
            "the oh-my-pi fix-issues.md precedent names explicitly."
        ),
    },
    {
        "name": "target",
        "rationale": (
            "Rust/Cargo build cache; the other directory the oh-my-pi precedent names, expensive "
            "to rebuild per worktree."
        ),
    },
    {
        "name": ".venv",
        "rationale": (
            "Python virtualenv created by uv/venv, reproducible from lockfiles; this repo's own "
            "tooling convention (see CLAUDE.md Build & Test)."
        ),
    },
    {
        "name": "build",
        "rationale": (
            "Generic build output directory used by many toolchains; only ever shared when the "
            "parent checkout itself gitignores it."
        ),
    },
)


def shared_artifacts_opted_out(env: Mapping[str, str] | None = None) -> bool:
    """Whether `OMH_FANOUT_SHARE_ARTIFACTS=0` disabled this feature for this dispatch."""
    source = env if env is not None else os.environ
    return str(source.get(OPT_OUT_ENV_VAR, "1")) == "0"


def plan_and_link_shared_artifacts(
    *,
    repo_root: Path,
    worktree_path: Path,
    runner: Callable[..., Any] = subprocess.run,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Symlink allowlisted, gitignored artifact directories into a fresh unit worktree.

    Called once, right after `git worktree add` creates `worktree_path` and
    before the unit's process spawns. Every allowlist entry is independent and
    best-effort: one entry failing never blocks another, and never blocks the
    dispatch -- a unit whose warm start did not work out still runs cold.
    """
    if shared_artifacts_opted_out(env):
        return {
            "schema_version": SHARED_ARTIFACT_SCHEMA_VERSION,
            "claim_boundary": SHARED_ARTIFACT_CLAIM_BOUNDARY,
            "opted_out": True,
            "entries": [],
            "linked": [],
        }
    entries = []
    linked: list[str] = []
    for allowed in SHAREABLE_ARTIFACT_ALLOWLIST:
        entry = _link_one(
            runner,
            repo_root=repo_root,
            worktree_path=worktree_path,
            name=allowed["name"],
            rationale=allowed["rationale"],
        )
        entries.append(entry)
        if entry["action"] == "linked":
            linked.append(allowed["name"])
    return {
        "schema_version": SHARED_ARTIFACT_SCHEMA_VERSION,
        "claim_boundary": SHARED_ARTIFACT_CLAIM_BOUNDARY,
        "opted_out": False,
        "entries": entries,
        "linked": linked,
    }


def _link_one(
    runner: Callable[..., Any],
    *,
    repo_root: Path,
    worktree_path: Path,
    name: str,
    rationale: str,
) -> dict[str, str]:
    entry = {"name": name, "rationale": rationale}
    source = repo_root / name
    target = worktree_path / name
    if target.exists() or target.is_symlink():
        # A fresh `git worktree add` checkout contains only tracked paths, so
        # a non-empty path here means a TRACKED file or directory of this name
        # already lives in the checkout -- exactly what the dossier's MUST NOT
        # warns against displacing.
        return {**entry, "action": "skipped", "reason": "path_exists_in_worktree"}
    if not source.is_dir():
        return {**entry, "action": "skipped", "reason": "source_missing"}
    if not _path_is_gitignored(runner, repo_root, name):
        return {**entry, "action": "skipped", "reason": "source_not_gitignored"}
    if not _symlinks_supported():
        # Windows CI runs this path. The safest available mechanism here is to
        # skip and record why, rather than reach for a junction/copy strategy
        # that would need its own correctness rails this feature has not
        # earned yet. Checked last, immediately before the syscall, so every
        # guard above it stays exercised (and its own reason reported) on
        # every platform.
        return {**entry, "action": "skipped", "reason": "platform_unsupported"}
    try:
        os.symlink(source, target, target_is_directory=True)
    except OSError as exc:
        return {**entry, "action": "skipped", "reason": f"symlink_failed:{exc}"}
    if not _path_is_gitignored(runner, worktree_path, name):
        # Re-checked from INSIDE the worktree, after the symlink now exists.
        # A dir-only gitignore pattern does not match a symlink of the same
        # name, so this can fail even though the source check above passed.
        # Undo rather than leave a link that would show up as an untracked
        # change and corrupt replay-safety detection.
        try:
            target.unlink()
        except OSError:
            pass
        return {**entry, "action": "skipped", "reason": "symlink_not_recognized_ignored"}
    return {**entry, "action": "linked", "reason": ""}


def _path_is_gitignored(runner: Callable[..., Any], cwd: Path, relative_name: str) -> bool:
    """Whether `relative_name` is gitignored, queried from `cwd` via `git check-ignore`.

    Read-only, and never raises: a runner that cannot answer is treated as
    "not ignored," which is the fail-closed direction -- it skips linking
    rather than risk sharing tracked content.
    """
    try:
        completed = runner(
            ["git", "check-ignore", "-q", "--", relative_name],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return False
    return _exit_code(getattr(completed, "returncode", 1)) == 0


def _exit_code(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _symlinks_supported() -> bool:
    """Whether this platform can create a directory symlink at all.

    A plain function, not an inlined condition, so a test can monkeypatch it
    directly to exercise the `platform_unsupported` branch on any host
    without needing an actual Windows runner.
    """
    return os.name != "nt" and hasattr(os, "symlink")
