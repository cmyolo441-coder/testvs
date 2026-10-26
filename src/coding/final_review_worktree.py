"""Read-only disposable worktrees for immutable final-review lanes."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import stat
import subprocess
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Iterator


_MAX_REVIEW_PATHS = 20_000


class FinalReviewWorktreeError(RuntimeError):
    """A review snapshot could not be isolated or cleaned safely."""


@contextmanager
def isolated_final_review_worktree(
    source: Path,
    expected_tree: str,
    git_lock: Lock,
) -> Iterator[Path]:
    """Yield one fixed detached checkout with filesystem writes removed."""
    if not final_review_worktree_matches(source, expected_tree):
        raise FinalReviewWorktreeError(
            "integrated worktree was dirty or moved before final-review isolation"
        )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    with TemporaryDirectory(prefix="omh-final-review-") as raw:
        checkout = Path(raw) / "checkout"
        with git_lock:
            added = subprocess.run(
                [
                    "git",
                    "worktree",
                    "add",
                    "--detach",
                    "--quiet",
                    str(checkout),
                    source_commit,
                ],
                cwd=source,
                capture_output=True,
                timeout=120,
            )
        if added.returncode != 0:
            raise FinalReviewWorktreeError(
                "final-review worktree creation failed"
            )
        cleanup_error = False
        try:
            if not final_review_worktree_matches(checkout, expected_tree):
                raise FinalReviewWorktreeError(
                    "final-review worktree revision did not match integration"
                )
            _remove_write_permissions(checkout)
            yield checkout
        finally:
            try:
                _restore_cleanup_permissions(checkout)
            except OSError:
                cleanup_error = True
            with git_lock:
                removed = subprocess.run(
                    [
                        "git",
                        "worktree",
                        "remove",
                        "--force",
                        str(checkout),
                    ],
                    cwd=source,
                    capture_output=True,
                    timeout=120,
                )
            if cleanup_error or removed.returncode != 0 or checkout.exists():
                raise FinalReviewWorktreeError(
                    "final-review worktree cleanup failed"
                )


def final_review_worktree_matches(
    checkout: Path,
    expected_tree: str,
) -> bool:
    """Return whether the isolated checkout stayed clean at the fixed tree."""
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=checkout,
        capture_output=True,
        text=True,
        timeout=30,
    )
    status = subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=checkout,
        capture_output=True,
        timeout=30,
    )
    return (
        tree.returncode == 0
        and tree.stdout.strip() == expected_tree
        and status.returncode == 0
        and not status.stdout
    )


def _review_paths(root: Path) -> tuple[Path, ...]:
    paths = (root, *root.rglob("*"))
    if len(paths) > _MAX_REVIEW_PATHS:
        raise FinalReviewWorktreeError(
            "final-review worktree exceeded its path bound"
        )
    return paths


def _remove_write_permissions(root: Path) -> None:
    paths = _review_paths(root)
    for path in reversed(paths):
        if path.is_symlink():
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        os.chmod(path, mode & ~0o222)


def _restore_cleanup_permissions(root: Path) -> None:
    if not root.exists():
        return
    for path in _review_paths(root):
        if path.is_symlink():
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        additions = stat.S_IWUSR
        if path.is_dir():
            additions |= stat.S_IRUSR | stat.S_IXUSR
        os.chmod(path, mode | additions)
