"""Bounded, read-only inspection of a local Hermes installation.

One seam, because several callers need the same facts and a second parser
would drift from the first. ``workflows.plugin_hook_contract`` asks whether the
pinned hook mapping was established for the Hermes a plugin is about to be
enabled on; doctor's OMH TUI check asks whether the launcher hook the OMH TUI
needs is there, and which commit the checkout is at, to print next to the one
the OMH TUI's gateway contract was declared against.

What "bounded" means here, exactly: paths derived from a directory the caller
named (the version module; the checkout's ``.git/HEAD``, the loose ref it
names under ``refs/``, and ``packed-refs``), each opened only when it is a
regular file under the size cap, decoded with replacement, and matched by one
regex. Nothing is imported, no binary is executed (git included), no
subprocess is spawned and no socket is opened. That is what lets a caller point
this at an installation it has not audited, and it is also the limit: a
version string or commit read off disk can be stale or edited, so it is
evidence of what the installation declares, never of what a running host would
report.
"""
from __future__ import annotations

from pathlib import Path
import re

HERMES_INSTALL_DIRNAME = "hermes-agent"

# Reading cap: `tools/delegate_tool.py`, the largest file read here, is ~46KB today.
MAX_BOUNDED_READ_BYTES = 512_000

# The module Hermes keeps `__version__` in, named in `HERMES_COMPAT_MATRIX`'s
# covered host contracts as `hermes_cli.__version__`. Public so a diagnostic can
# tell an operator which file was looked for without a second literal.
VERSION_MODULE_RELATIVE = Path("hermes_cli") / "__init__.py"

_VERSION_ASSIGNMENT = re.compile(r"__version__\s*=\s*[\"']([^\"']+)[\"']")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_HEAD_REF = re.compile(r"ref: (refs/[A-Za-z0-9._/-]+)")


def read_bounded_text(path: Path) -> str | None:
    """The file's text, or None when it is absent, oversized or unreadable."""
    try:
        if not path.is_file() or path.stat().st_size > MAX_BOUNDED_READ_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def hermes_install_dir(hermes_home: Path) -> Path:
    """Where Hermes installs itself under a Hermes home."""
    return hermes_home / HERMES_INSTALL_DIRNAME


def installed_hermes_version(install_dir: Path) -> str:
    """The version the installation declares, or "" when none was established.

    Empty covers an absent directory, an absent or oversized version module, and
    a module without the assignment. They are one answer because none of them
    yields a version, and a caller that needs to tell "you named nothing" from
    "what you named yielded nothing" has to keep that distinction itself -- this
    function is only ever asked about a directory a caller already named.
    """
    text = read_bounded_text(install_dir / VERSION_MODULE_RELATIVE)
    if text is None:
        return ""
    match = _VERSION_ASSIGNMENT.search(text)
    return match.group(1) if match else ""


def installed_hermes_commit(install_dir: Path) -> str:
    """The commit the installation's git checkout is at, or "" when that cannot be read off disk.

    A detached ``HEAD`` names it; a ``HEAD`` on a branch names a ref, read as
    a loose file under ``.git/refs/`` or else as a ``packed-refs`` line.
    Empty covers no ``.git`` directory (a ``.git`` file pointing elsewhere is
    not followed), an unreadable or oversized file, a ref name with a ``..``
    segment, and a ref found in neither place.
    """
    git_dir = install_dir / ".git"
    head = (read_bounded_text(git_dir / "HEAD") or "").strip()
    if _COMMIT.fullmatch(head):
        return head
    ref = _HEAD_REF.fullmatch(head)
    if ref is None or ".." in ref.group(1).split("/"):
        return ""
    name = ref.group(1)
    loose = (read_bounded_text(git_dir / name) or "").strip()
    if _COMMIT.fullmatch(loose):
        return loose
    for line in (read_bounded_text(git_dir / "packed-refs") or "").splitlines():
        commit, _, packed_name = line.partition(" ")
        if packed_name == name and _COMMIT.fullmatch(commit):
            return commit
    return ""
