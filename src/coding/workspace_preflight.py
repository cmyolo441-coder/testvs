"""Probe, in the isolation a unit will actually use, that the work can be done.

`executor_readiness` answers one question: does the executor's binary run? On
2026-09-11 that answer was `available: true` for a worker that then spent 36
minutes unable to write a git index it had been told was complete. The clone
was `blob:none` with network fetch forbidden, so objects the merge needed were
simply absent, and case-only filename collisions on a case-insensitive
filesystem blocked the checkout. Every one of those conditions was observable
before the unit started, and none of them was observed.

`--version` exiting 0 says the binary runs. It does not say a file can be
written, an index can be updated, the commits the work names exist, or the
tracked filenames can coexist on this filesystem. Those four are what this
module observes, in the worktree path the unit was handed, with no network
call and a bounded timeout on every subprocess.

A failing check is a BLOCKER, not a retry hint. None of the four clears by
running the same unit again under the same conditions: a permission denial
stays denied, a missing object stays missing until someone fetches it, and a
case collision is a property of the tree and the filesystem. The dispatcher
therefore refuses the spawn and reports which check failed with its detail, so
the repair happens at the preparation step where it belongs.

Nothing here spawns an agent, reads a credential, or reaches a network. It
runs `git` read/write commands against the given worktree, writes only scratch
paths it removes again, and never touches a tracked file or the real index.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any, Callable
from uuid import uuid4

from .unit_execution_state import (
    UNIT_STATE_DATA_MISSING,
    UNIT_STATE_PERMISSION_BLOCKED,
)

WORKSPACE_PREFLIGHT_SCHEMA_VERSION = "workspace_preflight/v1"

CHECK_FILE_WRITE = "file_write"
CHECK_GIT_INDEX_WRITE = "git_index_write"
CHECK_OBJECTS_PRESENT = "objects_present"
CHECK_CASE_COLLISION = "case_collision"

# Order is the order they run in, and the order a reader should read them: a
# filesystem that cannot take a file explains an index that cannot be written,
# which explains everything after it.
WORKSPACE_PREFLIGHT_CHECKS: tuple[str, ...] = (
    CHECK_FILE_WRITE,
    CHECK_GIT_INDEX_WRITE,
    CHECK_OBJECTS_PRESENT,
    CHECK_CASE_COLLISION,
)

WORKSPACE_PREFLIGHT_CLAIM_BOUNDARY = (
    "A workspace preflight observes four blockers inside the isolation the unit was handed: a file "
    "can be written, the git index can be written, the commits the work names are present, and no "
    "tracked filenames collide on a case-insensitive filesystem. A passing preflight is not a claim "
    "that the work will succeed, and it is not execution, verification, review, CI, or merge "
    "evidence -- only that these four blockers are absent right now."
)

# Which unit execution state a blocking check leaves the unit in. Permission
# and case collision are both `permission_blocked`: neither is cleared by
# retrying, and a case collision is a denial the filesystem issues rather than
# data the repository lacks.
_STATE_BY_CHECK: dict[str, str] = {
    CHECK_FILE_WRITE: UNIT_STATE_PERMISSION_BLOCKED,
    CHECK_GIT_INDEX_WRITE: UNIT_STATE_PERMISSION_BLOCKED,
    CHECK_OBJECTS_PRESENT: UNIT_STATE_DATA_MISSING,
    CHECK_CASE_COLLISION: UNIT_STATE_PERMISSION_BLOCKED,
}

# Every git call is bounded. A preflight that can hang is a preflight that
# reproduces the incident it exists to prevent.
_GIT_TIMEOUT_SECONDS = 30
# Commits walked when scanning a partial clone for absent objects. The scan is
# a blocker check, not an inventory: the first missing object already answers
# the question, and an unbounded walk on a large history would cost more than
# the spawn it is guarding.
_MISSING_SCAN_COMMIT_LIMIT = 200
# Colliding groups named in the detail. Enough to recognise the shape of the
# problem, not an attempt to list a whole tree.
_MAX_REPORTED_COLLISIONS = 3
_MAX_DETAIL = 400
# Exit code for a git that could not be run at all. Same convention
# `worktree_creator._git` uses: an unanswerable question is a refusal.
_GIT_UNRUNNABLE = 127


def probe_workspace(
    worktree: Path,
    *,
    base_ref: str | None,
    target_ref: str | None,
    runner: Callable[..., object] = subprocess.run,
) -> dict[str, Any]:
    """Observe the four pre-spawn blockers in `worktree`; never raise.

    `base_ref` is the commit the unit's work will merge into and `target_ref`
    the unit's own head when one already exists; either may be None, and the
    checks that need them are reported as skipped rather than guessed.

    All four checks always run. A file write that fails explains an index write
    that fails, and an operator repairing the isolation wants both facts at
    once rather than one per dispatch attempt.
    """
    root = Path(worktree)
    env = _git_environment()
    checks = [
        _check_file_write(root),
        _check_git_index_write(root, runner, env),
        _check_objects_present(root, runner, env, base_ref, target_ref),
        _check_case_collision(root, runner, env),
    ]
    blocking = [check["name"] for check in checks if not check["ok"]]
    return {
        "schema_version": WORKSPACE_PREFLIGHT_SCHEMA_VERSION,
        "ok": not blocking,
        "worktree": str(root),
        "checks": checks,
        "blocking": blocking,
        "claim_boundary": WORKSPACE_PREFLIGHT_CLAIM_BOUNDARY,
    }


def workspace_preflight_unit_state(report: dict[str, Any]) -> str:
    """The execution state a blocked preflight leaves the unit in, or an empty string.

    The FIRST blocking check decides, because the checks run in dependency
    order: a filesystem that refuses a write is why the index write failed, and
    reporting the later symptom would send the repair to the wrong place.
    """
    blocking = report.get("blocking")
    if not isinstance(blocking, list):
        return ""
    for name in blocking:
        state = _STATE_BY_CHECK.get(str(name))
        if state:
            return state
    return ""


def workspace_preflight_reason(report: dict[str, Any]) -> str:
    """One line naming every failing check and the first detail, for the envelope."""
    failed = [
        check
        for check in report.get("checks", [])
        if isinstance(check, dict) and not check.get("ok")
    ]
    if not failed:
        return ""
    names = ", ".join(str(check.get("name", "")) for check in failed)
    return _bounded(f"workspace preflight failed in the unit's isolation ({names}): "
                    f"{failed[0].get('detail', '')}")


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": _bounded(detail)}


def _bounded(text: object) -> str:
    """One bounded line. Git stderr arrives multi-line and CRLF-terminated on Windows."""
    collapsed = " ".join(str(text).split())
    return collapsed[:_MAX_DETAIL]


def _git_environment() -> dict[str, str]:
    """The child environment for every git call here: no network, no prompts.

    `GIT_NO_LAZY_FETCH` is what keeps a partial clone from reaching its
    promisor remote mid-check (git 2.41+; older git ignores it, and
    `--missing=print` already suppresses the fetch for the one command that
    could trigger one). `GIT_TERMINAL_PROMPT=0` means a credential prompt fails
    instead of hanging past the timeout, and `GIT_OPTIONAL_LOCKS=0` keeps a
    read from taking the index lock a concurrent unit may hold.
    """
    env = dict(os.environ)
    env["GIT_NO_LAZY_FETCH"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return env


def _stream(completed: object, name: str) -> str:
    raw = getattr(completed, name, "")
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw or "")


def _run_git(
    runner: Callable[..., object],
    worktree: Path,
    argv: list[str],
    *,
    env: dict[str, str],
    stdin: str | None = None,
) -> tuple[int, str, str]:
    """Run one bounded git command in the worktree, returning (code, stdout, stderr).

    `argv` arrives complete, `"git"` and its subcommand spelled as literals at
    the call site. That is not decoration: `tests/test_handoff_safety_contract
    _enforcement.py::NoRemoteMutation` proves structurally that no argv in
    `src/` can reach a remote, and it can only do so while every subcommand is
    a constant it can read out of the syntax tree.

    A git that cannot be run at all reports `_GIT_UNRUNNABLE` rather than
    raising: every caller is deciding whether a blocker is present, and an
    unanswerable question is a blocker, not a crash. `stdin`, when given, is
    passed as the runner's `input`; it is omitted otherwise, so a runner that
    does not take `input` still serves every other call, and the one call that
    needs it reports `_GIT_UNRUNNABLE` instead of raising.
    """
    try:
        completed = runner(
            argv,
            cwd=str(worktree),
            text=True,
            capture_output=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=env,
            **({} if stdin is None else {"input": stdin}),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _GIT_UNRUNNABLE, "", f"{type(exc).__name__}: {exc}"
    except TypeError as exc:
        # Only a call that hands git stdin can meet a runner without `input`.
        if stdin is None:
            raise
        return _GIT_UNRUNNABLE, "", f"the runner could not hand git its stdin: {exc}"
    code = getattr(completed, "returncode", None)
    # A runner that reports no usable exit code answered nothing; treating that
    # as success would let an unchecked blocker through.
    return (code if type(code) is int else 1), _stream(completed, "stdout"), _stream(completed, "stderr")


def _scratch_name(suffix: str) -> str:
    return f".omh-workspace-preflight-{uuid4().hex}{suffix}"


def _check_file_write(worktree: Path) -> dict[str, Any]:
    """Create, read back and remove one scratch file in the isolation.

    The scratch path is removed on every exit. A scratch file that survives
    would show up in the unit's own `git status` and change the work it
    reports, so a removal that fails is itself a failure of this check rather
    than a detail worth swallowing.
    """
    scratch = worktree / _scratch_name(".tmp")
    payload = "omh workspace preflight\n"
    try:
        scratch.write_text(payload, encoding="utf-8")
        read_back = scratch.read_text(encoding="utf-8")
    except OSError as exc:
        _remove(scratch)
        return _check(CHECK_FILE_WRITE, False, f"could not write a scratch file in {worktree}: {exc}")
    if read_back != payload:
        _remove(scratch)
        return _check(
            CHECK_FILE_WRITE,
            False,
            f"scratch file in {worktree} read back different bytes than were written",
        )
    try:
        scratch.unlink()
    except OSError as exc:
        return _check(
            CHECK_FILE_WRITE,
            False,
            f"wrote {scratch} but could not remove it ({exc}); the unit would inherit a stray file",
        )
    return _check(CHECK_FILE_WRITE, True, f"wrote, read and removed a scratch file in {worktree}")


def _remove(path: Path) -> None:
    """Best-effort cleanup of one scratch path we created.

    Deliberately silent: this runs only on a path that already failed, where
    the failure is what gets reported and a second error about the cleanup
    would bury it. The check that succeeded removes its own scratch path
    explicitly and reports a failure to do so.
    """
    try:
        path.unlink()
    except OSError:
        pass


def _check_git_index_write(
    worktree: Path, runner: Callable[..., object], env: dict[str, str]
) -> dict[str, Any]:
    """Prove the object store and the index can be written, without dirtying either.

    The write goes to a TEMPORARY index named by `GIT_INDEX_FILE`, so the real
    index the unit will use is never touched. `hash-object -w --stdin` is what
    observes that the object store itself is writable; `read-tree` plus
    `update-index --cacheinfo` is what observes that the index machinery works.

    The git directory is whatever the worktree's `.git` file names, and on a
    reused worktree that file is the unit's (#2040). So this process writes
    nothing under it: the blob reaches git on stdin, and the temporary index
    is created by git itself, through `runner`, which on a reused worktree is
    the unit's own fence. What this process does there is unlink the names
    that index write uses, which come from a fresh uuid: they exist only if
    the git call just above created them, and `unlink` does not follow a link.
    """
    code, stdout, stderr = _run_git(runner, worktree, ["git", "rev-parse", "--absolute-git-dir"], env=env)
    if code != 0:
        return _check(
            CHECK_GIT_INDEX_WRITE,
            False,
            f"could not resolve the git directory of {worktree}: {stderr or stdout}",
        )
    git_dir = Path(stdout.strip())
    if not git_dir.is_dir():
        return _check(
            CHECK_GIT_INDEX_WRITE,
            False,
            f"git reported {git_dir} as its directory, but it is not a directory",
        )
    index_path = git_dir / _scratch_name(".index")
    try:
        code, stdout, stderr = _run_git(
            runner, worktree, ["git", "hash-object", "-w", "--stdin"], env=env,
            stdin="omh workspace preflight\n",
        )
        if code != 0:
            return _check(
                CHECK_GIT_INDEX_WRITE,
                False,
                f"the object store under {git_dir} refused a write: {stderr or stdout}",
            )
        oid = stdout.strip()
        index_env = dict(env)
        index_env["GIT_INDEX_FILE"] = str(index_path)
        code, stdout, stderr = _run_git(runner, worktree, ["git", "read-tree", "HEAD"], env=index_env)
        if code != 0:
            return _check(
                CHECK_GIT_INDEX_WRITE,
                False,
                f"could not read HEAD into a temporary index at {index_path}: {stderr or stdout}",
            )
        code, stdout, stderr = _run_git(
            runner,
            worktree,
            ["git", "update-index", "--add", "--cacheinfo", f"100644,{oid},omh-workspace-preflight"],
            env=index_env,
        )
        if code != 0:
            return _check(
                CHECK_GIT_INDEX_WRITE,
                False,
                f"could not write an entry into a temporary index at {index_path}: {stderr or stdout}",
            )
    finally:
        _remove(index_path)
        _remove(index_path.with_name(index_path.name + ".lock"))
    return _check(
        CHECK_GIT_INDEX_WRITE,
        True,
        f"wrote a blob and an index entry through a temporary index under {git_dir}",
    )


def _check_objects_present(
    worktree: Path,
    runner: Callable[..., object],
    env: dict[str, str],
    base_ref: str | None,
    target_ref: str | None,
) -> dict[str, Any]:
    """Observe that the commits the work names exist here, and that nothing is promised-away.

    A partial clone is not a failure by itself -- it is a normal, useful thing
    to have. It becomes the incident's condition when objects the work needs
    are absent AND the fetch that would supply them cannot happen. So the
    promisor configuration is reported as a detail, and only an actually
    missing object fails the check.
    """
    notes: list[str] = []
    refs = [("HEAD", "HEAD")]
    if base_ref:
        refs.append(("base_ref", base_ref))
    if target_ref:
        refs.append(("target_ref", target_ref))
    for label, ref in refs:
        code, stdout, stderr = _run_git(runner, worktree, ["git", "cat-file", "-e", f"{ref}^{{commit}}"], env=env)
        if code != 0:
            return _check(
                CHECK_OBJECTS_PRESENT,
                False,
                f"{label} {ref} is not a commit present in {worktree}: {stderr or stdout}",
            )
    notes.append(f"present: {', '.join(f'{label}={ref}' for label, ref in refs)}")
    if base_ref and target_ref:
        code, stdout, stderr = _run_git(
            runner, worktree, ["git", "merge-base", base_ref, target_ref], env=env
        )
        if code != 0:
            return _check(
                CHECK_OBJECTS_PRESENT,
                False,
                f"no merge base between {base_ref} and {target_ref} in {worktree}: {stderr or stdout}",
            )
        notes.append(f"merge-base {stdout.strip()[:40]}")
    else:
        notes.append("merge-base not checked (only one side named)")
    partial = _partial_clone_note(worktree, runner, env)
    if not partial:
        notes.append("not a partial clone")
        return _check(CHECK_OBJECTS_PRESENT, True, "; ".join(notes))
    notes.append(partial)
    scan = (
        [f"{base_ref}..{target_ref}"]
        if base_ref and target_ref and base_ref != target_ref
        else ["HEAD"]
    )
    code, stdout, stderr = _run_git(
        runner,
        worktree,
        ["git", "rev-list", "--objects", "--missing=print", "-n", str(_MISSING_SCAN_COMMIT_LIMIT), *scan],
        env=env,
    )
    if code != 0:
        return _check(
            CHECK_OBJECTS_PRESENT,
            False,
            f"could not scan {' '.join(scan)} for absent objects in this partial clone: {stderr or stdout}",
        )
    missing = [line for line in stdout.splitlines() if line.startswith("?")]
    if missing:
        return _check(
            CHECK_OBJECTS_PRESENT,
            False,
            f"{len(missing)} object(s) the work needs are absent from this partial clone over "
            f"{' '.join(scan)} and fetching them is not permitted here; first: {missing[0][:60]}",
        )
    notes.append(f"no absent objects over {' '.join(scan)} within {_MISSING_SCAN_COMMIT_LIMIT} commits")
    return _check(CHECK_OBJECTS_PRESENT, True, "; ".join(notes))


def _partial_clone_note(
    worktree: Path, runner: Callable[..., object], env: dict[str, str]
) -> str:
    """A description of this clone's partial-clone configuration, or an empty string.

    Read from config rather than inferred: `extensions.partialClone` names the
    promisor remote and `remote.<name>.promisor` marks it, and either one alone
    is enough to mean objects may be promised rather than present.
    """
    found: list[str] = []
    for key in ("extensions.partialClone", "remote.origin.promisor", "remote.origin.partialclonefilter"):
        code, stdout, _stderr = _run_git(runner, worktree, ["git", "config", "--get", key], env=env)
        value = stdout.strip()
        if code == 0 and value and value.casefold() != "false":
            found.append(f"{key}={value[:60]}")
    if not found:
        return ""
    code, stdout, _stderr = _run_git(
        runner, worktree, ["git", "config", "--get", "core.repositoryFormatVersion"], env=env
    )
    version = stdout.strip() if code == 0 else ""
    return "partial clone (" + ", ".join(found) + (f", repositoryFormatVersion={version}" if version else "") + ")"


def _check_case_collision(
    worktree: Path, runner: Callable[..., object], env: dict[str, str]
) -> dict[str, Any]:
    """On a case-insensitive filesystem, fail when two tracked paths differ only in case.

    Case sensitivity is observed, never inferred from the platform: a macOS
    machine can hold a case-sensitive volume and a Linux machine can mount a
    case-insensitive one, and the wrong guess here is the difference between
    a checkout that works and one that silently loses a file.
    """
    sensitivity = _case_sensitivity(worktree)
    if sensitivity is None:
        return _check(
            CHECK_CASE_COLLISION,
            False,
            f"could not determine whether the filesystem under {worktree} is case-insensitive",
        )
    if sensitivity == "sensitive":
        return _check(
            CHECK_CASE_COLLISION,
            True,
            f"the filesystem under {worktree} is case-sensitive, so tracked filenames that differ "
            "only in case cannot collide here",
        )
    code, stdout, stderr = _run_git(runner, worktree, ["git", "ls-tree", "-r", "--name-only", "HEAD"], env=env)
    if code != 0:
        return _check(
            CHECK_CASE_COLLISION,
            False,
            f"could not list the tracked paths of HEAD in {worktree}: {stderr or stdout}",
        )
    groups: dict[str, list[str]] = {}
    for line in stdout.splitlines():
        path = line.strip()
        if path:
            groups.setdefault(path.casefold(), []).append(path)
    collisions = [sorted(set(paths)) for paths in groups.values() if len(set(paths)) > 1]
    if not collisions:
        return _check(
            CHECK_CASE_COLLISION,
            True,
            f"the filesystem under {worktree} is case-insensitive and no tracked path in HEAD "
            "collides with another under casefolding",
        )
    collisions.sort()
    named = "; ".join(" vs ".join(group) for group in collisions[:_MAX_REPORTED_COLLISIONS])
    return _check(
        CHECK_CASE_COLLISION,
        False,
        f"{len(collisions)} tracked path group(s) in HEAD differ only in case on a case-insensitive "
        f"filesystem, which blocks checkout and merge here: {named}",
    )


def _case_sensitivity(worktree: Path) -> str | None:
    """"sensitive", "insensitive", or None when the probe itself could not run."""
    probe = worktree / _scratch_name("-case.tmp")
    twin = probe.with_name(probe.name.upper())
    try:
        probe.write_text("case\n", encoding="utf-8")
    except OSError:
        _remove(probe)
        return None
    try:
        # `exists()` on the uppercase twin is the whole test: only a
        # case-insensitive filesystem resolves it to the file just written.
        # The names differ in case by construction, so a false positive would
        # require the twin to exist already, which a fresh uuid rules out.
        insensitive = twin.exists()
    except OSError:
        # An unanswerable probe is reported as unanswerable. Defaulting to
        # "sensitive" would skip the collision scan on exactly the filesystem
        # least able to answer questions about itself.
        return None
    finally:
        _remove(probe)
    return "insensitive" if insensitive else "sensitive"
