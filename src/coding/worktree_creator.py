"""Worktree observation ledger helpers.

OMH does not create Git worktrees for chat-prepared handoffs. Upstream Hermes
Agent manages worktrees natively (Kanban worktree-per-task since v0.15.0,
Desktop Projects since v0.18.0), so a chat-side creation path is redundant and
can collide with the worktree Hermes is already managing for the same task.
This module retains the observation-side helpers (reading the local worktree
ledger, recording observed worktree evidence) plus one scoped exception:
`ensure_fanout_unit_worktree`, used only by the explicit opt-in
`omh coding fanout dispatch` bridge, which needs one isolated worktree per
fanout unit before spawning a local agent CLI. Worktrees are never
auto-deleted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import threading
from typing import TYPE_CHECKING, Any, Callable, TypeGuard

from ..system.local_store import ensure_dir, ensure_file, file_lock, read_jsonl_objects, utc_now
from ..system.paths import OmhPaths, expand_path
from .fanout_failure_diagnostics import (
    ExitCodeSource, FailureDiagnostic, FailureReason, StreamDiagnostic,
    SCHEMA_VERSION, read_failure_diagnostic, is_string_map,
)
from .fanout_output import FanoutOutput
from .fanout_capacity import read_capacity_fields
from .fanout_executor_sessions import observe_session_workspace

if TYPE_CHECKING:
    from .fanout_confinement import FanoutFilesystemConfinement

WORKTREE_OBSERVATION_SCHEMA_VERSION = "omh_worktree_observation/v1"
WORKTREE_CLEANUP_EVENT = "worktree_cleanup"

_WORKTREE_ADD_LOCK = threading.Lock()

WORKTREE_CLAIM_BOUNDARY = (
    "An observed Git worktree is workspace-isolation evidence only. "
    "It is not executor dispatch, implementation, verification, review, CI, merge-readiness, or merge evidence."
)


def _secret_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, (tuple, list))


def _diagnostic_context(
    context: Mapping[str, object], *, unit_id: str, run_ref: str,
    base_sha: str, worktree_path: Path,
) -> tuple[FailureDiagnostic, tuple[str, ...]]:
    """Context is identity metadata, never permission to create or reuse a path."""
    identity = dict(context)
    secrets = identity.pop("known_secrets", ())
    if set(identity) != {"fanout_id", "unit_id", "run_ref", "owner", "attempt_id",
                         "worktree_ref", "base_sha", "observed_revision"}:
        raise ValueError("failure_diagnostic_context_invalid")
    fanout_id, attempt_id = identity.get("fanout_id"), identity.get("attempt_id")
    if (not isinstance(fanout_id, str) or not isinstance(attempt_id, str)
            or not attempt_id or not _secret_sequence(secrets) or len(secrets) > 128
            or identity.get("run_ref") != (run_ref or None)
            or identity.get("worktree_ref") != str(worktree_path)
            or identity.get("base_sha") != base_sha
            or identity.get("observed_revision") is not None):
        raise ValueError("failure_diagnostic_context_invalid")
    value = read_failure_diagnostic(
        {**identity, "schema_version": SCHEMA_VERSION, "phase": "worktree",
         "reason": "denial", "returncode": None, "exit_code_source": "not_observed", "streams": []},
        fanout_id=fanout_id, unit_id=unit_id, attempt_id=attempt_id,
    )
    if value is None:
        raise ValueError("failure_diagnostic_context_invalid")
    known_secrets: list[str] = []
    for secret in secrets:
        if not isinstance(secret, str):
            raise ValueError("failure_diagnostic_context_invalid")
        known_secrets.append(secret)
    # Check encoding before any subprocess or persistence, with bounded prefixes
    # matching the shared capture's conservative known-secret screening.
    try:
        _ = FanoutOutput(known_secrets=tuple(known_secrets))
    except UnicodeEncodeError as exc:
        raise ValueError("failure_diagnostic_context_invalid") from exc
    return value, tuple(known_secrets)


@dataclass
class _WorktreeFailure:
    known_secrets: tuple[str, ...] = ()
    reason: FailureReason = "denial"
    returncode: int | None = None
    exit_code_source: ExitCodeSource = "not_observed"
    streams: list[StreamDiagnostic] = field(default_factory=list)

    def capture(self, stdout: object, stderr: object, *, complete: bool = True) -> str:
        capture = FanoutOutput(known_secrets=self.known_secrets)
        for name, raw in (("stdout", stdout), ("stderr", stderr)):
            stream_complete = complete
            if isinstance(raw, bytes):
                capture.feed(name, raw)
            elif isinstance(raw, str):
                # Legacy injected text runners count re-encoded UTF-8, not wire
                # bytes. Screen every character, including beyond all tail caps.
                for start in range(0, len(raw), 8192):
                    capture.feed(name, raw[start:start + 8192].encode("utf-8", errors="surrogatepass"))
            elif raw is not None:
                stream_complete = False
            capture.finish(name, complete=stream_complete)
        self.streams = capture.streams()
        return self.streams[1]["text"]

    def process(self, completed: object) -> str:
        code: object = getattr(completed, "returncode", None)
        self.reason = "nonzero"
        if type(code) is int:
            self.returncode, self.exit_code_source = code, "process"
        return self.capture(getattr(completed, "stdout", None), getattr(completed, "stderr", None))

    def exception(self, exc: OSError | subprocess.TimeoutExpired) -> str:
        self.returncode, self.exit_code_source = None, "not_observed"
        if isinstance(exc, subprocess.TimeoutExpired):
            self.reason = "deadline"
            return self.capture(getattr(exc, "output", None), exc.stderr, complete=False)
        self.reason = "missing_binary" if isinstance(exc, FileNotFoundError) else "spawn_error"
        text = self.capture(None, str(exc))
        # An exception message is not captured process stderr.
        self.streams = []
        return text

    def diagnostic(self, identity: FailureDiagnostic) -> FailureDiagnostic:
        value = identity.copy()
        value.update(reason=self.reason, returncode=self.returncode,
                     exit_code_source=self.exit_code_source, streams=self.streams)
        return value


def list_worktree_records(paths: OmhPaths, *, limit: int = 20) -> tuple[list[dict[str, Any]], list[str]]:
    records, errors = read_jsonl_objects(paths.runtime_worktrees_path)
    return list(reversed(records))[:limit], errors


def latest_observed_worktree_record(paths: OmhPaths, worktree_path: str | Path) -> dict[str, Any]:
    records, _errors = read_jsonl_objects(paths.runtime_worktrees_path)
    target = str(expand_path(worktree_path))
    for record in reversed(records):
        if not isinstance(record, dict):
            continue
        if (
            str(record.get("worktree_path", "")) == target
            and record.get("observed")
            and record.get("created")
        ):
            return record
    return {}


def _observation_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": WORKTREE_OBSERVATION_SCHEMA_VERSION,
        "status": result["status"],
        "observed": result["observed"],
        "created": result["created"],
        "repo_root": result["repo_root"],
        "branch": result["branch"],
        "worktree_path": result["worktree_path"],
        "from_ref": result["from_ref"],
        "evidence_refs": result["evidence_refs"],
        "reason": result.get("reason", ""),
        "message": result.get("message", ""),
        "recorded_at": result.get("recorded_at", utc_now()),
        "claim_boundary": WORKTREE_CLAIM_BOUNDARY,
        # Empty on the happy path; a named refusal identifier otherwise, so a
        # caller can branch on which invariant stopped the add rather than
        # substring-matching the human-readable reason.
        "refusal": result.get("refusal", ""),
        **({"failure_diagnostic": result["failure_diagnostic"]} if "failure_diagnostic" in result else {}),
        **{key: result[key] for key in ('unit_id', 'run_ref', 'owner', 'incarnation_id') if key in result},
    }


def _append_worktree_record(paths: OmhPaths, record: dict[str, Any]) -> None:
    ensure_dir(paths.runtime_dir, private=True)
    ensure_file(paths.runtime_worktrees_path, private=True)
    # Dispatch appends from concurrent unit threads; lock the shared ledger.
    with file_lock(paths.runtime_worktrees_path, private=True):
        with paths.runtime_worktrees_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _git(
    runner: Callable[..., object],
    repo_root: Path,
    argv: list[str],
    *,
    timeout: int = 30,
    failure: _WorktreeFailure | None = None,
) -> tuple[int, str, str]:
    """Run one read-only git query, returning (exit code, stdout, stderr).

    A git binary that cannot be run at all is reported as exit 127 rather than
    raised: every caller here is deciding whether an invariant HOLDS, and an
    unanswerable question is a refusal, not a crash.
    """
    try:
        completed: object = runner(argv, cwd=str(repo_root), text=False, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", (failure or _WorktreeFailure()).exception(exc)
    code = _exit_code(getattr(completed, "returncode", 1))
    if code != 0:
        return code, "", (failure or _WorktreeFailure()).process(completed)
    stdout: object = getattr(completed, "stdout", "")
    return code, stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else str(stdout or ""), ""


def _exit_code(value: Any) -> int:
    """A runner that reports no usable exit code answered nothing.

    Test doubles and wrappers hand back objects whose `returncode` is `None`
    or non-numeric. Treating that as success would let an unanswered invariant
    check pass; it is reported as a nonzero code so the caller refuses instead.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _branch_checked_out_paths(runner: Callable[..., Any], repo_root: Path, branch: str) -> list[str]:
    """Worktree paths where `branch` is currently checked out.

    `git worktree add -b` already refuses a branch held by another worktree,
    but only after it has begun creating the new one; asking `git worktree
    list --porcelain` first turns that into a refusal with a name, before any
    directory exists to clean up.
    """
    code, stdout, _stderr = _git(runner, repo_root, ["git", "worktree", "list", "--porcelain"])
    if code != 0:
        return []
    target = f"refs/heads/{branch}"
    holders: list[str] = []
    current_path = ""
    for line in stdout.splitlines():
        if line.startswith("worktree "):
            current_path = line[len("worktree ") :].strip()
        elif line.startswith("branch ") and line[len("branch ") :].strip() == target:
            holders.append(current_path)
    return holders


def _registered_worktree_paths(runner: Callable[..., Any], repo_root: Path) -> set[str]:
    code, stdout, _stderr = _git(runner, repo_root, ["git", "worktree", "list", "--porcelain"])
    if code != 0:
        return set()
    return {
        line[len("worktree ") :].strip()
        for line in stdout.splitlines()
        if line.startswith("worktree ")
    }


def _pre_add_refusal(
    runner: Callable[..., Any],
    *,
    repo_root: Path,
    branch: str,
    base_sha: str,
    source_ref: str,
    worktree_path: Path,
    failure: _WorktreeFailure,
) -> tuple[str, str]:
    """The invariants checked BEFORE `git worktree add`, in refusal order.

    Returns `(refusal_name, reason)`, or `("", "")` when the add may proceed.
    """
    if worktree_path.exists() or str(worktree_path) in _registered_worktree_paths(runner, repo_root):
        return (
            "worktree_path_already_exists",
            f"worktree path already exists: {worktree_path}; remove it or dispatch --unit selectively",
        )
    code, _stdout, _stderr = _git(runner, repo_root, ["git", "check-ref-format", f"refs/heads/{branch}"], failure=failure)
    if code != 0:
        return ("branch_name_malformed", f"branch name is not a valid git ref: {branch!r}")
    if source_ref:
        # The freeze is runtime-shaped: `fanout_contract/v1` stores no base SHA,
        # so the only way to know the caller's `base_sha` still describes the
        # branch it was resolved from is to re-resolve that branch here, at
        # add time. A mismatch means the source branch moved between contract
        # freeze and dispatch, and the unit would silently build on a base
        # nobody agreed to.
        code, stdout, stderr = _git(runner, repo_root, ["git", "rev-parse", "--verify", f"{source_ref}^{{commit}}"], failure=failure)
        if code != 0:
            return (
                "source_ref_unresolvable",
                f"could not resolve source ref {source_ref!r} in {repo_root}: {stderr}",
            )
        observed_sha = stdout.strip()
        if observed_sha != base_sha:
            return (
                "base_sha_drifted_from_source_ref",
                (
                    f"base_sha {base_sha} no longer matches source ref {source_ref!r} "
                    "re-run fanout prepare against the current base"
                ),
            )
    code, _stdout, _stderr = _git(runner, repo_root, ["git", "rev-parse", "--verify", f"refs/heads/{branch}"])
    if code == 0:
        holders = _branch_checked_out_paths(runner, repo_root, branch)
        if holders:
            return (
                "branch_checked_out_in_worktree",
                f"branch {branch!r} is already checked out in {len(holders)} registered worktree(s)",
            )
        return (
            "branch_already_exists",
            f"branch {branch!r} already exists; delete it or dispatch --unit selectively",
        )
    return ("", "")


def _append_cleanup_receipt(
    paths: OmhPaths,
    *,
    unit_id: str,
    run_ref: str,
    refusal: str,
    reason: str,
    worktree_path: Path,
    removed: str,
    left: str,
    diagnostic: FailureDiagnostic | None = None,
) -> None:
    """Record what a refused or failed creation removed and left behind.

    OMH never auto-deletes a worktree, so `removed` is always "nothing" today;
    the field exists because a receipt that cannot say "nothing was removed" is
    a receipt nobody can trust when something is.
    """
    from ..runtime.artifacts import append_journal_observation

    summary = (
        f"{refusal}: removed {removed}; left {left}; worktree {worktree_path}; {reason}"
    )
    append_journal_observation(
        paths,
        {
            "target_type": "run" if run_ref else "runtime",
            "target_id": run_ref or unit_id,
            "run_id": run_ref,
            "event": WORKTREE_CLEANUP_EVENT,
            # A refusal is a failed creation; the diagnostic reader admits a
            # bound diagnostic only on failed/blocked observations.
            "status": "failed" if diagnostic is not None and diagnostic.get("attempt_id") else "observed",
            "summary": summary,
            "worker_ref": unit_id,
            "worktree_ref": str(worktree_path),
            "source": "fanout_worktree_creator",
            # A refused or failed creation belongs to the dispatcher's CURRENT
            # attempt; carrying its identity and sanitized diagnostic here is
            # what stops a later rerun from leaving a stale worker failure (and
            # its resume action) as the unit's public current attempt.
            # `runtime_profile` is the journal's canonical owner field; the
            # diagnostic reader binds the diagnostic's owner against it.
            **({"fanout_id": diagnostic["fanout_id"], "attempt_id": diagnostic["attempt_id"],
                "runtime_profile": diagnostic["owner"], "failure_diagnostic": diagnostic}
               if diagnostic is not None and diagnostic.get("attempt_id") else {}),
        },
    )


def ensure_fanout_unit_worktree(
    paths: OmhPaths,
    *,
    repo_root: Path,
    unit_id: str,
    branch: str,
    base_sha: str,
    source_ref: str = "",
    run_ref: str = "",
    runner: Callable[..., object] = subprocess.run,
    failure_diagnostic_context: Mapping[str, object] | None = None,
    capacity_resume: Mapping[str, object] | None = None,
    contract_digest: str = '',
    confinement: FanoutFilesystemConfinement | None = None,
) -> dict[str, Any]:
    """Create the per-unit worktree for the opt-in fanout dispatch bridge.

    A pre-existing branch or worktree path is an error, never silently reused:
    building on divergent state defeats the contract's isolation guarantee. The
    same holds for a branch some other registered worktree already holds, a
    branch name git would reject, and — when `source_ref` is given — a
    `base_sha` that no longer matches that ref in the live repository.

    Every refusal is recorded, never raised: dispatch reports the unit as
    `worktree_failed` and the operator keeps whatever was already on disk.
    Completed units are skipped by dispatch before this helper is called.

    Optional diagnostic context requires fanout_id, unit_id, run_ref (or None),
    owner, unique attempt_id, worktree_ref, base_sha and observed_revision=None;
    known_secrets is an optional transient list/tuple of strings. Invalid context
    is a typed refusal before Git. Failures return/store failure_diagnostic only
    with valid context. All subprocess/exception text is sanitized even without
    context, before either ledger or cleanup persistence. No raw spill is made.

    `confinement` fences the probe of an existing worktree a capacity resume
    would reuse: an earlier attempt wrote it, git config included (#1999).
    """
    worktree_path = repo_root.parent / f"{repo_root.name}-fanout-{unit_id}"
    result: dict[str, Any] = {
        "status": "failed",
        "observed": False,
        "created": False,
        "repo_root": str(repo_root),
        "branch": branch,
        "worktree_path": str(worktree_path),
        "from_ref": base_sha,
        "evidence_refs": [],
        "recorded_at": utc_now(),
        "refusal": "",
    }

    identity: FailureDiagnostic | None = None
    failure = _WorktreeFailure()

    def refuse(refusal: str, reason: str, *, left: str) -> dict[str, Any]:
        if identity is not None:
            result["failure_diagnostic"] = failure.diagnostic(identity)
        result["refusal"] = refusal
        result["reason"] = reason
        _append_worktree_record(paths, _observation_record(result))
        _append_cleanup_receipt(
            paths,
            unit_id=unit_id,
            run_ref=run_ref,
            refusal=refusal,
            reason=reason,
            worktree_path=worktree_path,
            removed="nothing",
            left=left,
            diagnostic=result.get("failure_diagnostic"),
        )
        return result

    if failure_diagnostic_context is not None:
        try:
            identity, secrets = _diagnostic_context(
                failure_diagnostic_context, unit_id=unit_id, run_ref=run_ref,
                base_sha=base_sha, worktree_path=worktree_path,
            )
        except ValueError:
            return refuse("failure_diagnostic_context_invalid", "Invalid worktree diagnostic context",
                          left="nothing (refused before git worktree add)")
        failure = _WorktreeFailure(known_secrets=secrets)

    if capacity_resume is not None:
        from .fanout_artifacts import fanout_dispatch_summary_path, unit_result_path
        from .inflight import INFLIGHT_DIR_NAME, write_inflight_marker
        fields = read_capacity_fields(capacity_resume)
        lineage, capacity = fields.get('capacity_lineage'), fields.get('capacity')
        expected = {'unit_id': unit_id, 'run_ref': run_ref, 'owner': (failure_diagnostic_context or {}).get('owner'),
                    'base_sha': base_sha, 'worktree_path': str(worktree_path.resolve()),
                    'branch': branch, 'contract_digest': contract_digest}
        if (not is_string_map(lineage) or not is_string_map(capacity)
                or any(lineage.get(key) != value for key, value in expected.items())
                or capacity_resume.get('replay_safe') is not True):
            return refuse('capacity_reuse_unsafe', 'Capacity lineage or replay evidence does not match',
                          left='existing worktree unchanged')
        fanout_id = str(capacity['fanout_id'])
        marker = fanout_dispatch_summary_path(paths, fanout_id).parent / INFLIGHT_DIR_NAME / f'{unit_id}.json'
        if worktree_path.exists():
            # Reuse the existing ledger lock and in-flight marker, not a second
            # queue/lease store. Check and reserve atomically across invocations.
            # No lock spans the child lifetime. An ambiguous marker means HOLD.
            with file_lock(paths.runtime_worktrees_path, private=True):
                observed = observe_session_workspace(str(worktree_path), confinement=confinement)
                ledger = latest_observed_worktree_record(paths, worktree_path)
                safe = (observed is not None and not observed.dirty and observed.head == base_sha
                    and observed.incarnation.incarnation_id == lineage.get('incarnation_id')
                    and ledger.get('incarnation_id') == observed.incarnation.incarnation_id
                    and all(ledger.get(key) == expected[key] for key in ('unit_id', 'run_ref', 'owner', 'branch'))
                    and ledger.get('repo_root') == str(repo_root) and ledger.get('from_ref') == base_sha
                    and not marker.exists() and not marker.is_symlink() and not marker.parent.is_symlink()
                    and not unit_result_path(paths, fanout_id, unit_id).exists())
                if safe and source_ref:
                    code, head, _ = _git(runner, repo_root, ['git', 'rev-parse', '--verify', f'{source_ref}^{{commit}}'])
                    safe = code == 0 and head.strip() == base_sha
                if safe:
                    _ = write_inflight_marker(paths, fanout_id, unit_id,
                        {'owner': expected['owner'], 'run_ref': run_ref, 'worktree': str(worktree_path)})
            if not safe:
                return refuse('capacity_reuse_unsafe', 'Capacity reselection requires the same clean owned idle worktree without a result artifact',
                              left='existing worktree unchanged')
            result.update(status='reused', observed=True, reused=True)
            return result
        if (lineage.get('incarnation_id') is not None or marker.exists() or marker.is_symlink()
                or unit_result_path(paths, fanout_id, unit_id).exists()):
            return refuse('capacity_reuse_unsafe', 'Prior workspace incarnation or in-flight state cannot be discarded',
                          left='existing state unchanged')

    refusal, reason = _pre_add_refusal(
        runner,
        repo_root=repo_root,
        branch=branch,
        base_sha=base_sha,
        source_ref=source_ref,
        worktree_path=worktree_path,
        failure=failure,
    )
    if refusal:
        left = (
            f"pre-existing worktree path {worktree_path}"
            if refusal == "worktree_path_already_exists"
            else "nothing (refused before git worktree add)"
        )
        return refuse(refusal, reason, left=left)
    try:
        # Dispatch creates unit worktrees from a thread pool, and concurrent
        # `git worktree add` calls against one repository contend on shared
        # repo-level lock files (packed-refs/config), which fails
        # intermittently on slow filesystems. Creation is cheap relative to
        # the agent run, so it is serialized process-wide.
        with _WORKTREE_ADD_LOCK:
            completed: object = runner(
                ["git", "worktree", "add", str(worktree_path), "-b", branch, base_sha],
                cwd=str(repo_root),
                text=False,
                capture_output=True,
                timeout=120,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return refuse(
            "worktree_add_failed",
            f"git worktree add failed to run: {failure.exception(exc)}",
            left=_partial_add_state(worktree_path),
        )
    if _exit_code(getattr(completed, "returncode", 1)) != 0:
        safe_stderr = failure.process(completed)
        return refuse(
            "worktree_add_failed",
            f"git worktree add exited nonzero: {safe_stderr}",
            left=_partial_add_state(worktree_path),
        )
    result.update({"status": "created", "observed": True, "created": True, "evidence_refs": [f"git-worktree:{branch}"]})
    observed = observe_session_workspace(str(worktree_path)) if failure_diagnostic_context is not None else None
    if observed is not None and failure_diagnostic_context is not None:
        result.update(unit_id=unit_id, run_ref=run_ref, owner=failure_diagnostic_context['owner'],
                      incarnation_id=observed.incarnation.incarnation_id)
    _append_worktree_record(paths, _observation_record(result))
    return result


def _partial_add_state(worktree_path: Path) -> str:
    """What a failed `git worktree add` left on disk, named for the receipt.

    Nothing is deleted here even when the directory is half-built: a directory
    this process did not finish creating may still hold work a previous run
    left, and no automatic delete is worth that risk (`git worktree prune` and
    an explicit operator `rm -rf` remain the recovery path).
    """
    if worktree_path.exists():
        return f"partially created worktree directory {worktree_path} (not deleted; run `git worktree prune` after removing it)"
    return "nothing on disk"
