"""A checked team of in-session helpers: split, dispatch, check, repair, stop.

The `omh_team` plugin tool is the only caller. What it adds over the ordinary
`delegate_task` fan-out is one rule: a part is done only when OMH has run that
part's frozen check command itself, in this session's workspace, and seen it
exit 0. A helper's own summary, its `schema_valid` flag and the host's child
status are reports; none of them moves a unit.

The record (`omh_team/v1`)
--------------------------
One JSON file per team under ``<omh_home>/runtime/teams/<session key>/``,
keyed by the DURABLE session the reading-session rule resolves (#1794), so a
created TUI's transport id and its durable key find the same team. Every
write is a temp file, ``fsync``, then ``os.replace``, under the record's OS
lock. The record is metadata: titles, the plan item each unit was approved
under, frozen commands, attempt keys, exit codes and times. No helper prompt,
summary or command output is stored -- a check keeps a digest of its output
tail, never the tail.

Team state (``running`` / ``done`` / ``blocked``) is never stored. It is
derived from the units on every read (`team_state`). The record is not a
trust root: anyone who can write the file can also recompute its unkeyed
``commands_digest``. So two facts live only in process memory
(`TeamContext`): which exact command lists a person approved through the
host's gate in this process, and the nonce of every check this process ran.
`team_reconcile` runs nothing for a team whose digest is not in the first,
and a check result whose nonce is not in the second is discarded on read and
the check re-run -- a planted `passed` accepts nothing, and after a restart
every pass is re-earned once the person has approved the team again.

Attempts are write-ahead
------------------------
Before a prepared `delegate_task` entry is returned, the attempt it belongs to
is written to the record with its deterministic key
``<team_id>/<unit_id>/attempt-<n>``. The key rides in the entry's goal as a
marker, which is how the host's `subagent_start` / `subagent_stop` callbacks
(not the model) tell OMH that the attempt is in flight and when it came back.
A marker only ever binds to an attempt the record already reserved; it is an
identifier OMH minted, looked up exactly, never text that is interpreted.

A return is matched on the child session id `subagent_start` recorded. Under
Hermes' rotating compaction (`compression.in_place: false`) a helper that
compacts goes on under a new session id, and `subagent_stop` names that one.
The caller then passes `lineage`, which reads the earlier segments that id
continues from Hermes' own compaction links, and the attempt that recorded
one of them is the one back. Nothing looser counts: not the same parent, not
the time. A return whose lineage cannot be read is not matched; it is kept in
the record as an unlinked return, said in `team_reconcile` while a part is
still out, and named when that part stops as `return_not_linked`. A helper
runs inside the Hermes process that started it, and a restart ends it with no
`subagent_stop`. So the dispatch also records that process (`host`), and the
first `team_start` after a restart hands back each part whose process is gone
(`TeamContext.dispatcher_alive`) instead of holding it for
DELEGATION_STALE_SECONDS. A part with no recorded process, or one whose
process cannot be told alive or gone, waits for that bound as before. The parent
compacts too: its tool calls and its `subagent_stop` then name a later
segment, and the team is found under the segment that started it through the
same links (`owning_session`), or refused as `team_lineage_unverified` when
they cannot be read.

Check outcomes
--------------
``passed`` accepts the unit. ``failed`` (the command ran and exited non-zero)
spends an attempt. ``inconclusive`` (timeout, command not found, killed by a
signal, missing workspace, a workspace outside git, the working tree changed
while the check ran, or a helper branch git could not read) spends nothing,
and three of them on one attempt block the unit with that reason so the team
still reaches a stop. ``refused`` (the command policy or Hermes' own floor
refuses it at run time) runs nothing and blocks the unit. One `team_reconcile`
runs at most MAX_CHECKS_PER_RECONCILE checks, each leased just before it runs.

Helpers in worktrees of their own
---------------------------------
With Hermes' ``delegation.worktree_isolation`` on, in a git workspace with
Hermes' terminal running locally, a helper edits a git worktree of its own, on
branch ``hermes-subagent/subagent-<id>`` named after the subagent id
`subagent_start` carries, and its work reaches this workspace only when the
session merges that branch. OMH never merges. A check run before
the merge tests a tree without the helper's work, so `team_reconcile` checks
nothing while a returned helper's branch has commits this workspace's HEAD
does not contain, or its worktree has uncommitted changes: it refuses with
``helper_branch_unmerged`` and names each such branch. The worktree is found
by the path Hermes gave it as well as by the branch, so a helper that left the
branch (a detached HEAD, a branch of its own) is still read. Merged means the
branch's last commit and the worktree's HEAD are both in HEAD's history; work
held by neither (Hermes prunes a worktree whose helper changed nothing, and
the session removes one it has merged) is not looked for.

What this does not see
----------------------
Without worktree isolation, helpers share one checkout. A check is attributed
to the unit whose command it is, but a different unit's edits are in the same
tree; the barrier (no check while any team delegation is in flight) narrows
that, it does not remove it. With isolation the merged branches put the same
mix in the tree. A check executes code the helpers wrote, as the operator's OS
user with the real HOME. Both are stated in `team_status` and in the approval
prompt.
The unmerged-branch guard keeps an honest session from checking too early; it
is not a trust boundary. The subagent id it reads is in the record, so a record
edited to drop it skips the guard, and the branch and worktree names are
Hermes' naming: a host that names them differently leaves the guard finding
nothing, and checks run as they did before it. Work a helper moved out of both
(a branch renamed or deleted after its worktree was removed) is not looked for.
The approval is the host's: under ``--yolo``, ``approvals.mode: off`` or a
cron ``approve`` mode Hermes answers it without asking anyone. The command
policy and Hermes' own floor still apply there, and `team_start` refuses in a
cron session, but a person who turned approvals off has not seen the list.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
from typing import Any, Final

from ..plugin_bundle.omh import dag_core
from ..plugin_bundle.omh.host_process import recorded_process
from ..system.local_store import _with_windows_retry, ensure_dir, file_lock

TEAM_SCHEMA_VERSION: Final = "omh_team/v1"
TEAM_RESULT_SCHEMA_VERSION: Final = "omh_team_result/v1"
# The check timeout, and the reason it is not the evidence tool's 180 s cap: a
# real repository suite runs for minutes, and a timeout here is inconclusive
# rather than a failure, so a cap below the suite's length would only ever
# produce "blocked: the check kept timing out". Ten minutes covers the suites
# this repository has measured while still ending a hung check inside one turn.
TEAM_CHECK_TIMEOUT_SECONDS: Final = 600
# A lease outlives the longest check by this margin, so a crashed checker's
# lease expires and the check is re-run without spending an attempt.
TEAM_LEASE_MARGIN_SECONDS: Final = 120
DEFAULT_MAX_REPAIR_ATTEMPTS: Final = 2
MAX_REPAIR_ATTEMPTS_CAP: Final = 3
MAX_TEAM_UNITS: Final = 8
MAX_TEAMS_PER_SESSION: Final = 16
# Three inconclusive checks on one attempt block the unit with that reason.
MAX_INCONCLUSIVE_PER_ATTEMPT: Final = 3
# A reserved attempt the host never reported dispatching is handed back at
# most this many times before the unit blocks as `dispatch_not_observed`; one
# whose helper a Hermes restart ended, as many times again, counted apart,
# before it blocks as `helper_lost`.
MAX_REEMITS: Final = 3
# A dispatched helper the host never reported returning is treated as lost
# after this long; the unit blocks rather than holding the barrier forever.
DELEGATION_STALE_SECONDS: Final = 4 * 3600
# An entry handed out but not yet reported started by the host counts as in
# flight for this long: the host fires `subagent_start` within seconds of a
# `delegate_task` call, so a later reconcile neither checks around it nor hands
# the same entry out twice. Past it, the entry is handed back (MAX_REEMITS).
EMIT_GRACE_SECONDS: Final = 120
# Checks one `team_reconcile` runs; the rest wait for the next call. Each check
# can take TEAM_CHECK_TIMEOUT_SECONDS, so this bounds one turn at two of them.
MAX_CHECKS_PER_RECONCILE: Final = 2
MAX_TEAM_RECORD_BYTES: Final = 262_144
# Returns kept per record that OMH saw but could not link to any part; the
# oldest drop first.
MAX_UNLINKED_RETURNS: Final = 8
MAX_TITLE_CHARS: Final = 80
MAX_COMMAND_CHARS: Final = 200
UNIT_STATES: Final = (
    "waiting", "prepared", "dispatched", "awaiting_check", "repairing", "accepted", "blocked",
)
TEAM_STATES: Final = ("running", "done", "blocked")
CHECK_OUTCOMES: Final = ("passed", "failed", "inconclusive", "refused")
EVIDENCE_KIND: Final = "team_check"
# Renderable team events: a header (`teammate` + `event`), one plain summary
# line, and a `detail_ref` into this record. Append-only with a monotonic
# `seq`; only the newest MAX_TEAM_EVENTS are kept (the oldest drop first), so
# a record's event history is bounded at 64 entries. Metadata only: no helper
# summary and no command output is ever an event.
TEAM_EVENTS: Final = ("started", "finished", "lost", "check_passed", "check_failed", "repairing", "blocked", "done")
MAX_TEAM_EVENTS: Final = 64
TEAM_CAVEAT: Final = (
    "Helpers run under this same Hermes profile, and each check runs the code they wrote as your own "
    "user account with your real home folder. Every check runs in this one workspace, which also holds "
    "the other parts' edits, so a check can see another part's work."
)
# INVARIANT 3 (`tests/test_handoff_safety_contract_enforcement.py`) applied to
# the one command-execution surface that does NOT go through the evidence
# allowlist: a team command is approved by the person, not matched against a
# prefix list, so these are refused by program wherever they appear in argv.
# This is one layer. The handler also passes every command through Hermes'
# own hardline floor and the person's `approvals.deny` rules before it is
# frozen and again before it runs (`TeamContext.host_guard`), and refuses when
# that floor cannot be reached.
TEAM_FORGE_PROGRAMS: Final = frozenset({"gh", "hub", "glab", "tea"})
TEAM_FORBIDDEN_GIT_VERBS: Final = frozenset({"push", "merge", "rebase", "remote", "fetch", "pull"})
# Git verbs that rewrite the local checkout or its history. A check reads the
# tree; it never moves it.
TEAM_GIT_MUTATING_VERBS: Final = frozenset({
    "reset", "clean", "checkout", "commit", "switch", "restore", "stash", "am", "apply", "cherry-pick",
    "revert", "tag", "branch", "config", "update-ref", "filter-branch", "submodule", "gc", "prune",
})
# `git -c alias.x=!cmd x` runs any program: configuration and a moved git
# directory are refused as options, and `alias.` / `core.` wherever they appear.
TEAM_GIT_CONFIG_OPTIONS: Final = frozenset({"-c", "-C", "--config", "--config-env", "--exec-path"})
# A shell, `env` or `xargs` as the program turns one approved line into any line.
TEAM_SHELL_PROGRAMS: Final = frozenset({
    "sh", "bash", "zsh", "dash", "ksh", "fish", "csh", "tcsh", "env", "xargs", "cmd", "cmd.exe",
    "powershell", "powershell.exe", "pwsh", "pwsh.exe",
})
# Interpreters whose options take program text inline. Any of these option
# letters in a short cluster (`-e`, `-Ic`, `-ec`), a long `--eval` / `--print`
# style option, or an `eval` subcommand is refused.
TEAM_INTERPRETERS: Final = frozenset({
    "perl", "ruby", "node", "nodejs", "deno", "bun", "php", "osascript", "lua", "luajit",
})
_INLINE_OPTION_LETTERS: Final = frozenset("ceEpr")
_INLINE_LONG_OPTIONS: Final = ("--eval", "--print", "--command", "--exec")
TEAM_PRIVILEGE_PROGRAMS: Final = frozenset({"sudo", "doas", "su"})
TEAM_NETWORK_PROGRAMS: Final = frozenset({
    "ssh", "scp", "sftp", "rsync", "curl", "wget", "nc", "ncat", "netcat", "socat", "telnet", "ftp",
})
TEAM_FILE_PROGRAMS: Final = frozenset({"rm", "rmdir", "mv", "chmod", "chown", "chgrp", "dd", "shred", "truncate"})
# Programs whose only job is to fetch, install or publish a package.
TEAM_PACKAGE_PROGRAMS: Final = frozenset({
    "pip", "pip3", "pipx", "twine", "npx", "pnpx", "bunx", "uvx", "ensurepip",
})
# Package managers whose ordinary test verbs stay allowed, with the verbs that
# install, run a fetched package, or publish.
TEAM_PACKAGE_VERBS: Final[dict[str, frozenset[str]]] = {
    "npm": frozenset({"publish", "unpublish", "install", "i", "ci", "add", "exec", "x", "link", "login",
                      "adduser", "dist-tag", "owner", "access", "deprecate"}),
    "pnpm": frozenset({"publish", "install", "i", "add", "dlx", "exec", "link", "login"}),
    "yarn": frozenset({"publish", "install", "add", "dlx", "exec", "link", "login", "npm"}),
    "bun": frozenset({"publish", "install", "i", "add", "x", "link", "pm"}),
    "uv": frozenset({"publish", "pip", "tool", "add", "remove", "self"}),
    "cargo": frozenset({"publish", "install", "login", "yank", "owner"}),
    "poetry": frozenset({"publish", "add", "remove", "install", "self", "config"}),
    "gem": frozenset({"push", "install", "owner", "yank", "signin"}),
    "go": frozenset({"install", "get"}),
    "docker": frozenset({"push", "login"}),
    "podman": frozenset({"push", "login"}),
}
_FIND_ACTIONS: Final = frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir"})
_SHELL_METACHARACTERS: Final = re.compile(r"[\n\r;&|`$<>(){}]")
# Bidirectional overrides and zero-width characters make a line read
# differently from what runs; refused in commands and titles.
_HIDDEN_CHARACTERS: Final = re.compile(
    "[\u00ad\u061c\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\u2028\u2029\ufeff]"
)
# The plan binding: a command is approved only as the exact text of a
# `check: `<command>`` field in an accepted plan item.
_CHECK_FIELD: Final = re.compile(r"check:\s*`([^`\n]+)`", re.IGNORECASE)
_TEAM_ID: Final = re.compile(r"[a-z0-9][a-z0-9-]{0,47}")
_UNIT_ID: Final = re.compile(r"[a-z0-9][a-z0-9_-]{0,47}")
# The attempt key a helper's goal ends with, so Hermes' 40-character previews show the part's title first. The last
# whole key in a goal binds: a goal handed out before an update opens with it, and a model may add text after it. A
# goal whose last marker is a dag node's is that node's child, even when an older title quoted a key before it
# (`delegation_ledger.dag_binding` reads the mirror).
_MARKER: Final = re.compile(
    r"\[(?:omh-team:([a-z0-9][a-z0-9-]{0,47})/([a-z0-9][a-z0-9_-]{0,47})/attempt-([1-9][0-9]?)"
    r"|dag:[a-z0-9][a-z0-9-]{0,31}/[a-z0-9][a-z0-9-]{0,31})\]"
)
_CONTROL: Final = re.compile(r"[\x00-\x1f\x7f-\x9f]")
# The subagent id Hermes hands `subagent_start` (`sa-<index>-<hex>` at v2026.9.7)
# and the branch it names a helper's worktree after
# (`tools/subagent_worktree.py:create_subagent_worktree`).
_SUBAGENT_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
HELPER_BRANCH_PREFIX: Final = "hermes-subagent/subagent-"
# What `TeamContext.helper_work` reports for one helper branch, seen from this
# workspace. Any other answer is read as `unreadable`.
HELPER_WORK_STATES: Final = ("absent", "merged", "commits_not_merged", "uncommitted_changes", "unreadable")
_HELPER_WORK_UNMERGED: Final = frozenset({"commits_not_merged", "uncommitted_changes"})
_FINGERPRINT_UNAVAILABLE: Final = frozenset({"timed_out", "unavailable"})
# Outside a git checkout the fingerprint cannot see a change, so a check there
# is never conclusive: the team is for git workspaces.
_FINGERPRINT_UNTRACKED: Final = frozenset({"not_a_repository", "unsupported"})


class TeamRefusal(ValueError):
    """A closed refusal. ``reason`` is a code; ``say`` is one plain sentence.

    ``unmerged`` names, for ``helper_branch_unmerged`` only, each helper branch
    whose work is not in this workspace yet: ``unit_id``, ``branch``, ``state``.
    """

    def __init__(self, reason: str, say: str, *, unmerged: tuple[dict[str, str], ...] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.say = say
        self.unmerged = unmerged


@dataclass(frozen=True)
class CheckRun:
    """What the runner observed: ``exited`` with a code, ``not_found`` or ``timeout``."""

    outcome: str
    exit_code: int | None
    output_tail: str


Runner = Callable[[list[str], Path, int], CheckRun]
Fingerprint = Callable[[Path], tuple[str, str | None]]
# The host's own command floor: None to allow, else a refusal reason code.
HostGuard = Callable[[str], str | None]
# Where one helper branch stands against this workspace: one of HELPER_WORK_STATES.
HelperWork = Callable[[Path, str], str]
# Whether the Hermes process a dispatch recorded still runs: True, False (gone), or None (cannot tell).
DispatcherAlive = Callable[[Mapping[str, Any]], bool | None]


@dataclass(frozen=True)
class TeamContext:
    """What one call needs. ``approved`` and ``observed_checks`` live in process memory.

    ``approved`` holds ``(session_ref, team_id, commands_digest)`` for every
    team whose `team_start` passed the host's approval gate in this process;
    the handler adds to it and nothing else does. ``observed_checks`` holds the
    nonce of every check this process ran. Neither is ever read from disk, so a
    record written by anyone else can neither run a command nor carry a pass.
    ``helper_work`` reads, through git, whether a helper's branch is merged
    into the workspace (`HELPER_WORK_STATES`). ``dispatcher_alive`` asks the
    host whether the process a dispatch recorded still runs; only False hands
    a part back.
    """

    omh_home: Path
    session_ref: str
    runner: Runner
    fingerprint: Fingerprint
    now: Callable[[], float]
    host_guard: HostGuard
    helper_work: HelperWork
    approved: set[tuple[str, str, str]]
    observed_checks: set[str]
    dispatcher_alive: DispatcherAlive


# --------------------------------------------------------------------------
# Command policy
# --------------------------------------------------------------------------


def validate_team_command(command: object) -> list[str]:
    """The argv of an approvable team check, or a refusal naming why not.

    Deliberately stricter than "not on a denylist": the command is refused when
    ANY argv word is a refused program, because `uv run gh ...` runs `gh` just
    as surely as `gh`. The lists are one layer, not the boundary: the handler
    also runs each command past Hermes' own floor (`TeamContext.host_guard`).
    """
    if not isinstance(command, str) or not command.strip():
        raise TeamRefusal("command_required", "Every part needs a check command.")
    text = command.strip()
    if _HIDDEN_CHARACTERS.search(text):
        raise TeamRefusal("command_hidden_characters", "A check command cannot contain invisible characters.")
    if len(text) > MAX_COMMAND_CHARS or _CONTROL.search(text):
        raise TeamRefusal("command_too_long", "A check command must be one short line.")
    if _SHELL_METACHARACTERS.search(text):
        raise TeamRefusal("command_shell_syntax", "A check command cannot use shell syntax such as pipes or ';'.")
    try:
        tokens = shlex.split(text)
    except ValueError:
        raise TeamRefusal("command_unparseable", "A check command could not be read as one command.") from None
    if not tokens:
        raise TeamRefusal("command_required", "Every part needs a check command.")
    for index, token in enumerate(tokens):
        _refuse_program(_program_name(token), tokens[index + 1:])
    return tokens


def _refuse_program(name: str, rest: list[str]) -> None:
    """Refuse one argv word as a program, given every word after it."""
    if name in TEAM_SHELL_PROGRAMS:
        raise TeamRefusal("command_runs_a_shell", "A check command cannot start a shell.")
    if name in TEAM_FORGE_PROGRAMS:
        raise TeamRefusal("command_talks_to_a_forge", "A check command cannot call GitHub or another forge.")
    if name in TEAM_PRIVILEGE_PROGRAMS:
        raise TeamRefusal("command_escalates_privilege", "A check command cannot run as another user.")
    if name in TEAM_NETWORK_PROGRAMS:
        raise TeamRefusal("command_uses_the_network", "A check command cannot reach another machine.")
    if name in TEAM_FILE_PROGRAMS or (name == "find" and set(rest) & _FIND_ACTIONS):
        raise TeamRefusal("command_changes_files", "A check command cannot delete, move or re-permission files.")
    if name in TEAM_PACKAGE_PROGRAMS or set(rest) & TEAM_PACKAGE_VERBS.get(name, frozenset()):
        raise TeamRefusal("command_installs_or_publishes", "A check command cannot install or publish a package.")
    if name == "git":
        _refuse_git(rest)
    if name.startswith(("python", "pypy")):
        # Python's own options end at `-m`; what follows belongs to the module.
        _refuse_inline(rest[:rest.index("-m")] if "-m" in rest else rest)
    elif name in TEAM_INTERPRETERS:
        _refuse_inline(rest)


def _refuse_git(rest: list[str]) -> None:
    words = set(rest)
    if words & TEAM_FORBIDDEN_GIT_VERBS:
        raise TeamRefusal("command_moves_a_remote", "A check command cannot push, pull, merge or rebase.")
    if words & TEAM_GIT_MUTATING_VERBS:
        raise TeamRefusal("command_rewrites_git", "A check command cannot change the checkout or its history.")
    for word in rest:
        lowered = word.lower()
        if (word in TEAM_GIT_CONFIG_OPTIONS or lowered.startswith(("--config", "--exec-path"))
                or "alias." in lowered or "core." in lowered):
            raise TeamRefusal("command_rewrites_git", "A check command cannot pass git settings or move git's folder.")


def _refuse_inline(options: list[str]) -> None:
    for word in options:
        cluster = word.startswith("-") and not word.startswith("--") and set(word[1:]) & _INLINE_OPTION_LETTERS
        if cluster or word.startswith(_INLINE_LONG_OPTIONS) or word == "eval":
            raise TeamRefusal("command_inline_program", "A check command cannot run program text written inline.")


def _program_name(token: str) -> str:
    return token.replace("\\", "/").rsplit("/", 1)[-1].lower()


def frozen_commands_digest(units: list[Mapping[str, Any]]) -> str:
    """One digest over every unit's id and command, in unit order."""
    pairs = [[str(unit.get("unit_id", "")), str(unit.get("verification_command", ""))] for unit in units]
    return _digest(pairs)


def approval_rule_key(commands: list[str]) -> str:
    """The `[a]lways` grain for one exact command list, never for the tool."""
    return "omh_team_start:" + _digest(commands)[:16]


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


def _digest(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def session_key(session_ref: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", session_ref).strip("_-")[:48]
    digest = hashlib.sha256(session_ref.encode("utf-8")).hexdigest()[:16]
    return f"{slug}-{digest}" if slug else digest


def teams_dir(omh_home: Path, session_ref: str) -> Path:
    return Path(omh_home) / "runtime" / "teams" / session_key(session_ref)


def team_path(omh_home: Path, session_ref: str, team_id: str) -> Path:
    if not isinstance(team_id, str) or not _TEAM_ID.fullmatch(team_id):
        raise TeamRefusal("invalid_team_id", "A team name must be short lowercase letters, digits and dashes.")
    return teams_dir(omh_home, session_ref) / f"{team_id}.json"


def _reject_links(path: Path, omh_home: Path) -> None:
    current = path
    root = Path(omh_home)
    while True:
        if current.is_symlink():
            raise TeamRefusal("team_store_unsafe", "The team record location is not safe to write.")
        if current == root or current.parent == current:
            return
        current = current.parent


def read_team(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_TEAM_RECORD_BYTES + 1)
    except FileNotFoundError:
        return None
    except OSError:
        raise TeamRefusal("team_store_unreadable", "The team record could not be read.") from None
    if len(raw) > MAX_TEAM_RECORD_BYTES:
        raise TeamRefusal("team_store_unreadable", "The team record could not be read.")
    try:
        record = json.loads(raw)
    except ValueError:
        raise TeamRefusal("team_store_unreadable", "The team record could not be read.") from None
    if not isinstance(record, dict) or record.get("schema_version") != TEAM_SCHEMA_VERSION:
        raise TeamRefusal("team_store_unreadable", "The team record could not be read.")
    if not _record_shape_ok(record):
        raise TeamRefusal("team_store_unreadable", "The team record could not be read.")
    return record


def _record_shape_ok(record: Mapping[str, Any]) -> bool:
    """Every field a reader indexes has the type it is read as, so a damaged record is refused, not a crash."""
    if not all(isinstance(record.get(key), str) for key in ("team_id", "session_ref", "commands_digest", "workdir")):
        return False
    if not _is_int(record.get("max_repair_attempts")) or not _is_int(record.get("next_seq")):
        return False
    units = record.get("units")
    events = record.get("events")
    if not isinstance(units, list) or not 1 <= len(units) <= MAX_TEAM_UNITS or not isinstance(events, list):
        return False
    if not all(isinstance(event, dict) and _is_int(event.get("seq")) and isinstance(event.get("event"), str)
               for event in events):
        return False
    unlinked = record.get("unlinked_returns", [])
    if not isinstance(unlinked, list) or not all(
        isinstance(note, dict) and all(isinstance(note.get(key), str)
                                       for key in ("child_session_id", "observed_at", "reason"))
        for note in unlinked
    ):
        return False
    ids = [unit.get("unit_id") if isinstance(unit, dict) else None for unit in units]
    return all(_unit_shape_ok(unit, ids) for unit in units)


def _unit_shape_ok(unit: object, ids: list[object]) -> bool:
    if not isinstance(unit, dict):
        return False
    if not all(isinstance(unit.get(key), str) for key in ("unit_id", "title", "verification_command", "plan_item")):
        return False
    depends_on = unit.get("depends_on")
    if not isinstance(depends_on, list) or not all(isinstance(item, str) and item in ids for item in depends_on):
        return False
    if not all(unit.get(key) is None or isinstance(unit.get(key), dict) for key in ("blocked", "lease")):
        return False
    attempts = unit.get("attempts")
    if not isinstance(attempts, list):
        return False
    for number, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, dict) or attempt.get("n") != number or not isinstance(attempt.get("inconclusive"), list):
            return False
        dispatch = attempt.get("dispatch")
        if dispatch is not None and not (isinstance(dispatch, dict) and isinstance(dispatch.get("child_session_id"), str)
                                         and isinstance(dispatch.get("subagent_id", ""), str)
                                         and (dispatch.get("host") is None or recorded_process(dispatch["host"]))):
            return False
        lost = attempt.get("lost", [])
        if not isinstance(lost, list) or not all(isinstance(item, dict) for item in lost):
            return False
        check = attempt.get("check")
        if check is not None and not (isinstance(check, dict) and check.get("outcome") in ("passed", "failed")
                                      and all(key in check for key in ("command", "exit_code", "observed_at"))):
            return False
    return True


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def write_team(path: Path, record: Mapping[str, Any]) -> None:
    """Temp file, fsync, `os.replace`: a reader sees the old record or the new one."""
    content = json.dumps(record, indent=2, sort_keys=True) + "\n"
    data = content.encode("utf-8")
    if len(data) > MAX_TEAM_RECORD_BYTES:
        raise TeamRefusal("team_record_too_large", "The team record grew past its size limit.")
    ensure_dir(path.parent, private=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}-{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _with_windows_retry(lambda: os.replace(temporary, path))
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


@contextmanager
def locked_team(omh_home: Path, session_ref: str, team_id: str) -> Iterator[Path]:
    path = team_path(omh_home, session_ref, team_id)
    _reject_links(path.parent, Path(omh_home))
    ensure_dir(path.parent, private=True)
    _reject_links(path.parent, Path(omh_home))
    try:
        with file_lock(path, private=True, timeout_seconds=5.0) as lock:
            if not lock.get("enforced"):
                raise TeamRefusal("team_lock_unavailable", "This system cannot lock the team record.")
            yield path
    except TimeoutError:
        raise TeamRefusal("team_record_busy", "Another step is updating this team; try again.") from None


# --------------------------------------------------------------------------
# Derived state
# --------------------------------------------------------------------------


def _latest(unit: Mapping[str, Any]) -> dict[str, Any] | None:
    attempts = unit.get("attempts") or []
    return attempts[-1] if attempts else None


def unit_state(unit: Mapping[str, Any]) -> str:
    if unit.get("blocked"):
        return "blocked"
    attempt = _latest(unit)
    if attempt is None:
        return "waiting"
    check = attempt.get("check")
    if check and check.get("outcome") == "passed":
        return "accepted"
    if attempt.get("returned_at"):
        return "awaiting_check"
    if attempt.get("n", 1) > 1:
        return "repairing"
    if attempt.get("dispatch"):
        return "dispatched"
    return "prepared"


def _blocked_ancestry(units: list[Mapping[str, Any]]) -> set[str]:
    """Unit ids that can never run: blocked, or downstream of a blocked unit."""
    by_id = {str(unit["unit_id"]): unit for unit in units}
    dead: set[str] = set()
    changed = True
    while changed:
        changed = False
        for unit_id, unit in by_id.items():
            if unit_id in dead:
                continue
            if unit.get("blocked") or any(parent in dead for parent in unit.get("depends_on", [])):
                dead.add(unit_id)
                changed = True
    return dead


def team_state(record: Mapping[str, Any]) -> str:
    """`done`, `blocked` or `running`, derived from the units -- never read from a field."""
    units = list(record.get("units") or [])
    if units and all(unit_state(unit) == "accepted" for unit in units):
        return "done"
    dead = _blocked_ancestry(units)
    live = [unit for unit in units if unit_state(unit) != "accepted" and str(unit["unit_id"]) not in dead]
    if not live and any(unit.get("blocked") for unit in units):
        return "blocked"
    return "running"


def _attempt_limit(record: Mapping[str, Any]) -> int:
    return 1 + int(record.get("max_repair_attempts", DEFAULT_MAX_REPAIR_ATTEMPTS))


def attempt_key(team_id: str, unit_id: str, n: int) -> str:
    return f"{team_id}/{unit_id}/attempt-{n}"


def evidence_ref(team_id: str, unit_id: str, n: int) -> dict[str, str]:
    return {"kind": EVIDENCE_KIND, "ref": f"{attempt_key(team_id, unit_id, n)}/check"}


# --------------------------------------------------------------------------
# Prepared delegate entries
# --------------------------------------------------------------------------


def _delegate_entry(team_id: str, unit: Mapping[str, Any], attempt: Mapping[str, Any]) -> dict[str, str]:
    key = attempt_key(team_id, str(unit["unit_id"]), int(attempt["n"]))
    command = str(unit["verification_command"])
    # True whether or not Hermes isolates helpers: the check always runs in the
    # parent session's workspace, and a helper in a worktree of its own learns that
    # from the note Hermes appends to its goal, not from this text.
    lines = [
        f"Part of a checked team: {unit['plan_item']}",
        f"When you stop, OMH runs this command in the parent session's workspace and the part counts as "
        f"done only if it exits 0 there: {command}",
        "If Hermes gave you a git worktree of your own, your edits reach that workspace only when the "
        "parent session merges your branch, so commit them on your branch before you stop.",
        "Stay inside this part; other parts are being worked on at the same time.",
    ]
    failing = attempt.get("repairs_check")
    if isinstance(failing, Mapping):
        lines.append("The previous try failed its check: " + json.dumps(
            {"command": failing.get("command"), "exit_code": failing.get("exit_code")}, sort_keys=True))
        lines.append("Fix the cause and stop when that command would exit 0.")
    return {"goal": f"{unit['title']} [omh-team:{key}]", "context": "\n".join(lines)}


def _reserve(unit: dict[str, Any], now: float, *, repairs_check: Mapping[str, Any] | None = None) -> dict[str, Any]:
    attempts = unit.setdefault("attempts", [])
    attempt: dict[str, Any] = {
        "n": len(attempts) + 1,
        "reserved_at": _iso(now),
        "reemits": 0,
        "dispatch": None,
        "returned_at": None,
        "check": None,
        "inconclusive": [],
    }
    if repairs_check is not None:
        attempt["repairs_check"] = {
            "command": repairs_check["command"], "exit_code": repairs_check["exit_code"],
        }
    attempts.append(attempt)
    return attempt


def _emit(
    record: dict[str, Any],
    now: float,
    *,
    unit: Mapping[str, Any] | None,
    event: str,
    summary: str,
    detail_ref: str,
) -> None:
    """Append one renderable event; `seq` never repeats, even after the cap drops old ones."""
    seq = int(record.get("next_seq", 1))
    record["next_seq"] = seq + 1
    events = record.setdefault("events", [])
    events.append({
        "seq": seq,
        "at": _iso(now),
        "unit_id": str(unit["unit_id"]) if unit is not None else "",
        "teammate": str(unit["title"]) if unit is not None else "The team",
        "event": event,
        "summary": summary,
        "detail_ref": detail_ref,
    })
    del events[:-MAX_TEAM_EVENTS]


def _block(record: dict[str, Any], unit: dict[str, Any], now: float, blocked: dict[str, Any]) -> None:
    unit["blocked"] = blocked
    why = _BLOCKED_SAY.get(str(blocked.get("reason")), "it cannot continue")
    _emit(record, now, unit=unit, event="blocked", summary=f"Stopped: {why}.",
          detail_ref=f"{record['team_id']}/{unit['unit_id']}/blocked")


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


# --------------------------------------------------------------------------
# team_start
# --------------------------------------------------------------------------


def _plain(value: object, limit: int, reason: str, say: str) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value.strip()) > limit or _CONTROL.search(value)
            or _HIDDEN_CHARACTERS.search(value)):
        raise TeamRefusal(reason, say)
    return value.strip()


def _validated_units(raw_units: object) -> list[dict[str, Any]]:
    if not isinstance(raw_units, list) or not 1 <= len(raw_units) <= MAX_TEAM_UNITS:
        raise TeamRefusal("invalid_units", f"A team needs between 1 and {MAX_TEAM_UNITS} parts.")
    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_units:
        if not isinstance(raw, Mapping) or set(raw) - {"unit_id", "title", "depends_on", "verification_command"}:
            raise TeamRefusal("invalid_units", "Each part takes only unit_id, title, depends_on and verification_command.")
        unit_id = raw.get("unit_id")
        if not isinstance(unit_id, str) or not _UNIT_ID.fullmatch(unit_id) or unit_id in seen:
            raise TeamRefusal("invalid_unit_id", "Each part needs its own short lowercase id.")
        seen.add(unit_id)
        depends_on = raw.get("depends_on", [])
        if not isinstance(depends_on, list) or not all(isinstance(item, str) for item in depends_on):
            raise TeamRefusal("invalid_depends_on", "depends_on must list other parts by id.")
        command = raw.get("verification_command")
        validate_team_command(command)
        title = _plain(raw.get("title"), MAX_TITLE_CHARS, "invalid_title", "Each part needs a short title.")
        # Either lane's reader also takes a marker that opens a goal (one handed out before an update), so a title
        # holding either marker could bind a helper to another part or dag node.
        if "[omh-team:" in title or "[dag:" in title:
            raise TeamRefusal("invalid_title",
                              "A part's title cannot hold an [omh-team:...] key or a [dag:...] marker; OMH adds its own.")
        units.append({
            "unit_id": unit_id,
            "title": title,
            "depends_on": sorted(set(depends_on)),
            "verification_command": str(command).strip(),
        })
    known = {unit["unit_id"] for unit in units}
    for unit in units:
        if dag_core.edge_problem(unit["unit_id"], unit["depends_on"], known) is not None:
            raise TeamRefusal("invalid_depends_on", "depends_on must name other parts of this team.")
    _require_acyclic(units)
    return units


def _require_acyclic(units: list[Mapping[str, Any]]) -> None:
    ids = [str(unit["unit_id"]) for unit in units]
    if dag_core.layers(ids, {str(unit["unit_id"]): unit["depends_on"] for unit in units}) is None:
        raise TeamRefusal("dependency_cycle", "The parts depend on each other in a circle.")


def _bind_to_plan(units: list[dict[str, Any]], plan: Mapping[str, Any], plan_ref: object) -> None:
    """The approval binding (R7): accepted plan, same digest, every command a `check:` field in it.

    A command binds only to an item that names it exactly as ``check: `<command>```.
    A command merely mentioned in prose ("do NOT run make deploy"), or a prefix
    of one, binds nothing.
    """
    if not plan.get("own_record") or plan.get("status") not in ("established", "all_done"):
        raise TeamRefusal("plan_not_found", "Start a team only from this session's own current plan.")
    if plan.get("plan_stage") != "accepted":
        raise TeamRefusal("plan_not_accepted", "The plan has not been accepted yet, so no team can start.")
    items = plan.get("items")
    if not isinstance(plan_ref, str) or not plan_ref or plan_ref != plan.get("items_digest"):
        raise TeamRefusal("plan_ref_mismatch", "The plan changed since it was accepted, or plan_ref names another plan.")
    texts = [str(item.get("text", "")) for item in items or [] if isinstance(item, Mapping)]
    for unit in units:
        item = next((text for text in texts if unit["verification_command"] in _check_fields(text)), None)
        if item is None:
            raise TeamRefusal(
                "command_not_in_accepted_plan",
                f"The check for '{unit['title']}' is not written in the accepted plan as check: `<command>`, "
                "so it was never approved.",
            )
        unit["plan_item"] = item


def _check_fields(text: str) -> list[str]:
    return [match.group(1).strip() for match in _CHECK_FIELD.finditer(text)]


def _require_host_allows(ctx: TeamContext, command: str) -> None:
    reason = ctx.host_guard(command)
    if reason is not None:
        raise TeamRefusal(reason, _HOST_REFUSAL_SAY.get(reason, _HOST_REFUSAL_SAY["command_refused_by_host"]))


_HOST_REFUSAL_SAY: Final[dict[str, str]] = {
    "command_refused_by_host": "Hermes' own command rules refuse this check command.",
    "host_command_floor_unavailable": "Hermes' own command rules could not be reached, so no check command is allowed.",
}


def team_start(
    ctx: TeamContext,
    *,
    team_id: object,
    units: object,
    plan: Mapping[str, Any],
    plan_ref: object,
    workdir: Path | None,
    max_repair_attempts: object = None,
) -> dict[str, Any]:
    validated = _validated_units(units)
    repairs = DEFAULT_MAX_REPAIR_ATTEMPTS if max_repair_attempts is None else max_repair_attempts
    if isinstance(repairs, bool) or not isinstance(repairs, int) or not 0 <= repairs <= MAX_REPAIR_ATTEMPTS_CAP:
        raise TeamRefusal("invalid_max_repair_attempts", f"Fix-up tries can be 0 to {MAX_REPAIR_ATTEMPTS_CAP}.")
    for unit in validated:
        _require_host_allows(ctx, unit["verification_command"])
    _bind_to_plan(validated, plan, plan_ref)
    if workdir is None or not Path(workdir).is_dir():
        raise TeamRefusal("workspace_missing", "This session has no workspace folder to run checks in.")
    tid = str(team_id)
    digest = frozen_commands_digest(validated)
    with locked_team(ctx.omh_home, ctx.session_ref, tid) as path:
        record = read_team(path)
        now = ctx.now()
        if record is not None:
            if (record.get("session_ref") != ctx.session_ref
                    or record.get("commands_digest") != digest
                    or frozen_commands_digest(record["units"]) != digest
                    or record.get("plan_ref") != plan_ref
                    or [unit["depends_on"] for unit in record["units"]] != [unit["depends_on"] for unit in validated]):
                raise TeamRefusal(
                    "commands_frozen",
                    "This team already started with different parts or checks; they cannot change after start.",
                )
            _distrust_unobserved(record, ctx.observed_checks)
            # A resume in the process that already approved this team is a
            # repeat call: entries still inside their grace are in flight and
            # are not handed out again. The first approval after a restart
            # hands back every undispatched entry, since no helper survived it,
            # and every dispatched one whose Hermes process is gone.
            same_process = (ctx.session_ref, tid, digest) in ctx.approved
            lost = frozenset() if same_process else _hand_back_lost(ctx, record, now)
            entries = _reemit(record, now, respect_grace=same_process, lost_now=lost)
            write_team(path, record)
            again = sum(1 for unit in record["units"] if not unit.get("blocked")
                        and attempt_key(tid, str(unit["unit_id"]), len(unit["attempts"])) in lost)
            return _result(record, "team_start", entries=entries, say=_say(record, resumed=True, restarted=again),
                           commands_digest=digest)
        if sum(1 for _ in path.parent.glob("*.json")) >= MAX_TEAMS_PER_SESSION:
            raise TeamRefusal("too_many_teams", "This session already has as many teams as it can keep.")
        record = {
            "schema_version": TEAM_SCHEMA_VERSION,
            "team_id": tid,
            "session_ref": ctx.session_ref,
            "plan_ref": plan_ref,
            "workdir": str(Path(workdir).resolve()),
            "created_at": _iso(now),
            "max_repair_attempts": repairs,
            "commands_digest": digest,
            "check_timeout_seconds": TEAM_CHECK_TIMEOUT_SECONDS,
            "units": [{**unit, "attempts": [], "blocked": None, "lease": None} for unit in validated],
            "events": [],
            "next_seq": 1,
        }
        entries = []
        for unit in record["units"]:
            if not unit["depends_on"]:
                entries.append(_hand_out(tid, unit, _reserve(unit, now), now))
        write_team(path, record)
    started, total = len(entries), len(validated)
    if total == 1:
        say = "Started 1 part; it is done only when its check passes."
    elif started == total:
        say = f"Split into {total} parts; {total} start now in parallel, and each part is done only when its check passes."
    else:
        now_line = "1 starts now" if started == 1 else f"{started} start now in parallel"
        say = (f"Split into {total} parts; {now_line} and the rest wait for the parts they depend on, and each part "
               "is done only when its check passes.")
    return _result(record, "team_start", entries=entries, say=say, commands_digest=digest)


def _reemit(
    record: dict[str, Any], now: float, *, reserved_now: frozenset[str] = frozenset(), respect_grace: bool = True,
    lost_now: frozenset[str] = frozenset(),
) -> list[dict[str, str]]:
    """Hand back every reserved attempt the host has not reported dispatching (H5.1).

    ``reserved_now`` names the attempts this same call just reserved: they are
    handed out once as new entries, not counted as a repeat of themselves. An
    entry handed out less than EMIT_GRACE_SECONDS ago is in flight, not lost,
    and is skipped unless ``respect_grace`` is False (the first start after a
    restart, when no helper can still be on its way). ``lost_now`` names the
    attempts `_hand_back_lost` just took back. Those are counted by their
    ``lost`` list, not by ``reemits``, so a part re-emitted earlier for a slow
    start is not stopped by fewer restarts; one past the bound blocks as
    `helper_lost`, since its helper was seen starting.
    """
    entries = []
    for unit in record["units"]:
        attempt = _latest(unit)
        if unit.get("blocked") or attempt is None or attempt.get("dispatch") or attempt.get("returned_at"):
            continue
        if attempt.get("check"):
            continue
        key = attempt_key(str(record["team_id"]), str(unit["unit_id"]), int(attempt["n"]))
        if key in reserved_now:
            continue
        if respect_grace and _emit_pending(attempt, now):
            continue
        if key in lost_now:
            reason, over = "helper_lost", len(attempt.get("lost") or []) > MAX_REEMITS
        else:
            attempt["reemits"] = int(attempt.get("reemits", 0)) + 1
            reason, over = "dispatch_not_observed", attempt["reemits"] > MAX_REEMITS
        if over:
            _block(record, unit, now, {"reason": reason, "observed_at": _iso(now)})
            continue
        entries.append(_hand_out(str(record["team_id"]), unit, attempt, now))
    return entries


def _hand_back_lost(ctx: TeamContext, record: dict[str, Any], now: float) -> frozenset[str]:
    """Undispatch every out attempt whose Hermes process is gone, so `_reemit` hands it back; their keys.

    A restart ends every helper the old process ran, and Hermes fires no
    `subagent_stop` for them, so without this each would hold the barrier for
    DELEGATION_STALE_SECONDS. Only a recorded process ``dispatcher_alive``
    says is gone counts. No attempt is spent: the hand-back is a re-emit,
    bounded by MAX_REEMITS hand-backs of its own (the ``lost`` list). The
    dispatch moves to ``lost``, so a stop that names its child later matches
    nothing.
    """
    lost = set()
    for unit in record["units"]:
        attempt = _latest(unit)
        if unit.get("blocked") or attempt is None or attempt.get("returned_at") or attempt.get("check"):
            continue
        dispatch = attempt.get("dispatch")
        if not dispatch or dispatch.get("host") is None or ctx.dispatcher_alive(dispatch["host"]) is not False:
            continue
        key = attempt_key(str(record["team_id"]), str(unit["unit_id"]), int(attempt["n"]))
        attempt["dispatch"] = None
        attempt.pop("emitted_at", None)
        gone = attempt.setdefault("lost", [])
        gone.append({**dispatch, "lost_at": _iso(now)})
        # Its length is the hand-back count `_reemit` bounds; the unit blocks at MAX_REEMITS + 1.
        del gone[:-(MAX_REEMITS + 1)]
        _emit(record, now, unit=unit, event="lost", summary="Its helper ended when Hermes restarted.", detail_ref=key)
        lost.add(key)
    return frozenset(lost)


def _hand_out(team_id: str, unit: Mapping[str, Any], attempt: dict[str, Any], now: float) -> dict[str, str]:
    """The entry for a reserved attempt, stamped as handed out now (emitted, not yet started)."""
    attempt["emitted_at"] = _iso(now)
    return _delegate_entry(team_id, unit, attempt)


def _emit_pending(attempt: Mapping[str, Any], now: float) -> bool:
    """Handed out, not yet reported started, and still inside its grace."""
    if attempt.get("dispatch") or attempt.get("returned_at") or attempt.get("check"):
        return False
    emitted = _epoch(attempt.get("emitted_at"))
    return emitted is not None and now - emitted < EMIT_GRACE_SECONDS


# --------------------------------------------------------------------------
# team_reconcile
# --------------------------------------------------------------------------


def _load(ctx: TeamContext, team_id: object) -> tuple[Path, dict[str, Any]]:
    path = team_path(ctx.omh_home, ctx.session_ref, str(team_id))
    record = read_team(path)
    if record is None or record.get("session_ref") != ctx.session_ref:
        raise TeamRefusal("team_not_found", "No team with that name was started in this session.")
    if record.get("commands_digest") != frozen_commands_digest(record["units"]):
        raise TeamRefusal("frozen_commands_changed", "The team's check commands were changed after start, so nothing runs.")
    _distrust_unobserved(record, ctx.observed_checks)
    return path, record


def _require_approved(ctx: TeamContext, record: Mapping[str, Any]) -> None:
    """Nothing runs for a team whose exact commands this process did not see approved.

    The record's digest is an unkeyed hash anyone who can write the file can
    recompute, so it proves the commands were not edited, never that a person
    approved them. The approval lives only in ``ctx.approved``.
    """
    if (ctx.session_ref, str(record["team_id"]), str(record["commands_digest"])) not in ctx.approved:
        raise TeamRefusal(
            "team_not_approved_here",
            "This team's check commands were not approved since Hermes started; call team_start again with "
            "the same parts so the person is asked again.",
        )


def _distrust_unobserved(record: dict[str, Any], observed: set[str]) -> None:
    """A check this process did not run is not a result: its unit is checked again.

    Applies to the latest attempt of every unit that is not blocked. The check
    is re-run rather than trusted, so a `passed` planted on disk (or left from
    before a restart) accepts nothing until OMH has run the command itself.
    """
    for unit in record["units"]:
        attempt = _latest(unit)
        if unit.get("blocked") or attempt is None:
            continue
        check = attempt.get("check")
        if isinstance(check, Mapping) and check.get("nonce") not in observed:
            attempt["check"] = None


def _in_flight(record: dict[str, Any], now: float, *, include_emitted: bool = True) -> list[str]:
    """Units with a helper out or about to start, after expiring ones the host never reported back.

    ``include_emitted=False`` is for a check inside the same reconcile that
    handed entries out: those have not reached the model yet, so nothing can
    have started them.
    """
    flying = []
    for unit in record["units"]:
        attempt = _latest(unit)
        if unit.get("blocked") or attempt is None:
            continue
        if _emit_pending(attempt, now) and include_emitted:
            flying.append(str(unit["unit_id"]))
            continue
        if not attempt.get("dispatch") or attempt.get("returned_at"):
            continue
        started = _epoch(attempt["dispatch"].get("dispatched_at"))
        if started is not None and now - started > DELEGATION_STALE_SECONDS:
            # A return OMH saw and could not link may have been this helper's,
            # so the stop says that rather than "never came back".
            reason = "return_not_linked" if _unlinked_since(record, started) else "delegation_not_returned"
            _block(record, unit, now, {"reason": reason, "observed_at": _iso(now)})
            continue
        flying.append(str(unit["unit_id"]))
    return flying


def _unlinked_since(record: Mapping[str, Any], started: float) -> list[dict[str, Any]]:
    """Returns OMH saw but could not link to any part, at or after ``started``."""
    return [dict(note) for note in record.get("unlinked_returns") or []
            if (_epoch(note.get("observed_at")) or 0.0) >= started]


def _out_attempts(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The latest attempt of every unit whose helper was seen starting and not yet coming back."""
    out = []
    for unit in record["units"]:
        attempt = _latest(unit)
        if not unit.get("blocked") and attempt is not None and attempt.get("dispatch") and not attempt.get("returned_at"):
            out.append(attempt)
    return out


def _unlinked_while_out(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Unlinked returns since the oldest helper still out was started; empty when none is out."""
    starts = []
    for attempt in _out_attempts(record):
        started = _epoch(attempt["dispatch"].get("dispatched_at"))
        if started is not None:
            starts.append(started)
    return _unlinked_since(record, min(starts)) if starts else []


def _lease_live(unit: Mapping[str, Any], now: float) -> bool:
    lease = unit.get("lease")
    if not isinstance(lease, Mapping):
        return False
    started = _epoch(lease.get("started_at"))
    return started is not None and now - started < TEAM_CHECK_TIMEOUT_SECONDS + TEAM_LEASE_MARGIN_SECONDS


def _unit(record: Mapping[str, Any], unit_id: str) -> dict[str, Any]:
    return next(item for item in record["units"] if str(item["unit_id"]) == unit_id)


def team_reconcile(ctx: TeamContext, *, team_id: object, since_seq: object = 0) -> dict[str, Any]:
    tid = str(team_id)
    # Phase A, under the lock: approval, barrier, and which units need a check.
    with locked_team(ctx.omh_home, ctx.session_ref, tid):
        path, record = _load(ctx, tid)
        _require_approved(ctx, record)
        now = ctx.now()
        flying = _in_flight(record, now)
        if flying:
            write_team(path, record)
            say = f"{len(flying)} part(s) are still being worked on; nothing is checked until they come back."
            unlinked = _unlinked_while_out(record)
            if unlinked:
                why = _UNLINKED_SAY.get(unlinked[-1]["reason"], _UNLINKED_SAY["lineage_unverified"])
                say = (f"{len(flying)} part(s) have not come back in a way OMH can confirm, so nothing is checked "
                       f"yet. A helper came back under a session OMH could not link to any part, because {why}.")
            return _result(record, "team_reconcile", entries=[], reason="delegations_in_flight",
                           say=say, since_seq=since_seq)
        waiting = [str(unit["unit_id"]) for unit in record["units"] if unit_state(unit) == "awaiting_check"]
        before = {str(unit["unit_id"]): len(unit.get("attempts") or []) for unit in record["units"]}
        helpers = _helper_branches(record, waiting)
        write_team(path, record)
    # Phase A2, outside the lock (git may be slow): nothing is checked while a
    # returned helper's work sits on a branch this workspace has not merged.
    unreadable = _require_helper_work_merged(ctx, Path(str(record["workdir"])), helpers)
    # Phase B, one unit at a time: lease it just before its own check, run the
    # check outside the lock, apply the result only while the lease is ours.
    entries: list[dict[str, str]] = []
    events: list[str] = []
    busy: list[str] = []
    checked = 0
    pending = 0
    for unit_id in waiting:
        if checked >= MAX_CHECKS_PER_RECONCILE:
            pending += 1
            continue
        leased = _take_lease(ctx, tid, unit_id)
        if leased is None:
            busy.append(unit_id)
            continue
        nonce, command, workdir = leased
        result = _run_check(ctx, command, workdir, helper_unreadable=unit_id in unreadable)
        checked += 1
        with locked_team(ctx.omh_home, ctx.session_ref, tid):
            path, record = _load(ctx, tid)
            unit = _unit(record, unit_id)
            lease = unit.get("lease")
            if isinstance(lease, Mapping) and lease.get("nonce") == nonce:
                unit["lease"] = None
                attempt = _latest(unit)
                if attempt is not None and attempt["n"] == lease.get("attempt") and not attempt.get("check"):
                    events.extend(_apply_check(ctx, record, unit, attempt, result, nonce, ctx.now(), entries))
                write_team(path, record)
    # Phase C, under the lock: release ready units, hand back lost entries, stop rule.
    with locked_team(ctx.omh_home, ctx.session_ref, tid):
        path, record = _load(ctx, tid)
        now = ctx.now()
        # Every parent accepted -- ALL of them, and accepted by a check, not by a report.
        accepted = {str(unit["unit_id"]) for unit in record["units"] if unit_state(unit) == "accepted"}
        waiting = {str(unit["unit_id"]): unit for unit in record["units"]
                   if unit_state(unit) == "waiting" and not unit.get("blocked")}
        for unit_id in dag_core.ready(waiting, {key: unit.get("depends_on", []) for key, unit in waiting.items()},
                                      accepted):
            unit = waiting[unit_id]
            entries.append(_hand_out(tid, unit, _reserve(unit, now), now))
        reserved_now = frozenset(
            attempt_key(tid, str(unit["unit_id"]), int(attempt["n"]))
            for unit in record["units"]
            for attempt in (unit.get("attempts") or [])[before.get(str(unit["unit_id"]), 0):]
        )
        entries.extend(_reemit(record, now, reserved_now=reserved_now))
        if team_state(record) == "done" and not any(item["event"] == "done" for item in record.get("events", [])):
            _emit(record, now, unit=None, event="done", summary=_all_passed(len(record["units"])), detail_ref=tid)
        write_team(path, record)
    reason = ""
    if pending:
        reason = "checks_pending"
        events.append(f"{pending} more part(s) wait for their check; call team_reconcile again.")
    elif busy and not checked:
        reason = "check_in_progress"
    return _result(record, "team_reconcile", entries=entries, reason=reason, say=_say(record, events=events),
                   since_seq=since_seq)


def helper_branch(subagent_id: str) -> str:
    """The branch Hermes' worktree isolation gives the helper with this subagent id."""
    return HELPER_BRANCH_PREFIX + subagent_id


def _helper_branches(record: Mapping[str, Any], unit_ids: list[str]) -> list[tuple[str, str, str]]:
    """``(unit_id, title, branch)`` for each listed unit whose latest helper reported a subagent id."""
    found = []
    for unit_id in unit_ids:
        unit = _unit(record, unit_id)
        attempt = _latest(unit)
        dispatch = attempt.get("dispatch") if attempt else None
        subagent_id = dispatch.get("subagent_id") if isinstance(dispatch, Mapping) else None
        if isinstance(subagent_id, str) and _SUBAGENT_ID.fullmatch(subagent_id):
            found.append((unit_id, str(unit["title"]), helper_branch(subagent_id)))
    return found


def _require_helper_work_merged(
    ctx: TeamContext, workdir: Path, helpers: list[tuple[str, str, str]],
) -> frozenset[str]:
    """Refuse while a helper's branch holds work this workspace lacks; return the unreadable ones.

    A branch git could not read is not refused here: its unit's check becomes
    inconclusive instead, so three such reads still end the unit as blocked
    rather than refusing every call forever.
    """
    if not helpers or not workdir.is_dir():
        # A missing workspace is the check's own `workspace_missing`.
        return frozenset()
    unmerged: list[dict[str, str]] = []
    unreadable: set[str] = set()
    for unit_id, title, branch in helpers:
        state = ctx.helper_work(workdir, branch)
        if state in _HELPER_WORK_UNMERGED:
            unmerged.append({"unit_id": unit_id, "title": title, "branch": branch, "state": state})
        elif state not in ("absent", "merged"):
            unreadable.add(unit_id)
    if unmerged:
        lines = [_UNMERGED_SAY[item["state"]].format(title=item["title"], branch=item["branch"]) for item in unmerged]
        raise TeamRefusal(
            "helper_branch_unmerged",
            " ".join(lines) + " Nothing was checked: merge each into this workspace, then call team_reconcile "
            "again. If one cannot be merged, report that part as blocked and stop.",
            unmerged=tuple({key: item[key] for key in ("unit_id", "branch", "state")} for item in unmerged),
        )
    return frozenset(unreadable)


_UNMERGED_SAY: Final[dict[str, str]] = {
    "commits_not_merged": "'{title}' left commits on branch {branch}, or in its worktree, that this workspace "
                          "does not have yet.",
    "uncommitted_changes": "'{title}' left uncommitted changes or untracked files in the worktree of branch "
                           "{branch}: commit what belongs to the part on that branch, and remove what does not, "
                           "or the whole worktree once its work is merged.",
}


def _take_lease(ctx: TeamContext, team_id: str, unit_id: str) -> tuple[str, str, Path] | None:
    """Stamp one unit's lease just before its check; None when it no longer needs one."""
    with locked_team(ctx.omh_home, ctx.session_ref, team_id):
        path, record = _load(ctx, team_id)
        _require_approved(ctx, record)
        now = ctx.now()
        if _in_flight(record, now, include_emitted=False):
            write_team(path, record)
            return None
        unit = _unit(record, unit_id)
        if unit_state(unit) != "awaiting_check" or _lease_live(unit, now):
            return None
        nonce = secrets.token_hex(8)
        unit["lease"] = {"attempt": _latest(unit)["n"], "nonce": nonce, "started_at": _iso(now)}
        write_team(path, record)
        return nonce, str(unit["verification_command"]), Path(str(record["workdir"]))


def _run_check(ctx: TeamContext, command: str, workdir: Path, *, helper_unreadable: bool = False) -> dict[str, Any]:
    started = ctx.now()
    try:
        tokens = validate_team_command(command)
        _require_host_allows(ctx, command)
    except TeamRefusal as refusal:
        return {"outcome": "refused", "reason": refusal.reason, "observed_at": _iso(started)}
    if not workdir.is_dir():
        return {"outcome": "inconclusive", "reason": "workspace_missing", "observed_at": _iso(started)}
    before = ctx.fingerprint(workdir)
    if before[0] in _FINGERPRINT_UNTRACKED:
        return {"outcome": "inconclusive", "reason": "workspace_not_tracked", "observed_at": _iso(started)}
    if before[0] in _FINGERPRINT_UNAVAILABLE:
        return {"outcome": "inconclusive", "reason": "workspace_fingerprint_unavailable", "observed_at": _iso(started)}
    if helper_unreadable:
        return {"outcome": "inconclusive", "reason": "helper_branch_unreadable", "observed_at": _iso(started)}
    run = ctx.runner(tokens, workdir, TEAM_CHECK_TIMEOUT_SECONDS)
    after = ctx.fingerprint(workdir)
    observed_at = _iso(ctx.now())
    if after[0] in _FINGERPRINT_UNAVAILABLE | _FINGERPRINT_UNTRACKED:
        return {"outcome": "inconclusive", "reason": "workspace_fingerprint_unavailable", "observed_at": observed_at}
    if before != after:
        return {"outcome": "inconclusive", "reason": "workspace_changed_during_check", "observed_at": observed_at}
    if run.outcome == "timeout":
        return {"outcome": "inconclusive", "reason": "check_timeout", "observed_at": observed_at}
    if run.outcome == "not_found" or run.exit_code is None:
        return {"outcome": "inconclusive", "reason": "command_not_found", "observed_at": observed_at}
    if run.exit_code < 0:
        # Killed by a signal (an OOM kill, a stray SIGTERM): not the check's answer.
        return {"outcome": "inconclusive", "reason": "check_killed", "observed_at": observed_at}
    return {
        "outcome": "passed" if run.exit_code == 0 else "failed",
        "command": command,
        "exit_code": run.exit_code,
        "observed_at": observed_at,
        "output_tail_digest": hashlib.sha256(run.output_tail.encode("utf-8", "replace")).hexdigest(),
        "workdir": str(workdir),
    }


def _apply_check(
    ctx: TeamContext,
    record: dict[str, Any],
    unit: dict[str, Any],
    attempt: dict[str, Any],
    result: Mapping[str, Any],
    nonce: str,
    now: float,
    entries: list[dict[str, str]],
) -> list[str]:
    title = str(unit["title"])
    if result["outcome"] == "refused":
        _block(record, unit, now, {"reason": result["reason"], "observed_at": result["observed_at"]})
        return [f"Stopped '{title}': its check command was refused before it ran."]
    if result["outcome"] == "inconclusive":
        attempt.setdefault("inconclusive", []).append(
            {"reason": result["reason"], "observed_at": result["observed_at"]})
        if len(attempt["inconclusive"]) >= MAX_INCONCLUSIVE_PER_ATTEMPT:
            _block(record, unit, now, {"reason": "check_inconclusive", "last_reason": result["reason"],
                                       "observed_at": result["observed_at"]})
            return [f"Stopped '{title}': its check could not give a clear answer ({_REASON_SAY[result['reason']]})."]
        return [f"The check for '{title}' could not give a clear answer ({_REASON_SAY[result['reason']]}); "
                "no try was used, and it will run again."]
    attempt["check"] = {**result, "nonce": nonce}
    ctx.observed_checks.add(nonce)
    check_ref = f"{attempt_key(str(record['team_id']), str(unit['unit_id']), int(attempt['n']))}/check"
    if result["outcome"] == "passed":
        _emit(record, now, unit=unit, event="check_passed", summary="Its check passed (exit code 0).",
              detail_ref=check_ref)
        return [f"'{title}' passed its check."]
    _emit(record, now, unit=unit, event="check_failed",
          summary=f"Its check failed with exit code {result['exit_code']}.", detail_ref=check_ref)
    if int(attempt["n"]) < _attempt_limit(record):
        repair = _reserve(unit, now, repairs_check=result)
        entries.append(_hand_out(str(record["team_id"]), unit, repair, now))
        _emit(record, now, unit=unit, event="repairing",
              summary=f"Sent back for a fix, try {repair['n']} of {_attempt_limit(record)}.",
              detail_ref=attempt_key(str(record["team_id"]), str(unit["unit_id"]), int(repair["n"])))
        return [f"The check for '{title}' failed (exit code {result['exit_code']}); sending it back for a fix, "
                f"try {repair['n']} of {_attempt_limit(record)}."]
    _block(record, unit, now, {
        "reason": "repair_budget_exhausted",
        "last_check": {key: result[key] for key in ("command", "exit_code", "observed_at")},
    })
    return [f"Stopped '{title}': it still fails its check after {attempt['n']} tries "
            f"(exit code {result['exit_code']})."]


_REASON_SAY: Final[dict[str, str]] = {
    "workspace_missing": "the workspace folder is gone",
    "workspace_not_tracked": "the workspace is not a git checkout, so OMH cannot tell whether files changed",
    "workspace_fingerprint_unavailable": "the workspace could not be read before and after",
    "workspace_changed_during_check": "files changed while it ran",
    "check_timeout": f"it ran past {TEAM_CHECK_TIMEOUT_SECONDS // 60} minutes",
    "check_killed": "it was stopped by a signal before it finished",
    "command_not_found": "the command was not found",
    "helper_branch_unreadable": "git could not say whether its helper's branch is merged into this workspace",
}

_BLOCKED_SAY: Final[dict[str, str]] = {
    "repair_budget_exhausted": "it still fails its check after every allowed fix",
    "check_inconclusive": "its check never gave a clear answer",
    "dispatch_not_observed": "its helper was never seen starting",
    "helper_lost": "Hermes kept restarting while its helper was out",
    "delegation_not_returned": "its helper never came back",
    "return_not_linked": "a helper came back that OMH could not link to it",
    "command_refused_by_host": "Hermes' own command rules refuse its check",
    "host_command_floor_unavailable": "Hermes' own command rules could not be reached",
}


# --------------------------------------------------------------------------
# Host lifecycle observations
# --------------------------------------------------------------------------


def owning_session(omh_home: Path, session_ref: str, team_id: str, earlier: Sequence[str] = ()) -> str:
    """The session whose record holds ``team_id``: ``session_ref``, else the nearest of ``earlier``, else "".

    ``earlier`` is the earlier segments of this conversation that Hermes'
    compaction links say it continues (`live_session.compaction_ancestors`).
    It is how a parent Hermes compacted into a new session id after
    `team_start` reaches the team it started. Nothing else is searched: not
    another session, not the newest team.
    """
    for ref in (session_ref, *earlier):
        path = team_path(omh_home, ref, team_id)
        if path.is_file() and not path.is_symlink():
            return ref
    return ""


def lineage_refusal(reason: str) -> TeamRefusal:
    """No team under this session, and its earlier segments could not be read either."""
    why = _UNLINKED_SAY.get(reason, _UNLINKED_SAY["lineage_unverified"])
    return TeamRefusal(
        "team_lineage_unverified",
        "No team with that name was started in this session, and OMH could not check whether this session "
        f"continues an earlier one Hermes compacted, because {why}.",
    )


def observe_dispatch(
    omh_home: Path, session_ref: str, *, child_session_id: object, goal: object, now: float,
    subagent_id: object = None, earlier: Callable[[], Sequence[str]] | None = None, host: object = None,
) -> bool:
    """`subagent_start` saw a goal carrying a team marker: bind that attempt to the child.

    ``subagent_id`` is the host's id for the child, kept so `team_reconcile`
    can find the branch a worktree-isolated helper left its work on.
    ``earlier`` is read only when this session holds no record of the team
    (see `owning_session`). ``host`` is the Hermes process the helper runs in
    (`host_process.this_process`), kept so a restart that ended it is seen.
    """
    found = _MARKER.findall(goal) if isinstance(goal, str) else []
    if not found or not found[-1][0] or not isinstance(child_session_id, str) or not child_session_id.strip():
        return False
    team_id, unit_id, number = found[-1]
    n = int(number)
    owner = owning_session(omh_home, session_ref, team_id)
    if not owner and earlier is not None:
        owner = owning_session(omh_home, session_ref, team_id, earlier())
    if not owner:
        return False
    with locked_team(omh_home, owner, team_id) as path:
        record = read_team(path)
        if record is None or record.get("session_ref") != owner:
            return False
        unit = next((item for item in record["units"] if item["unit_id"] == unit_id), None)
        attempt = _latest(unit) if unit is not None else None
        if attempt is None or attempt["n"] != n or attempt.get("dispatch") or attempt.get("returned_at"):
            return False
        attempt["dispatch"] = {"child_session_id": child_session_id.strip()[:160], "dispatched_at": _iso(now)}
        if isinstance(subagent_id, str) and _SUBAGENT_ID.fullmatch(subagent_id):
            attempt["dispatch"]["subagent_id"] = subagent_id
        if (process := recorded_process(host)) is not None:
            attempt["dispatch"]["host"] = process
        _emit(record, now, unit=unit, event="started",
              summary="Started a fix for this part." if n > 1 else "Started working on this part.",
              detail_ref=attempt_key(team_id, unit_id, n))
        write_team(path, record)
    return True


def observe_return(
    omh_home: Path, session_ref: str, *, child_session_id: object, now: float,
    lineage: Callable[[str], Mapping[str, Any]] | None = None,
    earlier: Callable[[], Sequence[str]] | None = None,
) -> bool:
    """`subagent_stop` for a child bound above: its attempt is back and may be checked.

    The id `subagent_start` recorded is matched first, in this session's
    teams and then, through ``earlier`` (see `owning_session`), in the teams of
    the segments this session continues. Failing that, ``lineage(child)``
    names the earlier segments Hermes' compaction links say this child
    continues (``{"status", "ancestors", "reason"}``), and an attempt that
    recorded one of them is the one back. A lineage that could not be read
    leaves every attempt out and is kept as an unlinked return.
    """
    if not isinstance(child_session_id, str) or not child_session_id.strip():
        return False
    child = child_session_id.strip()[:160]
    teams = _session_teams(omh_home, session_ref)
    if _mark_returned(omh_home, teams, child, {child}, now):
        return True
    if earlier is not None:
        more = [pair for ref in earlier() for pair in _session_teams(omh_home, ref)]
        if _mark_returned(omh_home, more, child, {child}, now):
            return True
        teams = teams + more
    if not teams or lineage is None:
        return False
    found = lineage(child)
    ancestors = {item[:160] for item in found.get("ancestors") or [] if isinstance(item, str) and item}
    if ancestors and _mark_returned(omh_home, teams, child, ancestors, now):
        return True
    if found.get("status") != "resolved":
        reason = str(found.get("reason") or "")
        _note_unlinked_return(omh_home, teams, child, reason if reason in _UNLINKED_SAY else "lineage_unverified", now)
    return False


def _session_teams(omh_home: Path, session_ref: str) -> list[tuple[str, str]]:
    """``(session, team_id)`` for every team record under one session."""
    directory = teams_dir(omh_home, session_ref)
    if not directory.is_dir() or directory.is_symlink():
        return []
    return [(session_ref, candidate.stem) for candidate in sorted(directory.glob("*.json"))[:MAX_TEAMS_PER_SESSION]]


def _mark_returned(
    omh_home: Path, teams: list[tuple[str, str]], child: str, started_as: set[str], now: float,
) -> bool:
    """Mark returned the out attempt whose recorded child id is in ``started_as``."""
    for session_ref, team_id in teams:
        with locked_team(omh_home, session_ref, team_id) as path:
            record = read_team(path)
            if record is None or record.get("session_ref") != session_ref:
                continue
            for unit in record["units"]:
                attempt = _latest(unit)
                dispatch = attempt.get("dispatch") if attempt else None
                if dispatch and dispatch.get("child_session_id") in started_as and not attempt.get("returned_at"):
                    attempt["returned_at"] = _iso(now)
                    summary = "Came back; its check runs next."
                    if dispatch["child_session_id"] != child:
                        # The later segment Hermes' compaction moved the helper to.
                        attempt["returned_as"] = child
                        summary = "Came back after Hermes compacted its session; its check runs next."
                    _emit(record, now, unit=unit, event="finished", summary=summary,
                          detail_ref=attempt_key(str(record["team_id"]), str(unit["unit_id"]), int(attempt["n"])))
                    write_team(path, record)
                    return True
    return False


def _note_unlinked_return(
    omh_home: Path, teams: list[tuple[str, str]], child: str, reason: str, now: float,
) -> None:
    """Keep a return OMH could not link, in every record that has a helper out."""
    for session_ref, team_id in teams:
        with locked_team(omh_home, session_ref, team_id) as path:
            record = read_team(path)
            if record is None or record.get("session_ref") != session_ref:
                continue
            if not _out_attempts(record):
                continue
            notes = record.setdefault("unlinked_returns", [])
            notes.append({"child_session_id": child, "observed_at": _iso(now), "reason": reason})
            del notes[:-MAX_UNLINKED_RETURNS]
            write_team(path, record)


# Why a return could not be linked, as a person reads it. The keys are the
# reasons `live_session.compaction_ancestors` gives, plus `lineage_unverified`
# for a reason this table does not know.
_UNLINKED_SAY: Final[dict[str, str]] = {
    "state_db_unavailable": "Hermes' session records could not be read",
    "session_row_missing": "Hermes has no record of the session",
    "segment_row_missing": "a session it continues is missing from Hermes' records",
    "continuation_not_authenticated": "its session history does not match how Hermes records a compaction",
    "tool_session_ambiguous": "Hermes records it as a tool session, which OMH cannot tell apart from a compacted one",
    "lineage_too_long": "its session history was too long to follow",
    "lineage_unverified": "its session history could not be checked",
}


# --------------------------------------------------------------------------
# team_status and results
# --------------------------------------------------------------------------


def team_status(
    ctx: TeamContext, *, team_id: object, cost: Mapping[str, Any] | None = None, since_seq: object = 0,
) -> dict[str, Any]:
    _, record = _load(ctx, team_id)
    result = _result(record, "team_status", entries=[], say=_say(record), since_seq=since_seq)
    result["cost"] = dict(cost) if cost is not None else {"status": "not_observed"}
    result["caveat"] = TEAM_CAVEAT
    result["check_timeout_seconds"] = TEAM_CHECK_TIMEOUT_SECONDS
    return result


def _unit_view(record: Mapping[str, Any], unit: Mapping[str, Any]) -> dict[str, Any]:
    attempts = list(unit.get("attempts") or [])
    last_check = next((attempt["check"] for attempt in reversed(attempts) if attempt.get("check")), None)
    view: dict[str, Any] = {
        "unit_id": unit["unit_id"],
        "title": unit["title"],
        "state": unit_state(unit),
        "attempts_used": len(attempts),
        "attempts_max": _attempt_limit(record),
        "verification_command": unit["verification_command"],
        "last_check": ({key: last_check[key] for key in ("command", "exit_code", "observed_at", "outcome")}
                       if last_check else None),
    }
    if unit.get("blocked"):
        view["blocked"] = dict(unit["blocked"])
    if view["state"] == "accepted":
        view["evidence"] = evidence_ref(str(record["team_id"]), str(unit["unit_id"]), len(attempts))
    return view


def _result(
    record: Mapping[str, Any],
    action: str,
    *,
    entries: list[dict[str, str]],
    say: str,
    reason: str = "",
    since_seq: object = 0,
    commands_digest: str = "",
) -> dict[str, Any]:
    units = [_unit_view(record, unit) for unit in record["units"]]
    counts = {state: 0 for state in UNIT_STATES}
    for view in units:
        counts[view["state"]] += 1
    result: dict[str, Any] = {
        "schema_version": TEAM_RESULT_SCHEMA_VERSION,
        "action": action,
        "status": "ok",
        "team_id": record["team_id"],
        "team_state": team_state(record),
        "counts": counts,
        "units": units,
        "say": say,
        "delegate_task": {"tool_name": "delegate_task", "arguments": {"tasks": entries}} if entries else None,
    }
    result.update(_events_since(record, since_seq))
    unlinked = _unlinked_while_out(record)
    if unlinked:
        # Returns OMH saw while a part was out and could not link to any part.
        # The parts stay out; this is what names why.
        result["unlinked_returns"] = unlinked
    if reason:
        result["reason"] = reason
    if commands_digest:
        # What the handler records as approved in process memory (`TeamContext.approved`).
        result["commands_digest"] = commands_digest
    return result


def _events_since(record: Mapping[str, Any], since_seq: object) -> dict[str, Any]:
    """Events newer than the caller's last seen `seq`, and whether the cap dropped some."""
    seen = since_seq if isinstance(since_seq, int) and not isinstance(since_seq, bool) and since_seq > 0 else 0
    events = [dict(item) for item in record.get("events") or [] if int(item["seq"]) > seen]
    oldest = int(events[0]["seq"]) if events else int(record.get("next_seq", 1))
    return {
        "events": events,
        "last_seq": int(record.get("next_seq", 1)) - 1,
        "events_truncated": oldest > seen + 1,
    }


def _say(record: Mapping[str, Any], *, events: list[str] | None = None, resumed: bool = False,
         restarted: int = 0) -> str:
    state = team_state(record)
    units = list(record["units"])
    if state == "done":
        return _all_passed(len(units))
    if events:
        return " ".join(events)
    if state == "blocked":
        blocked = [unit for unit in units if unit.get("blocked")]
        first = blocked[0]
        why = _BLOCKED_SAY.get(str(first["blocked"].get("reason")), "it cannot continue")
        return f"Stopped: '{first['title']}' is blocked because {why}."
    accepted = sum(1 for unit in units if unit_state(unit) == "accepted")
    passed = (f"{accepted} of 1 part has passed its check" if len(units) == 1
              else f"{accepted} of {len(units)} parts have passed their checks")
    if restarted:
        out = "1 part was out; it starts" if restarted == 1 else f"{restarted} parts were out; they start"
        return f"Hermes restarted while {out} again. {passed} so far."
    if resumed:
        return f"This team is already running: {passed}."
    return f"{passed} so far."


def _all_passed(count: int) -> str:
    return "The one part passed its check." if count == 1 else f"All {count} parts passed their checks."


def refusal_result(action: str, refusal: TeamRefusal) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": TEAM_RESULT_SCHEMA_VERSION,
        "action": action,
        "status": "refused",
        "reason": refusal.reason,
        "say": refusal.say,
        "delegate_task": None,
    }
    if refusal.unmerged:
        result["unmerged"] = [dict(item) for item in refusal.unmerged]
    return result
