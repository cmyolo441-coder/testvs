"""Whether a plan item marked done is closed by a recorded fact.

A done mark on the plan todo is a declaration: it says done was claimed, not
that anything ran. The continuation rule stops a plan when every item is done,
so a run that marked an item done over a failed command, or ticked several
items off one command, stopped its own loop without the work behind it. This
module is what the stop criterion reads instead of the word.

An item's evidence is a typed reference, ``{"kind": ..., "ref": ...}``, and
only two kinds resolve today, both against the result Hermes itself persisted
to ``<hermes_home>/state.db`` for one tool call id:

* ``tool_call``: a ``terminal`` call whose recorded ``exit_code`` is the
  integer 0 -- the field `remote_wait_nudge` reads;
* ``file_write``: a ``write_file`` result carrying ``bytes_written``, or a
  ``patch`` result whose ``success`` is true, neither with an ``error`` --
  #1922's `session_file_activity` outcome rules.

``effect_disposition`` ``none`` (the call had no effect) is a failure and
``unknown`` an unknown, the host's own two dispositions. A row Hermes rewound
out of the transcript (``active = 0`` and not ``compacted``) is not a record
of anything the session still stands on, so no query here reads one.

``pr``, ``ci_run`` and ``team_check`` are accepted and stored so every lane
writes one vocabulary, and none of them resolves yet: OMH makes no network
call, so it holds no record of a PR or a CI run, and the teammate lane that
will own ``team_check`` (``<team_id>/<unit_id>/attempt-<n>/check``) has not
landed. An unresolved reference keeps the item ``done_unverified``; a PR or
a CI run closes an item today through the ``tool_call`` that observed it --
``gh pr create``, ``gh run watch --exit-status``.

Who names the call. Models do not reliably see tool call ids -- no assistant
reply in the owner's session store quoted a ``toolu_`` id (measured
2026-09-28) -- and a reference a writer could send would be a reference it
could copy, so the ``omh_todo`` tool binds it from the records at the moment
of the done write and ignores any it is sent. Each item has its own WINDOW,
opened at its own last transition -- declared, pending to active, or out of
done -- and closed by the done write; it takes the latest evidence-capable
call recorded inside it that no other item holds (`observed_calls`).

What a plan write cannot hide (`_item_verdict`):

* a failed binding is sticky: it survives a reopen until a later call bound
  in its place replaces it;
* a binding a `set` drops -- a rename, a split, a removal -- is kept on the
  plan, and while it is a failure no item opened at or after the drop closes
  until one closes on a passing command of its own;
* an item is failed while the latest call before its done mark that no other
  item holds failed, so a failure nothing was bound to is not hidden by a
  rename or by passes that answer for other items;
* a file write does not close an item over a failing command in its window.

What it cannot prevent, because association is by time and not by content:
``clear`` removes the plan with its dropped bindings, so a failure bound
before a ``clear`` is hidden once any unheld passing call follows it; with
two items active at once, a call can be bound to the one that did not run it;
and an unheld passing call after a failure nothing was bound to answers for
it. It proves a command ran and how it ended, never that the command tested
the item.

What is never read: the item's text, the command's text, its output, or the
model's reply. A verdict comes from a result field or it does not come.

Nothing here raises into a host. A store that exists and cannot be read is a
reading of its own -- ``unreadable`` -- so a caller can say so instead of
treating it as evidence, and a missing store is ``absent``: a host with no
session store has recorded no commands, which is the conversational lane the
stop criterion leaves alone.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

EVIDENCE_KIND_TOOL_CALL: Final = "tool_call"
EVIDENCE_KIND_FILE_WRITE: Final = "file_write"
EVIDENCE_KIND_PR: Final = "pr"
EVIDENCE_KIND_CI_RUN: Final = "ci_run"
EVIDENCE_KIND_TEAM_CHECK: Final = "team_check"
EVIDENCE_KINDS: Final = (
    EVIDENCE_KIND_TOOL_CALL,
    EVIDENCE_KIND_FILE_WRITE,
    EVIDENCE_KIND_PR,
    EVIDENCE_KIND_CI_RUN,
    EVIDENCE_KIND_TEAM_CHECK,
)
# The host tools whose recorded result can close an item, by the kind that
# names them. Copied from Hermes rather than imported: the bundle may only
# import from inside itself.
_KIND_TOOLS: Final = {
    EVIDENCE_KIND_TOOL_CALL: ("terminal",),
    EVIDENCE_KIND_FILE_WRITE: ("write_file", "patch"),
}
EVIDENCE_TOOLS: Final = ("terminal", "write_file", "patch")
MAX_EVIDENCE_REF_CHARS: Final = 128
# Every ref is one token a reader can show without escaping, bounded before it
# is stored or queried. A call id (`toolu_...`, `call_...`) has no slash; a PR
# or run reference may (`owner/repo#12`); a team check has exactly its shape.
_CALL_ID: Final = re.compile(r"[A-Za-z0-9_.:-]{1,%d}" % MAX_EVIDENCE_REF_CHARS)
_REMOTE_REF: Final = re.compile(r"[A-Za-z0-9_.:/#-]{1,%d}" % MAX_EVIDENCE_REF_CHARS)
_TEAM_CHECK: Final = re.compile(r"[A-Za-z0-9_.:-]+/[A-Za-z0-9_.:-]+/attempt-[0-9]+/check")
_REF_SHAPES: Final = {
    EVIDENCE_KIND_TOOL_CALL: _CALL_ID,
    EVIDENCE_KIND_FILE_WRITE: _CALL_ID,
    EVIDENCE_KIND_PR: _REMOTE_REF,
    EVIDENCE_KIND_CI_RUN: _REMOTE_REF,
    EVIDENCE_KIND_TEAM_CHECK: _TEAM_CHECK,
}
# A compaction continues a session under a new id whose `parent_session_id`
# is the old one, and the calls recorded before it stay under the old id. The
# walk is bounded so a cycle in a hand-edited store costs a fixed number of
# queries.
MAX_LINEAGE_HOPS: Final = 8
# The most recent rows one done write considers. A window is the time since an
# item last opened, so this is far past any plan's work between two writes; it
# bounds the read on a session with a very long history.
MAX_OBSERVED_ROWS: Final = 500
# How many of the latest calls a verdict may skip because another item
# holds them: one per other item in the largest plan (`MAX_TODO_ITEMS`).
MAX_BOUND_SKIP: Final = 20
_CONNECT_TIMEOUT_SECONDS: Final = 0.5

# Store readings.
STORE_READ: Final = "read"
STORE_ABSENT: Final = "absent"
STORE_UNREADABLE: Final = "unreadable"

# Per-item verdicts. Only `closed` and `window_empty` close an item.
EVIDENCE_CLOSED: Final = "closed"
EVIDENCE_FAILED: Final = "failed"
EVIDENCE_UNRESOLVED: Final = "unresolved"
# No reference, and no evidence-capable call recorded inside the item's own
# window: nothing a command could have closed, so the done mark stands. This
# is what keeps a conversational item after a command-backed one from being
# held open for good.
WINDOW_EMPTY: Final = "window_empty"
# No reference, and calls WERE recorded inside the window -- each one already
# holding another item. One command closes at most one item.
WINDOW_HAS_COMMANDS: Final = "window_has_commands"


def valid_evidence(value: object) -> dict[str, str] | None:
    """``value`` as a storable ``{"kind", "ref"}``, or ``None`` when it is not one."""
    if not isinstance(value, dict):
        return None
    kind = value.get("kind")
    ref = value.get("ref")
    if not isinstance(kind, str) or not isinstance(ref, str) or kind not in _REF_SHAPES:
        return None
    ref = ref.strip()
    if len(ref) > MAX_EVIDENCE_REF_CHARS or not _REF_SHAPES[kind].fullmatch(ref):
        return None
    return {"kind": kind, "ref": ref}


def evidence_key(evidence: dict[str, str]) -> str:
    """One string per reference, for keying and de-duplicating."""
    return f"{evidence['kind']}:{evidence['ref']}"


def item_verdicts(
    hermes_home: str | Path | None,
    session_ref: str,
    items: list[dict[str, Any]],
    orphans: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Judge each item against the records of ``session_ref`` and its ancestors.

    Each entry of ``items`` is ``{"evidence": dict | None, "from": epoch | None,
    "to": epoch | None, "held": set of evidence keys other items hold}`` -- the
    reference, the item's window, and what the rest of the plan holds.
    ``orphans`` are ``{"evidence", "at": epoch}``: references the plan bound
    to an item that a later write dropped (a rename, a split, a removal).
    Returns ``{"store": ..., "verdicts": [verdict per item]}``; on ``absent``
    and ``unreadable`` the list is empty and the caller decides what each
    means.
    """
    session = str(session_ref or "").strip()
    if not session or not hermes_home:
        return {"store": STORE_ABSENT, "verdicts": []}
    opened = _open_readonly(hermes_home)
    if opened is None:
        return {"store": STORE_ABSENT, "verdicts": []}
    connection, store = opened
    if connection is None:
        return {"store": store, "verdicts": []}
    try:
        live = _live_rows_clause(connection)
        lineage = _lineage(connection, session)
        bound = [
            _evidence_verdict(connection, lineage, live, item["evidence"], item.get("from"))
            if isinstance(item.get("evidence"), dict)
            else None
            for item in items
        ]
        failures = _unresolved_orphan_failures(connection, lineage, live, orphans or [], items, bound)
        verdicts = [
            _item_verdict(connection, lineage, live, item, binding, failures)
            for item, binding in zip(items, bound, strict=True)
        ]
    except sqlite3.Error:
        return {"store": STORE_UNREADABLE, "verdicts": []}
    finally:
        connection.close()
    return {"store": STORE_READ, "verdicts": verdicts}


def _unresolved_orphan_failures(
    connection: sqlite3.Connection,
    lineage: list[str],
    live: str,
    orphans: list[dict[str, Any]],
    items: list[dict[str, Any]],
    bound: list[str | None],
) -> list[float]:
    """When each still-unresolved dropped failure was dropped.

    A dropped binding that failed is the plan's memory of work that did not
    pass under a name the plan no longer uses. It is resolved once an item
    opened at or after the drop closes on a passing COMMAND of its own -- the
    renamed or split work passing -- and not by a pass some other, older item
    holds, nor by a file write, which does not undo a failing check.
    """
    pending: list[float] = []
    for orphan in orphans:
        evidence = orphan.get("evidence")
        at = orphan.get("at")
        if not isinstance(evidence, dict) or not isinstance(at, (int, float)):
            continue
        if _evidence_verdict(connection, lineage, live, evidence, None) != EVIDENCE_FAILED:
            continue
        resolved = any(
            verdict == EVIDENCE_CLOSED
            and item["evidence"].get("kind") == EVIDENCE_KIND_TOOL_CALL
            and isinstance(item.get("from"), (int, float))
            and item["from"] >= at
            for item, verdict in zip(items, bound, strict=True)
        )
        if not resolved:
            pending.append(float(at))
    return pending


def observed_calls(
    hermes_home: str | Path | None,
    session_ref: str,
    *,
    after_epoch: float,
    until_epoch: float,
    exclude: set[str],
) -> list[dict[str, Any]]:
    """Evidence-capable calls recorded in ``(after_epoch, until_epoch]``, oldest first.

    Each entry is ``{"evidence": {kind, ref}, "at": epoch}`` so a caller can
    place a call inside one item's window. Read at a done write, so a call it
    names is one whose RESULT Hermes had already persisted: the host flushes a
    round's results before it runs the next round's tools, so a call from an
    earlier round is on disk and a sibling call in the same round as the
    ``omh_todo`` call is not. Calls in ``exclude`` (keys from `evidence_key`)
    already hold another item and are skipped, so no call is bound twice.
    """
    session = str(session_ref or "").strip()
    if not session or not hermes_home:
        return []
    opened = _open_readonly(hermes_home)
    if opened is None or opened[0] is None:
        return []
    connection = opened[0]
    try:
        live = _live_rows_clause(connection)
        lineage = _lineage(connection, session)
        rows = connection.execute(
            f"SELECT tool_name, tool_call_id, timestamp FROM messages "
            f"WHERE session_id IN ({_marks(lineage)}) "
            f"AND role = 'tool' AND tool_name IN ({_marks(EVIDENCE_TOOLS)}) "
            f"AND COALESCE(tool_call_id, '') <> '' AND timestamp > ? AND timestamp <= ?{live} "
            f"ORDER BY id DESC LIMIT {MAX_OBSERVED_ROWS}",
            (*lineage, *EVIDENCE_TOOLS, after_epoch, until_epoch),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        connection.close()
    calls: list[dict[str, Any]] = []
    seen: set[str] = set(exclude)
    for tool_name, call_id, at in reversed(rows):
        kind = EVIDENCE_KIND_TOOL_CALL if tool_name == "terminal" else EVIDENCE_KIND_FILE_WRITE
        evidence = valid_evidence({"kind": kind, "ref": call_id})
        if evidence is None or evidence_key(evidence) in seen or not isinstance(at, (int, float)):
            continue
        seen.add(evidence_key(evidence))
        calls.append({"evidence": evidence, "at": float(at)})
    return calls


def _marks(values: tuple[str, ...] | list[str]) -> str:
    return ",".join("?" for _ in values)


def _open_readonly(hermes_home: str | Path) -> tuple[sqlite3.Connection | None, str] | None:
    """``None`` when there is no store, else a connection or the reading that replaced one."""
    try:
        path = Path(hermes_home).expanduser() / "state.db"
        if not path.exists() and not path.is_symlink():
            return None
        # The kanban reader refuses a linked store for the same reason: a link
        # decides which database is read, and nothing here chose it.
        if path.is_symlink() or not path.is_file():
            return None, STORE_UNREADABLE
        connection = sqlite3.connect(
            f"file:{quote(str(path))}?mode=ro", uri=True, timeout=_CONNECT_TIMEOUT_SECONDS
        )
    except (OSError, ValueError, sqlite3.Error):
        return None, STORE_UNREADABLE
    return connection, STORE_READ


def _live_rows_clause(connection: sqlite3.Connection) -> str:
    """The filter that drops rewound rows, on a store that records rewinds at all.

    Hermes marks a rewound turn's rows ``active = 0`` and keeps rows a
    compaction folded as ``compacted = 1`` (`hermes_state_messages.py`); a
    store older than those columns has rewound nothing.
    """
    columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)").fetchall()}
    if {"active", "compacted"} <= columns:
        return " AND (active = 1 OR compacted = 1)"
    return ""


def _lineage(connection: sqlite3.Connection, session: str) -> list[str]:
    lineage = [session]
    current = session
    for _ in range(MAX_LINEAGE_HOPS):
        row = connection.execute(
            "SELECT parent_session_id FROM sessions WHERE id = ?", (current,)
        ).fetchone()
        parent = str(row[0]) if row and row[0] else ""
        if not parent or parent in lineage:
            break
        lineage.append(parent)
        current = parent
    return lineage


def _item_verdict(
    connection: sqlite3.Connection,
    lineage: list[str],
    live: str,
    item: dict[str, Any],
    binding: str | None,
    orphan_failures: list[float],
) -> str:
    """One done item's verdict over its window, ``from`` (exclusive) to ``to``.

    The rules, in order:

    1. A bound reference that FAILED stays failed wherever it sits in time: a
       failure is replaced only by a later call bound in its place.
    2. A bound reference that passed closes when its result lies inside the
       window -- except a file write where a failing check stands: the
       window's latest command no other item holds failed, or rule 3
       applies. A write alone does not undo a failing check; a later passing
       command does. A write still closes docs-only work, whose window held
       no failing command.
    3. An item opened at or after the plan dropped a still-unresolved
       failure (`_unresolved_orphan_failures`) is failed: a rename or split
       does not carry work past its failing check.
    4. The latest evidence-capable call at or before ``to`` that no OTHER
       item holds decides next: if it failed, the item is failed. A pass
       another item holds says nothing about this one, so it is skipped.
    5. Calls inside the window, each holding another item, leave the item
       ``no_evidence``; an empty window is a conversational item and closes.
    """
    evidence = item.get("evidence")
    start = item.get("from")
    end = item.get("to")
    if binding == EVIDENCE_FAILED:
        return EVIDENCE_FAILED
    held = item.get("held") if isinstance(item.get("held"), (set, frozenset)) else set()
    after_drop = isinstance(start, (int, float)) and any(at <= start for at in orphan_failures)
    if binding == EVIDENCE_CLOSED:
        if evidence.get("kind") == EVIDENCE_KIND_FILE_WRITE and (
            after_drop or _failing_command_in_window(connection, lineage, live, start, end, held)
        ):
            return EVIDENCE_FAILED
        return EVIDENCE_CLOSED
    if after_drop:
        return EVIDENCE_FAILED
    upper = ""
    upper_params: list[Any] = []
    if isinstance(end, (int, float)):
        upper = " AND timestamp <= ?"
        upper_params.append(end)
    recent = connection.execute(
        f"SELECT tool_name, content, {_disposition_column(connection)}, tool_call_id FROM messages "
        f"WHERE session_id IN ({_marks(lineage)}) AND role = 'tool' "
        f"AND tool_name IN ({_marks(EVIDENCE_TOOLS)}){upper}{live} "
        f"ORDER BY timestamp DESC, id DESC LIMIT {MAX_BOUND_SKIP + 1}",
        (*lineage, *EVIDENCE_TOOLS, *upper_params),
    ).fetchall()
    for tool_name, content, effect, call_id in recent:
        kind = EVIDENCE_KIND_TOOL_CALL if tool_name == "terminal" else EVIDENCE_KIND_FILE_WRITE
        if evidence_key({"kind": kind, "ref": str(call_id)}) in held:
            continue
        if _result_verdict(kind, (tool_name, content, effect)) == EVIDENCE_FAILED:
            return EVIDENCE_FAILED
        break
    if isinstance(evidence, dict):
        return EVIDENCE_UNRESOLVED
    window = ""
    params: list[Any] = [*lineage, *EVIDENCE_TOOLS]
    if isinstance(start, (int, float)):
        window += " AND timestamp > ?"
        params.append(start)
    window += upper
    params.extend(upper_params)
    row = connection.execute(
        f"SELECT 1 FROM messages WHERE session_id IN ({_marks(lineage)}) AND role = 'tool' "
        f"AND tool_name IN ({_marks(EVIDENCE_TOOLS)}){window}{live} LIMIT 1",
        params,
    ).fetchone()
    return WINDOW_HAS_COMMANDS if row is not None else WINDOW_EMPTY


def _failing_command_in_window(
    connection: sqlite3.Connection,
    lineage: list[str],
    live: str,
    start: object,
    end: object,
    held: set[str] | frozenset[str],
) -> bool:
    """Whether the window's latest ``terminal`` call no other item holds failed."""
    window = ""
    params: list[Any] = [*lineage]
    if isinstance(start, (int, float)):
        window += " AND timestamp > ?"
        params.append(start)
    if isinstance(end, (int, float)):
        window += " AND timestamp <= ?"
        params.append(end)
    rows = connection.execute(
        f"SELECT tool_name, content, {_disposition_column(connection)}, tool_call_id FROM messages "
        f"WHERE session_id IN ({_marks(lineage)}) AND role = 'tool' AND tool_name = 'terminal'"
        f"{window}{live} ORDER BY timestamp DESC, id DESC LIMIT {MAX_BOUND_SKIP + 1}",
        params,
    ).fetchall()
    for tool_name, content, effect, call_id in rows:
        if evidence_key({"kind": EVIDENCE_KIND_TOOL_CALL, "ref": str(call_id)}) in held:
            continue
        return _result_verdict(EVIDENCE_KIND_TOOL_CALL, (tool_name, content, effect)) == EVIDENCE_FAILED
    return False


def _disposition_column(connection: sqlite3.Connection) -> str:
    """``effect_disposition``, or ``NULL`` on a store older than the column."""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)").fetchall()}
    return "effect_disposition" if "effect_disposition" in columns else "NULL"


def _evidence_verdict(
    connection: sqlite3.Connection,
    lineage: list[str],
    live: str,
    evidence: dict[str, str],
    start: object,
) -> str:
    kind = evidence.get("kind")
    if kind not in _KIND_TOOLS:
        return EVIDENCE_UNRESOLVED
    # A compaction re-persists a result under a new row id; the first row by
    # id decides, the rule #1922 states for the same duplication.
    row = connection.execute(
        f"SELECT tool_name, content, {_disposition_column(connection)}, timestamp FROM messages "
        f"WHERE session_id IN ({_marks(lineage)}) AND role = 'tool' AND tool_call_id = ?{live} "
        "ORDER BY id LIMIT 1",
        (*lineage, evidence.get("ref")),
    ).fetchone()
    if row is None:
        return EVIDENCE_UNRESOLVED
    tool_name, content, effect, timestamp = row
    verdict = _result_verdict(str(kind), (tool_name, content, effect))
    # A failure is sticky and needs no window; a pass counts only inside it,
    # or a call from before the item opened would close it as a copy.
    if verdict == EVIDENCE_CLOSED and isinstance(start, (int, float)) and not (
        isinstance(timestamp, (int, float)) and timestamp > start
    ):
        return EVIDENCE_UNRESOLVED
    return verdict


def _result_verdict(kind: str, row: tuple[Any, Any, Any]) -> str:
    """One recorded result's verdict. An unknown id is `unresolved`, never `failed`."""
    tool_name, content, effect = row
    # A kind names which host tool may answer for it, so a `file_write`
    # pointing at a terminal call (or the reverse) is not a reference to
    # anything this kind can judge.
    if tool_name not in _KIND_TOOLS[kind]:
        return EVIDENCE_UNRESOLVED
    if effect == "none":
        return EVIDENCE_FAILED
    if effect == "unknown":
        return EVIDENCE_UNRESOLVED
    try:
        result = json.loads(content) if isinstance(content, str) else None
    except ValueError:
        result = None
    if not isinstance(result, dict):
        return EVIDENCE_UNRESOLVED
    if result.get("error"):
        return EVIDENCE_FAILED
    if tool_name == "terminal":
        exit_code = result.get("exit_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            # `None` is the yield-to-background result: the command was never
            # observed finishing, which is not the same thing as failing.
            return EVIDENCE_UNRESOLVED
        return EVIDENCE_CLOSED if exit_code == 0 else EVIDENCE_FAILED
    if tool_name == "write_file":
        return EVIDENCE_CLOSED if "bytes_written" in result else EVIDENCE_UNRESOLVED
    success = result.get("success")
    if success is True:
        return EVIDENCE_CLOSED
    return EVIDENCE_FAILED if success is False else EVIDENCE_UNRESOLVED
