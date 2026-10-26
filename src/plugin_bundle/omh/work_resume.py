"""What this person was doing in their other Hermes sessions, from records only.

A person who leaves the TUI for Hermes Desktop, or one Slack thread for
another, and says "continue what I was doing" reaches a session that knows
nothing. OMH already keeps two records a session leaves behind: the plan todo
(`todo_store`, one file per declaring session, hidden after 24 hours) and the
completion checkpoint (`completion_store`, kept 30 days). This module joins
them to Hermes' own session rows and returns the ones that belong to the same
person, so the new session can pick the work up instead of starting cold.

Who "the same person" is comes only from what Hermes records, never from text:

* A local surface (`cli`, `tui`, `desktop`, `acp`) is the profile's own user.
  Every local session of the profile is theirs.
* A messaging surface (Slack, Discord, ...) counts only in a direct message:
  the calling row's `source` and `user_id`, `chat_type = 'dm'`. Only that
  user's other direct messages on that platform are theirs. A group channel or
  thread is not: its row names one user while anyone in the room can type
  "continue", and the answer would be read by everyone there.
* Hermes records no link between a platform user and the local user, or
  between one platform's user and another's, so those crossings are not made.
  A surface outside both vocabularies (webhook, cron), a delegated subagent
  session (`model_config._delegate_from`), and a shared chat name no single
  person, and nothing is shown to them.

"Of this profile" is the calling profile's own `state.db`. A record whose
declaring session that database does not list -- another profile sharing the
same OMH home -- is never read into the summary.

Metadata only (DIRECTION.md, Observation-First Reporting): the plan items and
checkpoint scope OMH already stores, and Hermes' session metadata (surface,
times, counts, repository name and branch). No session title, no activity
description, no deferral reason, no transcript: those are free text a person
or a model wrote. Plan items and checkpoint results stay what they were when
written, model declarations and not observed execution.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ._governance_safety import contains_credential_like_material
from .completion_store import CompletionValidationError, _digest, read_completion_dossiers
from .completion_store import STALE_SECONDS as CHECKPOINT_STALE_SECONDS
from .jev_consent import LOCAL_ATTENDED_PLATFORMS, MESSAGING_ATTENDED_PLATFORMS
from .todo_store import (
    TODO_SCHEMA_VERSION,
    TODO_STALE_SECONDS,
    _SESSION_RECORD_NAME,
    _read_todo_record,
    todo_session_dir,
)

WORK_RESUME_SCHEMA = "omh_work_resume/v1"
WORK_RESUME_CLAIM_BOUNDARY = (
    "Metadata from OMH plan and checkpoint records and Hermes session rows. Plan items and "
    "checkpoint results are model declarations from those sessions, not observed execution, "
    "review, CI, or merge evidence."
)
DEFAULT_LIMIT = 3
MAX_LIMIT = 10
# How many of the person's sessions are considered. A checkpoint lives 30 days,
# so this is the newest rows inside that window; the bound keeps one read cheap
# on a gateway profile with a long history.
MAX_CANDIDATE_SESSIONS = 500

# Optional on older hosts; read only when the table has them.
_SESSION_COLUMNS = (
    "id", "source", "user_id", "started_at", "ended_at", "last_activity_at",
    "message_count", "tool_call_count", "git_branch", "git_repo_root",
    "chat_type", "model_config",
)


def _delegated(row: dict[str, Any]) -> bool:
    """A `delegate_task` child: Hermes marks it only inside `model_config`."""
    try:
        config = json.loads(row.get("model_config") or "{}")
    except (TypeError, ValueError):
        return False
    return isinstance(config, dict) and bool(config.get("_delegate_from"))


def _principal(row: dict[str, Any]) -> tuple[str, str, str] | None:
    """(realm, source, user) for the calling row, or None when it names no person."""
    source = str(row.get("source") or "")
    if _delegated(row):
        return None
    if source in LOCAL_ATTENDED_PLATFORMS:
        return ("local", "", "")
    user = str(row.get("user_id") or "")
    if source in MESSAGING_ATTENDED_PLATFORMS and user and row.get("chat_type") == "dm":
        return ("messaging", source, user)
    return None


def _session_rows(state_db: Path, session_ref: str, now: float) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]]]:
    """(status, calling row, the person's other rows) from `state.db`, read-only."""
    if not state_db.is_file():
        return "store_unavailable", None, []
    try:
        connection = sqlite3.connect(f"file:{quote(str(state_db))}?mode=ro", uri=True, timeout=0.25)
    except sqlite3.Error:
        return "store_unavailable", None, []
    try:
        connection.row_factory = sqlite3.Row
        present = {str(info["name"]) for info in connection.execute("PRAGMA table_info(sessions)")}
        if not {"id", "source", "started_at"} <= present:
            return "store_unavailable", None, []
        columns = ", ".join(name for name in _SESSION_COLUMNS if name in present)
        caller = connection.execute(f"SELECT {columns} FROM sessions WHERE id = ?", (session_ref,)).fetchone()
        if caller is None:
            return "session_not_observed", None, []
        caller_row = dict(caller)
        principal = _principal(caller_row)
        if principal is None:
            return "unlinked_surface", caller_row, []
        realm, source, user = principal
        since = now - CHECKPOINT_STALE_SECONDS
        activity = "COALESCE(last_activity_at, ended_at, started_at)" if "last_activity_at" in present else (
            "COALESCE(ended_at, started_at)" if "ended_at" in present else "started_at")
        if realm == "local":
            placeholders = ",".join("?" for _ in LOCAL_ATTENDED_PLATFORMS)
            where, parameters = f"source IN ({placeholders})", list(sorted(LOCAL_ATTENDED_PLATFORMS))
        else:
            where, parameters = "source = ? AND user_id = ? AND chat_type = 'dm'", [source, user]
        # Delegated children and sessions the person archived or hid are not
        # their earlier work; the child's own plan belongs to its parent's run.
        if "model_config" in present:
            where += (" AND (model_config IS NULL OR NOT json_valid(model_config)"
                      " OR json_extract(model_config, '$._delegate_from') IS NULL)")
        for flag in ("archived", "hidden"):
            if flag in present:
                where += f" AND COALESCE({flag}, 0) = 0"
        rows = connection.execute(
            f"SELECT {columns}, {activity} AS activity_at FROM sessions "
            f"WHERE {where} AND id != ? AND {activity} >= ? ORDER BY activity_at DESC LIMIT ?",
            (*parameters, session_ref, since, MAX_CANDIDATE_SESSIONS),
        ).fetchall()
        return "read", caller_row, [dict(row) for row in rows]
    except sqlite3.Error:
        return "store_unavailable", None, []
    finally:
        connection.close()


def _timestamp(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _plans_by_session(omh_home: Path, sessions: set[str], now: float) -> dict[str, dict[str, Any]]:
    """Fresh per-session plan records whose declaring session is one of ``sessions``."""
    directory = todo_session_dir(omh_home)
    try:
        names = sorted(entry.name for entry in directory.iterdir() if _SESSION_RECORD_NAME.fullmatch(entry.name))
    except OSError:
        return {}
    plans: dict[str, dict[str, Any]] = {}
    for name in names:
        record = _read_todo_record(directory / name)
        if record is None or record.get("schema_version") != TODO_SCHEMA_VERSION:
            continue
        session_ref = str(record.get("session_ref") or "")
        updated = _timestamp(record.get("updated_at"))
        if session_ref not in sessions or updated is None or not 0 <= now - updated <= TODO_STALE_SECONDS:
            continue
        items = record.get("items")
        if not isinstance(items, list):
            continue
        plans[session_ref] = {
            "title": str(record.get("title") or ""),
            "updated_at": updated,
            "plan_stage": str(record.get("plan_stage") or ""),
            "deferred": bool(record.get("deferred_reason")),
            "items": [
                {key: _redacted(item[key]) for key in ("text", "state", "blocked_reason")
                 if isinstance(item.get(key), str) and item[key]}
                for item in items if isinstance(item, dict)
            ],
        }
    return plans


def _redacted(value: str) -> str:
    # Checkpoint text is redacted when written; plan items are not, and this
    # read may be relayed to a chat, so the same guard runs here.
    return "[redacted]" if contains_credential_like_material(value) else value


def _checkpoints_by_session(omh_home: Path, sessions: set[str], now: float) -> dict[str, list[dict[str, Any]]]:
    """Checkpoints under 30 days old a session of ``sessions`` declared, keyed by that session."""
    try:
        read = read_completion_dossiers(omh_home)
    except CompletionValidationError:
        return {}
    if read.get("status") != "read":
        return {}
    by_author = {_digest(session): session for session in sessions}
    found: dict[str, list[dict[str, Any]]] = {}
    for dossier in read.get("dossiers", []):
        checkpoint = dossier.get("checkpoint", {})
        session = by_author.get(checkpoint.get("author", ""))
        created = checkpoint.get("created_at")
        if session is None or not isinstance(created, (int, float)) or now - created > CHECKPOINT_STALE_SECONDS:
            continue
        found.setdefault(session, []).append({
            "checkpoint_id": checkpoint.get("checkpoint_id", ""),
            "title": checkpoint.get("title", ""),
            "created_at": checkpoint.get("created_at"),
            "item_count": len(checkpoint.get("items", [])),
            "sources": dossier.get("sources", {}),
        })
    return found


def _repo_name(value: object) -> str:
    text = str(value or "").rstrip("/\\")
    return text.replace("\\", "/").rsplit("/", 1)[-1] if text else ""


def read_work_resume(
    omh_home: Path,
    hermes_home: Path,
    session_ref: str,
    *,
    limit: int = DEFAULT_LIMIT,
    now: float | None = None,
) -> dict[str, Any]:
    """The person's recent OMH work in this profile, newest first; never a write."""
    moment = time.time() if now is None else now
    bounded = max(1, min(MAX_LIMIT, int(limit)))
    payload: dict[str, Any] = {
        "schema_version": WORK_RESUME_SCHEMA,
        "claim_boundary": WORK_RESUME_CLAIM_BOUNDARY,
        "window": {"plan_hours": TODO_STALE_SECONDS // 3600, "checkpoint_days": CHECKPOINT_STALE_SECONDS // 86400},
        "work": [],
    }
    if not session_ref:
        payload["status"] = "no_session"
        payload["text"] = render_work_resume_text(payload)
        return payload
    status, caller, rows = _session_rows(hermes_home / "state.db", session_ref, moment)
    if caller is not None:
        principal = _principal(caller)
        payload["surface"] = str(caller.get("source") or "")
        payload["realm"] = principal[0] if principal else "none"
    if status != "read":
        payload["status"] = status
        payload["text"] = render_work_resume_text(payload)
        return payload
    by_id = {str(row["id"]): row for row in rows}
    sessions = set(by_id)
    plans = _plans_by_session(omh_home, sessions, moment)
    checkpoints = _checkpoints_by_session(omh_home, sessions, moment)
    work = []
    # Both readers keep only records a session in `sessions` declared.
    for session in set(plans) | set(checkpoints):
        row = by_id[session]
        plan = plans.get(session)
        entries = sorted(checkpoints.get(session, []), key=lambda c: -float(c.get("created_at") or 0))
        stamps = ([plan["updated_at"]] if plan else []) + [float(c.get("created_at") or 0) for c in entries]
        work.append({
            "surface": str(row.get("source") or ""),
            "started_at": row.get("started_at"),
            "last_activity_at": row.get("activity_at"),
            "ended": row.get("ended_at") is not None,
            "message_count": row.get("message_count"),
            "tool_call_count": row.get("tool_call_count"),
            "repo": _repo_name(row.get("git_repo_root")),
            "branch": str(row.get("git_branch") or ""),
            "record_at": max(stamps),
            "plan": plan,
            "checkpoints": entries,
        })
    work.sort(key=lambda entry: (-float(entry["record_at"] or 0), entry["surface"]))
    payload["work"] = work[:bounded]
    payload["status"] = "found" if work else "none"
    payload["text"] = render_work_resume_text(payload, now=moment)
    return payload


_ITEM_MARKS = {"done": "[x]", "active": "[>]", "pending": "[ ]"}
_REALM_WORDS = {
    "local": "your local sessions in this profile (TUI, Desktop, CLI)",
    "messaging": "your sessions on this chat platform in this profile",
}


def _ago(seconds_then: object, now: float) -> str:
    if not isinstance(seconds_then, (int, float)) or isinstance(seconds_then, bool):
        return "at an unrecorded time"
    minutes = max(0, int(now - float(seconds_then)) // 60)
    if minutes < 60:
        return f"{minutes}m ago"
    if minutes < 48 * 60:
        return f"{minutes // 60}h ago"
    return f"{minutes // 1440}d ago"


def render_work_resume_text(payload: dict[str, Any], *, now: float | None = None) -> str:
    """Plain-text rendering of the payload; the default surface for a person."""
    moment = time.time() if now is None else now
    status = payload.get("status")
    window = payload.get("window", {})
    if status == "no_session":
        return "Hermes did not name this session, so OMH cannot tell whose earlier work to show."
    if status == "session_not_observed":
        return "Hermes has no record of this session yet, so OMH cannot tell whose earlier work to show."
    if status == "store_unavailable":
        return "Hermes' session store could not be read, so no earlier work is shown."
    if status == "unlinked_surface":
        return (f"This surface ({payload.get('surface') or 'unknown'}) carries no person Hermes records, "
                "so no earlier work is shown.")
    realm = _REALM_WORDS.get(str(payload.get("realm")), "this profile")
    if status == "none":
        return (f"No earlier OMH work is recorded in {realm}: no plan from the last "
                f"{window.get('plan_hours')} hours and no checkpoint from the last {window.get('checkpoint_days')} days.")
    lines = [f"Earlier OMH work in {realm}, newest first:"]
    for number, entry in enumerate(payload.get("work", []), start=1):
        place = f" in {entry['repo']}" if entry.get("repo") else ""
        place += f" on {entry['branch']}" if entry.get("branch") else ""
        state = "ended" if entry.get("ended") else "last active"
        calls = entry.get("tool_call_count")
        count = f", {calls} tool calls" if isinstance(calls, int) and not isinstance(calls, bool) else ""
        lines.append(f"{number}. {entry.get('surface')} session{place}, {state} {_ago(entry.get('last_activity_at'), moment)}{count}")
        plan = entry.get("plan")
        if plan:
            items = plan["items"]
            done = sum(1 for item in items if item.get("state") == "done")
            title = f' "{plan["title"]}"' if plan.get("title") else ""
            notes = "".join((
                ", awaiting acceptance" if plan.get("plan_stage") == "awaiting_acceptance" else "",
                ", set aside by the person" if plan.get("deferred") else "",
            ))
            lines.append(f"   Plan{title}: {done} of {len(items)} declared done, updated {_ago(plan['updated_at'], moment)}{notes}")
            for item in items:
                blocked = f" (blocked: {item['blocked_reason']})" if item.get("blocked_reason") else ""
                lines.append(f"   {_ITEM_MARKS.get(item.get('state', ''), '[ ]')} {item.get('text', '')}{blocked}")
        for checkpoint in entry.get("checkpoints", []):
            title = f' "{checkpoint["title"]}"' if checkpoint.get("title") else ""
            verification = checkpoint.get("sources", {}).get("verification", "absent")
            lines.append(f"   Checkpoint {checkpoint['checkpoint_id']}{title}: {checkpoint['item_count']} accepted items, "
                         f"verification {verification.replace('_', ' ')}")
    lines.append("Items marked done are that session's declarations, not observed results; "
                 "check the work itself before building on them.")
    return "\n".join(lines)
