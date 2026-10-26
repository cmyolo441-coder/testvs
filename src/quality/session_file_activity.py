"""Which workspace files one Hermes session's file tools touched, read from ``state.db``.

Hermes persists every model tool call on the assistant row that issued it
(``messages.tool_calls``: a JSON list of ``{"id", "function": {"name",
"arguments"}}``) and every result as a ``role = 'tool'`` row carrying the same
``tool_call_id``, the tool's own JSON result and, for a call that never ran
or whose effect is unknown, ``effect_disposition``. This reader derives a
bounded ``session_file_activity/v1`` projection for one session from those
two records at query time. OMH keeps no store of its own for it and records
nothing while the session runs.

Covered tools are ``read_file``, ``write_file`` and ``patch``; a V4A patch
contributes every file it declares. The outcome of a call comes from the
recorded result's own fields -- the ones Hermes' own classifier reads -- and
never from wording: a call with no recorded result, an unknown effect, or a
result without a success field is ``unknown``, so a submitted path is never
presented as a successful write. Paths are shown workspace-relative only; a
path outside the workspace, or one that cannot be placed against it, is
omitted and counted by reason, never shown. A path inside the workspace by its
spelling is also omitted when a symlink on it leads outside the workspace:
the links are resolved at query time (``os.path.realpath``, link metadata
only), so the verdict reflects the links as they stand when the query runs.
The database is opened ``mode=ro`` and no file named in a call is opened or
read.

A compaction re-persists assistant and tool rows under new ids, so a call is
one distinct call id per session, the first row by id deciding both its
arguments and its result.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping

from .hermes_state import TOOL_CALL_KEY_SQL as _CALL_KEY, hermes_epoch, open_state_db_readonly, resolve_session_id


SESSION_FILE_ACTIVITY_SCHEMA_VERSION = "session_file_activity/v1"
FILE_ACTIVITY_TOOLS: tuple[str, ...] = ("read_file", "write_file", "patch")
OUTCOMES: tuple[str, ...] = ("succeeded", "failed", "unknown")
OPERATIONS: tuple[str, ...] = ("read", "write", "update", "add", "delete", "move_from", "move_to")
OMISSION_REASONS: tuple[str, ...] = (
    "outside_workspace",
    "symlink_escape",
    "workspace_unknown",
    "relative_without_cwd",
    "home_relative",
    "url_like",
    "control_characters",
    "oversized",
    "malformed",
)
DEFAULT_MAX_FILES = 100
MAX_PATH_CHARS = 4096

SESSION_FILE_ACTIVITY_CLAIM_BOUNDARY = (
    "Session file activity is a read-only projection of the file-tool calls one Hermes session "
    "persisted to its own session store. A path is the one the call declared, shown only when it "
    "lies inside the workspace by its spelling and through every symlink on it as the links stand at "
    "query time (link metadata only; no named file is opened or read). "
    "`succeeded` means Hermes recorded a result whose own fields report success; it is not "
    "file-content, diff, test, review, CI, or merge evidence."
)

_OUTCOME_RULES: dict[str, str] = {
    "call": "one distinct tool call id per session over assistant rows' tool_calls, the first row by id "
    "deciding its arguments; the first tool row by id with that tool_call_id decides its result",
    "unknown": "no recorded result, effect_disposition 'unknown', a result that is not a JSON object, "
    "or a result without the tool's success field",
    "failed": "effect_disposition 'none' (the call had no effect), a truthy 'error' field, "
    "or patch 'success' false",
    "succeeded": "read_file: a 'content' field; write_file: a 'bytes_written' field; "
    "patch: 'success' true -- each without an 'error' field",
    "patch_files": "a V4A patch applies its outcome to every file it declares",
    "timestamps": "the issuing assistant row's timestamp, as UTC",
    "workspace": "--workspace, else the session's git_repo_root, else its cwd (Hermes' own workspace key); "
    "a relative path is placed against the session's cwd",
    "symlinks": "a path inside the workspace by its spelling whose symlinks, resolved at query time, lead "
    "outside the workspace is omitted as symlink_escape",
}

# Hermes' V4A markers (tools/patch_parser.py): a marker is the whole line.
_V4A_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("update", re.compile(r"^\*\*\*\s*Update\s+File:\s*(.+)$")),
    ("add", re.compile(r"^\*\*\*\s*Add\s+File:\s*(.+)$")),
    ("delete", re.compile(r"^\*\*\*\s*Delete\s+File:\s*(.+)$")),
)
_V4A_MOVE = re.compile(r"^\*\*\*\s*Move\s+File:\s*(.+?)\s*->\s*(.+)$")


class SessionFileActivityError(ValueError):
    """The database, the session or an argument could not be used; the message says which."""


def build_session_file_activity(
    hermes_home: str | Path,
    session_id: str,
    *,
    workspace: str | None = None,
    max_files: int = DEFAULT_MAX_FILES,
) -> dict[str, Any]:
    """Project one session's ``read_file`` / ``write_file`` / ``patch`` calls per workspace file.

    ``session_id`` may be ``latest``. An explicit id without a session row is
    an error, so a mistyped id is never reported as a session that touched
    nothing. ``max_files`` bounds the listed files; the payload says when it
    cut the list and how many files it left out.
    """
    if max_files < 1:
        raise SessionFileActivityError("--max-files must be at least 1")
    path, connection = open_state_db_readonly(hermes_home, error=SessionFileActivityError)
    try:
        resolved = resolve_session_id(connection, session_id, None, error=SessionFileActivityError)
        session_row = _session_row(connection, resolved)
        call_rows = connection.execute(
            "SELECT id, tool_calls, timestamp FROM messages "
            "WHERE session_id = ? AND role = 'assistant' AND tool_calls IS NOT NULL ORDER BY id",
            (resolved,),
        ).fetchall()
        # A store older than Hermes' effect_disposition column has no disposition to read.
        message_columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)").fetchall()}
        disposition = "effect_disposition" if "effect_disposition" in message_columns else "NULL"
        result_rows = connection.execute(
            f"SELECT {_CALL_KEY}, tool_name, content, {disposition} FROM messages "
            "WHERE session_id = ? AND role = 'tool' AND tool_name IN (?, ?, ?) ORDER BY id",
            (resolved, *FILE_ACTIVITY_TOOLS),
        ).fetchall()
    except sqlite3.Error as exc:
        raise SessionFileActivityError(f"could not read {path}: {exc}") from exc
    finally:
        connection.close()
    if session_row is None:
        raise SessionFileActivityError(f"no Hermes session {resolved}")

    cwd = _clean_dir(session_row.get("cwd"))
    root, basis = _workspace_root(workspace, session_row, cwd)
    real_root = os.path.realpath(root) if root is not None else None

    results: dict[str, tuple[str, Any, Any]] = {}
    for call_key, tool_name, content, disposition in result_rows:
        results.setdefault(str(call_key), (str(tool_name), content, disposition))

    calls = _file_tool_calls(call_rows)
    files: dict[str, dict[str, Any]] = {}
    omitted = {reason: 0 for reason in OMISSION_REASONS}
    by_tool = {name: 0 for name in FILE_ACTIVITY_TOOLS}
    by_outcome = {outcome: 0 for outcome in OUTCOMES}
    for call in calls:
        by_tool[call["tool"]] += 1
        result = results.get(call["key"])
        outcome = _outcome(call["tool"], result)
        by_outcome[outcome] += 1
        targets = _call_targets(call["tool"], call["arguments"])
        if not targets:
            omitted["malformed"] += 1
            continue
        for raw_path, operation in targets:
            relative, reason = _workspace_relative(raw_path, root, real_root, cwd)
            if relative is None:
                omitted[reason] += 1
                continue
            _record(files, relative, operation, outcome, call["at"])
    call_keys = {call["key"] for call in calls}
    results_without_call = sum(1 for key in results if key not in call_keys)

    ordered = sorted(files.values(), key=lambda entry: (entry["first_epoch"] is None, entry["first_epoch"] or 0.0, entry["path"]))
    shown = [_finish_file(entry) for entry in ordered[:max_files]]
    return {
        "schema_version": SESSION_FILE_ACTIVITY_SCHEMA_VERSION,
        "source": {"kind": "hermes_state_db", "path": str(path), "session_id": resolved},
        "workspace": {"basis": basis, "established": root is not None},
        "tools": list(FILE_ACTIVITY_TOOLS),
        "calls": {
            "total": len(calls),
            "by_tool": by_tool,
            "by_outcome": by_outcome,
            "results_without_call": results_without_call,
        },
        "files": shown,
        "file_count": len(ordered),
        "shown_file_count": len(shown),
        "max_files": max_files,
        "truncated": len(ordered) > max_files,
        "omitted_file_count": len(ordered) - len(shown),
        "omitted_paths": {"count": sum(omitted.values()), "by_reason": omitted},
        "rules": dict(_OUTCOME_RULES),
        "observed": True,
        "claim_boundary": SESSION_FILE_ACTIVITY_CLAIM_BOUNDARY,
    }


def format_session_file_activity_summary(payload: Mapping[str, Any]) -> str:
    """Plain-text rendering: header, one line per file, omissions, then the boundary."""
    calls = payload.get("calls") or {}
    workspace = payload.get("workspace") or {}
    source = payload.get("source") or {}
    file_count = int(payload.get("file_count", 0))
    shown = int(payload.get("shown_file_count", 0))
    out = [f"OMH session file activity: session {source.get('session_id')}"]
    out.append(
        f"  workspace: {workspace.get('basis')}    files: {file_count}    calls: {int(calls.get('total', 0))}    "
        + "  ".join(f"{outcome} {int((calls.get('by_outcome') or {}).get(outcome, 0))}" for outcome in OUTCOMES)
    )
    out.append("Files")
    if not payload.get("files"):
        out.append("  (none)")
    for entry in payload.get("files") or ():
        activity = ", ".join(
            f"{item['operation']} {item['outcome']} x{item['calls']}" for item in entry.get("activity") or ()
        )
        out.append(f"  {entry['path']}: {activity}    {entry.get('first_at')} .. {entry.get('last_at')}")
    if payload.get("truncated"):
        out.append(
            f"  truncated: showing {shown} of {file_count} files (--max-files {payload.get('max_files')}); "
            f"{int(payload.get('omitted_file_count', 0))} not listed"
        )
    omitted = payload.get("omitted_paths") or {}
    reasons = {reason: count for reason, count in (omitted.get("by_reason") or {}).items() if count}
    without_call = int(calls.get("results_without_call", 0))
    if reasons or without_call:
        out.append("Omitted")
        if reasons:
            out.append("  paths not shown: " + "  ".join(f"{reason} {count}" for reason, count in reasons.items()))
        if without_call:
            out.append(f"  results without a recorded call: {without_call}")
    out.append("Boundary")
    out.append(f"  {payload.get('claim_boundary', SESSION_FILE_ACTIVITY_CLAIM_BOUNDARY)}")
    return "\n".join(out)


def _session_row(connection: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)").fetchall()}
    wanted = [name for name in ("cwd", "git_repo_root") if name in columns]
    select = ", ".join(["id", *wanted])
    row = connection.execute(f"SELECT {select} FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        return None
    return dict(zip(["id", *wanted], row))


def _clean_dir(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip() or _unsafe_reason(value) is not None:
        return None
    text = value.strip()
    return os.path.normpath(text) if os.path.isabs(text) else None


def _workspace_root(workspace: str | None, session_row: Mapping[str, Any], cwd: str | None) -> tuple[str | None, str]:
    if workspace is not None:
        if not workspace.strip() or _unsafe_reason(workspace) is not None:
            raise SessionFileActivityError(f"--workspace is not a usable directory path: {workspace!r}")
        return os.path.normpath(os.path.abspath(os.path.expanduser(workspace.strip()))), "argument"
    repo_root = _clean_dir(session_row.get("git_repo_root"))
    if repo_root is not None:
        return repo_root, "git_repo_root"
    if cwd is not None:
        return cwd, "cwd"
    return None, "unknown"


def _file_tool_calls(call_rows: list[tuple[Any, Any, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_id, tool_calls, stamp in call_rows:
        try:
            entries = json.loads(tool_calls) if isinstance(tool_calls, str) else None
        except ValueError:
            entries = None
        if not isinstance(entries, list):
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            function = entry.get("function")
            name = function.get("name") if isinstance(function, dict) else None
            if name not in FILE_ACTIVITY_TOOLS:
                continue
            call_id = entry.get("id") or entry.get("call_id")
            key = str(call_id) if isinstance(call_id, str) and call_id else f"row:{row_id}:{index}"
            if key in seen:
                continue
            seen.add(key)
            calls.append(
                {
                    "key": key,
                    "tool": name,
                    "arguments": _arguments(function.get("arguments")),
                    "at": hermes_epoch(stamp),
                }
            )
    return calls


def _arguments(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _call_targets(tool: str, arguments: Mapping[str, Any] | None) -> list[tuple[Any, str]]:
    """``(declared path, operation)`` for every file a call names; ``[]`` when it names none."""
    if arguments is None:
        return []
    if tool == "read_file":
        return [(arguments.get("path"), "read")] if "path" in arguments else []
    if tool == "write_file":
        return [(arguments.get("path"), "write")] if "path" in arguments else []
    if arguments.get("mode") == "patch":
        return _v4a_targets(arguments.get("patch"))
    return [(arguments.get("path"), "update")] if "path" in arguments else []


def _v4a_targets(patch: Any) -> list[tuple[Any, str]]:
    if not isinstance(patch, str):
        return []
    targets: list[tuple[Any, str]] = []
    for line in patch.split("\n"):
        line = line[:-1] if line.endswith("\r") else line
        move = _V4A_MOVE.match(line)
        if move is not None:
            targets.append((move.group(1).strip(), "move_from"))
            targets.append((move.group(2).strip(), "move_to"))
            continue
        for operation, marker in _V4A_MARKERS:
            match = marker.match(line)
            if match is not None:
                targets.append((match.group(1).strip(), operation))
                break
    return targets


def _outcome(tool: str, result: tuple[str, Any, Any] | None) -> str:
    """The closed outcome of one call, from its recorded result's own fields only."""
    if result is None:
        return "unknown"
    _name, content, disposition = result
    if disposition == "none":
        return "failed"
    if disposition == "unknown":
        return "unknown"
    data = _leading_json_object(content)
    if data is None:
        return "unknown"
    if data.get("error"):
        return "failed"
    if tool == "read_file":
        return "succeeded" if "content" in data else "unknown"
    if tool == "write_file":
        return "succeeded" if "bytes_written" in data else "unknown"
    if data.get("success") is True:
        return "succeeded"
    return "failed" if data.get("success") is False else "unknown"


def _leading_json_object(content: Any) -> dict[str, Any] | None:
    """The JSON object a result starts with; Hermes may append a notice after it."""
    if not isinstance(content, str):
        return None
    text = content.lstrip()
    if not text.startswith("{"):
        return None
    try:
        value, _end = json.JSONDecoder().raw_decode(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _unsafe_reason(value: str) -> str | None:
    if len(value) > MAX_PATH_CHARS:
        return "oversized"
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        return "control_characters"
    if "://" in value or value.lower().startswith("file:"):
        return "url_like"
    return None


def _workspace_relative(raw: Any, root: str | None, real_root: str | None, cwd: str | None) -> tuple[str | None, str]:
    """``(relative path, "")`` inside the workspace, else ``(None, reason)``; never the raw value."""
    if not isinstance(raw, str) or not raw.strip():
        return None, "malformed"
    unsafe = _unsafe_reason(raw)
    if unsafe is not None:
        return None, unsafe
    text = raw.strip()
    if text.startswith("~"):
        return None, "home_relative"
    if root is None:
        return None, "workspace_unknown"
    if os.path.isabs(text):
        absolute = os.path.normpath(text)
    elif cwd is not None:
        absolute = os.path.normpath(os.path.join(cwd, text))
    else:
        return None, "relative_without_cwd"
    if not _within(root, absolute) or os.path.normcase(absolute) == os.path.normcase(root):
        return None, "outside_workspace"
    # Both sides resolved, so a workspace reached through a link of its own still contains its files.
    if real_root is not None and not _within(real_root, os.path.realpath(absolute)):
        return None, "symlink_escape"
    return os.path.relpath(absolute, root).replace(os.sep, "/"), ""


def _within(root: str, path: str) -> bool:
    try:
        return os.path.commonpath([os.path.normcase(root), os.path.normcase(path)]) == os.path.normcase(root)
    except ValueError:
        return False


def _record(files: dict[str, dict[str, Any]], path: str, operation: str, outcome: str, at: float | None) -> None:
    entry = files.setdefault(path, {"path": path, "first_epoch": None, "last_epoch": None, "activity": {}})
    _widen(entry, "first_epoch", "last_epoch", at)
    item = entry["activity"].setdefault(
        (operation, outcome),
        {"operation": operation, "outcome": outcome, "calls": 0, "first_epoch": None, "last_epoch": None},
    )
    item["calls"] += 1
    _widen(item, "first_epoch", "last_epoch", at)


def _widen(target: dict[str, Any], first: str, last: str, at: float | None) -> None:
    if at is None:
        return
    if target[first] is None or at < target[first]:
        target[first] = at
    if target[last] is None or at > target[last]:
        target[last] = at


def _finish_file(entry: Mapping[str, Any]) -> dict[str, Any]:
    order = {operation: index for index, operation in enumerate(OPERATIONS)}
    rank = {outcome: index for index, outcome in enumerate(OUTCOMES)}
    activity = sorted(entry["activity"].values(), key=lambda item: (order[item["operation"]], rank[item["outcome"]]))
    return {
        "path": entry["path"],
        "first_at": _utc(entry["first_epoch"]),
        "last_at": _utc(entry["last_epoch"]),
        "activity": [
            {
                "operation": item["operation"],
                "outcome": item["outcome"],
                "calls": item["calls"],
                "first_at": _utc(item["first_epoch"]),
                "last_at": _utc(item["last_epoch"]),
            }
            for item in activity
        ],
    }


def _utc(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
