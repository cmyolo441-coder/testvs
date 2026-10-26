"""Did evaluating a person's tool-call rules fail, and when?

`toolcall_rules` is fail-open by design: a missing, malformed, or oversized
rules file degrades to "no intervention". That contract covers every failure
the module anticipated. It does not cover a failure it did not -- an
unexpected exception escaping `toolcall_rule_directive` -- and the host does
not cover it either: when a `pre_tool_call` callback raises, Hermes appends
nothing, logs one WARNING and then DEBUG only, so the person's blocks stop
running with no visible trace (`hermes_cli/plugins_dispatch.py`, read, not
reproduced).

That is the one state where "no rule matched" and "the rule gate is broken"
look identical from outside. This records the difference so `omh doctor` can
name it.

Counters, not an append-only log, for the reason `awareness_delivery` gives:
this sits on the hottest path in the system, and a per-call log there is how
a journal reached thousands of rows of noise. A fixed-shape record cannot
grow.

Metadata only, and by construction rather than by intention. The record
carries the dispatched tool's name and the exception's TYPE name, nothing
else from the failure. The exception's MESSAGE is deliberately dropped: it is
free text produced by whatever raised, so a `re.error` quotes the rule's own
pattern, a `KeyError` quotes a key, and any handler formatting a value with
`!r` quotes an argument fragment. Storing it would have made this sentence a
promise the code could not keep, on the same ledger whose redaction policy is
`metadata_only`. The type name is enough for doctor to name the fault and for
a person to recognise it; `_identifier_text` refuses anything that is not a
dotted identifier, so no caller can put free text back in through this field.
"""

from __future__ import annotations

from . import runtime_paths

import json
import os
import secrets
from pathlib import Path
from typing import Any

TOOLCALL_RULE_FAULTS_SCHEMA_VERSION = "omh_toolcall_rule_faults/v1"
TOOLCALL_RULE_FAULTS_FILE = "toolcall_rule_faults.json"

MAX_FAULT_TYPE_CHARS = 96
MAX_FAULT_TOOL_CHARS = 96
# What a value that is not a dotted identifier becomes. An exception type name
# always is one; anything else reaching this field is a caller bug, and naming
# it is more useful than storing whatever it was.
UNKNOWN_FAULT_TYPE = "unrecognized_error_type"


def toolcall_rule_faults_path(omh_home: str = "") -> Path:
    root = runtime_paths.expand_path(omh_home) if omh_home else runtime_paths.default_omh_home()
    return root / "runtime" / TOOLCALL_RULE_FAULTS_FILE


def empty_toolcall_rule_faults() -> dict[str, Any]:
    return {
        "schema_version": TOOLCALL_RULE_FAULTS_SCHEMA_VERSION,
        "fault_count": 0,
        "first_fault_at": "",
        "last_fault_at": "",
        "last_error_type": "",
        "last_tool": "",
        "unreadable": False,
    }


def read_toolcall_rule_faults(omh_home: str = "") -> dict[str, Any]:
    """Current fault counters, or the empty record when nothing ever failed."""
    try:
        data = json.loads(toolcall_rule_faults_path(omh_home).read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError):
        return empty_toolcall_rule_faults()
    except (OSError, json.JSONDecodeError):
        return {**empty_toolcall_rule_faults(), "unreadable": True}
    if not isinstance(data, dict) or not _valid_fault_record(data):
        return {**empty_toolcall_rule_faults(), "unreadable": True}
    return {**empty_toolcall_rule_faults(), **data}


def record_toolcall_rule_fault(
    *,
    tool_name: object,
    error_type: str,
    observed_at: str,
    omh_home: str = "",
) -> dict[str, Any] | None:
    """Count one rule-gate evaluation failure. Best-effort, never raises.

    ``error_type`` is the exception's type name and only that: pass
    ``type(exc).__name__``, never ``str(exc)``. See the module docstring for
    why the message is not stored.

    A lost increment under concurrency is acceptable and the read-modify-write
    below is deliberately unlocked: `omh doctor` asks whether the rule gate has
    ever failed and what kind of failure it was, and neither answer depends on
    the count being exact. What must not happen is this recorder breaking the
    hook it exists to report on, so every write fault returns None.
    """
    path = toolcall_rule_faults_path(omh_home)
    try:
        current = read_toolcall_rule_faults(omh_home)
        updated = {
            "schema_version": TOOLCALL_RULE_FAULTS_SCHEMA_VERSION,
            "fault_count": int(current["fault_count"]) + 1,
            "first_fault_at": current["first_fault_at"] or observed_at,
            "last_fault_at": observed_at,
            "last_error_type": _identifier_text(error_type, MAX_FAULT_TYPE_CHARS),
            "last_tool": _bounded(str(tool_name or ""), MAX_FAULT_TOOL_CHARS),
        }
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _write_record(path, updated)
    except (OSError, TypeError, ValueError):
        return None
    return updated


def _bounded(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _identifier_text(value: str, limit: int) -> str:
    """A dotted Python identifier, or the placeholder. Never free text.

    This is what makes the module's metadata-only claim structural instead of
    a convention. An exception type name passes; a message, a path, a pattern
    or an argument fragment cannot, because none of them is an identifier.
    """
    text = str(value or "").strip()
    if not text or len(text) > limit:
        return UNKNOWN_FAULT_TYPE
    parts = text.split(".")
    if all(part.isidentifier() for part in parts):
        return text
    return UNKNOWN_FAULT_TYPE


def _valid_fault_record(data: dict[str, Any]) -> bool:
    if data.get("schema_version") != TOOLCALL_RULE_FAULTS_SCHEMA_VERSION:
        return False
    count = data.get("fault_count", 0)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        return False
    return all(
        isinstance(data.get(key, ""), str)
        for key in ("first_fault_at", "last_fault_at", "last_error_type", "last_tool")
    )


def _write_record(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}-{secrets.token_hex(8)}.tmp")
    created = False
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            created = True
            handle.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        tmp.chmod(0o600)
        tmp.replace(path)
        path.chmod(0o600)
    except OSError:
        if created and tmp.exists() and not tmp.is_symlink():
            tmp.unlink()
        raise
