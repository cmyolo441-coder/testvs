"""The reminder that asks about an open memory record, and the ledger of asks.

An open record (``staleness.resolution == "open"``) is a question a person
chose not to settle. Past its review deadline it is still delivered, marked
``open · N days unresolved`` -- but a marker inside a pack is easy to stop
seeing. Owner direction (issue #1528): OMH should *ask*. The provider appends
one ``omh reminder:`` line to the prefetch pack when an open record crosses an
ask threshold, Hermes relays it as a question, and the answer maps to the
three verbs -- ``confirm`` (resolved), ``keep-open`` (still open), ``retire``
(drop it).

Cadence, so it costs attention without becoming noise: ask once when the
review deadline passes, then at most every ``open_ask_days`` per record;
"still open" resets that clock. Never more than one reminder line per pack;
the rest queue by age, oldest ``open_since`` first.

Boundary: the reminder only asks. This module writes the ask ledger and
nothing else -- never a record -- so a reminder can never promote or retire a
record on its own. Every state change still goes through confirm / correct /
retire. Stdlib only; no import of the ``omh`` control plane.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any

from .awareness_delivery import _awareness_delivery_lock as _ledger_lock
from .memory_governance import PRINCIPAL_PROJECT_MEMORY_RECORD_SCHEMA_VERSION, evaluate_renderable_strings
from .memory_recall_support import (
    PROJECT_MEMORY_RECORD_SCHEMA_VERSION,
    _MEMORY_CADENCE_DEFAULTS,
    _parse_utc_naive_as_utc,
    _record_staleness,
    _redact_admitted_text,
)
from .memory_records import RECORD_SUMMARY_LIMIT_CHARS
from .memory_state_files import write_text_atomic

MEMORY_OPEN_REMINDERS_SCHEMA_VERSION = "omh_memory_open_reminders/v1"
OPEN_REMINDERS_FILENAME = "open_reminders.json"


def open_reminders_path(omh_home: str | Path) -> Path:
    return Path(omh_home).expanduser() / "memory" / OPEN_REMINDERS_FILENAME


def read_open_reminders(homes: list[Path] | tuple[Path, ...] | Path | str) -> dict[str, dict[str, object]]:
    """The ask ledger across every home, the latest ``asked_at`` per record winning.

    The provider reads the project store and the user store; ``omh memory
    keep-open`` may have written either one, so an answer given through
    whichever scope the operator chose still resets the clock the provider
    reads. A foreign schema, a malformed file, or a malformed entry reads as
    absent -- never as "asked", which would silence a reminder.
    """
    if isinstance(homes, (str, Path)):
        homes = (Path(homes),)
    merged: dict[str, dict[str, object]] = {}
    for home in homes:
        for record_id, entry in _read_ledger(open_reminders_path(home)).items():
            current = merged.get(record_id)
            if current is None or str(entry.get("asked_at", "")) > str(current.get("asked_at", "")):
                merged[record_id] = entry
    return merged


def mark_open_reminder_asked(omh_home: str | Path, record_id: str, *, asked_at: str) -> dict[str, object]:
    """Record one ask (or one "still open" answer) for ``record_id`` at ``omh_home``.

    The read-modify-write runs under the bundle's own OS lock (the same
    two-backend lock the awareness ledger uses, on a sidecar lock file next
    to the ledger) so a provider prefetch racing an operator's
    ``omh memory keep-open`` cannot read the same ledger and overwrite the
    answer: both writers pass through this function. The write itself goes
    through ``write_text_atomic`` (temporary file and ``os.replace``) so a
    crash leaves a whole ledger, not half of one. Raises ``OSError`` (a lock timeout is one) on a
    home that cannot be written; the provider swallows that through
    ``_safely`` because a lost ledger line must never cost a turn, and the
    CLI reports it, so an operator's answer is either recorded or refused
    out loud, never dropped silently.
    """
    path = open_reminders_path(omh_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ledger_lock(path):
        ledger = _read_ledger(path)
        previous = ledger.get(str(record_id), {})
        count = previous.get("asked_count", 0)
        count = count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else 0
        entry: dict[str, object] = {"asked_at": str(asked_at), "asked_count": count + 1}
        ledger[str(record_id)] = entry
        payload = json.dumps(
            {"schema_version": MEMORY_OPEN_REMINDERS_SCHEMA_VERSION, "records": dict(sorted(ledger.items()))},
            ensure_ascii=False,
            sort_keys=True,
        )
        write_text_atomic(path, payload + "\n")
    return entry


def select_open_reminder(
    records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ledger: dict[str, dict[str, object]],
    *,
    now: datetime,
    allowed_scopes: list[dict[str, str]] | tuple[dict[str, str], ...],
    open_ask_days: int | None = None,
    eligible_record_ids: set[str] | frozenset[str] | None = None,
) -> dict[str, object] | None:
    """The one open record to ask about this turn, or None.

    A record qualifies when its freshness verdict is ``open`` (marked
    unresolved AND past its review deadline -- an open record still inside
    its deadline is not asked about, the deadline is the first ask), its
    scope is inside the pack's explicit allowlist (a record from another
    project must not be asked about in this session), the pack's own
    eligibility evaluation admitted it (``eligible_record_ids``: a
    superseded, archived, or otherwise refused record is a question two of
    the three answers would refuse, so it is never asked), its summary
    passes the same renderable-string safety the pack applies, and the
    ledger holds no ask for it or the last ask is older than
    ``open_ask_days``. Principal-bound (``v3``) records qualify on the same
    terms as ``v2`` ones: their audience and principal rules are the pack
    selector's, and ``eligible_record_ids`` is how those rules reach here --
    this selector never becomes a second, looser reader of them. Oldest
    ``open_since`` wins; the record id breaks ties so the choice is
    reproducible.
    """
    ask_days = open_ask_days if isinstance(open_ask_days, int) and open_ask_days >= 1 else _MEMORY_CADENCE_DEFAULTS["open_ask_days"]
    scopes = [dict(scope) for scope in allowed_scopes]
    candidates: list[tuple[str, str, dict[str, Any], dict[str, object]]] = []
    for record in records:
        if record.get("schema_version") not in {PROJECT_MEMORY_RECORD_SCHEMA_VERSION, PRINCIPAL_PROJECT_MEMORY_RECORD_SCHEMA_VERSION}:
            continue
        if record.get("scope") not in scopes:
            continue
        record_id = str(record.get("record_id", "") or "")
        if not record_id:
            continue
        if eligible_record_ids is not None and record_id not in eligible_record_ids:
            continue
        verdict = _record_staleness(record, now=now)
        if str(verdict.get("state", "")) != "open":
            continue
        if evaluate_renderable_strings({"summary": str(record.get("summary", "") or "")}).get("status") != "safe":
            continue
        asked = ledger.get(record_id, {})
        asked_at = _parse_utc_naive_as_utc(str(asked.get("asked_at", "") or ""))
        if asked_at is not None and asked_at + timedelta(days=ask_days) > now:
            continue
        staleness = record.get("staleness") if isinstance(record.get("staleness"), dict) else {}
        open_since = str(staleness.get("open_since", "") or "")
        candidates.append((open_since, record_id, record, verdict))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    open_since, record_id, record, verdict = candidates[0]
    asked = ledger.get(record_id, {})
    count = asked.get("asked_count", 0)
    return {
        "record_id": record_id,
        # The bounded, redacted projection the pack already uses for a summary.
        "summary": _redact_admitted_text(str(record.get("summary", "") or ""))[:RECORD_SUMMARY_LIMIT_CHARS],
        "open_days": int(verdict.get("open_days", 0) or 0),
        "open_since": open_since,
        "asked_count": count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else 0,
    }


def render_open_reminder(reminder: dict[str, object]) -> str:
    """Exactly one line. It asks; it decides nothing."""
    record_id = str(reminder.get("record_id", ""))
    summary = str(reminder.get("summary", "")).replace("\n", " ").replace('"', "'")
    days = int(reminder.get("open_days", 0) or 0)
    return (
        f'omh reminder: "{summary}" ({record_id}) has been unresolved for {days} days — '
        "resolved, still open, or drop it? Answer with "
        f"omh memory confirm {record_id} · omh memory keep-open {record_id} · omh memory retire {record_id}"
    )


def _read_ledger(path: Path) -> dict[str, dict[str, object]]:
    try:
        if path.is_symlink() or not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("schema_version") != MEMORY_OPEN_REMINDERS_SCHEMA_VERSION:
        return {}
    entries = data.get("records")
    if not isinstance(entries, dict):
        return {}
    ledger: dict[str, dict[str, object]] = {}
    for record_id, entry in entries.items():
        if not isinstance(entry, dict) or not str(record_id):
            continue
        asked_at = str(entry.get("asked_at", "") or "")
        if _parse_utc_naive_as_utc(asked_at) is None:
            continue
        count = entry.get("asked_count", 0)
        ledger[str(record_id)] = {
            "asked_at": asked_at,
            "asked_count": count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else 0,
        }
    return ledger


__all__ = [
    "MEMORY_OPEN_REMINDERS_SCHEMA_VERSION",
    "OPEN_REMINDERS_FILENAME",
    "mark_open_reminder_asked",
    "open_reminders_path",
    "read_open_reminders",
    "render_open_reminder",
    "select_open_reminder",
]
