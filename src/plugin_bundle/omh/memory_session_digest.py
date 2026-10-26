"""What OMH memory changed between two sessions, counted for the next session's first pack.

The user never saw what OMH memory did between sessions. Captures, demotions,
corrections and retirements each announce themselves once, in the turn that
made them, and nothing said "since last time" when the next conversation
opened. This module counts it, for one bounded ``<memory_digest>`` section the
provider serves in the first pack of a new session.

Every count is read from a structured field the writers already persist, never
from message or record text:

- ``remembered``: admitted records whose ``created_at`` falls in the window,
  other than the two kinds below.
- ``hermes_entries_copied``: records whose ``source`` is ``hermes_demotion``
  (``omh_memory`` demote, or the operator's staged demotion) and whose
  ``source_ref`` digest matches the record's own summary, so OMH holds the
  entry intact. An altered copy is not counted at all.
- ``hermes_entries_freed``: write-journal lines whose ``action`` is ``remove``
  and ``removed_entry_demoted`` is true: Hermes removed an entry OMH holds.
- ``corrected``: ``hermes_model`` records with a non-empty ``derived_from``.
  Only ``omh_memory`` correct writes one; a model capture carries none.
- ``retired``: retirement-journal lines whose ``retired_at`` falls in the
  window, whoever retired the record.

Only live records are read, so a record created and then corrected or retired
inside the same window counts as the correction or the retirement, not also as
remembered. On a shared surface only records the session's own recall lens
admits are counted, and the three counts that carry no scope (copies, frees,
retirements) are not taken at all.

The window is ``[since, until)``. ``until`` is the session start, floored to its
second; ``since`` is the later of the last recorded session close in this OMH
home (else the last served prefetch) and the end of the last digest served,
never more than seven days back. Both marks only move forward. Session marks or
a receipt that exist but cannot be read mean no digest: an unknown start is
never widened to the cap.

``omh quality-evidence memory`` (``src/quality/memory_evidence.py``) imports
``stamp_epoch`` and ``is_freeing_remove`` from here, so the report and the
digest parse stamps and recognise a freeing remove the same way by construction.

Stdlib only; vendored in the bundle because it runs inside the Hermes process.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

from .hermes_memory import demotion_origin_ref
from .memory_admission import MODEL_CAPTURE_SOURCE
from .memory_prefetch_receipt import MAX_PREFETCH_RECEIPT_BYTES, prefetch_receipt_path
from .memory_state_files import state_file_lock, write_text_atomic
from .memory_store_io import scan_project_memory_records
from .iso_datetime import strict_fromisoformat

DIGEST_WINDOW_CAP_DAYS = 7
SESSION_CLOSE_SCHEMA_VERSION = "omh_memory_session_close/v1"
SESSION_CLOSE_FILENAME = "session_close.json"
# The two forward-only marks the session file holds.
CLOSED_MARK = "last_session_closed_at"
REPORTED_MARK = "reported_through"
# A mark further ahead of the clock than this is ignored, as if absent.
MARK_FUTURE_SKEW_SECONDS = 300
# The `source` both demotion writers stamp (`memory_tool._demote_entry`,
# `workflows.memory` staged demotion).
DEMOTION_SOURCE = "hermes_demotion"
DIGEST_COUNT_KEYS = ("remembered", "hermes_entries_copied", "hermes_entries_freed", "corrected", "retired")
# Counts a shared surface keeps: the only two that come from records its lens can judge.
SHARED_COUNT_KEYS = ("remembered", "corrected")
# Five integers, one timestamp, one flag and a fixed sentence: far under this
# for any count a store can reach. A section that would exceed it is not served.
DIGEST_RENDER_BUDGET_CHARS = 512
DIGEST_INSTRUCTION = (
    "OMH memory changes in this OMH home since your last session. If they bear on the user's message, "
    "mention them in one short line at most; otherwise leave them out."
)
SHARED_DIGEST_INSTRUCTION = (
    "Changes since the last session to OMH memory this conversation can recall. If they bear on the user's "
    "message, mention them in one short line at most; otherwise leave them out."
)
_RECEIPT_SCHEMA_PREFIX = "omh_memory_prefetch_receipt/"


def session_close_path(omh_home: str | Path) -> Path:
    return Path(omh_home).expanduser() / "memory" / SESSION_CLOSE_FILENAME


def write_session_close(omh_home: str | Path, *, at: datetime) -> None:
    """Record that a session in this OMH home closed. Atomic, under the state lock; raises ``OSError``.

    The stamp keeps microseconds: record and journal stamps are whole seconds
    (floored), so every write made before this close sorts before it.
    """
    _advance_session_mark(omh_home, CLOSED_MARK, at)


def write_reported_through(omh_home: str | Path, *, until: datetime) -> None:
    """Record that a served digest reported everything before ``until``. Atomic, under the state lock.

    The next window never starts before it, so a row already reported does not
    come back after a session that ended without a close, or one that ran
    alongside; what that session wrote after ``until`` is still counted.
    """
    _advance_session_mark(omh_home, REPORTED_MARK, until)


def _advance_session_mark(omh_home: str | Path, mark: str, at: datetime) -> None:
    """Move one mark forward, never back: a clock stepped back, or a ``~/.omh``
    shared by two machines, must not widen the next window. An unreadable file
    is replaced, since nothing in it could be kept, and so is a mark further
    ahead than ``MARK_FUTURE_SKEW_SECONDS``, which no reader honours either."""
    path = session_close_path(omh_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    with state_file_lock(path):
        state, marks = _read_session_marks(path)
        kept = {name: raw for name, (_epoch, raw) in marks.items()} if state == "ok" else {}
        # Ahead of the clock, not of `at`: an older value written later must not displace a newer mark.
        current = _usable_mark(marks.get(mark), now=datetime.now(timezone.utc))
        if current is None or at.timestamp() > current:
            kept[mark] = stamp
        write_text_atomic(path, json.dumps({"schema_version": SESSION_CLOSE_SCHEMA_VERSION, **kept}, sort_keys=True))


def _usable_mark(mark: tuple[float, str] | None, *, now: datetime) -> float | None:
    """The mark's epoch, or None when absent or further in the future than the skew allows.

    One stamp far ahead -- a machine with a wrong clock sharing this home, a
    hand edit -- would otherwise silence every digest until that date; it is
    ignored instead, and the window falls back to the close or the receipt.
    """
    if mark is None or mark[0] > now.timestamp() + MARK_FUTURE_SKEW_SECONDS:
        return None
    return mark[0]


def clamp_to_reported(omh_home: str | Path, since: datetime, *, now: datetime) -> datetime | None:
    """``since`` raised to the reported-through mark as it stands now; None when the marks cannot be read.

    A window is fixed at its session start, but another session may serve its
    digest first and record a later mark. Read again at render time, the mark
    keeps that session's rows from being reported twice. Only this mark: a
    newer close would also hide rows no digest ever showed.
    """
    state, marks = _read_session_marks(session_close_path(omh_home))
    if state == "invalid":
        return None
    reported = _usable_mark(marks.get(REPORTED_MARK), now=now)
    return since if reported is None or reported <= since.timestamp() else _datetime(reported)


def digest_window_start(omh_home: str | Path, *, until: datetime, initializing_session: str = "") -> datetime | None:
    """Where the window starts, or None when no digest may be served.

    The later of the last recorded close and the reported-through mark; with
    no close ever recorded, the last served pack stands in for it; never more
    than seven days back. None when the session marks or the prefetch receipt
    exist but cannot be read, so an unknown start never widens to the cap; and,
    at an ``initialize``, when the receipt's last serve was this same session
    -- a gateway rebuilding the agent for a conversation already under way.

    Record, journal and receipt stamps are whole seconds (floored). The close
    stamp keeps microseconds, so every write made before a close sorts before
    it. The receipt's ``served_at`` does not: on that fallback a write made in
    the same second as the last served pack, before or after it, is inside the
    window. The reported-through mark is a window end, a whole second the last
    digest excluded. The start is inclusive. A mark more than
    ``MARK_FUTURE_SKEW_SECONDS`` ahead of ``until`` is ignored.
    """
    marks_state, marks = _read_session_marks(session_close_path(omh_home))
    receipt_state, receipt = _read_receipt_marks(prefetch_receipt_path(omh_home))
    if marks_state == "invalid" or receipt_state == "invalid":
        return None
    if initializing_session and receipt is not None and receipt["session_id"] == initializing_session:
        return None
    cap = until.timestamp() - DIGEST_WINDOW_CAP_DAYS * 86400
    closed = _usable_mark(marks.get(CLOSED_MARK), now=until)
    if closed is not None:
        start = closed
    elif receipt is not None:
        start = receipt["served_at"]
    else:
        start = cap
    reported = _usable_mark(marks.get(REPORTED_MARK), now=until)
    return _datetime(max(start, reported if reported is not None else cap, cap))


def session_digest_counts(
    record_homes: Iterable[Path],
    omh_home: Path,
    *,
    since: datetime,
    until: datetime,
    journal_limit: int,
    in_lens: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[dict[str, int], bool]:
    """The counts over ``[since, until)`` and whether the write journal was at its cap.

    A source that cannot be read adds nothing. ``in_lens`` is a shared
    surface's recall gate: only records it admits are counted, and copies,
    frees and retirements are not counted at all. At its cap the write journal
    has dropped its oldest lines, so ``hermes_entries_freed`` may be short.
    """
    low, high = since.timestamp(), until.timestamp()
    keys = SHARED_COUNT_KEYS if in_lens is not None else DIGEST_COUNT_KEYS
    counts = dict.fromkeys(keys, 0)
    seen_records: set[str] = set()
    seen_retirements: set[tuple[str, str]] = set()
    for home in dict.fromkeys(Path(home) for home in record_homes):
        for record in _records(home):
            record_id = str(record.get("record_id", ""))
            if record_id in seen_records:
                continue
            seen_records.add(record_id)
            if not _in_window(record.get("created_at"), low, high) or (in_lens is not None and not in_lens(record)):
                continue
            kind = _record_kind(record)
            if kind in counts:
                counts[kind] += 1
        if in_lens is not None:
            continue
        for entry in _journal_entries(home / "memory" / "archive" / "retirements.jsonl"):
            key = (str(entry.get("record_id", "")), str(entry.get("retired_at", "")))
            if key in seen_retirements or not _in_window(entry.get("retired_at"), low, high):
                continue
            seen_retirements.add(key)
            counts["retired"] += 1
    if in_lens is not None:
        return counts, False
    journal_path = Path(omh_home) / "memory" / "write_journal.jsonl"
    counts["hermes_entries_freed"] = sum(
        1 for entry in _journal_entries(journal_path) if is_freeing_remove(entry) and _in_window(entry.get("observed_at"), low, high)
    )
    # Every kept line counts toward the cap, parsed or not, as the report counts it.
    return counts, len(_journal_lines(journal_path)) >= journal_limit


def render_session_digest(
    counts: dict[str, int],
    *,
    since: datetime,
    shared: bool = False,
    journal_at_cap: bool = False,
) -> str:
    """One ``<memory_digest>`` section naming each nonzero count; "" when every count is zero.

    A shared surface's section names no ``since`` and carries its own sentence.
    """
    shown = [(key, counts.get(key, 0)) for key in DIGEST_COUNT_KEYS if counts.get(key, 0) > 0]
    if not shown:
        return ""
    attributes = [] if shared else [f'since="{since.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")}"']
    attributes.extend(f'{key}="{int(value)}"' for key, value in shown)
    if journal_at_cap and not shared:
        attributes.append('journal_at_cap="true"')
    sentence = SHARED_DIGEST_INSTRUCTION if shared else DIGEST_INSTRUCTION
    text = f"<memory_digest {' '.join(attributes)}>\n{sentence}\n</memory_digest>"
    return text if len(text) <= DIGEST_RENDER_BUDGET_CHARS else ""


def is_freeing_remove(entry: dict[str, Any]) -> bool:
    """A native remove of an entry OMH holds intact (the report's ``removed_entry_demoted.true``)."""
    return str(entry.get("action", "") or "").strip().lower() == "remove" and entry.get("removed_entry_demoted") is True


def stamp_epoch(value: Any) -> float | None:
    """Epoch seconds from a Hermes or OMH stamp, or None when it is not one.

    Hermes stores REAL epoch seconds; older rows, fixtures and OMH's own files
    carry ISO-8601 text. A non-finite or unrepresentable number -- ``nan``, a
    400-digit integer, a date past the platform's range -- is not a stamp in
    either form. ``quality.hermes_state.hermes_epoch`` is this function.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError:
            return None
        return number if math.isfinite(number) else None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        number = float(text)
    except ValueError:
        pass
    else:
        return number if math.isfinite(number) else None
    try:
        parsed = strict_fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (OverflowError, ValueError, OSError):
        return None


def _datetime(epoch: float) -> datetime | None:
    """The UTC moment for ``epoch``; None when the platform cannot represent it."""
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (OverflowError, ValueError, OSError):
        return None


def _read_session_marks(path: Path) -> tuple[str, dict[str, tuple[float, str]]]:
    """``("absent", {})``, ``("invalid", {})``, or ``("ok", {mark: (epoch, stamp)})``.

    Each mark is optional, but one that is present must be a stamp, and a file
    holding neither is not one this module wrote.
    """
    data = _read_json_object(path, max_bytes=4096)
    if data is None:
        return "absent", {}
    if data is False or data.get("schema_version") != SESSION_CLOSE_SCHEMA_VERSION:
        return "invalid", {}
    marks: dict[str, tuple[float, str]] = {}
    for mark in (CLOSED_MARK, REPORTED_MARK):
        if mark not in data:
            continue
        raw = data[mark]
        epoch = stamp_epoch(raw) if isinstance(raw, str) else None
        if epoch is None:
            return "invalid", {}
        marks[mark] = (epoch, raw)
    return ("ok", marks) if marks else ("invalid", {})


def _read_receipt_marks(path: Path) -> tuple[str, dict[str, Any] | None]:
    """The receipt's ``served_at`` and ``session_id`` read raw, so a receipt an older OMH wrote still counts.

    The full validation (`read_prefetch_receipt`) rejects a receipt from any
    other release; only these three fields are needed here, and none of them
    is covered by the receipt's own digest anyway.
    """
    data = _read_json_object(path, max_bytes=MAX_PREFETCH_RECEIPT_BYTES)
    if data is None:
        return "absent", None
    if data is False or not str(data.get("schema_version", "")).startswith(_RECEIPT_SCHEMA_PREFIX):
        return "invalid", None
    served_at = stamp_epoch(data.get("served_at"))
    session_id = data.get("session_id", "")
    if data.get("state") != "returned_to_host" or served_at is None or _datetime(served_at) is None or not isinstance(session_id, str):
        return "invalid", None
    return "ok", {"served_at": served_at, "session_id": session_id}


def _read_json_object(path: Path, *, max_bytes: int) -> dict[str, Any] | bool | None:
    """The file's JSON object; None when absent; False when present but unusable."""
    try:
        if not path.exists() and not path.is_symlink():
            return None
        if path.is_symlink() or not path.is_file():
            return False
        with path.open("rb") as handle:
            encoded = handle.read(max_bytes + 1)
    except OSError:
        return False
    if len(encoded) > max_bytes:
        return False
    try:
        data = json.loads(encoded)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return False
    return data if isinstance(data, dict) else False


def _record_kind(record: dict[str, Any]) -> str:
    source = str(record.get("source", "") or "")
    if source == DEMOTION_SOURCE:
        return "hermes_entries_copied" if _holds_intact_copy(record) else ""
    derived = record.get("derived_from")
    if source == MODEL_CAPTURE_SOURCE and isinstance(derived, list) and derived:
        return "corrected"
    return "remembered"


def _holds_intact_copy(record: dict[str, Any]) -> bool:
    """The record's ``source_ref`` names the digest of its own summary: the entry came over unaltered."""
    source_ref = str(record.get("source_ref", "") or "")
    summary = str(record.get("summary", "") or "")
    label = source_ref.removeprefix("hermes:").split("#", 1)[0]
    return bool(summary) and source_ref == demotion_origin_ref(label, summary)


def _records(home: Path) -> list[dict[str, Any]]:
    try:
        records, _unreadable = scan_project_memory_records(home)
    except OSError:
        return []
    return records


def _journal_lines(path: Path) -> list[str]:
    """The non-empty lines; an absent or unreadable journal has none."""
    try:
        if path.is_symlink() or not path.is_file():
            return []
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return [line for line in text.splitlines() if line.strip()]


def _journal_entries(path: Path) -> list[dict[str, Any]]:
    """Each JSON-object line; an absent or unreadable journal is empty."""
    entries = []
    for line in _journal_lines(path):
        try:
            entry = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _in_window(stamp: Any, low: float, high: float) -> bool:
    epoch = stamp_epoch(stamp)
    return epoch is not None and low <= epoch < high


__all__ = [
    "DIGEST_COUNT_KEYS",
    "DIGEST_RENDER_BUDGET_CHARS",
    "DIGEST_WINDOW_CAP_DAYS",
    "MARK_FUTURE_SKEW_SECONDS",
    "SESSION_CLOSE_SCHEMA_VERSION",
    "SHARED_COUNT_KEYS",
    "clamp_to_reported",
    "digest_window_start",
    "is_freeing_remove",
    "render_session_digest",
    "session_close_path",
    "session_digest_counts",
    "stamp_epoch",
    "write_reported_through",
    "write_session_close",
]
