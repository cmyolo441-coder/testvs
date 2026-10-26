"""Enumerate reviewed OMH memory records by the source they were admitted from.

Recovery from a source that turned out to be wrong starts with one question
the rest of `omh memory` could not answer: which records came from it?
`lineage` traverses a record id, `retire`/`prune`/`correct` act on a record id,
and the recall report that closes the recovery names record ids -- every one
of them needs the affected set to already exist. This module produces that set
as a read.

Two axes carry a record's origin, and a record can carry both:

- ``source`` -- the admission channel recorded at capture (``cli``, a wrapper
  label, an importer name).
- ``source_ref`` -- the thing the memory was taken from: a path, a URL, a
  ticket. When a document turns out to be wrong, this is the axis that names
  it.

``source_class`` is deliberately not an axis. It is a governance
classification with four fixed values, and every directly captured record is
``omh_local``, so indexing it would create one label matching most of the
store and tell an operator nothing about where a record came from.

Membership is per axis, never first-match: a record whose ``source`` and
``source_ref`` both name the queried value reports both axes, and a record
whose two axes carry different values is listed under both labels. Anything
else would quietly halve a blast radius.

A record that recorded no source at all is indeterminate, never clean. This
selector cannot exclude it, and the report says so rather than leaving it out
of both the matched set and the operator's attention.
"""

from __future__ import annotations

from typing import Any

from ..plugin_bundle.omh.memory_recall_support import (
    _normalize_tags,
    _redact_admitted_text,
    _redacted_metadata_label,
    _string_list,
)
from ..system.paths import OmhPaths
from .memory import _redacted_scope, scan_project_memory_records

MEMORY_SOURCE_INDEX_SCHEMA_VERSION = "memory_source_index/v1"

# The record fields that name where a record came from, in report order.
SOURCE_AXES = ("source", "source_ref")

# Why the selected set is empty. Each is a different answer to the operator's
# question and none of them may render as any of the others.
SELECTOR_OUTCOMES = (
    "not_requested",
    "matched",
    "no_records_for_source",
    "source_never_recorded",
    "empty_store",
)

_CLAIM_BOUNDARY = (
    "A source index enumerates OMH-local reviewed records by the source recorded on them. "
    "It reads the store and changes nothing: it quarantines, retires, and deletes nothing. "
    "It is prepared context, not execution, review, CI, merge, or Hermes internal-memory evidence."
)
_INDETERMINATE_BOUNDARY = (
    "A record with no recorded source cannot be excluded by this selector. It is indeterminate, "
    "never clean, and stays the operator's to judge; an unreadable record file is the same answer "
    "for a different reason."
)
_RECOVERY_SEQUENCE = (
    "omh memory lineage <record-id>",
    "omh memory retire <record-id> --apply",
    "omh memory prune <record-id> --revision <n> --apply --confirm-hard-delete-local",
    'omh memory recall "<the claim those records answered>"',
)
_RECOVERY_QUARANTINE = (
    "Quarantine is the default: retire moves the revision into the local archive and is reversible. "
    "prune hard-deletes the manifest-declared local target set and is an explicit second step."
)
_RECOVERY_COMPLETION = (
    "A recall report over this store showing the records absent: `omh memory recall`, or "
    "`omh memory recall-incident --record-id <id>` for one record's stored/eligible/selected stages."
)
_RECOVERY_NOT_COMPLETION = (
    "The retire command's exit code, because a sweep that archived nothing also exits 0. And "
    "`omh memory recall-suite`, which seeds its own fixture corpus in a temporary store: it proves "
    "the recall engine did not regress while you worked, never that this store's records left recall."
)


def build_memory_source_index(
    paths: OmhPaths, *, source: str | None = None, limit: int | None = None
) -> dict[str, object]:
    """Index the store by source, and select one source's records when asked.

    Report only. With no ``source`` the report is the label index an operator
    picks from; with one it also carries that source's blast radius. Both
    forms always carry the indeterminate set, because a selector that names
    what it matched without naming what it could not judge reads as complete
    when it is not.
    """
    selector = str(source or "").strip()
    records, unreadable = scan_project_memory_records(paths)
    labels: dict[str, dict[str, Any]] = {}
    indeterminate: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    axis_values: dict[str, set[str]] = {axis: set() for axis in SOURCE_AXES}
    for record in records:
        sources = _record_sources(record)
        if not sources:
            indeterminate.append(
                {
                    "record_id": _redacted_metadata_label(record.get("record_id", "")),
                    "reason": "source_not_recorded",
                }
            )
            continue
        for entry in sources:
            axis, value = str(entry["axis"]), str(entry["value"])
            axis_values[axis].add(value)
            bucket = labels.setdefault(value, {"axes": set(), "record_ids": []})
            bucket["axes"].add(axis)
            # One record reaching a label on two axes is still one record.
            if not bucket["record_ids"] or bucket["record_ids"][-1] != record.get("record_id"):
                bucket["record_ids"].append(record.get("record_id"))
        if selector:
            matched = [str(entry["axis"]) for entry in sources if str(entry["value"]) == selector]
            if matched:
                selected.append(_source_card(record, sources, matched_axes=matched))
    return {
        "schema_version": MEMORY_SOURCE_INDEX_SCHEMA_VERSION,
        "record_count": len(records),
        "axes": [
            {
                "axis": axis,
                "populated": bool(axis_values[axis]),
                "distinct_values": len(axis_values[axis]),
            }
            for axis in SOURCE_AXES
        ],
        "sources": [
            {
                "label": _redact_admitted_text(label),
                "axes": sorted(bucket["axes"]),
                "record_count": len(bucket["record_ids"]),
            }
            for label, bucket in sorted(labels.items())
        ],
        "source_count": len(labels),
        "selector": {"requested": bool(selector), "value": _redact_admitted_text(selector)},
        "selected": {
            "outcome": _selector_outcome(selector, selected, records, labels),
            "count": len(selected),
            "records": _limited(selected, limit),
            "truncated": _truncated(selected, limit),
        },
        "indeterminate": {
            "records": indeterminate,
            "unreadable": list(unreadable),
            "count": len(indeterminate) + len(unreadable),
            "boundary": _INDETERMINATE_BOUNDARY,
        },
        "recovery": {
            "sequence": list(_RECOVERY_SEQUENCE),
            "quarantine_default": _RECOVERY_QUARANTINE,
            "completion_evidence": _RECOVERY_COMPLETION,
            "not_completion_evidence": _RECOVERY_NOT_COMPLETION,
        },
        "redaction_policy": "metadata_only",
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def _record_sources(record: dict[str, Any]) -> list[dict[str, str]]:
    """Every source this record carries, one entry per populated axis."""
    entries: list[dict[str, str]] = []
    for axis in SOURCE_AXES:
        value = str(record.get(axis, "") or "").strip()
        if value:
            entries.append({"axis": axis, "value": value})
    return entries


def _selector_outcome(
    selector: str,
    selected: list[dict[str, object]],
    records: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
) -> str:
    """Why the selected set looks the way it does.

    An empty set has three causes and they are three different answers. A
    store with nothing readable in it, a store whose records never recorded a
    source -- where this selector can exclude nothing and the whole store is
    indeterminate -- and a store that does record sources and simply never saw
    this one. Rendering them alike would let the third reassure an operator
    who is actually in the second.
    """
    if not selector:
        return "not_requested"
    if selected:
        return "matched"
    if not records:
        return "empty_store"
    if not labels:
        return "source_never_recorded"
    return "no_records_for_source"


def _source_card(
    record: dict[str, Any], sources: list[dict[str, str]], *, matched_axes: list[str]
) -> dict[str, object]:
    attention = record.get("attention") if isinstance(record.get("attention"), dict) else {}
    retention = record.get("retention") if isinstance(record.get("retention"), dict) else {}
    revision = record.get("revision")
    return {
        "record_id": _redacted_metadata_label(record.get("record_id", "")),
        "revision": revision if isinstance(revision, int) and not isinstance(revision, bool) else None,
        "record_type": _redact_admitted_text(str(record.get("record_type", ""))),
        "summary": _redact_admitted_text(str(record.get("summary", "")))[:500],
        "scope": _redacted_scope(record.get("scope", {})),
        "tags": _normalize_tags(record.get("tags", [])),
        "approved_at": _redact_admitted_text(str(record.get("approved_at", ""))),
        "attention_tier": _redact_admitted_text(str(attention.get("tier", ""))),
        "retention_class": _redact_admitted_text(str(retention.get("class", ""))),
        "sources": [
            {"axis": entry["axis"], "value": _redact_admitted_text(entry["value"])} for entry in sources
        ],
        "matched_axes": sorted(matched_axes),
        "derived_from": [
            _redacted_metadata_label(ref) for ref in _string_list(record.get("derived_from", []))
        ],
    }


def _limited(items: list[dict[str, object]], limit: int | None) -> list[dict[str, object]]:
    if limit is None:
        return items
    return items[: max(limit, 0)]


def _truncated(items: list[dict[str, object]], limit: int | None) -> bool:
    return limit is not None and len(items) > max(limit, 0)
