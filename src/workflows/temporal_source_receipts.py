"""Point-in-time web evidence receipts (`temporal_source_receipt/v1`, issue #1403).

A research answer about "what was known as of a date" is only as good as the
proof that a page held its cited text at or before that date. Three clocks are
involved, and none of them stands in for another:

- **capture time** -- when an archive or capture provider observed the page.
  It is the only clock that can bound a historical claim, and it is always
  provider-reported: a receipt attributes it, it never authenticates it.
- **publication time** -- what the page says about itself. A self-reported
  date proves neither that the page existed nor that it was unchanged before
  the cutoff, so it never appears on a receipt at all.
- **retrieval time** -- when Hermes or a wrapper fetched the capture or the
  live page. It dates the observation, not the content.

A receipt binds one cited source to one requested `as_of` date or interval and
says which of two evidence surfaces the cited text came from: a historical
capture, or the live page. The two are typed apart on purpose. A live page is
valid *current* evidence and never historical evidence, whatever its
publication date says, so a receipt that presents one as historical is refused
rather than merely ineligible.

Eligibility is one predicate, `receipt_supports_as_of_claim`: an available
historical capture whose provider-reported capture time is at or before the
cutoff and which carries a stable capture identifier or content digest. A
receipt that fails it is still a valid record -- an unavailable capture, an
exhausted paid provider, or a capture whose time the provider did not report
are all facts worth keeping -- but the claim it would back moves to the
unresolved annex as a `temporal_retrieval_gap/v1`. Nothing here reaches the
network: the gap states that no retrieval was performed, and a missing archive
or spent provider authority is reported, never worked around.

The contract is provider-neutral. `capture_provider` is an opaque identity an
installed native tool or an external archive connector supplies; OMH installs
nothing, infers no credentials, and treats a receipt as prepared research
context rather than proof that anything was fetched.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, time, timezone
from typing import Any

from ..system.metadata_safety import is_sensitive_metadata_text, require_opaque_metadata_ref
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


TEMPORAL_SOURCE_RECEIPT_SCHEMA_VERSION = "temporal_source_receipt/v1"
TEMPORAL_RETRIEVAL_GAP_SCHEMA_VERSION = "temporal_retrieval_gap/v1"
TEMPORAL_EVIDENCE_SURFACES_SCHEMA_VERSION = "temporal_evidence_surfaces/v1"

RECEIPT_CLAIM_BOUNDARY = (
    "A temporal source receipt binds one cited source to one requested as-of cutoff. It attributes a "
    "provider-reported capture time; it does not authenticate that time, prove the page was true, or "
    "prove that any retrieval was executed."
)
GAP_CLAIM_BOUNDARY = (
    "A temporal retrieval gap records that no eligible capture backs a historical claim. It performed no "
    "retrieval, installed no provider, and is not evidence about the page at any time."
)
SURFACES_CLAIM_BOUNDARY = (
    "Historical captures and live evidence are separate surfaces for one as-of question. Neither is "
    "execution, review, CI, or merge evidence, and live evidence never answers a historical claim."
)

#: `as_of` is a single calendar date or a closed interval; the cutoff is the
#: date, or the interval's end.
AS_OF_KINDS = ("date", "interval")

#: Same axis the research briefing uses for a cited source.
SOURCE_CLASSES = ("upstream_official", "practitioner", "unattributed")

#: The two evidence surfaces a cited text can come from.
EVIDENCE_KINDS = ("historical_capture", "live_page")

#: Who supplied the capture. `none` is the live page: nothing captured it.
CAPTURE_PROVIDER_CLASSES = ("archive_service", "native_tool", "external_connector", "none")

#: A capture time is always the provider's word. `unknown` means the provider
#: reported none, and a receipt never fills that in from another clock.
CAPTURED_AT_ATTRIBUTIONS = ("provider_reported", "unknown")

CUTOFF_RELATIONS = ("at_or_before", "after", "unknown")

#: `provider_authority_exhausted` is the paid or credentialed provider whose
#: budget, quota, or authority gate refused the retrieval; `not_attempted` is
#: an archive that was never reachable from this run at all.
AVAILABILITIES = (
    "available",
    "unavailable",
    "access_denied",
    "provider_authority_exhausted",
    "not_attempted",
)

#: Why a claim went to the unresolved annex instead of being asserted.
GAP_KINDS = (
    "no_eligible_capture",
    "capture_after_cutoff",
    "capture_time_unknown",
    "capture_unverifiable",
    "capture_unavailable",
    "archive_access_unavailable",
    "provider_authority_exhausted",
    "live_page_only",
)

TEMPORAL_SOURCE_RECEIPT_KEYS = (
    "as_of",
    "availability",
    "capture_provider",
    "capture_provider_class",
    "capture_ref",
    "captured_at",
    "captured_at_attribution",
    "claim_boundary",
    "content_digest",
    "cutoff_relation",
    "evidence_kind",
    "failure_reason",
    "receipt_id",
    "retrieved_at",
    "schema_version",
    "source_class",
    "source_url",
)

_STRING_FIELDS = (
    "receipt_id",
    "source_url",
    "source_class",
    "evidence_kind",
    "retrieved_at",
    "capture_provider",
    "capture_provider_class",
    "captured_at",
    "captured_at_attribution",
    "capture_ref",
    "content_digest",
    "cutoff_relation",
    "availability",
    "failure_reason",
)

_VOCABULARIES = (
    ("source_class", SOURCE_CLASSES),
    ("evidence_kind", EVIDENCE_KINDS),
    ("capture_provider_class", CAPTURE_PROVIDER_CLASSES),
    ("captured_at_attribution", CAPTURED_AT_ATTRIBUTIONS),
    ("cutoff_relation", CUTOFF_RELATIONS),
    ("availability", AVAILABILITIES),
)

#: What identifies one observation of one source against one cutoff. The
#: retrieval stamp is excluded so re-recording the same capture does not mint
#: a second identity because the clock moved.
_IDENTITY_KEYS = (
    "as_of",
    "capture_provider",
    "capture_ref",
    "captured_at",
    "content_digest",
    "evidence_kind",
    "source_url",
)

MAX_SOURCE_URL_CHARS = 2048
MAX_FAILURE_REASON_CHARS = 200
MAX_GAP_DETAIL_CHARS = 200
_SHA256_HEX_CHARS = 64

_LABEL = "temporal_source_receipt"


class TemporalSourceReceiptError(ValueError):
    """Raised when a receipt or gap cannot be built from what was supplied."""


def as_of_date(date: str) -> dict[str, str]:
    """One calendar date, or the instant it names, as an `as_of` value."""
    return {"kind": "date", "date": str(date or "").strip()}


def as_of_interval(start: str, end: str) -> dict[str, str]:
    """A closed interval; the cutoff for a historical claim is its end."""
    return {"kind": "interval", "start": str(start or "").strip(), "end": str(end or "").strip()}


def as_of_errors(value: Any) -> list[str]:
    """Every reason an `as_of` value is not a date or a bounded interval."""
    if not isinstance(value, Mapping):
        return [f"{_LABEL} as_of must be a mapping"]
    kind = value.get("kind")
    if kind == "date":
        if set(value) != {"kind", "date"}:
            return [f"{_LABEL} as_of date carries exactly kind and date"]
        if _parse_stamp(value.get("date")) is None:
            return [f"{_LABEL} as_of date must be an ISO-8601 date or timestamp"]
        return []
    if kind == "interval":
        if set(value) != {"kind", "start", "end"}:
            return [f"{_LABEL} as_of interval carries exactly kind, start, and end"]
        start = _parse_stamp(value.get("start"))
        end = _parse_stamp(value.get("end"))
        if start is None or end is None:
            return [f"{_LABEL} as_of interval start and end must be ISO-8601 dates or timestamps"]
        if start > end:
            return [f"{_LABEL} as_of interval start must not be after its end"]
        return []
    return [f"{_LABEL} as_of kind must be one of {', '.join(AS_OF_KINDS)}"]


def as_of_cutoff(value: Mapping[str, Any]) -> datetime:
    """The last instant a capture may carry and still be at or before `as_of`.

    A calendar date means the whole of that day in UTC; a timestamp means that
    instant. Callers validate first; this raises on a malformed value.
    """
    errors = as_of_errors(value)
    if errors:
        raise TemporalSourceReceiptError(errors[0])
    raw = value["date"] if value["kind"] == "date" else value["end"]
    return _cutoff_instant(str(raw))


def cutoff_relation_for(captured_at: str, as_of: Mapping[str, Any]) -> str:
    """Where a provider-reported capture time sits relative to the cutoff.

    `unknown` when the provider reported no capture time. The relation is
    derived here and checked at validation, so a stored receipt cannot say
    `at_or_before` about a capture the timestamps put after the cutoff.
    """
    stamp = _parse_stamp(captured_at)
    if stamp is None:
        return "unknown"
    return "at_or_before" if stamp <= as_of_cutoff(as_of) else "after"


def build_temporal_source_receipt(
    *,
    as_of: Mapping[str, Any],
    source_url: str,
    source_class: str,
    evidence_kind: str,
    retrieved_at: str = "",
    capture_provider: str = "",
    capture_provider_class: str = "none",
    captured_at: str = "",
    capture_ref: str = "",
    content_digest: str = "",
    availability: str = "available",
    failure_reason: str = "",
) -> dict[str, Any]:
    """Mint one receipt from what a capture or live retrieval reported, or refuse.

    Nothing is inferred: the cutoff relation is computed from the supplied
    capture time and nothing else, an absent capture time stays absent, and a
    retrieval stamp is required rather than defaulted so a receipt never
    carries a time nobody observed.
    """
    captured = str(captured_at or "").strip()
    record = {
        "schema_version": TEMPORAL_SOURCE_RECEIPT_SCHEMA_VERSION,
        "receipt_id": "",
        "as_of": dict(as_of) if isinstance(as_of, Mapping) else as_of,
        "source_url": str(source_url or "").strip(),
        "source_class": str(source_class or ""),
        "evidence_kind": str(evidence_kind or ""),
        "retrieved_at": str(retrieved_at or "").strip(),
        "capture_provider": str(capture_provider or "").strip(),
        "capture_provider_class": str(capture_provider_class or ""),
        "captured_at": captured,
        "captured_at_attribution": "provider_reported" if captured else "unknown",
        "capture_ref": str(capture_ref or "").strip(),
        "content_digest": str(content_digest or "").strip().lower(),
        "cutoff_relation": "unknown",
        "availability": str(availability or ""),
        "failure_reason": str(failure_reason or "").strip(),
        "claim_boundary": RECEIPT_CLAIM_BOUNDARY,
    }
    if not as_of_errors(record["as_of"]) and record["evidence_kind"] == "historical_capture":
        record["cutoff_relation"] = cutoff_relation_for(captured, record["as_of"])
    record["receipt_id"] = temporal_source_receipt_id(record)
    errors = validate_temporal_source_receipt(record)
    if errors:
        raise TemporalSourceReceiptError(errors[0])
    return record


def temporal_source_receipt_id(record: Mapping[str, Any]) -> str:
    """Stable identity of one observation: a digest of what was observed."""
    return "tsr_" + temporal_source_receipt_fingerprint({key: record.get(key) for key in _IDENTITY_KEYS})[:24]


def temporal_source_receipt_fingerprint(record: Mapping[str, Any]) -> str:
    """Content digest a consumer can compare to prove it preserved a receipt."""
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_temporal_source_receipt(record: Any) -> list[str]:
    """Every contract violation, structural faults first."""
    if not isinstance(record, Mapping):
        return [f"{_LABEL} must be an object"]
    errors: list[str] = []
    extra = sorted(set(record) - set(TEMPORAL_SOURCE_RECEIPT_KEYS))
    if extra:
        errors.append(f"{_LABEL} has unsupported keys: {extra}")
    missing = sorted(set(TEMPORAL_SOURCE_RECEIPT_KEYS) - set(record))
    if missing:
        errors.append(f"{_LABEL} is missing keys: {missing}")
    if record.get("schema_version") != TEMPORAL_SOURCE_RECEIPT_SCHEMA_VERSION:
        errors.append(f"{_LABEL} schema_version must be {TEMPORAL_SOURCE_RECEIPT_SCHEMA_VERSION}")
    for key in _STRING_FIELDS:
        if not isinstance(record.get(key), str):
            errors.append(f"{_LABEL} {key} must be a string")
    if errors:
        return errors
    for key, allowed in _VOCABULARIES:
        if record[key] not in allowed:
            errors.append(f"{_LABEL} {key} must be one of {', '.join(allowed)}")
    errors.extend(as_of_errors(record["as_of"]))
    errors.extend(_source_url_errors(record["source_url"]))
    if record["claim_boundary"] != RECEIPT_CLAIM_BOUNDARY:
        errors.append(f"{_LABEL} claim_boundary must state the receipt boundary")
    if record["receipt_id"] != temporal_source_receipt_id(record):
        errors.append(f"{_LABEL} receipt_id does not match the observation it names")
    if errors:
        return errors
    errors.extend(_stamp_errors(record))
    errors.extend(_capture_errors(record))
    errors.extend(_availability_errors(record))
    return errors


def receipt_supports_as_of_claim(record: Mapping[str, Any]) -> bool:
    """Whether a receipt may back a claim about the page at or before the cutoff."""
    if validate_temporal_source_receipt(record):
        return False
    return as_of_claim_gap(record) is None


def as_of_claim_gap(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The gap a receipt leaves for a historical claim, or None when it backs one.

    Every ineligible receipt names one reason. The order is the order a reader
    needs: whether anything was retrieved, then which surface it came from,
    then where the capture sits against the cutoff, then whether it is stable
    enough to cite.
    """
    errors = validate_temporal_source_receipt(record)
    if errors:
        return build_temporal_retrieval_gap(
            as_of=record.get("as_of") if isinstance(record, Mapping) else None,
            source_url=str(record.get("source_url", "")) if isinstance(record, Mapping) else "",
            gap_kind="capture_unverifiable",
            detail=errors[0],
            receipt_ref=str(record.get("receipt_id", "")) if isinstance(record, Mapping) else "",
            strict=False,
        )
    kind = _availability_gap_kind(record["availability"])
    if kind is None and record["evidence_kind"] == "live_page":
        kind = "live_page_only"
    if kind is None and record["cutoff_relation"] == "after":
        kind = "capture_after_cutoff"
    if kind is None and record["cutoff_relation"] == "unknown":
        kind = "capture_time_unknown"
    if kind is None and not record["capture_ref"] and not record["content_digest"]:
        kind = "capture_unverifiable"
    if kind is None:
        return None
    return build_temporal_retrieval_gap(
        as_of=record["as_of"],
        source_url=record["source_url"],
        gap_kind=kind,
        detail=record["failure_reason"],
        receipt_ref=record["receipt_id"],
    )


def build_temporal_retrieval_gap(
    *,
    as_of: Any,
    source_url: str,
    gap_kind: str,
    detail: str = "",
    receipt_ref: str = "",
    strict: bool = True,
) -> dict[str, Any]:
    """One unresolved-annex entry: a historical claim OMH abstains from.

    `network_action` is a constant, not a report: the gap exists precisely
    because no retrieval closes it, and nothing here performs one.
    """
    if gap_kind not in GAP_KINDS:
        raise TemporalSourceReceiptError(f"{_LABEL} gap_kind must be one of {', '.join(GAP_KINDS)}")
    gap = {
        "schema_version": TEMPORAL_RETRIEVAL_GAP_SCHEMA_VERSION,
        "as_of": dict(as_of) if isinstance(as_of, Mapping) else as_of,
        "source_url": str(source_url or "").strip(),
        "gap_kind": gap_kind,
        "detail": _bounded_text(detail, MAX_GAP_DETAIL_CHARS),
        "receipt_ref": str(receipt_ref or ""),
        "resolution": "abstain",
        "network_action": "none",
        "claim_boundary": GAP_CLAIM_BOUNDARY,
    }
    if strict:
        errors = validate_temporal_retrieval_gap(gap)
        if errors:
            raise TemporalSourceReceiptError(errors[0])
    return gap


def validate_temporal_retrieval_gap(record: Any) -> list[str]:
    if not isinstance(record, Mapping):
        return ["temporal_retrieval_gap must be an object"]
    errors: list[str] = []
    if record.get("schema_version") != TEMPORAL_RETRIEVAL_GAP_SCHEMA_VERSION:
        errors.append(f"temporal_retrieval_gap schema_version must be {TEMPORAL_RETRIEVAL_GAP_SCHEMA_VERSION}")
    errors.extend(f"temporal_retrieval_gap {issue}" for issue in as_of_errors(record.get("as_of")))
    errors.extend(
        f"temporal_retrieval_gap {issue}" for issue in _source_url_errors(str(record.get("source_url", "")))
    )
    if record.get("gap_kind") not in GAP_KINDS:
        errors.append(f"temporal_retrieval_gap gap_kind must be one of {', '.join(GAP_KINDS)}")
    if record.get("resolution") != "abstain":
        errors.append("temporal_retrieval_gap resolution must be abstain")
    if record.get("network_action") != "none":
        errors.append("temporal_retrieval_gap network_action must be none; a gap performs no retrieval")
    if record.get("claim_boundary") != GAP_CLAIM_BOUNDARY:
        errors.append("temporal_retrieval_gap claim_boundary must state the gap boundary")
    return errors


def build_temporal_evidence_surfaces(
    *,
    as_of: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Split receipts for one as-of question into the two surfaces a reader sees.

    A then-versus-now answer needs both surfaces and must not flatten them:
    `historical_captures` are the receipts eligible to back a historical
    claim, `live_evidence` the available live pages that describe now, and
    `unresolved_annex` every claim neither surface can back. A receipt for a
    different cutoff is refused rather than silently re-dated.
    """
    errors = as_of_errors(as_of)
    if errors:
        raise TemporalSourceReceiptError(errors[0])
    requested = dict(as_of)
    historical: list[dict[str, Any]] = []
    live: list[dict[str, Any]] = []
    annex: list[dict[str, Any]] = []
    for receipt in receipts:
        preserved = preserve_temporal_source_receipt(receipt)
        if preserved["as_of"] != requested:
            raise TemporalSourceReceiptError(f"{_LABEL} {preserved['receipt_id']} answers a different as_of cutoff")
        gap = as_of_claim_gap(preserved)
        if gap is None:
            historical.append(preserved)
            continue
        if preserved["evidence_kind"] == "live_page" and preserved["availability"] == "available":
            live.append(preserved)
        annex.append(gap)
    return {
        "schema_version": TEMPORAL_EVIDENCE_SURFACES_SCHEMA_VERSION,
        "as_of": requested,
        "historical_captures": historical,
        "live_evidence": live,
        "unresolved_annex": annex,
        "claim_boundary": SURFACES_CLAIM_BOUNDARY,
    }


def preserve_temporal_source_receipt(record: Any) -> dict[str, Any]:
    """A validated copy for a consumer that carries the receipt without changing it.

    Consumers compare `temporal_source_receipt_fingerprint` of what they
    received and what they stored; equality is the proof of preservation.
    """
    errors = validate_temporal_source_receipt(record)
    if errors:
        raise TemporalSourceReceiptError(errors[0])
    copied = dict(record)
    copied["as_of"] = dict(record["as_of"])
    return copied


def _availability_gap_kind(availability: str) -> str | None:
    if availability == "available":
        return None
    if availability == "provider_authority_exhausted":
        return "provider_authority_exhausted"
    if availability in ("access_denied", "not_attempted"):
        return "archive_access_unavailable"
    return "capture_unavailable"


def _stamp_errors(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if record["retrieved_at"] and _parse_stamp(record["retrieved_at"]) is None:
        errors.append(f"{_LABEL} retrieved_at must be an ISO-8601 timestamp")
    if record["captured_at"] and _parse_stamp(record["captured_at"]) is None:
        errors.append(f"{_LABEL} captured_at must be an ISO-8601 timestamp")
    return errors


def _capture_errors(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    captured_at = record["captured_at"]
    attribution = record["captured_at_attribution"]
    if captured_at and attribution != "provider_reported":
        errors.append(f"{_LABEL} a captured_at is always provider_reported; nothing else authenticates it")
    if not captured_at and attribution != "unknown":
        errors.append(f"{_LABEL} captured_at_attribution must be unknown when no capture time was reported")
    digest = record["content_digest"]
    if digest and (len(digest) != _SHA256_HEX_CHARS or any(char not in "0123456789abcdef" for char in digest)):
        errors.append(f"{_LABEL} content_digest must be a lowercase sha256 hex digest")
    if record["capture_provider"]:
        try:
            require_opaque_metadata_ref(record["capture_provider"], field=f"{_LABEL} capture_provider")
        except ValueError as exc:
            errors.append(str(exc))
    if record["capture_ref"]:
        try:
            require_opaque_metadata_ref(record["capture_ref"], field=f"{_LABEL} capture_ref")
        except ValueError as exc:
            errors.append(str(exc))
    if record["evidence_kind"] == "live_page":
        if record["capture_provider_class"] != "none" or record["capture_provider"]:
            errors.append(f"{_LABEL} a live page has no capture provider")
        if captured_at or record["capture_ref"]:
            errors.append(f"{_LABEL} a live page presented with a capture time or capture id is not historical evidence")
        if record["cutoff_relation"] != "unknown":
            errors.append(f"{_LABEL} a live page has no cutoff relation; it is current evidence, never historical")
        return errors
    if record["capture_provider_class"] == "none" or not record["capture_provider"]:
        errors.append(f"{_LABEL} a historical capture names the provider that captured it")
    if captured_at and _parse_stamp(captured_at) is not None and not as_of_errors(record["as_of"]):
        expected = cutoff_relation_for(captured_at, record["as_of"])
        if record["cutoff_relation"] != expected:
            errors.append(f"{_LABEL} cutoff_relation must be {expected} for this captured_at and as_of")
    if not captured_at and record["cutoff_relation"] != "unknown":
        errors.append(f"{_LABEL} cutoff_relation {record['cutoff_relation']} requires a captured_at")
    return errors


def _availability_errors(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    reason = record["failure_reason"]
    if len(reason) > MAX_FAILURE_REASON_CHARS or is_sensitive_metadata_text(reason) or "\n" in reason:
        errors.append(f"{_LABEL} failure_reason must be one bounded line without secrets")
    if record["availability"] == "available":
        if not record["retrieved_at"]:
            errors.append(f"{_LABEL} an available receipt records when it was retrieved")
        if reason:
            errors.append(f"{_LABEL} an available receipt carries no failure_reason")
        return errors
    if not reason:
        errors.append(f"{_LABEL} availability {record['availability']} names its failure_reason")
    return errors


def _source_url_errors(value: str) -> list[str]:
    if not value:
        return [f"{_LABEL} source_url is required"]
    if len(value) > MAX_SOURCE_URL_CHARS:
        return [f"{_LABEL} source_url exceeds {MAX_SOURCE_URL_CHARS} characters"]
    if not (value.startswith("https://") or value.startswith("http://")):
        return [f"{_LABEL} source_url must be a canonical http(s) URL"]
    if any(char.isspace() or ord(char) < 32 for char in value):
        return [f"{_LABEL} source_url must not contain whitespace or control characters"]
    authority = value.split("://", 1)[1].split("/", 1)[0]
    if "@" in authority or not authority:
        return [f"{_LABEL} source_url must not carry credentials in its authority"]
    return []


def _bounded_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _cutoff_instant(raw: str) -> datetime:
    stamp = _parse_stamp(raw)
    if stamp is None:
        raise TemporalSourceReceiptError(f"{_LABEL} cutoff must be an ISO-8601 date or timestamp")
    if len(raw.strip()) == len("YYYY-MM-DD"):
        return datetime.combine(stamp.date(), time.max, tzinfo=timezone.utc)
    return stamp


def _parse_stamp(value: Any) -> datetime | None:
    """An aware UTC datetime from an ISO-8601 date or timestamp, else None.

    A naive timestamp is read as UTC; a provider that reports local time
    without an offset has reported an ambiguous time, and the receipt keeps
    that ambiguity visible in the stored string rather than resolving it.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = strict_fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
