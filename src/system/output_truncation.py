"""One versioned contract for every place OMH caps captured output.

Each cap used to invent its own marker: ``"\\n...[truncated]"`` in the release
smoke report, ``"[output truncated at 16384-byte capture limit]"`` in the
Hermes-child stream drainer, and a bare tail slice with no marker at all in the
fanout verification runner. All three tell a reader that something was lost;
none of them says *why*, *how much*, or *where the rest is*. A consumer
therefore cannot tell "this is the whole output" from "more exists, capped
here", and a follow-up step has nothing to resolve.

Every truncation built here carries a machine-readable record instead: a reason
code from a closed vocabulary, the original byte count, the byte ranges that
were kept, and a continuation hint. When a spill store is offered the full
content is written whole to a content-addressed path and the record gains a
resolvable back-reference (path + sha256 + byte length), so a later step can
fetch exactly what was cut. The pointer is emitted only when the write actually
succeeded -- this repo's `prepared_not_observed` discipline applied to
artifacts.

No notice this module renders contains an ellipsis. A bare "..." is the shape
that started the problem: it is indistinguishable from an ellipsis the producer
wrote itself, and it carries no recovery path.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .local_store import atomic_write_text

OUTPUT_TRUNCATION_SCHEMA_VERSION = "omh_output_truncation/v1"
OUTPUT_SPILL_REF_SCHEMA_VERSION = "omh_output_spill_ref/v1"

# Closed vocabulary. `not_truncated` is a real member, not a filler: recording
# it is what lets a reader distinguish "nothing more to show" from "capped".
# `fabricated_boundary` is `truncate_at_fabricated_boundary`'s reason: a caller-
# named authority marker turned up inside content a dispatched unit does not
# get to author, so everything from the marker onward was cut rather than
# trusted.
TRUNCATION_REASON_CODES: tuple[str, ...] = (
    "not_truncated",
    "output_cap",
    "capture_cap",
    "match_limit",
    "redacted",
    "fabricated_boundary",
)
# `kept_ranges` is a list rather than a single range so a head+tail keep can be
# added later without a schema bump. Only the two single-range keeps are
# implemented, because those are the only shapes any current cap site uses.
KEPT_RANGE_POSITIONS: tuple[str, ...] = ("head", "tail")
SPILL_STATUSES: tuple[str, ...] = (
    "written",
    "not_attempted",
    "content_not_retained",
    "store_unavailable",
)

OUTPUT_SPILL_DIR_NAME = "output-spills"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_SOURCE_CHARS = 120
_CONTINUATION_HINTS: dict[str, str] = {
    "not_attempted": (
        "no spill store was offered for this cap; re-run the producing command to see the discarded bytes"
    ),
    "content_not_retained": (
        "the discarded bytes were never retained in memory; re-run with a larger capture bound to see them"
    ),
    "store_unavailable": (
        "the spill write did not succeed, so the discarded bytes are not recoverable from this record"
    ),
}


@dataclass(frozen=True, slots=True)
class TruncatedOutput:
    """A bounded payload plus the contract record describing what was cut."""

    kept_text: str
    record: dict[str, Any]

    @property
    def truncated(self) -> bool:
        return bool(self.record.get("truncated"))

    @property
    def text(self) -> str:
        """The bounded payload with its continuation notice attached."""
        notice = truncation_notice(self.record)
        return f"{self.kept_text}\n{notice}\n" if notice else self.kept_text


def truncation_record(
    *,
    source: str,
    reason_code: str,
    limit_bytes: int,
    kept_bytes: int,
    original_bytes: int | None,
    keep: str = "tail",
    spill: Mapping[str, Any] | None = None,
    spill_status: str = "not_attempted",
) -> dict[str, Any]:
    """Build one truncation contract record.

    `original_bytes` is `None` when the cap was applied upstream and the total
    is genuinely unknowable -- recorded as unknown rather than guessed, because
    a fabricated size is worse than an absent one.
    """
    normalized_reason = reason_code if reason_code in TRUNCATION_REASON_CODES else "output_cap"
    truncated = normalized_reason != "not_truncated"
    status = spill_status if spill_status in SPILL_STATUSES else "not_attempted"
    if not truncated:
        status = "not_attempted"
    record: dict[str, Any] = {
        "schema_version": OUTPUT_TRUNCATION_SCHEMA_VERSION,
        "truncated": truncated,
        "reason_code": normalized_reason,
        "source": _bounded_source(source),
        "limit_bytes": max(0, int(limit_bytes)),
        "original_bytes": None if original_bytes is None else max(0, int(original_bytes)),
        "kept_bytes": max(0, int(kept_bytes)),
        "kept_ranges": _kept_ranges(
            keep=keep,
            kept_bytes=max(0, int(kept_bytes)),
            original_bytes=original_bytes,
        ),
        "spill_status": status,
    }
    # Emit-only-on-success: a pointer exists in the record only when a write
    # was actually observed to land, never because one was attempted.
    if truncated and status == "written" and isinstance(spill, Mapping):
        record["spill"] = dict(spill)
    record["continuation_hint"] = _continuation_hint(record)
    return record


def truncate_output(
    value: object,
    *,
    limit_bytes: int,
    source: str,
    reason_code: str = "output_cap",
    keep: str = "tail",
    spill_dir: Path | None = None,
) -> TruncatedOutput:
    """Cap `value` at `limit_bytes` and describe the cut in the contract.

    The bound is a byte bound, and the kept slice is aligned to a UTF-8
    character boundary so a cap can never emit a half-decoded character. Text
    exactly at the bound is not truncated: the record then carries the
    `not_truncated` reason so a consumer reads "this is the whole output"
    rather than having to infer it from an absent marker.
    """
    text = value if isinstance(value, str) else str(value or "")
    encoded = text.encode("utf-8")
    original_bytes = len(encoded)
    limit = max(0, int(limit_bytes))
    if original_bytes <= limit:
        return TruncatedOutput(
            text,
            truncation_record(
                source=source,
                reason_code="not_truncated",
                limit_bytes=limit,
                kept_bytes=original_bytes,
                original_bytes=original_bytes,
                keep="head",
            ),
        )

    position = keep if keep in KEPT_RANGE_POSITIONS else "tail"
    if position == "head":
        kept = _align_head(encoded[:limit])
    else:
        kept = _align_tail(encoded[original_bytes - limit :])

    spill: Mapping[str, Any] | None = None
    spill_status = "not_attempted"
    if spill_dir is not None:
        spill = write_output_spill(spill_dir, text)
        spill_status = "written" if spill else "store_unavailable"

    return TruncatedOutput(
        kept.decode("utf-8", errors="replace"),
        truncation_record(
            source=source,
            reason_code=reason_code,
            limit_bytes=limit,
            kept_bytes=len(kept),
            original_bytes=original_bytes,
            keep=position,
            spill=spill,
            spill_status=spill_status,
        ),
    )


def truncate_at_fabricated_boundary(
    value: object,
    *,
    markers: Sequence[str],
    source: str,
    spill_dir: Path | None = None,
) -> TruncatedOutput:
    """Cap `value` at the first occurrence of any string in `markers`.

    OMH's analogue of `tools.abortOnFabricatedResult`
    (`docs/toolconv/xml.md:188` in oh-my-pi; OMH absorption, P3 item Q): a
    dispatched unit's own captured output is not a trusted source for the
    markers named here -- only OMH's own dispatcher legitimately writes them.
    The first occurrence of any marker, wherever it falls, is a hard boundary:
    everything before it is kept as this capture's real content; everything
    from the marker onward is cut under the `fabricated_boundary` reason code
    and never parsed as if it were genuine, mirroring "content before the
    fabricated block is preserved, generation stops".

    `markers` is caller-supplied rather than a fixed OMH vocabulary: which
    strings count as an unforgeable authority marker is a property of the
    call site (what only ITS OWN dispatcher ever legitimately writes), not of
    this generic contract. No marker found is not truncation -- `value`
    passes through unchanged with the `not_truncated` reason, exactly like
    `truncate_output` at or under its limit.
    """
    text = value if isinstance(value, str) else str(value or "")
    original_bytes = len(text.encode("utf-8"))
    boundary = _first_marker_offset(text, markers)
    if boundary is None:
        return TruncatedOutput(
            text,
            truncation_record(
                source=source,
                reason_code="not_truncated",
                limit_bytes=original_bytes,
                kept_bytes=original_bytes,
                original_bytes=original_bytes,
                keep="head",
            ),
        )

    kept = text[:boundary]
    kept_bytes = len(kept.encode("utf-8"))
    spill: Mapping[str, Any] | None = None
    spill_status = "not_attempted"
    if spill_dir is not None:
        # The full, unredacted text spills -- including the fabricated
        # portion -- so a reviewer can inspect exactly what was cut, the same
        # discipline `truncate_output` already applies to a length cap.
        spill = write_output_spill(spill_dir, text)
        spill_status = "written" if spill else "store_unavailable"

    return TruncatedOutput(
        kept,
        truncation_record(
            source=source,
            reason_code="fabricated_boundary",
            limit_bytes=kept_bytes,
            kept_bytes=kept_bytes,
            original_bytes=original_bytes,
            keep="head",
            spill=spill,
            spill_status=spill_status,
        ),
    )


def _first_marker_offset(text: str, markers: Sequence[str]) -> int | None:
    offsets = [text.index(marker) for marker in markers if marker and marker in text]
    return min(offsets) if offsets else None


def write_output_spill(spill_dir: Path, text: str) -> dict[str, Any]:
    """Write the full text to a content-addressed path; return its back-reference.

    Content addressing is what makes the path deterministic *and* resume-safe:
    identical content re-spills onto the same file, different content can never
    land on an existing one, so a resumed or re-reaped dispatch cannot clobber
    an earlier run's spill. That is the property the scan-then-allocate counter
    rule buys elsewhere, without a counter to keep consistent across processes.

    Returns `{}` when the write did not succeed, so the caller emits no pointer
    at all rather than one nothing can resolve.
    """
    encoded = text.encode("utf-8")
    digest = sha256(encoded).hexdigest()
    path = spill_dir / f"{digest}.txt"
    try:
        # Temp-then-rename, so a crash mid-write leaves either the previous
        # file or no file -- never a half-written one a digest check would
        # later reject.
        atomic_write_text(path, text, private=True)
    except OSError:
        return {}
    return {
        "schema_version": OUTPUT_SPILL_REF_SCHEMA_VERSION,
        "path": str(path),
        "sha256": digest,
        "byte_count": len(encoded),
    }


def resolve_spill_reference(spill: Mapping[str, Any]) -> str:
    """Read back exactly what a spill pointer promised, or raise.

    Digest and byte length are re-checked against the file, so a pointer into a
    rewritten, truncated, or partially written file fails loudly instead of
    returning content that is not what was cut.
    """
    if str(spill.get("schema_version", "")) != OUTPUT_SPILL_REF_SCHEMA_VERSION:
        raise ValueError("output spill reference schema is unsupported")
    digest = str(spill.get("sha256", ""))
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError("output spill reference digest is invalid")
    path = Path(str(spill.get("path", "")))
    if path.is_symlink():
        raise ValueError("output spill file must not be a symlink")
    text = path.read_text(encoding="utf-8")
    encoded = text.encode("utf-8")
    if len(encoded) != int(spill.get("byte_count", -1)):
        raise ValueError("output spill file length does not match its reference")
    if sha256(encoded).hexdigest() != digest:
        raise ValueError("output spill file digest does not match its reference")
    return text


def truncation_notice(record: Mapping[str, Any], *, compact: bool = False) -> str:
    """One rendered line saying truncated, why, how much, and where the rest lives.

    Empty for an untruncated record: a reader gets a notice only when there is
    something to continue from.

    `compact` is for surfaces that impose their own character ceiling on prose
    -- the observation journal caps `summary` at 500 -- where a full notice
    would itself be cut and leave a half-written pointer nothing can resolve.
    It drops the schema and source fields and, when the record has a spill,
    points at `evidence_refs` instead of inlining the path: such a surface is
    expected to carry `spill_evidence_ref(record)` in that uncapped field.
    """
    if not record.get("truncated"):
        return ""
    original = record.get("original_bytes")
    fields = [
        f"reason={record.get('reason_code', 'output_cap')}",
        f"original_bytes={'unknown' if original is None else original}",
        f"kept_bytes={record.get('kept_bytes', 0)}",
        f"kept_ranges={_rendered_ranges(record.get('kept_ranges'))}",
    ]
    if compact:
        spilled = isinstance(record.get("spill"), Mapping)
        fields.append(
            f"continuation={'evidence_refs' if spilled else record.get('spill_status', 'not_attempted')}"
        )
    else:
        fields.insert(0, f"schema={record.get('schema_version', OUTPUT_TRUNCATION_SCHEMA_VERSION)}")
        fields.insert(2, f"source={record.get('source', '')}")
        fields.append(f"continuation={record.get('continuation_hint', '')}")
    return "[output truncated: " + "; ".join(fields) + "]"


def spill_evidence_ref(record: Mapping[str, Any]) -> str:
    """A one-string evidence ref for a spilled truncation, or "" when none spilled.

    For surfaces that carry `evidence_refs` and cap their prose: the pointer
    stays whole in a field with no character ceiling, instead of riding in a
    summary line a bare slice could cut in half and leave unresolvable.
    """
    spill = record.get("spill") if isinstance(record, Mapping) else None
    if not isinstance(spill, Mapping):
        return ""
    return (
        f"output_spill:{spill.get('path', '')}"
        f":sha256:{spill.get('sha256', '')}:{spill.get('byte_count', 0)}"
    )


def _kept_ranges(*, keep: str, kept_bytes: int, original_bytes: int | None) -> list[dict[str, Any]]:
    position = keep if keep in KEPT_RANGE_POSITIONS else "tail"
    if position == "tail" and original_bytes is not None:
        start = max(0, original_bytes - kept_bytes)
        return [{"position": "tail", "start_byte": start, "end_byte": start + kept_bytes}]
    # A tail keep whose original size is unknown cannot name an absolute
    # offset, so it is reported as the head of what survived rather than as an
    # invented range.
    return [{"position": "head", "start_byte": 0, "end_byte": kept_bytes}]


def _rendered_ranges(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or not value:
        return "none"
    parts = []
    for entry in value:
        if not isinstance(entry, Mapping):
            continue
        parts.append(
            f"{entry.get('position', 'head')} {entry.get('start_byte', 0)}-{entry.get('end_byte', 0)}"
        )
    return ",".join(parts) if parts else "none"


def _continuation_hint(record: Mapping[str, Any]) -> str:
    if not record.get("truncated"):
        return ""
    spill = record.get("spill")
    if isinstance(spill, Mapping):
        return (
            f"read {spill.get('path', '')} "
            f"(sha256 {spill.get('sha256', '')}, {spill.get('byte_count', 0)} bytes) for the full output"
        )
    return _CONTINUATION_HINTS.get(str(record.get("spill_status", "")), _CONTINUATION_HINTS["not_attempted"])


def _bounded_source(value: object) -> str:
    text = " ".join(str(value or "").split())
    return text[:_MAX_SOURCE_CHARS]


def _align_head(chunk: bytes) -> bytes:
    """Drop a trailing partial UTF-8 sequence so the kept prefix decodes whole."""
    for back in range(1, min(4, len(chunk)) + 1):
        byte = chunk[-back]
        if byte < 0x80:
            return chunk
        if byte >= 0xC0:
            width = 2 if byte < 0xE0 else 3 if byte < 0xF0 else 4
            return chunk if back == width else chunk[: len(chunk) - back]
    return chunk


def _align_tail(chunk: bytes) -> bytes:
    """Drop leading continuation bytes so the kept suffix decodes whole."""
    index = 0
    while index < len(chunk) and index < 4 and 0x80 <= chunk[index] < 0xC0:
        index += 1
    return chunk[index:]
