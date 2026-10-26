"""One mechanical ceiling on what an unobserved source is allowed to claim.

The defect this closes: OMH's evidence vocabulary answers exactly one question --
did OMH watch this happen. ``src/evidence/labels.py`` renders that as Phase x
Confidence, and every wire value on it (``prepared_not_observed``,
``observed_partial``, ...) is a statement about observation. Knowledge that
arrived from outside -- an upstream document, a practitioner's write-up -- has no
place on that axis, so it had two ways to be handled and both were wrong: drop it,
or let it ride into a report next to observed facts where a reader cannot tell
the two apart. The second is laundering, and prose telling an agent not to do it
is not a guard.

Two axes, kept apart on purpose, because welding them is the confusion:

* **Observation** -- did OMH see it. Owned by ``evidence.labels`` and the
  observed-claim guards in ``workflows.operator_productivity``. Nothing here
  touches it.
* **Source trust** -- what class of source stands behind a claim OMH never
  observed. ``upstream_official`` is primary or maintainer-published material.
  ``practitioner_heuristic`` is a named practitioner's report of what works for
  them, useful and unverified. ``unattributed`` is a claim with no identifiable
  source behind it.

The ceiling is mechanical, not advisory: :data:`TRUST_CLAIM_CEILING` says which
claim kinds each tier may back, and :func:`build_source_trust_claim` refuses the
record rather than recording a claim the tier cannot support. A
``practitioner_heuristic`` may back ``approach`` -- it can tell you how to go
about something -- and nothing further, so a tip structurally cannot appear as an
established finding. An ``unattributed`` source backs nothing at all; it is
recordable so that it is visible rather than silently dropped, which is the
difference between discarding knowledge and refusing to let it vote.

The strongest rule falls out of the table rather than being written on top of it:
NO tier reaches ``completion``. Not even ``upstream_official``. What is done,
passing, verified, reviewed, or merged is settled by observation alone, and
:data:`SOURCE_TRUST_TIERS` has no member that can say otherwise --
:func:`completion_is_never_source_backed` re-derives that from the table instead
of trusting the sentence you are reading.

Everything here is pure. OMH fetches nothing, subscribes to nothing, and ranks no
publisher: a caller states the tier, and this module decides only what that tier
is permitted to claim. Deciding WHICH sources are trustworthy is the reader's
judgement and stays outside OMH, per the ownership boundary in
``docs/DIRECTION.md`` that puts source-backed research on the Hermes side.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from ..system.append_only_store import (
    RAW_OR_HIDDEN_KEYS,
    is_unsafe_metadata_line,
    opaque_ref,
    reference_errors,
)
from .source_finder import SOURCE_CANDIDATE_SCHEMA_VERSION


SOURCE_TRUST_CLAIM_SCHEMA_VERSION: Final = "source_trust_claim/v1"
SOURCE_TRUST_SUMMARY_SCHEMA_VERSION: Final = "source_trust_summary/v1"

# The three classes of source a claim can rest on, strongest to weakest. Closed,
# because the claim kinds a tier may back is the whole mechanism and a free-text
# tier would route around it.
SOURCE_TRUST_TIERS: Final = ("upstream_official", "practitioner_heuristic", "unattributed")

# The tier an unreadable or unsupported value falls back to. Fail-closed, the way
# `source_finder.normalize_observation_provenance` falls back to `unknown`: an
# unrecognised tier must lose authority, never borrow it.
SOURCE_TRUST_TIER_FALLBACK: Final = "unattributed"

# What a claim asserts, weakest to strongest.
#   approach   -- this is a way to go about the work. Informs a plan.
#   finding    -- this is so. Enters a report as an established fact.
#   completion -- this work is done, passing, verified, reviewed, or merged.
CLAIM_KINDS: Final = ("approach", "finding", "completion")

# What a summary reports when no accepted claim backs anything.
CLAIM_KIND_NONE: Final = "none"

# Which claim kinds each tier may back. `practitioner_heuristic` may back only
# `approach`, which is what makes "a tip never becomes an established finding" a
# property of the schema rather than a rule someone remembers. No tier lists
# `completion`: an outside source cannot settle whether work happened.
TRUST_CLAIM_CEILING: Final = {
    "upstream_official": ("approach", "finding"),
    "practitioner_heuristic": ("approach",),
    "unattributed": (),
}

SOURCE_TRUST_CLAIM_KEYS: Final = (
    "schema_version",
    "tier",
    "claim_kind",
    "claim",
    "source_ref",
    "recorded_at",
)

SOURCE_TRUST_SUMMARY_KEYS: Final = (
    "schema_version",
    "topic",
    "strongest_claim_kind",
    "tiers_present",
    "accepted_count",
    "rejected_count",
    "rejected_claims",
    "claim_boundary",
)
_REJECTED_CLAIM_KEYS: Final = ("index", "reason")

SOURCE_TRUST_CLAIM_BOUNDARY: Final = (
    "This summary reports what class of source stands behind each claim, not whether any claim is "
    "true. OMH fetched nothing, read no source, and verified no statement here; a tier says who "
    "stands behind a claim, never that OMH checked it, and no claim on this axis is execution, "
    "verification, review, CI, or merge evidence."
)

_MAX_TOPIC_CHARS: Final = 160
_MAX_CLAIM_CHARS: Final = 400
_MAX_REJECTED_ROWS: Final = 8
_MAX_REJECTION_REASON_CHARS: Final = 200

_LABEL: Final = "source trust claim"
_SUMMARY_LABEL: Final = "source trust summary"

_CLAIM_KIND_LEVELS: Final = {kind: level for level, kind in enumerate(CLAIM_KINDS, start=1)}


class SourceTrustError(ValueError):
    """Raised when a claim cannot be accepted at the tier it was offered under."""


def normalize_source_trust_tier(tier: str | None) -> str:
    """The supplied tier, or the fallback when it is not one OMH knows.

    Fail-closed: an unrecognised tier becomes `unattributed`, which backs
    nothing. Normalising upward would let a typo buy authority.
    """
    normalized = " ".join(str(tier or "").strip().lower().split()).replace(" ", "_").replace("-", "_")
    return normalized if normalized in SOURCE_TRUST_TIERS else SOURCE_TRUST_TIER_FALLBACK


def claim_kinds_for_tier(tier: str) -> tuple[str, ...]:
    """Every claim kind this tier may back, weakest first. Empty when none."""
    return tuple(TRUST_CLAIM_CEILING.get(normalize_source_trust_tier(tier), ()))


def tier_may_claim(tier: str, claim_kind: str) -> bool:
    """Whether one tier is permitted to back one claim kind."""
    return claim_kind in claim_kinds_for_tier(tier)


def completion_is_never_source_backed() -> bool:
    """Re-derive from the table that no tier can back a completion claim.

    Stated as a derivation rather than a constant so that widening a tier's
    ceiling to include `completion` cannot leave a comment claiming otherwise.
    """
    return all("completion" not in kinds for kinds in TRUST_CLAIM_CEILING.values())


def build_source_trust_claim(
    *,
    tier: str,
    claim_kind: str,
    claim: str,
    recorded_at: str,
    source_ref: str = "",
) -> dict[str, Any]:
    """Mint one source-trust claim, or refuse.

    The refusal is the feature. A `practitioner_heuristic` offered as a `finding`
    does not get downgraded to `approach` and stored, because a silently
    downgraded record reads later as though the caller had claimed correctly;
    it is rejected so the caller has to say what they actually mean.

    `source_ref` is required for a tier that names a source and must be empty for
    `unattributed` -- having no identifiable source is what that tier means, so a
    reference on it would make the tier a lie.
    """
    if tier not in SOURCE_TRUST_TIERS:
        raise SourceTrustError(f"source trust tier is unsupported: {tier!r}")
    if claim_kind not in CLAIM_KINDS:
        raise SourceTrustError(f"source trust claim kind is unsupported: {claim_kind!r}")
    if not tier_may_claim(tier, claim_kind):
        allowed = ", ".join(claim_kinds_for_tier(tier)) or "nothing"
        raise SourceTrustError(
            f"source trust tier {tier} may not back a {claim_kind} claim; it may back: {allowed}"
        )
    has_ref = bool(str(source_ref or "").strip())
    if tier == "unattributed" and has_ref:
        raise SourceTrustError("source trust tier unattributed must not name a source")
    if tier != "unattributed" and not has_ref:
        raise SourceTrustError(f"source trust tier {tier} must name the source it rests on")
    record = {
        "schema_version": SOURCE_TRUST_CLAIM_SCHEMA_VERSION,
        "tier": tier,
        "claim_kind": claim_kind,
        "claim": _bounded_claim(claim),
        "source_ref": _opaque(source_ref, field="source trust claim source_ref") if has_ref else "",
        "recorded_at": _opaque(recorded_at, field="source trust claim recorded_at"),
    }
    errors = source_trust_claim_errors(record)
    if errors:
        raise SourceTrustError(errors[0])
    return record


def source_trust_claim_errors(record: Any) -> list[str]:
    """Every reason one claim is not usable. Both directions on keys."""
    if not isinstance(record, Mapping):
        return [f"{_LABEL} must be an object"]
    errors: list[str] = []
    forbidden = sorted(key for key in record if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if forbidden:
        errors.append(f"{_LABEL} must not carry raw or hidden keys: {forbidden}")
    present = {str(key) for key in record}
    missing = sorted(set(SOURCE_TRUST_CLAIM_KEYS) - present)
    if missing:
        errors.append(f"{_LABEL} is missing keys: {missing}")
    unexpected = sorted(present - set(SOURCE_TRUST_CLAIM_KEYS) - set(forbidden))
    if unexpected:
        errors.append(f"{_LABEL} has unsupported keys: {unexpected}")
    if record.get("schema_version") != SOURCE_TRUST_CLAIM_SCHEMA_VERSION:
        errors.append(f"{_LABEL} schema_version must be {SOURCE_TRUST_CLAIM_SCHEMA_VERSION}")
    tier = record.get("tier")
    if tier not in SOURCE_TRUST_TIERS:
        errors.append(f"{_LABEL} tier is unsupported: {tier!r}")
    claim_kind = record.get("claim_kind")
    if claim_kind not in CLAIM_KINDS:
        errors.append(f"{_LABEL} claim kind is unsupported: {claim_kind!r}")
    elif tier in SOURCE_TRUST_TIERS and not tier_may_claim(str(tier), str(claim_kind)):
        allowed = ", ".join(claim_kinds_for_tier(str(tier))) or "nothing"
        errors.append(f"{_LABEL} tier {tier} may not back a {claim_kind} claim; it may back: {allowed}")
    errors.extend(_bounded_claim_errors(record))
    source_ref = record.get("source_ref")
    if tier == "unattributed" and isinstance(source_ref, str) and source_ref:
        errors.append(f"{_LABEL} tier unattributed must not name a source")
    if tier in SOURCE_TRUST_TIERS and tier != "unattributed" and not source_ref:
        errors.append(f"{_LABEL} tier {tier} must name the source it rests on")
    errors.extend(reference_errors(source_ref, field="source_ref", label=_LABEL, required=False))
    errors.extend(reference_errors(record.get("recorded_at"), field="recorded_at", label=_LABEL, required=True))
    return errors


def summarize_source_trust(
    *,
    topic: str,
    claims: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """What a gathered set of claims is collectively allowed to support.

    Invalid claims are rejected before anything is derived, so a malformed record
    structurally cannot raise the summary. `strongest_claim_kind` is the
    strongest kind any accepted claim backs, and because no tier reaches
    `completion` it can never read as one -- a caller reading this summary to
    decide whether work is done gets `finding` at most and has to go to the
    observation axis for the rest.
    """
    supplied = list(claims)
    accepted: list[Mapping[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejected_count = 0
    for index, record in enumerate(supplied):
        errors = source_trust_claim_errors(record)
        if errors:
            rejected_count += 1
            if len(rejected) < _MAX_REJECTED_ROWS:
                rejected.append({"index": index, "reason": errors[0][:_MAX_REJECTION_REASON_CHARS]})
            continue
        accepted.append(record)

    strongest = CLAIM_KIND_NONE
    best_level = 0
    tiers: set[str] = set()
    for record in accepted:
        tiers.add(str(record["tier"]))
        level = _CLAIM_KIND_LEVELS.get(str(record["claim_kind"]), 0)
        if level > best_level:
            best_level, strongest = level, str(record["claim_kind"])

    return {
        "schema_version": SOURCE_TRUST_SUMMARY_SCHEMA_VERSION,
        "topic": _bounded_topic(topic),
        "strongest_claim_kind": strongest,
        "tiers_present": [tier for tier in SOURCE_TRUST_TIERS if tier in tiers],
        "accepted_count": len(accepted),
        "rejected_count": rejected_count,
        "rejected_claims": rejected,
        "claim_boundary": SOURCE_TRUST_CLAIM_BOUNDARY,
    }


def validate_source_trust_summary(summary: Any) -> list[str]:
    """Every reason one summary is not a well-formed report."""
    if not isinstance(summary, Mapping):
        return [f"{_SUMMARY_LABEL} must be an object"]
    errors: list[str] = []
    present = {str(key) for key in summary}
    missing = sorted(set(SOURCE_TRUST_SUMMARY_KEYS) - present)
    if missing:
        errors.append(f"{_SUMMARY_LABEL} is missing keys: {missing}")
    unexpected = sorted(present - set(SOURCE_TRUST_SUMMARY_KEYS))
    if unexpected:
        errors.append(f"{_SUMMARY_LABEL} has unsupported keys: {unexpected}")
    if summary.get("schema_version") != SOURCE_TRUST_SUMMARY_SCHEMA_VERSION:
        errors.append(f"{_SUMMARY_LABEL} schema_version must be {SOURCE_TRUST_SUMMARY_SCHEMA_VERSION}")
    topic = summary.get("topic")
    if not isinstance(topic, str) or not topic.strip() or len(topic) > _MAX_TOPIC_CHARS:
        errors.append(f"{_SUMMARY_LABEL} topic must be a nonempty string of at most {_MAX_TOPIC_CHARS} characters")
    elif is_unsafe_metadata_line(topic):
        errors.append(f"{_SUMMARY_LABEL} topic must not carry secrets, links, paths, or raw text")
    strongest = summary.get("strongest_claim_kind")
    if strongest not in (CLAIM_KIND_NONE, *CLAIM_KINDS):
        errors.append(f"{_SUMMARY_LABEL} strongest_claim_kind is unsupported: {strongest!r}")
    elif strongest == "completion":
        errors.append(f"{_SUMMARY_LABEL} strongest_claim_kind must never be completion; no tier backs one")
    tiers = summary.get("tiers_present")
    if not isinstance(tiers, list) or any(tier not in SOURCE_TRUST_TIERS for tier in tiers):
        errors.append(f"{_SUMMARY_LABEL} tiers_present must list known source trust tiers")
    for field in ("accepted_count", "rejected_count"):
        value = summary.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{_SUMMARY_LABEL} {field} must be a non-negative integer")
    errors.extend(_rejected_claim_errors(summary.get("rejected_claims")))
    if summary.get("claim_boundary") != SOURCE_TRUST_CLAIM_BOUNDARY:
        errors.append(f"{_SUMMARY_LABEL} claim_boundary must be the frozen boundary sentence")
    return errors


def claim_from_source_candidate(
    *,
    candidate: Mapping[str, Any],
    tier: str,
    claim_kind: str,
    claim: str,
    recorded_at: str,
) -> dict[str, Any]:
    """Bind one claim to a `source_candidate/v1` that `source_finder` already minted.

    This is the producer the ceiling was missing. :func:`build_source_trust_claim`
    can only refuse a claim someone hands it; until something in OMH actually
    mints one, the ceiling is a guard with nothing to guard. `source_finder`
    already produces the one thing a claim needs and a caller usually does not
    have: an opaque identifier for a source.

    **`candidate_id` is the source reference, and `uri` is never read.** The two
    fields sit side by side in a candidate (``source_finder.build_source_candidate``
    keeps the raw ``uri`` verbatim), and passing the wrong one is the mistake this
    function exists to make impossible. ``source_ref`` goes through
    ``reference_errors``, which refuses anything URL-shaped, so a ``uri`` would be
    rejected -- but at the far end of a pipeline, where the failure is hardest to
    read. Reading only ``candidate_id`` moves that from a late refusal to a
    structural impossibility.

    The tier is stated by the caller and never inferred from the candidate.
    Deriving trust from a source's kind -- treating a `docs_spec` as official or a
    `web_link` as a heuristic -- would make OMH the judge of which sources deserve
    trust, which `docs/DIRECTION.md` puts on the Hermes side. OMH decides only
    what a stated tier may claim.

    ``unattributed`` is refused here even though it is a valid tier elsewhere: a
    candidate is a named, identified source, so claiming there is no identifiable
    source behind it contradicts the record being cited.
    """
    if not isinstance(candidate, Mapping):
        raise SourceTrustError("source trust claim requires a source candidate object")
    if candidate.get("schema_version") != SOURCE_CANDIDATE_SCHEMA_VERSION:
        raise SourceTrustError(
            f"source trust claim requires a {SOURCE_CANDIDATE_SCHEMA_VERSION} candidate"
        )
    candidate_id = str(candidate.get("candidate_id", "") or "").strip()
    if not candidate_id:
        raise SourceTrustError("source trust claim requires the candidate's candidate_id")
    if tier == "unattributed":
        raise SourceTrustError(
            "source trust tier unattributed cannot cite a source candidate; the candidate is the attribution"
        )
    return build_source_trust_claim(
        tier=tier,
        claim_kind=claim_kind,
        claim=claim,
        recorded_at=recorded_at,
        source_ref=candidate_id,
    )


def _bounded_claim(claim: str) -> str:
    text = " ".join(str(claim or "").split())
    if not text:
        raise SourceTrustError(f"{_LABEL} requires the claim it carries")
    if len(text) > _MAX_CLAIM_CHARS:
        raise SourceTrustError(f"{_LABEL} claim must be at most {_MAX_CLAIM_CHARS} characters")
    if is_unsafe_metadata_line(text):
        raise SourceTrustError(
            f"{_LABEL} claim must be one bounded metadata line without secrets, links, or paths"
        )
    return text


def _bounded_topic(topic: str) -> str:
    text = " ".join(str(topic or "").split())
    if not text:
        raise SourceTrustError(f"{_SUMMARY_LABEL} requires the topic it summarizes")
    if len(text) > _MAX_TOPIC_CHARS:
        raise SourceTrustError(f"{_SUMMARY_LABEL} topic must be at most {_MAX_TOPIC_CHARS} characters")
    if is_unsafe_metadata_line(text):
        raise SourceTrustError(
            f"{_SUMMARY_LABEL} topic must be one bounded metadata line without secrets, links, or paths"
        )
    return text


def _bounded_claim_errors(record: Mapping[str, Any]) -> list[str]:
    value = record.get("claim")
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_CLAIM_CHARS:
        return [f"{_LABEL} claim must be a nonempty string of at most {_MAX_CLAIM_CHARS} characters"]
    if is_unsafe_metadata_line(value):
        return [f"{_LABEL} claim must not carry secrets, links, paths, or raw text"]
    return []


def _rejected_claim_errors(value: Any) -> list[str]:
    if not isinstance(value, list):
        return [f"{_SUMMARY_LABEL} rejected_claims must be a list"]
    if len(value) > _MAX_REJECTED_ROWS:
        return [f"{_SUMMARY_LABEL} rejected_claims must hold at most {_MAX_REJECTED_ROWS} rows"]
    errors: list[str] = []
    for row in value:
        if not isinstance(row, Mapping):
            errors.append(f"{_SUMMARY_LABEL} rejected_claims rows must be objects")
            continue
        if sorted(str(key) for key in row) != sorted(_REJECTED_CLAIM_KEYS):
            errors.append(f"{_SUMMARY_LABEL} rejected_claims rows must carry exactly {list(_REJECTED_CLAIM_KEYS)}")
            continue
        index = row.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            errors.append(f"{_SUMMARY_LABEL} rejected_claims index must be a non-negative integer")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > _MAX_REJECTION_REASON_CHARS:
            errors.append(
                f"{_SUMMARY_LABEL} rejected_claims reason must be a nonempty string of at most "
                f"{_MAX_REJECTION_REASON_CHARS} characters"
            )
    return errors


def _opaque(value: str, *, field: str) -> str:
    return opaque_ref(value, field=field, error=SourceTrustError)
