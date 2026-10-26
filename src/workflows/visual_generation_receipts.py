"""Append-only receipts binding one image result to the route a producer attested.

`visual_observation/v1` records that a referenced image artifact was reported.
That is a real observation and it stays what it is, but it cannot answer the
question a user actually asks about a generated image: *which* backend and model
produced *these* bytes, from *which* prompt-card revision and reference inputs,
for *which* accepted attempt. A host can accept a requested model value without
attesting that it honoured the selection, so a returned file proves a file, never
a route.

A `visual_generation_receipt/v1` is therefore built around one separation that
runs through every field on it:

- `requested_route` is what the request asked for. It is intent, and intent is
  never evidence.
- `observed_route` is what the producer attested. Every field on it that is not
  named in `attested_route_fields` is exactly `"unknown"`, and a field that *is*
  named may not be `"unknown"` -- attesting nothing is not an attestation.

Nothing copies across that line. A successful artifact cannot promote a
requested provider, model, quality, operation, dimension, or credential class
into an observed one, because the two blocks are validated independently and the
attested set is the only bridge. `route_evidence` reports the gap rather than
closing it.

The rest follows the same rule that a record may only say what something
observed:

- A receipt is minted per attempt, and its identity binds the prompt-card
  digest, the accepted action, and the attempt. Re-reporting one attempt is
  idempotent by that identity; a retry links through `supersedes_receipt_ref`
  instead of overwriting.
- Only `outcome="succeeded"` carries an artifact, and only a succeeded receipt
  can back a generated-image observation. A failed or partial attempt keeps a
  bounded `failure_stage` and mints no image evidence.
- The artifact is bound to its content digest, not to its path. A path reused
  for different bytes is a different artifact and therefore a different receipt
  identity, so an earlier result cannot survive under a replaced file.
- Usage and cost are producer-attributed readings with `measured`, `estimated`,
  or `unavailable` semantics. A missing metric is unavailable and carries a null
  value; it never becomes zero, and an estimate is never measured spend.

Everything stored is metadata: bounded opaque identifiers, content digests,
closed-vocabulary values, and one redacted summary line. No raw prompt, source
image, private path, credential, or provider request/response body can be
stored, and the key-name guard rejects the names those arrive under as well as
the shapes.

The store mechanics -- the torn-tail-safe append, the supersede-chain walk, the
opaque-reference guards -- are `system/append_only_store.py`, shared with the
external-effect and approval receipt families for the reasons that module gives.
Only the meaning is local.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..system.append_only_store import (
    RAW_OR_HIDDEN_KEYS,
    append_store_line,
    is_unsafe_metadata_line,
    latest_record_in,
    mint_record_id,
    opaque_ref,
    record_fingerprint,
    reference_errors,
    supersede_chain_errors,
)
from ..system.local_store import file_lock, read_jsonl_objects, utc_now
from ..system.paths import OmhPaths
from .visual_summary import (
    SUPPORTED_IMAGE_MIME_TYPES,
    VISUAL_GENERATION_BINDING_KEYS,
    valid_visual_id,
    validate_visual_observation,
)


VISUAL_GENERATION_RECEIPT_SCHEMA_VERSION = "visual_generation_receipt/v1"
VISUAL_GENERATION_ROUTE_EVIDENCE_SCHEMA_VERSION = "visual_generation_route_evidence/v1"
VISUAL_GENERATION_RECEIPT_PROJECTION_SCHEMA_VERSION = "visual_generation_receipt_projection/v1"
VISUAL_GENERATION_RECEIPT_STORE_VALIDATION_SCHEMA_VERSION = "visual_generation_receipt_store_validation/v1"
VISUAL_GENERATION_RECEIPT_STORE_NAME = "generation_receipts.jsonl"

RECEIPT_PRIVACY = "metadata_only"
CLAIM_BOUNDARY = (
    "A visual generation receipt is one producer's report of one image attempt. "
    "It is not visual QA, attachment, posting, sharing, or delivery evidence, and it does not "
    "prove that the image content is correct."
)
ROUTE_CLAIM_BOUNDARY = (
    "Requested route is intent. Only route fields named in attested_route_fields were observed; "
    "every other observed field is unknown and must not be filled in from configuration, "
    "capability state, or the existence of a returned file."
)

# The value every unattested route field carries. It is a single spelling on
# purpose: a renderer that special-cases "", None, and "unknown" is a renderer
# that will eventually print one of them as a model name.
UNKNOWN = "unknown"

# What the request asked the producer to do. `edit` is the operation that has
# source images and preserve/remove/replace constraints; `generate` has neither.
OPERATIONS = ("generate", "edit")

# The route, field by field. Requested and observed carry exactly these keys so
# a renderer can put the two side by side without a per-field lookup table.
ROUTE_FIELDS = ("provider", "model", "quality", "operation", "dimensions", "credential_class")
# Which route fields a mismatch or an unknown is worth warning about. Dimensions
# and credential class are reported but do not raise a warning on their own:
# a host legitimately normalises a requested size, and a credential path is not
# a claim about what produced the image.
ROUTE_CLAIM_FIELDS = ("provider", "model", "quality", "operation")
# How the producer was authorised, without naming or storing anything issued.
CREDENTIAL_CLASSES = ("host_managed", "user_supplied", "subscription", "none", UNKNOWN)

# The terminal states a producer can report. `unknown` is a producer that saw
# the attempt end but cannot classify how -- the same meaning the external-effect
# store gives the word.
RECEIPT_OUTCOMES = ("succeeded", "partial", "failed", UNKNOWN)
# Where a non-succeeded attempt stopped. `none` belongs to `succeeded` alone, so
# every other outcome has to say where it got to.
FAILURE_STAGES = ("none", "setup", "authorization", "request", "provider", "download", "validation", UNKNOWN)

# What an edit says about the source image, without storing the instruction.
EDIT_CONSTRAINT_KINDS = ("preserve", "remove", "replace")

# Usage semantics. `measured` is a figure the producer read off the provider,
# `estimated` is one it computed, `unavailable` is one it does not have. There is
# no fourth state, and in particular there is no zero that means "not reported".
USAGE_MEASUREMENTS = ("measured", "estimated", "unavailable")
USAGE_METRIC_NAMES = ("input_tokens", "output_tokens", "image_units", "cost_usd")
INTEGER_USAGE_METRICS = ("input_tokens", "output_tokens", "image_units")

# What a generation receipt never proves, carried on the record so a reader that
# has only the record still has the boundary.
DOES_NOT_PROVE = ("visual_qa_passed", "delivered", "image_content_correct")

RECEIPT_KEYS = (
    "action_id",
    "artifact",
    "attempt_id",
    "attested_route_fields",
    "card_digest",
    "card_id",
    "claim_boundary",
    "does_not_prove",
    "effect_id",
    "evidence_refs",
    "external_effect_ref",
    "failure_stage",
    "input_lineage",
    "observed_at",
    "observed_route",
    "outcome",
    "privacy",
    "producer",
    "receipt_id",
    "requested_route",
    "route_claim_boundary",
    "schema_version",
    "summary",
    "supersedes_receipt_ref",
    "usage",
)
ARTIFACT_KEYS = ("artifact_ref", "byte_size", "content_sha256", "mime_type", "provider_response_ref")
INPUT_LINEAGE_KEYS = ("edit_constraints", "source_image_count", "source_image_digests")
USAGE_READING_KEYS = ("measurement", "value")

# Identifiers that must always name something.
_REQUIRED_RECEIPT_REFS = ("receipt_id", "effect_id", "action_id", "attempt_id", "observed_at", "producer")
# Identifiers that are legitimately absent. `external_effect_ref` is empty until
# a wrapper supplies an identity it already minted for this action, which is the
# whole point of the field: OMH does not invent a second claim that the action
# occurred.
_OPTIONAL_RECEIPT_REFS = ("external_effect_ref", "supersedes_receipt_ref")
_RECEIPT_TEXT_FIELDS = ("summary",)
_RECEIPT_STRING_FIELDS = (
    "receipt_id",
    "effect_id",
    "card_id",
    "card_digest",
    "action_id",
    "attempt_id",
    "producer",
    "outcome",
    "failure_stage",
    "observed_at",
    "external_effect_ref",
    "summary",
    "supersedes_receipt_ref",
)

# What identifies *which attempt report* a receipt is, ignoring when it was
# written. The content digest is in here on purpose: the same path carrying
# different bytes is a different result and must mint its own receipt rather
# than dedupe onto the earlier one.
_ATTEMPT_IDENTITY_KEYS = (
    "action_id",
    "artifact",
    "attempt_id",
    "attested_route_fields",
    "card_digest",
    "card_id",
    "effect_id",
    "evidence_refs",
    "external_effect_ref",
    "failure_stage",
    "input_lineage",
    "observed_route",
    "outcome",
    "producer",
    "requested_route",
    "summary",
    "usage",
)

MAX_SUMMARY_CHARS = 200
MAX_EVIDENCE_REFS = 8
MAX_SOURCE_IMAGES = 8
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_USAGE_VALUE = 1_000_000_000

_LABEL = "visual_generation_receipt"
_DIGEST = re.compile(r"^[a-f0-9]{64}$")
# `observed_at` is the one required reference that is not an identifier, and the
# store's whole ordering claim rests on it, so it carries a format bound as well
# as the reference guard.
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")
# A reference that names a place on a filesystem is not an opaque handle. The
# shared guard permits `/` and `:` because other families store platform ids
# that contain them; this family's spec names private paths explicitly, so the
# path shapes are refused here rather than in the shared guard.
_PATH_SHAPED = re.compile(r"/|^[A-Za-z]:|\\")
_EFFECT_PREFIX = "visual_generation"


class VisualGenerationReceiptError(ValueError):
    """Raised when an attempt report cannot become a receipt."""


def visual_generation_effect_id(card_id: str, action_id: str, attempt_id: str) -> str:
    """Stable identity of one image attempt against one accepted action.

    The action is part of the identity, not decoration. A producer that numbers
    attempts per action -- the obvious numbering -- would otherwise file
    `act-A/att-1` and `act-B/att-1` under one identity, and the second would
    supersede the first: one action's result made unreachable, and a replacement
    recorded that never happened.
    """
    return f"{_EFFECT_PREFIX}:{card_id}:{action_id}:{attempt_id}"


def empty_route(*, operation: str = UNKNOWN) -> dict[str, str]:
    """A route with nothing observed on it."""
    route = {field: UNKNOWN for field in ROUTE_FIELDS}
    route["operation"] = operation
    return route


def build_visual_generation_receipt(
    *,
    card_id: str,
    card_digest: str,
    action_id: str,
    attempt_id: str,
    producer: str,
    outcome: str,
    requested_route: Mapping[str, str],
    observed_route: Mapping[str, str] | None = None,
    attested_route_fields: Sequence[str] = (),
    artifact: Mapping[str, Any] | None = None,
    input_lineage: Mapping[str, Any] | None = None,
    usage: Mapping[str, Mapping[str, Any]] | None = None,
    failure_stage: str = "none",
    external_effect_ref: str = "",
    evidence_refs: Sequence[str] = (),
    summary: str = "",
    observed_at: str = "",
    supersedes_receipt_ref: str = "",
) -> dict[str, Any]:
    """Mint one receipt for one attempt, or refuse.

    Every refusal here is the separation this module exists for: a route field
    is observed only when the producer names it as attested, and image evidence
    exists only when the attempt succeeded and the bytes have a digest.
    """
    if outcome not in RECEIPT_OUTCOMES:
        raise VisualGenerationReceiptError(f"{_LABEL} outcome is unsupported: {outcome!r}")
    if failure_stage not in FAILURE_STAGES:
        raise VisualGenerationReceiptError(f"{_LABEL} failure_stage is unsupported: {failure_stage!r}")
    if not valid_visual_id(str(card_id)):
        raise VisualGenerationReceiptError(f"{_LABEL} card_id must contain only letters, digits, and hyphens")
    if not _DIGEST.match(str(card_digest or "")):
        raise VisualGenerationReceiptError(f"{_LABEL} card_digest must be a sha256 hex digest of the prompt card")
    safe_action_id = _opaque(action_id, field=f"{_LABEL} action_id")
    safe_attempt_id = _opaque(attempt_id, field=f"{_LABEL} attempt_id")
    safe_producer = _opaque(producer, field=f"{_LABEL} producer")
    safe_external = (
        _opaque(external_effect_ref, field=f"{_LABEL} external_effect_ref") if external_effect_ref else ""
    )
    safe_supersedes = (
        _opaque(supersedes_receipt_ref, field=f"{_LABEL} supersedes_receipt_ref") if supersedes_receipt_ref else ""
    )
    stamp = _opaque(str(observed_at or utc_now()), field=f"{_LABEL} observed_at")
    attested = _normalized_attested(attested_route_fields)
    record = {
        "schema_version": VISUAL_GENERATION_RECEIPT_SCHEMA_VERSION,
        "receipt_id": "",
        "effect_id": _opaque(
            visual_generation_effect_id(str(card_id), safe_action_id, safe_attempt_id),
            field=f"{_LABEL} effect_id",
        ),
        "card_id": str(card_id),
        "card_digest": str(card_digest),
        "action_id": safe_action_id,
        "attempt_id": safe_attempt_id,
        "producer": safe_producer,
        "outcome": outcome,
        "failure_stage": failure_stage,
        "requested_route": _normalized_route(requested_route),
        "observed_route": _normalized_route(observed_route or {}),
        "attested_route_fields": attested,
        "artifact": _normalized_artifact(artifact or {}),
        "input_lineage": _normalized_lineage(input_lineage or {}),
        "usage": _normalized_usage(usage or {}),
        "external_effect_ref": safe_external,
        "evidence_refs": _bounded_refs(evidence_refs),
        "summary": bounded_receipt_summary(summary),
        "observed_at": stamp,
        "supersedes_receipt_ref": safe_supersedes,
        "does_not_prove": list(DOES_NOT_PROVE),
        "privacy": RECEIPT_PRIVACY,
        "claim_boundary": CLAIM_BOUNDARY,
        "route_claim_boundary": ROUTE_CLAIM_BOUNDARY,
    }
    record["receipt_id"] = mint_record_id(
        prefix="vgr",
        identity={
            "effect_id": str(record["effect_id"]),
            "outcome": outcome,
            "content_sha256": str(record["artifact"]["content_sha256"]),
            "observed_at": stamp,
        },
    )
    errors = validate_visual_generation_receipt(record)
    if errors:
        raise VisualGenerationReceiptError(errors[0])
    return record


def validate_visual_generation_receipt(record: Any) -> list[str]:
    """Every reason a stored record is not a receipt, reported at once."""
    if not isinstance(record, dict):
        return [f"{_LABEL} must be an object"]
    errors: list[str] = []
    forbidden = sorted(key for key in record if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if forbidden:
        errors.append(f"{_LABEL} must not carry raw or hidden keys: {forbidden}")
    extra_keys = sorted(set(record) - set(RECEIPT_KEYS) - set(forbidden))
    if extra_keys:
        errors.append(f"{_LABEL} has unsupported keys: {extra_keys}")
    missing = sorted(set(RECEIPT_KEYS) - set(record))
    if missing:
        errors.append(f"{_LABEL} is missing keys: {missing}")
    if record.get("schema_version") != VISUAL_GENERATION_RECEIPT_SCHEMA_VERSION:
        errors.append(f"{_LABEL} schema_version must be {VISUAL_GENERATION_RECEIPT_SCHEMA_VERSION}")
    for key in _RECEIPT_STRING_FIELDS:
        if not isinstance(record.get(key), str):
            errors.append(f"{_LABEL} {key} must be a string")
    if record.get("privacy") != RECEIPT_PRIVACY:
        errors.append(f"{_LABEL} privacy must be {RECEIPT_PRIVACY}")
    if record.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append(f"{_LABEL} claim_boundary must state the receipt boundary")
    if record.get("route_claim_boundary") != ROUTE_CLAIM_BOUNDARY:
        errors.append(f"{_LABEL} route_claim_boundary must state the requested-versus-observed boundary")
    if record.get("does_not_prove") != list(DOES_NOT_PROVE):
        errors.append(f"{_LABEL} does_not_prove must be {list(DOES_NOT_PROVE)}")
    if not valid_visual_id(str(record.get("card_id", ""))):
        errors.append(f"{_LABEL} card_id must contain only letters, digits, and hyphens")
    if not _DIGEST.match(str(record.get("card_digest", ""))):
        errors.append(f"{_LABEL} card_digest must be a sha256 hex digest of the prompt card")
    for field in _REQUIRED_RECEIPT_REFS:
        errors.extend(reference_errors(record.get(field), field=field, label=_LABEL, required=True))
    for field in _OPTIONAL_RECEIPT_REFS:
        errors.extend(reference_errors(record.get(field), field=field, label=_LABEL, required=False))
    if not _TIMESTAMP.match(str(record.get("observed_at", ""))):
        errors.append(f"{_LABEL} observed_at must be an ISO-8601 UTC timestamp")
    expected_effect = visual_generation_effect_id(
        str(record.get("card_id", "")), str(record.get("action_id", "")), str(record.get("attempt_id", ""))
    )
    if str(record.get("effect_id", "")) != expected_effect:
        errors.append(f"{_LABEL} effect_id must be {expected_effect}; a receipt names the attempt it is about")
    errors.extend(_outcome_errors(record))
    errors.extend(_route_errors(record))
    errors.extend(_artifact_errors(record))
    errors.extend(_lineage_errors(record))
    errors.extend(_usage_errors(record))
    errors.extend(_refs_errors(record))
    errors.extend(_text_errors(record))
    return errors


# --- route reading -------------------------------------------------------


def observed_route_field(receipt: Mapping[str, Any], field: str) -> str:
    """One observed route field, or `unknown`.

    The only reader any surface should use. It answers from the attested set
    rather than from the stored value, so a hand-edited store that filled an
    observed field in without attesting it still reads as unknown.
    """
    if field not in ROUTE_FIELDS:
        return UNKNOWN
    attested = receipt.get("attested_route_fields")
    if not isinstance(attested, list) or field not in attested:
        return UNKNOWN
    route = receipt.get("observed_route")
    if not isinstance(route, Mapping):
        return UNKNOWN
    value = str(route.get(field, UNKNOWN))
    return value or UNKNOWN


def requested_route_field(receipt: Mapping[str, Any], field: str) -> str:
    if field not in ROUTE_FIELDS:
        return UNKNOWN
    route = receipt.get("requested_route")
    if not isinstance(route, Mapping):
        return UNKNOWN
    return str(route.get(field, UNKNOWN)) or UNKNOWN


def route_evidence(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Requested against observed, plus every gap between them.

    This is the shape a chat or status surface renders. It never resolves a gap:
    an unattested field stays unknown here exactly as it is on the receipt, and a
    disagreement is reported as a mismatch rather than settled in favour of
    either side.
    """
    requested = {field: requested_route_field(receipt, field) for field in ROUTE_FIELDS}
    observed = {field: observed_route_field(receipt, field) for field in ROUTE_FIELDS}
    unknown_fields = [field for field in ROUTE_CLAIM_FIELDS if observed[field] == UNKNOWN]
    mismatched_fields = [
        field
        for field in ROUTE_CLAIM_FIELDS
        if observed[field] != UNKNOWN and requested[field] != UNKNOWN and observed[field] != requested[field]
    ]
    warnings: list[str] = []
    warnings.extend(f"route_mismatch:{field}" for field in mismatched_fields)
    warnings.extend(f"unknown_observed_route:{field}" for field in unknown_fields)
    return {
        "schema_version": VISUAL_GENERATION_ROUTE_EVIDENCE_SCHEMA_VERSION,
        "receipt_id": str(receipt.get("receipt_id", "")),
        "effect_id": str(receipt.get("effect_id", "")),
        "card_id": str(receipt.get("card_id", "")),
        "attempt_id": str(receipt.get("attempt_id", "")),
        "producer": str(receipt.get("producer", "")),
        "outcome": str(receipt.get("outcome", "")),
        "failure_stage": str(receipt.get("failure_stage", "")),
        "requested_route": requested,
        "observed_route": observed,
        "attested_route_fields": [
            field for field in ROUTE_FIELDS if observed_route_field(receipt, field) != UNKNOWN
        ],
        "unknown_route_fields": unknown_fields,
        "mismatched_route_fields": mismatched_fields,
        "warnings": warnings,
        "claim_boundary": ROUTE_CLAIM_BOUNDARY,
    }


def usage_reading(receipt: Mapping[str, Any], name: str) -> dict[str, Any]:
    """One usage figure as the producer attributed it, or an explicit unavailable.

    Absence is the honest reading, never zero: a receipt that omits a metric is
    a producer that did not report it, and a zero it did report is kept as the
    measured or estimated zero it is.
    """
    usage = receipt.get("usage")
    reading = usage.get(name) if isinstance(usage, Mapping) else None
    if name not in USAGE_METRIC_NAMES or not isinstance(reading, Mapping):
        return {"value": None, "measurement": "unavailable"}
    measurement = reading.get("measurement")
    if measurement not in USAGE_MEASUREMENTS or measurement == "unavailable":
        return {"value": None, "measurement": "unavailable"}
    return {"value": reading.get("value"), "measurement": str(measurement)}


def route_status_warnings(
    receipts: Sequence[Mapping[str, Any]],
    *,
    card_digest: str = "",
) -> list[str]:
    """Cross-receipt warnings a single receipt cannot raise on its own.

    Two of them exist only in the history: a receipt filed against a prompt-card
    revision that is no longer the current one, and one artifact reference that
    two receipts saw carrying different bytes. The second is the reason the
    artifact is bound to a digest at all -- a path is mutable and a reused one
    would otherwise look like the same result.
    """
    warnings: list[str] = []
    if card_digest:
        stale = sorted(
            {
                str(receipt.get("receipt_id", ""))
                for receipt in receipts
                if str(receipt.get("card_digest", "")) not in ("", card_digest)
            }
        )
        warnings.extend(f"stale_card_identity:{receipt_id}" for receipt_id in stale if receipt_id)
    digests_by_ref: dict[str, set[str]] = {}
    attempts_by_response: dict[str, set[str]] = {}
    for receipt in receipts:
        artifact = receipt.get("artifact")
        if not isinstance(artifact, Mapping):
            continue
        ref = str(artifact.get("artifact_ref", ""))
        digest = str(artifact.get("content_sha256", ""))
        if ref and digest:
            digests_by_ref.setdefault(ref, set()).add(digest)
        response_ref = str(artifact.get("provider_response_ref", ""))
        effect_id = str(receipt.get("effect_id", ""))
        if response_ref and effect_id:
            attempts_by_response.setdefault(response_ref, set()).add(effect_id)
    warnings.extend(
        f"artifact_digest_drift:{ref}" for ref in sorted(digests_by_ref) if len(digests_by_ref[ref]) > 1
    )
    # One provider response cited by two attempts is either a producer replaying
    # a response or two attempts credited to one call. Either way the second
    # attempt's result is not established by that response, so it is reported
    # rather than accepted as separate evidence.
    warnings.extend(
        f"provider_response_reuse:{ref}"
        for ref in sorted(attempts_by_response)
        if len(attempts_by_response[ref]) > 1
    )
    return warnings


# --- generated-image binding ---------------------------------------------


def generated_image_binding_errors(
    receipt: Mapping[str, Any],
    *,
    card_id: str,
    artifact: Mapping[str, Any] | None = None,
) -> list[str]:
    """Every reason a receipt cannot back a generated-image observation.

    `artifact` is the observation's own artifact block. It is checked rather
    than overwritten: binding a receipt to an image it is demonstrably not about
    would reintroduce, inside the binding step, the exact gap this contract
    exists to close.
    """
    errors = validate_visual_generation_receipt(receipt)
    if errors:
        return errors
    faults: list[str] = []
    if str(receipt.get("card_id", "")) != str(card_id):
        faults.append(
            f"{_LABEL} card_id {receipt.get('card_id')!r} does not match the observed card {card_id!r}"
        )
    outcome = str(receipt.get("outcome", ""))
    if outcome != "succeeded":
        faults.append(
            f"a {outcome} visual generation attempt is not generated-image evidence; "
            "record the receipt and leave the observation unmade"
        )
    receipt_artifact = receipt.get("artifact")
    receipt_artifact = receipt_artifact if isinstance(receipt_artifact, Mapping) else {}
    observed_artifact = artifact if isinstance(artifact, Mapping) else {}
    if observed_artifact:
        observed_mime = str(observed_artifact.get("mime_type", ""))
        receipt_mime = str(receipt_artifact.get("mime_type", ""))
        if observed_mime and receipt_mime and observed_mime != receipt_mime:
            faults.append(
                f"the observed artifact is {observed_mime} and the receipt reports {receipt_mime}; "
                "a receipt binds the image it is about, not another one"
            )
        observed_digest = str(observed_artifact.get("content_sha256", ""))
        receipt_digest = str(receipt_artifact.get("content_sha256", ""))
        if observed_digest and receipt_digest and observed_digest != receipt_digest:
            faults.append(
                "the observed artifact already carries a different content digest than the receipt; "
                "a binding never overwrites a digest someone else reported"
            )
    return faults


def bind_visual_generation_receipt(
    observation: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach one succeeded receipt to a generated-image observation.

    The binding is a reference plus the two facts a reader needs in order not to
    overclaim from the observation alone: the attempt it belongs to and the
    digest of the bytes it is about. The route itself is deliberately not copied
    -- it lives on the receipt, where the attested set travels with it, so an
    observation can never carry a route without the attestation that qualifies
    it.
    """
    card_id = str(observation.get("visual_card_id", ""))
    if str(observation.get("observation_type", "")) != "generated_image_observed":
        raise VisualGenerationReceiptError(
            "a visual generation receipt binds a generated_image_observed observation only; "
            "visual QA and delivery are separate evidence states"
        )
    stored_artifact = observation.get("artifact")
    faults = generated_image_binding_errors(
        receipt,
        card_id=card_id,
        artifact=stored_artifact if isinstance(stored_artifact, Mapping) else None,
    )
    if faults:
        raise VisualGenerationReceiptError(faults[0])
    artifact = receipt.get("artifact")
    artifact = artifact if isinstance(artifact, Mapping) else {}
    binding = {
        "receipt_id": str(receipt.get("receipt_id", "")),
        "effect_id": str(receipt.get("effect_id", "")),
        "attempt_id": str(receipt.get("attempt_id", "")),
        "action_id": str(receipt.get("action_id", "")),
        "card_digest": str(receipt.get("card_digest", "")),
        "content_sha256": str(artifact.get("content_sha256", "")),
        "byte_size": artifact.get("byte_size"),
        "producer": str(receipt.get("producer", "")),
    }
    if sorted(binding) != sorted(VISUAL_GENERATION_BINDING_KEYS):
        raise VisualGenerationReceiptError(
            "visual generation binding keys must match the observation contract"
        )
    bound = dict(observation)
    bound["generation_receipt"] = binding
    if isinstance(stored_artifact, Mapping):
        merged = dict(stored_artifact)
        merged["content_sha256"] = binding["content_sha256"]
        merged["byte_size"] = binding["byte_size"]
        bound["artifact"] = merged
    # The observation stops saying it cannot name a route the moment a receipt
    # names one. Everything else it does not prove -- visual QA, delivery -- is
    # untouched, because a receipt is not evidence for either of those.
    claims = bound.get("does_not_prove")
    if isinstance(claims, list):
        bound["does_not_prove"] = [claim for claim in claims if claim != "generation_route_attested"]
    observation_errors = validate_visual_observation(bound)
    if observation_errors:
        raise VisualGenerationReceiptError(observation_errors[0])
    return bound


def project_legacy_visual_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Read a `visual_observation/v1` record through the receipt contract.

    A record written before receipts existed, or one recorded without a
    producer, still has to be readable. It is projected with every route,
    attempt, lineage, digest, and usage field unknown, because that is what it
    says: a path and a MIME type were reported. Nothing here infers a provider
    from configuration or a digest from the file being on disk.
    """
    binding = observation.get("generation_receipt")
    binding = binding if isinstance(binding, Mapping) else {}
    artifact = observation.get("artifact")
    artifact = artifact if isinstance(artifact, Mapping) else {}
    return {
        "schema_version": VISUAL_GENERATION_RECEIPT_PROJECTION_SCHEMA_VERSION,
        "source_schema": str(observation.get("schema_version", "")),
        "observation_id": str(observation.get("observation_id", "")),
        "card_id": str(observation.get("visual_card_id", "")),
        "observation_type": str(observation.get("observation_type", "")),
        "observed_at": str(observation.get("observed_at", "")),
        "receipt_id": str(binding.get("receipt_id", "")),
        "effect_id": str(binding.get("effect_id", "")),
        "attempt_id": str(binding.get("attempt_id", "")) or UNKNOWN,
        "action_id": str(binding.get("action_id", "")) or UNKNOWN,
        "card_digest": str(binding.get("card_digest", "")) or UNKNOWN,
        "producer": str(binding.get("producer", "")) or UNKNOWN,
        "requested_route": empty_route(),
        "observed_route": empty_route(),
        "attested_route_fields": [],
        "artifact": {
            "artifact_ref": UNKNOWN,
            "content_sha256": str(artifact.get("content_sha256", "")) or UNKNOWN,
            "mime_type": str(artifact.get("mime_type", "")) or UNKNOWN,
            "byte_size": artifact.get("byte_size") if isinstance(artifact.get("byte_size"), int) else None,
            "provider_response_ref": UNKNOWN,
        },
        "input_lineage": {"source_image_count": None, "source_image_digests": [], "edit_constraints": []},
        "usage": {name: {"value": None, "measurement": "unavailable"} for name in USAGE_METRIC_NAMES},
        "route_claim_boundary": ROUTE_CLAIM_BOUNDARY,
        "projection_note": (
            "Projected from a visual_observation/v1 record. Unknown fields were never observed; "
            "they are not defaults and must not be filled in from configuration or file existence."
        ),
    }


# --- store ---------------------------------------------------------------


def record_visual_generation_receipt(paths: OmhPaths, **fields: Any) -> tuple[dict[str, Any], bool]:
    """The receipt now on record for this attempt, and whether this call wrote it.

    Re-reporting an attempt identically appends nothing and returns the receipt
    already on record, so a producer that retries its report still has exactly
    one receipt. A report that differs -- a later outcome, an attested field the
    producer learned, replaced bytes at the same reference -- appends a new
    receipt linked to its predecessor through `supersedes_receipt_ref`. Nothing
    on disk is ever rewritten.

    The read of the prior receipt and the append happen under one lock, so two
    concurrent reports of one attempt cannot both link to the same predecessor.
    """
    path = paths.visual_generation_receipts_path
    with file_lock(path, private=True):
        # The same normalisation `build_visual_generation_receipt` applies, so a
        # padded identifier finds its own predecessor instead of minting a
        # duplicate attempt beside it.
        prior = latest_receipt_for_effect(
            _read_receipts(path),
            visual_generation_effect_id(
                str(fields.get("card_id", "")).strip(),
                str(fields.get("action_id", "")).strip(),
                str(fields.get("attempt_id", "")).strip(),
            ),
        )
        record = build_visual_generation_receipt(
            supersedes_receipt_ref=str(prior.get("receipt_id", "")) if prior else "",
            **fields,
        )
        if prior and _attempt_fingerprint(prior) == _attempt_fingerprint(record):
            return prior, False
        append_store_line(path, record)
    return record, True


def read_visual_generation_receipts(
    paths: OmhPaths,
    *,
    card_id: str | None = None,
    effect_id: str | None = None,
) -> list[dict[str, Any]]:
    receipts = _read_receipts(paths.visual_generation_receipts_path)
    if card_id is not None:
        receipts = [receipt for receipt in receipts if str(receipt.get("card_id", "")) == card_id]
    if effect_id is not None:
        receipts = [receipt for receipt in receipts if str(receipt.get("effect_id", "")) == effect_id]
    return receipts


def read_visual_generation_receipt(paths: OmhPaths, receipt_id: str) -> dict[str, Any]:
    """The receipt with this id, or an empty mapping."""
    return latest_record_in(
        _read_receipts(paths.visual_generation_receipts_path), key="receipt_id", value=str(receipt_id)
    )


def latest_receipt_for_effect(
    receipts: Sequence[Mapping[str, Any]],
    effect_id: str,
) -> dict[str, Any]:
    """The receipt that speaks for one attempt: the one selection rule.

    The store is append-only, so arrival order is history and the last record
    for an attempt is its current state. Every surface that judges an attempt
    selects through this function, which is what stops a validator and a status
    renderer from judging different receipts for one attempt.
    """
    return latest_record_in(receipts, key="effect_id", value=str(effect_id))


def validate_visual_generation_receipt_store(path: Path, *, card_id: str | None = None) -> dict[str, Any]:
    """Validate the store, optionally scoped to one card's receipts."""
    receipts, read_errors = read_jsonl_objects(path)
    indexed = [
        record for record in receipts if card_id is None or str(record.get("card_id", "")) == card_id
    ]
    errors: list[str] = list(read_errors) if card_id is None else []
    for index, record in enumerate(receipts, start=1):
        if card_id is not None and str(record.get("card_id", "")) != card_id:
            continue
        errors.extend(f"{path}:{index}: {error}" for error in validate_visual_generation_receipt(record))
    # The chain is walked over the whole store even when the report is scoped to
    # one card: a link into a receipt for another card is still a link into a
    # receipt that exists, and a chain fault is the store's fault rather than
    # any one card's. Scoping narrows which records are schema-checked, never
    # which chain is walked.
    errors.extend(supersede_chain_errors(path, receipts, run_id=None, label=_LABEL))
    return {
        "schema_version": VISUAL_GENERATION_RECEIPT_STORE_VALIDATION_SCHEMA_VERSION,
        "path": str(path),
        "card_id": card_id or "",
        "ok": not errors,
        "receipt_count": len(indexed),
        "errors": errors,
    }


def parse_usage_arg(value: str) -> tuple[str, dict[str, Any]]:
    """One `NAME=VALUE:MEASUREMENT` usage argument.

    The measurement is not optional and `unavailable` is not accepted: a
    producer that has no figure omits the argument, and an omitted metric
    already reads as unavailable. Requiring the word is what stops a caller
    from passing a bare number whose measured-or-estimated meaning nobody
    recorded.
    """
    text = str(value).strip()
    name, separator, remainder = text.partition("=")
    figure, colon, measurement = remainder.partition(":")
    if not separator or not colon:
        raise ValueError("--usage must use NAME=VALUE:MEASUREMENT with exactly one '=' and one ':'")
    name = name.strip()
    measurement = measurement.strip()
    if name not in USAGE_METRIC_NAMES:
        raise ValueError(f"--usage name must be one of {', '.join(USAGE_METRIC_NAMES)}")
    if measurement not in ("measured", "estimated"):
        raise ValueError(
            "--usage measurement must be measured or estimated; omit the metric when it is unavailable"
        )
    try:
        number: float | int = int(figure.strip()) if name in INTEGER_USAGE_METRICS else float(figure.strip())
    except ValueError as exc:
        raise ValueError(f"--usage value for {name} must be a number: {figure.strip()!r}") from exc
    return name, {"value": number, "measurement": measurement}


def bounded_receipt_summary(value: str) -> str:
    """One bounded metadata line, or nothing.

    A summary that carries a link, a path, a secret, or a control character is
    dropped rather than truncated: the value is not a summary that ran long, it
    is the wrong kind of value.
    """
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    if is_unsafe_metadata_line(text):
        return "[redacted]"
    return text[:MAX_SUMMARY_CHARS]


# --- internals -----------------------------------------------------------


def _opaque(value: str, *, field: str) -> str:
    return opaque_ref(value, field=field, error=VisualGenerationReceiptError)


def _handle(value: str, *, field: str) -> str:
    """One opaque handle that is also not a filesystem path."""
    text = _opaque(value, field=field)
    if _PATH_SHAPED.search(text):
        raise VisualGenerationReceiptError(f"{field} must be an opaque handle, not a filesystem path")
    return text


def _handle_errors(value: Any, *, field: str, required: bool) -> list[str]:
    """The validation-time counterpart of `_handle`: collects rather than raises."""
    errors = reference_errors(value, field=field, label=_LABEL, required=required)
    if errors or not isinstance(value, str) or not value:
        return errors
    if _PATH_SHAPED.search(value):
        return [f"{_LABEL} {field} must be an opaque handle, not a filesystem path"]
    return []


def _read_receipts(path: Path) -> list[dict[str, Any]]:
    receipts, _ = read_jsonl_objects(path)
    return receipts


def _attempt_fingerprint(record: Mapping[str, Any]) -> str:
    return record_fingerprint(record, _ATTEMPT_IDENTITY_KEYS)


def _normalized_attested(values: Sequence[str]) -> list[str]:
    return [field for field in ROUTE_FIELDS if field in {str(value) for value in values}]


def _normalized_route(route: Mapping[str, str]) -> dict[str, str]:
    return {field: (str(route.get(field, "")).strip() or UNKNOWN) for field in ROUTE_FIELDS}


def _normalized_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    byte_size = artifact.get("byte_size")
    for field in ("artifact_ref", "provider_response_ref"):
        value = str(artifact.get(field, "")).strip()
        if value:
            _handle(value, field=f"{_LABEL} artifact {field}")
    return {
        "artifact_ref": str(artifact.get("artifact_ref", "")).strip(),
        "content_sha256": str(artifact.get("content_sha256", "")).strip().lower(),
        "mime_type": str(artifact.get("mime_type", "")).strip().lower(),
        "byte_size": byte_size if isinstance(byte_size, int) and not isinstance(byte_size, bool) else None,
        "provider_response_ref": str(artifact.get("provider_response_ref", "")).strip(),
    }


def _normalized_lineage(lineage: Mapping[str, Any]) -> dict[str, Any]:
    count = lineage.get("source_image_count", 0)
    digests = lineage.get("source_image_digests", ())
    constraints = lineage.get("edit_constraints", ())
    # A malformed value is passed through, not normalised away. Silently
    # dropping a source digest a producer got wrong would turn a producer error
    # into a receipt that positively asserts there were no source images.
    return {
        "source_image_count": count if isinstance(count, int) and not isinstance(count, bool) else count,
        "source_image_digests": [str(value).strip().lower() if isinstance(value, str) else value for value in digests]
        if isinstance(digests, (list, tuple))
        else digests,
        "edit_constraints": list(constraints) if isinstance(constraints, (list, tuple)) else constraints,
    }


def _normalized_usage(usage: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    readings: dict[str, dict[str, Any]] = {}
    for name in USAGE_METRIC_NAMES:
        reading = usage.get(name)
        if not isinstance(reading, Mapping):
            continue
        measurement = str(reading.get("measurement", "")).strip()
        if measurement == "unavailable" or not measurement:
            continue
        readings[name] = {"value": reading.get("value"), "measurement": measurement}
    return readings


def _bounded_refs(refs: Sequence[str]) -> list[str]:
    return [
        _handle(str(value), field=f"{_LABEL} evidence_refs[{index}]")
        for index, value in enumerate(list(refs)[:MAX_EVIDENCE_REFS])
    ]


def _outcome_errors(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    outcome = record.get("outcome")
    stage = record.get("failure_stage")
    if outcome not in RECEIPT_OUTCOMES:
        errors.append(f"{_LABEL} outcome is unsupported: {outcome!r}")
    if stage not in FAILURE_STAGES:
        errors.append(f"{_LABEL} failure_stage is unsupported: {stage!r}")
    if outcome in RECEIPT_OUTCOMES and stage in FAILURE_STAGES:
        if outcome == "succeeded" and stage != "none":
            errors.append(f"{_LABEL} succeeded must carry failure_stage none")
        if outcome != "succeeded" and stage == "none":
            errors.append(
                f"{_LABEL} {outcome} must name the stage it stopped at; "
                f"expected one of {', '.join(name for name in FAILURE_STAGES if name != 'none')}"
            )
    return errors


def _route_errors(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    attested = record.get("attested_route_fields")
    if not isinstance(attested, list):
        errors.append(f"{_LABEL} attested_route_fields must be a list")
        attested_set: set[str] = set()
    else:
        unsupported = sorted({str(value) for value in attested} - set(ROUTE_FIELDS))
        if unsupported:
            errors.append(f"{_LABEL} attested_route_fields has unsupported fields: {unsupported}")
        if attested != _normalized_attested(attested):
            errors.append(
                f"{_LABEL} attested_route_fields must be unique and in route field order: {list(ROUTE_FIELDS)}"
            )
        attested_set = {str(value) for value in attested} & set(ROUTE_FIELDS)
    for key, required_operation in (("requested_route", True), ("observed_route", False)):
        route = record.get(key)
        if not isinstance(route, Mapping):
            errors.append(f"{_LABEL} {key} must be an object")
            continue
        unsupported = sorted(set(route) - set(ROUTE_FIELDS))
        if unsupported:
            errors.append(f"{_LABEL} {key} has unsupported fields: {unsupported}")
        missing = sorted(set(ROUTE_FIELDS) - set(route))
        if missing:
            errors.append(f"{_LABEL} {key} is missing fields: {missing}")
        for field in ROUTE_FIELDS:
            if field not in route:
                continue
            errors.extend(_route_field_errors(route[field], key=key, field=field))
        operation = route.get("operation")
        if required_operation and operation not in OPERATIONS:
            errors.append(
                f"{_LABEL} requested_route operation must be one of {', '.join(OPERATIONS)}; "
                "a request always states whether it generates or edits"
            )
        if not required_operation and operation not in OPERATIONS and operation != UNKNOWN:
            errors.append(
                f"{_LABEL} observed_route operation must be one of {', '.join((*OPERATIONS, UNKNOWN))}"
            )
    observed = record.get("observed_route")
    if isinstance(observed, Mapping):
        for field in ROUTE_FIELDS:
            value = str(observed.get(field, UNKNOWN))
            if field in attested_set and value == UNKNOWN:
                errors.append(
                    f"{_LABEL} observed_route {field} is attested but unknown; "
                    "attesting nothing is not an attestation"
                )
            if field not in attested_set and value != UNKNOWN:
                errors.append(
                    f"{_LABEL} observed_route {field} must be unknown unless it is named in "
                    "attested_route_fields; a returned file does not confirm a requested route"
                )
    return errors


def _route_field_errors(value: Any, *, key: str, field: str) -> list[str]:
    if not isinstance(value, str):
        return [f"{_LABEL} {key} {field} must be a string"]
    if field == "credential_class":
        if value not in CREDENTIAL_CLASSES:
            return [
                f"{_LABEL} {key} credential_class is unsupported: {value!r}; "
                f"expected one of {', '.join(CREDENTIAL_CLASSES)}"
            ]
        return []
    if field == "operation":
        return []
    if value == UNKNOWN:
        return []
    return reference_errors(value, field=f"{key} {field}", label=_LABEL, required=True)


def _artifact_errors(record: Mapping[str, Any]) -> list[str]:
    artifact = record.get("artifact")
    if not isinstance(artifact, Mapping):
        return [f"{_LABEL} artifact must be an object"]
    errors: list[str] = []
    unsupported = sorted(set(artifact) - set(ARTIFACT_KEYS))
    if unsupported:
        errors.append(f"{_LABEL} artifact has unsupported keys: {unsupported}")
    missing = sorted(set(ARTIFACT_KEYS) - set(artifact))
    if missing:
        errors.append(f"{_LABEL} artifact is missing keys: {missing}")
    succeeded = record.get("outcome") == "succeeded"
    digest = artifact.get("content_sha256")
    mime = artifact.get("mime_type")
    byte_size = artifact.get("byte_size")
    ref = artifact.get("artifact_ref")
    response_ref = artifact.get("provider_response_ref")
    if not succeeded:
        for field, empty in (
            ("artifact_ref", ""),
            ("content_sha256", ""),
            ("mime_type", ""),
            ("byte_size", None),
            ("provider_response_ref", ""),
        ):
            if field in artifact and artifact[field] != empty:
                errors.append(
                    f"{_LABEL} artifact {field} must be empty unless the attempt succeeded; "
                    "a failed or partial attempt is not image evidence"
                )
        return errors
    if not isinstance(digest, str) or not _DIGEST.match(digest):
        errors.append(f"{_LABEL} artifact content_sha256 must be a sha256 hex digest of the returned bytes")
    if mime not in SUPPORTED_IMAGE_MIME_TYPES:
        errors.append(f"{_LABEL} artifact mime_type must be one of {', '.join(SUPPORTED_IMAGE_MIME_TYPES)}")
    if not isinstance(byte_size, int) or isinstance(byte_size, bool) or byte_size < 1:
        errors.append(f"{_LABEL} artifact byte_size must be a positive integer")
    elif byte_size > MAX_ARTIFACT_BYTES:
        errors.append(f"{_LABEL} artifact byte_size must be at most {MAX_ARTIFACT_BYTES}")
    errors.extend(_handle_errors(ref, field="artifact artifact_ref", required=True))
    errors.extend(_handle_errors(response_ref, field="artifact provider_response_ref", required=False))
    return errors


def _lineage_errors(record: Mapping[str, Any]) -> list[str]:
    lineage = record.get("input_lineage")
    if not isinstance(lineage, Mapping):
        return [f"{_LABEL} input_lineage must be an object"]
    errors: list[str] = []
    unsupported = sorted(set(lineage) - set(INPUT_LINEAGE_KEYS))
    if unsupported:
        errors.append(f"{_LABEL} input_lineage has unsupported keys: {unsupported}")
    missing = sorted(set(INPUT_LINEAGE_KEYS) - set(lineage))
    if missing:
        errors.append(f"{_LABEL} input_lineage is missing keys: {missing}")
    count = lineage.get("source_image_count")
    digests = lineage.get("source_image_digests")
    constraints = lineage.get("edit_constraints")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        errors.append(f"{_LABEL} input_lineage source_image_count must be a non-negative integer")
        count = None
    elif count > MAX_SOURCE_IMAGES:
        errors.append(f"{_LABEL} input_lineage source_image_count must be at most {MAX_SOURCE_IMAGES}")
    if not isinstance(digests, list):
        errors.append(f"{_LABEL} input_lineage source_image_digests must be a list")
        digests = None
    else:
        if len(digests) > MAX_SOURCE_IMAGES:
            errors.append(f"{_LABEL} input_lineage source_image_digests must have at most {MAX_SOURCE_IMAGES} items")
        for index, value in enumerate(digests):
            if not isinstance(value, str) or not _DIGEST.match(value):
                errors.append(
                    f"{_LABEL} input_lineage source_image_digests[{index}] must be a sha256 hex digest"
                )
    if not isinstance(constraints, list):
        errors.append(f"{_LABEL} input_lineage edit_constraints must be a list")
        constraints = None
    else:
        unsupported_kinds = sorted({str(value) for value in constraints} - set(EDIT_CONSTRAINT_KINDS))
        if unsupported_kinds:
            errors.append(f"{_LABEL} input_lineage edit_constraints has unsupported kinds: {unsupported_kinds}")
    operation = UNKNOWN
    requested = record.get("requested_route")
    if isinstance(requested, Mapping):
        operation = str(requested.get("operation", UNKNOWN))
    if operation == "generate":
        if count:
            errors.append(f"{_LABEL} a generate request has no source images; source_image_count must be 0")
        if digests:
            errors.append(f"{_LABEL} a generate request has no source images; source_image_digests must be empty")
        if constraints:
            errors.append(
                f"{_LABEL} a generate request has no preserve, remove, or replace constraints; "
                "edit_constraints must be empty"
            )
    elif operation == "edit":
        if count is not None and count < 1:
            errors.append(f"{_LABEL} an edit request names at least one source image; source_image_count must be 1 or more")
        if isinstance(digests, list) and digests and count is not None and len(digests) != count:
            errors.append(
                f"{_LABEL} input_lineage source_image_digests must be empty or name every source image; "
                f"got {len(digests)} digests for {count} source images"
            )
    return errors


def _usage_errors(record: Mapping[str, Any]) -> list[str]:
    usage = record.get("usage")
    if not isinstance(usage, Mapping):
        return [f"{_LABEL} usage must be an object"]
    errors: list[str] = []
    unsupported = sorted(set(usage) - set(USAGE_METRIC_NAMES))
    if unsupported:
        errors.append(f"{_LABEL} usage has unsupported metric names: {unsupported}")
    for name in USAGE_METRIC_NAMES:
        if name not in usage:
            continue
        reading = usage[name]
        path = f"usage {name}"
        if not isinstance(reading, Mapping):
            errors.append(f"{_LABEL} {path} must be an object")
            continue
        unsupported_keys = sorted(set(reading) - set(USAGE_READING_KEYS))
        if unsupported_keys:
            errors.append(f"{_LABEL} {path} has unsupported keys: {unsupported_keys}")
        missing_keys = sorted(set(USAGE_READING_KEYS) - set(reading))
        if missing_keys:
            errors.append(f"{_LABEL} {path} is missing keys: {missing_keys}")
        measurement = reading.get("measurement")
        if measurement not in USAGE_MEASUREMENTS:
            errors.append(f"{_LABEL} {path} measurement is unsupported: {measurement!r}")
            continue
        if measurement == "unavailable":
            errors.append(
                f"{_LABEL} {path} must be omitted rather than stored as unavailable; "
                "an absent metric already reads as unavailable"
            )
            continue
        value = reading.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{_LABEL} {path} value must be a number when it is {measurement}")
            continue
        if value < 0 or value > MAX_USAGE_VALUE:
            errors.append(f"{_LABEL} {path} value must be from 0 to {MAX_USAGE_VALUE}")
        elif name in INTEGER_USAGE_METRICS and not isinstance(value, int):
            errors.append(f"{_LABEL} {path} value must be an integer")
    return errors


def _refs_errors(record: Mapping[str, Any]) -> list[str]:
    refs = record.get("evidence_refs")
    if not isinstance(refs, list):
        return [f"{_LABEL} evidence_refs must be a list"]
    errors: list[str] = []
    if len(refs) > MAX_EVIDENCE_REFS:
        errors.append(f"{_LABEL} evidence_refs must have at most {MAX_EVIDENCE_REFS} items")
    for index, value in enumerate(refs):
        errors.extend(_handle_errors(value, field=f"evidence_refs[{index}]", required=True))
    return errors


def _text_errors(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in _RECEIPT_TEXT_FIELDS:
        text = record.get(field)
        if not isinstance(text, str):
            continue
        if len(text) > MAX_SUMMARY_CHARS:
            errors.append(f"{_LABEL} {field} must be at most {MAX_SUMMARY_CHARS} characters")
        if is_unsafe_metadata_line(text):
            errors.append(
                f"{_LABEL} {field} must not carry secrets, links, paths, or raw text; "
                "it is one bounded metadata line"
            )
    return errors
