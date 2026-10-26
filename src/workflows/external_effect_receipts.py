"""Append-only receipts for external effects an acting surface actually observed.

An external effect is something that happened outside this machine: a message
that reached a chat platform, a review that landed on a change, a CI run that
executed, a merge that moved a branch. OMH performs none of them. It can only
record that a surface which *does* act observed one, and a receipt is that
record.

The store is therefore mint-restricted, in the same shape as the
prepared-vs-observed handshake in `adapter_quality`:

- `build_external_effect_receipt` refuses `observed_result="requested"`
  outright, and every producer refuses to mint from a record whose own
  `observed` flag is false, so no prepared, requested, or merely intended
  record can become a receipt.
- `requested` and `attempted` are reportable states that are *projected* over
  the absence of a receipt for an effect a run asked for (`PROJECTED_RESULTS`).
  They are never minted from an unobserved record.
- `acting_surface` is a closed vocabulary of surfaces that already require
  observed evidence before they write. There is no free-text surface and no
  CLI that mints a receipt from operator input.
- `succeeded` requires a non-empty `external_ref`. A surface that observed
  success but cannot name what it succeeded at records `unknown`, because an
  unnameable success is not a citable one.

Retries and reversals link (`supersedes_receipt_ref`) instead of overwriting.
The store is a JSONL file appended under `file_lock`, whose OS lock is real on
both POSIX and Windows, so history is structurally immutable on either. A short
write that leaves a partial line cannot swallow the next record: the append
terminates an unterminated tail before writing.

Everything on a receipt is metadata: bounded identifiers, a closed-vocabulary
result, and one redacted summary line. No prompt, payload, transcript, URL, or
filesystem path can be stored, and `validate_external_effect_receipt` rejects
the key names those arrive under as well as the shapes they arrive in.

The guards are declared by field *class* rather than by field name (see the
class tables beside `EXTERNAL_EFFECT_RECEIPT_KEYS`) and applied in all three
places a receipt is handled -- built, validated, and rendered -- because a guard
that covers the fields someone thought of is how the next field arrives without
one.

One predicate, `receipt_satisfies_success_claim`, decides whether a receipt
backs a success claim, and one selector, `select_effect_receipt`, decides which
receipt that predicate is asked about. Runtime validation and the claim ladder
call both, so the two can never disagree about what a receipt proves or judge
different receipts for the same effect.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..system.append_only_store import (
    RAW_OR_HIDDEN_KEYS,
    append_sidecar_line,
    append_store_line,
    closed_vocabulary_value,
    is_unsafe_metadata_line,
    latest_record_in,
    mint_record_id,
    opaque_ref,
    record_fingerprint,
    redacted_ref,
    reference_errors,
    store_errors,
)
from ..system.local_store import file_lock, read_jsonl_objects, utc_now
from ..plugin_bundle.omh.run_records import (
    EXTERNAL_EFFECT_CLAIM_BOUNDARY as CLAIM_BOUNDARY,
    EXTERNAL_EFFECT_RECEIPT_SCHEMA_VERSION,
    # Re-exported (`X as X`): the runtime store registry imports it from here.
    EXTERNAL_EFFECT_RECEIPT_STORE_NAME as EXTERNAL_EFFECT_RECEIPT_STORE_NAME,
)
from ..system.paths import OmhPaths


EXTERNAL_EFFECT_PROJECTION_SCHEMA_VERSION = "external_effect_projection/v1"
EXTERNAL_EFFECT_RECEIPT_STORE_VALIDATION_SCHEMA_VERSION = "external_effect_receipt_store_validation/v1"
EXTERNAL_EFFECT_MINT_RESULT_SCHEMA_VERSION = "external_effect_mint_result/v1"
EXTERNAL_EFFECT_MINT_FAILURE_SCHEMA_VERSION = "external_effect_mint_failure/v1"
RECEIPT_PRIVACY = "metadata_only"
EXTERNAL_EFFECT_MINT_FAILURE_STORE_NAME = "external_effect_mint_failures.jsonl"

# Only effects a surface in `ACTING_SURFACES` genuinely produces today. Adding
# an action without a producer would put a state in status reports that nothing
# can ever reach.
ACTIONS = ("message_sent", "review_submitted", "ci_run", "merge", "browser_mutation")
TARGET_CLASSES = ("chat_channel", "code_review", "ci_provider", "repository", "endpoint")
ACTION_TARGET_CLASSES = {
    "message_sent": "chat_channel",
    "review_submitted": "code_review",
    "ci_run": "ci_provider",
    "merge": "repository",
    "browser_mutation": "endpoint",
}

# The five states AC1 of issue #836 requires status reports to tell apart.
#   requested  -- the effect was asked for; no acting surface reported it.
#   attempted  -- the effect started; no acting surface reported an outcome.
#   succeeded  -- a surface observed the effect complete, and named it.
#   failed     -- a surface observed the effect not happen.
#   unknown    -- a surface observed a terminal state it cannot classify, or
#                 observed success it cannot name.
OBSERVED_RESULTS = ("requested", "attempted", "succeeded", "failed", "unknown")
# `requested` is deliberately absent: a request is not an observation.
RECEIPT_RESULTS = ("attempted", "succeeded", "failed", "unknown")
# The two states a status report may reach with no receipt at all. Both are
# projected over the absence of one, so neither can be mistaken for evidence.
PROJECTED_RESULTS = ("requested", "attempted")

# Surfaces that already require observed evidence before they write. A receipt
# names one of these, so every success claim can say who acted (AC2).
ACTING_SURFACES = (
    "adapter_quality_delivery",
    "runtime_review_record",
    "runtime_ci_record",
    "runtime_merge_record",
    "browser_adapter_readback",
)

# Which action and which acting surface a run-scoped effect must carry to back a
# success claim about that gate. A `merge` receipt cannot satisfy a CI claim and
# a receipt from another surface cannot satisfy either.
EXTERNAL_EFFECT_CLAIM_SURFACES = {
    "review": ("review_submitted", "runtime_review_record"),
    "ci": ("ci_run", "runtime_ci_record"),
    "merge": ("merge", "runtime_merge_record"),
}

EXTERNAL_EFFECT_RECEIPT_KEYS = (
    "acting_surface",
    "action",
    "claim_boundary",
    "effect_id",
    "evidence_refs",
    "external_ref",
    "observed_at",
    "observed_result",
    "privacy",
    "receipt_id",
    "run_id",
    "schema_version",
    "summary",
    "supersedes_receipt_ref",
    "target_class",
)

# Which guard each string field on a receipt gets, by class rather than by
# name. Guarding the two fields a review happened to notice is how a third
# arrives unguarded: `receipt_id` is what every AC2 citation is built from, and
# a `receipt_id` carrying `\x1b[2K\r` would otherwise validate clean and print
# straight into one. Every field below is enforced in all three places a receipt
# is handled -- build, validate, and render -- so no path is the loose one.
#
# Identifiers are opaque references (`require_opaque_metadata_ref`): bounded,
# non-navigable, and free of control characters by construction.
_REQUIRED_RECEIPT_REFS = ("receipt_id", "effect_id", "observed_at")
# Identifiers that are legitimately absent. `run_id` is empty for a session-
# scoped effect such as an adapter delivery; the other two are empty until
# there is something to name.
_OPTIONAL_RECEIPT_REFS = ("external_ref", "run_id", "supersedes_receipt_ref")
# Closed vocabularies, checked by membership everywhere including rendering.
_RECEIPT_VOCABULARIES = (
    ("action", ACTIONS),
    ("acting_surface", ACTING_SURFACES),
    ("observed_result", RECEIPT_RESULTS),
    ("target_class", TARGET_CLASSES),
)
# Free text, through the URL / secret / path / control-character guard.
_RECEIPT_TEXT_FIELDS = ("summary",)
# Every string field, in the order the string-type check reports them. Exactly
# the union of the four classes above, and exactly the receipt keys that are
# not a container or a fixed constant -- `tests/test_external_effect_receipts.py`
# holds both halves of that so a new field cannot be added without a class.
_RECEIPT_STRING_FIELDS = (
    "receipt_id",
    "effect_id",
    "run_id",
    "action",
    "target_class",
    "acting_surface",
    "observed_result",
    "observed_at",
    "external_ref",
    "summary",
    "supersedes_receipt_ref",
)

# Everything that identifies *which observation* a receipt records. The
# timestamp, the receipt id, and the supersede link are excluded on purpose:
# re-recording the same observation must not mint a second receipt just because
# the clock moved.
_OBSERVATION_IDENTITY_KEYS = (
    "acting_surface",
    "action",
    "effect_id",
    "evidence_refs",
    "external_ref",
    "observed_result",
    "run_id",
    "summary",
)

MAX_SUMMARY_CHARS = 200
MAX_EVIDENCE_REFS = 8

REDACTED_SUMMARY = "[redacted]"

# The store label every error line and every reference fault is reported under.
_LABEL = "external_effect_receipt"


class ExternalEffectReceiptError(ValueError):
    """Raised when a receipt cannot be minted from what a surface observed."""


def external_effect_id(kind: str, run_id: str) -> str:
    """Stable identity of one logical external effect on one run."""
    return f"{kind}:{run_id}"


def build_external_effect_receipt(
    *,
    effect_id: str,
    action: str,
    acting_surface: str,
    observed_result: str,
    run_id: str = "",
    external_ref: str = "",
    evidence_refs: list[str] | tuple[str, ...] = (),
    summary: str = "",
    observed_at: str = "",
    supersedes_receipt_ref: str = "",
) -> dict[str, Any]:
    """Mint one receipt, or refuse.

    Every refusal here is the invariant this module exists for: a receipt can
    only describe something an acting surface observed.
    """
    if observed_result == "requested":
        raise ExternalEffectReceiptError(
            "a requested external effect is not an observed receipt; receipts require an observed result"
        )
    if observed_result not in RECEIPT_RESULTS:
        raise ExternalEffectReceiptError(f"external_effect_receipt observed_result is unsupported: {observed_result!r}")
    if action not in ACTIONS:
        raise ExternalEffectReceiptError(f"external_effect_receipt action is unsupported: {action!r}")
    if acting_surface not in ACTING_SURFACES:
        raise ExternalEffectReceiptError(
            f"external_effect_receipt acting_surface is unsupported: {acting_surface!r}"
        )
    safe_effect_id = _opaque(effect_id, field="external_effect_receipt effect_id")
    safe_external_ref = _opaque(external_ref, field="external_effect_receipt external_ref") if external_ref else ""
    # A supersede link is a reference like any other, so it goes through the
    # same opaque-reference guard instead of being copied in verbatim.
    safe_supersedes = (
        _opaque(supersedes_receipt_ref, field="external_effect_receipt supersedes_receipt_ref")
        if supersedes_receipt_ref
        else ""
    )
    # `run_id` and `observed_at` are identifiers like every other reference on a
    # receipt, so they go through the same guard rather than being copied in as
    # whatever the caller held. A caller that passes a control character in
    # either is refused here, by name, instead of one field further on.
    safe_run_id = _opaque(run_id, field="external_effect_receipt run_id") if run_id else ""
    if observed_result == "succeeded" and not safe_external_ref:
        raise ExternalEffectReceiptError(
            "external_effect_receipt succeeded requires an external_ref naming the observed effect"
        )
    stamp = _opaque(str(observed_at or utc_now()), field="external_effect_receipt observed_at")
    record = {
        "schema_version": EXTERNAL_EFFECT_RECEIPT_SCHEMA_VERSION,
        "receipt_id": _receipt_id(stamp, safe_effect_id, acting_surface, observed_result),
        "effect_id": safe_effect_id,
        "run_id": safe_run_id,
        "action": action,
        "target_class": ACTION_TARGET_CLASSES[action],
        "acting_surface": acting_surface,
        "observed_result": observed_result,
        "observed_at": stamp,
        "external_ref": safe_external_ref,
        "evidence_refs": _bounded_refs(evidence_refs),
        "summary": bounded_receipt_summary(summary),
        "supersedes_receipt_ref": safe_supersedes,
        "privacy": RECEIPT_PRIVACY,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    errors = validate_external_effect_receipt(record)
    if errors:
        raise ExternalEffectReceiptError(errors[0])
    return record


def validate_external_effect_receipt(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["external_effect_receipt must be an object"]
    forbidden = sorted(key for key in record if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if forbidden:
        errors.append(f"external_effect_receipt must not carry raw or hidden keys: {forbidden}")
    extra_keys = sorted(set(record) - set(EXTERNAL_EFFECT_RECEIPT_KEYS) - set(forbidden))
    if extra_keys:
        errors.append(f"external_effect_receipt has unsupported keys: {extra_keys}")
    missing = sorted(set(EXTERNAL_EFFECT_RECEIPT_KEYS) - set(record))
    if missing:
        errors.append(f"external_effect_receipt is missing keys: {missing}")
    if record.get("schema_version") != EXTERNAL_EFFECT_RECEIPT_SCHEMA_VERSION:
        errors.append(f"external_effect_receipt schema_version must be {EXTERNAL_EFFECT_RECEIPT_SCHEMA_VERSION}")
    for key in _RECEIPT_STRING_FIELDS:
        if not isinstance(record.get(key), str):
            errors.append(f"external_effect_receipt {key} must be a string")
    if record.get("observed_result") == "requested":
        errors.append("external_effect_receipt observed_result must not be requested; a request is not an observation")
    elif record.get("observed_result") not in RECEIPT_RESULTS:
        errors.append(f"external_effect_receipt observed_result is unsupported: {record.get('observed_result')!r}")
    if record.get("action") not in ACTIONS:
        errors.append(f"external_effect_receipt action is unsupported: {record.get('action')!r}")
    elif record.get("target_class") != ACTION_TARGET_CLASSES[str(record.get("action"))]:
        errors.append("external_effect_receipt target_class does not match its action")
    if record.get("target_class") not in TARGET_CLASSES:
        errors.append(f"external_effect_receipt target_class is unsupported: {record.get('target_class')!r}")
    if record.get("acting_surface") not in ACTING_SURFACES:
        errors.append(f"external_effect_receipt acting_surface is unsupported: {record.get('acting_surface')!r}")
    if record.get("privacy") != RECEIPT_PRIVACY:
        errors.append("external_effect_receipt privacy must be metadata_only")
    if record.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("external_effect_receipt claim_boundary must state the receipt boundary")
    # Every identifier on the receipt, not the two a review named. `receipt_id`
    # and `observed_at` are checked here for the same reason `effect_id` is:
    # they are rendered, and a citation built from an unguarded one is exactly
    # the string this store exists to keep out of a status report.
    for field in _REQUIRED_RECEIPT_REFS:
        errors.extend(reference_errors(record.get(field), field=field, label=_LABEL, required=True))
    for field in _OPTIONAL_RECEIPT_REFS:
        errors.extend(reference_errors(record.get(field), field=field, label=_LABEL, required=False))
    if record.get("observed_result") == "succeeded" and not str(record.get("external_ref", "")):
        errors.append("external_effect_receipt succeeded requires an external_ref naming the observed effect")
    refs = record.get("evidence_refs")
    if not isinstance(refs, list):
        errors.append("external_effect_receipt evidence_refs must be a list")
    else:
        if len(refs) > MAX_EVIDENCE_REFS:
            errors.append(f"external_effect_receipt evidence_refs must have at most {MAX_EVIDENCE_REFS} items")
        for index, value in enumerate(refs):
            errors.extend(
                reference_errors(value, field=f"evidence_refs[{index}]", label=_LABEL, required=True)
            )
    for field in _RECEIPT_TEXT_FIELDS:
        text = record.get(field)
        if not isinstance(text, str):
            continue
        if len(text) > MAX_SUMMARY_CHARS:
            errors.append(f"external_effect_receipt {field} must be at most {MAX_SUMMARY_CHARS} characters")
        if is_unsafe_receipt_text(text):
            errors.append(
                f"external_effect_receipt {field} must not carry secrets, links, paths, or raw text; "
                "it is one bounded metadata line"
            )
    return errors


def append_external_effect_receipt(paths: OmhPaths, record: dict[str, Any]) -> dict[str, Any]:
    """Append one validated receipt. Never rewrites, never reorders."""
    errors = validate_external_effect_receipt(record)
    if errors:
        raise ExternalEffectReceiptError(errors[0])
    path = paths.runtime_external_effect_receipts_path
    # `file_lock` rather than a bare `fcntl.flock`: it holds a real OS lock on
    # both POSIX and Windows, and this store's mutual exclusion has to be a
    # property CI can assert on either platform.
    with file_lock(path, private=True):
        append_store_line(path, record)
    return record


def record_external_effect(
    paths: OmhPaths,
    *,
    effect_id: str,
    action: str,
    acting_surface: str,
    observed_result: str,
    run_id: str = "",
    external_ref: str = "",
    evidence_refs: list[str] | tuple[str, ...] = (),
    summary: str = "",
) -> dict[str, Any]:
    """Mint and append one receipt, or return the receipt that already records it.

    The strict entry point: it raises when the observation cannot become a
    receipt and when the store cannot be written. Producers that must not fail
    because of a receipt call `mint_external_effect_receipt` instead.

    A retry or a reversal of an effect that already has a receipt links to it
    through `supersedes_receipt_ref`; nothing on disk is changed. Re-recording
    an observation identical to the effect's latest receipt appends nothing and
    returns that receipt, so a record written three times byte-identically
    still has exactly one receipt.

    The read of the prior receipt and the append happen under one lock, so two
    concurrent retries of one effect cannot both link to the same predecessor
    and lose a link in the chain.
    """
    record, _ = _record_external_effect(
        paths.runtime_external_effect_receipts_path,
        effect_id=effect_id,
        action=action,
        acting_surface=acting_surface,
        observed_result=observed_result,
        run_id=run_id,
        external_ref=external_ref,
        evidence_refs=evidence_refs,
        summary=summary,
    )
    return record


def _record_external_effect(
    path: Path,
    *,
    effect_id: str,
    action: str,
    acting_surface: str,
    observed_result: str,
    run_id: str,
    external_ref: str,
    evidence_refs: list[str] | tuple[str, ...],
    summary: str,
) -> tuple[dict[str, Any], bool]:
    """The receipt now on record, and whether this call is what put it there."""
    with file_lock(path, private=True):
        prior = _latest_receipt_for_effect_in(path, effect_id)
        record = build_external_effect_receipt(
            effect_id=effect_id,
            action=action,
            acting_surface=acting_surface,
            observed_result=observed_result,
            run_id=run_id,
            external_ref=external_ref,
            evidence_refs=evidence_refs,
            summary=summary,
            supersedes_receipt_ref=str(prior.get("receipt_id", "")) if prior else "",
        )
        if prior and _observation_fingerprint(prior) == _observation_fingerprint(record):
            return prior, False
        append_store_line(path, record)
    return record, True


def mint_external_effect_receipt(
    paths: OmhPaths,
    *,
    effect_id: str,
    action: str,
    acting_surface: str,
    observed_result: str,
    run_id: str = "",
    external_ref: str = "",
    evidence_refs: list[str] | tuple[str, ...] = (),
    summary: str = "",
) -> dict[str, Any]:
    """Record one observed external effect without ever raising into the producer.

    A producer records a receipt *after* it has already observed the effect and
    written its own record. Letting the receipt store fail that producer would
    make an unwritable store look like an effect that never happened, so every
    refusal and every write failure is returned instead of raised.

    Returns an `external_effect_mint_result/v1` mapping:

        schema_version  -- EXTERNAL_EFFECT_MINT_RESULT_SCHEMA_VERSION
        minted          -- True only when a new receipt reached the store
        outcome         -- "recorded" | "already_recorded" | "refused" | "not_written"
        effect_id       -- the effect this call was about
        receipt_id      -- the receipt now on record, "" when there is none
        observed_result -- the result now on record, "" when there is none
        error           -- the refusal or write failure, "" otherwise

    `already_recorded` is what makes this idempotent by effect identity: the
    same observation of the same effect mints once no matter how often it is
    reported. A failed write records nothing, so the next report of the same
    observation still mints exactly one receipt.

    Failures are also appended to `external_effect_mint_failures.jsonl` beside
    the store, best effort, so an effect that went unreceipted is visible
    without the producer having to invent a channel for it.
    """
    return mint_external_effect_receipt_at(
        paths.runtime_external_effect_receipts_path,
        effect_id=effect_id,
        action=action,
        acting_surface=acting_surface,
        observed_result=observed_result,
        run_id=run_id,
        external_ref=external_ref,
        evidence_refs=evidence_refs,
        summary=summary,
    )


def mint_external_effect_receipt_at(
    store_path: Path,
    *,
    effect_id: str,
    action: str,
    acting_surface: str,
    observed_result: str,
    run_id: str = "",
    external_ref: str = "",
    evidence_refs: list[str] | tuple[str, ...] = (),
    summary: str = "",
) -> dict[str, Any]:
    """`mint_external_effect_receipt` against an already-resolved store path.

    Producers that hold a run directory rather than an `OmhPaths` use this, so
    every path they touch is derived from the directory they were given and
    none is reconstructed from an environment they are not running in.
    """
    try:
        record, minted = _record_external_effect(
            store_path,
            effect_id=effect_id,
            action=action,
            acting_surface=acting_surface,
            observed_result=observed_result,
            run_id=run_id,
            external_ref=external_ref,
            evidence_refs=evidence_refs,
            summary=summary,
        )
    except ExternalEffectReceiptError as exc:
        return _mint_failure(
            store_path,
            outcome="refused",
            error=str(exc),
            effect_id=effect_id,
            action=action,
            acting_surface=acting_surface,
            observed_result=observed_result,
            run_id=run_id,
        )
    except OSError as exc:
        # Covers FileLockTimeout, which is a TimeoutError and therefore an
        # OSError: an unavailable lock is a store that could not be written.
        return _mint_failure(
            store_path,
            outcome="not_written",
            error=str(exc),
            effect_id=effect_id,
            action=action,
            acting_surface=acting_surface,
            observed_result=observed_result,
            run_id=run_id,
        )
    return {
        "schema_version": EXTERNAL_EFFECT_MINT_RESULT_SCHEMA_VERSION,
        "minted": minted,
        "outcome": "recorded" if minted else "already_recorded",
        "effect_id": str(record.get("effect_id", "")),
        "receipt_id": str(record.get("receipt_id", "")),
        "observed_result": str(record.get("observed_result", "")),
        "error": "",
    }


def read_external_effect_receipts_result(paths: OmhPaths) -> tuple[list[dict[str, Any]], list[str]]:
    return read_jsonl_objects(paths.runtime_external_effect_receipts_path)


def read_external_effect_receipts(
    paths: OmhPaths,
    *,
    run_id: str | None = None,
    effect_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    receipts, _ = read_external_effect_receipts_result(paths)
    if run_id is not None:
        receipts = [receipt for receipt in receipts if str(receipt.get("run_id", "")) == run_id]
    if effect_id is not None:
        receipts = [receipt for receipt in receipts if str(receipt.get("effect_id", "")) == effect_id]
    if limit is None:
        return receipts
    if limit < 1:
        return []
    return receipts[-limit:]


def index_external_effect_receipts_by_run(paths: OmhPaths) -> dict[str, list[dict[str, Any]]]:
    """Every receipt grouped by the run it belongs to, from one read of the store.

    The store is runtime-wide, so a caller that walks runs would otherwise
    re-parse the whole file once per run. Callers that iterate runs index once
    and hand each run its own slice.
    """
    receipts, _ = read_external_effect_receipts_result(paths)
    index: dict[str, list[dict[str, Any]]] = {}
    for receipt in receipts:
        index.setdefault(str(receipt.get("run_id", "")), []).append(receipt)
    return index


def latest_receipt_in(receipts: list[dict[str, Any]] | tuple[dict[str, Any], ...], effect_id: str) -> dict[str, Any]:
    """The receipt that speaks for `effect_id`: the one selection rule.

    The store is append-only, so store order is arrival order and the last
    record for an effect is its latest state; a later append always wins the
    tie. Every surface that judges an effect selects through this function over
    the same order, which is what stops the validator and the claim ladder from
    picking different receipts for one effect and reaching opposite conclusions
    about it.
    """
    return latest_record_in(receipts, key="effect_id", value=str(effect_id))


def select_effect_receipt(
    receipts: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    kind: str,
    run_id: str,
) -> dict[str, Any]:
    """The receipt a run-scoped gate is judged against, for every call site.

    Both gate call sites -- runtime validation and the projection the claim
    ladder reads -- name the effect the same way and select through
    `latest_receipt_in`, so neither can end up judging a receipt the other never
    saw.
    """
    return latest_receipt_in(receipts, external_effect_id(kind, run_id))


def receipt_satisfies_success_claim(
    receipt: Mapping[str, Any] | None,
    *,
    kind: str,
    run_id: str,
) -> bool:
    """The one predicate that decides whether a receipt backs a success claim.

    Runtime validation and the claim ladder both call this, so a `failed` or
    `attempted` receipt can never satisfy one of them while the other refuses
    it. A receipt backs a success claim only when it observed the effect
    *succeed*, is about *that* effect, records *that* action, and names the
    acting surface that gate is allowed to be observed by.
    """
    expected = EXTERNAL_EFFECT_CLAIM_SURFACES.get(kind)
    if expected is None or not isinstance(receipt, Mapping):
        return False
    action, acting_surface = expected
    return (
        str(receipt.get("observed_result", "")) == "succeeded"
        and str(receipt.get("effect_id", "")) == external_effect_id(kind, run_id)
        and str(receipt.get("action", "")) == action
        and str(receipt.get("acting_surface", "")) == acting_surface
        and bool(str(receipt.get("receipt_id", "")))
    )


def receipt_contradicts_success_claim(
    receipt: Mapping[str, Any] | None,
    *,
    kind: str,
    run_id: str,
) -> bool:
    """Whether a receipt says the *opposite* of a success record, not merely less.

    Defined against `receipt_satisfies_success_claim` so the two can never
    drift. A receipt that satisfies the claim never contradicts it. A receipt
    that observed the effect start (`attempted`) or reach a terminal state it
    could not name (`unknown`) withholds the claim on the ladder rather than
    condemning the record: it is less evidence, not opposite evidence. Only a
    receipt that observed the effect *not happen*, or one that is filed against
    this effect while naming another action or another acting surface,
    contradicts a record that says the effect succeeded.
    """
    expected = EXTERNAL_EFFECT_CLAIM_SURFACES.get(kind)
    if expected is None or not isinstance(receipt, Mapping) or not receipt:
        return False
    if receipt_satisfies_success_claim(receipt, kind=kind, run_id=run_id):
        return False
    if str(receipt.get("effect_id", "")) != external_effect_id(kind, run_id):
        return False
    action, acting_surface = expected
    if str(receipt.get("action", "")) != action or str(receipt.get("acting_surface", "")) != acting_surface:
        return True
    return str(receipt.get("observed_result", "")) == "failed"


def validate_external_effect_receipt_store(path: Path, *, run_id: str | None = None) -> dict[str, Any]:
    """Validate the receipt store, optionally scoped to one run's records.

    Unscoped (`run_id=None`) this is the store's own report: unparseable lines,
    every record's schema, and the integrity of the supersede chain. A line
    that does not parse belongs to no run -- it has no `run_id` to read -- so it
    is the store's fault and is reported here, once, never against a run that
    happened to be validated while it sat there.

    Scoped to a run, only that run's records and their chain are considered, so
    one bad line elsewhere cannot fault a run that has nothing to do with it.
    """
    receipts, read_errors = read_jsonl_objects(path)
    receipt_count, errors = store_errors(
        path,
        receipts,
        read_errors,
        run_id=run_id,
        validate_record=validate_external_effect_receipt,
        label=_LABEL,
    )
    return {
        "schema_version": EXTERNAL_EFFECT_RECEIPT_STORE_VALIDATION_SCHEMA_VERSION,
        "path": str(path),
        "run_id": run_id or "",
        "ok": not errors,
        "receipt_count": receipt_count,
        "mint_failure_count": len(read_external_effect_mint_failures(path)),
        "errors": errors,
    }


def external_effect_mint_failures_path(store_path: Path) -> Path:
    """Where unrecorded mints are logged, beside the store they failed to reach."""
    return store_path.with_name(EXTERNAL_EFFECT_MINT_FAILURE_STORE_NAME)


def read_external_effect_mint_failures(store_path: Path) -> list[dict[str, Any]]:
    failures, _ = read_jsonl_objects(external_effect_mint_failures_path(store_path))
    return failures


def project_external_effects(
    paths: OmhPaths,
    run_id: str,
    *,
    requested_effects: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    """Project one run's external effects into AC1's five states.

    `requested_effects` are effects the run's own records say are needed, each
    carrying the state its own records justify: `requested` when nothing has
    started, `attempted` when the record says the effect is in flight with no
    outcome. Both are projections over the *absence* of a receipt and are only
    used when the effect has none, so neither can be mistaken for an
    observation. `succeeded`, `failed`, and `unknown` come only from a receipt.
    """
    known = read_external_effect_receipts(paths, run_id=str(run_id))
    counts: dict[str, int] = {}
    for receipt in known:
        effect_id = str(receipt.get("effect_id", ""))
        counts[effect_id] = counts.get(effect_id, 0) + 1
    # Through `latest_receipt_in` rather than by overwriting a dict in place, so
    # this projection and runtime validation select an effect's receipt by one
    # rule instead of two that happen to agree today.
    latest = {effect_id: latest_receipt_in(known, effect_id) for effect_id in counts}
    buckets: dict[str, list[dict[str, Any]]] = {result: [] for result in OBSERVED_RESULTS}
    for effect_id, receipt in latest.items():
        # A result outside the receipt vocabulary is not a sixth state; it is a
        # value with no meaning, and the honest bucket for that is `unknown`.
        result = closed_receipt_value(receipt.get("observed_result"), RECEIPT_RESULTS) or "unknown"
        buckets[result].append(_observed_effect_row(receipt, counts.get(effect_id, 1)))
    for requested in requested_effects:
        effect_id = str(requested.get("effect_id", ""))
        if not effect_id or effect_id in latest:
            continue
        state = str(requested.get("state", "requested"))
        if state not in PROJECTED_RESULTS:
            state = "requested"
        buckets[state].append(_projected_effect_row(requested, state))
    for result in OBSERVED_RESULTS:
        buckets[result].sort(key=lambda row: str(row.get("effect_id", "")))
    return {
        "schema_version": EXTERNAL_EFFECT_PROJECTION_SCHEMA_VERSION,
        "run_id": str(run_id),
        "requested": buckets["requested"],
        "attempted": buckets["attempted"],
        "succeeded": buckets["succeeded"],
        "failed": buckets["failed"],
        "unknown": buckets["unknown"],
        "receipt_count": len(known),
        "effect_count": sum(len(buckets[result]) for result in OBSERVED_RESULTS),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def receipt_claim_for_effect(projection: dict[str, Any], effect_id: str) -> dict[str, Any]:
    """The citation a success claim about `effect_id` must carry (AC2)."""
    for result in OBSERVED_RESULTS:
        rows = projection.get(result)
        for row in rows if isinstance(rows, list) else []:
            if str(row.get("effect_id", "")) == str(effect_id):
                return {
                    # Projected states carry no receipt id, which is exactly
                    # what makes them unobserved.
                    "observed": bool(str(row.get("receipt_id", ""))),
                    "effect_id": str(row.get("effect_id", "")),
                    "receipt_id": str(row.get("receipt_id", "")),
                    "action": str(row.get("action", "")),
                    "acting_surface": str(row.get("acting_surface", "")),
                    "observed_result": result,
                    "external_ref": str(row.get("external_ref", "")),
                    "receipt_count": int(row.get("receipt_count", 0) or 0),
                }
    return {
        "observed": False,
        "effect_id": str(effect_id),
        "receipt_id": "",
        "action": "",
        "acting_surface": "",
        "observed_result": "requested",
        "external_ref": "",
        "receipt_count": 0,
    }


def success_claim_citation(projection: dict[str, Any]) -> str:
    """One rendered clause naming every succeeded effect and who acted."""
    rows = projection.get("succeeded")
    if not isinstance(rows, list) or not rows:
        return ""
    parts = [
        f"{row.get('action', '')} receipt {row.get('receipt_id', '')} via {row.get('acting_surface', '')}"
        for row in rows
        if isinstance(row, dict)
    ]
    return "; ".join(parts)


def redacted_external_effect_ref(value: str) -> str:
    """Fold any external handle into a bounded, non-navigable receipt reference.

    A URL, a secret-shaped string, or anything longer than a bounded identifier
    becomes a stable digest handle instead. Stable because callers dedupe rows
    by it; non-navigable because a status report is not a place to publish
    someone's links.
    """
    return redacted_ref(value, field="external_ref")


def is_unsafe_receipt_text(value: str) -> bool:
    """Whether one free-text receipt field carries something it must not.

    A secret, a link, a filesystem path, or a control character. The last one
    is the tell that the value is a prompt, a transcript, or a log slice rather
    than the single bounded metadata line a receipt summary is allowed to be.
    """
    return is_unsafe_metadata_line(value)


def bounded_receipt_summary(value: Any) -> str:
    """One bounded, link-free, secret-free metadata line, or `[redacted]`.

    The check runs on the raw value, before whitespace is collapsed: a summary
    that arrived with newlines in it was never a summary, and collapsing them
    first would hide that.
    """
    raw = str(value or "")
    if not raw.strip():
        return ""
    if is_unsafe_receipt_text(raw):
        return REDACTED_SUMMARY
    return " ".join(raw.split())[:MAX_SUMMARY_CHARS]


def closed_receipt_value(value: Any, allowed: tuple[str, ...]) -> str:
    """A closed-vocabulary field renders only what its vocabulary contains.

    Rendering reads whatever is on disk, and a store can be hand-edited. A value
    outside the vocabulary is not a new state, it is a value with no meaning, so
    it renders empty rather than as itself.
    """
    return closed_vocabulary_value(value, allowed)


def compact_external_effect_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """The user-facing shape, with every field through its class's guard.

    Rendering is the last point at which a hand-edited or tampered store reaches
    a human, so nothing is copied through verbatim. Identifiers fold to a stable
    non-navigable handle when they are not opaque -- a `receipt_id` carrying an
    ANSI erase sequence is precisely what an AC2 citation would otherwise print
    -- closed vocabularies render empty outside their vocabulary, and free text
    goes through the bounded-summary guard.
    """
    return {
        "receipt_id": redacted_external_effect_ref(str(receipt.get("receipt_id", ""))),
        "effect_id": redacted_external_effect_ref(str(receipt.get("effect_id", ""))),
        "action": closed_receipt_value(receipt.get("action"), ACTIONS),
        "target_class": closed_receipt_value(receipt.get("target_class"), TARGET_CLASSES),
        "acting_surface": closed_receipt_value(receipt.get("acting_surface"), ACTING_SURFACES),
        "observed_result": closed_receipt_value(receipt.get("observed_result"), RECEIPT_RESULTS),
        "observed_at": redacted_external_effect_ref(str(receipt.get("observed_at", ""))),
        "external_ref": redacted_external_effect_ref(str(receipt.get("external_ref", ""))),
        "supersedes_receipt_ref": redacted_external_effect_ref(str(receipt.get("supersedes_receipt_ref", ""))),
        "summary": bounded_receipt_summary(receipt.get("summary", "")),
    }


def _mint_failure(
    store_path: Path,
    *,
    outcome: str,
    error: str,
    effect_id: str,
    action: str,
    acting_surface: str,
    observed_result: str,
    run_id: str,
) -> dict[str, Any]:
    failure = {
        "schema_version": EXTERNAL_EFFECT_MINT_FAILURE_SCHEMA_VERSION,
        "observed_at": utc_now(),
        "outcome": outcome,
        "effect_id": str(effect_id),
        "action": str(action),
        "acting_surface": str(acting_surface),
        "observed_result": str(observed_result),
        "run_id": str(run_id),
        "error": bounded_receipt_summary(error),
    }
    _append_mint_failure(store_path, failure)
    return {
        "schema_version": EXTERNAL_EFFECT_MINT_RESULT_SCHEMA_VERSION,
        "minted": False,
        "outcome": outcome,
        "effect_id": str(effect_id),
        "receipt_id": "",
        "observed_result": "",
        "error": str(error),
    }


def _append_mint_failure(store_path: Path, failure: dict[str, Any]) -> None:
    append_sidecar_line(external_effect_mint_failures_path(store_path), failure)


def _observed_effect_row(receipt: dict[str, Any], receipt_count: int) -> dict[str, Any]:
    return {**compact_external_effect_receipt(receipt), "receipt_count": receipt_count}


def _projected_effect_row(requested: dict[str, Any], state: str) -> dict[str, Any]:
    action = str(requested.get("action", ""))
    return {
        "receipt_id": "",
        "effect_id": str(requested.get("effect_id", "")),
        "action": action,
        "target_class": ACTION_TARGET_CLASSES.get(action, ""),
        "acting_surface": "",
        "observed_result": state,
        "observed_at": "",
        "external_ref": "",
        "supersedes_receipt_ref": "",
        "summary": bounded_receipt_summary(requested.get("summary", "")),
        "receipt_count": 0,
    }


def _latest_receipt_for_effect_in(path: Path, effect_id: str) -> dict[str, Any]:
    receipts, _ = read_jsonl_objects(path)
    return latest_receipt_in(receipts, str(effect_id))


def _observation_fingerprint(receipt: Mapping[str, Any]) -> str:
    """Which observation a receipt records, ignoring when it was recorded."""
    return record_fingerprint(receipt, _OBSERVATION_IDENTITY_KEYS)


def _opaque(value: str, *, field: str) -> str:
    return opaque_ref(value, field=field, error=ExternalEffectReceiptError)


def _receipt_id(observed_at: str, effect_id: str, acting_surface: str, observed_result: str) -> str:
    return mint_record_id(
        prefix="receipt",
        identity={
            "observed_at": observed_at,
            "effect_id": effect_id,
            "acting_surface": acting_surface,
            "observed_result": observed_result,
        },
    )


def _bounded_refs(values: list[str] | tuple[str, ...]) -> list[str]:
    refs: list[str] = []
    for value in values or ():
        text = str(value or "").strip()
        if not text:
            continue
        refs.append(_opaque(text, field="external_effect_receipt evidence_ref"))
        if len(refs) == MAX_EVIDENCE_REFS:
            break
    return refs
