"""Metadata-only what-if children of an accepted Hermes plan (issue #827).

A `plan_variant/v1` record answers "what if we had assumed something else"
without touching the plan that was accepted. It names the parent plan by its
immutable digest, lists every input that differs, lists the references it
inherits unchanged, and leaves the next-handoff choice open for a human.

Why a child record and not a second plan
----------------------------------------

Re-running `omh hermes plan` for the alternative produces a sibling artifact
with no link back: nothing says which assumption changed, nothing says the two
plans came from the same decision, and evidence recorded against one reads as
evidence for the other. Rewriting the accepted plan in place is worse -- it
destroys the artifact a handoff already points at by digest.

A variant is deliberately *not* a plan. It carries no `status` field, so no
reader can find `accepted` on it; its own key set is closed, so a caller cannot
staple one on. Choosing a variant as the next handoff is a separate act that
goes through the normal plan lifecycle commands, not something this record can
perform.

What the record does not do
---------------------------

Building a variant is a pure local metadata operation. It runs no replay, no
tool, no network call, and no dispatch -- `PLAN_VARIANT_NOT_OBSERVED` names the
classes a variant is never evidence for, and this module imports nothing that
could produce one.

Determinism
-----------

`variant_id` and `variant_digest` are derived from the parent digest, the
variant name, and the deltas. Wall-clock time is a caller-supplied parameter
and stays out of both, so the same fork of the same plan is the same record no
matter when it is built.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from ..system.local_store import atomic_write_json
from ..system.paths import OmhPaths, ensure_project_store_ignored, project_artifact_dir


PLAN_VARIANT_SCHEMA_VERSION = "plan_variant/v1"
PLAN_VARIANT_DELTA_SCHEMA_VERSION = "plan_variant_delta/v1"
PLAN_VARIANT_REF_SCHEMA_VERSION = "plan_variant_ref/v1"
PLAN_VARIANT_HANDOFF_SCHEMA_VERSION = "plan_variant_handoff/v1"
PLAN_VARIANT_PARENT_SCHEMA_VERSION = "hermes_plan/v1"
PLAN_VARIANT_PARENT_STATUS = "accepted"

PLAN_VARIANT_DELTA_DIMENSIONS = (
    "assumption",
    "scope",
    "coding_owner",
    "policy",
    "acceptance_criteria",
    "verification",
    "other",
)
PLAN_VARIANT_REF_KINDS = (
    "plan_artifact",
    "context_pack",
    "source",
    "evidence",
    "memory_record",
    "unknown",
)
PLAN_VARIANT_HANDOFF_DECISIONS = ("undecided", "parent_plan", "variant")
PLAN_VARIANT_HANDOFF_CANDIDATE_KINDS = ("parent_plan", "variant")
PLAN_VARIANT_PREPARED_STATE = "prepared_not_observed"
PLAN_VARIANT_HANDOFF_QUESTION = (
    "Which plan should become the next handoff: the accepted parent plan or this variant?"
)
PLAN_VARIANT_HANDOFF_BOUNDARY = (
    "Recording a decision here is intent only. The accepted plan artifact stays unchanged "
    "until a separate plan lifecycle command runs."
)
PLAN_VARIANT_CLAIM_BOUNDARY = (
    "A plan_variant/v1 record is a metadata-only what-if child of an accepted Hermes plan. "
    "It is not the accepted plan, not plan acceptance, and not replay, dispatch, execution, "
    "verification, review, CI, merge-readiness, or merge evidence."
)
PLAN_VARIANT_NOT_OBSERVED = (
    "plan_acceptance",
    "plan_replay",
    "tool_execution",
    "network_call",
    "coding_dispatch",
    "executor_result",
    "verification",
    "review",
)

PLAN_VARIANT_KEYS = (
    "schema_version",
    "variant_id",
    "variant_digest",
    "name",
    "rationale",
    "created_at",
    "parent_plan_path",
    "parent_plan_status",
    "parent_plan_schema_version",
    "parent_plan_sha256",
    "parent_task_statement_sha256",
    "deltas",
    "delta_count",
    "changed_inputs",
    "inherited_refs",
    "inherited_ref_count",
    "refs_requiring_reevaluation",
    "prepared_state",
    "next_handoff",
    "claim_boundary",
    "not_evidence_until_observed",
)
PLAN_VARIANT_DELTA_KEYS = ("schema_version", "dimension", "label", "parent_value", "variant_value", "reason")
PLAN_VARIANT_REF_KEYS = ("schema_version", "kind", "ref", "reviewed", "note")
PLAN_VARIANT_HANDOFF_KEYS = (
    "schema_version",
    "decision",
    "question",
    "requires_user_choice",
    "candidates",
    "boundary",
)
PLAN_VARIANT_HANDOFF_CANDIDATE_KEYS = ("kind", "ref", "digest")


def normalize_plan_variant_dimension(dimension: str | None) -> str:
    """Map a delta dimension onto the closed vocabulary, defaulting to `other`."""
    normalized = _key(dimension or "")
    aliases = {
        "assumption": "assumption",
        "assumptions": "assumption",
        "clarified assumption": "assumption",
        "scope": "scope",
        "goal": "scope",
        "non goal": "scope",
        "coding owner": "coding_owner",
        "owner": "coding_owner",
        "executor": "coding_owner",
        "policy": "policy",
        "constraint": "policy",
        "acceptance criteria": "acceptance_criteria",
        "acceptance": "acceptance_criteria",
        "verification": "verification",
        "verification plan": "verification",
        "other": "other",
    }
    return aliases.get(normalized, "other")


def normalize_plan_variant_ref_kind(kind: str | None) -> str:
    """Map an inherited-reference kind onto the closed vocabulary."""
    normalized = _key(kind or "").replace(" ", "_")
    return normalized if normalized in PLAN_VARIANT_REF_KINDS else "unknown"


def build_plan_variant_delta(
    *,
    dimension: str,
    label: str,
    parent_value: str,
    variant_value: str,
    reason: str = "",
) -> dict[str, object]:
    """One explicit difference between the accepted plan and the variant."""
    return {
        "schema_version": PLAN_VARIANT_DELTA_SCHEMA_VERSION,
        "dimension": normalize_plan_variant_dimension(dimension),
        "label": label.strip(),
        "parent_value": parent_value.strip(),
        "variant_value": variant_value.strip(),
        "reason": reason.strip(),
    }


def build_plan_variant_ref(
    *,
    kind: str,
    ref: str,
    reviewed: bool = False,
    note: str = "",
) -> dict[str, object]:
    """One reference the variant carries over from its parent.

    `reviewed` is the whole point of the field: only reviewed references are
    inherited. Everything else is routed to `refs_requiring_reevaluation`, so a
    variant never silently reuses context nobody checked against its new
    assumption.
    """
    return {
        "schema_version": PLAN_VARIANT_REF_SCHEMA_VERSION,
        "kind": normalize_plan_variant_ref_kind(kind),
        "ref": ref.strip(),
        "reviewed": bool(reviewed),
        "note": note.strip(),
    }


def build_plan_variant(
    *,
    parent_artifact: dict[str, Any],
    name: str,
    deltas: Iterable[dict[str, Any]],
    refs: Iterable[dict[str, Any]] = (),
    rationale: str = "",
    created_at: str = "",
) -> dict[str, object]:
    """Build a `plan_variant/v1` child of one accepted plan artifact.

    `parent_artifact` is a `read_hermes_plan_artifact` result. Nothing is
    written and the parent is never mutated: this function only reads the
    digest and identity fields off the dict it was handed.

    `created_at` is a caller-supplied timestamp so the identity fields stay
    reproducible; it is deliberately excluded from both derived digests.
    """
    variant_name = name.strip()
    if not variant_name:
        raise ValueError("plan variant requires a name")
    parent_schema = str(parent_artifact.get("schema_version", ""))
    if parent_schema != PLAN_VARIANT_PARENT_SCHEMA_VERSION:
        raise ValueError(f"plan variant requires a {PLAN_VARIANT_PARENT_SCHEMA_VERSION} parent artifact")
    parent_status = str(parent_artifact.get("status", ""))
    if parent_status != PLAN_VARIANT_PARENT_STATUS:
        raise ValueError("plan variant requires an accepted parent plan")
    parent_sha256 = str(parent_artifact.get("sha256", ""))
    if not _is_sha256(parent_sha256):
        raise ValueError("plan variant requires a parent plan sha256 digest")
    normalized_deltas = [_coerce_delta(delta) for delta in deltas]
    if not normalized_deltas:
        raise ValueError("plan variant requires at least one delta")

    parent_path = str(parent_artifact.get("path", ""))
    # The accepted parent is always an inherited reference: a variant that
    # listed no inheritance would read as an unrelated plan.
    supplied_refs = [
        build_plan_variant_ref(
            kind="plan_artifact",
            ref=parent_path,
            reviewed=True,
            note="Accepted parent plan artifact; every input not named in deltas is inherited from it.",
        ),
        *(_coerce_ref(ref) for ref in refs),
    ]
    inherited = [ref for ref in supplied_refs if ref["reviewed"]]
    reevaluate = [ref for ref in supplied_refs if not ref["reviewed"]]

    digest = _variant_digest(parent_sha256, variant_name, normalized_deltas)
    variant_id = f"plan-variant-{digest[:12]}"
    return {
        "schema_version": PLAN_VARIANT_SCHEMA_VERSION,
        "variant_id": variant_id,
        "variant_digest": digest,
        "name": variant_name,
        "rationale": rationale.strip(),
        "created_at": created_at.strip(),
        "parent_plan_path": parent_path,
        "parent_plan_status": parent_status,
        "parent_plan_schema_version": parent_schema,
        "parent_plan_sha256": parent_sha256,
        "parent_task_statement_sha256": str(parent_artifact.get("task_statement_sha256", "")),
        "deltas": normalized_deltas,
        "delta_count": len(normalized_deltas),
        "changed_inputs": [_changed_input_line(delta) for delta in normalized_deltas],
        "inherited_refs": inherited,
        "inherited_ref_count": len(inherited),
        "refs_requiring_reevaluation": reevaluate,
        "prepared_state": PLAN_VARIANT_PREPARED_STATE,
        "next_handoff": {
            "schema_version": PLAN_VARIANT_HANDOFF_SCHEMA_VERSION,
            "decision": "undecided",
            "question": PLAN_VARIANT_HANDOFF_QUESTION,
            "requires_user_choice": True,
            "candidates": [
                {"kind": "parent_plan", "ref": parent_path, "digest": parent_sha256},
                {"kind": "variant", "ref": variant_id, "digest": digest},
            ],
            "boundary": PLAN_VARIANT_HANDOFF_BOUNDARY,
        },
        "claim_boundary": PLAN_VARIANT_CLAIM_BOUNDARY,
        "not_evidence_until_observed": list(PLAN_VARIANT_NOT_OBSERVED),
    }


def validate_plan_variant(payload: dict[str, Any]) -> list[str]:
    """Return every contract violation in one pass; empty means valid."""
    errors: list[str] = []
    if payload.get("schema_version") != PLAN_VARIANT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PLAN_VARIANT_SCHEMA_VERSION}")
    errors.extend(_key_set_errors(payload, PLAN_VARIANT_KEYS, "plan_variant"))
    if not str(payload.get("variant_id", "")).startswith("plan-variant-"):
        errors.append("variant_id must start with plan-variant-")
    if not _is_sha256(str(payload.get("variant_digest", ""))):
        errors.append("variant_digest must be a sha256 hex digest")
    if not str(payload.get("name", "")).strip():
        errors.append("name must not be empty")
    if payload.get("parent_plan_schema_version") != PLAN_VARIANT_PARENT_SCHEMA_VERSION:
        errors.append(f"parent_plan_schema_version must be {PLAN_VARIANT_PARENT_SCHEMA_VERSION}")
    if payload.get("parent_plan_status") != PLAN_VARIANT_PARENT_STATUS:
        errors.append(f"parent_plan_status must be {PLAN_VARIANT_PARENT_STATUS}")
    if not _is_sha256(str(payload.get("parent_plan_sha256", ""))):
        errors.append("parent_plan_sha256 must be a sha256 hex digest")
    if payload.get("prepared_state") != PLAN_VARIANT_PREPARED_STATE:
        errors.append(f"prepared_state must be {PLAN_VARIANT_PREPARED_STATE}")
    if payload.get("claim_boundary") != PLAN_VARIANT_CLAIM_BOUNDARY:
        errors.append("claim_boundary must deny that a variant is the accepted plan or evidence")
    if not set(PLAN_VARIANT_NOT_OBSERVED).issubset(set(_string_list(payload.get("not_evidence_until_observed")))):
        errors.append("not_evidence_until_observed must include all plan-variant boundaries")

    deltas = payload.get("deltas")
    if not isinstance(deltas, list) or not deltas:
        errors.append("deltas must be a non-empty list")
    else:
        for index, delta in enumerate(deltas):
            for error in _delta_errors(delta):
                errors.append(f"deltas[{index}]: {error}")
        if payload.get("delta_count") != len(deltas):
            errors.append("delta_count must match deltas")
        if len(_string_list(payload.get("changed_inputs"))) != len(deltas):
            errors.append("changed_inputs must list one line per delta")

    inherited = payload.get("inherited_refs")
    if not isinstance(inherited, list) or not inherited:
        errors.append("inherited_refs must be a non-empty list")
    else:
        for index, ref in enumerate(inherited):
            for error in _ref_errors(ref):
                errors.append(f"inherited_refs[{index}]: {error}")
            if isinstance(ref, dict) and not ref.get("reviewed"):
                errors.append(f"inherited_refs[{index}]: only reviewed references may be inherited")
        if payload.get("inherited_ref_count") != len(inherited):
            errors.append("inherited_ref_count must match inherited_refs")

    reevaluate = payload.get("refs_requiring_reevaluation")
    if not isinstance(reevaluate, list):
        errors.append("refs_requiring_reevaluation must be a list")
    else:
        for index, ref in enumerate(reevaluate):
            for error in _ref_errors(ref):
                errors.append(f"refs_requiring_reevaluation[{index}]: {error}")
            if isinstance(ref, dict) and ref.get("reviewed"):
                errors.append(f"refs_requiring_reevaluation[{index}]: a reviewed reference belongs in inherited_refs")

    errors.extend(_handoff_errors(payload.get("next_handoff")))
    return errors


def render_plan_variant_text(variant: dict[str, Any]) -> str:
    """Plain-text rendering: only the differences, then the open question."""
    lines = [
        f"Plan variant: {variant.get('name', '')} ({variant.get('variant_id', '')})",
        f"Parent plan: {variant.get('parent_plan_path', '')}",
        f"Parent digest: {variant.get('parent_plan_sha256', '')} (unchanged by this variant)",
    ]
    rationale = str(variant.get("rationale", ""))
    if rationale:
        lines.append(f"Rationale: {rationale}")
    lines.append("")
    lines.append("Changed inputs:")
    lines.extend(f"  - {line}" for line in _string_list(variant.get("changed_inputs")))
    lines.append("")
    lines.append("Inherited references:")
    lines.extend(_ref_lines(variant.get("inherited_refs")))
    reevaluate = variant.get("refs_requiring_reevaluation")
    if isinstance(reevaluate, list) and reevaluate:
        lines.append("")
        lines.append("Re-evaluate before any handoff:")
        lines.extend(_ref_lines(reevaluate))
    handoff = variant.get("next_handoff")
    lines.append("")
    if isinstance(handoff, dict):
        lines.append(f"Next handoff: {handoff.get('decision', 'undecided')}")
        lines.append(f"  {handoff.get('question', '')}")
        for candidate in handoff.get("candidates", []):
            if isinstance(candidate, dict):
                lines.append(f"  - {candidate.get('kind', '')}: {candidate.get('ref', '')}")
    lines.append("")
    lines.append(f"Prepared state: {variant.get('prepared_state', '')}")
    lines.append(str(variant.get("claim_boundary", "")))
    return "\n".join(lines)


def write_plan_variant(paths: OmhPaths, variant: dict[str, Any]) -> dict[str, object]:
    """Persist one variant under the project store; the parent is never opened."""
    errors = validate_plan_variant(variant)
    if errors:
        raise ValueError("; ".join(errors))
    variants_dir = project_artifact_dir(paths, "plan-variants")
    ensure_project_store_ignored(variants_dir)
    # Named by variant_id, not by a timestamp: re-forking the same plan the same
    # way rewrites one file instead of accumulating near-identical siblings.
    path = variants_dir / f"{variant['variant_id']}.json"
    atomic_write_json(path, dict(variant), private=True)
    return {
        "path": str(path),
        "kind": "plan_variant",
        "schema_version": PLAN_VARIANT_SCHEMA_VERSION,
        "variant_id": variant["variant_id"],
        "parent_plan_sha256": variant["parent_plan_sha256"],
    }


def _coerce_delta(delta: dict[str, Any]) -> dict[str, object]:
    return build_plan_variant_delta(
        dimension=str(delta.get("dimension", "")),
        label=str(delta.get("label", "")),
        parent_value=str(delta.get("parent_value", "")),
        variant_value=str(delta.get("variant_value", "")),
        reason=str(delta.get("reason", "")),
    )


def _coerce_ref(ref: dict[str, Any]) -> dict[str, object]:
    return build_plan_variant_ref(
        kind=str(ref.get("kind", "")),
        ref=str(ref.get("ref", "")),
        reviewed=bool(ref.get("reviewed", False)),
        note=str(ref.get("note", "")),
    )


def _changed_input_line(delta: dict[str, Any]) -> str:
    label = str(delta.get("label", "")) or "unnamed input"
    return (
        f"{delta.get('dimension', 'other')} | {label}: "
        f"{delta.get('parent_value', '')} -> {delta.get('variant_value', '')}"
    )


def _ref_lines(refs: object) -> list[str]:
    if not isinstance(refs, list) or not refs:
        return ["  - None recorded."]
    lines: list[str] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        note = str(ref.get("note", ""))
        suffix = f" — {note}" if note else ""
        lines.append(f"  - {ref.get('kind', 'unknown')}: {ref.get('ref', '')}{suffix}")
    return lines or ["  - None recorded."]


def _delta_errors(delta: object) -> list[str]:
    if not isinstance(delta, dict):
        return ["delta must be an object"]
    errors = _key_set_errors(delta, PLAN_VARIANT_DELTA_KEYS, "delta")
    if delta.get("schema_version") != PLAN_VARIANT_DELTA_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PLAN_VARIANT_DELTA_SCHEMA_VERSION}")
    if delta.get("dimension") not in PLAN_VARIANT_DELTA_DIMENSIONS:
        errors.append("dimension is unsupported")
    if not str(delta.get("label", "")).strip():
        errors.append("label must not be empty")
    if str(delta.get("parent_value", "")) == str(delta.get("variant_value", "")):
        errors.append("parent_value and variant_value must differ")
    return errors


def _ref_errors(ref: object) -> list[str]:
    if not isinstance(ref, dict):
        return ["reference must be an object"]
    errors = _key_set_errors(ref, PLAN_VARIANT_REF_KEYS, "reference")
    if ref.get("schema_version") != PLAN_VARIANT_REF_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PLAN_VARIANT_REF_SCHEMA_VERSION}")
    if ref.get("kind") not in PLAN_VARIANT_REF_KINDS:
        errors.append("kind is unsupported")
    if not str(ref.get("ref", "")).strip():
        errors.append("ref must not be empty")
    if not isinstance(ref.get("reviewed"), bool):
        errors.append("reviewed must be a boolean")
    return errors


def _handoff_errors(handoff: object) -> list[str]:
    if not isinstance(handoff, dict):
        return ["next_handoff must be an object"]
    errors = _key_set_errors(handoff, PLAN_VARIANT_HANDOFF_KEYS, "next_handoff")
    if handoff.get("schema_version") != PLAN_VARIANT_HANDOFF_SCHEMA_VERSION:
        errors.append(f"next_handoff schema_version must be {PLAN_VARIANT_HANDOFF_SCHEMA_VERSION}")
    if handoff.get("decision") not in PLAN_VARIANT_HANDOFF_DECISIONS:
        errors.append("next_handoff decision is unsupported")
    if not isinstance(handoff.get("requires_user_choice"), bool):
        errors.append("next_handoff requires_user_choice must be a boolean")
    if handoff.get("boundary") != PLAN_VARIANT_HANDOFF_BOUNDARY:
        errors.append("next_handoff boundary must keep the accepted plan unchanged")
    candidates = handoff.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        errors.append("next_handoff must offer the parent plan and the variant")
        return errors
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            errors.append(f"next_handoff candidates[{index}]: candidate must be an object")
            continue
        for error in _key_set_errors(candidate, PLAN_VARIANT_HANDOFF_CANDIDATE_KEYS, "candidate"):
            errors.append(f"next_handoff candidates[{index}]: {error}")
        if candidate.get("kind") not in PLAN_VARIANT_HANDOFF_CANDIDATE_KINDS:
            errors.append(f"next_handoff candidates[{index}]: kind is unsupported")
    kinds = [candidate.get("kind") for candidate in candidates if isinstance(candidate, dict)]
    if kinds != list(PLAN_VARIANT_HANDOFF_CANDIDATE_KINDS):
        errors.append("next_handoff candidates must be the parent plan then the variant")
    return errors


def _key_set_errors(payload: object, expected: tuple[str, ...], label: str) -> list[str]:
    if not isinstance(payload, dict):
        return [f"{label} must be an object"]
    present = set(payload)
    allowed = set(expected)
    errors: list[str] = []
    missing = sorted(allowed - present)
    if missing:
        errors.append(f"{label} is missing keys: {', '.join(missing)}")
    unexpected = sorted(present - allowed)
    if unexpected:
        errors.append(f"{label} has unexpected keys: {', '.join(unexpected)}")
    return errors


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _variant_digest(parent_sha256: str, name: str, deltas: list[dict[str, object]]) -> str:
    seed = "|".join(
        [
            parent_sha256,
            _key(name),
            *[
                "~".join(
                    (
                        str(delta["dimension"]),
                        _key(str(delta["label"])),
                        str(delta["parent_value"]),
                        str(delta["variant_value"]),
                    )
                )
                for delta in deltas
            ],
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _key(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())
