"""The whole OMH change a requested capability needs, shown before it is built.

The defect this closes (#794): asking OMH to handle something natively produces
a generic implementation plan, and a generic implementation plan hides the shape
of the change. A new capability routinely moves chat routing, a generated skill,
what memory retains, what a wrapper session carries, what a coding handoff
delegates, what counts as evidence, what the docs claim, and which tests hold the
line -- and a plan that lists tasks says nothing about which of those eight it
touches. The product-wide blast radius arrives after the work, not before it.

`native_capability_change_preview/v1` is the artifact that puts it before. One
requested capability, one described target experience against one stated
baseline, and eight axes that are each answered rather than each mentioned.

Three things make it hold:

* **Every axis is answered.** :data:`CHANGE_AXES` is exhaustive and closed. A
  preview carries all eight, once each, in canonical order, and a missing axis
  is a validation error naming it -- silence is not an available answer.
  :data:`AXIS_STATES` has three members on purpose: ``unchanged`` says the axis
  was considered and this change leaves it alone, ``not_applicable`` says the
  axis has no bearing on this capability at all, and collapsing them into one
  "no" would lose which of those two a reviewer was told. An axis marked
  ``changed`` must name at least one affected surface, and an axis not marked
  ``changed`` must name none, so the blast radius cannot be asserted without
  being enumerated or enumerated without being asserted.

* **A material change moves the digest.** :func:`native_capability_change_preview_digest`
  hashes the material content, and a review is bound to the digest it reviewed.
  When the material content moves, the preview supersedes itself and the earlier
  acceptance stops covering it. That is structural, not remembered:
  ``accepted`` is only a valid status while ``reviewed_digest`` equals
  ``preview_digest``, so a preview that claims an acceptance it outgrew cannot
  pass :func:`validate_native_capability_change_preview`.

* **An accepted preview is a plan.** :data:`PREVIEW_STATUSES` contains no word
  that asserts the capability exists, :data:`REFUSED_PREVIEW_STATUSES` names the
  words it will not accept so the refusal reads as a rule rather than as an
  unrecognised value, and :data:`PREVIEW_STATUS_CLAIMS` gives the one sentence
  each status may be described with -- every one of which says nothing has been
  built.

Where the material line is drawn
--------------------------------

``note`` is the only explanatory field name in this schema, and it is the only
content the digest excludes, at every nesting level. Everything else is
material: the capability identity, the described experience, the baseline, every
axis state, every named surface, every risk and its severity, every verification
expectation and its kind, the rollback boundary, the privacy impact, the
delegated and retained handoff scope, and the optional blueprint reference.

The rule that produces that line: the digest covers what a reviewer is being
asked to agree to, and excludes the prose that explains an item whose identity
and classification did not move. Rewording the ``docs`` axis note from "adds a
section" to "documents the contract" is the same decision described better, and
invalidating an acceptance over it would train reviewers to re-approve noise.
Adding ``README`` to that axis's surfaces is a wider change than the one that was
accepted, and it moves the digest.

Two further exclusions are structural rather than editorial.
:data:`REVIEW_STATE_KEYS` -- ``status``, ``reviewed_by``, ``reviewed_digest`` --
are the review's own state, not the change under review; folding them in would
make the act of accepting a preview move the digest that the acceptance names.
:data:`DERIVED_PREVIEW_KEYS` -- ``preview_digest``, ``claim_boundary`` -- are a
value that cannot cover itself and a module constant identical in every preview.

Everything here is pure. Nothing is written, no clock is read, and no wall-clock
value ever enters the digest seed: the two review stamps a caller supplies are
its own parameters and stay outside the material content. The hashing scheme is
`quality.skill_governance.policy_decision_digest`, reused rather than reinvented,
because a second stable-encoding rule in this repository would be a second thing
to keep in agreement with the first.

`blueprint_ref` is the named seam for `native_capability_blueprint/v1` (#791),
which is being built separately. It is optional, unvalidated beyond being an
opaque reference, and nothing here imports or requires a blueprint; it exists so
the two artifacts can be joined later without reshaping this one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from ..quality.skill_governance import policy_decision_digest
from ..system.append_only_store import (
    RAW_OR_HIDDEN_KEYS,
    is_unsafe_metadata_line,
    opaque_ref,
    reference_errors,
)


NATIVE_CAPABILITY_CHANGE_PREVIEW_SCHEMA_VERSION: Final = "native_capability_change_preview/v1"

# The eight axes a capability change moves in this product, from AC1. Closed and
# exhaustive: a preview answers all of them, and there is no ninth to add
# without deciding what it means for every existing preview.
CHANGE_AXES: Final = (
    "routing",
    "skill",
    "memory",
    "wrapper",
    "handoff",
    "evidence",
    "docs",
    "tests",
)

# Three, not two. `unchanged` means the axis exists for this capability and this
# change leaves it alone; `not_applicable` means the axis has no bearing on the
# capability. A reviewer reading one of those cannot infer the other.
AXIS_STATES: Final = ("changed", "unchanged", "not_applicable")

# What the change does to data leaving OMH's metadata-only default.
PRIVACY_IMPACTS: Final = ("none", "metadata_only", "local_content", "external_disclosure")

RISK_SEVERITIES: Final = ("low", "medium", "high")

# The classes of gate this repository actually runs, so a verification
# expectation names something checkable rather than an intention.
VERIFICATION_KINDS: Final = ("unit_test", "generated_artifact_gate", "lint_gate", "manual_observation")

# How far back the change can be taken once it exists. `persisted_records_retained`
# is the honest middle: the code reverts, the records it already wrote do not.
ROLLBACK_BOUNDARIES: Final = (
    "no_persisted_state",
    "local_artifacts_removable",
    "persisted_records_retained",
    "irreversible",
)

# The lifecycle of the preview itself. None of these four says the capability
# exists, which is AC3 made a property of the vocabulary.
PREVIEW_STATUSES: Final = ("draft", "in_review", "accepted", "superseded")

# Words a status will not be. Held by name so the refusal says why rather than
# reporting an unrecognised value, and so the AC3 boundary is testable as data.
REFUSED_PREVIEW_STATUSES: Final = (
    "implemented",
    "available",
    "shipped",
    "released",
    "live",
    "enabled",
    "installed",
    "done",
    "complete",
    "working",
)

# The one sentence each status may be described with. Every one of them states
# that nothing was built, so no rendering of any status can read as delivery.
PREVIEW_STATUS_CLAIMS: Final = {
    "draft": "A change is described and not yet reviewed. Nothing has been built.",
    "in_review": "A described change is under review. Nothing has been built.",
    "accepted": "A described change was reviewed and accepted as the plan. Nothing has been built.",
    "superseded": (
        "The reviewed content moved, so the earlier acceptance no longer covers it. Nothing has been built."
    ),
}

# How the current material content stands against the content that was reviewed.
REVIEW_STATES: Final = ("not_reviewed", "current", "stale")

AXIS_KEYS: Final = ("axis", "state", "surfaces", "note")
RISK_KEYS: Final = ("id", "severity", "note")
VERIFICATION_KEYS: Final = ("id", "kind", "note")
ROLLBACK_KEYS: Final = ("boundary", "note")
HANDOFF_SCOPE_KEYS: Final = ("delegated", "retained", "note")

NATIVE_CAPABILITY_CHANGE_PREVIEW_KEYS: Final = (
    "schema_version",
    "capability_id",
    "target_experience",
    "baseline",
    "axes",
    "privacy_impact",
    "compatibility_risks",
    "verification_expectations",
    "rollback",
    "handoff_scope",
    "blueprint_ref",
    "status",
    "reviewed_by",
    "reviewed_digest",
    "preview_digest",
    "claim_boundary",
)

# The single explanatory field name in the schema, at every nesting level.
EXPLANATORY_FIELD: Final = "note"

# The review's own state. Outside the digest because accepting a preview must
# not move the digest that the acceptance names.
REVIEW_STATE_KEYS: Final = ("status", "reviewed_by", "reviewed_digest")

# A value that cannot cover itself, and a module constant identical everywhere.
DERIVED_PREVIEW_KEYS: Final = ("preview_digest", "claim_boundary")

# Everything else. Derived rather than listed so a key added to the schema is
# material by default and has to be argued out, never silently left out.
MATERIAL_PREVIEW_KEYS: Final = tuple(
    key
    for key in NATIVE_CAPABILITY_CHANGE_PREVIEW_KEYS
    if key not in set(REVIEW_STATE_KEYS) | set(DERIVED_PREVIEW_KEYS)
)

NATIVE_CAPABILITY_CHANGE_PREVIEW_CLAIM_BOUNDARY: Final = (
    "This preview describes a capability OMH has not built. Accepting it records that the described "
    "change was reviewed and nothing further: none of the routing, skill, memory, wrapper, handoff, "
    "evidence, docs, and tests surfaces named here exists yet, and an accepted preview is never "
    "implementation, availability, execution, code-review, CI, merge-readiness, or merge evidence."
)

_MAX_NARRATIVE_CHARS: Final = 400
_MAX_NOTE_CHARS: Final = 240
_MAX_AXIS_SURFACES: Final = 12
_MAX_RISKS: Final = 12
_MAX_VERIFICATIONS: Final = 12
_MAX_SCOPE_ITEMS: Final = 16

_LABEL: Final = "native capability change preview"

_SHA256_LENGTH: Final = 64


class NativeCapabilityChangePreviewError(ValueError):
    """Raised when a preview cannot be built or a revision cannot be applied."""


def build_native_capability_change_preview(
    *,
    capability_id: str,
    target_experience: str,
    baseline: str,
    axes: Sequence[Mapping[str, Any]],
    privacy_impact: str,
    rollback: Mapping[str, Any],
    handoff_scope: Mapping[str, Any],
    verification_expectations: Sequence[Mapping[str, Any]],
    compatibility_risks: Sequence[Mapping[str, Any]] = (),
    blueprint_ref: str = "",
    status: str = "draft",
    reviewed_by: str = "",
    reviewed_digest: str = "",
) -> dict[str, Any]:
    """Mint one preview, or refuse.

    All eight axes are required here, not defaulted: a preview that answered
    only the axes a caller remembered would be exactly the generic plan this
    artifact replaces.

    `status` defaults to `draft` and the two review fields default to empty,
    because minting a preview is not reviewing one. A caller may pass a
    reviewed triple to rebuild an existing preview, and
    :func:`validate_native_capability_change_preview` decides whether that
    triple still holds against the material content it was given.
    """
    preview = {
        "schema_version": NATIVE_CAPABILITY_CHANGE_PREVIEW_SCHEMA_VERSION,
        "capability_id": _opaque(capability_id, field=f"{_LABEL} capability_id"),
        "target_experience": _bounded_line(
            target_experience, field=f"{_LABEL} target_experience", limit=_MAX_NARRATIVE_CHARS
        ),
        "baseline": _bounded_line(baseline, field=f"{_LABEL} baseline", limit=_MAX_NARRATIVE_CHARS),
        "axes": _axis_rows(axes),
        "privacy_impact": _closed(privacy_impact, allowed=PRIVACY_IMPACTS, field=f"{_LABEL} privacy_impact"),
        "compatibility_risks": _risk_rows(compatibility_risks),
        "verification_expectations": _verification_rows(verification_expectations),
        "rollback": _rollback(rollback),
        "handoff_scope": _handoff_scope(handoff_scope),
        "blueprint_ref": _optional_opaque(blueprint_ref, field=f"{_LABEL} blueprint_ref"),
        "status": _status(status),
        "reviewed_by": _optional_opaque(reviewed_by, field=f"{_LABEL} reviewed_by"),
        "reviewed_digest": str(reviewed_digest or ""),
        "claim_boundary": NATIVE_CAPABILITY_CHANGE_PREVIEW_CLAIM_BOUNDARY,
    }
    preview["preview_digest"] = native_capability_change_preview_digest(preview)
    errors = validate_native_capability_change_preview(preview)
    if errors:
        raise NativeCapabilityChangePreviewError("; ".join(errors))
    return preview


def validate_native_capability_change_preview(preview: Any) -> list[str]:
    """Every reason a preview is not a valid one. Both directions on keys."""
    if not isinstance(preview, Mapping):
        return [f"{_LABEL} must be an object"]
    errors: list[str] = []
    forbidden = sorted(key for key in preview if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if forbidden:
        errors.append(f"{_LABEL} must not carry raw or hidden keys: {forbidden}")
    present = {str(key) for key in preview}
    missing = sorted(set(NATIVE_CAPABILITY_CHANGE_PREVIEW_KEYS) - present)
    if missing:
        errors.append(f"{_LABEL} is missing keys: {missing}")
    unexpected = sorted(present - set(NATIVE_CAPABILITY_CHANGE_PREVIEW_KEYS) - set(forbidden))
    if unexpected:
        errors.append(f"{_LABEL} has unsupported keys: {unexpected}")
    if preview.get("schema_version") != NATIVE_CAPABILITY_CHANGE_PREVIEW_SCHEMA_VERSION:
        errors.append(f"{_LABEL} schema_version must be {NATIVE_CAPABILITY_CHANGE_PREVIEW_SCHEMA_VERSION}")
    if preview.get("claim_boundary") != NATIVE_CAPABILITY_CHANGE_PREVIEW_CLAIM_BOUNDARY:
        errors.append(f"{_LABEL} claim_boundary must state the unbuilt-capability boundary")
    errors.extend(
        reference_errors(preview.get("capability_id"), field="capability_id", label=_LABEL, required=True)
    )
    errors.extend(
        reference_errors(preview.get("blueprint_ref"), field="blueprint_ref", label=_LABEL, required=False)
    )
    for field in ("target_experience", "baseline"):
        errors.extend(_line_errors(preview.get(field), field=field, limit=_MAX_NARRATIVE_CHARS, required=True))
    if preview.get("privacy_impact") not in PRIVACY_IMPACTS:
        errors.append(f"{_LABEL} privacy_impact is unsupported: {preview.get('privacy_impact')!r}")
    errors.extend(_axis_errors(preview.get("axes")))
    errors.extend(_risk_errors(preview.get("compatibility_risks")))
    errors.extend(_verification_errors(preview.get("verification_expectations")))
    errors.extend(_rollback_errors(preview.get("rollback")))
    errors.extend(_handoff_scope_errors(preview.get("handoff_scope")))
    errors.extend(_review_errors(preview))
    return errors


def native_capability_change_preview_digest(preview: Mapping[str, Any]) -> str:
    """Content hash of the material part of a preview.

    Reuses `quality.skill_governance.policy_decision_digest` rather than adding
    a second hashing scheme: it is already this repository's stable,
    order-independent, JSON-free content hash over a metadata mapping, and a
    preview digest that disagreed with it about encoding would be a second rule
    to keep correct. That function drops a key named `omh_observed_record` from
    its seed; :data:`MATERIAL_PREVIEW_KEYS` never contains one, and the closed
    key set is what keeps that true.
    """
    return policy_decision_digest(material_preview_content(preview))


def material_preview_content(preview: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a preview the digest covers: everything except `note`.

    A missing key projects as `None` rather than raising, so a malformed preview
    still has a digest and the validator -- not the hash -- is what reports the
    fault.
    """
    return {key: _without_notes(preview.get(key)) for key in MATERIAL_PREVIEW_KEYS}


def preview_review_state(preview: Mapping[str, Any]) -> str:
    """Whether the current material content is still the reviewed content.

    `not_reviewed` when nothing was ever reviewed, `current` when the reviewed
    digest matches what the preview now holds, `stale` when it does not.
    """
    reviewed = str(preview.get("reviewed_digest", "") or "")
    if not reviewed:
        return "not_reviewed"
    return "current" if reviewed == native_capability_change_preview_digest(preview) else "stale"


def preview_requires_renewed_review(preview: Mapping[str, Any]) -> bool:
    """True when the content moved past the review that was given to it."""
    return preview_review_state(preview) != "current"


def preview_status_claim(status: str) -> str:
    """The one sentence a status may be described with.

    There is no rendering path that produces anything else, which is how "an
    accepted preview cannot be described as implemented or available" survives
    contact with a surface that wants a human-readable line.
    """
    text = str(status or "")
    if text not in PREVIEW_STATUS_CLAIMS:
        raise NativeCapabilityChangePreviewError(f"{_LABEL} status is unsupported: {text!r}")
    return PREVIEW_STATUS_CLAIMS[text]


def accept_native_capability_change_preview(
    preview: Mapping[str, Any], *, reviewer: str
) -> dict[str, Any]:
    """Bind a review to the material content it reviewed.

    Accepting is the only way `accepted` is reached, and it stamps the digest of
    the content in front of the reviewer. A later material change moves that
    digest away from the stamp, which is what makes the acceptance expire
    instead of quietly widening.
    """
    errors = validate_native_capability_change_preview(preview)
    if errors:
        raise NativeCapabilityChangePreviewError("; ".join(errors))
    accepted = dict(preview)
    accepted["status"] = "accepted"
    accepted["reviewed_by"] = _opaque(reviewer, field=f"{_LABEL} reviewed_by")
    accepted["reviewed_digest"] = native_capability_change_preview_digest(preview)
    remaining = validate_native_capability_change_preview(accepted)
    if remaining:
        raise NativeCapabilityChangePreviewError("; ".join(remaining))
    return accepted


def revise_native_capability_change_preview(
    preview: Mapping[str, Any], **changes: Any
) -> dict[str, Any]:
    """Apply changes and let the digest decide whether the review survives.

    A revision that leaves the material content identical -- rewording a note,
    for instance -- keeps the acceptance exactly as it was. A revision that
    moves the material content supersedes the preview: the earlier reviewer and
    the digest they reviewed are kept as the record of what was accepted, and
    the status says the acceptance no longer covers what is there now.
    """
    errors = validate_native_capability_change_preview(preview)
    if errors:
        raise NativeCapabilityChangePreviewError("; ".join(errors))
    unknown = sorted(set(changes) - set(MATERIAL_PREVIEW_KEYS) - {"blueprint_ref"})
    if unknown:
        raise NativeCapabilityChangePreviewError(
            f"{_LABEL} revision may only change material content, not: {unknown}"
        )
    if "schema_version" in changes:
        raise NativeCapabilityChangePreviewError(f"{_LABEL} revision may not change schema_version")
    reviewed_digest = str(preview.get("reviewed_digest", "") or "")
    reviewed_by = str(preview.get("reviewed_by", "") or "")
    fields = {
        "capability_id": preview.get("capability_id", ""),
        "target_experience": preview.get("target_experience", ""),
        "baseline": preview.get("baseline", ""),
        "axes": preview.get("axes", ()),
        "privacy_impact": preview.get("privacy_impact", ""),
        "compatibility_risks": preview.get("compatibility_risks", ()),
        "verification_expectations": preview.get("verification_expectations", ()),
        "rollback": preview.get("rollback", {}),
        "handoff_scope": preview.get("handoff_scope", {}),
        "blueprint_ref": preview.get("blueprint_ref", ""),
    }
    fields.update({key: value for key, value in changes.items() if key != "schema_version"})
    revised = build_native_capability_change_preview(**fields)
    if not reviewed_digest:
        revised["status"] = _status(str(preview.get("status", "draft")))
        errors = validate_native_capability_change_preview(revised)
        if errors:
            raise NativeCapabilityChangePreviewError("; ".join(errors))
        return revised
    still_reviewed = reviewed_digest == revised["preview_digest"]
    revised["status"] = "accepted" if still_reviewed else "superseded"
    revised["reviewed_by"] = reviewed_by
    revised["reviewed_digest"] = reviewed_digest
    errors = validate_native_capability_change_preview(revised)
    if errors:
        raise NativeCapabilityChangePreviewError("; ".join(errors))
    return revised


def _axis_rows(axes: Any) -> list[dict[str, Any]]:
    if not _is_row_sequence(axes):
        raise NativeCapabilityChangePreviewError(f"{_LABEL} axes must be a list of objects")
    seen: dict[str, dict[str, Any]] = {}
    for row in axes:
        unknown = sorted({str(key) for key in row} - set(AXIS_KEYS))
        if unknown:
            raise NativeCapabilityChangePreviewError(f"{_LABEL} change axis has unsupported keys: {unknown}")
        axis = str(row.get("axis", ""))
        if axis not in CHANGE_AXES:
            raise NativeCapabilityChangePreviewError(f"{_LABEL} has an unsupported change axis: {axis!r}")
        if axis in seen:
            raise NativeCapabilityChangePreviewError(f"{_LABEL} marks change axis {axis} more than once")
        state = row.get("state")
        if state not in AXIS_STATES:
            raise NativeCapabilityChangePreviewError(
                f"{_LABEL} change axis {axis} has an unsupported state: {state!r}; "
                f"one of {list(AXIS_STATES)} is required"
            )
        surfaces = _surface_refs(row.get("surfaces", ()), axis=axis)
        _check_surface_rule(axis, str(state), surfaces, error=True)
        seen[axis] = {
            "axis": axis,
            "state": state,
            "surfaces": surfaces,
            "note": _optional_line(row.get("note", ""), field=f"{_LABEL} axis {axis} note", limit=_MAX_NOTE_CHARS),
        }
    absent = [axis for axis in CHANGE_AXES if axis not in seen]
    if absent:
        raise NativeCapabilityChangePreviewError(f"{_LABEL} is missing change axes: {absent}")
    return [seen[axis] for axis in CHANGE_AXES]


def _axis_errors(axes: Any) -> list[str]:
    if not isinstance(axes, list) or not all(isinstance(row, Mapping) for row in axes):
        return [f"{_LABEL} axes must be a list of objects"]
    errors: list[str] = []
    seen: list[str] = []
    for row in axes:
        unknown = sorted({str(key) for key in row} - set(AXIS_KEYS))
        if unknown:
            errors.append(f"{_LABEL} change axis has unsupported keys: {unknown}")
        absent_keys = sorted(set(AXIS_KEYS) - {str(key) for key in row})
        if absent_keys:
            errors.append(f"{_LABEL} change axis is missing keys: {absent_keys}")
        axis = str(row.get("axis", ""))
        if axis not in CHANGE_AXES:
            errors.append(f"{_LABEL} has an unsupported change axis: {row.get('axis')!r}")
            continue
        if axis in seen:
            errors.append(f"{_LABEL} marks change axis {axis} more than once")
        seen.append(axis)
        state = row.get("state")
        if state not in AXIS_STATES:
            errors.append(
                f"{_LABEL} change axis {axis} has an unsupported state: {state!r}; "
                f"one of {list(AXIS_STATES)} is required"
            )
        errors.extend(_surface_errors(row.get("surfaces"), axis=axis))
        if state in AXIS_STATES and isinstance(row.get("surfaces"), list):
            rule = _check_surface_rule(axis, str(state), [str(item) for item in row["surfaces"]], error=False)
            if rule:
                errors.append(rule)
        errors.extend(
            _line_errors(row.get("note"), field=f"axis {axis} note", limit=_MAX_NOTE_CHARS, required=False)
        )
    absent = [axis for axis in CHANGE_AXES if axis not in seen]
    if absent:
        errors.append(f"{_LABEL} is missing change axes: {absent}")
    elif seen != list(CHANGE_AXES):
        errors.append(f"{_LABEL} axes must be listed in canonical order: {list(CHANGE_AXES)}")
    return errors


def _check_surface_rule(axis: str, state: str, surfaces: Sequence[str], *, error: bool) -> str:
    """A changed axis names its surfaces; an unchanged one names none."""
    message = ""
    if state == "changed" and not surfaces:
        message = f"{_LABEL} change axis {axis} is marked changed and must name at least one affected surface"
    elif state in ("unchanged", "not_applicable") and surfaces:
        message = f"{_LABEL} change axis {axis} is marked {state} and must not name an affected surface"
    if message and error:
        raise NativeCapabilityChangePreviewError(message)
    return message


def _surface_refs(value: Any, *, axis: str) -> list[str]:
    items = _string_items(value, field=f"{_LABEL} axis {axis} surfaces")
    if len(items) > _MAX_AXIS_SURFACES:
        raise NativeCapabilityChangePreviewError(
            f"{_LABEL} change axis {axis} must name at most {_MAX_AXIS_SURFACES} surfaces"
        )
    refs = [_opaque(item, field=f"{_LABEL} axis {axis} surface") for item in items]
    if len(set(refs)) != len(refs):
        raise NativeCapabilityChangePreviewError(f"{_LABEL} change axis {axis} names the same surface twice")
    return sorted(refs)


def _surface_errors(value: Any, *, axis: str) -> list[str]:
    if not isinstance(value, list):
        return [f"{_LABEL} change axis {axis} surfaces must be a list"]
    if len(value) > _MAX_AXIS_SURFACES:
        return [f"{_LABEL} change axis {axis} must name at most {_MAX_AXIS_SURFACES} surfaces"]
    errors: list[str] = []
    for item in value:
        errors.extend(reference_errors(item, field=f"axis {axis} surface", label=_LABEL, required=True))
    names = [str(item) for item in value]
    if len(set(names)) != len(names):
        errors.append(f"{_LABEL} change axis {axis} names the same surface twice")
    if names != sorted(names):
        errors.append(f"{_LABEL} change axis {axis} surfaces must be sorted")
    return errors


def _risk_rows(risks: Any) -> list[dict[str, Any]]:
    return _identified_rows(
        risks,
        keys=RISK_KEYS,
        kind="compatibility risk",
        limit=_MAX_RISKS,
        closed_field="severity",
        allowed=RISK_SEVERITIES,
    )


def _risk_errors(risks: Any) -> list[str]:
    return _identified_row_errors(
        risks,
        keys=RISK_KEYS,
        kind="compatibility risk",
        field="compatibility_risks",
        limit=_MAX_RISKS,
        closed_field="severity",
        allowed=RISK_SEVERITIES,
        required=False,
    )


def _verification_rows(expectations: Any) -> list[dict[str, Any]]:
    rows = _identified_rows(
        expectations,
        keys=VERIFICATION_KEYS,
        kind="verification expectation",
        limit=_MAX_VERIFICATIONS,
        closed_field="kind",
        allowed=VERIFICATION_KINDS,
    )
    if not rows:
        raise NativeCapabilityChangePreviewError(
            f"{_LABEL} must name at least one verification expectation"
        )
    return rows


def _verification_errors(expectations: Any) -> list[str]:
    return _identified_row_errors(
        expectations,
        keys=VERIFICATION_KEYS,
        kind="verification expectation",
        field="verification_expectations",
        limit=_MAX_VERIFICATIONS,
        closed_field="kind",
        allowed=VERIFICATION_KINDS,
        required=True,
    )


def _identified_rows(
    value: Any,
    *,
    keys: tuple[str, ...],
    kind: str,
    limit: int,
    closed_field: str,
    allowed: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not _is_row_sequence(value):
        raise NativeCapabilityChangePreviewError(f"{_LABEL} {kind} rows must be a list of objects")
    rows = list(value)
    if len(rows) > limit:
        raise NativeCapabilityChangePreviewError(f"{_LABEL} must hold at most {limit} {kind} rows")
    built: list[dict[str, Any]] = []
    for row in rows:
        unknown = sorted({str(key) for key in row} - set(keys))
        if unknown:
            raise NativeCapabilityChangePreviewError(f"{_LABEL} {kind} has unsupported keys: {unknown}")
        identifier = _opaque(row.get("id", ""), field=f"{_LABEL} {kind} id")
        built.append(
            {
                "id": identifier,
                closed_field: _closed(
                    row.get(closed_field), allowed=allowed, field=f"{_LABEL} {kind} {closed_field}"
                ),
                "note": _optional_line(
                    row.get("note", ""), field=f"{_LABEL} {kind} note", limit=_MAX_NOTE_CHARS
                ),
            }
        )
    identifiers = [row["id"] for row in built]
    if len(set(identifiers)) != len(identifiers):
        raise NativeCapabilityChangePreviewError(f"{_LABEL} lists the same {kind} id twice")
    return sorted(built, key=lambda row: str(row["id"]))


def _identified_row_errors(
    value: Any,
    *,
    keys: tuple[str, ...],
    kind: str,
    field: str,
    limit: int,
    closed_field: str,
    allowed: tuple[str, ...],
    required: bool,
) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        return [f"{_LABEL} {field} must be a list of objects"]
    if len(value) > limit:
        return [f"{_LABEL} must hold at most {limit} {kind} rows"]
    if required and not value:
        return [f"{_LABEL} must name at least one {kind}"]
    errors: list[str] = []
    for row in value:
        if {str(key) for key in row} != set(keys):
            errors.append(f"{_LABEL} {kind} must carry exactly {', '.join(keys)}")
            continue
        errors.extend(reference_errors(row.get("id"), field=f"{kind} id", label=_LABEL, required=True))
        if row.get(closed_field) not in allowed:
            errors.append(f"{_LABEL} {kind} {closed_field} is unsupported: {row.get(closed_field)!r}")
        errors.extend(
            _line_errors(row.get("note"), field=f"{kind} note", limit=_MAX_NOTE_CHARS, required=False)
        )
    identifiers = [str(row.get("id", "")) for row in value]
    if len(set(identifiers)) != len(identifiers):
        errors.append(f"{_LABEL} lists the same {kind} id twice")
    if identifiers != sorted(identifiers):
        errors.append(f"{_LABEL} {field} must be sorted by id")
    return errors


def _rollback(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeCapabilityChangePreviewError(f"{_LABEL} rollback must be an object")
    unknown = sorted({str(key) for key in value} - set(ROLLBACK_KEYS))
    if unknown:
        raise NativeCapabilityChangePreviewError(f"{_LABEL} rollback has unsupported keys: {unknown}")
    return {
        "boundary": _closed(
            value.get("boundary"), allowed=ROLLBACK_BOUNDARIES, field=f"{_LABEL} rollback boundary"
        ),
        "note": _optional_line(value.get("note", ""), field=f"{_LABEL} rollback note", limit=_MAX_NOTE_CHARS),
    }


def _rollback_errors(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{_LABEL} rollback must be an object"]
    errors: list[str] = []
    if {str(key) for key in value} != set(ROLLBACK_KEYS):
        errors.append(f"{_LABEL} rollback must carry exactly {', '.join(ROLLBACK_KEYS)}")
    if value.get("boundary") not in ROLLBACK_BOUNDARIES:
        errors.append(f"{_LABEL} rollback boundary is unsupported: {value.get('boundary')!r}")
    errors.extend(_line_errors(value.get("note"), field="rollback note", limit=_MAX_NOTE_CHARS, required=False))
    return errors


def _handoff_scope(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeCapabilityChangePreviewError(f"{_LABEL} handoff_scope must be an object")
    unknown = sorted({str(key) for key in value} - set(HANDOFF_SCOPE_KEYS))
    if unknown:
        raise NativeCapabilityChangePreviewError(f"{_LABEL} handoff_scope has unsupported keys: {unknown}")
    delegated = _work_refs(value.get("delegated", ()), field="delegated")
    if not delegated:
        raise NativeCapabilityChangePreviewError(
            f"{_LABEL} handoff_scope must name at least one delegated unit of work"
        )
    return {
        "delegated": delegated,
        "retained": _work_refs(value.get("retained", ()), field="retained"),
        "note": _optional_line(
            value.get("note", ""), field=f"{_LABEL} handoff_scope note", limit=_MAX_NOTE_CHARS
        ),
    }


def _handoff_scope_errors(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{_LABEL} handoff_scope must be an object"]
    errors: list[str] = []
    if {str(key) for key in value} != set(HANDOFF_SCOPE_KEYS):
        errors.append(f"{_LABEL} handoff_scope must carry exactly {', '.join(HANDOFF_SCOPE_KEYS)}")
    for field in ("delegated", "retained"):
        items = value.get(field)
        if not isinstance(items, list):
            errors.append(f"{_LABEL} handoff_scope {field} must be a list")
            continue
        if len(items) > _MAX_SCOPE_ITEMS:
            errors.append(f"{_LABEL} handoff_scope {field} must hold at most {_MAX_SCOPE_ITEMS} items")
        for item in items:
            errors.extend(
                reference_errors(item, field=f"handoff_scope {field} item", label=_LABEL, required=True)
            )
        names = [str(item) for item in items]
        if len(set(names)) != len(names):
            errors.append(f"{_LABEL} handoff_scope {field} lists the same work twice")
        if names != sorted(names):
            errors.append(f"{_LABEL} handoff_scope {field} must be sorted")
    if isinstance(value.get("delegated"), list) and not value["delegated"]:
        errors.append(f"{_LABEL} handoff_scope must name at least one delegated unit of work")
    errors.extend(
        _line_errors(value.get("note"), field="handoff_scope note", limit=_MAX_NOTE_CHARS, required=False)
    )
    return errors


def _work_refs(value: Any, *, field: str) -> list[str]:
    items = _string_items(value, field=f"{_LABEL} handoff_scope {field}")
    if len(items) > _MAX_SCOPE_ITEMS:
        raise NativeCapabilityChangePreviewError(
            f"{_LABEL} handoff_scope {field} must hold at most {_MAX_SCOPE_ITEMS} items"
        )
    refs = [_opaque(item, field=f"{_LABEL} handoff_scope {field} item") for item in items]
    if len(set(refs)) != len(refs):
        raise NativeCapabilityChangePreviewError(f"{_LABEL} handoff_scope {field} lists the same work twice")
    return sorted(refs)


def _review_errors(preview: Mapping[str, Any]) -> list[str]:
    """The invariant AC2 rests on, checked on the finished preview.

    `accepted` is only valid while the reviewed digest is the current one, and
    `superseded` is only valid while it is not. Between them there is no way to
    hold an acceptance over content the reviewer never saw.
    """
    errors: list[str] = []
    status = preview.get("status")
    if isinstance(status, str) and status.strip().lower() in REFUSED_PREVIEW_STATUSES:
        errors.append(
            f"{_LABEL} status may not claim the capability exists: {status!r}; a preview is a plan, "
            f"and one of {list(PREVIEW_STATUSES)} is required"
        )
    elif status not in PREVIEW_STATUSES:
        errors.append(f"{_LABEL} status is unsupported: {status!r}")
    errors.extend(reference_errors(preview.get("reviewed_by"), field="reviewed_by", label=_LABEL, required=False))
    reviewed_digest = preview.get("reviewed_digest")
    if not isinstance(reviewed_digest, str):
        errors.append(f"{_LABEL} reviewed_digest must be a string")
        return errors
    if reviewed_digest and not _is_sha256(reviewed_digest):
        errors.append(f"{_LABEL} reviewed_digest must be a sha256 hex digest")
    digest = preview.get("preview_digest")
    current = native_capability_change_preview_digest(preview)
    if digest != current:
        errors.append(f"{_LABEL} preview_digest must be the digest of its own material content")
    reviewed_by = str(preview.get("reviewed_by", "") or "")
    if status in ("draft", "in_review"):
        if reviewed_digest or reviewed_by:
            errors.append(f"{_LABEL} status {status} must not carry a review")
    elif status == "accepted":
        if not reviewed_by:
            errors.append(f"{_LABEL} an accepted preview must name the reviewer")
        if reviewed_digest != current:
            errors.append(
                f"{_LABEL} an accepted preview must carry the digest it was reviewed at; the material "
                "content moved and the review has to be renewed"
            )
    elif status == "superseded":
        if not reviewed_by or not reviewed_digest:
            errors.append(f"{_LABEL} a superseded preview must keep the review it outgrew")
        elif reviewed_digest == current:
            errors.append(
                f"{_LABEL} a superseded preview must differ from the content that was reviewed"
            )
    return errors


def _without_notes(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_notes(item)
            for key, item in value.items()
            if str(key) != EXPLANATORY_FIELD
        }
    if isinstance(value, (list, tuple)):
        return [_without_notes(item) for item in value]
    return value


def _is_row_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and all(isinstance(row, Mapping) for row in value)
    )


def _string_items(value: Any, *, field: str) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise NativeCapabilityChangePreviewError(f"{field} must be a list")
    return [str(item) for item in value]


def _closed(value: Any, *, allowed: tuple[str, ...], field: str) -> str:
    text = str(value or "")
    if text not in allowed:
        raise NativeCapabilityChangePreviewError(f"{field} is unsupported: {value!r}")
    return text


def _status(value: Any) -> str:
    text = str(value or "")
    if text.strip().lower() in REFUSED_PREVIEW_STATUSES:
        raise NativeCapabilityChangePreviewError(
            f"{_LABEL} status may not claim the capability exists: {text!r}; a preview is a plan, "
            f"and one of {list(PREVIEW_STATUSES)} is required"
        )
    if text not in PREVIEW_STATUSES:
        raise NativeCapabilityChangePreviewError(f"{_LABEL} status is unsupported: {value!r}")
    return text


def _opaque(value: Any, *, field: str) -> str:
    return opaque_ref(str(value or ""), field=field, error=NativeCapabilityChangePreviewError)


def _optional_opaque(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    return _opaque(text, field=field) if text else ""


def _bounded_line(value: Any, *, field: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise NativeCapabilityChangePreviewError(f"{field} is required")
    if len(text) > limit:
        raise NativeCapabilityChangePreviewError(f"{field} must be at most {limit} characters")
    if is_unsafe_metadata_line(text):
        raise NativeCapabilityChangePreviewError(
            f"{field} must be one bounded metadata line without secrets, links, or paths"
        )
    return text


def _optional_line(value: Any, *, field: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return _bounded_line(text, field=field, limit=limit) if text else ""


def _line_errors(value: Any, *, field: str, limit: int, required: bool) -> list[str]:
    if not isinstance(value, str):
        return [f"{_LABEL} {field} must be a string"]
    if not value.strip():
        return [f"{_LABEL} {field} is required"] if required else []
    if len(value) > limit:
        return [f"{_LABEL} {field} must be at most {limit} characters"]
    if is_unsafe_metadata_line(value):
        return [f"{_LABEL} {field} must not carry secrets, links, paths, or raw text"]
    return []


def _is_sha256(value: str) -> bool:
    if len(value) != _SHA256_LENGTH or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
