"""Definition of done for one native capability: `native_capability_quality_gate/v1` (issue #795).

What was missing
----------------

A capability looks present as soon as a route fires or a skill name resolves.
The chat copy, the memory rules, the handoff behavior, the evidence ladder, the
tests, and the documentation are each enforced by their own gate, and every one
of those gates answers a question about the repository rather than a question
about one capability. Nobody could ask "is this capability genuinely complete?"
and get an answer, because the answer was scattered across a dozen suites that
each pass happily while a capability is half-built.

What this is
------------

One capability, checked against the surfaces #791 says an installable capability
moves, with three verdicts a maintainer can act on:

    pass     every expected surface is answered, and generated guidance reproduces
    revise   a surface this capability needs is absent
    blocked  the repository's own checks do not reproduce, so completeness is unknowable

`blocked` is not a worse `revise`. It says the question could not be answered:
while the catalog does not validate or a generated file is stale, "this
capability is missing an awareness lane" is a statement nobody should act on,
because the surfaces are being read out of a tree that no longer agrees with
itself. Telling an author to add a lane in that state is worse than telling them
nothing.

The surface vocabulary is #791's, imported
------------------------------------------

This module names no surface of its own. `REQUIRED_NATIVE_CAPABILITY_SURFACES`,
the wider vocabulary, and the per-surface anchors all come from
`native_capability_blueprint`, and `tests/test_native_capability_quality_gate.py`
fails if a surface name is ever written as a literal here. Two lists of native
surfaces that drift apart would be worse than one list nobody enforces.

When a blueprint is supplied, `blueprint_expected_surfaces` -- the accessor #791
built for this reader -- widens the expected set with whatever conditional
surfaces the design promised. A capability whose blueprint promised a memory
policy has to answer for one; a capability whose blueprint did not, does not.
An invalid blueprint raises out of that accessor and is deliberately not
re-wrapped: the fault is in the blueprint, and the sentence that names it is
#791's, not this module's.

Absent, or exempt with a reason. Never silent
---------------------------------------------

AC1's two halves are one rule. Every expected surface carries a row, and a
surface with no row at all is refused by name and by the file it lives in --
silence is not an available answer. A row is `present`, `missing`, or `exempt`,
and an exemption carries the reason it is one. An exemption with no reason is a
validation error, because "this surface does not apply here" is an argument
somebody has to actually make; without it, `exempt` is just `missing` spelled in
a way that passes.

The reason is bounded to one metadata line for the same reason every other
free-text field in this repository is: it is a place a transcript would
otherwise get parked.

A required surface can be exempted, and that is deliberate even though #791
calls the required set the one no installable capability can avoid. The required
set is what a capability must *answer for*; the exemption is the answer that
says why this one does not need it, written down where a reviewer reads it.
Refusing the exemption outright would not make the surface appear -- it would
make an author write `present` and move on, which is the failure with the
evidence removed.

Reproducibility is reported, never recomputed
----------------------------------------------

AC2 asks whether generated guidance still reproduces. This repository already
answers that twice -- `catalog_validation/v1` for the catalog the guidance is
rendered from, and the generated-artifact half of `omh_drift_report/v1` for the
committed files, which is the same byte comparison `omh docs workflows --check`,
`omh docs roles --check`, and `omh docs capability-families --check` make.
`generated_guidance_reproducibility` projects those two and adds nothing. It
regenerates nothing either; the regenerate commands stay where they already are.

Count and budget metrics are deliberately left out of that projection: they are
exact-count fixtures, not generated guidance, and a capability's completeness
should not hinge on a number in a test file that a different change moved.

The verdict is derived on the way in and on the way out
-------------------------------------------------------

`build_native_capability_quality_gate` has no verdict parameter, so a caller
cannot state one. That alone would only move the problem to whoever hands a
payload around, so `validate_native_capability_quality_gate` re-derives the
verdict from the findings the payload carries and refuses one that disagrees.
The same rule runs one level down on the reproducibility block: its
`reproducible` flag is re-derived from its own parts, and a reasonless exemption
counts as unanswered inside the derivation itself rather than only in the key
checks. Between them there is no arrangement of a payload where `pass` sits on
top of an incomplete gate.

Deliberately no digest. A digest seals content against edits made after a
reviewer agreed to it, and this schema carries no review to bind one to. The
property it would buy -- a payload that cannot be quietly rearranged -- is
already bought by re-deriving the verdict, and a second hashing rule in this
repository is a second thing to keep in agreement with the first.

Structure is not evidence
-------------------------

A `pass` says the structure is complete. It does not say the capability was
exercised, that a test ran, that anyone reviewed it, that CI is green, or that
anything shipped. `NATIVE_CAPABILITY_QUALITY_GATE_CLAIM_BOUNDARY` says so on
every payload, every verdict's one permitted sentence repeats the denial,
`REFUSED_QUALITY_GATE_VERDICTS` refuses the words that would blur it by name,
and #791's `IMPLEMENTATION_CLAIM_KEYS` refuses a payload reshaped to say
otherwise.

What this does not check
------------------------

That an answer is true. A row saying a surface is `present` is the author's
statement, and this gate makes it impossible to leave one out, not impossible to
get one wrong. The coverage gates in `tests/test_capabilities.py`,
`tests/test_wrapper_contract.py`, and the routing suites are what fail on a
false `present`, and duplicating them here would be a second set of rules to
keep in agreement with the first.

Determinism
-----------

No clock is read. `prepared_at` is a parameter and defaults to empty; it is the
caller's stamp for when the reproducibility block -- which is a point-in-time
read of a working tree -- was taken.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from ..maintenance.drift import drift_report
from ..skills.validation import validate_catalog_contract
from ..system.append_only_store import RAW_OR_HIDDEN_KEYS, is_unsafe_metadata_line
from .native_capability_blueprint import (
    IMPLEMENTATION_CLAIM_KEYS,
    NATIVE_CAPABILITY_SURFACES,
    REQUIRED_NATIVE_CAPABILITY_SURFACES,
    blueprint_expected_surfaces,
    blueprint_surface_anchor,
    is_canonical_capability_id,
    missing_required_surfaces,
    unknown_surfaces,
)


NATIVE_CAPABILITY_QUALITY_GATE_SCHEMA_VERSION: Final[str] = "native_capability_quality_gate/v1"

QUALITY_GATE_PRIVACY: Final[str] = "metadata_only"

# Three, and they are not a severity scale. `revise` is about the capability;
# `blocked` is about whether the question can be answered at all.
QUALITY_GATE_VERDICTS: Final[tuple[str, ...]] = ("pass", "revise", "blocked")

# Words a verdict will not be. Held by name so the refusal reads as a rule
# rather than as an unrecognised value, and so AC3 is testable as data.
REFUSED_QUALITY_GATE_VERDICTS: Final[tuple[str, ...]] = (
    "approved",
    "certified",
    "ci_green",
    "complete",
    "delivered",
    "deployed",
    "done",
    "green",
    "merged",
    "observed",
    "passing",
    "released",
    "reviewed",
    "shipped",
    "tested",
    "validated",
    "verified",
    "working",
)

# Repeated verbatim in every verdict's sentence. One string so no rendering of
# any verdict can exist that forgets it.
VERDICT_CLAIM_DENIAL: Final[str] = "Nothing here was run, reviewed, checked by CI, merged, or released."

# The one sentence each verdict may be described with.
QUALITY_GATE_VERDICT_CLAIMS: Final[dict[str, str]] = {
    "pass": (
        "Every expected native surface is answered and the repository's generated guidance reproduces. "
        f"{VERDICT_CLAIM_DENIAL}"
    ),
    "revise": (
        "A native surface this capability needs is absent or unanswered, so the capability is not complete. "
        f"{VERDICT_CLAIM_DENIAL}"
    ),
    "blocked": (
        "The repository's own catalog and generated-output checks do not reproduce, so this capability's "
        f"completeness cannot be judged from this tree yet. {VERDICT_CLAIM_DENIAL}"
    ),
}

# What one surface row may say. `exempt` is the recorded not-applicable, and it
# is the only state that carries a reason.
SURFACE_STATES: Final[tuple[str, ...]] = ("present", "missing", "exempt")

SURFACE_FINDING_KEYS: Final[tuple[str, ...]] = ("surface", "state", "reason")

GENERATED_GUIDANCE_KEYS: Final[tuple[str, ...]] = (
    "catalog_validation_errors",
    "catalog_validation_ok",
    "checked",
    "reproducible",
    "stale_artifacts",
)

NATIVE_CAPABILITY_QUALITY_GATE_KEYS: Final[tuple[str, ...]] = (
    "capability_id",
    "claim_boundary",
    "expected_surfaces",
    "generated_guidance",
    "prepared_at",
    "privacy",
    "schema_version",
    "surfaces",
    "unmet_surfaces",
    "verdict",
    "verdict_claim",
)

NATIVE_CAPABILITY_QUALITY_GATE_CLAIM_BOUNDARY: Final[str] = (
    "A native capability quality gate is an OMH-local structural check of one capability against the "
    "required native surfaces of native_capability_blueprint/v1 and this repository's own catalog and "
    "generated-output checks. A pass means the structure is complete and nothing more: it is never "
    "execution, runtime, test, code-review, CI, merge-readiness, merge, or release evidence, and it never "
    "certifies, ranks, or admits an external package."
)

_LABEL: Final[str] = "native_capability_quality_gate"

# An exemption is an argument. Long enough that `n/a`, `none`, `-`, and `skip`
# are not one.
_MIN_REASON_CHARS: Final[int] = 12
_MAX_REASON_CHARS: Final[int] = 240


class NativeCapabilityQualityGateError(ValueError):
    """Raised when a set of findings cannot become a quality gate."""


# ---------------------------------------------------------------------------
# Expected surfaces
# ---------------------------------------------------------------------------


def expected_gate_surfaces(
    *,
    blueprint: Mapping[str, Any] | None = None,
    answered: Iterable[str] = (),
) -> tuple[str, ...]:
    """The surfaces one capability has to answer for, in #791's declared order.

    Always every required surface. Plus every surface a supplied blueprint
    promised, so a design that named a conditional surface cannot then skip it.
    Plus every surface the caller answered anyway, because answering more than
    was asked is not an error and dropping the extra answer silently would be.

    A supplied blueprint is read through `blueprint_expected_surfaces`, which
    refuses an invalid blueprint rather than handing back a partial list.
    """
    named = set(REQUIRED_NATIVE_CAPABILITY_SURFACES)
    if blueprint is not None:
        named |= set(blueprint_expected_surfaces(blueprint))
    answered_names = [str(surface) for surface in answered]
    unknown = unknown_surfaces(answered_names)
    if unknown:
        raise NativeCapabilityQualityGateError(
            f"{_LABEL} names surfaces that do not exist in this repository: {list(unknown)}; "
            f"the vocabulary is {list(NATIVE_CAPABILITY_SURFACES)}"
        )
    named |= set(answered_names)
    return tuple(surface for surface in NATIVE_CAPABILITY_SURFACES if surface in named)


def unmet_surface_anchors(gate: Mapping[str, Any]) -> tuple[str, ...]:
    """Every unanswered surface with the file and structure #791 says it lives in.

    Derived on demand rather than stored: the anchor text belongs to
    `native_capability_blueprint/v1`, and a copy of it inside a report is a copy
    that can disagree with the original.
    """
    return tuple(
        f"{surface} ({blueprint_surface_anchor(surface)})"
        for surface in quality_gate_unmet_surfaces(gate)
    )


# ---------------------------------------------------------------------------
# Reproducibility of generated guidance
# ---------------------------------------------------------------------------


def generated_guidance_reproducibility(*, repo_root: Path | None = None) -> dict[str, Any]:
    """Whether generated guidance reproduces, as this repository already checks it.

    Two existing checks, projected and nothing else: `catalog_validation/v1`
    for the catalog every generated file is rendered from, and the
    generated-artifact half of `omh_drift_report/v1` for the committed files,
    which is the byte comparison the `--check` gates already make.

    Nothing is regenerated here. When something is stale, the fix is the
    regenerate command the drift report already names.
    """
    validation = validate_catalog_contract()
    report = drift_report(repo_root=repo_root, counts=(), budgets=())
    stale = sorted(
        {
            str(item.get("name", ""))
            for item in report.get("drift", [])
            if isinstance(item, Mapping) and str(item.get("name", ""))
        }
    )
    block: dict[str, Any] = {
        "catalog_validation_errors": [str(error) for error in _string_list(validation.get("errors"))],
        "catalog_validation_ok": validation.get("ok") is True,
        "checked": ["catalog_validation", *(str(name) for name in _string_list(report.get("checked")))],
        "reproducible": False,
        "stale_artifacts": stale,
    }
    block["reproducible"] = generated_guidance_reproduces(block)
    return block


def generated_guidance_reproduces(block: Mapping[str, Any]) -> bool:
    """Re-derived from the block's own parts, never read off its stored flag.

    An empty `checked` list is not reproducible: nothing having been checked is
    not the same answer as everything having reproduced.
    """
    return (
        block.get("catalog_validation_ok") is True
        and not _string_list(block.get("catalog_validation_errors"))
        and not _string_list(block.get("stale_artifacts"))
        and bool(_string_list(block.get("checked")))
    )


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def quality_gate_unmet_surfaces(gate: Mapping[str, Any]) -> tuple[str, ...]:
    """Every expected surface this capability has not answered for.

    A surface is answered when it is `present`, or `exempt` with a reason. A
    required surface with no row at all is unanswered by the same rule that
    makes a `missing` row unanswered -- an omission cannot be quieter than a
    stated absence.
    """
    answered: dict[str, Mapping[str, Any]] = {}
    for row in _mapping_rows(gate.get("surfaces")):
        surface = str(row.get("surface", ""))
        if surface and surface not in answered:
            answered[surface] = row
    expected = {
        str(surface) for surface in _string_list(gate.get("expected_surfaces"))
    } | set(REQUIRED_NATIVE_CAPABILITY_SURFACES)
    return tuple(
        surface
        for surface in NATIVE_CAPABILITY_SURFACES
        if surface in expected and not _surface_is_answered(answered.get(surface))
    )


def derive_quality_gate_verdict(gate: Mapping[str, Any]) -> str:
    """The verdict the findings produce. `blocked` outranks `revise` outranks `pass`.

    The only place a verdict is ever decided. The builder calls it because it
    accepts no verdict, and the validator calls it because a payload that
    arrived from somewhere else must not be taken at its word.
    """
    guidance = gate.get("generated_guidance")
    if not isinstance(guidance, Mapping) or not generated_guidance_reproduces(guidance):
        return "blocked"
    if quality_gate_unmet_surfaces(gate):
        return "revise"
    return "pass"


def quality_gate_verdict_claim(verdict: str) -> str:
    """The one sentence a verdict may be described with."""
    text = str(verdict or "")
    if text not in QUALITY_GATE_VERDICT_CLAIMS:
        raise NativeCapabilityQualityGateError(f"{_LABEL} verdict is unsupported: {verdict!r}")
    return QUALITY_GATE_VERDICT_CLAIMS[text]


# ---------------------------------------------------------------------------
# Build and validate
# ---------------------------------------------------------------------------


def build_native_capability_quality_gate(
    *,
    capability_id: str,
    surfaces: Sequence[Mapping[str, Any]],
    blueprint: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
    prepared_at: str = "",
) -> dict[str, Any]:
    """Gate one capability, or refuse.

    There is no verdict parameter and there will not be one. The verdict is a
    conclusion about the findings, and a caller who could state it could state
    `pass` over an incomplete gate, which is the entire failure this artifact
    exists to prevent.

    `repo_root` selects the tree whose generated guidance is read; the default
    is this installation's own repository.
    """
    rows = _finding_rows(surfaces)
    expected = expected_gate_surfaces(
        blueprint=blueprint, answered=[str(row["surface"]) for row in rows]
    )
    gate: dict[str, Any] = {
        "schema_version": NATIVE_CAPABILITY_QUALITY_GATE_SCHEMA_VERSION,
        "capability_id": str(capability_id).strip().casefold(),
        "expected_surfaces": list(expected),
        "surfaces": rows,
        "generated_guidance": generated_guidance_reproducibility(repo_root=repo_root),
        "unmet_surfaces": [],
        "verdict": "",
        "verdict_claim": "",
        "prepared_at": str(prepared_at).strip(),
        "privacy": QUALITY_GATE_PRIVACY,
        "claim_boundary": NATIVE_CAPABILITY_QUALITY_GATE_CLAIM_BOUNDARY,
    }
    gate["unmet_surfaces"] = list(quality_gate_unmet_surfaces(gate))
    gate["verdict"] = derive_quality_gate_verdict(gate)
    gate["verdict_claim"] = quality_gate_verdict_claim(gate["verdict"])
    errors = validate_native_capability_quality_gate(gate)
    if errors:
        raise NativeCapabilityQualityGateError("; ".join(errors))
    return gate


def validate_native_capability_quality_gate(gate: Any) -> list[str]:
    """Every reason one payload is not a native capability quality gate."""
    if not isinstance(gate, Mapping):
        return [f"{_LABEL} must be an object"]
    errors: list[str] = []
    claims_implementation = sorted(key for key in gate if str(key).lower() in IMPLEMENTATION_CLAIM_KEYS)
    if claims_implementation:
        errors.append(
            f"{_LABEL} must not carry implementation-claim keys: {claims_implementation}; a structural "
            "gate reports what exists in this tree and never reports that something ran"
        )
    forbidden = sorted(key for key in gate if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if forbidden:
        errors.append(f"{_LABEL} must not carry raw or hidden keys: {forbidden}")
    unexpected = sorted(
        set(gate) - set(NATIVE_CAPABILITY_QUALITY_GATE_KEYS) - set(claims_implementation) - set(forbidden)
    )
    if unexpected:
        errors.append(f"{_LABEL} has unsupported keys: {unexpected}")
    missing = sorted(set(NATIVE_CAPABILITY_QUALITY_GATE_KEYS) - set(gate))
    if missing:
        errors.append(f"{_LABEL} is missing keys: {missing}")
    if gate.get("schema_version") != NATIVE_CAPABILITY_QUALITY_GATE_SCHEMA_VERSION:
        errors.append(f"{_LABEL} schema_version must be {NATIVE_CAPABILITY_QUALITY_GATE_SCHEMA_VERSION}")
    if gate.get("privacy") != QUALITY_GATE_PRIVACY:
        errors.append(f"{_LABEL} privacy must be {QUALITY_GATE_PRIVACY}")
    if gate.get("claim_boundary") != NATIVE_CAPABILITY_QUALITY_GATE_CLAIM_BOUNDARY:
        errors.append(
            f"{_LABEL} claim_boundary must state the structural boundary: a complete structure is not "
            "execution, review, CI, merge, or release evidence"
        )
    if not isinstance(gate.get("prepared_at"), str):
        errors.append(f"{_LABEL} prepared_at must be a string")
    if not is_canonical_capability_id(gate.get("capability_id")):
        errors.append(
            f"{_LABEL} capability_id must be the canonical skill name as a lowercase slug: "
            f"{gate.get('capability_id')!r}"
        )
    errors.extend(_expected_surface_errors(gate))
    errors.extend(_finding_errors(gate))
    errors.extend(_generated_guidance_errors(gate.get("generated_guidance")))
    errors.extend(_verdict_errors(gate))
    return errors


# ---------------------------------------------------------------------------
# Validation parts
# ---------------------------------------------------------------------------


def _expected_surface_errors(gate: Mapping[str, Any]) -> list[str]:
    """AC1's first half: the expected set is #791's, and it is complete."""
    expected = gate.get("expected_surfaces")
    if not isinstance(expected, list) or not all(isinstance(item, str) for item in expected):
        return [f"{_LABEL} expected_surfaces must be a list of strings"]
    errors: list[str] = []
    if len(set(expected)) != len(expected):
        errors.append(f"{_LABEL} expected_surfaces must not repeat a surface")
    unknown = unknown_surfaces(expected)
    if unknown:
        errors.append(
            f"{_LABEL} expected_surfaces names surfaces that do not exist in this repository: "
            f"{list(unknown)}; the vocabulary is {list(NATIVE_CAPABILITY_SURFACES)}"
        )
    absent = missing_required_surfaces(expected)
    if absent:
        described = [f"{surface} ({blueprint_surface_anchor(surface)})" for surface in absent]
        errors.append(
            f"{_LABEL} expected_surfaces is missing required surfaces: {described}; every installable "
            "capability moves all of them, so a gate that does not expect one cannot report on it"
        )
    canonical = [surface for surface in NATIVE_CAPABILITY_SURFACES if surface in set(expected)]
    if not unknown and expected != canonical:
        errors.append(f"{_LABEL} expected_surfaces must be listed in the declared surface order")
    return errors


def _finding_errors(gate: Mapping[str, Any]) -> list[str]:
    """AC1's second half: absent, or exempt with a reason, and never silent."""
    rows = gate.get("surfaces")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        return [f"{_LABEL} surfaces must be a list of objects"]
    errors: list[str] = []
    seen: list[str] = []
    for row in rows:
        unknown_keys = sorted({str(key) for key in row} - set(SURFACE_FINDING_KEYS))
        if unknown_keys:
            errors.append(f"{_LABEL} surface finding has unsupported keys: {unknown_keys}")
        absent_keys = sorted(set(SURFACE_FINDING_KEYS) - {str(key) for key in row})
        if absent_keys:
            errors.append(f"{_LABEL} surface finding is missing keys: {absent_keys}")
        surface = str(row.get("surface", ""))
        if surface not in set(NATIVE_CAPABILITY_SURFACES):
            errors.append(
                f"{_LABEL} surface finding names a surface that does not exist in this repository: "
                f"{row.get('surface')!r}; the vocabulary is {list(NATIVE_CAPABILITY_SURFACES)}"
            )
            continue
        if surface in seen:
            errors.append(f"{_LABEL} answers for surface {surface} more than once")
        seen.append(surface)
        errors.extend(_state_errors(surface, row))
    ordered = [surface for surface in NATIVE_CAPABILITY_SURFACES if surface in set(seen)]
    if sorted(seen) == sorted(set(seen)) and seen != ordered:
        errors.append(f"{_LABEL} surfaces must be listed in the declared surface order")
    expected = [str(item) for item in _string_list(gate.get("expected_surfaces"))]
    unanswered = [surface for surface in expected if surface not in set(seen)]
    if unanswered:
        described = [f"{surface} ({blueprint_surface_anchor(surface)})" for surface in unanswered]
        errors.append(
            f"{_LABEL} does not answer for every expected surface: {described}; a surface is present, "
            "absent, or exempt with a reason, and never unmentioned"
        )
    derived = list(quality_gate_unmet_surfaces(gate))
    if gate.get("unmet_surfaces") != derived:
        errors.append(
            f"{_LABEL} unmet_surfaces must be derived from the findings: {derived}, "
            f"not {gate.get('unmet_surfaces')!r}"
        )
    return errors


def _state_errors(surface: str, row: Mapping[str, Any]) -> list[str]:
    """One row's state and the reason rule that goes with it, both directions."""
    state = row.get("state")
    if state not in SURFACE_STATES:
        return [
            f"{_LABEL} surface {surface} has an unsupported state: {state!r}; "
            f"one of {list(SURFACE_STATES)} is required"
        ]
    reason = row.get("reason")
    if not isinstance(reason, str):
        return [f"{_LABEL} surface {surface} reason must be a string"]
    text = " ".join(reason.split())
    if state != "exempt":
        if text:
            return [
                f"{_LABEL} surface {surface} is marked {state} and must not carry a reason; a reason "
                "records why an exemption is one"
            ]
        return []
    if not text:
        return [
            f"{_LABEL} surface {surface} is exempt and must record why it does not apply to this "
            "capability; an exemption without a reason is an omission"
        ]
    errors: list[str] = []
    if len(text) < _MIN_REASON_CHARS:
        errors.append(
            f"{_LABEL} surface {surface} exemption reason must be at least {_MIN_REASON_CHARS} "
            "characters; an exemption is an argument, not a shrug"
        )
    if len(text) > _MAX_REASON_CHARS:
        errors.append(
            f"{_LABEL} surface {surface} exemption reason must be at most {_MAX_REASON_CHARS} characters"
        )
    if is_unsafe_metadata_line(text):
        errors.append(
            f"{_LABEL} surface {surface} exemption reason must be one bounded metadata line without "
            "secrets, links, or paths"
        )
    return errors


def _generated_guidance_errors(block: Any) -> list[str]:
    """AC2: reported from the repository's own checks, and never stapled on."""
    if not isinstance(block, Mapping):
        return [f"{_LABEL} generated_guidance must be an object"]
    errors: list[str] = []
    unknown_keys = sorted({str(key) for key in block} - set(GENERATED_GUIDANCE_KEYS))
    if unknown_keys:
        errors.append(f"{_LABEL} generated_guidance has unsupported keys: {unknown_keys}")
    absent_keys = sorted(set(GENERATED_GUIDANCE_KEYS) - {str(key) for key in block})
    if absent_keys:
        errors.append(f"{_LABEL} generated_guidance is missing keys: {absent_keys}")
    if not isinstance(block.get("catalog_validation_ok"), bool):
        errors.append(f"{_LABEL} generated_guidance catalog_validation_ok must be a boolean")
    for field in ("catalog_validation_errors", "checked", "stale_artifacts"):
        value = block.get(field)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            errors.append(f"{_LABEL} generated_guidance {field} must be a list of strings")
    if errors:
        return errors
    if block.get("catalog_validation_ok") is True and block.get("catalog_validation_errors"):
        errors.append(
            f"{_LABEL} generated_guidance cannot report a passing catalog validation alongside "
            f"{len(block['catalog_validation_errors'])} validation error(s)"
        )
    unchecked = sorted(set(block["stale_artifacts"]) - set(block["checked"]))
    if unchecked:
        errors.append(f"{_LABEL} generated_guidance reports stale artifacts it did not check: {unchecked}")
    reproducible = block.get("reproducible")
    if not isinstance(reproducible, bool):
        errors.append(f"{_LABEL} generated_guidance reproducible must be a boolean")
    elif reproducible != generated_guidance_reproduces(block):
        errors.append(
            f"{_LABEL} generated_guidance reproducible must be derived from the checks it reports, "
            f"not {reproducible!r}"
        )
    return errors


def _verdict_errors(gate: Mapping[str, Any]) -> list[str]:
    """AC3, and the rule that makes the verdict a conclusion rather than a claim."""
    errors: list[str] = []
    verdict = gate.get("verdict")
    if isinstance(verdict, str) and verdict.strip().casefold() in REFUSED_QUALITY_GATE_VERDICTS:
        errors.append(
            f"{_LABEL} verdict may not claim the capability was run, reviewed, or released: {verdict!r}; "
            f"a gate reports structure, and one of {list(QUALITY_GATE_VERDICTS)} is required"
        )
    elif verdict not in QUALITY_GATE_VERDICTS:
        errors.append(f"{_LABEL} verdict is unsupported: {verdict!r}")
    derived = derive_quality_gate_verdict(gate)
    if verdict != derived:
        errors.append(
            f"{_LABEL} verdict must be derived from the findings; these findings are {derived}, "
            f"not {verdict!r}"
        )
    claim = gate.get("verdict_claim")
    expected_claim = QUALITY_GATE_VERDICT_CLAIMS.get(derived, "")
    if claim != expected_claim:
        errors.append(
            f"{_LABEL} verdict_claim must be the one sentence the {derived} verdict is described with"
        )
    return errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finding_rows(surfaces: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize authored rows into declared surface order, or refuse."""
    if isinstance(surfaces, (str, bytes, bytearray)) or not isinstance(surfaces, Sequence):
        raise NativeCapabilityQualityGateError(f"{_LABEL} surfaces must be a list of objects")
    if not all(isinstance(row, Mapping) for row in surfaces):
        raise NativeCapabilityQualityGateError(f"{_LABEL} surfaces must be a list of objects")
    seen: dict[str, dict[str, Any]] = {}
    for row in surfaces:
        unknown_keys = sorted({str(key) for key in row} - set(SURFACE_FINDING_KEYS))
        if unknown_keys:
            raise NativeCapabilityQualityGateError(
                f"{_LABEL} surface finding has unsupported keys: {unknown_keys}"
            )
        surface = str(row.get("surface", ""))
        if surface not in set(NATIVE_CAPABILITY_SURFACES):
            raise NativeCapabilityQualityGateError(
                f"{_LABEL} surface finding names a surface that does not exist in this repository: "
                f"{row.get('surface')!r}; the vocabulary is {list(NATIVE_CAPABILITY_SURFACES)}"
            )
        if surface in seen:
            raise NativeCapabilityQualityGateError(f"{_LABEL} answers for surface {surface} more than once")
        state = row.get("state")
        if state not in SURFACE_STATES:
            raise NativeCapabilityQualityGateError(
                f"{_LABEL} surface {surface} has an unsupported state: {state!r}; "
                f"one of {list(SURFACE_STATES)} is required"
            )
        built = {
            "surface": surface,
            "state": str(state),
            "reason": " ".join(str(row.get("reason", "") or "").split()),
        }
        state_errors = _state_errors(surface, built)
        if state_errors:
            raise NativeCapabilityQualityGateError("; ".join(state_errors))
        seen[surface] = built
    return [seen[surface] for surface in NATIVE_CAPABILITY_SURFACES if surface in seen]


def _surface_is_answered(row: Mapping[str, Any] | None) -> bool:
    """Present, or exempt with a reason. Nothing else counts as answered.

    The reason is read here rather than only in the key checks so that a
    hand-written payload carrying a reasonless exemption cannot derive `pass`.
    """
    if not isinstance(row, Mapping):
        return False
    state = str(row.get("state", ""))
    if state == "present":
        return True
    reason = row.get("reason")
    return state == "exempt" and isinstance(reason, str) and bool(reason.strip())


def _mapping_rows(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return []
    return [item for item in value if isinstance(item, str) and item]
