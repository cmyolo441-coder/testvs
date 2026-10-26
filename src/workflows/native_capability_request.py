"""Asking for a capability, not for somebody else's package (issue #789).

What was missing
----------------

A person shows Hermes a feature they liked in another agent tool and says
"study this and make Hermes capable of the useful part". Everything OMH had for
that request answered a different question. `source_finder` records where the
material came from. `capability_inspiration_snapshot/v1` (#790) freezes what was
read. `native_capability_blueprint/v1` (#791) describes how the capability
should behave once somebody decides to build it. None of them holds the request
itself, and so nothing separated the two things the request confuses:

    the behavior the person wants Hermes to have
    the implementation they happened to see it in

Collapsed into one field, that request reads as "adopt this plugin". Kept apart,
it reads as "here is an outcome, here is what OMH already does, here is the gap".

Three fields, never one
-----------------------

`observed_reference_behavior`, `desired_user_outcome`, and
`missing_native_behavior` are three required keys, three separate digest inputs,
and refused when any two carry the same text. The separation is the feature: a
request whose observed behavior and desired outcome are the same sentence has
not distinguished the reference implementation from the outcome, and merging
them is how "make Hermes do this" quietly becomes "make Hermes run that".

Coverage is cited, not asserted
-------------------------------

`current_coverage` names real OMH capability ids and gives each one a verdict.
The id space is re-derived from this repository --
`native_capability_coverage_vocabulary()` joins `installable_skill_names()` (the
same id space `awesome_hermes_plugin_outcome_matrix/v1` fills its
`native_capability_ids` with) and the capability-family ids from
`src/capabilities/families.py`. A coverage answer that names something OMH does
not ship is a validation error rather than a plausible sentence, which is what
makes "what does OMH already handle?" checkable by a reader.

A request that answers `covered` for every capability it names contradicts its
own `missing_native_behavior`, and is refused for that reason.

Installation is never the answer
--------------------------------

#789 AC2 is a product boundary, so it is enforced twice and from two directions.
`INSTALLATION_RESOLUTIONS` names the shapes an adopt-the-package answer arrives
in and refuses them by name -- before the generic unsupported-value check,
because "unsupported value" is the wrong sentence for an answer that is
perfectly well-formed and simply not this product's. `resolution_summary` is
then scanned for an installation directive or a package specifier, because a
resolution can carry the same answer as prose under an allowed action.

The scan is deliberately scoped to the resolution text alone. "Never require
installing the source extension" is exactly the right sentence in
`safety_constraints` and exactly the wrong one in the field that says what OMH
should do about it.

OMH fetches nothing
-------------------

A feature URL is a reference the caller observed and supplied. It is recorded in
the `capability_inspiration_snapshot/v1` the request cites, never on the request
itself, and this module cannot open it: it imports nothing that can reach a
network and calls nothing that can. `url` is additionally in
`RAW_OR_HIDDEN_KEYS`, so a request that tries to carry one is refused by key
name. `claim_boundary` says all of this on every payload.

Evidence comes from #790 and nowhere else. `inspiration_citation` is a
`capability_inspiration_citation/v1` checked with that module's own validator,
and the citation must be frozen for the capability the request is about. A
second evidence record would mean two answers to "what was this based on".

The brief is executor-neutral by construction
---------------------------------------------

`build_native_capability_request_brief` refuses any request that is not
`accepted`, and produces a payload whose only owner-dependent value is `owner`.
`brief_digest` excludes `owner` and `prepared_at`, so the same accepted request
produces the same number for Codex, Claude Code, a Hermes runtime handoff, and a
generic executor profile -- executor neutrality as an equality a test can assert
rather than a promise the prose makes.

Accepting is not building. The brief's `claim_status` is
`prepared_not_observed`, its `not_observed` list names implementation,
verification, review, CI, and merge, and `IMPLEMENTATION_CLAIM_KEYS` -- reused
from #791 rather than restated, so the two families cannot drift -- refuses any
key shaped to say otherwise. That reuse is also why the field is `claim_status`
rather than `status`: a bare `status` key is one of the shapes that set refuses,
and being refused by it is the correct outcome rather than an inconvenience.

Determinism
-----------

Nothing here reads a clock. `prepared_at` is a parameter on both payloads and is
excluded from both digests; a wall clock inside a compared value turns an
equality check into a race this repository has already lost Windows CI time to.
Closed vocabularies and coverage entries are normalized into a canonical order
before the digest is taken, so two authors who supply the same request in a
different order produce the same digest.

`review_state` is excluded from `request_digest` on purpose: the digest seals
what was asked, not where the ask has got to. A request keeps its `request_id`
from prepared through accepted, which is what lets a brief point back at it.

What reads these today
----------------------

Stated because the vocabulary is wider than the wiring. No production surface
mints a request yet; `tests/test_native_capability_request.py` is the only
caller. `blueprint_ref` and `native_capability_request_blueprint_gap` are the
seams to #791: a request references a blueprint by digest rather than restating
one, and the gap accessor reports which required surfaces a blueprint would
still have to add.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any, Final

from ..capabilities.families import capability_family_projection
from ..coding.executors import EXECUTOR_PROFILES
from ..quality.capability_inspiration_snapshot import (
    CAPABILITY_INSPIRATION_CITATION_KEYS,
    validate_capability_inspiration_citation,
)
from ..skills.catalog import installable_skill_names
from ..system.append_only_store import RAW_OR_HIDDEN_KEYS
from .native_capability_blueprint import (
    IMPLEMENTATION_CLAIM_KEYS,
    NATIVE_CAPABILITY_SURFACES,
    SOURCE_HOST_MECHANICS,
    missing_required_surfaces,
    unknown_surfaces,
)


NATIVE_CAPABILITY_REQUEST_SCHEMA_VERSION: Final[str] = "native_capability_request/v1"
NATIVE_CAPABILITY_REQUEST_BRIEF_SCHEMA_VERSION: Final[str] = "native_capability_request_brief/v1"

REQUEST_PRIVACY: Final[str] = "metadata_only"

# The repository's canonical prepared state, under the key name
# `awesome_hermes_plugin_outcome_matrix/v1` already uses for it. A bare
# `status` key is in `IMPLEMENTATION_CLAIM_KEYS` and refused there for good
# reason -- it is the shape "it ran" arrives in -- and `claim_status` says
# what this value actually is: the boundary on a claim, not the outcome of a
# run.
BRIEF_CLAIM_STATUS: Final[str] = "prepared_not_observed"

REQUEST_CLAIM_BOUNDARY: Final[str] = (
    "A native capability request is an OMH-local record of an outcome somebody asked for. The referenced "
    "feature was observed and supplied by the caller; OMH performed no fetch, download, or network call to "
    "produce this and cannot perform one. Recording, reviewing, or accepting a request is not implementation, "
    "verification, review, CI, merge evidence, or a claim that the capability exists, and the resolution is "
    "always OMH-native work rather than installing, enabling, or requiring the referenced extension."
)

BRIEF_CLAIM_BOUNDARY: Final[str] = (
    "An executor-neutral brief is prepared work description only. It names the selected coding owner without "
    "dispatching to one, and an accepted request behind it is a reviewed decision rather than a built "
    "capability: nothing here is implementation, verification, review, CI, or merge evidence, and none of it "
    "means the capability exists, is installed, or is available."
)

# Every claim neither payload makes, listed so a consumer can check the list
# instead of inferring the boundary from prose.
NATIVE_CAPABILITY_REQUEST_NOT_OBSERVED: Final[tuple[str, ...]] = (
    "reference_feature_fetch",
    "reference_feature_execution",
    "reference_implementation_review",
    "native_capability_implementation",
    "verification_execution",
    "review",
    "ci",
    "merge",
)

# What OMH already does about one named capability. Three verdicts and no
# "unknown": a coverage answer nobody could give is a request that has not
# answered "what does OMH already handle", which is half of AC1.
COVERAGE_STATES: Final[tuple[str, ...]] = ("covered", "partially_covered", "not_covered")

# The verdicts that leave something for OMH to build. A request every one of
# whose coverage answers falls outside this set says the outcome already exists.
COVERAGE_GAP_STATES: Final[tuple[str, ...]] = ("partially_covered", "not_covered")

# Where a request has got to. None of these means the capability is available;
# `accepted` means a reviewer agreed OMH should build it.
REVIEW_STATES: Final[tuple[str, ...]] = ("prepared", "reviewed", "accepted", "rejected")

# The OMH-native answers. Planning first, because "study this feature" usually
# resolves to a design artifact rather than straight to code.
OMH_PLANNING_RESOLUTIONS: Final[tuple[str, ...]] = (
    "prepare_native_capability_blueprint",
    "prepare_product_proposal",
)
OMH_CODING_RESOLUTIONS: Final[tuple[str, ...]] = (
    "prepare_executor_neutral_brief",
    "prepare_coding_handoff",
)
NATIVE_CAPABILITY_RESOLUTIONS: Final[tuple[str, ...]] = (
    *OMH_PLANNING_RESOLUTIONS,
    *OMH_CODING_RESOLUTIONS,
)

RESOLUTION_LANES: Final[tuple[str, ...]] = ("omh_planning", "omh_coding")

# The answers #789 AC2 exists to refuse: resolve the user's need by adopting
# somebody else's package. Every member is well-formed and simply not this
# product's, so they are refused by name rather than as unsupported values.
INSTALLATION_RESOLUTIONS: Final[tuple[str, ...]] = (
    "enable_host_setting",
    "install_host_extension",
    "install_host_plugin",
    "install_mcp_server",
    "install_source_package",
    "subscribe_to_host_marketplace",
    "vendor_source_package",
)

# What the executor-neutral brief refuses on every payload, regardless of who
# authored the request it came from.
BRIEF_NON_GOALS: Final[tuple[str, ...]] = (
    "Installing, enabling, vendoring, or requiring the referenced extension, plugin, package, or MCP server.",
    "Operating an extension aggregator, marketplace, connector manager, or plugin catalog.",
    "Reproducing the reference implementation's internals instead of the outcome a person asked for.",
    "Claiming the capability exists, is available, or has been implemented.",
)

# Every coding owner this repository treats as first-class. Taken from the
# executor profiles rather than restated, so "executor-neutral" means neutral
# across the same set the rest of OMH selects from.
NATIVE_CAPABILITY_BRIEF_OWNERS: Final[tuple[str, ...]] = EXECUTOR_PROFILES

NATIVE_CAPABILITY_REQUEST_KEYS: Final[tuple[str, ...]] = (
    "affected_surfaces",
    "blueprint_ref",
    "capability_id",
    "claim_boundary",
    "current_coverage",
    "desired_user_outcome",
    "example_requests",
    "inspiration_citation",
    "missing_native_behavior",
    "not_observed",
    "observed_reference_behavior",
    "observed_source_mechanics",
    "prepared_at",
    "privacy",
    "request_digest",
    "request_id",
    "resolution_action",
    "resolution_summary",
    "review_state",
    "safety_constraints",
    "schema_version",
)

NATIVE_CAPABILITY_COVERAGE_KEYS: Final[tuple[str, ...]] = ("capability_id", "coverage", "note")

# What `request_digest` seals: the ask, and nothing else. `review_state` is
# excluded so a request keeps one id from prepared through accepted, and
# `prepared_at` is excluded because a clock inside a compared value is a race.
REQUEST_DIGEST_KEYS: Final[tuple[str, ...]] = (
    "affected_surfaces",
    "blueprint_ref",
    "capability_id",
    "current_coverage",
    "desired_user_outcome",
    "example_requests",
    "inspiration_citation",
    "missing_native_behavior",
    "observed_reference_behavior",
    "observed_source_mechanics",
    "resolution_action",
    "resolution_summary",
    "safety_constraints",
)

NATIVE_CAPABILITY_REQUEST_BRIEF_KEYS: Final[tuple[str, ...]] = (
    "acceptance_criteria",
    "affected_surfaces",
    "brief_digest",
    "capability_id",
    "claim_boundary",
    "claim_status",
    "current_coverage",
    "desired_user_outcome",
    "inspiration_citation",
    "missing_native_behavior",
    "non_goals",
    "not_observed",
    "observed_reference_behavior",
    "owner",
    "prepared_at",
    "privacy",
    "request_digest",
    "request_id",
    "resolution_action",
    "safety_constraints",
    "schema_version",
)

# `owner` and `prepared_at` are excluded, which is #789 AC3 as an equality: the
# same accepted request describes the same work whoever is asked to do it.
BRIEF_DIGEST_KEYS: Final[tuple[str, ...]] = (
    "acceptance_criteria",
    "affected_surfaces",
    "capability_id",
    "current_coverage",
    "desired_user_outcome",
    "inspiration_citation",
    "missing_native_behavior",
    "non_goals",
    "observed_reference_behavior",
    "request_digest",
    "request_id",
    "resolution_action",
    "safety_constraints",
)

_LABEL: Final[str] = "native_capability_request"
_BRIEF_LABEL: Final[str] = "native_capability_request_brief"

# The canonical skill name the requested capability would use: the identifier a
# catalog entry, a tap directory, and a routing key would all share.
_CAPABILITY_ID: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}$")

# A blueprint is referenced by its `blueprint_digest`, which is a sha256.
_DIGEST: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

# An example ask is a sentence a person types, not a command they run.
# `docs/DIRECTION.md` rule 1: chat users stay free of command knowledge, so an
# example that teaches one is the wrong example.
_COMMAND_SHAPED: Final[re.Pattern[str]] = re.compile(r"^\s*(?:omh\b|/|-)|--")

# Installation phrased as prose under an allowed action. Checked against the
# resolution text only -- the same words belong in `safety_constraints`, where
# they state the boundary rather than propose crossing it.
_INSTALLATION_DIRECTIVES: Final[tuple[str, ...]] = (
    "add a plugin",
    "add an extension",
    "add an mcp server",
    "add the extension",
    "add the mcp server",
    "add the plugin",
    "brew install",
    "download the extension",
    "download the plugin",
    "enable the extension",
    "enable the plugin",
    "install a",
    "install the",
    "install this",
    "installing the",
    "npm install",
    "npx ",
    "pip install",
    "pipx ",
    "register the mcp server",
    "uv add",
    "vendor the package",
)

# A scoped package specifier, the way a resolution names the thing it wants
# adopted when it does not use a verb at all.
_PACKAGE_SPEC: Final[re.Pattern[str]] = re.compile(r"@[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]+")

# Free-text bounds. A request is authored product prose, and these keep it from
# becoming somewhere a transcript can be parked.
_MAX_TEXT_LENGTH: Final[int] = 240
_MIN_EXAMPLE_REQUESTS: Final[int] = 2
_MAX_EXAMPLE_REQUESTS: Final[int] = 8
_MIN_COVERAGE_ENTRIES: Final[int] = 1
_MAX_COVERAGE_ENTRIES: Final[int] = 12
_MIN_SAFETY_CONSTRAINTS: Final[int] = 1
_MAX_SAFETY_CONSTRAINTS: Final[int] = 8


class NativeCapabilityRequestError(ValueError):
    """Raised when an ask cannot become a native capability request."""


# ---------------------------------------------------------------------------
# Capability vocabulary
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def native_capability_coverage_vocabulary() -> tuple[str, ...]:
    """Every OMH capability id a coverage answer may name, sorted.

    Two real id spaces joined: the installable skills this repository ships and
    the capability-family ids a user-facing answer groups them under. Both are
    re-derived here rather than copied, so a coverage answer cannot cite a
    capability that stopped existing.
    """
    projection = capability_family_projection()
    families = projection.get("families", [])
    family_ids = [
        str(family.get("id", ""))
        for family in families
        if isinstance(family, Mapping) and str(family.get("id", ""))
    ]
    return tuple(sorted({*installable_skill_names(), *family_ids}))


def unknown_coverage_capabilities(capability_ids: Iterable[str]) -> tuple[str, ...]:
    """Every named capability id that is not an OMH capability, sorted."""
    vocabulary = set(native_capability_coverage_vocabulary())
    return tuple(sorted({str(item) for item in capability_ids} - vocabulary))


def resolution_lane(resolution_action: str) -> str:
    """`omh_planning`, `omh_coding`, or an empty string for anything else."""
    action = str(resolution_action)
    if action in OMH_PLANNING_RESOLUTIONS:
        return "omh_planning"
    if action in OMH_CODING_RESOLUTIONS:
        return "omh_coding"
    return ""


# ---------------------------------------------------------------------------
# Build and validate the request
# ---------------------------------------------------------------------------


def build_native_capability_request(
    *,
    capability_id: str,
    observed_reference_behavior: str,
    desired_user_outcome: str,
    missing_native_behavior: str,
    example_requests: Sequence[str],
    current_coverage: Sequence[Mapping[str, Any]],
    inspiration_citation: Mapping[str, Any],
    resolution_action: str,
    resolution_summary: str,
    safety_constraints: Sequence[str],
    affected_surfaces: Sequence[str] = (),
    observed_source_mechanics: Sequence[str] = (),
    blueprint_ref: str = "",
    review_state: str = "prepared",
    prepared_at: str = "",
) -> dict[str, Any]:
    """Mint one native capability request, or refuse.

    Closed-vocabulary lists are normalized into their declared order and
    coverage entries are sorted by capability id, so two callers who supply the
    same request in a different order produce the same `request_digest`.
    Authored prose keeps the order it was written in.

    Nothing here reads a clock, opens a file, or reaches a network.
    """
    request: dict[str, Any] = {
        "schema_version": NATIVE_CAPABILITY_REQUEST_SCHEMA_VERSION,
        "request_id": "",
        "capability_id": str(capability_id).strip().casefold(),
        "observed_reference_behavior": str(observed_reference_behavior).strip(),
        "desired_user_outcome": str(desired_user_outcome).strip(),
        "missing_native_behavior": str(missing_native_behavior).strip(),
        "example_requests": _authored_list(example_requests),
        "current_coverage": _normalized_coverage(current_coverage),
        "inspiration_citation": _copied_citation(inspiration_citation),
        "resolution_action": str(resolution_action).strip(),
        "resolution_summary": str(resolution_summary).strip(),
        "safety_constraints": _authored_list(safety_constraints),
        "affected_surfaces": _vocabulary_list(affected_surfaces, NATIVE_CAPABILITY_SURFACES),
        "observed_source_mechanics": _vocabulary_list(observed_source_mechanics, SOURCE_HOST_MECHANICS),
        "blueprint_ref": str(blueprint_ref).strip().casefold(),
        "review_state": str(review_state).strip(),
        "prepared_at": str(prepared_at).strip(),
        "privacy": REQUEST_PRIVACY,
        "claim_boundary": REQUEST_CLAIM_BOUNDARY,
        "not_observed": list(NATIVE_CAPABILITY_REQUEST_NOT_OBSERVED),
        "request_digest": "",
    }
    request["request_digest"] = request_digest_of(request)
    request["request_id"] = f"native-capability-request-{request['request_digest'][:16]}"
    errors = validate_native_capability_request(request)
    if errors:
        raise NativeCapabilityRequestError(errors[0])
    return request


def request_digest_of(request: Mapping[str, Any]) -> str:
    """A sha256 over the ask, covering no clock, no review state, and no id."""
    identity = {key: request.get(key) for key in REQUEST_DIGEST_KEYS}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def validate_native_capability_request(request: Mapping[str, Any]) -> list[str]:
    """Every reason one payload is not a `native_capability_request/v1`."""
    if not isinstance(request, Mapping):
        return [f"{_LABEL} must be an object"]
    errors = _envelope_errors(request, NATIVE_CAPABILITY_REQUEST_KEYS, _LABEL)
    if request.get("schema_version") != NATIVE_CAPABILITY_REQUEST_SCHEMA_VERSION:
        errors.append(f"{_LABEL} schema_version must be {NATIVE_CAPABILITY_REQUEST_SCHEMA_VERSION}")
    if request.get("privacy") != REQUEST_PRIVACY:
        errors.append(f"{_LABEL} privacy must be {REQUEST_PRIVACY}")
    if request.get("claim_boundary") != REQUEST_CLAIM_BOUNDARY:
        errors.append(
            f"{_LABEL} claim_boundary must state that the reference was supplied rather than fetched and that "
            "recording a request is not implementation"
        )
    errors.extend(_not_observed_errors(request, _LABEL))
    if not isinstance(request.get("prepared_at"), str):
        errors.append(f"{_LABEL} prepared_at must be a string")
    errors.extend(_separation_errors(request))
    errors.extend(_evidence_errors(request))
    errors.extend(_coverage_errors(request))
    errors.extend(_resolution_errors(request))
    errors.extend(_surface_errors(request))
    errors.extend(_review_errors(request))
    errors.extend(_request_digest_errors(request))
    return errors


def native_capability_request_offered_actions(request: Mapping[str, Any]) -> tuple[str, ...]:
    """The OMH actions a response may offer, chosen action first.

    #789 AC2 in its offered form. Every member is an OMH planning or coding
    action because the tuple is built from `NATIVE_CAPABILITY_RESOLUTIONS`;
    there is no branch through which an installation action could reach it.
    """
    errors = validate_native_capability_request(request)
    if errors:
        raise NativeCapabilityRequestError(errors[0])
    chosen = str(request.get("resolution_action", ""))
    return (chosen, *(action for action in NATIVE_CAPABILITY_RESOLUTIONS if action != chosen))


def native_capability_request_blueprint_gap(request: Mapping[str, Any]) -> tuple[str, ...]:
    """Required blueprint surfaces this request has not named yet.

    The seam to `native_capability_blueprint/v1`. A request is earlier than a
    blueprint and is not expected to have identified every surface, so the
    unnamed ones are reported rather than refused: the answer is what a
    blueprint would still have to add, not a defect in the request.
    """
    errors = validate_native_capability_request(request)
    if errors:
        raise NativeCapabilityRequestError(errors[0])
    return missing_required_surfaces(request.get("affected_surfaces", []))


# ---------------------------------------------------------------------------
# Build and validate the executor-neutral brief
# ---------------------------------------------------------------------------


def build_native_capability_request_brief(
    request: Mapping[str, Any],
    *,
    owner: str,
    prepared_at: str = "",
) -> dict[str, Any]:
    """Populate an executor-neutral brief from one accepted request.

    Only an `accepted` request produces a brief. A prepared or reviewed request
    has not been agreed to, and a rejected one has been declined; minting work
    descriptions from either would make review decorative.

    `owner` is the selected coding owner and is the only value on the payload
    that depends on it. `brief_digest` excludes it.
    """
    errors = validate_native_capability_request(request)
    if errors:
        raise NativeCapabilityRequestError(errors[0])
    if request.get("review_state") != "accepted":
        raise NativeCapabilityRequestError(
            f"{_BRIEF_LABEL} needs an accepted request; review_state is "
            f"{str(request.get('review_state'))!r}, and accepting is what authorizes a brief"
        )
    safe_owner = str(owner).strip().casefold()
    if safe_owner not in NATIVE_CAPABILITY_BRIEF_OWNERS:
        raise NativeCapabilityRequestError(
            f"{_BRIEF_LABEL} owner is unsupported: {owner!r}; allowed: {list(NATIVE_CAPABILITY_BRIEF_OWNERS)}"
        )
    brief: dict[str, Any] = {
        "schema_version": NATIVE_CAPABILITY_REQUEST_BRIEF_SCHEMA_VERSION,
        "owner": safe_owner,
        "claim_status": BRIEF_CLAIM_STATUS,
        "request_id": str(request["request_id"]),
        "request_digest": str(request["request_digest"]),
        "capability_id": str(request["capability_id"]),
        "observed_reference_behavior": str(request["observed_reference_behavior"]),
        "desired_user_outcome": str(request["desired_user_outcome"]),
        "missing_native_behavior": str(request["missing_native_behavior"]),
        "current_coverage": _copied_coverage(request["current_coverage"]),
        "inspiration_citation": _copied_citation(request["inspiration_citation"]),
        "affected_surfaces": list(request["affected_surfaces"]),
        "safety_constraints": list(request["safety_constraints"]),
        "acceptance_criteria": _acceptance_criteria(request),
        "non_goals": list(BRIEF_NON_GOALS),
        "resolution_action": str(request["resolution_action"]),
        "prepared_at": str(prepared_at).strip(),
        "privacy": REQUEST_PRIVACY,
        "claim_boundary": BRIEF_CLAIM_BOUNDARY,
        "not_observed": list(NATIVE_CAPABILITY_REQUEST_NOT_OBSERVED),
        "brief_digest": "",
    }
    brief["brief_digest"] = brief_digest_of(brief)
    brief_errors = validate_native_capability_request_brief(brief)
    if brief_errors:
        raise NativeCapabilityRequestError(brief_errors[0])
    return brief


def brief_digest_of(brief: Mapping[str, Any]) -> str:
    """A sha256 over the work, covering no owner, no clock, and no digest."""
    identity = {key: brief.get(key) for key in BRIEF_DIGEST_KEYS}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def validate_native_capability_request_brief(brief: Mapping[str, Any]) -> list[str]:
    """Every reason one payload is not a `native_capability_request_brief/v1`."""
    if not isinstance(brief, Mapping):
        return [f"{_BRIEF_LABEL} must be an object"]
    errors = _envelope_errors(brief, NATIVE_CAPABILITY_REQUEST_BRIEF_KEYS, _BRIEF_LABEL)
    if brief.get("schema_version") != NATIVE_CAPABILITY_REQUEST_BRIEF_SCHEMA_VERSION:
        errors.append(f"{_BRIEF_LABEL} schema_version must be {NATIVE_CAPABILITY_REQUEST_BRIEF_SCHEMA_VERSION}")
    if brief.get("privacy") != REQUEST_PRIVACY:
        errors.append(f"{_BRIEF_LABEL} privacy must be {REQUEST_PRIVACY}")
    if brief.get("claim_status") != BRIEF_CLAIM_STATUS:
        errors.append(
            f"{_BRIEF_LABEL} claim_status must be {BRIEF_CLAIM_STATUS}; a brief describes work nobody "
            "has done yet"
        )
    if brief.get("claim_boundary") != BRIEF_CLAIM_BOUNDARY:
        errors.append(
            f"{_BRIEF_LABEL} claim_boundary must state that a named owner is not a dispatched one and that an "
            "accepted request is not a built capability"
        )
    errors.extend(_not_observed_errors(brief, _BRIEF_LABEL))
    if brief.get("owner") not in NATIVE_CAPABILITY_BRIEF_OWNERS:
        errors.append(
            f"{_BRIEF_LABEL} owner is unsupported: {brief.get('owner')!r}; allowed: "
            f"{list(NATIVE_CAPABILITY_BRIEF_OWNERS)}"
        )
    if brief.get("resolution_action") not in NATIVE_CAPABILITY_RESOLUTIONS:
        errors.append(
            f"{_BRIEF_LABEL} resolution_action must be an OMH planning or coding action: "
            f"{brief.get('resolution_action')!r}"
        )
    if list(brief.get("non_goals", [])) != list(BRIEF_NON_GOALS):
        errors.append(f"{_BRIEF_LABEL} non_goals must be {list(BRIEF_NON_GOALS)}")
    for key in ("request_id", "request_digest", "capability_id"):
        if not isinstance(brief.get(key), str) or not str(brief.get(key)).strip():
            errors.append(f"{_BRIEF_LABEL} {key} must be a non-empty string")
    if not isinstance(brief.get("prepared_at"), str):
        errors.append(f"{_BRIEF_LABEL} prepared_at must be a string")
    errors.extend(
        _text_list_errors(brief.get("acceptance_criteria"), "acceptance_criteria", _BRIEF_LABEL, minimum=2, maximum=32)
    )
    digest = brief.get("brief_digest")
    if not isinstance(digest, str) or not _DIGEST.match(digest):
        errors.append(f"{_BRIEF_LABEL} brief_digest must be a sha256 hex digest")
    elif digest != brief_digest_of(brief):
        errors.append(
            f"{_BRIEF_LABEL} brief_digest does not match the work it seals; the payload was edited after it "
            "was built"
        )
    return errors


# ---------------------------------------------------------------------------
# Validation parts
# ---------------------------------------------------------------------------


def _envelope_errors(payload: Mapping[str, Any], keys: tuple[str, ...], label: str) -> list[str]:
    """Closed key set, plus the two key shapes refused by name before it."""
    errors: list[str] = []
    claims_implementation = sorted(key for key in payload if str(key).lower() in IMPLEMENTATION_CLAIM_KEYS)
    if claims_implementation:
        errors.append(
            f"{label} must not carry implementation-claim keys: {claims_implementation}; a request and its "
            "brief describe a capability nobody has built and never report one that exists"
        )
    forbidden = sorted(key for key in payload if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if forbidden:
        errors.append(
            f"{label} must not carry raw or hidden keys: {forbidden}; a referenced feature is cited through "
            "its frozen snapshot, never as a link or a transcript on this payload"
        )
    unexpected = sorted(set(payload) - set(keys) - set(claims_implementation) - set(forbidden))
    if unexpected:
        errors.append(f"{label} has unsupported keys: {unexpected}")
    missing = sorted(set(keys) - set(payload))
    if missing:
        errors.append(f"{label} is missing keys: {missing}")
    return errors


def _not_observed_errors(payload: Mapping[str, Any], label: str) -> list[str]:
    declared = payload.get("not_observed")
    if not isinstance(declared, list) or list(declared) != list(NATIVE_CAPABILITY_REQUEST_NOT_OBSERVED):
        return [f"{label} not_observed must be {list(NATIVE_CAPABILITY_REQUEST_NOT_OBSERVED)}"]
    return []


def _separation_errors(request: Mapping[str, Any]) -> list[str]:
    """The three fields the issue exists to keep apart, kept apart."""
    capability_id = request.get("capability_id")
    errors: list[str] = []
    if not isinstance(capability_id, str) or not _CAPABILITY_ID.match(capability_id):
        errors.append(
            f"{_LABEL} capability_id must be the canonical skill name as a lowercase slug: {capability_id!r}"
        )
    fields = ("observed_reference_behavior", "desired_user_outcome", "missing_native_behavior")
    for field in fields:
        errors.extend(_text_errors(request.get(field), field, _LABEL))
    present = {field: _folded(request.get(field)) for field in fields if isinstance(request.get(field), str)}
    collapsed = sorted(
        {
            tuple(sorted((first, second)))
            for first in present
            for second in present
            if first != second and present[first] and present[first] == present[second]
        }
    )
    if collapsed:
        errors.append(
            f"{_LABEL} keeps the observed reference behavior, the desired outcome, and the missing native "
            f"behavior apart; these carry the same text: {[list(pair) for pair in collapsed]}"
        )
    errors.extend(
        _text_list_errors(
            request.get("example_requests"),
            "example_requests",
            _LABEL,
            minimum=_MIN_EXAMPLE_REQUESTS,
            maximum=_MAX_EXAMPLE_REQUESTS,
        )
    )
    if isinstance(request.get("example_requests"), list):
        commands = [
            example for example in request["example_requests"] if _COMMAND_SHAPED.search(str(example))
        ]
        if commands:
            errors.append(
                f"{_LABEL} example_requests must be natural-language asks, not commands: {commands}"
            )
    errors.extend(
        _text_list_errors(
            request.get("safety_constraints"),
            "safety_constraints",
            _LABEL,
            minimum=_MIN_SAFETY_CONSTRAINTS,
            maximum=_MAX_SAFETY_CONSTRAINTS,
        )
    )
    return errors


def _evidence_errors(request: Mapping[str, Any]) -> list[str]:
    """#789 AC1's first half: the request cites frozen evidence from #790."""
    citation = request.get("inspiration_citation")
    if not isinstance(citation, Mapping) or not citation:
        return [
            f"{_LABEL} inspiration_citation must be a capability_inspiration_citation/v1; a request with no "
            "cited snapshot has no evidence a reader can re-ask"
        ]
    citation_errors = validate_capability_inspiration_citation(citation)
    if citation_errors:
        return [f"{_LABEL} inspiration_citation is not a valid citation: {citation_errors[0]}"]
    if str(citation.get("capability_id", "")) != str(request.get("capability_id", "")):
        return [
            f"{_LABEL} inspiration_citation must be frozen for the capability being requested: cited "
            f"{str(citation.get('capability_id'))!r}, requested {str(request.get('capability_id'))!r}"
        ]
    return []


def _coverage_errors(request: Mapping[str, Any]) -> list[str]:
    """#789 AC1's second half: a coverage answer per named OMH capability."""
    coverage = request.get("current_coverage")
    if not isinstance(coverage, list) or not all(isinstance(entry, Mapping) for entry in coverage):
        return [f"{_LABEL} current_coverage must be a list of coverage answers"]
    errors: list[str] = []
    if len(coverage) < _MIN_COVERAGE_ENTRIES:
        errors.append(
            f"{_LABEL} current_coverage must answer for at least {_MIN_COVERAGE_ENTRIES} OMH capability; "
            "a request that never says what OMH already handles has not isolated the gap"
        )
    if len(coverage) > _MAX_COVERAGE_ENTRIES:
        errors.append(f"{_LABEL} current_coverage must name at most {_MAX_COVERAGE_ENTRIES} capabilities")
    named: list[str] = []
    for index, entry in enumerate(coverage):
        errors.extend(
            _envelope_errors(entry, NATIVE_CAPABILITY_COVERAGE_KEYS, f"{_LABEL} current_coverage[{index}]")
        )
        capability_id = entry.get("capability_id")
        if not isinstance(capability_id, str) or not capability_id.strip():
            errors.append(f"{_LABEL} current_coverage[{index}] capability_id must be a non-empty string")
            continue
        named.append(capability_id)
        if entry.get("coverage") not in COVERAGE_STATES:
            errors.append(
                f"{_LABEL} current_coverage[{index}] needs a coverage answer for {capability_id!r}: "
                f"{entry.get('coverage')!r} is not one of {list(COVERAGE_STATES)}"
            )
        errors.extend(
            _text_errors(entry.get("note"), f"current_coverage[{index}] note", _LABEL, allow_empty=True)
        )
    unknown = unknown_coverage_capabilities(named)
    if unknown:
        errors.append(
            f"{_LABEL} current_coverage names capabilities OMH does not ship: {list(unknown)}; a coverage "
            "answer cites an installable skill name or a capability-family id, never prose"
        )
    if len(set(named)) != len(named):
        errors.append(f"{_LABEL} current_coverage must answer once per capability")
    if named != sorted(named):
        errors.append(f"{_LABEL} current_coverage must be sorted by capability_id")
    verdicts = {str(entry.get("coverage")) for entry in coverage if isinstance(entry, Mapping)}
    if named and verdicts and not verdicts & set(COVERAGE_GAP_STATES):
        errors.append(
            f"{_LABEL} current_coverage says every named capability already covers the outcome, which "
            "contradicts missing_native_behavior; a request with no gap is not a request"
        )
    return errors


def _resolution_errors(request: Mapping[str, Any]) -> list[str]:
    """#789 AC2: OMH planning or coding, and never installing somebody's package."""
    action = request.get("resolution_action")
    errors: list[str] = []
    if action in INSTALLATION_RESOLUTIONS:
        errors.append(
            f"{_LABEL} resolution_action must not resolve the request by adopting a package: {action!r}; "
            "OMH answers a referenced feature with a native planning or coding action, and installing, "
            "enabling, vendoring, or requiring the source extension is never the resolution"
        )
    elif action not in NATIVE_CAPABILITY_RESOLUTIONS:
        errors.append(
            f"{_LABEL} resolution_action must be an OMH planning or coding action: {action!r}; allowed: "
            f"{list(NATIVE_CAPABILITY_RESOLUTIONS)}"
        )
    summary = request.get("resolution_summary")
    errors.extend(_text_errors(summary, "resolution_summary", _LABEL))
    if isinstance(summary, str):
        folded = _folded(summary)
        directives = [phrase.strip() for phrase in _INSTALLATION_DIRECTIVES if phrase in folded]
        packages = sorted(set(_PACKAGE_SPEC.findall(folded)))
        if directives or packages:
            named = [*directives, *packages]
            errors.append(
                f"{_LABEL} resolution_summary must not name an installable package or extension as the "
                f"resolution: {named}; the same need belongs here as the OMH planning or coding action that "
                "would produce the outcome natively"
            )
    return errors


def _surface_errors(request: Mapping[str, Any]) -> list[str]:
    """The seam to #791: surfaces come from that vocabulary or from nowhere."""
    surfaces = request.get("affected_surfaces")
    if not isinstance(surfaces, list) or not all(isinstance(surface, str) for surface in surfaces):
        return [f"{_LABEL} affected_surfaces must be a list of strings"]
    errors: list[str] = []
    if len(set(surfaces)) != len(surfaces):
        errors.append(f"{_LABEL} affected_surfaces must not repeat a surface")
    unknown = unknown_surfaces(surfaces)
    if unknown:
        errors.append(
            f"{_LABEL} affected_surfaces names surfaces that do not exist in this repository: {list(unknown)}; "
            f"the vocabulary is {list(NATIVE_CAPABILITY_SURFACES)}"
        )
    mechanics = request.get("observed_source_mechanics")
    if not isinstance(mechanics, list) or not all(isinstance(item, str) for item in mechanics):
        errors.append(f"{_LABEL} observed_source_mechanics must be a list of strings")
        return errors
    if len(set(mechanics)) != len(mechanics):
        errors.append(f"{_LABEL} observed_source_mechanics must not repeat a mechanic")
    unsupported = sorted({item for item in mechanics if item not in set(SOURCE_HOST_MECHANICS)})
    if unsupported:
        errors.append(
            f"{_LABEL} observed_source_mechanics has unsupported entries: {unsupported}; allowed: "
            f"{list(SOURCE_HOST_MECHANICS)}"
        )
    return errors


def _review_errors(request: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if request.get("review_state") not in REVIEW_STATES:
        errors.append(
            f"{_LABEL} review_state is unsupported: {request.get('review_state')!r}; allowed: "
            f"{list(REVIEW_STATES)}, none of which means the capability is available"
        )
    blueprint_ref = request.get("blueprint_ref")
    if not isinstance(blueprint_ref, str):
        errors.append(f"{_LABEL} blueprint_ref must be a string")
    elif blueprint_ref and not _DIGEST.match(blueprint_ref):
        errors.append(
            f"{_LABEL} blueprint_ref must be a native_capability_blueprint/v1 blueprint_digest: "
            f"{blueprint_ref!r}"
        )
    return errors


def _request_digest_errors(request: Mapping[str, Any]) -> list[str]:
    digest = request.get("request_digest")
    if not isinstance(digest, str) or not _DIGEST.match(digest):
        return [f"{_LABEL} request_digest must be a sha256 hex digest"]
    if digest != request_digest_of(request):
        return [
            f"{_LABEL} request_digest does not match the ask it seals; the payload was edited after it was "
            "minted"
        ]
    if request.get("request_id") != f"native-capability-request-{digest[:16]}":
        return [f"{_LABEL} request_id does not match its request_digest"]
    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _acceptance_criteria(request: Mapping[str, Any]) -> list[str]:
    """What "done" means for this brief, templated from the request itself.

    Bounded by `_MAX_COVERAGE_ENTRIES` plus the three fixed lines, so a brief
    cannot grow an unbounded criteria list from a bounded request.
    """
    gaps = [
        entry
        for entry in request["current_coverage"]
        if str(entry.get("coverage")) in COVERAGE_GAP_STATES
    ]
    return [
        f"Hermes produces this outcome natively: {request['desired_user_outcome']}",
        f"OMH implements the missing native behavior: {request['missing_native_behavior']}",
        *[
            f"Coverage gap closed for {entry['capability_id']} (currently {entry['coverage']})"
            for entry in gaps
        ],
        "No source-host mechanic becomes an OMH runtime requirement, and no referenced package is installed, "
        "vendored, or required.",
    ]


def _normalized_coverage(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Coverage answers in canonical order, so supply order cannot move the digest."""
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise NativeCapabilityRequestError(f"{_LABEL} current_coverage entry must be an object")
        unsupported = sorted(set(entry) - set(NATIVE_CAPABILITY_COVERAGE_KEYS))
        if unsupported:
            raise NativeCapabilityRequestError(
                f"{_LABEL} current_coverage entry has unsupported keys: {unsupported}"
            )
        normalized.append(
            {
                "capability_id": str(entry.get("capability_id", "")).strip().casefold(),
                "coverage": str(entry.get("coverage", "")).strip(),
                "note": str(entry.get("note", "")).strip(),
            }
        )
    normalized.sort(key=lambda entry: str(entry["capability_id"]))
    return normalized


def _copied_coverage(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: entry.get(key) for key in NATIVE_CAPABILITY_COVERAGE_KEYS} for entry in entries]


def _copied_citation(citation: Mapping[str, Any]) -> dict[str, Any]:
    """The cited citation, copied by value in #790's declared key order.

    Copied rather than aliased so a caller mutating its citation afterwards
    cannot silently move a minted request's digest. Keys outside
    `CAPABILITY_INSPIRATION_CITATION_KEYS` are carried through rather than
    dropped, so an unexpected key is refused by that module's validator instead
    of disappearing between the caller's payload and the sealed one.
    """
    if not isinstance(citation, Mapping):
        return {}
    ordered = [key for key in CAPABILITY_INSPIRATION_CITATION_KEYS if key in citation]
    ordered.extend(sorted(key for key in citation if key not in set(CAPABILITY_INSPIRATION_CITATION_KEYS)))
    return {
        key: list(citation[key]) if isinstance(citation[key], list) else citation[key]
        for key in ordered
    }


def _authored_list(values: Sequence[str]) -> list[str]:
    """Authored prose, in the order it was written, deduplicated."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            cleaned.append(text)
            seen.add(text)
    return cleaned


def _vocabulary_list(values: Sequence[str], vocabulary: tuple[str, ...]) -> list[str]:
    """Closed-vocabulary members in declared order, plus anything unrecognised.

    Unrecognised values are kept rather than dropped: silently discarding one
    would turn a caller's typo into a request that validates and describes
    something nobody asked for.
    """
    named = [str(value).strip() for value in values if str(value).strip()]
    known = [item for item in vocabulary if item in named]
    unknown = sorted({item for item in named if item not in set(vocabulary)})
    return [*known, *unknown]


def _folded(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _text_errors(value: object, field: str, label: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, str):
        return [f"{label} {field} must be a string"]
    if not value.strip() and not allow_empty:
        return [f"{label} {field} must be a non-empty string"]
    if len(value) > _MAX_TEXT_LENGTH:
        return [f"{label} {field} must be at most {_MAX_TEXT_LENGTH} characters"]
    return []


def _text_list_errors(
    value: object,
    field: str,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return [f"{label} {field} must be a list of strings"]
    errors: list[str] = []
    if len(value) < minimum:
        errors.append(f"{label} {field} must name at least {minimum}")
    if len(value) > maximum:
        errors.append(f"{label} {field} must name at most {maximum}")
    if len(set(value)) != len(value):
        errors.append(f"{label} {field} must not repeat an entry")
    if [item for item in value if not item.strip()]:
        errors.append(f"{label} {field} must not contain an empty entry")
    if [item for item in value if len(item) > _MAX_TEXT_LENGTH]:
        errors.append(f"{label} {field} entries must be at most {_MAX_TEXT_LENGTH} characters")
    return errors
