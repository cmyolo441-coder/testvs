"""Reusable behavior maps: `reusable_behavior_map/v1` (issue #796).

What was missing
----------------

Two halves of this already existed and nothing joined them.

`plugin_risk_audit/v1` reads a bundle without running it. It walks one
explicitly named local directory through symlink-refusing dirfd-anchored opens,
returns aggregate risk categories, a manifest status, and two counts, and says
in its own `claim_boundary` that it imported nothing, registered nothing, and
executed nothing. What it deliberately does not return is *behavior*: it never
says what the bundle is for.

`awesome_hermes_plugin_outcome_matrix/v1` maps a demonstrated outcome onto OMH
capability ids and refuses a claim that outruns its evidence. What it does not
do is start from a bundle somebody just handed over; its rows are a reviewed
catalog pinned to an upstream commit.

So a person holding a complex demonstrated workflow could learn that it touches
a network and spawns a process, or could read a curated table about a plugin
somebody already reviewed, and had nothing that answered the actual question:
*of the things this bundle does, which ones are outcomes OMH should provide
natively, and which are the host's machinery.*

What a map is
-------------

One inventory over one cited audit. Each row is one behavior somebody observed
in the bundle, carrying the user outcome it produces, the host-free procedure
that would reproduce it, the OMH capability that already covers it, the host
mechanics that make it unreusable, the risk categories the audit already found,
and a note on how OMH would stand it up independently.

The map cites an audit; it never re-scans. `source_audit` is a projection of a
real `plugin_risk_audit/v1` payload, and a behavior may only name a risk
category that the cited audit actually observed. That is what keeps the risky
verdict evidence-backed instead of a second opinion about the same files.

Six states, and no seventh
--------------------------

#796 AC2 names six classifications. They are a closed vocabulary, and an
unrecognised value is refused rather than passed through, because a map whose
verdicts are open-ended is a list of adjectives.

They are made distinguishable structurally rather than by prose.
`CLASSIFICATION_REQUIRED_EVIDENCE` gives each classification the exact set of
evidence fields it must carry, and every evidence field it does not require is
forbidden. A behavior's non-empty evidence fields therefore *are* its
classification's signature, the six signatures are distinct, and no payload can
satisfy two classifications at once:

    reusable    reusable_procedure
    covered     reusable_procedure + omh_capability_ids
    host_bound  host_mechanics
    risky       risk_categories
    unclear     missing_evidence
    irrelevant  (nothing)

One consequence is deliberate: a verdict is singular. A behavior whose
demonstrated mechanic is risky is `risky`, not `reusable` with a risk note
attached, and a behavior welded to a host hook is `host_bound` even when that
hook is also risky. The nuance goes in `independence_note`, which every row
carries. Letting a row be two things at once is how "we know what this bundle
does" quietly becomes "we have opinions about this bundle".

Covered is a citation, not an adjective
---------------------------------------

#796 AC3 -- no pattern is called supported or implemented without OMH evidence
-- is enforced as resolution, not as format. `covered` requires at least one
`omh_capability_ids` entry, and every entry must resolve against
`omh_capability_vocabulary()`, which is re-derived from
`POPULAR_PLUGIN_FAMILIES` in `src/quality/popular_plugin_coverage.py`. A
well-formed slug naming a capability this repository does not have is refused
with the ids listed. Asserting coverage therefore costs the author a real
lookup, and a reviewer can check the claim without leaving the repo.

`not_observed` carries the other half: naming an existing OMH capability says
that capability exists, and says nothing about this map having built, run, or
verified anything.

The bundle is never run
-----------------------

The audit's boundary is inherited and restated. `CLAIM_BOUNDARY` says OMH does
not import, install, register, or execute the inspected bundle, does not install
its dependencies, and does not reach its endpoints. `NOT_OBSERVED` says the same
thing as data. `IMPLEMENTATION_CLAIM_KEYS` -- reused from
`native_capability_blueprint`, not forked, so the two families cannot drift on
what an execution claim looks like -- refuses a payload shaped to say otherwise
by key name, on the envelope and on every row.

Determinism
-----------

#796 AC1 is literally determinism, so nothing here reads a clock. `prepared_at`
is a parameter, defaults to empty, and is excluded from `map_digest`; a wall
clock inside a compared value turns an equality check into a race, and this
repo has already lost Windows CI time to exactly that.

Behaviors are sorted by `behavior_id` and closed-vocabulary lists are normalized
into their declared order, so two researchers who found the same things in a
different order produce the same map, byte for byte. `summary` is derived rather
than supplied, and the validator recomputes it, so a hand-edited count is a
validation error instead of a quiet disagreement with the rows above it.

What reads these today
----------------------

Stated because the vocabulary is wider than the wiring. No production surface
mints a map yet; this module is the contract and
`tests/test_reusable_behavior_map.py` is its only caller. The intended next step
is the one #796 describes: a behavior classified `reusable` becomes a separate
native capability request, and `host_mechanics` is deliberately
`SOURCE_HOST_MECHANICS` from `native_capability_blueprint` so a host-bound
exclusion here and a blueprint's `observed_source_mechanics` there are the same
words.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any, Final, get_args

from ..quality.popular_plugin_coverage import POPULAR_PLUGIN_FAMILIES
from ..system.append_only_store import RAW_OR_HIDDEN_KEYS
from .native_capability_blueprint import IMPLEMENTATION_CLAIM_KEYS, SOURCE_HOST_MECHANICS
from .plugin_risk_audit import PLUGIN_RISK_AUDIT_SCHEMA_VERSION, ManifestStatus, RiskCategory


REUSABLE_BEHAVIOR_MAP_SCHEMA_VERSION: Final[str] = "reusable_behavior_map/v1"

MAP_PRIVACY: Final[str] = "metadata_only"

# On every payload. It restates the audit's boundary rather than narrowing it:
# the map knows less about the bundle than the audit did, never more.
CLAIM_BOUNDARY: Final[str] = (
    "A reusable behavior map is an OMH-local research inventory over one cited plugin_risk_audit/v1, which "
    "read bounded text from one explicitly named local directory. OMH does not import, install, register, or "
    "execute the inspected bundle, does not install its dependencies, and does not reach its endpoints. A "
    "covered classification cites an OMH capability that already exists and is not execution, verification, "
    "review, CI, merge, or delivery evidence; every other classification is a research judgement about work "
    "OMH has not done."
)

# The same shape `plugin_risk_audit/v1` uses, so a reader who knows one knows
# the other. `native_implementation` is the row #796 AC3 needs: citing an OMH
# capability says that capability exists, not that this map built anything.
NOT_OBSERVED: Final[dict[str, dict[str, str]]] = {
    "bundle_import": {"status": "not_observed"},
    "bundle_registration": {"status": "not_observed"},
    "bundle_installation": {"status": "not_observed"},
    "bundle_execution": {"status": "not_observed"},
    "dependency_installation": {"status": "not_observed"},
    "network_access": {"status": "not_observed"},
    "native_implementation": {"status": "not_observed"},
}

# #796 AC2's six states, in the order the issue names them. Closed: an
# unrecognised verdict is refused, never carried.
BEHAVIOR_CLASSIFICATIONS: Final[tuple[str, ...]] = (
    "reusable",
    "covered",
    "host_bound",
    "risky",
    "unclear",
    "irrelevant",
)

# The per-behavior fields that carry evidence. Everything else on a row is
# required of every row and so distinguishes nothing.
BEHAVIOR_EVIDENCE_FIELDS: Final[tuple[str, ...]] = (
    "reusable_procedure",
    "omh_capability_ids",
    "host_mechanics",
    "risk_categories",
    "missing_evidence",
)

# What each verdict has to show, and -- by omission -- what it may not show. The
# six required sets are distinct, so a row's non-empty evidence fields identify
# exactly one classification. That is #796 AC2's "distinguishable" as a
# structural property rather than an assertion in prose.
CLASSIFICATION_REQUIRED_EVIDENCE: Final[dict[str, tuple[str, ...]]] = {
    "reusable": ("reusable_procedure",),
    "covered": ("reusable_procedure", "omh_capability_ids"),
    "host_bound": ("host_mechanics",),
    "risky": ("risk_categories",),
    "unclear": ("missing_evidence",),
    "irrelevant": (),
}

# What the cited audit is allowed to have found. Re-derived from the audit's own
# Literal aliases so the map cannot drift from the scanner it cites; adding a
# category there is immediately citable here with no edit.
AUDIT_RISK_CATEGORIES: Final[tuple[str, ...]] = tuple(get_args(RiskCategory))
AUDIT_MANIFEST_STATUSES: Final[tuple[str, ...]] = tuple(get_args(ManifestStatus))

# The projection of a `plugin_risk_audit/v1` payload the map keeps. Counts and
# manifest status are carried so a reader can tell how much of the bundle the
# cited audit actually saw before trusting an inventory built on it.
SOURCE_AUDIT_KEYS: Final[tuple[str, ...]] = (
    "manifest_status",
    "risk_categories",
    "scanned_byte_count",
    "scanned_file_count",
    "schema_version",
)

BEHAVIOR_KEYS: Final[tuple[str, ...]] = (
    "behavior_id",
    "classification",
    "host_mechanics",
    "independence_note",
    "missing_evidence",
    "omh_capability_ids",
    "reusable_procedure",
    "risk_categories",
    "user_outcome",
)

SUMMARY_KEYS: Final[tuple[str, ...]] = (
    "behavior_count",
    "cited_capability_ids",
    "cited_risk_categories",
    "classification_counts",
)

REUSABLE_BEHAVIOR_MAP_KEYS: Final[tuple[str, ...]] = (
    "behaviors",
    "claim_boundary",
    "map_digest",
    "not_observed",
    "prepared_at",
    "privacy",
    "schema_version",
    "source_audit",
    "summary",
)

# Exactly what `map_digest` seals: the cited evidence and the inventory built on
# it. The three module constants are not findings, and `prepared_at` is excluded
# because a clock inside a compared value is a race.
MAP_DIGEST_KEYS: Final[tuple[str, ...]] = ("behaviors", "source_audit", "summary")

_LABEL: Final[str] = "reusable_behavior_map"

# One observed behavior's handle: a slug a follow-up request or blueprint can
# reuse as an identifier.
_BEHAVIOR_ID: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}$")

_SHA256_HEX: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

# Free-text bounds. A map is authored research notes, not somewhere a bundle's
# source or a transcript can be parked.
_MAX_TEXT_LENGTH: Final[int] = 200
_MAX_BEHAVIORS: Final[int] = 64
_MAX_PROCEDURE_STEPS: Final[int] = 12
_MAX_CAPABILITY_IDS: Final[int] = 8
_MAX_MISSING_EVIDENCE: Final[int] = 6


class ReusableBehaviorMapError(ValueError):
    """Raised when observed behaviors cannot become a reusable behavior map."""


# ---------------------------------------------------------------------------
# OMH capability vocabulary
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def omh_capability_vocabulary() -> tuple[str, ...]:
    """Every OMH capability id a `covered` behavior may cite, sorted.

    Re-derived from `POPULAR_PLUGIN_FAMILIES` rather than restated: the family
    ids and the common-request case ids they map to are this repository's
    existing answer to "which OMH capability covers this kind of request", and a
    second hand-maintained list would be the thing that goes stale while still
    validating.
    """
    ids = {family.family_id for family in POPULAR_PLUGIN_FAMILIES}
    ids.update(case_id for family in POPULAR_PLUGIN_FAMILIES for case_id in family.case_ids)
    return tuple(sorted(ids))


def unresolved_capability_ids(capability_ids: Sequence[str]) -> tuple[str, ...]:
    """Every cited id that is not an OMH capability in this repository, sorted."""
    return tuple(sorted({str(item) for item in capability_ids} - set(omh_capability_vocabulary())))


# ---------------------------------------------------------------------------
# Build and validate
# ---------------------------------------------------------------------------


def build_reusable_behavior_map(
    *,
    audit: Mapping[str, Any] | None,
    behaviors: Sequence[Mapping[str, Any]],
    prepared_at: str = "",
) -> dict[str, Any]:
    """Mint one map over one cited audit, or refuse.

    `audit` is a `plugin_risk_audit/v1` payload, cited and projected -- nothing
    here opens the bundle again. Behaviors are sorted by `behavior_id` and
    closed-vocabulary lists are normalized into their declared order, so the same
    bounded evidence yields the same map whatever order it arrived in.

    Nothing reads a clock. `prepared_at` is whatever the caller passes and is
    excluded from `map_digest`.
    """
    rows = [_normalized_behavior(entry) for entry in behaviors]
    rows.sort(key=_behavior_sort_key)
    behavior_map: dict[str, Any] = {
        "schema_version": REUSABLE_BEHAVIOR_MAP_SCHEMA_VERSION,
        "source_audit": _audit_citation(audit),
        "behaviors": rows,
        "summary": _summary(rows),
        "not_observed": _copied_not_observed(),
        "prepared_at": str(prepared_at).strip(),
        "privacy": MAP_PRIVACY,
        "claim_boundary": CLAIM_BOUNDARY,
        "map_digest": "",
    }
    behavior_map["map_digest"] = map_digest_of(behavior_map)
    errors = validate_reusable_behavior_map(behavior_map)
    if errors:
        raise ReusableBehaviorMapError(errors[0])
    return behavior_map


def map_digest_of(behavior_map: Mapping[str, Any]) -> str:
    """A sha256 over the cited evidence and the inventory, covering no clock."""
    identity = {key: behavior_map.get(key) for key in MAP_DIGEST_KEYS}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def validate_reusable_behavior_map(behavior_map: Mapping[str, Any]) -> list[str]:
    """Every reason one payload is not a reusable behavior map."""
    if not isinstance(behavior_map, Mapping):
        return [f"{_LABEL} must be an object"]
    errors = _forbidden_key_errors(behavior_map, _LABEL)
    reported = {key for key in behavior_map if str(key).lower() in IMPLEMENTATION_CLAIM_KEYS | RAW_OR_HIDDEN_KEYS}
    unexpected = sorted(set(behavior_map) - set(REUSABLE_BEHAVIOR_MAP_KEYS) - reported)
    if unexpected:
        errors.append(f"{_LABEL} has unsupported keys: {unexpected}")
    missing = sorted(set(REUSABLE_BEHAVIOR_MAP_KEYS) - set(behavior_map))
    if missing:
        errors.append(f"{_LABEL} is missing keys: {missing}")
    if behavior_map.get("schema_version") != REUSABLE_BEHAVIOR_MAP_SCHEMA_VERSION:
        errors.append(f"{_LABEL} schema_version must be {REUSABLE_BEHAVIOR_MAP_SCHEMA_VERSION}")
    if behavior_map.get("privacy") != MAP_PRIVACY:
        errors.append(f"{_LABEL} privacy must be {MAP_PRIVACY}")
    if behavior_map.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append(
            f"{_LABEL} claim_boundary must state that OMH never imports, installs, registers, or executes the "
            "inspected bundle"
        )
    if behavior_map.get("not_observed") != NOT_OBSERVED:
        errors.append(
            f"{_LABEL} not_observed must record every unobserved bundle interaction: "
            f"{sorted(NOT_OBSERVED)}"
        )
    if not isinstance(behavior_map.get("prepared_at"), str):
        errors.append(f"{_LABEL} prepared_at must be a string")
    audit_errors = _source_audit_errors(behavior_map.get("source_audit"))
    errors.extend(audit_errors)
    errors.extend(_behaviors_errors(behavior_map.get("behaviors"), cited_risks=_cited_risks(behavior_map)))
    errors.extend(_summary_errors(behavior_map))
    errors.extend(_digest_errors(behavior_map))
    return errors


# ---------------------------------------------------------------------------
# Cited audit
# ---------------------------------------------------------------------------


def _audit_citation(audit: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project the parts of a `plugin_risk_audit/v1` payload the map cites.

    Values are carried through unconverted. A malformed audit becomes a
    validation error naming the field, not a coercion that invents a count the
    scanner never reported.
    """
    payload = audit if isinstance(audit, Mapping) else {}
    source = payload.get("source")
    summary = payload.get("summary")
    source = source if isinstance(source, Mapping) else {}
    summary = summary if isinstance(summary, Mapping) else {}
    risk_categories = summary.get("risk_categories")
    return {
        "schema_version": payload.get("schema_version", ""),
        "manifest_status": source.get("manifest_status", ""),
        "scanned_file_count": summary.get("scanned_file_count", -1),
        "scanned_byte_count": summary.get("scanned_byte_count", -1),
        "risk_categories": list(risk_categories) if isinstance(risk_categories, list) else risk_categories,
    }


def _source_audit_errors(source_audit: object) -> list[str]:
    """A map cites an audit. No audit, no map."""
    if not isinstance(source_audit, Mapping):
        return [f"{_LABEL} source_audit must cite a {PLUGIN_RISK_AUDIT_SCHEMA_VERSION} payload"]
    errors: list[str] = []
    if sorted(source_audit) != list(SOURCE_AUDIT_KEYS):
        errors.append(f"{_LABEL} source_audit keys must be exactly {list(SOURCE_AUDIT_KEYS)}")
    if source_audit.get("schema_version") != PLUGIN_RISK_AUDIT_SCHEMA_VERSION:
        errors.append(
            f"{_LABEL} source_audit must cite a {PLUGIN_RISK_AUDIT_SCHEMA_VERSION} payload; the map cites a "
            "bounded audit and never scans the bundle itself"
        )
    if source_audit.get("manifest_status") not in AUDIT_MANIFEST_STATUSES:
        errors.append(
            f"{_LABEL} source_audit manifest_status is unsupported: {source_audit.get('manifest_status')!r}"
        )
    for field in ("scanned_file_count", "scanned_byte_count"):
        value = source_audit.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{_LABEL} source_audit {field} must be a non-negative integer")
    errors.extend(
        _vocabulary_list_errors(
            source_audit.get("risk_categories"),
            "source_audit risk_categories",
            AUDIT_RISK_CATEGORIES,
        )
    )
    return errors


def _cited_risks(behavior_map: Mapping[str, Any]) -> frozenset[str]:
    source_audit = behavior_map.get("source_audit")
    if not isinstance(source_audit, Mapping):
        return frozenset()
    categories = source_audit.get("risk_categories")
    if not isinstance(categories, list):
        return frozenset()
    return frozenset(str(item) for item in categories)


# ---------------------------------------------------------------------------
# Behaviors
# ---------------------------------------------------------------------------


def _normalized_behavior(raw: Mapping[str, Any]) -> Any:
    """One row in canonical form, with anything unrecognised carried through.

    Unknown keys are kept rather than dropped so the validator can refuse them:
    silently discarding a caller's typo produces a map that validates and says
    something nobody wrote.
    """
    if not isinstance(raw, Mapping):
        return raw
    row: dict[str, Any] = {
        "behavior_id": str(raw.get("behavior_id", "")).strip().casefold(),
        "classification": str(raw.get("classification", "")).strip(),
        "user_outcome": str(raw.get("user_outcome", "")).strip(),
        "reusable_procedure": _authored_list(raw.get("reusable_procedure")),
        "omh_capability_ids": _sorted_list(raw.get("omh_capability_ids")),
        "host_mechanics": _vocabulary_list(raw.get("host_mechanics"), SOURCE_HOST_MECHANICS),
        "risk_categories": _vocabulary_list(raw.get("risk_categories"), AUDIT_RISK_CATEGORIES),
        "missing_evidence": _authored_list(raw.get("missing_evidence")),
        "independence_note": str(raw.get("independence_note", "")).strip(),
    }
    for key, value in raw.items():
        if key not in row:
            row[key] = value
    return {key: row[key] for key in sorted(row)}


def _behavior_sort_key(row: object) -> str:
    if not isinstance(row, Mapping):
        return ""
    return str(row.get("behavior_id", ""))


def _behaviors_errors(behaviors: object, *, cited_risks: frozenset[str]) -> list[str]:
    if not isinstance(behaviors, list) or not all(isinstance(row, Mapping) for row in behaviors):
        return [f"{_LABEL} behaviors must be a list of objects"]
    errors: list[str] = []
    if not behaviors:
        errors.append(
            f"{_LABEL} behaviors must name at least 1 observed behavior; an empty inventory records no "
            "research and is not a finding about the bundle"
        )
    if len(behaviors) > _MAX_BEHAVIORS:
        errors.append(f"{_LABEL} behaviors must name at most {_MAX_BEHAVIORS}")
    identifiers = [str(row.get("behavior_id", "")) for row in behaviors]
    if identifiers != sorted(identifiers):
        errors.append(f"{_LABEL} behaviors must be sorted by behavior_id so the same evidence renders identically")
    duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
    if duplicates:
        errors.append(f"{_LABEL} behaviors must not repeat a behavior_id: {duplicates}")
    for row in behaviors:
        errors.extend(_behavior_errors(row, cited_risks=cited_risks))
    return errors


def _behavior_errors(row: Mapping[str, Any], *, cited_risks: frozenset[str]) -> list[str]:
    identifier = str(row.get("behavior_id", ""))
    label = f"{_LABEL} behavior {identifier!r}"
    errors = _forbidden_key_errors(row, label)
    reported = {key for key in row if str(key).lower() in IMPLEMENTATION_CLAIM_KEYS | RAW_OR_HIDDEN_KEYS}
    unexpected = sorted(set(row) - set(BEHAVIOR_KEYS) - reported)
    if unexpected:
        errors.append(f"{label} has unsupported keys: {unexpected}")
    missing = sorted(set(BEHAVIOR_KEYS) - set(row))
    if missing:
        errors.append(f"{label} is missing keys: {missing}")
    if not _BEHAVIOR_ID.match(identifier):
        errors.append(f"{label} behavior_id must be a lowercase slug")
    errors.extend(_text_errors(row.get("user_outcome"), f"{label} user_outcome"))
    errors.extend(_text_errors(row.get("independence_note"), f"{label} independence_note"))
    errors.extend(_evidence_shape_errors(row, label))
    errors.extend(_evidence_content_errors(row, label, cited_risks=cited_risks))
    return errors


def _evidence_shape_errors(row: Mapping[str, Any], label: str) -> list[str]:
    """#796 AC2: six states, each with its own evidence signature and no other.

    A classification outside the vocabulary stops here. Checking a row's
    evidence against an unknown verdict would report which fields the unknown
    state wants, and it does not want anything -- it does not exist.
    """
    classification = row.get("classification")
    if classification not in CLASSIFICATION_REQUIRED_EVIDENCE:
        return [
            f"{label} classification is unsupported: {classification!r}; the six states are "
            f"{list(BEHAVIOR_CLASSIFICATIONS)}"
        ]
    errors: list[str] = []
    required = set(CLASSIFICATION_REQUIRED_EVIDENCE[classification])
    for field in BEHAVIOR_EVIDENCE_FIELDS:
        value = row.get(field)
        if not isinstance(value, list):
            errors.append(f"{label} {field} must be a list of strings")
            continue
        if field in required and not value:
            errors.append(f"{label} classified {classification} must name {field}")
        if field not in required and value:
            errors.append(
                f"{label} classified {classification} must not name {field}; a behavior carries one verdict, "
                f"and {classification} is evidenced by {sorted(required) or 'no evidence field'}"
            )
    return errors


def _evidence_content_errors(row: Mapping[str, Any], label: str, *, cited_risks: frozenset[str]) -> list[str]:
    """#796 AC3, plus the rule that keeps the risky verdict tied to the audit."""
    errors = _text_list_errors(row.get("reusable_procedure"), f"{label} reusable_procedure", _MAX_PROCEDURE_STEPS)
    errors.extend(_text_list_errors(row.get("missing_evidence"), f"{label} missing_evidence", _MAX_MISSING_EVIDENCE))
    errors.extend(_vocabulary_list_errors(row.get("host_mechanics"), f"{label} host_mechanics", SOURCE_HOST_MECHANICS))
    errors.extend(
        _vocabulary_list_errors(row.get("risk_categories"), f"{label} risk_categories", AUDIT_RISK_CATEGORIES)
    )
    capability_ids = row.get("omh_capability_ids")
    if not isinstance(capability_ids, list) or not all(isinstance(item, str) for item in capability_ids):
        errors.append(f"{label} omh_capability_ids must be a list of strings")
    else:
        if len(capability_ids) > _MAX_CAPABILITY_IDS:
            errors.append(f"{label} omh_capability_ids must name at most {_MAX_CAPABILITY_IDS}")
        if len(set(capability_ids)) != len(capability_ids):
            errors.append(f"{label} omh_capability_ids must not repeat an entry")
        unresolved = unresolved_capability_ids(capability_ids)
        if unresolved:
            errors.append(
                f"{label} omh_capability_ids do not name an OMH capability: {list(unresolved)}; a covered "
                "behavior cites a capability id from src/quality/popular_plugin_coverage.py, so coverage is a "
                "reference a reviewer can resolve rather than an assertion"
            )
    risk_categories = row.get("risk_categories")
    if isinstance(risk_categories, list):
        uncited = sorted({str(item) for item in risk_categories} - cited_risks)
        if uncited:
            errors.append(
                f"{label} risk_categories are absent from the cited audit: {uncited}; the map cites "
                f"{PLUGIN_RISK_AUDIT_SCHEMA_VERSION} for risky mechanics and never re-scans the bundle"
            )
    return errors


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _summary(rows: Sequence[Any]) -> dict[str, Any]:
    """Counts derived from the rows, with all six states present at every build.

    Every classification appears even at zero. A summary that omits the states
    nobody used reads as though those states were not considered.
    """
    mappings = [row for row in rows if isinstance(row, Mapping)]
    classifications = [str(row.get("classification", "")) for row in mappings]
    capability_ids: set[str] = set()
    risk_categories: set[str] = set()
    for row in mappings:
        capability_ids.update(str(item) for item in _string_list(row.get("omh_capability_ids")))
        risk_categories.update(str(item) for item in _string_list(row.get("risk_categories")))
    return {
        "behavior_count": len(rows),
        "classification_counts": {name: classifications.count(name) for name in BEHAVIOR_CLASSIFICATIONS},
        "cited_capability_ids": sorted(capability_ids),
        "cited_risk_categories": [name for name in AUDIT_RISK_CATEGORIES if name in risk_categories],
    }


def _summary_errors(behavior_map: Mapping[str, Any]) -> list[str]:
    summary = behavior_map.get("summary")
    if not isinstance(summary, Mapping):
        return [f"{_LABEL} summary must be an object"]
    if sorted(summary) != list(SUMMARY_KEYS):
        return [f"{_LABEL} summary keys must be exactly {list(SUMMARY_KEYS)}"]
    behaviors = behavior_map.get("behaviors")
    if not isinstance(behaviors, list):
        return []
    derived = _summary(behaviors)
    if dict(summary) != derived:
        return [
            f"{_LABEL} summary must be derived from the behaviors it summarizes; expected {derived}",
        ]
    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _copied_not_observed() -> dict[str, dict[str, str]]:
    return {key: dict(value) for key, value in NOT_OBSERVED.items()}


def _digest_errors(behavior_map: Mapping[str, Any]) -> list[str]:
    digest = behavior_map.get("map_digest")
    if not isinstance(digest, str) or not _SHA256_HEX.match(digest):
        return [f"{_LABEL} map_digest must be a sha256 hex digest"]
    if digest != map_digest_of(behavior_map):
        return [f"{_LABEL} map_digest does not match the inventory it seals; the payload was edited after it was minted"]
    return []


def _forbidden_key_errors(payload: Mapping[str, Any], label: str) -> list[str]:
    """Key names under which "we ran it" arrives, refused by name on every row."""
    errors: list[str] = []
    claims = sorted(key for key in payload if str(key).lower() in IMPLEMENTATION_CLAIM_KEYS)
    if claims:
        errors.append(
            f"{label} must not carry implementation-claim keys: {claims}; a map records what a bounded read "
            "suggests the bundle does and never reports running, installing, or building anything"
        )
    hidden = sorted(key for key in payload if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if hidden:
        errors.append(f"{label} must not carry raw or hidden keys: {hidden}")
    return errors


def _is_list_shaped(value: object) -> bool:
    """A sequence that is not itself text.

    A caller who passes a string where a list belongs gets it back unchanged and
    a validation error naming the field, rather than a normalized list of that
    string's characters.
    """
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _authored_list(value: object) -> Any:
    """Authored prose in the order it was written, deduplicated."""
    if not _is_list_shaped(value):
        return [] if value is None else value
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:  # type: ignore[union-attr]
        text = str(item).strip()
        if text and text not in seen:
            cleaned.append(text)
            seen.add(text)
    return cleaned


def _sorted_list(value: object) -> Any:
    """Identifiers with no meaningful order, sorted and deduplicated."""
    if not _is_list_shaped(value):
        return [] if value is None else value
    return sorted({str(item).strip() for item in value if str(item).strip()})  # type: ignore[union-attr]


def _vocabulary_list(value: object, vocabulary: tuple[str, ...]) -> Any:
    """Closed-vocabulary members in declared order, plus anything unrecognised.

    Unrecognised values survive normalization so the validator can name them.
    """
    if not _is_list_shaped(value):
        return [] if value is None else value
    named = [str(item).strip() for item in value if str(item).strip()]  # type: ignore[union-attr]
    known = [item for item in vocabulary if item in named]
    unknown = sorted({item for item in named if item not in set(vocabulary)})
    return [*known, *unknown]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _text_errors(value: object, field: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return [f"{field} must be a non-empty string"]
    if len(value) > _MAX_TEXT_LENGTH:
        return [f"{field} must be at most {_MAX_TEXT_LENGTH} characters"]
    return []


def _text_list_errors(value: object, field: str, maximum: int) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return [f"{field} must be a list of strings"]
    errors: list[str] = []
    if len(value) > maximum:
        errors.append(f"{field} must name at most {maximum}")
    if len(set(value)) != len(value):
        errors.append(f"{field} must not repeat an entry")
    if any(not item.strip() for item in value):
        errors.append(f"{field} must not contain an empty entry")
    if any(len(item) > _MAX_TEXT_LENGTH for item in value):
        errors.append(f"{field} entries must be at most {_MAX_TEXT_LENGTH} characters")
    return errors


def _vocabulary_list_errors(value: object, field: str, vocabulary: tuple[str, ...]) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return [f"{field} must be a list of strings"]
    errors: list[str] = []
    if len(set(value)) != len(value):
        errors.append(f"{field} must not repeat an entry")
    unsupported = sorted({item for item in value if item not in set(vocabulary)})
    if unsupported:
        errors.append(f"{field} has unsupported entries: {unsupported}; allowed: {list(vocabulary)}")
    return errors
