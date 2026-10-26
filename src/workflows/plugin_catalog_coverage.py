"""Plugin catalog coverage (`plugin_catalog_coverage/v1`, issue #1449).

Reconciles one host-supplied `plugin_catalog_snapshot/v1` against OMH's own
workflow ownership and answers four questions a maintainer currently cannot
ask: which OMH workflow owns each plugin outcome, what changed since the prior
snapshot, whether a compatibility or removal status blocks use, and what
evidence OMH does not have.

Everything here is a pure function of the two snapshots it is handed. Nothing
is fetched, imported, installed, activated, or executed, and no host
configuration is read or written.

Four separations do the work, and none of them collapses into another:

- **Coverage class is not change class.** An entry gets exactly one coverage
  class -- what OMH can do with it -- and, separately, exactly one change class
  -- how it differs from the prior snapshot. A first sighting of an entry that
  OMH cannot map is `added` and `generic_review`, not one merged verdict.
- **A rediscovered entry is not a new one.** Change classification keys on
  `plugin_id`, and `changed` requires a difference in tracked metadata. An
  identical snapshot replayed produces `unchanged` for every row and no new
  row at all. A reviewed content revision that moved under a known identity is
  `changed`, and both revisions survive into the row -- reporting that as
  `added` would lose the fact that the identity was already reviewed once.
- **Catalog presence is not readiness.** No entry ever reaches a ready state
  here. A mapped entry is routed to the workflow that owns its outcome, and
  that workflow's own readiness contract is where an adoption answer comes
  from. An unsatisfied host-version constraint or a catalog removal is held
  outright, and a stale snapshot holds every row it carries.
- **Absence does not inherit presence.** When no snapshot is available the
  answer is `unavailable` with no entries. It never falls back to the packaged
  catalog OMH ships, because that catalog describes a repository revision and
  says nothing about the host's live catalog.

A removal hold says the catalog withdrew the entry. It never says an installed
copy was disabled or uninstalled: OMH cannot see installed copies, and the
host owns that state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from ..catalogs.awesome_hermes_agent import (
    PACKAGED_COVERAGE_SCOPE,
    PACKAGED_COVERAGE_SCOPE_NOTE,
    UPSTREAM_SOURCE_COMMIT,
)
from .plugin_catalog_snapshots import (
    MAX_DIAGNOSTICS,
    snapshot_entries,
    snapshot_freshness,
    snapshot_identity,
    validate_plugin_catalog_snapshot,
)


PLUGIN_CATALOG_COVERAGE_SCHEMA_VERSION: Final = "plugin_catalog_coverage/v1"

COVERAGE_CLAIM_BOUNDARY: Final = (
    "Plugin catalog coverage interprets one supplied catalog snapshot against OMH workflow ownership. "
    "Declared capabilities are catalog claims, not observed plugin behavior, and coverage is not plugin "
    "installation, admission, compatibility enforcement, permission, activation, update, removal, or "
    "execution evidence. A removal hold reports a catalog withdrawal only and says nothing about whether "
    "an installed copy exists, was disabled, or was uninstalled."
)

#: Exactly one of these per row. `unknown` is the honest class for an entry
#: that declared no capabilities at all -- there is nothing to interpret, which
#: is different from having declared something OMH does not own.
COVERAGE_CLASSES: Final = ("mapped", "generic_review", "incompatible", "removed", "unknown")

#: Exactly one of these per row, against the prior snapshot.
CHANGE_CLASSES: Final = ("added", "changed", "unchanged", "removed")

#: What OMH tells the reader to do next. There is deliberately no `ready`:
#: catalog presence cannot produce one, and each owner workflow keeps its own
#: readiness contract.
ADOPTION_GUIDANCE: Final = ("route_to_owner", "generic_review", "host_owned", "held")

#: Declared capability token -> the OMH workflow that owns that outcome, in
#: precedence order. The order is the tie-break when one entry declares
#: several, so a multi-capability entry routes to the same primary owner on
#: every run.
CAPABILITY_OWNERS: Final = (
    ("provider", "provider-profile-posture"),
    ("connector", "external-connector-readiness"),
    ("observability", "ops-observability-card"),
    ("memory", "memory-sync"),
    ("media", "media-input-operator"),
)

#: The capability that names host loader, transport, permission, and
#: installation mechanics. OMH owns no workflow for it on purpose; the row is
#: reported with a host owner boundary rather than routed into an OMH lane.
HOST_CORE_CAPABILITY: Final = "host_core"

#: The bounded route for an entry OMH cannot map. Capability scouting is what
#: answers "is there an OMH surface for this?", so an unmapped entry lands
#: there rather than disappearing from the report.
GENERIC_REVIEW_ROUTE: Final = "skill-scout"

#: The bounded route for a held entry. A withdrawal or an unmet host-version
#: constraint is a review question, not an adoption one.
HELD_REVIEW_ROUTE: Final = "security-safety-review"

#: Evidence OMH never has about any catalog entry, whatever its class.
BASELINE_UNAVAILABLE_EVIDENCE: Final = (
    "host_observed_behavior",
    "host_permission_grant",
    "plugin_execution",
)

#: Metadata whose movement makes an existing identity `changed`. Identity
#: itself is excluded: a different `plugin_id` is a different entry.
TRACKED_ENTRY_FIELDS: Final = (
    "content_revision",
    "declared_capabilities",
    "host_version_constraint",
    "host_version_status",
    "platform_limits",
    "removal_status",
    "required_env_names",
    "tier",
)

_SNAPSHOT_STATES: Final = ("current", "stale", "unavailable")


class PluginCatalogCoverageError(ValueError):
    """The supplied inputs cannot produce a coverage answer at all."""


def build_plugin_catalog_coverage(
    snapshot: Mapping[str, Any],
    *,
    previous: Mapping[str, Any] | None = None,
    profile_ref: str = "",
    now: str = "",
) -> dict[str, Any]:
    """Reconcile one snapshot, optionally against the prior one.

    Raises `PluginCatalogCoverageError` for every input that cannot be
    interpreted -- malformed, oversized, duplicate-identity, or from a
    different profile than the caller asked about. Those fail closed rather
    than producing a partial answer, because a partial coverage report reads
    exactly like a complete one.
    """
    diagnostics = validate_plugin_catalog_snapshot(snapshot)
    if diagnostics:
        raise PluginCatalogCoverageError("; ".join(diagnostics))
    previous_snapshot: Mapping[str, Any] | None = None
    if previous is not None:
        previous_diagnostics = validate_plugin_catalog_snapshot(previous)
        if previous_diagnostics:
            raise PluginCatalogCoverageError(
                "; ".join(f"previous snapshot: {item}" for item in previous_diagnostics[:MAX_DIAGNOSTICS])
            )
        previous_snapshot = previous
    identity = snapshot_identity(snapshot)
    requested_profile = profile_ref.strip()
    if requested_profile and requested_profile != identity["profile_ref"]:
        raise PluginCatalogCoverageError(
            "snapshot was taken for a different profile than the one requested; "
            "a cross-profile snapshot is a real observation of something else"
        )
    if previous_snapshot is not None:
        previous_identity = snapshot_identity(previous_snapshot)
        if previous_identity["profile_ref"] != identity["profile_ref"]:
            raise PluginCatalogCoverageError(
                "previous snapshot was taken for a different profile; comparing across profiles would "
                "report host changes that never happened"
            )
    freshness = snapshot_freshness(snapshot, now=now)
    snapshot_state = "current" if freshness["state"] == "fresh" else "stale"
    current = {entry["plugin_id"]: entry for entry in snapshot_entries(snapshot)}
    prior = (
        {entry["plugin_id"]: entry for entry in snapshot_entries(previous_snapshot)}
        if previous_snapshot is not None
        else {}
    )
    rows = [
        _coverage_row(
            plugin_id,
            current.get(plugin_id),
            prior.get(plugin_id),
            snapshot_state=snapshot_state,
        )
        for plugin_id in sorted(set(current) | set(prior))
    ]
    return _coverage_payload(
        rows,
        identity=identity,
        previous_identity=snapshot_identity(previous_snapshot) if previous_snapshot is not None else None,
        freshness=freshness,
        snapshot_state=snapshot_state,
        entry_count=len(current),
        diagnostics=[],
    )


def plugin_catalog_coverage_unavailable(reason: str) -> dict[str, Any]:
    """The answer when no snapshot is available.

    Carries no entries and no packaged-catalog substitute. The packaged
    reference block is still attached, labelled as the snapshot-limited
    repository projection it is, so a reader can see that OMH declined to use
    it rather than wondering whether it silently did.
    """
    bounded = reason.strip()[:200] or "no plugin catalog snapshot was supplied"
    return _coverage_payload(
        [],
        identity={
            "producer_kind": "",
            "producer_ref": "",
            "host_version": "",
            "profile_ref": "",
            "observed_at": "",
            "catalog_revision": "",
        },
        previous_identity=None,
        freshness={
            "state": "unknown",
            "age_seconds": None,
            "stale_after_seconds": None,
            "evaluated_at": "",
        },
        snapshot_state="unavailable",
        entry_count=0,
        diagnostics=[bounded],
    )


def render_plugin_catalog_coverage(payload: Mapping[str, Any]) -> str:
    """The concise chat status for one coverage answer."""
    summary = payload["summary"]
    lines = [f"Plugin catalog coverage: {payload['snapshot_state']}"]
    if payload["snapshot_state"] == "unavailable":
        lines.append("No host catalog snapshot was supplied, so OMH reports no host coverage.")
        for diagnostic in payload["diagnostics"]:
            lines.append(f"  - {diagnostic}")
        lines.extend(["", _packaged_reference_line(payload), "", str(payload["claim_boundary"])])
        return "\n".join(lines)
    identity = payload["snapshot"]
    lines.append(
        f"Snapshot: revision {identity['catalog_revision']} from {identity['producer_kind']} "
        f"{identity['producer_ref']} observed {identity['observed_at']} (profile {identity['profile_ref']})"
    )
    freshness = payload["freshness"]
    lines.append(f"Freshness: {freshness['state']} (age {freshness['age_seconds']}s)")
    lines.append(f"Entries: {payload['entry_count']}")
    lines.append("Coverage:")
    for name in COVERAGE_CLASSES:
        lines.append(f"  {name}: {summary['coverage_class_counts'][name]}")
    lines.append("Changes since the prior snapshot:")
    if payload["previous_snapshot"] is None:
        lines.append("  no prior snapshot supplied; every entry is a first sighting")
    for name in CHANGE_CLASSES:
        lines.append(f"  {name}: {summary['change_class_counts'][name]}")
    if payload["held_entries"]:
        lines.append("Held:")
        for row in payload["entries"]:
            if row["adoption_guidance"] == "held":
                lines.append(f"  {row['plugin_id']}: {', '.join(row['holds'])}")
    if payload["unresolved_entries"]:
        lines.append(f"Unresolved: {', '.join(payload['unresolved_entries'])} -> {GENERIC_REVIEW_ROUTE}")
    lines.extend(["", _packaged_reference_line(payload), "", str(payload["claim_boundary"])])
    return "\n".join(lines)


def _coverage_payload(
    rows: Sequence[Mapping[str, Any]],
    *,
    identity: Mapping[str, Any],
    previous_identity: Mapping[str, Any] | None,
    freshness: Mapping[str, Any],
    snapshot_state: str,
    entry_count: int,
    diagnostics: Sequence[str],
) -> dict[str, Any]:
    if snapshot_state not in _SNAPSHOT_STATES:
        raise PluginCatalogCoverageError(f"unsupported snapshot state: {snapshot_state}")
    coverage_counts = {name: 0 for name in COVERAGE_CLASSES}
    change_counts = {name: 0 for name in CHANGE_CLASSES}
    for row in rows:
        coverage_counts[str(row["coverage_class"])] += 1
        change_counts[str(row["change_class"])] += 1
    held = sorted(str(row["plugin_id"]) for row in rows if row["adoption_guidance"] == "held")
    unresolved = sorted(
        str(row["plugin_id"]) for row in rows if row["coverage_class"] in {"generic_review", "unknown"}
    )
    return {
        "schema_version": PLUGIN_CATALOG_COVERAGE_SCHEMA_VERSION,
        "snapshot_state": snapshot_state,
        "snapshot": dict(identity),
        "previous_snapshot": dict(previous_identity) if previous_identity is not None else None,
        "freshness": dict(freshness),
        "entry_count": entry_count,
        "summary": {
            "coverage_class_counts": coverage_counts,
            "change_class_counts": change_counts,
            "held_count": len(held),
            "unresolved_count": len(unresolved),
        },
        "entries": [dict(row) for row in rows],
        "held_entries": held,
        "unresolved_entries": unresolved,
        "packaged_reference": {
            "catalog_revision": UPSTREAM_SOURCE_COMMIT,
            "coverage_scope": PACKAGED_COVERAGE_SCOPE,
            "coverage_scope_note": PACKAGED_COVERAGE_SCOPE_NOTE,
            "used_as_host_coverage": False,
        },
        "diagnostics": list(diagnostics),
        "claim_boundary": COVERAGE_CLAIM_BOUNDARY,
    }


def _coverage_row(
    plugin_id: str,
    entry: Mapping[str, Any] | None,
    prior: Mapping[str, Any] | None,
    *,
    snapshot_state: str,
) -> dict[str, Any]:
    present = entry is not None
    source = entry if entry is not None else (prior if prior is not None else {})
    capabilities = tuple(str(item) for item in source.get("declared_capabilities", ()))
    platform_limits = tuple(str(item) for item in source.get("platform_limits", ()))
    env_names = tuple(str(item) for item in source.get("required_env_names", ()))
    removal_status = str(source.get("removal_status", ""))
    host_version_status = str(source.get("host_version_status", ""))
    classification = _classify(
        present=present,
        capabilities=capabilities,
        removal_status=removal_status,
        host_version_status=host_version_status,
    )
    holds = list(classification["holds"])
    guidance = str(classification["adoption_guidance"])
    if snapshot_state == "stale":
        holds.append("stale_snapshot")
        guidance = "held"
    advisories = _advisories(
        tier=str(source.get("tier", "")),
        removal_status=removal_status,
        platform_limits=platform_limits,
        env_names=env_names,
        host_core=HOST_CORE_CAPABILITY in capabilities,
    )
    return {
        "plugin_id": plugin_id,
        "present_in_snapshot": present,
        "coverage_class": classification["coverage_class"],
        "change_class": _change_class(entry, prior),
        "changed_fields": _changed_fields(entry, prior),
        "content_revision": str(entry.get("content_revision", "")) if entry is not None else "",
        "previous_content_revision": str(prior.get("content_revision", "")) if prior is not None else "",
        "tier": str(source.get("tier", "")),
        "declared_capabilities": list(capabilities),
        "owner_boundary": classification["owner_boundary"],
        "route": classification["route"],
        "additional_routes": list(classification["additional_routes"]),
        "adoption_guidance": guidance,
        "holds": sorted(set(holds)),
        "advisories": advisories,
        "unavailable_evidence": _unavailable_evidence(
            coverage_class=str(classification["coverage_class"]),
            host_version_status=host_version_status,
            platform_limits=platform_limits,
        ),
        "required_env_names": list(env_names),
        "platform_limits": list(platform_limits),
        "host_version_constraint": str(source.get("host_version_constraint", "")),
        "host_version_status": host_version_status,
        "removal_status": removal_status,
    }


def _classify(
    *,
    present: bool,
    capabilities: tuple[str, ...],
    removal_status: str,
    host_version_status: str,
) -> dict[str, Any]:
    """Exactly one coverage class per entry, by fixed precedence.

    Withdrawal and incompatibility come before capability mapping on purpose:
    routing a withdrawn or refused entry to its outcome owner would read as
    adoption guidance for something the host will not run.
    """
    if not present:
        return _held("removed", "withdrawn_from_catalog")
    if removal_status == "removed":
        return _held("removed", "catalog_removal")
    if host_version_status == "unsatisfied":
        return _held("incompatible", "host_version_unsatisfied")
    owned = [workflow for capability, workflow in CAPABILITY_OWNERS if capability in capabilities]
    if owned:
        return {
            "coverage_class": "mapped",
            "owner_boundary": "omh",
            "route": owned[0],
            "additional_routes": tuple(owned[1:]),
            "adoption_guidance": "route_to_owner",
            "holds": (),
        }
    if HOST_CORE_CAPABILITY in capabilities:
        return {
            "coverage_class": "generic_review",
            "owner_boundary": "hermes_host",
            "route": GENERIC_REVIEW_ROUTE,
            "additional_routes": (),
            "adoption_guidance": "host_owned",
            "holds": (),
        }
    return {
        "coverage_class": "generic_review" if capabilities else "unknown",
        "owner_boundary": "omh",
        "route": GENERIC_REVIEW_ROUTE,
        "additional_routes": (),
        "adoption_guidance": "generic_review",
        "holds": (),
    }


def _held(coverage_class: str, hold: str) -> dict[str, Any]:
    return {
        "coverage_class": coverage_class,
        "owner_boundary": "omh",
        "route": HELD_REVIEW_ROUTE,
        "additional_routes": (),
        "adoption_guidance": "held",
        "holds": (hold,),
    }


def _change_class(entry: Mapping[str, Any] | None, prior: Mapping[str, Any] | None) -> str:
    """One change class per row against the prior snapshot.

    With no prior snapshot every row is `added` -- a first sighting. Calling it
    `unchanged` would claim a comparison that never happened, and the payload
    says separately that no previous snapshot was supplied so a reader can tell
    a first run from one admitted entry.
    """
    if entry is None:
        return "removed"
    if prior is None:
        return "added"
    return "changed" if _changed_fields(entry, prior) else "unchanged"


def _changed_fields(entry: Mapping[str, Any] | None, prior: Mapping[str, Any] | None) -> list[str]:
    if entry is None or prior is None:
        return []
    return [field for field in TRACKED_ENTRY_FIELDS if entry.get(field) != prior.get(field)]


def _advisories(
    *,
    tier: str,
    removal_status: str,
    platform_limits: tuple[str, ...],
    env_names: tuple[str, ...],
    host_core: bool,
) -> list[str]:
    advisories: list[str] = []
    if tier == "unreviewed":
        advisories.append("unreviewed_tier")
    if removal_status == "deprecated":
        advisories.append("catalog_deprecation")
    if platform_limits:
        advisories.append("platform_limited")
    if env_names:
        advisories.append("requires_environment_names")
    if host_core:
        advisories.append("host_owned_mechanics")
    return sorted(advisories)


def _unavailable_evidence(
    *,
    coverage_class: str,
    host_version_status: str,
    platform_limits: tuple[str, ...],
) -> list[str]:
    evidence = set(BASELINE_UNAVAILABLE_EVIDENCE)
    if platform_limits:
        evidence.add("host_platform_match")
    if host_version_status == "unknown":
        evidence.add("host_version_evaluation")
    if coverage_class == "removed":
        evidence.add("installed_copy_state")
    return sorted(evidence)


def _packaged_reference_line(payload: Mapping[str, Any]) -> str:
    reference = payload["packaged_reference"]
    return (
        f"Packaged catalog {reference['catalog_revision']} is {reference['coverage_scope']} and was not "
        "used as host coverage."
    )


__all__ = [
    "ADOPTION_GUIDANCE",
    "BASELINE_UNAVAILABLE_EVIDENCE",
    "CAPABILITY_OWNERS",
    "CHANGE_CLASSES",
    "COVERAGE_CLAIM_BOUNDARY",
    "COVERAGE_CLASSES",
    "GENERIC_REVIEW_ROUTE",
    "HELD_REVIEW_ROUTE",
    "HOST_CORE_CAPABILITY",
    "PLUGIN_CATALOG_COVERAGE_SCHEMA_VERSION",
    "TRACKED_ENTRY_FIELDS",
    "PluginCatalogCoverageError",
    "build_plugin_catalog_coverage",
    "plugin_catalog_coverage_unavailable",
    "render_plugin_catalog_coverage",
]
