"""Task-scoped capability projection (`omh_capability_projection/v1`).

Every capability payload OMH had before this module was a whole-catalog
snapshot: `capability_snapshot()` and `capability_summary()` answer "what can
this install do at all?", which is the right answer to a discovery question and
the wrong answer to a request. A request that needs two capabilities was still
handed all of them, so the inventory ate context, diluted the routing signal,
and put authority the task never asked for in front of the model.

This module answers the other question -- "what does THIS request need?" -- as
a view over the same catalog:

- Selection reuses the router. `recommend_skills` already ranks the catalog
  against a message; this module takes that shortlist and the capability-family
  mapping rather than inventing a second ranking, so a projection can never
  disagree with the route the user is about to be offered.
- Two levels. The projection carries a compact summary per relevant capability.
  Exact detail is a separate `expand_capability` call, never produced as a side
  effect of projecting.
- Budgeted. Content size is measured against `omh_run_context_budget/v1`, the
  same ledger the runtime observe surfaces spend from. Over budget, the
  projection degrades through `degrade_capability_projection_payload` and names
  every capability it dropped.
- Closed exclusion vocabulary. Every offered capability that is not included is
  accounted for by one of `EXCLUSION_REASON_CODES`, and an unrecognized reason
  is refused rather than passed through.

Authority is a frozen input, never an output. `CapabilityAuthority` is what an
approval covered; a projection is a VIEW over it and can only intersect. The
structural guarantee is in the shape of the API, not in a rule a caller must
remember:

- `CapabilityAuthority` is a frozen dataclass holding a `frozenset`, so the
  granted set cannot be edited in place.
- `approve_capability_authority` is the only function that builds one.
- `refresh_capability_projection` takes no authority parameter at all. It reads
  the frozen grant off the projection it is refreshing, so a refresh against a
  larger catalog has no seam through which a new capability could enter the
  grant -- it appears as an exclusion carrying `not_granted_by_authority`.

Determinism: nothing here reads a clock, the environment, or the filesystem.
The context budget arrives as a parameter, exactly like a time would. Note also
that none of this work belongs behind the `@lru_cache` on
`registry._capability_summary_template()`: that cache advertises
`"determinism": "static_projection_no_runtime_clock"` and is keyed on nothing,
so a per-request result cached there would be served to the next, different
request. Per-request work stays in this module, outside that cache; the only
cache it relies on is `recommend_skills`, which is keyed on the query itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
from typing import Any, Iterable

from ..routing.recommend import recommend_skills
from ..runtime.context_budget import degrade_capability_projection_payload, public_budget
from ..skills.catalog import installable_skill_names
from .families import capability_family_projection
from .registry import inspect_capability

CAPABILITY_AUTHORITY_SCHEMA_VERSION = "omh_capability_authority/v1"
CAPABILITY_PROJECTION_SCHEMA_VERSION = "omh_capability_projection/v1"
CAPABILITY_EXPANSION_SCHEMA_VERSION = "omh_capability_expansion/v1"

# Closed vocabulary. A projection that cannot name why a capability is absent
# is a projection that hides authority, so there is no "other" member and
# `capability_exclusion` refuses anything outside this tuple.
EXCLUSION_REASON_CODES = (
    "beyond_context_budget",
    "not_granted_by_authority",
    "not_relevant_to_request",
    "outranked_by_shortlist",
)

_EXCLUSION_EXPLANATIONS = {
    "beyond_context_budget": (
        "Relevant and granted, but dropped so the projection stays inside the declared context budget."
    ),
    "not_granted_by_authority": (
        "Offered by the catalog but outside the approved authority for this task; approving it again is "
        "the only way in."
    ),
    "not_relevant_to_request": "The router scored no trigger, phrase, or metadata match against this request.",
    "outranked_by_shortlist": "Matched the request, but ranked below the capabilities that were included.",
}

# How far down the router's ranking a capability still counts as "considered".
# Only considered capabilities are itemized in `exclusions`; everything below is
# counted in `exclusion_summary` and never named. That split is the whole point
# of the feature: naming ninety irrelevant workflows to explain why they were
# excluded is the tool-list dump the projection exists to avoid.
CONSIDERED_LIMIT = 12
DEFAULT_PROJECTION_LIMIT = 5

PROJECTION_CLAIM_BOUNDARY = (
    "A capability projection is a task-scoped view over the approved authority. It grants nothing, "
    "widens nothing, and is not execution, review, CI, merge-readiness, or merge evidence."
)
AUTHORITY_CLAIM_BOUNDARY = (
    "An approved capability authority records what a task was allowed to see. Re-projecting it can only "
    "intersect; a projection never writes authority."
)
EXPANSION_CLAIM_BOUNDARY = (
    "Expanded capability detail is catalog metadata returned because it was explicitly requested. It is "
    "not execution, review, CI, merge-readiness, or merge evidence."
)

EXPANSION_POLICY = "explicit_request_only"


class CapabilityProjectionError(ValueError):
    """Raised for an unknown exclusion reason, an unapproved expansion, or a changed grant."""


@dataclass(frozen=True)
class CapabilityAuthority:
    """What one task's approval covered, frozen at approval time.

    Frozen dataclass plus a `frozenset` on purpose: neither the record nor the
    granted set can be edited after approval, so "the projection widened the
    grant" is not a bug that can be written, only a constructor call that would
    have to be added.
    """

    task_id: str
    permission_profile: str
    envelope_digest: str
    granted: frozenset[str]
    granted_families: tuple[str, ...]

    def permits(self, capability: str) -> bool:
        return str(capability) in self.granted

    def granted_capabilities(self) -> tuple[str, ...]:
        return tuple(sorted(self.granted))

    @property
    def digest(self) -> str:
        """Stable identity for this grant; the value a caller re-approves against."""
        encoded = "\x1f".join(
            [
                CAPABILITY_AUTHORITY_SCHEMA_VERSION,
                self.task_id,
                self.permission_profile,
                self.envelope_digest,
                *self.granted_capabilities(),
            ]
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Emit the grant identity, never the grant contents.

        The full granted list is the whole-catalog inventory this feature exists
        to keep out of context. A reader gets the count, the families, and the
        digest to compare against; `omh capabilities list` still returns the
        inventory when someone actually wants it.
        """
        return {
            "schema_version": CAPABILITY_AUTHORITY_SCHEMA_VERSION,
            "task_id": self.task_id,
            "permission_profile": self.permission_profile,
            "envelope_digest": self.envelope_digest,
            "digest": self.digest,
            "granted_capability_count": len(self.granted),
            "granted_families": list(self.granted_families),
            "frozen": True,
            "claim_boundary": AUTHORITY_CLAIM_BOUNDARY,
        }


@dataclass(frozen=True)
class CapabilityProjection:
    """One task-scoped view. `authority` is the approval it was built against."""

    authority: CapabilityAuthority
    request: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.payload)

    def included_capabilities(self) -> tuple[str, ...]:
        included = self.payload.get("included")
        if not isinstance(included, list):
            return ()
        return tuple(str(entry.get("capability", "")) for entry in included if isinstance(entry, dict))


def capability_exclusion(capability: str, reason_code: str, explanation: str = "") -> dict[str, str]:
    """Build one exclusion entry, refusing any reason outside the closed vocabulary."""
    if reason_code not in EXCLUSION_REASON_CODES:
        raise CapabilityProjectionError(
            f"unsupported capability exclusion reason: {reason_code}; "
            f"expected one of {', '.join(EXCLUSION_REASON_CODES)}"
        )
    return {
        "capability": str(capability),
        "reason_code": reason_code,
        "explanation": str(explanation or _EXCLUSION_EXPLANATIONS[reason_code]),
    }


def approve_capability_authority(
    *,
    task_id: str,
    granted_capabilities: Iterable[str],
    permission_profile: str = "",
    authority_envelope: dict[str, Any] | None = None,
) -> CapabilityAuthority:
    """Freeze one task's approved capability grant.

    The only constructor of `CapabilityAuthority`. When an approved
    `task_authority_envelope/v1` is supplied its permission profile wins and its
    digest is recorded, so a later envelope change shows up as a different
    authority digest instead of being absorbed silently.
    """
    resolved_profile = str(permission_profile or "")
    envelope_digest = ""
    if authority_envelope is not None:
        envelope_profile, envelope_digest = _envelope_identity(authority_envelope)
        if resolved_profile and resolved_profile != envelope_profile:
            raise CapabilityProjectionError(
                "permission_profile disagrees with the supplied task_authority_envelope; "
                f"envelope says {envelope_profile!r} and the caller says {resolved_profile!r}"
            )
        resolved_profile = envelope_profile
    if resolved_profile and resolved_profile not in _permission_profiles():
        raise CapabilityProjectionError(
            f"unsupported permission profile: {resolved_profile}; "
            f"expected one of {', '.join(_permission_profiles())}"
        )
    granted = frozenset(str(item) for item in granted_capabilities if str(item))
    mapping = _workflow_to_family()
    families = tuple(sorted({mapping[name] for name in granted if name in mapping}))
    return CapabilityAuthority(
        task_id=str(task_id),
        permission_profile=resolved_profile,
        envelope_digest=envelope_digest,
        granted=granted,
        granted_families=families,
    )


def project_capabilities(
    request: str,
    *,
    authority: CapabilityAuthority,
    budget: dict[str, Any],
    offered_capabilities: Iterable[str] | None = None,
    limit: int = DEFAULT_PROJECTION_LIMIT,
) -> CapabilityProjection:
    """Build the task-scoped view of `offered_capabilities` for one request.

    `budget` is an `omh_run_context_budget/v1` payload supplied by the caller --
    a parameter for the same reason a clock reading would be, so the projection
    stays reproducible and testable without a filesystem.
    """
    if limit < 1:
        raise CapabilityProjectionError("capability projection limit must be at least 1")

    offered = _offered_set(offered_capabilities)
    ranked = _ranked_candidates(request, offered)
    considered = tuple(name for name, _score, _entry in ranked)
    granted_candidates = [item for item in ranked if authority.permits(item[0])]

    families = _workflow_to_family()
    included = [_compact_summary(entry, families) for _name, _score, entry in granted_candidates[:limit]]
    dropped: list[dict[str, str]] = []

    payload = _projection_payload(
        request=request,
        authority=authority,
        offered=offered,
        considered=considered,
        included=included,
        dropped=dropped,
        budget=budget,
    )
    remaining = max(0, int(budget.get("remaining_bytes", 0) or 0))
    content_bytes = int(payload["projected_bytes"])

    # Trim the lowest-ranked capability at a time, recording each drop, until
    # the content fits. A drop is never silent: it is itemized with
    # `beyond_context_budget` and counted in `budget_drop`.
    while included and content_bytes > remaining:
        removed = included.pop()
        dropped.append(capability_exclusion(str(removed["capability"]), "beyond_context_budget"))
        payload = _projection_payload(
            request=request,
            authority=authority,
            offered=offered,
            considered=considered,
            included=included,
            dropped=dropped,
            budget=budget,
        )
        content_bytes = int(payload["projected_bytes"])

    if content_bytes > remaining:
        # Nothing fits, so fall through to the same degrade contract the runtime
        # observe surfaces use: summary only, named drops, pointer to the
        # command that returns the full view.
        payload = degrade_capability_projection_payload(
            payload,
            budget,
            dropped=dropped,
            content_bytes=content_bytes,
        )

    return CapabilityProjection(authority=authority, request=str(request), payload=payload)


def refresh_capability_projection(
    projection: CapabilityProjection,
    *,
    budget: dict[str, Any],
    offered_capabilities: Iterable[str] | None = None,
    limit: int = DEFAULT_PROJECTION_LIMIT,
    request: str | None = None,
) -> CapabilityProjection:
    """Re-project against a possibly-changed catalog, reusing the frozen grant.

    There is deliberately no authority parameter. The grant comes off the
    projection being refreshed and is passed through by identity, so a catalog
    that grew since approval cannot widen what the task may see -- the new
    capability is reported as an exclusion carrying `not_granted_by_authority`.
    """
    return project_capabilities(
        projection.request if request is None else str(request),
        authority=projection.authority,
        budget=budget,
        offered_capabilities=offered_capabilities,
        limit=limit,
    )


def expand_capability(projection: CapabilityProjection, capability: str) -> dict[str, Any]:
    """Return exact detail for one projected capability, on explicit request only.

    Nothing in `project_capabilities` calls this. Expansion is a second, opt-in
    round trip; that is what keeps the compact level compact.
    """
    name = str(capability or "").strip()
    if not name:
        raise CapabilityProjectionError("capability expansion requires a capability id")
    if not projection.authority.permits(name):
        raise CapabilityProjectionError(
            f"capability {name} is outside the approved authority for task {projection.authority.task_id}"
        )
    if name not in projection.included_capabilities():
        raise CapabilityProjectionError(
            f"capability {name} is not in this projection; project a request that needs it before expanding"
        )
    inspected = inspect_capability(name, section="skills")
    return {
        "schema_version": CAPABILITY_EXPANSION_SCHEMA_VERSION,
        "task_id": projection.authority.task_id,
        "authority_digest": projection.authority.digest,
        "capability": name,
        "family": _workflow_to_family().get(name, ""),
        "requested": True,
        "automatic": False,
        "detail": inspected["capability"],
        "claim_boundary": EXPANSION_CLAIM_BOUNDARY,
    }


def authority_change_report(approved_digest: str, current: CapabilityAuthority) -> dict[str, Any]:
    """Compare a re-derived grant against the digest a caller approved.

    The digest is the whole point: an install whose capability policy changed
    between two projections produces a different grant, and this makes that
    visible as a refusal instead of a quietly different view. It deliberately
    does not name what changed -- a digest is all the caller held, and naming
    the delta would re-emit the grant contents this feature keeps out of
    context. `omh capability-policy status` is where that question belongs.
    """
    expected = str(approved_digest or "")
    matches = expected == current.digest
    return {
        "schema_version": "omh_capability_authority_change/v1",
        "task_id": current.task_id,
        "approved_digest": expected,
        "current_digest": current.digest,
        "unchanged": matches,
        "next_action": "reuse_projection" if matches else "re_approve_capability_authority",
        "compare_command": "omh capability-policy status",
        "claim_boundary": AUTHORITY_CLAIM_BOUNDARY,
    }


def _projection_payload(
    *,
    request: str,
    authority: CapabilityAuthority,
    offered: frozenset[str],
    considered: tuple[str, ...],
    included: list[dict[str, Any]],
    dropped: list[dict[str, str]],
    budget: dict[str, Any],
) -> dict[str, Any]:
    included_names = {str(entry["capability"]) for entry in included}
    dropped_names = {str(entry["capability"]) for entry in dropped}
    considered_set = set(considered)

    summary: dict[str, int] = {code: 0 for code in EXCLUSION_REASON_CODES}
    itemized: list[dict[str, str]] = list(dropped)
    for name in sorted(offered - included_names):
        reason = _exclusion_reason(
            name,
            authority=authority,
            dropped_names=dropped_names,
            considered=considered_set,
        )
        summary[reason] += 1
        if reason != "beyond_context_budget" and name in considered_set:
            itemized.append(capability_exclusion(name, reason))

    content = {
        "included": included,
        "exclusions": itemized,
        "exclusion_summary": summary,
    }
    payload: dict[str, Any] = {
        "schema_version": CAPABILITY_PROJECTION_SCHEMA_VERSION,
        "determinism": "static_projection_no_runtime_clock",
        "task_id": authority.task_id,
        "request_digest": request_digest(request),
        "authority": authority.to_dict(),
        "included": included,
        "included_count": len(included),
        "offered_count": len(offered),
        "excluded_count": len(offered) - len(included),
        "exclusions": itemized,
        "exclusion_summary": summary,
        "exclusion_reason_vocabulary": list(EXCLUSION_REASON_CODES),
        "expansion": {
            "policy": EXPANSION_POLICY,
            "automatic": False,
            "expandable": sorted(included_names),
            "command_template": "omh capabilities project <request> --expand <capability>",
        },
        # Measured over the content block only. Including the envelope would
        # make the number self-referential: writing it into the payload would
        # change the payload it measures.
        "projected_bytes": _content_bytes(content),
        "context_budget": public_budget(budget),
        "degraded": False,
        "claim_boundary": PROJECTION_CLAIM_BOUNDARY,
    }
    if dropped:
        payload["budget_drop"] = {
            "dropped_capabilities": [str(entry["capability"]) for entry in dropped],
            "dropped_count": len(dropped),
            "projected_bytes": payload["projected_bytes"],
            "budget_bytes": int(budget.get("budget_bytes", 0) or 0),
            "remaining_bytes": max(0, int(budget.get("remaining_bytes", 0) or 0)),
        }
    return payload


def _exclusion_reason(
    name: str,
    *,
    authority: CapabilityAuthority,
    dropped_names: set[str],
    considered: set[str],
) -> str:
    if name in dropped_names:
        return "beyond_context_budget"
    if not authority.permits(name):
        return "not_granted_by_authority"
    if name in considered:
        return "outranked_by_shortlist"
    return "not_relevant_to_request"


def _ranked_candidates(request: str, offered: frozenset[str]) -> list[tuple[str, int, dict[str, Any]]]:
    """The router's own shortlist, filtered to what this install offers.

    `recommend_skills` is the selection algorithm. This module adds no ranking
    of its own so a projection and a route cannot disagree.
    """
    ranked: list[tuple[str, int, dict[str, Any]]] = []
    for entry in recommend_skills(str(request), limit=CONSIDERED_LIMIT):
        name = str(entry.get("skill", ""))
        if name not in offered:
            continue
        ranked.append((name, int(entry.get("score", 0) or 0), entry))
    return ranked


def _compact_summary(entry: dict[str, Any], families: dict[str, str]) -> dict[str, Any]:
    """One capability at the discovery level: enough to choose, not to execute."""
    name = str(entry.get("skill", ""))
    return {
        "capability": name,
        "family": families.get(name, ""),
        "summary": str(entry.get("description", "")),
        "next_action": str(entry.get("next_action", "")),
        "match_reason": str(entry.get("why", "")),
        "score": int(entry.get("score", 0) or 0),
    }


def request_digest(request: str) -> str:
    """Reproducible identity for one request; also the default task-id source."""
    return hashlib.sha256(str(request).encode("utf-8")).hexdigest()


def _content_bytes(content: dict[str, Any]) -> int:
    return len(json.dumps(content, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def _offered_set(offered_capabilities: Iterable[str] | None) -> frozenset[str]:
    if offered_capabilities is None:
        return frozenset(installable_skill_names())
    return frozenset(str(item) for item in offered_capabilities if str(item))


def _workflow_to_family() -> dict[str, str]:
    mapping = capability_family_projection().get("workflow_to_family", {})
    if not isinstance(mapping, dict):
        return {}
    return {str(key): str(value) for key, value in mapping.items()}


def _envelope_identity(envelope: dict[str, Any]) -> tuple[str, str]:
    """Read the permission profile and a stable digest off an approved envelope."""
    if not isinstance(envelope, dict):
        raise CapabilityProjectionError("authority_envelope must be a task_authority_envelope/v1 object")
    if str(envelope.get("schema_version", "")) != "task_authority_envelope/v1":
        raise CapabilityProjectionError("authority_envelope must be a task_authority_envelope/v1 object")
    profile = str(envelope.get("permission_profile", ""))
    digest = hashlib.sha256(
        json.dumps(envelope, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return profile, digest


def _permission_profiles() -> tuple[str, ...]:
    """Imported lazily, matching `action_gate`, to keep this module import-light."""
    from ..workflows.goal_loop import PERMISSION_PROFILES

    return PERMISSION_PROFILES
