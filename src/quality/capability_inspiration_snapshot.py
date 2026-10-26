"""The evidence behind a native capability design, frozen so it can be re-asked.

A native OMH capability is often designed by reading somebody else's docs or
release. Months later two questions arrive and neither had an answer: "base the
OMH version on this exact release" -- which release was that? -- and "has the
inspiration changed?" -- changed from what? Source finding records a candidate
and how far its acquisition got; it records no revision, no content digest, and
no license observation, so there was nothing to compare a later reading against.

What this module freezes
------------------------

One snapshot per capability design, holding the sources that informed it (uri,
revision, a digest of the content as it read at that revision, the license note
the observer saw), the bounded findings drawn from them, and the requirements
derived for the OMH-native version. The snapshot carries who observed the
material and when, a reviewer reference once someone has reviewed it, and a link
to the snapshot it replaces.

OMH does not fetch
------------------

Every byte of observed material is supplied by the caller. This module imports
nothing that can open a socket and calls nothing that can, by construction: it
digests a string it was handed and records the revision string it was told. The
`claim_boundary` on every artifact says so, because the failure this guards
against is a reader treating a frozen digest as proof that OMH just checked the
source. It did not, and it cannot.

Content is digested, never stored. A snapshot is metadata about evidence; the
evidence itself belongs wherever the observer got it.

Why the digest excludes the clock
---------------------------------

`evidence_digest` is canonical over the observed material and nothing else. The
observation time, the observer, the reviewer, and the supersede link are all
excluded, so re-observing identical material at a later time reproduces the
identical digest -- which is the only way "has the inspiration changed?" can be
answered by comparing two numbers. A timestamp inside the seed would make every
re-observation look like a change.

The encoding is `skill_governance._stable_encode`, the one this tree already
uses for a canonical digest over a nested value. A second encoding scheme would
mean two definitions of "the same evidence" and eventually two answers.

Sources are sorted by `(uri, revision)` in the built snapshot, so the order they
were supplied in cannot change the artifact or its digest -- a set of observed
sources has no meaningful order. Findings and derived requirements keep the
order they were given: a requirement list reads as a list, and reordering it is
an authoring change the digest should notice.

Supersession, not rewriting
---------------------------

Changed evidence produces a new snapshot that links to the old one through
`supersedes_snapshot_ref`. Nothing is mutated and nothing is deleted, so the
evidence that actually shaped the shipped design stays readable after the source
moves on. `snapshot_id` is derived from `snapshot_digest`, which has a useful
consequence: a snapshot whose evidence did not change gets the same id, so
"supersede it with itself" is a self-link and the chain walk refuses it.

`capability_inspiration_chain_errors` is that walk, and it is deliberately not
`system.append_only_store.supersede_chain_errors`. That function reports faults
against a store path, and this family has no store: a chain here is a sequence a
caller holds. Passing it a fabricated `Path` to satisfy the signature would put
a separator-dependent string into every error line, which is precisely the shape
that breaks on Windows. The four refusals are the same four; when a later issue
gives these records a file, that caller should use the store walk.

Citation
--------

`capability_inspiration_citation` is the accessor other capability artifacts
bind to. It carries the digests, the capability it belongs to, and the counts --
never the source URIs, so a cited snapshot cannot read as "go look here now".
Its `source_availability` is pinned to the single member of
`SOURCE_AVAILABILITY_STATES`: this vocabulary has exactly one value on purpose,
so recording an observed availability is a visible widening of a named tuple
rather than a string someone typed.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from ..workflows.source_finder import (
    SOURCE_FINDER_OBSERVATION_PROVENANCE,
    normalize_observation_provenance,
)
from .skill_governance import _stable_encode


CAPABILITY_INSPIRATION_SNAPSHOT_SCHEMA_VERSION: Final = "capability_inspiration_snapshot/v1"
CAPABILITY_INSPIRATION_COMPARISON_SCHEMA_VERSION: Final = "capability_inspiration_comparison/v1"
CAPABILITY_INSPIRATION_CITATION_SCHEMA_VERSION: Final = "capability_inspiration_citation/v1"

# Only a full commit-shaped hex digest is treated as immutable evidence. A
# branch moves by design and a tag can be moved by hand, so both are `mutable`
# and render as visibly unstable evidence; a revision nobody supplied is
# `unknown` rather than either. Conservative on purpose: calling moving evidence
# immutable is the failure that makes a frozen snapshot lie.
REVISION_STABILITIES: Final[tuple[str, ...]] = ("immutable", "mutable", "unknown")

# The verdicts a comparison can reach. `unverifiable` is the honest answer when
# the source could not be re-observed at all -- unavailable, unreadable, or
# simply never supplied a second time -- and is never collapsed into `changed`.
COMPARISON_VERDICTS: Final[tuple[str, ...]] = ("changed", "current", "unverifiable")

# How one source differs between two snapshots. A source is matched across
# snapshots by its uri; everything else about it is a change to report.
SOURCE_CHANGE_KINDS: Final[tuple[str, ...]] = (
    "added",
    "content_changed",
    "license_changed",
    "removed",
    "revision_changed",
    "stability_changed",
)

# The whole vocabulary this family may use to describe whether a cited source
# can be reached right now. One member, and it is "not observed": OMH performs
# no fetch, so it has nothing else it could truthfully say.
SOURCE_AVAILABILITY_STATES: Final[tuple[str, ...]] = ("not_observed",)

# Every claim a snapshot or a citation does not make, named so a consumer can
# check the list instead of inferring the boundary from prose.
CAPABILITY_INSPIRATION_NOT_OBSERVED: Final[tuple[str, ...]] = (
    "source_availability",
    "source_reachability",
    "source_content_re_observation",
    "license_verification",
    "derived_requirement_review",
    "derived_requirement_implementation",
)

CAPABILITY_INSPIRATION_CLAIM_BOUNDARY: Final = (
    "This snapshot records evidence a caller supplied. OMH performed no fetch, download, "
    "clone, or network call to produce it and cannot perform one. The revision, content "
    "digest, license note, and observation time are what the observer reported, not what "
    "OMH verified. Freezing evidence is not a statement that the source is reachable, "
    "unchanged, or available now."
)

CAPABILITY_INSPIRATION_CITATION_CLAIM_BOUNDARY: Final = (
    "A cited snapshot is frozen evidence, not a live source check. Availability, "
    "reachability, and current content stay unobserved until a caller supplies a new "
    "observation and compares it. Citing a snapshot is not review, implementation, or "
    "verification of the requirements derived from it."
)

CAPABILITY_INSPIRATION_SNAPSHOT_KEYS: Final[tuple[str, ...]] = (
    "capability_id",
    "claim_boundary",
    "derived_requirements",
    "derived_requirements_digest",
    "evidence_digest",
    "findings",
    "not_evidence_until_observed",
    "observed_at",
    "observed_sources",
    "observer",
    "reviewer_ref",
    "schema_version",
    "snapshot_digest",
    "snapshot_id",
    "source_availability",
    "supersedes_snapshot_ref",
)

CAPABILITY_INSPIRATION_SOURCE_KEYS: Final[tuple[str, ...]] = (
    "content_digest",
    "license_note",
    "observation_provenance",
    "revision",
    "revision_stability",
    "source_id",
    "uri",
)

CAPABILITY_INSPIRATION_COMPARISON_KEYS: Final[tuple[str, ...]] = (
    "capability_id",
    "changed_sources",
    "claim_boundary",
    "derived_requirements_changed",
    "earlier_evidence_digest",
    "earlier_snapshot_id",
    "later_evidence_digest",
    "later_snapshot_id",
    "not_evidence_until_observed",
    "schema_version",
    "supersedes_earlier",
    "verdict",
)

CAPABILITY_INSPIRATION_CHANGED_SOURCE_KEYS: Final[tuple[str, ...]] = (
    "changes",
    "earlier_source_id",
    "later_source_id",
    "uri",
)

CAPABILITY_INSPIRATION_CITATION_KEYS: Final[tuple[str, ...]] = (
    "capability_id",
    "cited_source_count",
    "claim_boundary",
    "derived_requirements_digest",
    "evidence_digest",
    "not_evidence_until_observed",
    "observed_at",
    "observer",
    "reviewer_ref",
    "schema_version",
    "snapshot_digest",
    "snapshot_id",
    "source_availability",
)

# Bounded because the issue's design is bounded findings, and because an
# unbounded list turns a metadata artifact into a document store. Over the cap
# is refused rather than truncated: silently dropping the twenty-first finding
# would change the digest without telling anyone why.
MAX_OBSERVED_SOURCES: Final = 20
MAX_FINDINGS: Final = 20
MAX_DERIVED_REQUIREMENTS: Final = 20
MAX_LINE_LENGTH: Final = 200
MAX_URI_LENGTH: Final = 512

_HEX_DIGEST = re.compile(r"\A(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class CapabilityInspirationSnapshotError(ValueError):
    """A snapshot that cannot be built without asserting something untrue."""


def revision_stability(revision: str) -> str:
    """Whether a revision identifier names a fixed point or a moving one."""
    text = str(revision or "").strip()
    if not text:
        return "unknown"
    return "immutable" if _HEX_DIGEST.match(text) else "mutable"


def build_capability_inspiration_source(
    *,
    uri: str,
    revision: str = "",
    content: str = "",
    license_note: str = "",
    observation_provenance: str | None = None,
) -> dict[str, Any]:
    """One observed source, with its supplied content reduced to a digest.

    `content` is what the observer read at `revision`. It is hashed and dropped:
    the snapshot is metadata about evidence and never a copy of it. An empty
    `content` means no content was supplied, which is recorded as an empty
    digest rather than as the digest of an empty document.
    """
    safe_uri = str(uri or "").strip()
    if not safe_uri:
        raise CapabilityInspirationSnapshotError("capability_inspiration_snapshot source uri is required")
    if len(safe_uri) > MAX_URI_LENGTH:
        raise CapabilityInspirationSnapshotError(
            f"capability_inspiration_snapshot source uri exceeds {MAX_URI_LENGTH} characters"
        )
    safe_revision = _bounded_line(revision, field="source revision")
    return {
        "source_id": _source_id(safe_uri, safe_revision),
        "uri": safe_uri,
        "revision": safe_revision,
        "revision_stability": revision_stability(safe_revision),
        "content_digest": _content_digest(content),
        "license_note": _bounded_line(license_note, field="source license_note"),
        "observation_provenance": normalize_observation_provenance(observation_provenance),
    }


def build_capability_inspiration_snapshot(
    *,
    capability_id: str,
    observed_sources: Sequence[Mapping[str, Any]],
    observer: str,
    observed_at: str,
    findings: Sequence[str] = (),
    derived_requirements: Sequence[str] = (),
    reviewer_ref: str = "",
    supersedes_snapshot_ref: str = "",
) -> dict[str, Any]:
    """Freeze the evidence behind one native capability design.

    `observed_at` is a parameter and never a clock read here, so a caller can
    reproduce a snapshot exactly. It is excluded from every digest for the same
    reason.

    `observer` is required. A frozen artifact with nobody standing behind it
    cannot be weighed later against the evidence somebody else supplied, so a
    snapshot with no observer is refused rather than recorded as anonymous.
    """
    safe_capability = _bounded_line(capability_id, field="capability_id")
    if not safe_capability:
        raise CapabilityInspirationSnapshotError("capability_inspiration_snapshot capability_id is required")
    safe_observer = _bounded_line(observer, field="observer")
    if not safe_observer:
        raise CapabilityInspirationSnapshotError("capability_inspiration_snapshot observer is required")
    sources = _normalized_sources(observed_sources)
    if not sources:
        raise CapabilityInspirationSnapshotError(
            "capability_inspiration_snapshot requires at least one observed source"
        )
    safe_findings = _bounded_lines(findings, field="findings", limit=MAX_FINDINGS)
    safe_requirements = _bounded_lines(
        derived_requirements, field="derived_requirements", limit=MAX_DERIVED_REQUIREMENTS
    )
    evidence_digest = _digest({"sources": [_evidence_identity(source) for source in sources]})
    requirements_digest = _digest({"derived_requirements": safe_requirements})
    snapshot_digest = _digest(
        {
            "capability_id": safe_capability,
            "derived_requirements_digest": requirements_digest,
            "evidence_digest": evidence_digest,
            "findings": safe_findings,
        }
    )
    return {
        "schema_version": CAPABILITY_INSPIRATION_SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": f"capability-inspiration-{snapshot_digest[:16]}",
        "capability_id": safe_capability,
        "observed_sources": sources,
        "findings": safe_findings,
        "derived_requirements": safe_requirements,
        "evidence_digest": evidence_digest,
        "derived_requirements_digest": requirements_digest,
        "snapshot_digest": snapshot_digest,
        "observer": safe_observer,
        "observed_at": _bounded_line(observed_at, field="observed_at"),
        "reviewer_ref": _bounded_line(reviewer_ref, field="reviewer_ref"),
        "supersedes_snapshot_ref": _bounded_line(
            supersedes_snapshot_ref, field="supersedes_snapshot_ref"
        ),
        "source_availability": SOURCE_AVAILABILITY_STATES[0],
        "claim_boundary": CAPABILITY_INSPIRATION_CLAIM_BOUNDARY,
        "not_evidence_until_observed": list(CAPABILITY_INSPIRATION_NOT_OBSERVED),
    }


def compare_capability_inspiration_snapshots(
    earlier: Mapping[str, Any], later: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Whether the frozen evidence still describes what a later reading found.

    `later` is `None` when the sources could not be re-observed. That is
    `unverifiable` and never `current`: nothing was compared, so nothing can be
    reported as unchanged.

    Neither argument is modified. The earlier snapshot stays exactly as it was
    frozen, which is the whole point of superseding rather than rewriting.
    """
    earlier_sources = _source_index(earlier)
    if later is None:
        return _comparison(
            capability_id=str(earlier.get("capability_id", "")),
            verdict="unverifiable",
            earlier=earlier,
            later={},
            changed_sources=[],
            derived_requirements_changed=False,
        )
    later_sources = _source_index(later)
    changed_sources = _changed_sources(earlier_sources, later_sources)
    requirements_changed = earlier.get("derived_requirements_digest") != later.get(
        "derived_requirements_digest"
    )
    same_evidence = (
        bool(earlier.get("evidence_digest"))
        and earlier.get("evidence_digest") == later.get("evidence_digest")
    )
    return _comparison(
        capability_id=str(later.get("capability_id", "")),
        verdict="current" if same_evidence else "changed",
        earlier=earlier,
        later=later,
        changed_sources=changed_sources,
        derived_requirements_changed=bool(requirements_changed),
    )


def capability_inspiration_citation(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """The bindable reference other capability artifacts cite.

    Digests and counts only. The source URIs stay in the snapshot: a coding
    handoff citing this needs to name which evidence it was built on, not to
    invite its reader to go and fetch that evidence now.
    """
    sources = snapshot.get("observed_sources")
    return {
        "schema_version": CAPABILITY_INSPIRATION_CITATION_SCHEMA_VERSION,
        "snapshot_id": str(snapshot.get("snapshot_id", "")),
        "capability_id": str(snapshot.get("capability_id", "")),
        "snapshot_digest": str(snapshot.get("snapshot_digest", "")),
        "evidence_digest": str(snapshot.get("evidence_digest", "")),
        "derived_requirements_digest": str(snapshot.get("derived_requirements_digest", "")),
        "cited_source_count": len(sources) if isinstance(sources, list) else 0,
        "observer": str(snapshot.get("observer", "")),
        "observed_at": str(snapshot.get("observed_at", "")),
        "reviewer_ref": str(snapshot.get("reviewer_ref", "")),
        "source_availability": SOURCE_AVAILABILITY_STATES[0],
        "claim_boundary": CAPABILITY_INSPIRATION_CITATION_CLAIM_BOUNDARY,
        "not_evidence_until_observed": list(CAPABILITY_INSPIRATION_NOT_OBSERVED),
    }


def validate_capability_inspiration_snapshot(snapshot: Any) -> list[str]:
    """Every reason a payload is not a `capability_inspiration_snapshot/v1`."""
    label = "capability_inspiration_snapshot"
    if not isinstance(snapshot, Mapping):
        return [f"{label} must be an object"]
    errors = _key_set_errors(snapshot, CAPABILITY_INSPIRATION_SNAPSHOT_KEYS, label)
    if snapshot.get("schema_version") != CAPABILITY_INSPIRATION_SNAPSHOT_SCHEMA_VERSION:
        errors.append(f"{label} schema_version must be {CAPABILITY_INSPIRATION_SNAPSHOT_SCHEMA_VERSION}")
    for key in ("capability_id", "observer", "snapshot_id"):
        if not _non_empty_text(snapshot.get(key)):
            errors.append(f"{label} {key} must be a non-empty string")
    for key in ("observed_at", "reviewer_ref", "supersedes_snapshot_ref"):
        if not isinstance(snapshot.get(key), str):
            errors.append(f"{label} {key} must be a string")
    if snapshot.get("source_availability") not in SOURCE_AVAILABILITY_STATES:
        errors.append(f"{label} source_availability must be one of {list(SOURCE_AVAILABILITY_STATES)}")
    if snapshot.get("claim_boundary") != CAPABILITY_INSPIRATION_CLAIM_BOUNDARY:
        errors.append(f"{label} claim_boundary must state that OMH observed nothing itself")
    errors.extend(_not_observed_errors(snapshot, label))
    errors.extend(_bounded_list_errors(snapshot.get("findings"), field="findings", label=label, limit=MAX_FINDINGS))
    errors.extend(
        _bounded_list_errors(
            snapshot.get("derived_requirements"),
            field="derived_requirements",
            label=label,
            limit=MAX_DERIVED_REQUIREMENTS,
        )
    )
    errors.extend(_observed_source_errors(snapshot.get("observed_sources"), label))
    if str(snapshot.get("snapshot_id", "")) == str(snapshot.get("supersedes_snapshot_ref", "")) and str(
        snapshot.get("supersedes_snapshot_ref", "")
    ):
        errors.append(f"{label} supersedes_snapshot_ref must not name itself")
    if not errors:
        errors.extend(_digest_errors(snapshot, label))
    return errors


def validate_capability_inspiration_citation(citation: Any) -> list[str]:
    """Every reason a payload is not a `capability_inspiration_citation/v1`."""
    label = "capability_inspiration_citation"
    if not isinstance(citation, Mapping):
        return [f"{label} must be an object"]
    errors = _key_set_errors(citation, CAPABILITY_INSPIRATION_CITATION_KEYS, label)
    if citation.get("schema_version") != CAPABILITY_INSPIRATION_CITATION_SCHEMA_VERSION:
        errors.append(f"{label} schema_version must be {CAPABILITY_INSPIRATION_CITATION_SCHEMA_VERSION}")
    for key in ("capability_id", "observer", "snapshot_id", "snapshot_digest", "evidence_digest"):
        if not _non_empty_text(citation.get(key)):
            errors.append(f"{label} {key} must be a non-empty string")
    count = citation.get("cited_source_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        errors.append(f"{label} cited_source_count must be a positive integer")
    if citation.get("source_availability") not in SOURCE_AVAILABILITY_STATES:
        errors.append(f"{label} source_availability must be one of {list(SOURCE_AVAILABILITY_STATES)}")
    if citation.get("claim_boundary") != CAPABILITY_INSPIRATION_CITATION_CLAIM_BOUNDARY:
        errors.append(f"{label} claim_boundary must state that availability stays unobserved")
    errors.extend(_not_observed_errors(citation, label))
    return errors


def validate_capability_inspiration_comparison(comparison: Any) -> list[str]:
    """Every reason a payload is not a `capability_inspiration_comparison/v1`."""
    label = "capability_inspiration_comparison"
    if not isinstance(comparison, Mapping):
        return [f"{label} must be an object"]
    errors = _key_set_errors(comparison, CAPABILITY_INSPIRATION_COMPARISON_KEYS, label)
    if comparison.get("schema_version") != CAPABILITY_INSPIRATION_COMPARISON_SCHEMA_VERSION:
        errors.append(f"{label} schema_version must be {CAPABILITY_INSPIRATION_COMPARISON_SCHEMA_VERSION}")
    verdict = comparison.get("verdict")
    if verdict not in COMPARISON_VERDICTS:
        errors.append(f"{label} verdict must be one of {list(COMPARISON_VERDICTS)}")
    if not isinstance(comparison.get("derived_requirements_changed"), bool):
        errors.append(f"{label} derived_requirements_changed must be a boolean")
    if not isinstance(comparison.get("supersedes_earlier"), bool):
        errors.append(f"{label} supersedes_earlier must be a boolean")
    if comparison.get("claim_boundary") != CAPABILITY_INSPIRATION_CLAIM_BOUNDARY:
        errors.append(f"{label} claim_boundary must state that OMH observed nothing itself")
    errors.extend(_not_observed_errors(comparison, label))
    errors.extend(_changed_source_errors(comparison.get("changed_sources"), label))
    if verdict == "current" and comparison.get("changed_sources"):
        errors.append(f"{label} verdict current cannot name changed sources")
    if verdict == "unverifiable" and _non_empty_text(comparison.get("later_snapshot_id")):
        errors.append(f"{label} verdict unverifiable cannot name a later snapshot")
    return errors


def capability_inspiration_chain_errors(snapshots: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject every shape a supersede chain must not take.

    A chain is a line, not a graph: a snapshot cannot repeat an id, cannot
    supersede itself, cannot supersede a snapshot that does not already exist
    earlier in the sequence, and cannot supersede a snapshot something else
    already superseded. A fork means two writers each believe they replaced the
    same frozen evidence, which makes "the snapshot in force" ambiguous.
    """
    label = "capability_inspiration_snapshot"
    seen: set[str] = set()
    successors: dict[str, str] = {}
    errors: list[str] = []
    for index, snapshot in enumerate(snapshots):
        snapshot_id = str(snapshot.get("snapshot_id", ""))
        if snapshot_id and snapshot_id in seen:
            errors.append(f"{label}[{index}]: snapshot_id is not unique: {snapshot_id}")
        superseded = str(snapshot.get("supersedes_snapshot_ref", ""))
        if superseded:
            if superseded == snapshot_id:
                errors.append(
                    f"{label}[{index}]: supersedes_snapshot_ref must not name itself: {snapshot_id}"
                )
            elif superseded not in seen:
                errors.append(
                    f"{label}[{index}]: supersedes_snapshot_ref does not name an earlier "
                    f"snapshot: {superseded}"
                )
            elif superseded in successors:
                errors.append(
                    f"{label}[{index}]: supersedes_snapshot_ref forks the chain: {superseded} is "
                    f"already superseded by {successors[superseded]}"
                )
            else:
                successors[superseded] = snapshot_id
        seen.add(snapshot_id)
    return errors


def _comparison(
    *,
    capability_id: str,
    verdict: str,
    earlier: Mapping[str, Any],
    later: Mapping[str, Any],
    changed_sources: list[dict[str, Any]],
    derived_requirements_changed: bool,
) -> dict[str, Any]:
    return {
        "schema_version": CAPABILITY_INSPIRATION_COMPARISON_SCHEMA_VERSION,
        "capability_id": capability_id,
        "verdict": verdict,
        "earlier_snapshot_id": str(earlier.get("snapshot_id", "")),
        "earlier_evidence_digest": str(earlier.get("evidence_digest", "")),
        "later_snapshot_id": str(later.get("snapshot_id", "")),
        "later_evidence_digest": str(later.get("evidence_digest", "")),
        "changed_sources": changed_sources,
        "derived_requirements_changed": derived_requirements_changed,
        "supersedes_earlier": bool(
            later.get("supersedes_snapshot_ref")
            and later.get("supersedes_snapshot_ref") == earlier.get("snapshot_id")
        ),
        "claim_boundary": CAPABILITY_INSPIRATION_CLAIM_BOUNDARY,
        "not_evidence_until_observed": list(CAPABILITY_INSPIRATION_NOT_OBSERVED),
    }


def _source_index(snapshot: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    sources = snapshot.get("observed_sources")
    if not isinstance(sources, list):
        return {}
    return {
        str(source.get("uri", "")): source
        for source in sources
        if isinstance(source, Mapping) and str(source.get("uri", ""))
    }


def _changed_sources(
    earlier: Mapping[str, Mapping[str, Any]], later: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    changed: list[dict[str, Any]] = []
    for uri in sorted(set(earlier) | set(later)):
        before = earlier.get(uri)
        after = later.get(uri)
        if before is None:
            changes = ["added"]
        elif after is None:
            changes = ["removed"]
        else:
            changes = sorted(
                kind
                for field, kind in (
                    ("revision", "revision_changed"),
                    ("content_digest", "content_changed"),
                    ("license_note", "license_changed"),
                    ("revision_stability", "stability_changed"),
                )
                if before.get(field) != after.get(field)
            )
        if not changes:
            continue
        changed.append(
            {
                "uri": uri,
                "earlier_source_id": str(before.get("source_id", "")) if before else "",
                "later_source_id": str(after.get("source_id", "")) if after else "",
                "changes": changes,
            }
        )
    return changed


def _normalized_sources(observed_sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(observed_sources, (str, bytes)) or not isinstance(observed_sources, Sequence):
        raise CapabilityInspirationSnapshotError(
            "capability_inspiration_snapshot observed_sources must be a sequence of sources"
        )
    if len(observed_sources) > MAX_OBSERVED_SOURCES:
        raise CapabilityInspirationSnapshotError(
            f"capability_inspiration_snapshot observed_sources exceeds {MAX_OBSERVED_SOURCES} entries"
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in observed_sources:
        if not isinstance(source, Mapping):
            raise CapabilityInspirationSnapshotError(
                "capability_inspiration_snapshot observed source must be an object"
            )
        # Closed on the way in as well as on the way out. A caller handing this
        # a `content` key expects the content to be digested here, and it is
        # not: `build_capability_inspiration_source` already did that and
        # dropped it. Refusing the key is the only answer that is not a silent
        # loss of the evidence the caller thought it supplied.
        unsupported = sorted(set(source) - set(CAPABILITY_INSPIRATION_SOURCE_KEYS))
        if unsupported:
            raise CapabilityInspirationSnapshotError(
                f"capability_inspiration_snapshot observed source has unsupported keys: {unsupported}"
            )
        rebuilt = build_capability_inspiration_source(
            uri=str(source.get("uri", "")),
            revision=str(source.get("revision", "")),
            license_note=str(source.get("license_note", "")),
            observation_provenance=str(source.get("observation_provenance", "") or ""),
        )
        # A source arriving as a mapping has already had its content digested by
        # `build_capability_inspiration_source`; re-digesting a digest would
        # produce a different one. Carry the recorded digest through instead.
        rebuilt["content_digest"] = _carried_content_digest(source)
        if rebuilt["uri"] in seen:
            raise CapabilityInspirationSnapshotError(
                f"capability_inspiration_snapshot observed_sources repeats uri: {rebuilt['uri']}"
            )
        seen.add(str(rebuilt["uri"]))
        normalized.append(rebuilt)
    normalized.sort(key=lambda source: (str(source["uri"]), str(source["revision"])))
    return normalized


def _carried_content_digest(source: Mapping[str, Any]) -> str:
    digest = str(source.get("content_digest", "")).strip().lower()
    if not digest:
        return ""
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise CapabilityInspirationSnapshotError(
            "capability_inspiration_snapshot source content_digest must be a sha256 hex digest"
        )
    return digest


def _evidence_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    """The observed material only: no derived id, no observation time."""
    return {
        "content_digest": source.get("content_digest", ""),
        "license_note": source.get("license_note", ""),
        "observation_provenance": source.get("observation_provenance", ""),
        "revision": source.get("revision", ""),
        "revision_stability": source.get("revision_stability", ""),
        "uri": source.get("uri", ""),
    }


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_stable_encode(value).encode("utf-8")).hexdigest()


def _content_digest(content: str) -> str:
    text = str(content or "")
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_id(uri: str, revision: str) -> str:
    seed = _stable_encode({"revision": revision, "uri": uri})
    return f"inspiration-source-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"


def _bounded_line(value: Any, *, field: str) -> str:
    """One metadata line: whitespace collapsed, length capped, never wrapped.

    Collapsing whitespace also removes the newline a caller pasted in, which is
    what keeps a supplied value from arriving as `\\r\\n` on one platform and
    `\\n` on another and digesting differently.
    """
    text = " ".join(str(value or "").split())
    if len(text) > MAX_LINE_LENGTH:
        raise CapabilityInspirationSnapshotError(
            f"capability_inspiration_snapshot {field} exceeds {MAX_LINE_LENGTH} characters"
        )
    return text


def _bounded_lines(values: Sequence[str], *, field: str, limit: int) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CapabilityInspirationSnapshotError(
            f"capability_inspiration_snapshot {field} must be a sequence of strings"
        )
    lines: list[str] = []
    for value in values:
        line = _bounded_line(value, field=field)
        if line and line not in lines:
            lines.append(line)
    if len(lines) > limit:
        raise CapabilityInspirationSnapshotError(
            f"capability_inspiration_snapshot {field} exceeds {limit} entries"
        )
    return lines


def _key_set_errors(payload: Mapping[str, Any], keys: tuple[str, ...], label: str) -> list[str]:
    errors: list[str] = []
    extra = sorted(set(payload) - set(keys))
    if extra:
        errors.append(f"{label} has unsupported keys: {extra}")
    missing = sorted(set(keys) - set(payload))
    if missing:
        errors.append(f"{label} is missing keys: {missing}")
    return errors


def _not_observed_errors(payload: Mapping[str, Any], label: str) -> list[str]:
    declared = payload.get("not_evidence_until_observed")
    if not isinstance(declared, list):
        return [f"{label} not_evidence_until_observed must be a list"]
    if list(declared) != list(CAPABILITY_INSPIRATION_NOT_OBSERVED):
        return [
            f"{label} not_evidence_until_observed must be "
            f"{list(CAPABILITY_INSPIRATION_NOT_OBSERVED)}"
        ]
    return []


def _bounded_list_errors(values: Any, *, field: str, label: str, limit: int) -> list[str]:
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        return [f"{label} {field} must be a list of strings"]
    errors: list[str] = []
    if len(values) > limit:
        errors.append(f"{label} {field} exceeds {limit} entries")
    if any(not value or len(value) > MAX_LINE_LENGTH for value in values):
        errors.append(f"{label} {field} entries must be non-empty and at most {MAX_LINE_LENGTH} characters")
    if len(set(values)) != len(values):
        errors.append(f"{label} {field} must not repeat an entry")
    return errors


def _observed_source_errors(sources: Any, label: str) -> list[str]:
    if not isinstance(sources, list) or not sources:
        return [f"{label} observed_sources must be a non-empty list"]
    errors: list[str] = []
    if len(sources) > MAX_OBSERVED_SOURCES:
        errors.append(f"{label} observed_sources exceeds {MAX_OBSERVED_SOURCES} entries")
    order: list[tuple[str, str]] = []
    for source in sources:
        if not isinstance(source, Mapping):
            errors.append(f"{label} observed source must be an object")
            continue
        errors.extend(_key_set_errors(source, CAPABILITY_INSPIRATION_SOURCE_KEYS, f"{label} observed source"))
        if not _non_empty_text(source.get("uri")):
            errors.append(f"{label} observed source uri must be a non-empty string")
        if source.get("revision_stability") not in REVISION_STABILITIES:
            errors.append(f"{label} observed source revision_stability must be one of {list(REVISION_STABILITIES)}")
        elif source.get("revision_stability") != revision_stability(str(source.get("revision", ""))):
            errors.append(f"{label} observed source revision_stability does not match its revision")
        if source.get("observation_provenance") not in SOURCE_FINDER_OBSERVATION_PROVENANCE:
            errors.append(f"{label} observed source observation_provenance is unsupported")
        digest = source.get("content_digest")
        if not isinstance(digest, str) or (digest and not re.fullmatch(r"[0-9a-f]{64}", digest)):
            errors.append(f"{label} observed source content_digest must be empty or a sha256 hex digest")
        expected_id = _source_id(str(source.get("uri", "")), str(source.get("revision", "")))
        if source.get("source_id") != expected_id:
            errors.append(f"{label} observed source source_id does not match its uri and revision")
        order.append((str(source.get("uri", "")), str(source.get("revision", ""))))
    if order != sorted(order):
        errors.append(f"{label} observed_sources must be sorted by (uri, revision)")
    if len({uri for uri, _ in order}) != len(order):
        errors.append(f"{label} observed_sources must not repeat a uri")
    return errors


def _changed_source_errors(changed: Any, label: str) -> list[str]:
    if not isinstance(changed, list):
        return [f"{label} changed_sources must be a list"]
    errors: list[str] = []
    order: list[str] = []
    for entry in changed:
        if not isinstance(entry, Mapping):
            errors.append(f"{label} changed source must be an object")
            continue
        errors.extend(_key_set_errors(entry, CAPABILITY_INSPIRATION_CHANGED_SOURCE_KEYS, f"{label} changed source"))
        if not _non_empty_text(entry.get("uri")):
            errors.append(f"{label} changed source uri must be a non-empty string")
        kinds = entry.get("changes")
        if not isinstance(kinds, list) or not kinds or not all(kind in SOURCE_CHANGE_KINDS for kind in kinds):
            errors.append(f"{label} changed source changes must name one or more of {list(SOURCE_CHANGE_KINDS)}")
        elif list(kinds) != sorted(set(kinds)):
            errors.append(f"{label} changed source changes must be sorted and distinct")
        order.append(str(entry.get("uri", "")))
    if order != sorted(order):
        errors.append(f"{label} changed_sources must be sorted by uri")
    return errors


def _digest_errors(snapshot: Mapping[str, Any], label: str) -> list[str]:
    """Re-derive every digest from the payload's own material.

    A snapshot whose digest does not follow from what it records is not a
    frozen artifact, it is two claims that disagree.
    """
    sources = [source for source in snapshot["observed_sources"] if isinstance(source, Mapping)]
    evidence_digest = _digest({"sources": [_evidence_identity(source) for source in sources]})
    requirements_digest = _digest({"derived_requirements": list(snapshot["derived_requirements"])})
    snapshot_digest = _digest(
        {
            "capability_id": str(snapshot["capability_id"]),
            "derived_requirements_digest": requirements_digest,
            "evidence_digest": evidence_digest,
            "findings": list(snapshot["findings"]),
        }
    )
    errors: list[str] = []
    if snapshot.get("evidence_digest") != evidence_digest:
        errors.append(f"{label} evidence_digest does not match its observed sources")
    if snapshot.get("derived_requirements_digest") != requirements_digest:
        errors.append(f"{label} derived_requirements_digest does not match its derived requirements")
    if snapshot.get("snapshot_digest") != snapshot_digest:
        errors.append(f"{label} snapshot_digest does not match its frozen material")
    elif snapshot.get("snapshot_id") != f"capability-inspiration-{snapshot_digest[:16]}":
        errors.append(f"{label} snapshot_id does not match its snapshot_digest")
    return errors


def _non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
