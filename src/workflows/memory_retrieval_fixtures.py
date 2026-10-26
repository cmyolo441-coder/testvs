"""Retained retrieval fixtures for the project-memory recall regression suite.

This module is data plus one materializer. It holds no ranking, eligibility, or
budget logic: every expectation below is a recorded verdict about what the
production recall-pack builder must return, never a second implementation of
how it decides.

Two properties make the corpus a regression gate rather than a snapshot:

- The clock is part of the fixture. Every timestamp is absolute and the case
  clock is ``FIXTURE_CLOCK``, so a case that depends on expiry, review
  deadlines, or age tiers reads the same on any host on any day. No fixture
  here reads the wall clock, directly or through a default argument.
- Record order is the order written here, and the store reader sorts by file
  name, so neither dict iteration nor filesystem order can reorder a pack.

Each case declares the records, the pins, the delivery counters, the query,
the scope and perspective lens, the budgets, and the expected disposition of
every record: included in this order, excluded for this named reason, or
absent because a lens filtered it before the pack could name it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from ..plugin_bundle.omh.memory_governance import (
    MEMORY_GOVERNANCE_POLICY_VERSION,
    PROJECT_MEMORY_RECORD_SCHEMA_VERSION,
    PROJECT_MEMORY_REVIEW_RECORD_SCHEMA_VERSION,
    canonical_payload_digest,
    stable_artifact_identity,
)

FIXTURE_CORPUS_VERSION = "omh-memory-retrieval-fixtures/v5"
FIXTURE_PROJECT_IDENTITY = "repo:" + hashlib.sha256(b"example.invalid/memory-fixture").hexdigest()[:32]
# The corpus clock. Absolute, declared once, never derived from the host.
FIXTURE_CLOCK = datetime(2031, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
FIXTURE_CLOCK_ISO = "2031-01-02T03:04:05Z"

_RECORD_CLAIM_BOUNDARY = (
    "Reviewed OMH project memory is prepared context only; it is not execution, "
    "review, CI, merge, or Hermes internal-memory evidence."
)
_REVIEW_CLAIM_BOUNDARY = (
    "Project memory review decisions are prepared governance only, never executor-use evidence."
)
# Contamination classes label a record that a lens, a deadline, or a tier is
# supposed to keep out of the pack. A labelled record appearing in a pack that
# did not expect it increments the counter named here, so a leak is reported
# under its own name instead of as a generic relevance miss.
CONTAMINATION_CLASSES = ("foreign_scope", "foreign_perspective", "stale", "expired", "archived")


def _record(
    record_id: str,
    summary: str,
    *,
    approved_at: str,
    tags: tuple[str, ...] = (),
    record_type: str = "fact",
    scope_kind: str = "project",
    scope_ref: str = FIXTURE_PROJECT_IDENTITY,
    observer: str = "",
    observed: str = "",
    retention_class: str = "standard",
    expires_at: str = "",
    review_due_at: str = "",
    superseded_by: str = "",
    attention_tier: str = "active",
    admission_state: str = "approved_manual",
    contamination_class: str = "",
    source: str = "",
    subject: str = "",
) -> dict[str, object]:
    """One fixture record specification, in the retained corpus's own shape.

    ``source`` names the admission channel the stored record carries; a case
    about recovering from one wrong source needs records that differ on it.
    ``subject`` is the model-declared subject slug a case about same-subject
    supersession needs. Each is emitted only when set, so a spec that does not
    care stays byte-identical.
    """
    if contamination_class and contamination_class not in CONTAMINATION_CLASSES:
        raise ValueError(f"unsupported contamination class: {contamination_class}")
    return {
        **({"source": source} if source else {}),
        **({"subject": subject} if subject else {}),
        "record_id": record_id,
        "summary": summary,
        "approved_at": approved_at,
        "tags": list(tags),
        "record_type": record_type,
        "scope_kind": scope_kind,
        "scope_ref": scope_ref,
        "observer": observer,
        "observed": observed,
        "retention_class": retention_class,
        "expires_at": expires_at,
        "review_due_at": review_due_at,
        "superseded_by": superseded_by,
        "attention_tier": attention_tier,
        "admission_state": admission_state,
        "contamination_class": contamination_class,
    }


def materialize_fixture_record(spec: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    """Turn one fixture specification into a stored record and its review.

    The pair is written straight into an isolated fixture store rather than
    produced through capture and approval, because approval stamps the record
    with ``utc_now()`` and mints a random record id. Both would make the pack
    depend on the day the suite ran and on the bytes of ``os.urandom``.
    """
    approved_at = str(spec["approved_at"])
    record_id = str(spec["record_id"])
    review_id = f"review_{record_id}"
    retention: dict[str, object] = {"class": str(spec["retention_class"]), "admitted_at": approved_at}
    expires_at = str(spec["expires_at"])
    if expires_at:
        retention["expires_at"] = expires_at
    review_due_at = str(spec["review_due_at"])
    tags = spec["tags"] if isinstance(spec.get("tags"), list) else []
    record: dict[str, object] = {
        "schema_version": PROJECT_MEMORY_RECORD_SCHEMA_VERSION,
        "record_id": record_id,
        "candidate_id": f"cand_{record_id}",
        "revision": 1,
        "record_type": str(spec["record_type"]),
        "summary": str(spec["summary"]),
        "scope": {"kind": str(spec["scope_kind"]), "ref": str(spec["scope_ref"])},
        "tags": [str(tag) for tag in tags],
        "source": str(spec.get("source") or "retrieval_fixture"),
        "source_class": "omh_local",
        "source_ref": "",
        "derived_from": [],
        "admission": {
            "state": str(spec["admission_state"]),
            "review_id": review_id,
            "reviewer_claim": "operator",
            "admitted_at": approved_at,
            "policy_version": MEMORY_GOVERNANCE_POLICY_VERSION,
        },
        "retention": retention,
        "revalidation": {"deadline": review_due_at} if review_due_at else {},
        "approved_at": approved_at,
        "created_at": approved_at,
        "updated_at": approved_at,
        "ttl": {"expires_at": expires_at, "ttl_days": None},
        "staleness": {"stale_after": review_due_at, "review_due_at": review_due_at, "stale_after_days": None},
        "attention": {
            "schema_version": "omh_memory_attention/v1",
            "tier": str(spec["attention_tier"]),
            "reason": "retrieval_fixture",
            "previous_tier": "",
            "changed_at": approved_at,
        },
        "safety": {},
        "redaction_policy": "metadata_only",
        "claim_boundary": _RECORD_CLAIM_BOUNDARY,
    }
    if str(spec["observed"]):
        record["perspective"] = {"observer": str(spec["observer"]), "observed": str(spec["observed"])}
    if str(spec["superseded_by"]):
        record["superseded_by"] = str(spec["superseded_by"])
    if str(spec.get("subject", "")):
        record["subject"] = str(spec["subject"])
    digest = canonical_payload_digest(record)
    admission = record["admission"]
    if isinstance(admission, dict):
        admission["payload_digest"] = digest
    review = {
        "schema_version": PROJECT_MEMORY_REVIEW_RECORD_SCHEMA_VERSION,
        "review_id": review_id,
        "artifact_identity": stable_artifact_identity(record),
        "decision": str(spec["admission_state"]),
        "reviewer_claim": "operator",
        "payload_digest": digest,
        "policy_version": MEMORY_GOVERNANCE_POLICY_VERSION,
        "reviewed_at": approved_at,
        "claim_boundary": _REVIEW_CLAIM_BOUNDARY,
    }
    return record, review


def _case(
    case_id: str,
    intent: str,
    *,
    records: tuple[dict[str, object], ...],
    query: str = "",
    scope_kind: str | None = None,
    scope_ref: str | None = None,
    observer: str | None = None,
    observed: str | None = None,
    limit: int = 6,
    max_chars: int | None = None,
    include_stale: bool = False,
    include_archived: bool = False,
    pins: tuple[str, ...] = (),
    usage: tuple[tuple[str, int], ...] = (),
    included_order: tuple[str, ...] = (),
    excluded_reasons: tuple[tuple[str, str], ...] = (),
    absent: tuple[str, ...] = (),
    truncated: bool = False,
    sibling_hints: tuple[tuple[str, str], ...] = (),
    ineligible_included: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "intent": intent,
        "clock": FIXTURE_CLOCK_ISO,
        "query": query,
        "lens": {"scope_kind": scope_kind, "scope_ref": scope_ref, "observer": observer, "observed": observed},
        "budget": {"limit": limit, "max_chars": max_chars},
        "flags": {"include_stale": include_stale, "include_archived": include_archived},
        "pins": list(pins),
        "usage": {record_id: times for record_id, times in usage},
        "records": list(records),
        "expected": {
            "included_order": list(included_order),
            "excluded_reasons": dict(excluded_reasons),
            "absent": list(absent),
            "truncated": truncated,
            "sibling_hints": dict(sibling_hints),
            "ineligible_included": list(ineligible_included),
        },
    }


# Timestamps used across the corpus. Named so a case reads as a statement about
# freshness rather than as a date arithmetic puzzle.
_YOUNG = "2030-12-20T00:00:00Z"
_YOUNGER = "2030-12-28T00:00:00Z"
_AGING = "2030-10-01T00:00:00Z"
_OLD = "2029-06-01T00:00:00Z"
_PAST_DEADLINE = "2030-11-15T00:00:00Z"
_FUTURE_DEADLINE = "2032-06-01T00:00:00Z"

# An ordinary store's worth of unrelated records. Cases about term weighting
# carry them because how rare a word is belongs to a store, not to one record:
# in a two-record store "the" is exactly as rare as "backup".
_BACKGROUND: tuple[dict[str, object], ...] = (
    _record("mem_bg1", "The team uses squash merges for every pull request", approved_at=_AGING, tags=("vcs",)),
    _record("mem_bg2", "Release notes are drafted by the author of the change", approved_at=_AGING, tags=("release",)),
    _record("mem_bg3", "The staging cluster is rebuilt every sunday night", approved_at=_AGING, tags=("staging",)),
    _record("mem_bg4", "Error budgets are reviewed in the monthly ops meeting", approved_at=_AGING, tags=("sre",)),
    _record("mem_bg5", "Python 3.12 is the minimum supported version for the CLI", approved_at=_AGING, tags=("python",)),
    _record("mem_bg6", "The user prefers concise replies without emoji", approved_at=_AGING, tags=("style",)),
)
_BACKGROUND_IDS = tuple(str(spec["record_id"]) for spec in _BACKGROUND)

# A store where most records name the team: newer than the one record about
# backups, so on the words alone they would fill the pack ahead of it.
_TEAM_FLOOD: tuple[dict[str, object], ...] = (
    _record("mem_tf1", "The team meets on mondays", approved_at=_YOUNG, tags=("misc",)),
    _record("mem_tf2", "The team uses squash merges", approved_at=_YOUNG, tags=("misc",)),
    _record("mem_tf3", "Team lunch is on fridays", approved_at=_YOUNG, tags=("misc",)),
    _record("mem_tf4", "The team owns the billing service", approved_at=_YOUNGER, tags=("misc",)),
    _record("mem_tf5", "Team docs live in the wiki", approved_at=_YOUNGER, tags=("misc",)),
    _record("mem_tf6", "The team prefers short replies", approved_at=_YOUNGER, tags=("misc",)),
)
_TEAM_FLOOD_IDS = tuple(str(spec["record_id"]) for spec in _TEAM_FLOOD)

# The long onboarding record two length cases share: it mentions many topics
# once each, the way a checklist captured in one sentence does.
_LONG_ONBOARDING = _record(
    "mem_s12_long",
    "Onboarding: install docker and homebrew, clone the monorepo, run the bootstrap script, request VPN access "
    "from IT, then use the node seed script in tools to load sample data into the local database",
    approved_at=_YOUNG,
    tags=("onboarding",),
)


RETRIEVAL_CASES: tuple[dict[str, object], ...] = (
    _case(
        "exact_and_partial_relevance",
        "An exact multi-token match leads a partial one, and a record with no overlap is named, not dropped.",
        records=(
            _record("mem_r01_exact", "Staging deploys run canary batches before promotion", approved_at=_YOUNG, tags=("deploy", "staging")),
            _record("mem_r02_partial", "Deploys are announced in the release channel", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_r03_unrelated", "The parser owner is the platform team", approved_at=_YOUNG, tags=("parser",)),
        ),
        query="staging deploys canary",
        included_order=("mem_r01_exact", "mem_r02_partial"),
        excluded_reasons=(("mem_r03_unrelated", "no_query_overlap"),),
    ),
    _case(
        "no_match_falls_back_to_the_active_tier",
        "A query nothing answers serves the active records as an unqueried pack would, instead of an empty pack.",
        records=(
            _record("mem_r01_exact", "Staging deploys run canary batches before promotion", approved_at=_YOUNG, tags=("deploy", "staging")),
            _record("mem_r02_partial", "Deploys are announced in the release channel", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_r03_unrelated", "The parser owner is the platform team", approved_at=_YOUNG, tags=("parser",)),
        ),
        query="quantum ledger reconciliation",
        included_order=("mem_r01_exact", "mem_r02_partial", "mem_r03_unrelated"),
    ),
    _case(
        "no_match_fallback_leaves_reference_keyword_gated",
        "The no-match fallback re-admits only the active tier; a reference record still needs a keyword.",
        records=(
            _record("mem_r04_active", "Deploys wait for a green nightly run", approved_at=_YOUNG, tags=("deploy",)),
            _record(
                "mem_r05_reference",
                "The parser owner is the platform team",
                approved_at=_YOUNG,
                tags=("parser",),
                attention_tier="reference",
            ),
        ),
        query="quantum ledger reconciliation",
        included_order=("mem_r04_active",),
        excluded_reasons=(("mem_r05_reference", "no_query_overlap"),),
    ),
    _case(
        "partial_match_never_falls_back",
        "Once any record overlaps the query, an active record that does not stays out as no_query_overlap.",
        records=(
            _record("mem_r06_match", "Staging deploys run canary batches before promotion", approved_at=_YOUNG, tags=("staging",)),
            _record("mem_r07_miss", "The parser owner is the platform team", approved_at=_YOUNGER, tags=("parser",)),
        ),
        query="staging quantum",
        included_order=("mem_r06_match",),
        excluded_reasons=(("mem_r07_miss", "no_query_overlap"),),
    ),
    _case(
        "duplicate_records_collapse_to_the_newest",
        "Two approved records with the same normalized summary take one slot; the older one names why it is out.",
        records=(
            _record("mem_r08_older", "Hotfixes go through the release branch", approved_at=_AGING, tags=("hotfix",)),
            _record("mem_r09_newer", "  hotfixes GO through   the release branch ", approved_at=_YOUNGER, tags=("hotfix",)),
        ),
        query="hotfixes release branch",
        included_order=("mem_r09_newer",),
        excluded_reasons=(("mem_r08_older", "duplicate_record"),),
    ),
    _case(
        "same_topic_disagreement_names_the_cut_sibling",
        "Two records disagree on one topic and the budget keeps one: the cut side names the sibling that survived.",
        records=(
            _record("mem_r10_keeps", "Deploys wait for a green nightly run", approved_at=_YOUNGER, tags=("deploy",)),
            _record("mem_r11_cut", "Deploys ship without waiting for the nightly run", approved_at=_OLD, tags=("deploy",)),
        ),
        query="deploys nightly",
        limit=1,
        included_order=("mem_r10_keeps",),
        excluded_reasons=(("mem_r11_cut", "over_budget"),),
        truncated=True,
        sibling_hints=(("mem_r11_cut", "mem_r10_keeps"),),
    ),
    _case(
        "supersession_removes_the_corrected_revision",
        "A corrected record points at its replacement and leaves the pack as superseded, not as a relevance miss.",
        records=(
            _record("mem_r20_current", "Release notes are written before the tag is pushed", approved_at=_YOUNG, tags=("release",)),
            _record(
                "mem_r21_superseded",
                "Release notes are written after the tag is pushed",
                approved_at=_AGING,
                tags=("release",),
                superseded_by="mem_r20_current",
            ),
        ),
        query="release notes tag",
        included_order=("mem_r20_current",),
        excluded_reasons=(("mem_r21_superseded", "superseded"),),
    ),
    _case(
        "expired_volatile_record_leaves_the_pack",
        "A volatile record whose retention deadline has passed is excluded under its retention class, not silently.",
        records=(
            _record("mem_r30_live", "The staging database runs postgres 17", approved_at=_YOUNG, tags=("staging",)),
            _record(
                "mem_r31_expired",
                "The staging database runs postgres 16",
                approved_at=_AGING,
                tags=("staging",),
                retention_class="volatile",
                expires_at=_PAST_DEADLINE,
                contamination_class="expired",
            ),
        ),
        query="staging database postgres",
        included_order=("mem_r30_live",),
        excluded_reasons=(("mem_r31_expired", "expired_volatile"),),
    ),
    _case(
        "stale_review_is_excluded_by_default",
        "A passed revalidation deadline holds the record out of the default pack under its own reason code.",
        records=(
            _record("mem_r40_fresh", "The nightly job runs on the ubuntu runner", approved_at=_YOUNG, tags=("nightly",), review_due_at=_FUTURE_DEADLINE),
            _record(
                "mem_r41_stale",
                "The nightly job runs on the macos runner",
                approved_at=_AGING,
                tags=("nightly",),
                review_due_at=_PAST_DEADLINE,
                contamination_class="stale",
            ),
        ),
        query="nightly job runner",
        included_order=("mem_r40_fresh",),
        excluded_reasons=(("mem_r41_stale", "stale_review_required"),),
    ),
    _case(
        "include_stale_surfaces_the_record_carrying_ineligible_evidence",
        "The inspection affordance returns the stale record, and it arrives still marked ineligible.",
        records=(
            _record("mem_r40_fresh", "The nightly job runs on the ubuntu runner", approved_at=_YOUNG, tags=("nightly",), review_due_at=_FUTURE_DEADLINE),
            _record(
                "mem_r41_stale",
                "The nightly job runs on the macos runner",
                approved_at=_AGING,
                tags=("nightly",),
                review_due_at=_PAST_DEADLINE,
                contamination_class="stale",
            ),
        ),
        query="nightly job runner",
        include_stale=True,
        included_order=("mem_r40_fresh", "mem_r41_stale"),
        ineligible_included=("mem_r41_stale",),
    ),
    _case(
        "scope_lens_isolates_a_foreign_scope",
        "A record filed under another scope never reaches the pack the lens asked for.",
        records=(
            _record("mem_r50_in_scope", "The default scope owns the release checklist", approved_at=_YOUNG, tags=("release",)),
            _record(
                "mem_r51_other_scope",
                "The other scope owns the release checklist",
                approved_at=_YOUNG,
                tags=("release",),
                scope_ref="other",
                contamination_class="foreign_scope",
            ),
        ),
        query="release checklist",
        scope_kind="project",
        scope_ref=FIXTURE_PROJECT_IDENTITY,
        included_order=("mem_r50_in_scope",),
        absent=("mem_r51_other_scope",),
    ),
    _case(
        "perspective_lens_isolates_another_actors_record",
        "An unscoped record and this actor's record pass; a record observed for another actor does not.",
        records=(
            _record("mem_r60_unscoped", "Handoffs cite the branch name", approved_at=_YOUNG, tags=("handoff",)),
            _record("mem_r61_mine", "Handoffs cite the branch name and the base revision", approved_at=_YOUNG, tags=("handoff",), observer="omh", observed="claude-code"),
            _record(
                "mem_r62_theirs",
                "Handoffs cite the branch name and the ticket id",
                approved_at=_YOUNG,
                tags=("handoff",),
                observer="omh",
                observed="codex",
                contamination_class="foreign_perspective",
            ),
        ),
        query="handoffs branch name",
        observer="omh",
        observed="claude-code",
        included_order=("mem_r60_unscoped", "mem_r61_mine"),
        absent=("mem_r62_theirs",),
    ),
    _case(
        "pins_lead_without_owning_the_whole_budget",
        "A pinned anchor with no query overlap leads the pack, and the strongest match still takes a slot.",
        records=(
            _record("mem_r70_pinned", "The team reviews architecture decisions on friday", approved_at=_OLD, tags=("process",)),
            _record("mem_r71_strong", "Cache invalidation runs on every deploy", approved_at=_YOUNG, tags=("cache", "deploy")),
            _record("mem_r72_weak", "Cache warmers are optional", approved_at=_YOUNG, tags=("cache",)),
        ),
        query="cache invalidation deploy",
        limit=2,
        pins=("mem_r70_pinned",),
        included_order=("mem_r70_pinned", "mem_r71_strong"),
        excluded_reasons=(("mem_r72_weak", "over_budget"),),
        truncated=True,
    ),
    _case(
        "attention_tier_outranks_relevance",
        "An active record leads a reference record that matches the query more strongly.",
        records=(
            _record("mem_r80_active", "Migrations are applied by the release job", approved_at=_YOUNG, tags=("migration",)),
            _record(
                "mem_r81_reference",
                "Migrations are applied by the release job during the release window",
                approved_at=_YOUNG,
                tags=("migration", "release"),
                attention_tier="reference",
            ),
        ),
        query="migrations release job window",
        included_order=("mem_r80_active", "mem_r81_reference"),
    ),
    _case(
        "archive_tier_leaves_the_default_pack_named",
        "An archived record is held back under its own reason code while it stays in the store.",
        records=(
            _record("mem_r90_active", "Feature flags are removed after two releases", approved_at=_YOUNG, tags=("flags",)),
            _record(
                "mem_r91_archived",
                "Feature flags are removed after five releases",
                approved_at=_AGING,
                tags=("flags",),
                attention_tier="archive",
                contamination_class="archived",
            ),
        ),
        query="feature flags releases",
        included_order=("mem_r90_active",),
        excluded_reasons=(("mem_r91_archived", "archived_tier"),),
    ),
    _case(
        "archive_tier_returns_on_an_explicit_archived_query",
        "The archived record comes back when the operator asks for it, behind its active peer.",
        records=(
            _record("mem_r90_active", "Feature flags are removed after two releases", approved_at=_YOUNG, tags=("flags",)),
            _record(
                "mem_r91_archived",
                "Feature flags are removed after five releases",
                approved_at=_AGING,
                tags=("flags",),
                attention_tier="archive",
                contamination_class="archived",
            ),
        ),
        query="feature flags releases",
        include_archived=True,
        included_order=("mem_r90_active", "mem_r91_archived"),
    ),
    _case(
        "age_tie_breaks_deterministically_by_record_id",
        "Two records of equal relevance, equal age, and equal usage order by record id, never by store order.",
        records=(
            _record("mem_ra1_first", "Backups run every night", approved_at=_YOUNG, tags=("backup",)),
            _record("mem_ra2_second", "Backups run each night", approved_at=_YOUNG, tags=("backup",)),
        ),
        query="backups night",
        included_order=("mem_ra1_first", "mem_ra2_second"),
    ),
    _case(
        "record_limit_truncation_reports_over_budget",
        "A record-count budget cuts the tail and says so, rather than returning a short pack silently.",
        records=(
            _record("mem_rb1", "Queue workers retry failed jobs twice", approved_at=_YOUNGER, tags=("queue",)),
            _record("mem_rb2", "Queue workers log every retry", approved_at=_YOUNG, tags=("queue",)),
            _record("mem_rb3", "Queue depth alerts fire at one thousand", approved_at=_AGING, tags=("queue",)),
            _record("mem_rb4", "Queue names are prefixed by the service", approved_at=_OLD, tags=("queue",)),
        ),
        query="queue workers retry",
        limit=2,
        included_order=("mem_rb1", "mem_rb2"),
        excluded_reasons=(("mem_rb3", "over_budget"), ("mem_rb4", "over_budget")),
        truncated=True,
    ),
    _case(
        "character_budget_truncation_is_reported_separately",
        "A character budget cuts the pack under the same over_budget reason with the record limit left slack.",
        records=(
            _record("mem_rc1", "Queue workers retry failed jobs twice", approved_at=_YOUNGER, tags=("queue",)),
            _record("mem_rc2", "Queue workers log every retry attempt for audit", approved_at=_YOUNG, tags=("queue",)),
            _record("mem_rc3", "Queue depth alerts fire at one thousand messages", approved_at=_AGING, tags=("queue",)),
        ),
        query="queue workers retry",
        limit=6,
        max_chars=40,
        included_order=("mem_rc1",),
        excluded_reasons=(("mem_rc2", "over_budget"), ("mem_rc3", "over_budget")),
        truncated=True,
    ),
    _case(
        "accent_folded_query_matches_an_accented_record",
        "A query typed without accents overlaps the accented word it spells, so a French record is found, not fallen back to.",
        records=(
            _record("mem_rd1_deploy", "Le déploiement passe par le Mac Studio le vendredi", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_rd2_theme", "Le thème sombre reste le choix par défaut", approved_at=_YOUNGER, tags=("ui",)),
        ),
        query="deploiement",
        included_order=("mem_rd1_deploy",),
        excluded_reasons=(("mem_rd2_theme", "no_query_overlap"),),
    ),
    _case(
        "accented_word_is_not_split_into_fragments",
        "An accented query word matches whole: its ASCII fragments never reach an unrelated record that shares one.",
        records=(
            _record("mem_re1_preference", "La préférence d'affichage reste le thème sombre", approved_at=_YOUNG, tags=("ui",)),
            _record("mem_re2_pr", "Every PR needs one approving review", approved_at=_YOUNGER, tags=("review",)),
        ),
        query="préférence",
        included_order=("mem_re1_preference",),
        excluded_reasons=(("mem_re2_pr", "no_query_overlap"),),
    ),
    _case(
        "inflection_deployment_reaches_deploy_records",
        "A user asking how deployment is handled is answered by the deploy record, ahead of a record that only shares 'handled'.",
        records=(
            _record("mem_s01_deploys", "Deploys go out through the release pipeline", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_s02_handled", "Hotfixes are handled by the on-call engineer", approved_at=_YOUNG, tags=("hotfix", "on-call")),
            _record("mem_s03_design", "The design review happens on thursdays", approved_at=_YOUNG, tags=("design",)),
        ),
        query="how is deployment handled?",
        included_order=("mem_s01_deploys", "mem_s02_handled"),
        excluded_reasons=(("mem_s03_design", "no_query_overlap"),),
    ),
    _case(
        "inflection_migration_reaches_a_plural_summary",
        "A singular query word finds the plural summary, and the specific term outranks a generic one two records share.",
        records=(
            _record("mem_s04_migrations", "Database migrations run from the deploy job, never by hand", approved_at=_YOUNG, tags=("database", "deploy")),
            _record("mem_s05_lint", "Lint rules live in ruff.toml", approved_at=_YOUNG, tags=("lint",)),
            _record("mem_s06_protection", "Branch protection rules require one approval", approved_at=_YOUNG, tags=("github",)),
        ),
        query="schema migration rules",
        included_order=("mem_s04_migrations", "mem_s05_lint", "mem_s06_protection"),
    ),
    _case(
        "stemmed_only_match_leads_the_fallback_pack",
        # Pre-registered as "'caching' alone answers the cache record"; that
        # changed with the stemmed-only rule (review of #2091, M1): no record
        # shares an unstemmed word, so the stemmed match may not suppress the
        # active-tier fallback, and it leads the fallback pack instead.
        "'caching' meets the cache record only through stemming, so the pack falls back to the active tier with the cache record first.",
        records=(
            _record("mem_s07_cache", "Cache keys include the schema version", approved_at=_YOUNG, tags=("cache",)),
            _record("mem_s08_branches", "Feature branches are deleted after merge", approved_at=_YOUNG, tags=("vcs",)),
            _record("mem_s09_oncall", "On-call rotates every monday", approved_at=_YOUNG, tags=("oncall",)),
        ),
        query="caching strategy",
        included_order=("mem_s07_cache", "mem_s08_branches", "mem_s09_oncall"),
    ),
    _case(
        "tag_match_ranks_above_an_incidental_summary_mention",
        "The record tagged with the query term leads a record whose summary only lists the word in passing.",
        records=(
            _record("mem_s10_compose", "Local services start with compose up from the repo root", approved_at=_YOUNG, tags=("docker", "local-dev")),
            _record("mem_s11_doc", "The onboarding doc lists docker, homebrew and the VPN client", approved_at=_YOUNG, tags=("onboarding",)),
        ),
        query="docker",
        included_order=("mem_s10_compose", "mem_s11_doc"),
    ),
    _case(
        "tag_and_inflection_find_the_flaky_test_policy",
        "The record tagged flaky and tests leads; a summary about snapshot tests matches 'test' and follows it.",
        records=(
            _record("mem_s24_retry", "Retry the integration suite once before reporting a failure", approved_at=_YOUNG, tags=("flaky", "tests")),
            _record("mem_s25_snapshot", "Snapshot tests are updated with the update flag", approved_at=_YOUNG, tags=("snapshot",)),
        ),
        query="flaky test",
        included_order=("mem_s24_retry", "mem_s25_snapshot"),
    ),
    _case(
        "focused_short_record_beats_a_long_incidental_mention",
        "Equal dates, and the long record has the smaller id: only its length can put the focused Node fact first.",
        records=(
            _LONG_ONBOARDING,
            _record("mem_s13_node", "Node 22 is pinned in .nvmrc", approved_at=_YOUNG, tags=("toolchain",)),
            _record("mem_s14_prettier", "Prettier formats all TypeScript files on commit", approved_at=_YOUNG, tags=("format",)),
            _record("mem_s15_review", "Pull requests need one approving review", approved_at=_YOUNG, tags=("review",)),
        ),
        query="which node version",
        included_order=("mem_s13_node", "mem_s12_long"),
        excluded_reasons=(("mem_s14_prettier", "no_query_overlap"), ("mem_s15_review", "no_query_overlap")),
    ),
    _case(
        "long_record_still_leads_when_it_carries_the_query",
        "Length never outweighs coverage: the long record answering three query terms leads a short one answering one.",
        records=(
            _LONG_ONBOARDING,
            _record("mem_s16_vpn", "VPN profiles are stored in 1Password", approved_at=_YOUNG, tags=("vpn",)),
            _record("mem_s13_node", "Node 22 is pinned in .nvmrc", approved_at=_YOUNG, tags=("toolchain",)),
            _record("mem_s15_review", "Pull requests need one approving review", approved_at=_YOUNG, tags=("review",)),
        ),
        query="onboarding vpn access",
        included_order=("mem_s12_long", "mem_s16_vpn"),
        excluded_reasons=(("mem_s13_node", "no_query_overlap"), ("mem_s15_review", "no_query_overlap")),
    ),
    _case(
        "korean_query_with_particles_finds_the_deploy_branch_record",
        "A Korean question with particles finds the deploy-branch record; a record sharing only the bigram 브랜 follows it.",
        records=(
            _record("mem_k01_branch", "배포는 release 브랜치에서만 한다", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_k02_brand", "브랜드 색상은 파란색이다", approved_at=_YOUNG, tags=("design",)),
            _record("mem_k03_review", "코드 리뷰는 한 명의 승인이 필요하다", approved_at=_YOUNG, tags=("review",)),
        ),
        query="배포 브랜치가 뭐였지?",
        included_order=("mem_k01_branch", "mem_k02_brand"),
        excluded_reasons=(("mem_k03_review", "no_query_overlap"),),
    ),
    _case(
        "chinese_query_bigrams_find_the_deploy_process_record",
        "A Chinese query finds the record that states the deploy process ahead of one that shares only 流程.",
        records=(
            _record("mem_c01_deploy", "部署流程需要两个人批准", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_c02_test", "测试流程在CI中运行", approved_at=_YOUNG, tags=("ci",)),
        ),
        query="部署流程",
        included_order=("mem_c01_deploy", "mem_c02_test"),
    ),
    _case(
        "japanese_query_finds_the_deploy_procedure_record",
        "A Japanese query finds the deploy procedure ahead of a test procedure that shares only 手順.",
        records=(
            _record("mem_j01_deploy", "デプロイ手順はREADMEにある", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_j02_test", "テスト手順はCIで動く", approved_at=_YOUNG, tags=("ci",)),
        ),
        query="デプロイ手順",
        included_order=("mem_j01_deploy", "mem_j02_test"),
    ),
    _case(
        "mixed_language_query_ranks_full_coverage_first",
        "An English-and-Korean query leads with the record covering all three terms, then the same environment, then the shared noun.",
        records=(
            _record("mem_x01_address", "staging 서버 주소는 10.0.0.12 이다", approved_at=_YOUNG, tags=("staging", "infra")),
            _record("mem_x02_production", "production 서버는 세 대이다", approved_at=_YOUNG, tags=("production", "infra")),
            _record("mem_x03_deploy", "staging 배포는 자동이다", approved_at=_YOUNG, tags=("staging", "deploy")),
        ),
        query="staging 서버 주소",
        included_order=("mem_x01_address", "mem_x03_deploy", "mem_x02_production"),
    ),
    _case(
        "common_words_do_not_outrank_the_rare_match",
        "A chat message sharing 'plan' (and function words) with an offsite note must not outrank the backup rotation record.",
        records=(
            _record("mem_s21_backup", "Backups rotate weekly and four snapshots are kept", approved_at=_YOUNG, tags=("backup",)),
            _record("mem_s22_offsite", "The plan for the offsite is to book the venue for the team in March", approved_at=_YOUNG, tags=("team",)),
            *_BACKGROUND,
        ),
        query="what is the plan for the backup rotation?",
        limit=1,
        included_order=("mem_s21_backup",),
        # The question's function words ("what", "is", "the", "for") are not
        # query terms, so records sharing only those do not overlap at all.
        excluded_reasons=(("mem_s22_offsite", "over_budget"), *((record_id, "no_query_overlap") for record_id in _BACKGROUND_IDS)),
        truncated=True,
    ),
    _case(
        "pull_request_wording_does_not_answer_who_reviews_docs",
        "Asked who reviews docs pull requests, the docs-review record answers; the squash-merge record only shares wording.",
        records=(
            _record("mem_s23_docs", "Docs changes are reviewed by the tech writer", approved_at=_YOUNG, tags=("docs", "review")),
            *_BACKGROUND,
        ),
        query="who reviews pull requests for the docs?",
        limit=1,
        included_order=("mem_s23_docs",),
        excluded_reasons=(
            ("mem_bg1", "over_budget"),
            ("mem_bg4", "over_budget"),
            *((record_id, "no_query_overlap") for record_id in _BACKGROUND_IDS if record_id not in {"mem_bg1", "mem_bg4"}),
        ),
        truncated=True,
    ),
    _case(
        "stemming_reads_news_as_new_below_the_real_match",
        # The right answer serves mem_s17_hires alone; this pins the known
        # cost of the light stemmer, which reads 'news' as 'new'. The old
        # exact-word match kept the news record out.
        "Known cost, pinned: the light stemmer reads 'news' as 'new', so the release-news record is served after the onboarding record.",
        records=(
            _record("mem_s17_hires", "New hires get a laptop and a buddy on day one", approved_at=_YOUNG, tags=("onboarding",)),
            _record("mem_s18_news", "Release news is posted in the announce channel", approved_at=_YOUNG, tags=("release",)),
        ),
        query="new hire onboarding",
        included_order=("mem_s17_hires", "mem_s18_news"),
    ),
    _case(
        "stemming_keeps_apply_apart_from_app",
        "'apply' is not cut down to 'app': the mobile app record stays out while the staging patch record answers.",
        records=(
            _record("mem_s19_patches", "Patches land on staging before production", approved_at=_YOUNG, tags=("staging",)),
            _record("mem_s20_app", "The mobile app ships through TestFlight", approved_at=_YOUNG, tags=("mobile",)),
        ),
        query="apply patch staging",
        included_order=("mem_s19_patches",),
        excluded_reasons=(("mem_s20_app", "no_query_overlap"),),
    ),
    _case(
        "same_subject_serves_only_the_newest_value",
        "Two records declare subject deploy-branch: only the newest approved value is served, the older one is named.",
        records=(
            _record("mem_u01_release", "Deploys go from the release branch", approved_at=_AGING, tags=("deploy",), subject="deploy-branch"),
            _record("mem_u02_main", "Deploys go from main and the release branch is retired", approved_at=_YOUNG, tags=("deploy",), subject="deploy-branch"),
        ),
        query="which branch do deploys go from",
        included_order=("mem_u02_main",),
        excluded_reasons=(("mem_u01_release", "superseded_by_subject"),),
    ),
    _case(
        "same_subject_old_wording_never_brings_the_old_value_back",
        # Pre-registered as "the current value leads"; that changed with the
        # score-transfer rule (review of #2091, Q3): the current value shares
        # no query word, so it takes no score from the old one. The style
        # record matches only through stemming, so the pack falls back.
        "The query names the old wording of reply-language: the old value stays out, and the current value arrives through the fallback.",
        records=(
            _record("mem_u03_korean", "Reply language for this user is Korean", approved_at=_AGING, tags=("style",), subject="reply-language"),
            _record("mem_u04_english", "The user wants answers in English", approved_at=_YOUNG, tags=("style",), subject="reply-language"),
            _record("mem_u05_concise", "The user prefers concise replies without emoji", approved_at=_YOUNG, tags=("style",)),
        ),
        query="reply language",
        included_order=("mem_u05_concise", "mem_u04_english"),
        excluded_reasons=(("mem_u03_korean", "superseded_by_subject"),),
    ),
    _case(
        "different_subjects_never_collapse",
        "Two deploy records with different subjects are separate facts and both stay in the pack.",
        records=(
            _record("mem_v01_branch", "Deploys go from main", approved_at=_YOUNG, tags=("deploy",), subject="deploy-branch"),
            _record("mem_v02_window", "Deploys happen between 10:00 and 16:00 KST", approved_at=_YOUNG, tags=("deploy",), subject="deploy-window"),
        ),
        query="deploy setup",
        included_order=("mem_v01_branch", "mem_v02_window"),
    ),
    _case(
        "same_subject_in_different_scopes_stays_separate",
        "A project record and a user-global record share a subject; each scope keeps its own value, newest first.",
        records=(
            _record("mem_w01_repo", "Reply language in this repository is Korean", approved_at=_AGING, tags=("style",), subject="reply-language"),
            _record(
                "mem_w02_global",
                "Reply language everywhere else is English",
                approved_at=_YOUNG,
                tags=("style",),
                subject="reply-language",
                scope_kind="user-global",
                scope_ref="default",
            ),
        ),
        query="reply language",
        included_order=("mem_w02_global", "mem_w01_repo"),
    ),
    # The review of #2091 probed the first BM25 revision with these stores and
    # queries; they are copied verbatim, with the dispositions the review asked for.
    _case(
        "english_question_word_outranks_topic",
        "Review of #2091 (H2): 'how' is a function word, so the record opening with 'How expenses' never outranks the deploy records.",
        records=(
            _record("mem_q1", "Deploy through the release pipeline after CI is green", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_q2", "Deploy approvals need two reviewers", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_q3", "Deploy freezes start on december 20", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_q4", "How expenses are filed is documented in the wiki", approved_at=_YOUNG, tags=("finance",)),
        ),
        query="how do we deploy?",
        included_order=("mem_q1", "mem_q2", "mem_q3"),
        excluded_reasons=(("mem_q4", "no_query_overlap"),),
    ),
    _case(
        "all_korean_store",
        "Review of #2091 (H2): 어떻게 is a function word, so '어떻게 해야 할지 모르면' never outranks the deploy records.",
        records=(
            _record("mem_h1", "배포는 release 브랜치에서만 한다", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_h2", "배포 승인은 두 명이 필요하다", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_h3", "코드 리뷰는 한 명의 승인이 필요하다", approved_at=_YOUNG, tags=("review",)),
            _record("mem_h4", "어떻게 해야 할지 모르면 팀장에게 묻는다", approved_at=_YOUNG, tags=("process",)),
        ),
        query="배포는 어떻게 해?",
        included_order=("mem_h1", "mem_h2"),
        excluded_reasons=(("mem_h3", "no_query_overlap"), ("mem_h4", "no_query_overlap"),),
    ),
    _case(
        "korean_question_word_who",
        "Review of #2091 (H2): 누가 is a function word, so the meeting-notes record never outranks the deploy records.",
        records=(
            _record("mem_r1", "배포 승인은 두 명의 리뷰어가 한다", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_r2", "배포는 release 브랜치에서만 한다", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_r3", "배포 동결은 12월 20일부터다", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_r4", "누가 회의록을 쓰는지는 주간 회의에서 정한다", approved_at=_YOUNG, tags=("meeting",)),
        ),
        query="배포 승인은 누가 해?",
        included_order=("mem_r1", "mem_r2", "mem_r3"),
        excluded_reasons=(("mem_r4", "no_query_overlap"),),
    ),
    _case(
        "stats_conflates_with_state",
        "Review of #2091 (M1): 'stats' meets 'State' only through stemming, so the pack falls back to the active tier.",
        records=(
            _record("mem_a1_state", "State is kept in a Redux store", approved_at=_YOUNG, tags=("frontend",)),
            _record("mem_a2_grafana", "Grafana dashboards hold the latency metrics", approved_at=_YOUNG, tags=("observability",)),
            _record("mem_a3_release", "Releases ship on fridays", approved_at=_YOUNG, tags=("release",)),
        ),
        query="stats location?",
        included_order=("mem_a1_state", "mem_a2_grafana", "mem_a3_release"),
    ),
    _case(
        "news_conflates_with_new",
        "Review of #2091 (M1): 'news' meets 'New hires' only through stemming, so the pack falls back to the active tier.",
        records=(
            _record("mem_b1_hires", "New hires get a laptop on day one", approved_at=_YOUNG, tags=("onboarding",)),
            _record("mem_b2_announce", "Release notes are posted in announce", approved_at=_YOUNG, tags=("release",)),
        ),
        query="any news?",
        included_order=("mem_b1_hires", "mem_b2_announce"),
    ),
    _case(
        "like_conflates_with_likely",
        "Review of #2091 (M1): 'like' meets 'likely' only through stemming, so the pack falls back to the active tier.",
        records=(
            _record("mem_c1_flaky", "Flaky tests are likely caused by shared fixtures", approved_at=_YOUNG, tags=("tests",)),
            _record("mem_c2_style", "The user prefers short replies", approved_at=_YOUNG, tags=("style",)),
        ),
        query="I would like concise answers",
        included_order=("mem_c1_flaky", "mem_c2_style"),
    ),
    _case(
        "even_conflates_with_evening",
        "Review of #2091 (M1): 'even' meets 'evening' only through stemming, so the pack falls back to the active tier.",
        records=(
            _record("mem_d1_freeze", "Deploys freeze in the evening", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_d2_oncall", "On-call rotates every monday", approved_at=_YOUNG, tags=("oncall",)),
        ),
        query="even weekends?",
        included_order=("mem_d1_freeze", "mem_d2_oncall"),
    ),
    _case(
        "statement_conflates_with_state",
        "Review of #2091 (M1): 'statement' meets 'state' only through stemming, so the pack falls back to the active tier.",
        records=(
            _record("mem_e1_state", "Global state lives in zustand", approved_at=_YOUNG, tags=("frontend",)),
            _record("mem_e2_pay", "Invoices are paid by wire transfer", approved_at=_YOUNG, tags=("finance",)),
        ),
        query="bank statement",
        included_order=("mem_e1_state", "mem_e2_pay"),
    ),
    _case(
        "temporal_recent_deploy_decision",
        "Review of #2091 (M2): a temporal query bands relevance, so the newer deploy-window decision leads the older one that repeats 'deploy'.",
        records=(
            _record("mem_p1_old", "Deploy windows are 10:00 to 16:00 and deploys need approval from the release manager", approved_at="2030-06-01T00:00:00Z", tags=("deploy",)),
            _record("mem_p2_new", "Deploy windows moved to 09:00 to 12:00", approved_at="2030-12-30T00:00:00Z", tags=("deploy",)),
        ),
        query="what did we decide recently about deploy windows",
        included_order=("mem_p2_new", "mem_p1_old"),
    ),
    _case(
        "record_type_plural_floods_pack",
        "Review of #2091 (M3): 'facts' names a record type; with no summary or tag hit it adds nothing, so only the deploy record is served.",
        records=(
            _record("mem_j00", "Python 3.12 is the minimum version", approved_at=_YOUNG, tags=("python",)),
            _record("mem_j01", "Lint rules live in ruff.toml", approved_at=_YOUNG, tags=("lint",)),
            _record("mem_j02", "The staging cluster is rebuilt nightly", approved_at=_YOUNG, tags=("staging",)),
            _record("mem_j03", "Error budgets are reviewed monthly", approved_at=_YOUNG, tags=("sre",)),
            _record("mem_j04", "Release notes are drafted by the author", approved_at=_YOUNG, tags=("release",)),
            _record("mem_j05", "Squash merges are used for every pull request", approved_at=_YOUNG, tags=("vcs",)),
            _record("mem_j06", "The VPN profile is stored in 1Password", approved_at=_YOUNG, tags=("vpn",)),
            _record("mem_j99_deploy", "Deploys go out through the release pipeline", approved_at=_YOUNG, tags=("deploy",)),
        ),
        query="any facts about deploys?",
        included_order=("mem_j99_deploy",),
        excluded_reasons=(("mem_j00", "no_query_overlap"), ("mem_j01", "no_query_overlap"), ("mem_j02", "no_query_overlap"), ("mem_j03", "no_query_overlap"), ("mem_j04", "no_query_overlap"), ("mem_j05", "no_query_overlap"), ("mem_j06", "no_query_overlap"),),
    ),
    _case(
        "subject_newest_expired_resurrects_old",
        "Review of #2091 (M5): the newest deploy-branch value expired; the older value stays superseded instead of coming back.",
        records=(
            _record("mem_n1_old", "Deploys go from the release branch", approved_at=_AGING, tags=("deploy",), retention_class="durable", subject="deploy-branch"),
            _record("mem_n2_new", "Deploys go from main", approved_at=_YOUNG, tags=("deploy",), retention_class="volatile", expires_at="2030-12-30T00:00:00Z", subject="deploy-branch"),
        ),
        query="deploy branch",
        included_order=(),
        excluded_reasons=(("mem_n1_old", "superseded_by_subject"), ("mem_n2_new", "expired_volatile"),),
    ),
    _case(
        "subject_newest_archived_resurrects_old",
        "Review of #2091 (M5): the newest deploy-branch value is archived; the older value stays superseded instead of coming back.",
        records=(
            _record("mem_o1_old", "Deploys go from the release branch", approved_at=_AGING, tags=("deploy",), subject="deploy-branch"),
            _record("mem_o2_new", "Deploys go from main", approved_at=_YOUNG, tags=("deploy",), attention_tier="archive", subject="deploy-branch"),
        ),
        query="deploy branch",
        included_order=(),
        excluded_reasons=(("mem_o1_old", "superseded_by_subject"), ("mem_o2_new", "archived_tier"),),
    ),
    # The re-review of #2091 probed the second revision with these question-shaped
    # stores; copied verbatim. The merge bar: no unrelated record above the topic
    # record the old scorer ranked first, and a listed word in capitals still scores.
    _case(
        "ja_interrogative_how",
        "Review of #2091 (H2): どう is cut out of the query, and coverage keeps every deploy record above the record that only shares やる and るの.",
        records=(
            _record("mem_ja1", "デプロイ手順はREADMEにある", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_ja2", "デプロイ前にテストを走らせる", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_ja3", "デプロイは金曜日にしない", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_ja4", "どうやるのか分からない時はSlackで聞く", approved_at=_YOUNG, tags=("process",)),
        ),
        query="デプロイはどうやるの？",
        included_order=("mem_ja3", "mem_ja1", "mem_ja2", "mem_ja4"),
    ),
    _case(
        "zh_interrogative_how",
        "Review of #2091 (H2): 怎么 is cut out of the query before bigrams, so only the deploy records match 部署.",
        records=(
            _record("mem_zh1", "部署流程需要两个人批准", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_zh2", "部署前要跑测试", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_zh3", "周五不部署", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_zh4", "不知道怎么办就问组长", approved_at=_YOUNG, tags=("process",)),
        ),
        query="怎么部署？",
        included_order=("mem_zh1", "mem_zh2", "mem_zh3"),
        excluded_reasons=(("mem_zh4", "no_query_overlap"),),
    ),
    _case(
        "ko_variant_eonjejjeum",
        "Review of #2091 (H2): a run starting with 언제 is dropped whole, so 언제쯤이야 never matches the meeting record.",
        records=(
            _record("mem_k1", "배포는 release 브랜치에서만 한다", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_k2", "배포 승인은 두 명이 필요하다", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_k3", "배포 동결은 12월 20일부터다", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_k4", "언제 회의할지는 팀장이 정한다", approved_at=_YOUNG, tags=("meeting",)),
            _record("mem_k5", "누가요 라는 질문에는 담당자 표를 보여준다", approved_at=_YOUNG, tags=("process",)),
        ),
        query="배포 언제쯤이야?",
        included_order=("mem_k1", "mem_k2", "mem_k3"),
        excluded_reasons=(("mem_k4", "no_query_overlap"), ("mem_k5", "no_query_overlap"),),
    ),
    _case(
        "en_remind",
        "Review of #2091 (H2): 'remind' is a chat filler, and coverage keeps a deploy record with its tag above any stray-word record.",
        records=(
            _record("mem_e1", "Deploy through the release pipeline after CI is green", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_e2", "Deploy approvals need two reviewers", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_e3", "Deploy freezes start on december 20", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_e4", "We remind new hires to enable 2FA", approved_at=_YOUNG, tags=("security",)),
            _record("mem_e5", "Quick fixes go through the hotfix branch", approved_at=_YOUNG, tags=("hotfix",)),
            _record("mem_e6", "Open questions are tracked in the RFC doc", approved_at=_YOUNG, tags=("rfc",)),
        ),
        query="can you remind me how we deploy?",
        included_order=("mem_e1", "mem_e2", "mem_e3"),
        excluded_reasons=(("mem_e4", "no_query_overlap"), ("mem_e5", "no_query_overlap"), ("mem_e6", "no_query_overlap"),),
    ),
    _case(
        "en_quick_question",
        "Review of #2091 (H2): 'quick question' is filler, so neither the hotfix nor the RFC record outranks the deploy records.",
        records=(
            _record("mem_e1", "Deploy through the release pipeline after CI is green", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_e2", "Deploy approvals need two reviewers", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_e3", "Deploy freezes start on december 20", approved_at=_YOUNG, tags=("deploy",)),
            _record("mem_e4", "We remind new hires to enable 2FA", approved_at=_YOUNG, tags=("security",)),
            _record("mem_e5", "Quick fixes go through the hotfix branch", approved_at=_YOUNG, tags=("hotfix",)),
            _record("mem_e6", "Open questions are tracked in the RFC doc", approved_at=_YOUNG, tags=("rfc",)),
        ),
        query="quick question: do we deploy on fridays?",
        included_order=("mem_e1", "mem_e2", "mem_e3"),
        excluded_reasons=(("mem_e4", "no_query_overlap"), ("mem_e5", "no_query_overlap"), ("mem_e6", "no_query_overlap"),),
    ),
    _case(
        "who_org_as_topic_alone",
        "Review of #2091 (H2 rule 3): 'WHO' in capitals names something, so it scores although 'who' is a function word.",
        records=(
            _record("mem_w1", "WHO travel advisories are checked before booking offsites", approved_at=_YOUNG, tags=("travel",)),
            _record("mem_w2", "Offsite venues are booked by the ops lead", approved_at=_YOUNG, tags=("offsite",)),
            _record("mem_w3", "Budget reviews happen monthly", approved_at=_YOUNG, tags=("budget",)),
        ),
        query="WHO?",
        included_order=("mem_w1",),
        excluded_reasons=(("mem_w2", "no_query_overlap"), ("mem_w3", "no_query_overlap"),),
    ),
    _case(
        "who_org_with_words",
        "Review of #2091 (H2 rule 3): capital 'WHO' mid-question still scores, beside 'travel'.",
        records=(
            _record("mem_w1", "WHO travel advisories are checked before booking offsites", approved_at=_YOUNG, tags=("travel",)),
            _record("mem_w2", "Offsite venues are booked by the ops lead", approved_at=_YOUNG, tags=("offsite",)),
            _record("mem_w3", "Budget reviews happen monthly", approved_at=_YOUNG, tags=("budget",)),
        ),
        query="what does WHO say about travel",
        included_order=("mem_w1",),
        excluded_reasons=(("mem_w2", "no_query_overlap"), ("mem_w3", "no_query_overlap"),),
    ),
    _case(
        "will_person_topic",
        "Review of #2091 (H2 rule 3): 'Will' capitalized mid-sentence names a person and scores.",
        records=(
            _record("mem_wl1", "Will owns the budget spreadsheet", approved_at=_YOUNG, tags=("budget",)),
            _record("mem_wl2", "Budget reviews happen monthly", approved_at=_YOUNG, tags=("budget",)),
            _record("mem_wl3", "The CAN bus logger runs on the test rig", approved_at=_YOUNG, tags=("hardware",)),
            _record("mem_wl4", "Bus timeouts are retried twice", approved_at=_YOUNG, tags=("hardware",)),
        ),
        query="ask Will",
        included_order=("mem_wl1",),
        excluded_reasons=(("mem_wl2", "no_query_overlap"), ("mem_wl3", "no_query_overlap"), ("mem_wl4", "no_query_overlap"),),
    ),
    _case(
        "can_bus_topic",
        "Review of #2091 (H2 rule 3): 'CAN' in capitals names the bus and scores with 'logger'.",
        records=(
            _record("mem_wl1", "Will owns the budget spreadsheet", approved_at=_YOUNG, tags=("budget",)),
            _record("mem_wl2", "Budget reviews happen monthly", approved_at=_YOUNG, tags=("budget",)),
            _record("mem_wl3", "The CAN bus logger runs on the test rig", approved_at=_YOUNG, tags=("hardware",)),
            _record("mem_wl4", "Bus timeouts are retried twice", approved_at=_YOUNG, tags=("hardware",)),
        ),
        query="CAN logger",
        included_order=("mem_wl3",),
        excluded_reasons=(("mem_wl1", "no_query_overlap"), ("mem_wl2", "no_query_overlap"), ("mem_wl4", "no_query_overlap"),),
    ),
    _case(
        "store_common_term_never_matches_alone",
        "A word most records hold never makes a record match by itself: the backup record leads alone, though "
        "only stemming matches it, and the team records are named common_terms_only instead of filling the pack.",
        records=(*_TEAM_FLOOD, _record("mem_tf_backup", "Backups rotate weekly", approved_at=_AGING, tags=("ops",))),
        query="team backup",
        included_order=("mem_tf_backup",),
        excluded_reasons=tuple((record_id, "common_terms_only") for record_id in _TEAM_FLOOD_IDS),
    ),
    _case(
        "store_common_term_stemmed_only_query_falls_back",
        "When no servable record shares an unstemmed query word ('teams' against 'team'), the common word's hits are "
        "stemmed-only too, so the query falls back: the backup record is served and the team records stay out.",
        records=(*_TEAM_FLOOD, _record("mem_tf_backup", "Backups rotate weekly", approved_at=_AGING, tags=("ops",))),
        query="teams backup",
        included_order=("mem_tf_backup",),
        excluded_reasons=tuple((record_id, "common_terms_only") for record_id in _TEAM_FLOOD_IDS),
    ),
    _case(
        "subject_history_never_makes_its_own_word_common",
        "A subject's replaced values are not servable, so they never count toward a common word: the current "
        "value still answers a query worded like every value of it.",
        records=(
            *(
                _record(f"mem_sh{index}", f"Deploys go from branch v{index}", approved_at=_AGING, tags=("ops",), subject="deploy-branch")
                for index in range(4)
            ),
            _record("mem_sh_current", "Deploys go from the release branch", approved_at=_YOUNGER, tags=("ops",), subject="deploy-branch"),
            _record("mem_sh_ci", "CI runs on Python 3.12", approved_at=_YOUNG, tags=("ci",)),
            _record("mem_sh_region", "Staging lives in region eu", approved_at=_YOUNG, tags=("infra",)),
            _record("mem_sh_owner", "Sungmin owns billing", approved_at=_YOUNG, tags=("owner",)),
        ),
        query="which branch do deploys use now",
        included_order=("mem_sh_current",),
        excluded_reasons=(
            *((f"mem_sh{index}", "superseded_by_subject") for index in range(4)),
            ("mem_sh_ci", "no_query_overlap"),
            ("mem_sh_region", "no_query_overlap"),
            ("mem_sh_owner", "no_query_overlap"),
        ),
    ),
    _case(
        "duplicate_copies_count_once_toward_a_common_word",
        "Copies of one record take one pack slot, so they count once toward a common word and cannot hold back "
        "the one other record about the same thing.",
        records=(
            _record("mem_dc_rotate", "Rotate the backup encryption key monthly", approved_at=_AGING, tags=("ops",)),
            _record("mem_dc_ci", "CI runs on Python 3.12", approved_at=_YOUNG, tags=("ci",)),
            _record("mem_dc_owner", "Sungmin owns billing", approved_at=_YOUNG, tags=("owner",)),
            _record("mem_dc_docs", "Docs live in the wiki", approved_at=_YOUNG, tags=("docs",)),
            *(
                _record(f"mem_dc_copy{index}", "Backup questions go to the external portal", approved_at=_YOUNGER, tags=("backup",))
                for index in range(4)
            ),
        ),
        query="backup restore steps",
        included_order=("mem_dc_copy0", "mem_dc_rotate"),
        excluded_reasons=(
            ("mem_dc_ci", "no_query_overlap"),
            ("mem_dc_owner", "no_query_overlap"),
            ("mem_dc_docs", "no_query_overlap"),
            *((f"mem_dc_copy{index}", "duplicate_record") for index in range(1, 4)),
        ),
    ),
    _case(
        "project_name_alone_never_fills_the_pack",
        "A task no record answers, naming the project most records mention, serves only the record tagged with "
        "the project name; the others met the query on that name alone.",
        records=(
            _record("mem_pn_currency", "Tally's default currency is KRW and USD stays supported", approved_at=_AGING, tags=("currency", "tally")),
            _record("mem_pn_port", "Tally serve listens on port 8765 by default", approved_at=_YOUNG, tags=("serve",)),
            _record("mem_pn_export", "Monthly tally exports are uploaded to the finance bucket", approved_at=_YOUNG, tags=("export",)),
            _record("mem_pn_report", "Tally reports bucket expenses by Seoul calendar month", approved_at=_YOUNG, tags=("report",)),
            _record("mem_pn_ci", "CI runs the test suite on Python 3.11 and 3.12", approved_at=_YOUNG, tags=("ci",)),
            _record("mem_pn_owner", "Sungmin owns the reports module", approved_at=_YOUNG, tags=("owner",)),
        ),
        query="`tally list --limit 0` prints the whole ledger instead of nothing. Fix it.",
        included_order=("mem_pn_currency",),
        excluded_reasons=(
            ("mem_pn_port", "common_terms_only"),
            ("mem_pn_export", "common_terms_only"),
            ("mem_pn_report", "common_terms_only"),
            ("mem_pn_ci", "no_query_overlap"),
            ("mem_pn_owner", "no_query_overlap"),
        ),
    ),
)


def fixture_digest(cases: tuple[dict[str, object], ...] = RETRIEVAL_CASES) -> str:
    """One digest over the retained corpus, keys sorted, separators fixed."""
    return hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
