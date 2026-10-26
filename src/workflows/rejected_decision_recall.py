from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re

from ..paths import OmhPaths
from ..plugin_bundle.omh.memory_governance import (
    PROJECT_MEMORY_RECORD_SCHEMA_VERSION,
    canonical_payload_digest,
    evaluate_memory_replay,
    stable_artifact_identity,
)
from ..system.metadata_safety import is_sensitive_metadata_text
from .rejected_decision_evidence import (
    ALLOWED_SCOPE_KINDS,
    SAFE_REF,
    RejectedDecisionEvidence,
    metadata_text,
    normalized_tags,
    read_rejected_decision_evidence,
)


REJECTED_DECISION_RECALL_SCHEMA_VERSION = "rejected_decision_recall/v1"
REJECTED_DECISION_RECALL_CLAIM_BOUNDARY = (
    "This is a reviewed-decision surface only: bounded negative evidence, separate from "
    "approved-memory recall, never an approved fact or instruction, and never auto-attached to a coding prompt."
)
_TOKEN_RE = re.compile(r"[a-z0-9_/-]+")


@dataclass(frozen=True)
class RejectedDecisionRecallRequest:
    query: str
    scope_kind: str
    scope_ref: str
    tags: tuple[str, ...] = ()
    include_stale: bool = False
    limit: int = 6


def build_rejected_decision_recall(
    paths: OmhPaths,
    request: RejectedDecisionRecallRequest,
    *,
    now: datetime | None = None,
) -> dict[str, object]:
    scope_kind, scope_ref = _validated_scope(request)
    requested_tags, limit = normalized_tags(request.tags), _validated_limit(request.limit)
    query, current_time = metadata_text(request.query, limit=240), _as_utc(now)
    matches: list[tuple[RejectedDecisionEvidence, dict[str, object], int]] = []
    excluded: list[dict[str, object]] = []
    for decision, review_id in read_rejected_decision_evidence(paths):
        if (decision.scope_kind, decision.scope_ref) != (scope_kind, scope_ref):
            continue
        evaluation = _evaluate(decision, review_id, current_time, scope_kind, scope_ref)
        if not bool(evaluation["eligible"]):
            excluded.append(_item(decision, evaluation))
            continue
        if not set(requested_tags).issubset(decision.tags):
            continue
        score = _match_score(decision, _tokens(query))
        if _tokens(query) and score == 0:
            continue
        matches.append((decision, {**evaluation, "reason_code": "eligible_legacy_read_only" if decision.legacy else "eligible"}, score))
    matches.sort(key=lambda value: value[0].candidate_id)
    matches.sort(key=lambda value: value[0].reviewed_at, reverse=True)
    matches.sort(key=lambda value: value[2], reverse=True)
    excluded.sort(key=lambda value: (str(value["candidate_id"]), int(value["decision_revision"])))
    return {
        "schema_version": REJECTED_DECISION_RECALL_SCHEMA_VERSION,
        "query": query,
        "scope": {"kind": scope_kind, "ref": scope_ref},
        "requested_tags": list(requested_tags),
        "include_stale": request.include_stale,
        "limit": limit,
        "matches": [_item(decision, evaluation, score) for decision, evaluation, score in matches[:limit]],
        "excluded_matches": excluded[:limit],
        "excluded_truncated": len(excluded) > limit,
        "claim_boundary": REJECTED_DECISION_RECALL_CLAIM_BOUNDARY,
    }


def _evaluate(
    decision: RejectedDecisionEvidence,
    review_id: str,
    now: datetime,
    scope_kind: str,
    scope_ref: str,
) -> dict[str, object]:
    # The shared evaluator accepts replay artifacts, not review records. This
    # private adapter evaluates immutable decision evidence without promoting
    # the rejected subject to approved memory or returning the adapter itself.
    artifact: dict[str, object] = {
        "schema_version": PROJECT_MEMORY_RECORD_SCHEMA_VERSION,
        "record_id": review_id,
        "revision": decision.decision_revision,
        "record_type": decision.record_type,
        "summary": decision.summary,
        "value": decision.rejection_reason,
        "scope": {"kind": decision.scope_kind, "ref": decision.scope_ref},
        "source_class": decision.source_class,
        "retention": decision.retention,
    }
    if decision.revalidation is not None:
        artifact["revalidation"] = decision.revalidation
    if decision.superseded_by is not None:
        artifact["superseded_by"] = decision.superseded_by
    payload_digest, identity = canonical_payload_digest(artifact), stable_artifact_identity(artifact)
    artifact["admission"] = {"state": "approved_manual", "review_id": review_id, "payload_digest": payload_digest}
    return evaluate_memory_replay(
        artifact,
        now=now,
        requested_scope={"kind": scope_kind, "ref": scope_ref},
        review_resolver={review_id: {"artifact_identity": identity, "payload_digest": payload_digest}},
    )


def _item(
    decision: RejectedDecisionEvidence,
    evaluation: dict[str, object],
    score: int | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "candidate_id": decision.candidate_id,
        "decision_revision": decision.decision_revision,
        "record_type": decision.record_type,
        "scope": {"kind": decision.scope_kind, "ref": decision.scope_ref},
        "reviewed_at": decision.reviewed_at,
        "admission_mode": "legacy_rejected_snapshot" if decision.legacy else "rejected_review",
        "source_class": decision.source_class,
        "retention_class": decision.retention["class"],
        "evaluation_timestamp": evaluation["evaluated_at"],
        "eligibility_reason": evaluation["reason_code"],
        "legacy": decision.legacy,
        "authoritative": not decision.legacy,
        "approved_memory": False,
        "surface_kind": "reviewed_negative_decision",
        "renderable_as_instruction": False,
    }
    if score is not None:
        item.update({"summary": decision.summary, "rejection_reason": decision.rejection_reason, "tags": list(decision.tags), "match_score": score})
    return item


def _match_score(decision: RejectedDecisionEvidence, query_tokens: frozenset[str]) -> int:
    return len(query_tokens & _tokens(f"{decision.summary} {decision.record_type}")) + len(query_tokens & set(decision.tags))


def _validated_scope(request: RejectedDecisionRecallRequest) -> tuple[str, str]:
    if request.scope_kind not in ALLOWED_SCOPE_KINDS:
        raise ValueError(f"unsupported rejected-decision scope kind: {request.scope_kind}")
    if not SAFE_REF.fullmatch(request.scope_ref) or is_sensitive_metadata_text(request.scope_ref):
        raise ValueError(f"unsafe rejected-decision scope ref: {request.scope_ref!r}")
    return request.scope_kind, request.scope_ref


def _validated_limit(limit: int) -> int:
    if not 1 <= limit <= 20:
        raise ValueError("rejected-decision recall limit must be between 1 and 20")
    return limit


def _tokens(value: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(value.lower()))


def _as_utc(value: datetime | None) -> datetime:
    now = value if value is not None else datetime.now(timezone.utc)
    return now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
