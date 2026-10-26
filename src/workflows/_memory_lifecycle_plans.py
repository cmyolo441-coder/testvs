from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from ..plugin_bundle.omh.memory_governance import (
    MEMORY_CLASSIFIER_VERSION,
    MEMORY_GOVERNANCE_POLICY_VERSION,
    PRINCIPAL_PROJECT_MEMORY_RECORD_SCHEMA_VERSION,
    PRINCIPAL_PROJECT_MEMORY_REVIEW_RECORD_SCHEMA_VERSION,
    PROJECT_MEMORY_RECORD_SCHEMA_VERSION,
    PROJECT_MEMORY_REVIEW_RECORD_SCHEMA_VERSION,
    build_retention,
    canonical_memory_scope,
    canonical_payload_digest,
    resolve_record_expiry_deadline,
    stable_artifact_identity,
)
from ..system.paths import OmhPaths
from ._memory_lifecycle_model import LifecycleMutation, LifecyclePlan, LifecycleTransactionExecutor
from ._memory_lifecycle_scan import ScanFinding, journal_findings, linked_to, matching_json_findings, matching_scope_findings, read_json, safe_token, stamp
from .memory import MEMORY_ATTENTION_SCHEMA_VERSION, MEMORY_ATTENTION_TIERS


def _scope(value: Mapping[str, object]) -> dict[str, object]:
    scope = canonical_memory_scope(dict(value.get("scope", {})))
    if not all(safe_token(scope[key]) for key in ("kind", "ref")):
        raise ValueError("scope_invalid")
    return scope


def _admission_state(value: Mapping[str, object]) -> str:
    admission = value.get("admission")
    return str(admission.get("state", "")) if isinstance(admission, Mapping) else ""


def _retention_class(value: Mapping[str, object]) -> str:
    retention = value.get("retention")
    return str(retention.get("class", "")) if isinstance(retention, Mapping) else ""


def _expiry_reason(value: Mapping[str, object], now: datetime) -> str:
    """Whether this revision is past its deadline, on the one deadline there is.

    This used to read `retention.expires_at` alone while recall, retirement,
    status, and the dreaming counts read `ttl.expires_at`. Both fields are
    written from a single value at approval and nothing enforced that they stay
    equal, so one record could be expired for half the system and eligible for
    the other half at the same moment -- `memory retire` reporting `expired: 1`
    for the record `memory prune` was rejecting as
    `prune_requires_expired_approved_volatile`.

    `resolve_record_expiry_deadline` reads both spellings and takes the earlier,
    which is what `classify_record_expiry` now reads too, so the claim in its
    docstring -- that recall and retirement "can never disagree" -- is true of
    the lifecycle plans as well.
    """
    retention = value.get("retention")
    if not isinstance(retention, Mapping) or _retention_class(value) not in {"volatile", "standard", "durable"}:
        return "retention_invalid"
    deadline, state = resolve_record_expiry_deadline(value)
    if state == "absent":
        return "eligible"
    if deadline is None:
        return "retention_parse_error"
    current = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
    return f"expired_{_retention_class(value)}" if current >= deadline else "eligible"


def _current_record(paths: OmhPaths, record_id: str, revision: int) -> tuple[dict[str, object], str | None]:
    if not safe_token(record_id) or not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise ValueError("unsafe_record_identity")
    value, error = read_json(paths.memory_dir, f"records/{record_id}.json")
    if error or value is None:
        return {}, error or "record_not_found"
    if value.get("schema_version") not in {PROJECT_MEMORY_RECORD_SCHEMA_VERSION, PRINCIPAL_PROJECT_MEMORY_RECORD_SCHEMA_VERSION} or value.get("record_id") != record_id or value.get("revision") != revision:
        return {}, "exact_revision_not_current"
    if value.get("source_class") != "omh_local":
        return {}, "external_target"
    try:
        _scope(value)
    except ValueError:
        return {}, "scope_invalid"
    return value, None


def _archived_record(paths: OmhPaths, record_id: str, revision: int) -> tuple[dict[str, object], str, str | None]:
    if not safe_token(record_id) or not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise ValueError("unsafe_record_identity")
    matches, errors = matching_json_findings(paths.memory_dir, "archive", record_id, revision, "archive")
    if errors:
        return {}, "", errors[0]
    if len(matches) != 1:
        return {}, "", "archive_exact_revision_not_found" if not matches else "archive_exact_revision_ambiguous"
    value, relative = dict(matches[0].value), matches[0].relative_path
    if value.get("schema_version") not in {PROJECT_MEMORY_RECORD_SCHEMA_VERSION, PRINCIPAL_PROJECT_MEMORY_RECORD_SCHEMA_VERSION}:
        return {}, relative, "archive_exact_revision_not_found"
    try:
        _scope(value)
    except ValueError:
        return {}, relative, "scope_invalid"
    return value, relative, None


def _tombstone(tombstone_id: str, record_id: str, revision: int, scope: Mapping[str, object], reason: str, now: datetime) -> dict[str, object]:
    operation = "retire" if reason == "retired_archive" else "prune"
    return {"schema_version": "memory_tombstone/v1", "tombstone_id": tombstone_id, "record_id": record_id, "revision": revision, "scope": dict(scope), "operation_id": f"memory-{operation}-{record_id}-r{revision}", "reason_code": reason, "actor_class": "operator", "tombstoned_at": stamp(now)}


def _pending_candidate(record: Mapping[str, object], candidate_id: str, revision: int, lifecycle: str) -> dict[str, object]:
    origin = stable_artifact_identity(dict(record))
    replacement = {key: record[key] for key in ("record_type", "summary", "scope", "source_class", "source_ref", "source_evidence", "retention", "revalidation", "ttl", "staleness", "derived_from", "perspective", "attention", "identity") if key in record}
    return {"schema_version": "project_memory_candidate/v2", "candidate_id": candidate_id, "candidate_revision": revision, "record_id": str(record["record_id"]), "artifact_kind": "record", "status": "pending_review", "admission": {"state": "pending_review"}, "origin": origin, "lifecycle": lifecycle, "replacement": replacement}


def _source_evidence(replacement: Mapping[str, object]) -> dict[str, object]:
    """Carry a reviewed revision's source digest into its successor, scalar-only.

    Without this the corrected revision would silently lose the digest of the
    file it cites, and a later source change would stop being observable
    after the first correction.
    """
    evidence = replacement.get("source_evidence")
    if not isinstance(evidence, Mapping) or not str(evidence.get("sha256", "") or ""):
        return {}
    return {key: str(evidence.get(key, "") or "") for key in ("path", "sha256", "captured_at")}


def _attention(replacement: Mapping[str, object]) -> dict[str, object]:
    """Carry a reviewed revision's attention tier into its successor, scalar-only.

    Without this a correction or a restore silently returns an archived or
    reference record to the active working set -- the exact opposite of what
    the operator asked for, and invisible until the record reappeared in a
    handoff. An unreadable tier carries nothing, so the successor falls back to
    the ordinary active default rather than inheriting a corrupt value.
    """
    attention = replacement.get("attention")
    if not isinstance(attention, Mapping) or str(attention.get("tier", "") or "") not in MEMORY_ATTENTION_TIERS:
        return {}
    return {
        "schema_version": MEMORY_ATTENTION_SCHEMA_VERSION,
        **{key: str(attention.get(key, "") or "") for key in ("tier", "reason", "previous_tier", "changed_at")},
    }


def _resolution(replacement: Mapping[str, object]) -> dict[str, object]:
    """Carry a reviewed revision's open/resolved marker into its successor, scalar-only.

    A corrected open record arrives here already marked `resolved` (the
    correction plan wrote the answer); a restored open record arrives still
    `open`, and restoring a question must restore it as a question, ceiling
    included. Without this the successor would silently read as a settled
    fact -- the exact promotion an open record exists to refuse. Any other
    value carries nothing, so a corrupt marker cannot exempt a record.
    """
    staleness = replacement.get("staleness")
    if not isinstance(staleness, Mapping) or str(staleness.get("resolution", "") or "") not in {"open", "resolved"}:
        return {}
    keys = ("resolution", "open_since", "open_expires_at", "resolved_at")
    return {key: str(staleness.get(key, "") or "") for key in keys if str(staleness.get(key, "") or "")}


def _approved_record(replacement: Mapping[str, object], record_id: str, revision: int, reviewer: str, now: datetime) -> dict[str, object]:
    record_type = str(replacement.get("record_type", "fact"))
    retention = replacement.get("retention") if isinstance(replacement.get("retention"), Mapping) else {}
    ttl_days = retention.get("ttl_days") if isinstance(retention.get("ttl_days"), int) and not isinstance(retention.get("ttl_days"), bool) else None
    identity = replacement.get("identity") if isinstance(replacement.get("identity"), Mapping) else None
    if identity is not None:
        identity = {
            **identity,
            "reviewer": {"principal": identity.get("reviewer", {}).get("principal") if isinstance(identity.get("reviewer"), Mapping) else None, "review_ref": f"review-{record_id}-r{revision}"},
            "audience": {**identity.get("audience", {}), "review_ref": f"review-{record_id}-r{revision}"} if isinstance(identity.get("audience"), Mapping) else {},
        }
    record: dict[str, object] = {"schema_version": PRINCIPAL_PROJECT_MEMORY_RECORD_SCHEMA_VERSION if identity is not None else PROJECT_MEMORY_RECORD_SCHEMA_VERSION, "record_id": record_id, "revision": revision, "record_type": record_type, "summary": str(replacement.get("summary", "")), "scope": _scope(replacement), "source_class": str(replacement.get("source_class", "omh_local")), "derived_from": [str(ref) for ref in (replacement.get("derived_from") if isinstance(replacement.get("derived_from"), list) else []) if isinstance(ref, str)], **({"perspective": {"observer": str(replacement["perspective"].get("observer", "")), "observed": str(replacement["perspective"].get("observed", ""))}} if isinstance(replacement.get("perspective"), Mapping) and str(replacement["perspective"].get("observed", "")) else {}), **({"source_evidence": evidence} if (evidence := _source_evidence(replacement)) else {}), **({"attention": attention} if (attention := _attention(replacement)) else {}), **({"staleness": resolution} if (resolution := _resolution(replacement)) else {}), **({"identity": dict(identity)} if identity is not None else {}), "retention": build_retention(str(retention.get("class", "standard")), record_type=record_type, admitted_at=now, ttl_days=ttl_days), "revalidation": {}, "admission": {"state": "approved_manual", "review_id": f"review-{record_id}-r{revision}", "reviewer_claim": reviewer, "admitted_at": stamp(now), "policy_version": MEMORY_GOVERNANCE_POLICY_VERSION, "classifier_version": MEMORY_CLASSIFIER_VERSION}}
    identity = stable_artifact_identity(record)
    record["admission"] = {**record["admission"], "artifact_identity": identity, "payload_digest": canonical_payload_digest(record)}
    return record


def project_identity_successor(record: Mapping[str, Any], scope_ref: str, now: datetime, *, operation_id: str) -> tuple[dict[str, Any], dict[str, object]]:
    """Reviewed scope-only successor; never renew retention or rewrite source evidence."""
    replacement = {**record, "scope": {"kind": "project", "ref": scope_ref}}
    successor: dict[str, Any] = _approved_record(replacement, str(record["record_id"]), int(record["revision"]) + 1, "reviewed_project_identity_migration", now)
    successor = {**record, "operation_id": operation_id, **{key: successor[key] for key in ("revision", "scope", "admission")},
                 **({"identity": successor["identity"]} if "identity" in successor else {})}
    successor["admission"] = {**successor["admission"], "artifact_identity": stable_artifact_identity(successor), "payload_digest": canonical_payload_digest(successor)}
    review = _review(successor, str(successor["admission"]["review_id"]), "reviewed_project_identity_migration")
    return successor, review


def _review(record: Mapping[str, object], review_id: str, reviewer: str) -> dict[str, object]:
    return {"schema_version": PRINCIPAL_PROJECT_MEMORY_REVIEW_RECORD_SCHEMA_VERSION if record.get("schema_version") == PRINCIPAL_PROJECT_MEMORY_RECORD_SCHEMA_VERSION else PROJECT_MEMORY_REVIEW_RECORD_SCHEMA_VERSION, "review_id": review_id, "artifact_identity": stable_artifact_identity(dict(record)), "decision": "approved_manual", "reviewer_claim": reviewer, **({"identity": dict(record["identity"])} if isinstance(record.get("identity"), Mapping) else {}), "payload_digest": canonical_payload_digest(dict(record)), "policy_version": MEMORY_GOVERNANCE_POLICY_VERSION, "classifier_version": MEMORY_CLASSIFIER_VERSION}


def _manifest(items: Sequence[tuple[str, str, str]]) -> list[dict[str, object]]:
    return [{"target_id": target_id, "artifact_kind": kind, "relative_path": relative} for target_id, kind, relative in items]


def _plan(operation: str, record_id: str, revision: int, scope: Mapping[str, object], now: datetime, manifest: Sequence[Mapping[str, object]], mutations: Sequence[LifecycleMutation], preserved: Sequence[Mapping[str, object]] = ()) -> LifecyclePlan:
    next_actions = {"retire": "Apply only after review: archive the exact expired standard revision; it is not deleted.", "restore": "Review/reactivate the pending restored revision; the archive remains unchanged.", "prune": "Apply only with --confirm-hard-delete-local; the receipt is not deletion or erasure proof.", "correct": "Review and approve the pending replacement revision.", "reapprove": "The new active revision remains governed by its immutable review record."}
    report = {"schema_version": "memory_lifecycle_report/v1", "operation_type": operation, "record_id": record_id, "revision": revision, "scope": dict(scope), "eligible": True, "applied": False, "reason_code": "planned", "manifest": list(manifest), "outcomes": [], "exclusions": ["backups", "filesystem_snapshots", "trash", "synced_copies", "hermes_native", "external_providers", "vector_stores", "executors", "unlinked_artifacts", "recall_usage_counters", "pin_markers"], "next_action": next_actions[operation], "claim_boundary": "This is an OMH-local observed target-set plan, not deletion, erasure, backup, provider, native-memory, executor, or legal-erasure proof."}
    return LifecyclePlan(f"memory-{operation}-{record_id}-r{revision}", operation, record_id, revision, dict(scope), now, report, tuple(mutations), tuple(preserved))


def _rejected(operation: str, record_id: str, revision: int, now: datetime, reason: str, outcomes: Sequence[Mapping[str, object]] = ()) -> LifecyclePlan:
    safe_id = record_id if safe_token(record_id) else "invalid"
    safe_revision = revision if isinstance(revision, int) and not isinstance(revision, bool) and revision > 0 else 1
    report = {"schema_version": "memory_lifecycle_report/v1", "operation_type": operation, "record_id": safe_id, "revision": safe_revision, "scope": {"kind": "project", "ref": "default"}, "eligible": False, "applied": False, "reason_code": reason, "manifest": [], "outcomes": list(outcomes) or [{"target_id": "rejected:0", "artifact_kind": "local_target", "outcome": "rejected", "reason_code": reason}], "exclusions": [], "next_action": "No apply action: resolve the reported lifecycle guard.", "claim_boundary": "No mutation was planned. This report does not claim deletion, erasure, or any provider/native/executor effect."}
    return LifecyclePlan(f"memory-{operation}-{safe_id}-r{safe_revision}", operation, safe_id, safe_revision, report["scope"], now, report, ())


def _prune_findings(paths: OmhPaths, record_id: str, revision: int) -> tuple[list[ScanFinding], list[str], list[dict[str, object]]]:
    root, findings, errors = paths.memory_dir, [], []
    for directory, kind in (("records", "record"), ("archive", "archive"), ("history", "history"), ("candidates", "candidate"), ("reviews", "review"), ("blocks", "block"), ("tombstones", "tombstone")):
        matched, found_errors = matching_json_findings(root, directory, record_id, revision, kind)
        findings.extend(matched)
        errors.extend(found_errors)
    scoped, found_errors = matching_scope_findings(root, record_id, revision)
    journals, journal_errors, preserved = journal_findings(root, record_id, revision)
    findings.extend(scoped)
    findings.extend(journals)
    errors.extend(found_errors + journal_errors)
    for finding in findings:
        source = str(finding.value.get("source_class", ""))
        if source in {"provider", "external", "hermes_native"} or (finding.artifact_kind == "tombstone" and finding.value.get("reason_code") == "hard_deleted_local"):
            errors.append("external_target" if source else "tombstoned_identity")
    if (root / f"tombstones/hard-deleted-{record_id}-r{revision}.json").exists():
        errors.append("tombstone_collision")
    index, index_error = read_json(root, "index.json")
    if index_error not in {None, "already_absent"}:
        errors.append(index_error)
    elif index is not None and (any(finding.relative_path in str(index) for finding in findings) or record_id in str(index)):
        findings.append(ScanFinding("index.json", "index_entry", f"index_entry:{record_id}:r{revision}", index))
    return findings, errors, preserved


def _prune_mutations(root: Any, findings: Sequence[ScanFinding], record_id: str, revision: int, scope: Mapping[str, object], now: datetime) -> tuple[LifecycleMutation, ...]:
    mutations: list[LifecycleMutation] = []
    seen: set[str] = set()
    for finding in findings:
        if finding.relative_path in seen:
            continue
        seen.add(finding.relative_path)
        if finding.artifact_kind == "scope_item":
            items = finding.value.get("items", {})
            payload = {**finding.value, "items": {key: value for key, value in dict(items).items() if not linked_to(value, record_id, revision)}}
            action = "write"
        elif finding.artifact_kind == "journal_entry":
            lines = finding.value.get("lines", [])
            payload = tuple(line for line in lines if not linked_to(line, record_id, revision))
            action = "rewrite_jsonl"
        elif finding.artifact_kind == "index_entry":
            removed = {item.relative_path for item in findings if item.relative_path != "index.json"}
            payload = _remove_index_paths(finding.value, removed, record_id)
            action = "write"
        else:
            payload, action = None, "delete"
        mutations.append(LifecycleMutation(f"prune_{len(mutations)}", action, finding.relative_path, finding.target_id, finding.artifact_kind, payload=payload))
    tombstone_id = f"hard-deleted-{record_id}-r{revision}"
    mutations.append(LifecycleMutation("write_hard_delete_tombstone", "write", f"tombstones/{tombstone_id}.json", f"tombstone:{tombstone_id}", "tombstone", payload=_tombstone(tombstone_id, record_id, revision, scope, "hard_deleted_local", now)))
    return tuple(mutations)


def _remove_index_paths(value: Mapping[str, object], removed: set[str], record_id: str = "") -> dict[str, object]:
    exact = set(removed) | {path.removesuffix(".json") for path in removed}
    if record_id:
        exact.add(record_id)
    omitted = object()

    def clean(item: object) -> object:
        if isinstance(item, str):
            return omitted if item in exact else item
        if isinstance(item, list):
            return [child for value in item if (child := clean(value)) is not omitted]
        if isinstance(item, Mapping):
            result: dict[str, object] = {}
            removed_any = False
            for key, child in item.items():
                key_text = str(key)
                if key_text in exact:
                    if isinstance(child, str) and child in exact:
                        removed_any = True
                        continue
                    elif isinstance(child, (dict, list)) and not child:
                        removed_any = True
                        continue
                cleaned = clean(child)
                if cleaned is omitted:
                    removed_any = True
                    continue
                result[key_text] = cleaned
            return omitted if (removed_any and not result) else result
        return item

    cleaned = clean(value)
    return dict(cleaned) if isinstance(cleaned, Mapping) else {}


def _require_guard(paths: OmhPaths, plan: LifecyclePlan, directory: str) -> None:
    guarded = [item for item in plan.report.get("manifest", []) if isinstance(item, Mapping) and str(item.get("relative_path", "")).startswith(f"{directory}/")]
    if not guarded:
        raise ValueError("operation_guard_missing")
    value, error = read_json(paths.memory_dir, str(guarded[0]["relative_path"]))
    if error or value is None:
        raise ValueError("operation_guard_mismatch")
    if directory in {"records", "archive"} and (value.get("record_id") != plan.record_id or value.get("revision") != plan.revision):
        raise ValueError("operation_guard_mismatch")


def _execute_plan(paths: OmhPaths, plan: LifecyclePlan, executor: LifecycleTransactionExecutor, validator: Callable[[Mapping[str, object]], list[str]]) -> dict[str, object]:
    if not plan.report.get("eligible"):
        raise ValueError(str(plan.report.get("reason_code", "operation_rejected")))
    result = executor(paths, plan)
    receipt = result.get("receipt") if isinstance(result, Mapping) else None
    if not isinstance(receipt, Mapping) or validator(receipt):
        raise ValueError("invalid_transaction_receipt")
    expected = {mutation.target_id for mutation in plan.mutations}
    actual = {str(item.get("target_id", "")) for item in receipt.get("outcomes", []) if isinstance(item, Mapping)}
    if not expected.issubset(actual):
        raise ValueError("transaction_receipt_missing_target")
    return {"operation_id": plan.operation_id, "applied": True, "receipt": dict(receipt), "next_action": plan.report["next_action"], "claim_boundary": "Receipt records only what the supplied transaction executor reported for named OMH-local targets; it never proves deletion or erasure outside that set."}
