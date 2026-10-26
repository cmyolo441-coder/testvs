"""Crash-safe lifecycle for approved project-local skills.

`SKILL.md` is the only visibility commit.  Everything else is immutable
history or private pre-entry staging; status derives truth from bytes, never a
mutable active-state record.

Since #1571 the same lifecycle carries either promotion source -- an approved,
replay-passing browser workflow trace, or a reviewed and activated
`skill_draft/v1`.  Nothing here was duplicated for the second kind: locking,
staging, the activation index, receipt validation, rollback, retry and status
are one implementation, and the three things that genuinely differ by kind --
which record the source lock names, how an entry's metadata is spelled, and what
"the source drifted" means -- go through `skill_promotion_source`.  The module,
its function names and its `browser_skill_*` schema strings keep their shipped
spelling so no digest, receipt or installed entry moves.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
import re
import secrets
import stat
from pathlib import Path
from typing import Literal

from ..system.local_store import file_lock, utc_now
from .browser_skill_promotion_approval import (
    BrowserSkillPromotionApprovalError, HermesPromotionNativeHost, PromotionNativeHost,
    approve_browser_skill_promotion, read_browser_skill_promotion_approval_receipt,
    review_browser_skill_promotion,
)
from .browser_skill_promotion_plan import read_browser_skill_package
from .browser_workflow_learning import BrowserTraceError
from .browser_workflow_learning_store import _root as observed_git_root, read_browser_workflow_trace, resolved_browser_workflow_promotion_reference
from .skill_promotion_source import (
    SKILL_DRAFT_SOURCE, SkillPromotionSourceError, entry_source_binding,
    parse_promotion_entry_metadata, promotion_manifest_file_sets, promotion_source_lock_parts,
    read_project_skill_draft, receipt_source_binding, skill_draft_digest,
)

_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_SLUG = re.compile(r"^[a-z][a-z0-9-]{2,48}$")
_MAX_FILE_BYTES = 262144
# The receipt fields that name the promotion source. Compared as a block so a
# draft entry, which spells them differently in its own metadata, is checked
# against the same four values the receipt bound.
_SOURCE_BINDING_KEYS = ("trace_id", "trace_revision", "trace_digest", "fixture_digests")


class BrowserSkillPromotionError(ValueError):
    pass


def review_browser_skill_lifecycle(
    project_root: str | Path, source_id: str, skill_name: str, *,
    operation: Literal["install", "update"] = "install", host: PromotionNativeHost | None = None,
) -> dict[str, object]:
    root = observed_git_root(project_root)
    active = _active(root, skill_name)
    inventory = _managed_inventory(root, skill_name)
    if active is not None:
        _require_current_source(root, active)
        if _same_identity(active, source_id):
            return _unchanged_review(root, skill_name, active)
        operation = "update"
    elif operation == "update":
        raise BrowserSkillPromotionError("update requires a verified active managed SKILL.md")
    return review_browser_skill_promotion(
        root, source_id, skill_name, operation=operation,
        previous_generation=str(active["generation"]) if active else None,
        base_entry_digest=str(active["entry_digest"]) if active else "absent",
        existing_files=inventory, host=host or HermesPromotionNativeHost(),
    )


def approve_browser_skill_lifecycle(
    project_root: str | Path, source_id: str, skill_name: str, *, reviewed_diff_digest: str,
    reviewer_identity: str, operation: Literal["install", "update"] = "install",
    host: PromotionNativeHost | None = None,
) -> dict[str, object]:
    root = observed_git_root(project_root)
    active = _active(root, skill_name)
    inventory = _managed_inventory(root, skill_name)
    if active is not None:
        _require_current_source(root, active)
        if _same_identity(active, source_id):
            if reviewed_diff_digest != _digest(b""):
                raise BrowserSkillPromotionError("unchanged promotion requires the empty exact diff digest")
            return _mapping(active, "receipt")
        operation = "update"
    elif operation == "update":
        raise BrowserSkillPromotionError("update requires a verified active managed SKILL.md")
    return approve_browser_skill_promotion(
        root, source_id, skill_name, reviewed_diff_digest=reviewed_diff_digest, reviewer_identity=reviewer_identity,
        operation=operation, previous_generation=str(active["generation"]) if active else None,
        base_entry_digest=str(active["entry_digest"]) if active else "absent", existing_files=inventory,
        host=host or HermesPromotionNativeHost(),
    )


def review_browser_skill_rollback(project_root: str | Path, skill_name: str, generation: str, *, host: PromotionNativeHost | None = None) -> dict[str, object]:
    root = observed_git_root(project_root)
    active = _require_active(root, skill_name)
    historical = _historical_entry(root, skill_name, generation)
    _require_current_source(root, historical)
    inventory = _managed_inventory(root, skill_name)
    return review_browser_skill_promotion(
        root, _source_id(historical), skill_name, operation="rollback", previous_generation=str(active["generation"]),
        base_entry_digest=str(active["entry_digest"]), rollback_of=generation, existing_files=inventory,
        host=host or HermesPromotionNativeHost(),
    )


def approve_browser_skill_rollback(project_root: str | Path, skill_name: str, generation: str, *, reviewed_diff_digest: str, reviewer_identity: str, host: PromotionNativeHost | None = None) -> dict[str, object]:
    root = observed_git_root(project_root)
    active = _require_active(root, skill_name)
    historical = _historical_entry(root, skill_name, generation)
    _require_current_source(root, historical)
    return approve_browser_skill_promotion(
        root, _source_id(historical), skill_name, reviewed_diff_digest=reviewed_diff_digest, reviewer_identity=reviewer_identity,
        operation="rollback", previous_generation=str(active["generation"]), base_entry_digest=str(active["entry_digest"]),
        rollback_of=generation, existing_files=_managed_inventory(root, skill_name), host=host or HermesPromotionNativeHost(),
    )


def review_browser_skill_removal(project_root: str | Path, skill_name: str, *, host: PromotionNativeHost | None = None) -> dict[str, object]:
    root = observed_git_root(project_root)
    active = _active(root, skill_name)
    if active is None:
        previous = _last_observation(root, skill_name)
        if previous is not None and previous.get("status") in {"stale", "quarantined", "removed"}:
            return {**previous, "status": "already_deactivated", "reused": True}
        raise BrowserSkillPromotionError("a verified managed SKILL.md is required")
    _require_current_source(root, active)
    return review_browser_skill_promotion(root, _source_id(active), skill_name, operation="remove", previous_generation=str(active["generation"]), base_entry_digest=str(active["entry_digest"]), existing_files=_managed_inventory(root, skill_name), host=host or HermesPromotionNativeHost())


def approve_browser_skill_removal(project_root: str | Path, skill_name: str, *, reviewed_diff_digest: str, reviewer_identity: str, host: PromotionNativeHost | None = None) -> dict[str, object]:
    root = observed_git_root(project_root)
    active = _active(root, skill_name)
    if active is None:
        previous = _last_observation(root, skill_name)
        if previous is not None and previous.get("status") in {"stale", "quarantined", "removed"}:
            return {**previous, "status": "already_deactivated", "reused": True}
        raise BrowserSkillPromotionError("a verified managed SKILL.md is required")
    _require_current_source(root, active)
    return approve_browser_skill_promotion(root, _source_id(active), skill_name, reviewed_diff_digest=reviewed_diff_digest, reviewer_identity=reviewer_identity, operation="remove", previous_generation=str(active["generation"]), base_entry_digest=str(active["entry_digest"]), existing_files=_managed_inventory(root, skill_name), host=host or HermesPromotionNativeHost())


def promote_approved_browser_skill(project_root: str | Path, receipt_id: str, *, host: PromotionNativeHost | None = None) -> dict[str, object]:
    """Perform the one visibility commit covered by an immutable receipt."""
    root = observed_git_root(project_root)
    receipt = read_browser_skill_promotion_approval_receipt(root, receipt_id)
    skill_name = _skill_from_receipt(root, receipt)
    if receipt["operation"] == "remove":
        return _remove_approved(root, skill_name, receipt, host or HermesPromotionNativeHost())
    active = _active(root, skill_name)
    if active is not None and active.get("receipt_id") == receipt_id:
        return _reuse_or_deactivate(root, skill_name, active)
    with file_lock(_source_lock_path(root, str(receipt["trace_id"])), private=True):
        with file_lock(_lock_path(root, skill_name), private=True):
            active = _active(root, skill_name)
            if active is not None and active.get("receipt_id") == receipt_id:
                return _reuse_or_deactivate(root, skill_name, active)
            inventory = _managed_inventory(root, skill_name, allow_unindexed_generation=str(receipt["generation"]))
            # A crash can leave complete immutable resources but no visibility
            # commit. Retry is explicit and accepts them only byte-for-byte.
            pre_stage = _drop_generation(inventory, str(receipt["generation"]))
            plan = _receipt_plan(root, receipt, pre_stage, host or HermesPromotionNativeHost())
            if _without_staged_generation(inventory, plan) != pre_stage:
                raise BrowserSkillPromotionError("incomplete staging differs from the approved immutable generation")
            _stage_and_verify(root, skill_name, plan)
            current = _managed_inventory(root, skill_name, allow_unindexed_generation=str(receipt["generation"]))
            if _without_staged_generation(current, plan) != pre_stage:
                raise BrowserSkillPromotionError("promotion target changed during immutable staging")
            # Re-resolve source, policy and exact base immediately before the
            # only visible write while both cooperating locks are held.
            plan = _receipt_plan(root, receipt, pre_stage, host or HermesPromotionNativeHost())
            _write_activation_index(root, skill_name, plan, receipt)
            _replace_entry(root, skill_name, _text(plan, "package")["SKILL.md"], str(plan["activation_id"]))
            verified = _active(root, skill_name)
            if verified is None or verified.get("receipt_id") != receipt_id:
                raise BrowserSkillPromotionError("SKILL.md readback did not validate the approved activation")
            return _observe(root, skill_name, "rolled_back" if receipt["operation"] == "rollback" else "active", verified, reused=False)


def browser_skill_promotion_status(project_root: str | Path, skill_name: str, *, check_source: bool = True) -> dict[str, object]:
    root = observed_git_root(project_root)
    active = _active(root, skill_name)
    if active is None:
        target = _target(root, skill_name)
        if os.path.lexists(target / "SKILL.md"):
            return {"schema_version": "browser_skill_promotion/v1", "status": "unverified_managed_state", "skill_name": skill_name, "source_checked": False}
        previous = _last_observation(root, skill_name)
        if previous is not None and previous.get("status") in {"stale", "quarantined", "removed"}:
            return {**previous, "source_checked": False, "current_entry": "absent"}
        return {"schema_version": "browser_skill_promotion/v1", "status": "inactive", "skill_name": skill_name, "source_checked": False}
    if not check_source:
        return _observe(root, skill_name, "active_unchecked", active, reused=True, reason="source check intentionally skipped")
    return _reuse_or_deactivate(root, skill_name, active)


def retry_browser_skill_promotion(project_root: str | Path, receipt_id: str, *, host: PromotionNativeHost | None = None) -> dict[str, object]:
    """Explicitly resume an incomplete pre-entry attempt; never runs in status."""
    return promote_approved_browser_skill(project_root, receipt_id, host=host)


def _remove_approved(root: Path, skill_name: str, receipt: Mapping[str, object], host: PromotionNativeHost) -> dict[str, object]:
    with file_lock(_source_lock_path(root, str(receipt["trace_id"])), private=True):
        with file_lock(_lock_path(root, skill_name), private=True):
            active = _active(root, skill_name)
            if active is None:
                # An explicit repeat after successful removal, or after drift
                # already deactivated the verified entry, is a safe no-op.
                previous = _last_observation(root, skill_name)
                if previous is not None:
                    return {**previous, "status": "removed", "reused": True}
                return _observe(root, skill_name, "removed", None, reused=True)
            inventory = _managed_inventory(root, skill_name)
            _receipt_plan(root, receipt, inventory, host)
            if active["entry_digest"] != receipt["base_entry_digest"]:
                raise BrowserSkillPromotionError("removal approval base is stale")
            _unlink_entry(_target(root, skill_name) / "SKILL.md")
            return _observe(root, skill_name, "removed", active, reused=False)


def _receipt_plan(root: Path, receipt: Mapping[str, object], inventory: Mapping[str, str], host: PromotionNativeHost) -> dict[str, object]:
    plan_review = review_browser_skill_promotion(
        # The receipt's `trace_id` is its shipped `/v1` name for the promotion
        # source id; for a draft receipt it holds that draft's id.
        root, str(receipt["trace_id"]), Path(str(receipt["target_path"])).name,
        operation=str(receipt["operation"]), previous_generation=_nullable(receipt.get("previous_generation"), receipt.get("operation")),
        base_entry_digest=str(receipt["base_entry_digest"]), rollback_of=_nullable(receipt.get("rollback_of"), "rollback"),
        existing_files=inventory, host=host,
    )
    plan = _mapping(plan_review, "plan")
    source = _mapping(plan, "source")
    keys = ("operation", "activation_id", "payload_digest", "rollback_of", "base_digest", "base_entry_digest", "previous_generation", "project_root", "project_identity", "target_path", "trace_id", "trace_revision", "trace_digest", "fixture_digests", "generic_draft_digest", "generation", "package_digest", "entry_digest", "manifest_digest", "reviewed_diff_digest", "policy_revision")
    preflight = _mapping(plan_review, "native_preflight")
    policy = _mapping(preflight, "policy")
    expected: dict[str, object] = {**plan, **source, **receipt_source_binding(source)}
    expected["previous_generation"] = source.get("previous_generation")
    expected["reviewed_diff_digest"] = plan.get("diff_digest")
    expected["policy_revision"] = policy.get("revision")
    for key in keys:
        if receipt.get(key) != expected.get(key):
            raise BrowserSkillPromotionError(f"promotion approval is stale at {key}")
    return plan


def _nullable(value: object, operation: object) -> str | None:
    if operation == "install":
        return None
    return str(value) if isinstance(value, str) and _DIGEST.fullmatch(value) else None


def _active(root: Path, skill_name: str) -> dict[str, object] | None:
    target = _target(root, skill_name)
    entry_path = target / "SKILL.md"
    if not os.path.lexists(entry_path):
        return None
    try:
        entry = _read_text(entry_path)
        metadata = _entry_metadata(entry)
        generation = _digest_value(metadata.get("generation"), "entry generation")
        owned = _validated_generation(root, skill_name, generation)
        if entry != owned["entry"]:
            raise BrowserSkillPromotionError("managed entry differs from immutable entry resource")
        return owned
    except (BrowserSkillPromotionError, BrowserSkillPromotionApprovalError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


def _validated_generation(root: Path, skill_name: str, generation: str) -> dict[str, object]:
    target = _target(root, skill_name)
    entry = _read_text(target / "resources" / generation / "entry.md")
    metadata = _entry_metadata(entry)
    entry_digest = _digest(entry.encode("utf-8"))
    if metadata.get("generation") != generation:
        raise BrowserSkillPromotionError("immutable entry generation mismatch")
    manifest_path = target / "resources" / generation / "manifest.json"
    manifest_bytes = _read_bytes(manifest_path)
    manifest = _manifest(manifest_path, generation)
    _verify_manifest(target, manifest)
    index = _read_json(_activation_path(root, skill_name, entry_digest))
    expected_index = {"schema_version", "entry_digest", "activation_id", "receipt_id", "project_identity", "target_relative_path", "generation", "manifest_digest", "payload_digest"}
    if set(index) != expected_index or index.get("schema_version") != "browser_skill_activation/v1":
        raise BrowserSkillPromotionError("activation index has an unsupported schema")
    receipt = read_browser_skill_promotion_approval_receipt(root, str(index.get("receipt_id", "")))
    package = {"SKILL.md": entry, **{name: _read_text(target / name) for name in _mapping(manifest, "files")}, f"resources/{generation}/manifest.json": manifest_bytes.decode("utf-8")}
    if (
        index.get("entry_digest") != entry_digest or index.get("activation_id") != metadata.get("activation_id")
        or index.get("generation") != generation or index.get("manifest_digest") != _digest(manifest_bytes)
        or index.get("project_identity") != receipt.get("project_identity") or index.get("target_relative_path") != str(target.relative_to(root))
        or index.get("payload_digest") != receipt.get("payload_digest")
        or receipt.get("entry_digest") != entry_digest or receipt.get("generation") != generation
        or receipt.get("manifest_digest") != _digest(manifest_bytes) or receipt.get("package_digest") != _package_digest(package)
        or receipt.get("activation_id") != metadata.get("activation_id") or receipt.get("target_path") != str(target)
        or receipt.get("project_identity") != _project_identity(root)
        or {key: receipt.get(key) for key in _SOURCE_BINDING_KEYS} != entry_source_binding(metadata)
        or receipt.get("generic_draft_digest") != metadata.get("generic_draft_digest")
    ):
        raise BrowserSkillPromotionError("approval receipt does not bind immutable generation bytes")
    return {**metadata, "entry": entry, "entry_digest": entry_digest, "receipt_id": receipt["receipt_id"], "receipt": receipt, "manifest": manifest}


def _require_active(root: Path, skill_name: str) -> dict[str, object]:
    active = _active(root, skill_name)
    if active is None:
        raise BrowserSkillPromotionError("a verified managed SKILL.md is required")
    return active


def _require_current_source(root: Path, active: Mapping[str, object]) -> None:
    if active.get("source_kind") == SKILL_DRAFT_SOURCE:
        _require_current_draft(root, active)
        return
    try:
        reference = resolved_browser_workflow_promotion_reference(root, str(active["trace_id"]))
    except BrowserTraceError as exc:
        raise BrowserSkillPromotionError("approved browser trace is unavailable or no longer replay-passing") from exc
    required = {"trace_digest": "trace_digest", "trace_revision": "trace_revision", "origins": "origins", "fixture_digests": "fixture_digests", "output_schema_digest": "output_schema_digest", "replay_digest": "replay_digest"}
    for entry_key, reference_key in required.items():
        if active.get(entry_key) != reference.get(reference_key):
            raise BrowserSkillPromotionError("approved browser trace has drifted")


def _require_current_draft(root: Path, active: Mapping[str, object]) -> None:
    """A draft source drifts when its record stops resolving as the reviewed one.

    A draft id is a content hash of the proposed name, the selected runs and the
    instruction sections, so editing any of those produces a different draft
    rather than a changed one. `draft_digest` covers the rest of the record,
    which is where a later review that withdrew the approval shows up -- so an
    approval revoked after promotion deactivates the entry on the next status.
    """
    try:
        draft = read_project_skill_draft(root, str(active["draft_id"]))
    except (SkillPromotionSourceError, KeyError) as exc:
        raise BrowserSkillPromotionError(
            "approved skill draft is unavailable or no longer an active proposal"
        ) from exc
    if skill_draft_digest(draft) != active.get("draft_digest"):
        raise BrowserSkillPromotionError("approved skill draft has drifted")


def _skill_from_receipt(root: Path, receipt: Mapping[str, object]) -> str:
    target = Path(str(receipt.get("target_path", "")))
    skill_name = target.name
    if target != _target(root, skill_name):
        raise BrowserSkillPromotionError("approval receipt has an unsafe target")
    return skill_name


def _same_identity(active: Mapping[str, object], source_id: str) -> bool:
    return _source_id(active) == source_id


def _source_id(active: Mapping[str, object]) -> str:
    """The promotion source id an installed entry names, whichever kind wrote it."""
    return str(entry_source_binding(active)["trace_id"])


def _unchanged_review(root: Path, skill_name: str, active: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "browser_skill_promotion/v1", "status": "active", "reused": True,
        "plan": {"schema_version": "browser_skill_promotion_plan/v1", "operation": "unchanged", "target_path": str(_target(root, skill_name)), "diff": "", "diff_digest": _digest(b"")},
        "receipt_id": active["receipt_id"], "generation": active["generation"],
    }


def _reuse_or_deactivate(root: Path, skill_name: str, active: Mapping[str, object]) -> dict[str, object]:
    try:
        _require_current_source(root, active)
    except BrowserSkillPromotionError as exc:
        with file_lock(_source_lock_path(root, _source_id(active)), private=True):
            with file_lock(_lock_path(root, skill_name), private=True):
                again = _active(root, skill_name)
                if again is not None and again.get("entry_digest") == active.get("entry_digest"):
                    _unlink_entry(_target(root, skill_name) / "SKILL.md")
        return _observe(root, skill_name, _source_failure_status(root, active), active, reused=False, reason=str(exc), deactivated=True)
    return _observe(root, skill_name, "active", active, reused=True)


def _source_failure_status(root: Path, active: Mapping[str, object]) -> str:
    # Only a browser trace has a quarantine lifecycle; a draft that no longer
    # resolves as an approved proposal is stale, which is the honest word for it.
    if active.get("source_kind") == SKILL_DRAFT_SOURCE:
        return "stale"
    try:
        trace = read_browser_workflow_trace(root, str(active["trace_id"]))
        lifecycle = trace.get("lifecycle")
        if isinstance(lifecycle, dict) and lifecycle.get("status") == "quarantined":
            return "quarantined"
    except BrowserTraceError:
        return "stale"
    return "stale"


def _observe(root: Path, skill_name: str, status: str, active: Mapping[str, object] | None, *, reused: bool, reason: str = "", deactivated: bool = False) -> dict[str, object]:
    receipt = _mapping(active, "receipt") if active is not None else {}
    previous = _last_observation(root, skill_name)
    same_receipt = previous is not None and previous.get("receipt_id") == receipt.get("receipt_id")
    prior = previous if same_receipt and previous is not None else {}
    payload: dict[str, object] = {
        "schema_version": "browser_skill_promotion/v1", "status": status, "skill_name": skill_name,
        "project_root": str(root), "generation": active.get("generation", "") if active else "",
        "receipt_id": receipt.get("receipt_id", ""), "promoter": receipt.get("reviewer_identity", ""),
        "promoted_at": prior.get("promoted_at", "") if same_receipt else (utc_now() if status in {"active", "rolled_back"} and not reused else ""),
        "observed_at": prior.get("observed_at", "") if same_receipt else utc_now(),
        # `trace_id` keeps its shipped payload name and carries the promotion
        # source id, which for a draft-sourced skill is that draft's id.
        "lineage": {"previous_generation": active.get("previous_generation", "") if active else "", "rollback_of": active.get("rollback_of", "") if active else "", "trace_id": _source_id(active) if active else ""},
        "reused": reused, "reason": reason, "deactivated": deactivated,
    }
    # Read/reuse paths are projections only. Persist a terminal transition or
    # a newly committed activation; never replace checked state with unchecked.
    if not reused:
        _write_observation(root, skill_name, payload)
    return payload


def _write_observation(root: Path, skill_name: str, payload: Mapping[str, object]) -> None:
    # Observation history is not consulted for activation; actual entry/index
    # bytes remain the only active truth after a crash.
    path = _state_root(root, skill_name) / "last-observation.json"
    text = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}-{secrets.token_hex(8)}.tmp")
    _write_exact(temporary, text, replace=False, private=True)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.lexists(temporary) and not temporary.is_symlink():
            temporary.unlink()


def _last_observation(root: Path, skill_name: str) -> dict[str, object] | None:
    path = _state_root(root, skill_name) / "last-observation.json"
    if not os.path.lexists(path):
        return None
    try:
        value = _read_json(path)
        return value if value.get("schema_version") == "browser_skill_promotion/v1" else None
    except (BrowserSkillPromotionError, OSError):
        return None


def _managed_inventory(root: Path, skill_name: str, *, allow_unindexed_generation: str | None = None) -> dict[str, str]:
    target = _target(root, skill_name)
    files = read_browser_skill_package(target)
    if not files:
        return {}
    allowed: set[str] = {"SKILL.md"} if "SKILL.md" in files else set()
    manifests = [name for name in files if name.startswith("resources/") and name.endswith("/manifest.json")]
    if not manifests:
        raise BrowserSkillPromotionError("existing target has unmanaged files")
    for name in manifests:
        generation = name.split("/")[1] if len(name.split("/")) == 3 else ""
        manifest = _manifest(target / name, generation)
        _verify_manifest(target, manifest)
        try:
            _validated_generation(root, skill_name, generation)
            owned = True
        except (BrowserSkillPromotionError, BrowserSkillPromotionApprovalError, OSError):
            owned = False
        if not owned and generation != allow_unindexed_generation:
            raise BrowserSkillPromotionError("retained generation is not owned by a matching approval")
        allowed.update(_mapping(manifest, "files"))
        allowed.add(name)
    if set(files) != allowed:
        raise BrowserSkillPromotionError("existing target contains unmanaged files")
    if "SKILL.md" in files and _active(root, skill_name) is None:
        raise BrowserSkillPromotionError("existing SKILL.md is not a verified managed browser skill")
    return files


def _historical_entry(root: Path, skill_name: str, generation: str) -> dict[str, object]:
    if _DIGEST.fullmatch(generation) is None:
        raise BrowserSkillPromotionError("rollback generation is invalid")
    return _validated_generation(root, skill_name, generation)


def _stage_and_verify(root: Path, skill_name: str, plan: Mapping[str, object]) -> None:
    package = _text(plan, "package")
    generation = str(plan["generation"])
    resources = {name: text for name, text in package.items() if name.startswith(f"resources/{generation}/")}
    if len(resources) != 4:
        raise BrowserSkillPromotionError("approved package does not contain one complete immutable generation")
    stage = _state_root(root, skill_name) / "staging" / str(plan["activation_id"])
    for name, text in resources.items():
        _write_exact(stage / name, text, replace=False)
    for name, text in resources.items():
        if _read_text(stage / name) != text:
            raise BrowserSkillPromotionError("private immutable staging verification failed")
    target = _target(root, skill_name)
    for name, text in resources.items():
        _write_exact(target / name, text, replace=False)
    manifest = _manifest(target / f"resources/{generation}/manifest.json", generation)
    _verify_manifest(target, manifest)


def _write_activation_index(root: Path, skill_name: str, plan: Mapping[str, object], receipt: Mapping[str, object]) -> None:
    entry_digest = str(plan["entry_digest"])
    value = {"schema_version": "browser_skill_activation/v1", "entry_digest": entry_digest, "activation_id": plan["activation_id"], "receipt_id": receipt["receipt_id"], "project_identity": plan["project_identity"], "target_relative_path": str(Path(str(plan["target_path"])).relative_to(root)), "generation": plan["generation"], "manifest_digest": plan["manifest_digest"], "payload_digest": plan["payload_digest"]}
    path = _activation_path(root, skill_name, entry_digest)
    if os.path.lexists(path):
        if _read_json(path) != value:
            raise BrowserSkillPromotionError("activation index digest collides with different bytes")
        return
    _write_exact(path, json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", replace=False, private=True)


def _replace_entry(root: Path, skill_name: str, text: str, activation_id: str) -> None:
    target = _target(root, skill_name)
    target.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise BrowserSkillPromotionError("target is a symlink")
    destination = target / "SKILL.md"
    temporary = target / f".SKILL.md.{activation_id}.tmp"
    _write_exact(temporary, text, replace=False)
    try:
        if destination.is_symlink():
            raise BrowserSkillPromotionError("SKILL.md is a symlink")
        os.replace(temporary, destination)
        _fsync_directory(target)
    finally:
        if os.path.lexists(temporary) and not temporary.is_symlink():
            temporary.unlink()


def _without_staged_generation(inventory: Mapping[str, str], plan: Mapping[str, object]) -> dict[str, str]:
    package = _text(plan, "package")
    resources = {name: text for name, text in package.items() if name.startswith(f"resources/{plan['generation']}/")}
    if resources and all(inventory.get(name) == text for name, text in resources.items()):
        return {name: text for name, text in inventory.items() if name not in resources}
    return dict(inventory)


def _drop_generation(inventory: Mapping[str, str], generation: str) -> dict[str, str]:
    prefix = f"resources/{_digest_value(generation, 'receipt generation')}/"
    return {name: text for name, text in inventory.items() if not name.startswith(prefix)}


def _entry_metadata(entry: str) -> dict[str, object]:
    """Read a managed entry's promotion metadata, whichever kind wrote it."""
    try:
        return parse_promotion_entry_metadata(entry)
    except SkillPromotionSourceError as exc:
        raise BrowserSkillPromotionError(str(exc)) from exc


def _manifest(path: Path, generation: str) -> dict[str, object]:
    value = _read_json(path)
    if value.get("schema_version") != "browser_skill_resource_manifest/v1" or value.get("generation") != generation:
        raise BrowserSkillPromotionError("resource manifest is invalid")
    files = _mapping(value, "files")
    if set(files) not in promotion_manifest_file_sets(generation) or not all(_DIGEST.fullmatch(str(digest)) for digest in files.values()):
        raise BrowserSkillPromotionError("resource manifest has invalid file bindings")
    return value


def _verify_manifest(target: Path, manifest: Mapping[str, object]) -> None:
    for name, digest in _mapping(manifest, "files").items():
        if _digest(_read_bytes(target / name)) != digest:
            raise BrowserSkillPromotionError("immutable resource digest mismatch")


def _target(root: Path, skill_name: str) -> Path:
    _validate_skill_name(skill_name)
    return _safe_under(root, (".hermes", "skills", skill_name), "project-local skill target")


def _state_root(root: Path, skill_name: str) -> Path:
    _validate_skill_name(skill_name)
    return _safe_under(root, (".omh", "browser-skill-promotions", skill_name), "promotion state")


def _safe_under(root: Path, parts: tuple[str, ...], label: str) -> Path:
    current = root
    for part in parts:
        if not part or part in {".", ".."} or "/" in part or "\\" in part:
            raise BrowserSkillPromotionError(f"unsafe {label} path")
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise BrowserSkillPromotionError(f"{label} contains a symlink")
    return current


def _validate_skill_name(skill_name: str) -> None:
    if not _SLUG.fullmatch(skill_name):
        raise BrowserSkillPromotionError("skill name must be a lowercase-hyphen project-local slug")


def _lock_path(root: Path, skill_name: str) -> Path: return _state_root(root, skill_name) / ".lock"
def _source_lock_path(root: Path, source_id: str) -> Path:
    """The source record a promotion holds open while it commits, per kind."""
    try:
        parts = promotion_source_lock_parts(source_id)
    except SkillPromotionSourceError as exc:
        raise BrowserSkillPromotionError("promotion source id is invalid") from exc
    return _safe_under(root, parts, "promotion source")
def _activation_path(root: Path, skill_name: str, entry_digest: str) -> Path: return _state_root(root, skill_name) / "activation-by-entry" / f"{_digest_value(entry_digest, 'entry digest')}.json"
def _digest_value(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None: raise BrowserSkillPromotionError(f"{label} is invalid")
    return value

def _text(value: Mapping[str, object], key: str) -> dict[str, str]:
    nested = _mapping(value, key)
    if not all(isinstance(name, str) and isinstance(text, str) for name, text in nested.items()): raise BrowserSkillPromotionError(f"{key} is malformed")
    return {str(name): str(text) for name, text in nested.items()}
def _mapping(value: Mapping[str, object], key: str) -> dict[str, object]:
    nested = value.get(key)
    if not isinstance(nested, dict): raise BrowserSkillPromotionError(f"{key} is malformed")
    return dict(nested)
def _read_bytes(path: Path) -> bytes:
    if path.is_symlink(): raise BrowserSkillPromotionError("promotion path is a symlink")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_FILE_BYTES: raise BrowserSkillPromotionError("promotion input is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = _MAX_FILE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally: os.close(descriptor)
def _read_text(path: Path) -> str:
    raw = _read_bytes(path)
    if len(raw) > _MAX_FILE_BYTES: raise BrowserSkillPromotionError("promotion input exceeds bounds")
    return raw.decode("utf-8")
def _read_json(path: Path) -> dict[str, object]:
    try: value = json.loads(_read_text(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise BrowserSkillPromotionError("promotion JSON is malformed") from exc
    if not isinstance(value, dict): raise BrowserSkillPromotionError("promotion JSON is not an object")
    return value
def _write_exact(path: Path, text: str, *, replace: bool, private: bool = False) -> None:
    for parent in (path.parent, *path.parent.parents):
        if parent == parent.parent: break
        if parent.is_symlink(): raise BrowserSkillPromotionError("promotion path is a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        if path.is_symlink() or _read_text(path) != text: raise BrowserSkillPromotionError("immutable promotion file already differs")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600 if private else 0o644)
    try:
        raw = text.encode("utf-8")
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally: os.close(descriptor)
    if _read_text(path) != text: raise BrowserSkillPromotionError("promotion file readback differs")
def _unlink_entry(path: Path) -> None:
    if path.is_symlink(): raise BrowserSkillPromotionError("SKILL.md is a symlink")
    if path.exists(): path.unlink(); _fsync_directory(path.parent)
def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        # CRT cannot open directories. Regular files are still fsynced before
        # replace; Windows directory-entry power-loss durability is not claimed.
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise BrowserSkillPromotionError("promotion directory fsync failed") from exc

def _project_identity(root: Path) -> str: return _digest(str(root).encode("utf-8"))
def _package_digest(package: Mapping[str, str]) -> str: return _digest(b"".join(name.encode("utf-8") + b"\0" + package[name].encode("utf-8") for name in sorted(package)))
def _digest(value: bytes) -> str: return hashlib.sha256(value).hexdigest()
