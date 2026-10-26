from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import TYPE_CHECKING

from .domain_intelligence_contracts import (
    SAFE_CANDIDATE_ID,
    SAFE_PROFILE_ID,
    ensure_no_forbidden_keys,
)
from .domain_intelligence_json_io import read_stable_json_at, stable_file_identity
from .domain_intelligence_lineage import ProfileValidationContext
from .domain_intelligence_review_validation import (
    canonical_reason_code,
    canonical_reviewer_claim,
    validate_review_artifact_for_status,
)
from .domain_intelligence_schema import REJECTED_REVIEW_KEYS, validate_review_contract
from .domain_intelligence_snapshot_budget import DomainSnapshotBudget
from .domain_intelligence_store_security import MAX_DOMAIN_ARTIFACT_FILES
from .domain_intelligence_validation import validate_profile_artifact_for_resolution

if TYPE_CHECKING:
    from .domain_project_context import HostProjectBinding


_HEALTH_DIRECTORIES = ("profiles", "reviews", "history")


@dataclass(frozen=True)
class _DirectorySnapshot:
    identity: tuple[int, int, int, int, int, int]
    manifest: tuple[tuple[str, int, int, int, int, int, int], ...]
    total_bytes: int


def read_validated_domain_profiles_at(
    binding: object,
) -> tuple[dict[str, object], ...]:
    """Read one complete, stable, descriptor-bound profile health universe."""
    from .domain_project_context import HostProjectBinding

    if not isinstance(binding, HostProjectBinding):
        raise ValueError("host_project_binding_required")
    with binding.shared_store_lock(), ExitStack() as stack:
        directories = {
            name: stack.enter_context(binding.open_directory(name))
            for name in _HEALTH_DIRECTORIES
        }
        before = {
            name: _snapshot_directory(descriptor)
            for name, descriptor in directories.items()
        }
        for name, descriptor in directories.items():
            _require_bound_directory(binding.domain_store_fd, name, descriptor)
        budget = DomainSnapshotBudget()
        budget.require_total_bytes(
            sum(snapshot.total_bytes for snapshot in before.values())
        )
        records = _read_budgeted_records(directories, before, budget)
        profiles = _validate_resolution_records(binding, records)
        after = {
            name: _snapshot_directory(descriptor)
            for name, descriptor in directories.items()
        }
        for name, descriptor in directories.items():
            _require_bound_directory(binding.domain_store_fd, name, descriptor)
        if before != after:
            raise ValueError("domain_profile_snapshot_changed")
        return profiles


def _read_budgeted_records(
    directories: dict[str, int | Path],
    snapshots: dict[str, _DirectorySnapshot],
    budget: DomainSnapshotBudget,
) -> dict[str, tuple[tuple[str, dict[str, object]], ...]]:
    records: dict[str, tuple[tuple[str, dict[str, object]], ...]] = {}
    for name in _HEALTH_DIRECTORIES:
        values: list[tuple[str, dict[str, object]]] = []
        for filename, *_rest in snapshots[name].manifest:
            value = read_stable_json_at(directories[name], filename)
            budget.consume_json(value)
            values.append((filename, value))
        records[name] = tuple(values)
    return records


def _snapshot_directory(directory_fd: int | Path) -> _DirectorySnapshot:
    directory_stat = (
        os.fstat(directory_fd)
        if isinstance(directory_fd, int)
        else os.stat(directory_fd, follow_symlinks=False)
    )
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise ValueError("domain_health_directory_invalid")
    names = _bounded_json_names(directory_fd)
    manifest: list[tuple[str, int, int, int, int, int, int]] = []
    total_bytes = 0
    for name in names:
        item = (
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if isinstance(directory_fd, int)
            else os.stat(directory_fd / name, follow_symlinks=False)
        )
        if not stat.S_ISREG(item.st_mode):
            raise ValueError("symlink_or_not_file")
        manifest.append((name, *stable_file_identity(item)))
        total_bytes += item.st_size
    return _DirectorySnapshot(
        identity=stable_file_identity(directory_stat),
        manifest=tuple(manifest),
        total_bytes=total_bytes,
    )


def _bounded_json_names(directory_fd: int | Path) -> tuple[str, ...]:
    names: list[str] = []
    scan_limit = max(MAX_DOMAIN_ARTIFACT_FILES * 2 + 1, 1)
    scanned = 0
    with os.scandir(directory_fd) as entries:
        for entry in entries:
            scanned += 1
            if scanned > scan_limit:
                raise ValueError("artifact_file_count_exceeded")
            if entry.name.endswith(".json"):
                names.append(entry.name)
                if len(names) > MAX_DOMAIN_ARTIFACT_FILES:
                    raise ValueError("artifact_file_count_exceeded")
    return tuple(sorted(names))


def _require_bound_directory(
    root_fd: int | Path,
    name: str,
    directory_fd: int | Path,
) -> None:
    bound = (
        os.fstat(directory_fd)
        if isinstance(directory_fd, int)
        else os.stat(directory_fd, follow_symlinks=False)
    )
    current = (
        os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if isinstance(root_fd, int)
        else os.stat(root_fd / name, follow_symlinks=False)
    )
    if stable_file_identity(bound) != stable_file_identity(current):
        raise ValueError("domain_health_directory_changed")


def _validate_resolution_records(
    binding: HostProjectBinding,
    records: dict[str, tuple[tuple[str, dict[str, object]], ...]],
) -> tuple[dict[str, object], ...]:
    profiles = _identity_records(records["profiles"], "profile_id")
    reviews = _identity_records(records["reviews"], "review_id")
    history = _history_records(records["history"])
    all_profiles = (*profiles, *history)
    profile_index = _unique_profile_index(all_profiles)
    review_index = {str(review["review_id"]): review for review in reviews}
    context = ProfileValidationContext(
        history={_profile_identity(profile): profile for profile in history},
        candidates={},
        reviews=review_index,
    )
    paths = binding.project_paths
    for profile in all_profiles:
        validate_profile_artifact_for_resolution(paths, profile, context=context)
    for review in reviews:
        if review.get("decision") == "rejected":
            _validate_rejected_review_without_candidate(review)
        else:
            validate_review_artifact_for_status(
                review,
                candidates={},
                profiles=profile_index,
            )
    return tuple(profile for profile in profiles if profile.get("status") == "active")


def _identity_records(
    records: tuple[tuple[str, dict[str, object]], ...], identity_field: str
) -> tuple[dict[str, object], ...]:
    values: list[dict[str, object]] = []
    identities: set[str] = set()
    for filename, value in records:
        ensure_no_forbidden_keys(value)
        identity = value.get(identity_field)
        if (
            not isinstance(identity, str)
            or not identity
            or filename != f"{identity}.json"
        ):
            raise ValueError("artifact_identity_mismatch")
        if identity in identities:
            raise ValueError("duplicate_embedded_id")
        identities.add(identity)
        values.append(value)
    return tuple(values)


def _history_records(
    records: tuple[tuple[str, dict[str, object]], ...],
) -> tuple[dict[str, object], ...]:
    values: list[dict[str, object]] = []
    identities: set[tuple[str, int]] = set()
    for filename, value in records:
        ensure_no_forbidden_keys(value)
        identity = _profile_identity(value)
        if filename != f"{identity[0]}_r{identity[1]}.json":
            raise ValueError("artifact_identity_mismatch")
        if identity in identities:
            raise ValueError("duplicate_embedded_id")
        identities.add(identity)
        values.append(value)
    return tuple(values)


def _unique_profile_index(
    profiles: tuple[dict[str, object], ...],
) -> dict[tuple[str, int], dict[str, object]]:
    index: dict[tuple[str, int], dict[str, object]] = {}
    for profile in profiles:
        key = _profile_identity(profile)
        if key in index:
            raise ValueError("duplicate_embedded_id")
        index[key] = profile
    return index


def _profile_identity(profile: dict[str, object]) -> tuple[str, int]:
    profile_id = profile.get("profile_id")
    revision = profile.get("revision")
    if (
        not isinstance(profile_id, str)
        or not SAFE_PROFILE_ID.fullmatch(profile_id)
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
    ):
        raise ValueError("artifact_identity_mismatch")
    return profile_id, revision


def _validate_rejected_review_without_candidate(review: dict[str, object]) -> None:
    ensure_no_forbidden_keys(review)
    validate_review_contract(review, REJECTED_REVIEW_KEYS)
    candidate_id = review.get("candidate_id")
    profile_id = review.get("profile_id")
    if (
        not isinstance(candidate_id, str)
        or not SAFE_CANDIDATE_ID.fullmatch(candidate_id)
        or review.get("review_id") != f"direview_{candidate_id}"
        or not isinstance(profile_id, str)
        or not SAFE_PROFILE_ID.fullmatch(profile_id)
        or review.get("revision") is not None
    ):
        raise ValueError("review_identity_mismatch")
    canonical_reviewer_claim(review.get("reviewer_claim"))
    canonical_reason_code(review.get("reason_code"))
