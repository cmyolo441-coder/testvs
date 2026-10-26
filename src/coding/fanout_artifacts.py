from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Mapping

from ..plugin_bundle.omh.run_records import CONTRACT_PROVENANCE_FILE, DISPATCH_SUMMARY_FILE, FANOUT_CONTRACT_FILE
from ..system.local_store import atomic_write_json
from ..system.local_store import ensure_dir
from ..system.paths import OmhPaths
from .fanout_contracts import FANOUT_ID_PATTERN
from .fanout_contracts import FANOUT_CONTRACT_PROVENANCE_SCHEMA_VERSION

_FANOUT_ID_RE = re.compile(FANOUT_ID_PATTERN)
# The same slug shape `fanout._UNIT_ID_RE` accepts, restated here rather than
# imported so the path validator does not depend on the contract builder.
_UNIT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_PROVENANCE_KEYS = {
    "schema_version",
    "fanout_id",
    "contract_schema_version",
    "contract_sha256",
    "privacy",
}


def write_fanout_contract(paths: OmhPaths, contract: dict[str, object]) -> dict[str, object]:
    fanout_id = _validated_fanout_id(contract.get("fanout_id"))
    contract_dir = _managed_fanout_dir(paths, fanout_id)
    contract_path = contract_dir / FANOUT_CONTRACT_FILE
    ensure_dir(paths.fanout_contracts_dir, private=True)
    ensure_dir(contract_dir, private=True)

    payload = deepcopy(contract)
    payload["artifacts"] = {"contract_path": str(contract_path), "privacy": "metadata_only"}
    atomic_write_json(contract_path, payload, private=True)
    atomic_write_json(
        contract_dir / CONTRACT_PROVENANCE_FILE,
        {
            "schema_version": FANOUT_CONTRACT_PROVENANCE_SCHEMA_VERSION,
            "fanout_id": fanout_id,
            "contract_schema_version": str(payload.get("schema_version", "")),
            "contract_sha256": fanout_contract_digest(payload),
            "privacy": "metadata_only",
        },
        private=True,
    )
    return payload


def fanout_dispatch_summary_path(paths: OmhPaths, fanout_id: str) -> Path:
    """Validated dispatch-summary path for one fanout (id pattern + containment)."""
    return _managed_fanout_dir(paths, _validated_fanout_id(fanout_id)) / DISPATCH_SUMMARY_FILE


def fanout_run_journal_path(paths: OmhPaths, fanout_id: str) -> Path:
    """Validated run-journal path for one fanout (id pattern + containment)."""
    return _managed_fanout_dir(paths, _validated_fanout_id(fanout_id)) / "run_journal.json"


def fanout_review_dispatch_budget_path(paths: OmhPaths, fanout_id: str) -> Path:
    """Validated durable review-dispatch budget path for one fanout goal."""
    return _managed_fanout_dir(paths, _validated_fanout_id(fanout_id)) / "review_dispatch_budget.json"


def fanout_contract_provenance_path(paths: OmhPaths, fanout_id: str) -> Path:
    return _managed_fanout_dir(
        paths,
        _validated_fanout_id(fanout_id),
    ) / CONTRACT_PROVENANCE_FILE


def unit_result_path(paths: OmhPaths, fanout_id: str, unit_id: str) -> Path:
    """Validated sidecar path for one executor-reported unit result."""
    if not _UNIT_ID_RE.match(str(unit_id or "")):
        raise ValueError(f"invalid unit_id: {unit_id!r}")
    result_dir = _managed_fanout_dir(paths, _validated_fanout_id(fanout_id)) / "unit_results"
    if result_dir.is_symlink():
        raise ValueError("fanout unit-result directory must not be a symlink")
    return result_dir / f"{unit_id}.json"


def fanout_unit_recovery_path(paths: OmhPaths, fanout_id: str, unit_id: str) -> Path:
    """Validated recovery-record path for one unit of one fanout.

    The unit id rides through the same slug check the contract builder applies,
    so a crafted id cannot walk out of the fanout directory.
    """
    if not _UNIT_ID_RE.match(str(unit_id or "")):
        raise ValueError(f"invalid unit_id: {unit_id!r}")
    recovery_dir = _managed_fanout_dir(paths, _validated_fanout_id(fanout_id)) / "recovery"
    if recovery_dir.is_symlink():
        raise ValueError("fanout recovery directory must not be a symlink")
    return recovery_dir / f"{unit_id}.json"


def write_fanout_unit_recovery(
    paths: OmhPaths,
    fanout_id: str,
    unit_id: str,
    record: dict[str, object],
) -> Path:
    """Persist one unit's metadata-only recovery record and return its path."""
    recovery_path = fanout_unit_recovery_path(paths, fanout_id, unit_id)
    ensure_dir(paths.fanout_contracts_dir, private=True)
    ensure_dir(recovery_path.parent.parent, private=True)
    ensure_dir(recovery_path.parent, private=True)
    atomic_write_json(recovery_path, record, private=True)
    return recovery_path


def clear_fanout_unit_recovery(paths: OmhPaths, fanout_id: str, unit_id: str) -> None:
    """Drop a unit's stored recovery record, if it has one.

    Called before a unit re-runs, so a record describing an earlier attempt
    cannot outlive the worktree it points at. Same posture as the in-flight
    marker: best effort, never blocks a dispatch.
    """
    try:
        fanout_unit_recovery_path(paths, fanout_id, unit_id).unlink(missing_ok=True)
    except (OSError, ValueError):
        return


def read_fanout_contract(paths: OmhPaths, fanout_id: str) -> dict[str, object]:
    contract_path = _managed_fanout_dir(paths, _validated_fanout_id(fanout_id)) / FANOUT_CONTRACT_FILE
    return json.loads(contract_path.read_text(encoding="utf-8"))


def read_fanout_contract_provenance(
    paths: OmhPaths,
    fanout_id: str,
    contract: Mapping[str, object],
) -> dict[str, object]:
    validated_id = _validated_fanout_id(fanout_id)
    provenance_path = fanout_contract_provenance_path(paths, validated_id)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise ValueError("fanout contract schema provenance must be an object")
    validate_fanout_contract_provenance(
        contract,
        provenance,
        expected_fanout_id=validated_id,
    )
    return provenance


def fanout_contract_digest(contract: Mapping[str, object]) -> str:
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def validate_fanout_contract_provenance(
    contract: Mapping[str, object],
    provenance: Mapping[str, object],
    *,
    expected_fanout_id: str | None = None,
) -> None:
    if set(provenance) != _PROVENANCE_KEYS:
        raise ValueError("fanout contract schema provenance has unsupported fields")
    if provenance.get("schema_version") != FANOUT_CONTRACT_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("fanout contract schema provenance is unsupported")
    fanout_id = str(contract.get("fanout_id", ""))
    if provenance.get("fanout_id") != fanout_id or (
        expected_fanout_id is not None and fanout_id != expected_fanout_id
    ):
        raise ValueError("fanout contract schema provenance fanout_id does not match")
    if provenance.get("contract_schema_version") != contract.get("schema_version"):
        raise ValueError("fanout contract schema provenance does not match the contract")
    if provenance.get("privacy") != "metadata_only":
        raise ValueError("fanout contract schema provenance privacy must be metadata_only")
    digest = provenance.get("contract_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("fanout contract schema provenance digest is invalid")
    if digest != fanout_contract_digest(contract):
        raise ValueError("fanout contract schema provenance digest does not match")


def _validated_fanout_id(value: object) -> str:
    fanout_id = str(value or "")
    if not _FANOUT_ID_RE.match(fanout_id):
        raise ValueError(f"invalid fanout_id: {fanout_id!r}")
    return fanout_id


def _managed_fanout_dir(paths: OmhPaths, fanout_id: str) -> Path:
    root = paths.fanout_contracts_dir
    if root.is_symlink():
        raise ValueError("fanout contract storage must not be a symlink")
    root_resolved = root.resolve(strict=False)
    if not root_resolved.is_relative_to(paths.omh_home.resolve(strict=False)):
        raise ValueError("fanout contract storage must resolve under OMH home")

    contract_dir = root / fanout_id
    if contract_dir.is_symlink():
        raise ValueError("fanout contract directory must not be a symlink")
    if contract_dir.resolve(strict=False).parent != root_resolved:
        raise ValueError("fanout_id must resolve under the fanout contracts directory")
    return contract_dir
