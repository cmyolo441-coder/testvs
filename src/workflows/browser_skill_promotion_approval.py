"""Native-preflight and exact-diff approval admission for browser skills.

This is deliberately not an installer.  It produces one durable receipt after
rechecking the current trace, target bytes, and native policy.  Activation,
rollback, removal, and status belong to the later lifecycle group.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol

from ..install.plugin_loader_observation import _find_hermes_python

from ..system.local_store import file_lock
from .browser_skill_promotion_plan import (
    build_skill_promotion_plan,
    read_browser_skill_package,
)
from .browser_workflow_learning_store import _root as observed_git_root
from .skill_promotion_source import receipt_source_binding


PROMOTION_APPROVAL_RECEIPT_SCHEMA_VERSION = "browser_skill_promotion_approval_receipt/v1"
PROMOTION_NATIVE_PREFLIGHT_SCHEMA_VERSION = "browser_skill_promotion_native_preflight/v1"
_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_REVIEWER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@:-]{0,127}$")


class BrowserSkillPromotionApprovalError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NativeWritePolicy:
    requirement: Literal["not_required", "required"]
    approval: Literal["not_applicable", "not_obtained"]
    revision: str
    support: Literal["available", "unsupported"]


@dataclass(frozen=True)
class NativePromotionPreflight:
    """Evidence returned by a host collaborator, never a bare grant boolean."""

    schema_version: str
    project_root: str
    package_digest: str
    trusted: bool
    structure_error: str | None
    lint_errors: tuple[str, ...]
    security_verdict: Literal["safe", "caution", "dangerous"]
    policy: NativeWritePolicy


class PromotionNativeHost(Protocol):
    """Read-only collaborator supplied by Hermes or a fixed host probe.

    A future non-Hermes CLI must obtain this record through one fixed,
    read-only probe executed by the installed Hermes interpreter.  It must not
    provide a general subprocess bridge or synthesize a boolean grant.
    """

    def inspect(
        self, project_root: Path, package: Mapping[str, str]
    ) -> NativePromotionPreflight: ...


class HermesPromotionNativeHost:
    """Closed read-only bridge to the installed Hermes interpreter.

    The normal OMH interpreter deliberately does not import Hermes.  This
    adapter only runs the shipped fixed probe with the discovered Hermes venv
    and source checkout; it cannot execute caller-provided commands or code.
    """

    def inspect(
        self, project_root: Path, package: Mapping[str, str]
    ) -> NativePromotionPreflight:
        python, source = _fixed_hermes_runtime()
        probe = Path(__file__).with_name("browser_skill_promotion_native_probe.py")
        if not probe.is_file():
            raise BrowserSkillPromotionApprovalError("Hermes native promotion probe is unavailable")
        request = _canonical({
            "schema_version": PROMOTION_NATIVE_PREFLIGHT_SCHEMA_VERSION,
            "project_root": str(project_root),
            "package": dict(package),
        })
        if len(request) > 512 * 1024:
            raise BrowserSkillPromotionApprovalError("native promotion request exceeds its byte bound")
        environment = {
            key: os.environ[key]
            for key in ("HOME", "HERMES_HOME", "HERMES_MANAGED_DIR", "PATH")
            if key in os.environ
        }
        try:
            completed = subprocess.run(
                [str(python), str(probe)],
                cwd=source,
                input=request,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BrowserSkillPromotionApprovalError("Hermes native promotion probe is unavailable") from exc
        if completed.returncode != 0:
            raise BrowserSkillPromotionApprovalError("Hermes native promotion probe is unavailable")
        return _decode_native_preflight(completed.stdout)


def _fixed_hermes_runtime() -> tuple[Path, Path]:
    python = _find_hermes_python()
    if python is None:
        raise BrowserSkillPromotionApprovalError("Hermes native promotion APIs are unavailable")
    python = python.expanduser().absolute()
    source = python.parent.parent.parent
    if (
        python.parts[-3:] != ("venv", "bin", "python")
        or not python.is_file()
        or not (source / "agent" / "skill_utils.py").is_file()
        or not (source / "tools" / "skills_guard.py").is_file()
        or not (source / "hermes_cli" / "config.py").is_file()
    ):
        raise BrowserSkillPromotionApprovalError("Hermes native promotion APIs are unavailable")
    return python, source


def _decode_native_preflight(raw: bytes) -> NativePromotionPreflight:
    prefix = b"OMH_BROWSER_SKILL_PROMOTION_NATIVE_PREFLIGHT="
    lines = [line for line in raw.splitlines() if line.startswith(prefix)]
    if len(lines) != 1 or len(lines[0]) > 512 * 1024:
        raise BrowserSkillPromotionApprovalError("Hermes native promotion probe returned no valid observation")
    try:
        value = json.loads(lines[0][len(prefix):].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrowserSkillPromotionApprovalError("Hermes native promotion probe returned no valid observation") from exc
    if isinstance(value, dict) and set(value) == {"error", "detail"} and value.get("error") == "native preflight unavailable":
        raise BrowserSkillPromotionApprovalError("native write policy is unavailable")
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "project_root", "package_digest", "trusted",
        "structure_error", "lint_errors", "security_verdict", "policy",
    }:
        raise BrowserSkillPromotionApprovalError("Hermes native promotion probe returned no valid observation")
    policy = value.get("policy")
    lint_errors = value.get("lint_errors")
    if (
        not isinstance(policy, dict)
        or set(policy) != {"requirement", "approval", "support", "revision"}
        or not isinstance(lint_errors, list)
        or not all(isinstance(item, str) for item in lint_errors)
        or not isinstance(value.get("project_root"), str)
        or not isinstance(value.get("package_digest"), str)
        or not isinstance(value.get("trusted"), bool)
        or value.get("structure_error") is not None and not isinstance(value.get("structure_error"), str)
        or value.get("security_verdict") not in {"safe", "caution", "dangerous"}
        or policy.get("requirement") not in {"not_required", "required"}
        or policy.get("approval") not in {"not_applicable", "not_obtained"}
        or policy.get("support") not in {"available", "unsupported"}
        or not isinstance(policy.get("revision"), str)
    ):
        raise BrowserSkillPromotionApprovalError("Hermes native promotion probe returned no valid observation")
    return NativePromotionPreflight(
        schema_version=value["schema_version"],
        project_root=value["project_root"],
        package_digest=value["package_digest"],
        trusted=value["trusted"],
        structure_error=value["structure_error"],
        lint_errors=tuple(lint_errors),
        security_verdict=value["security_verdict"],
        policy=NativeWritePolicy(
            requirement=policy["requirement"],
            approval=policy["approval"],
            support=policy["support"],
            revision=policy["revision"],
        ),
    )


def review_browser_skill_promotion(
    project_root: str | Path,
    source_id: str,
    skill_name: str,
    *,
    operation: str = "install",
    previous_generation: str | None = None,
    base_entry_digest: str = "absent",
    rollback_of: str | None = None,
    existing_files: Mapping[str, str] | None = None,
    host: PromotionNativeHost | None = None,
) -> dict[str, object]:
    """Render and inspect the exact package an operator must review."""
    root = observed_git_root(project_root)
    plan, preflight = _current_review(root, source_id, skill_name, host or HermesPromotionNativeHost(), operation=operation, previous_generation=previous_generation, base_entry_digest=base_entry_digest, rollback_of=rollback_of, existing_files=existing_files)
    return {"plan": plan, "native_preflight": _preflight_payload(preflight), "native_preflight_digest": _preflight_digest(preflight)}


def approve_browser_skill_promotion(
    project_root: str | Path,
    source_id: str,
    skill_name: str,
    *,
    reviewed_diff_digest: str,
    reviewer_identity: str,
    operation: str = "install",
    previous_generation: str | None = None,
    base_entry_digest: str = "absent",
    rollback_of: str | None = None,
    existing_files: Mapping[str, str] | None = None,
    host: PromotionNativeHost | None = None,
) -> dict[str, object]:
    """Save one receipt only when the caller names the currently reviewed diff."""
    if _DIGEST.fullmatch(reviewed_diff_digest) is None:
        raise BrowserSkillPromotionApprovalError("reviewed diff digest is invalid")
    if _REVIEWER.fullmatch(reviewer_identity) is None:
        raise BrowserSkillPromotionApprovalError("reviewer identity is invalid")

    root = observed_git_root(project_root)
    collaborator = host or HermesPromotionNativeHost()

    # Do not create a receipt directory or lock sidecar until a complete,
    # read-only review succeeded.
    plan, preflight = _current_review(root, source_id, skill_name, collaborator, operation=operation, previous_generation=previous_generation, base_entry_digest=base_entry_digest, rollback_of=rollback_of, existing_files=existing_files)
    if reviewed_diff_digest != plan["diff_digest"]:
        raise BrowserSkillPromotionApprovalError("reviewed diff digest is not the current exact diff")

    receipt = _receipt(plan, preflight, reviewer_identity)
    path = _receipt_path(root, str(receipt["receipt_id"]))
    with file_lock(path, private=True):
        # The source, target package, scanner/linter output, trust decision,
        # and native policy are all re-resolved after acquiring the receipt
        # lock.  A stale review cannot become a persisted approval.
        current_plan, current_preflight = _current_review(
            root, source_id, skill_name, collaborator, operation=operation,
            previous_generation=previous_generation, base_entry_digest=base_entry_digest,
            rollback_of=rollback_of, existing_files=existing_files,
        )
        if reviewed_diff_digest != current_plan["diff_digest"]:
            raise BrowserSkillPromotionApprovalError(
                "reviewed diff digest is not the current exact diff"
            )
        current_receipt = _receipt(
            current_plan, current_preflight, reviewer_identity
        )
        if current_receipt != receipt:
            raise BrowserSkillPromotionApprovalError(
                "promotion review changed before approval was persisted"
            )
        if _lexists(path):
            stored = read_browser_skill_promotion_approval_receipt(
                root, str(receipt["receipt_id"])
            )
            if stored != receipt:
                raise BrowserSkillPromotionApprovalError(
                    "promotion receipt id collides with different bytes"
                )
            return stored
        _atomic_json(path, receipt)
    return receipt


def read_browser_skill_promotion_approval_receipt(
    project_root: str | Path, receipt_id: str
) -> dict[str, object]:
    root = observed_git_root(project_root)
    path = _receipt_path(root, receipt_id)
    if not _lexists(path):
        raise BrowserSkillPromotionApprovalError("promotion approval receipt was not found")
    value = _read_json(path)
    required = {
        "schema_version",
        "receipt_id",
        "operation",
        "activation_id",
        "payload_digest",
        "rollback_of",
        "base_digest",
        "base_entry_digest",
        "previous_generation",
        "reviewer_identity",
        "reviewed_diff_digest",
        "project_root",
        "project_identity",
        "target_path",
        "trace_id",
        "trace_revision",
        "trace_digest",
        "fixture_digests",
        "generic_draft_digest",
        "generation",
        "package_digest",
        "entry_digest",
        "manifest_digest",
        "base_package_digest",
        "native_preflight_digest",
        "policy_revision",
    }
    if set(value) != required or value.get("schema_version") != PROMOTION_APPROVAL_RECEIPT_SCHEMA_VERSION:
        raise BrowserSkillPromotionApprovalError("promotion approval receipt has an invalid schema")
    if value.get("receipt_id") != receipt_id:
        raise BrowserSkillPromotionApprovalError("promotion approval receipt id does not match its path")
    if value.get("project_root") != str(root):
        raise BrowserSkillPromotionApprovalError("promotion approval receipt belongs to another project")
    binding = {key: item for key, item in value.items() if key != "receipt_id"}
    if _digest(_canonical(binding)) != receipt_id:
        raise BrowserSkillPromotionApprovalError("promotion approval receipt digest is invalid")
    return value


def _current_review(
    root: Path, source_id: str, skill_name: str, host: PromotionNativeHost, *,
    operation: str, previous_generation: str | None, base_entry_digest: str,
    rollback_of: str | None, existing_files: Mapping[str, str] | None,
) -> tuple[dict[str, object], NativePromotionPreflight]:
    # Lifecycle callers pass a verified managed inventory.  The public review
    # path retains a read-only fallback for initial adoption checks.
    initial = build_skill_promotion_plan(root, source_id, skill_name, operation=operation, previous_generation=previous_generation, base_entry_digest=base_entry_digest, rollback_of=rollback_of, existing_files=existing_files)
    existing = dict(existing_files) if existing_files is not None else read_browser_skill_package(Path(str(initial["target_path"])))
    plan = build_skill_promotion_plan(root, source_id, skill_name, operation=operation, previous_generation=previous_generation, base_entry_digest=base_entry_digest, rollback_of=rollback_of, existing_files=existing)
    package = _string_map(plan, "package")
    preflight = host.inspect(root, package)
    _validate_preflight(root, package, preflight)
    return plan, preflight


def _validate_preflight(
    root: Path, package: Mapping[str, str], preflight: NativePromotionPreflight
) -> None:
    if not isinstance(preflight, NativePromotionPreflight):
        raise BrowserSkillPromotionApprovalError("native preflight must be a typed host observation")
    if preflight.schema_version != PROMOTION_NATIVE_PREFLIGHT_SCHEMA_VERSION:
        raise BrowserSkillPromotionApprovalError("native preflight schema is unsupported")
    if preflight.project_root != str(root):
        raise BrowserSkillPromotionApprovalError("native preflight belongs to another project root")
    if preflight.package_digest != _package_digest(package):
        raise BrowserSkillPromotionApprovalError("native preflight does not bind the reviewed package")
    if not preflight.trusted:
        raise BrowserSkillPromotionApprovalError(
            "project is not trusted by Hermes; promotion never auto-trusts"
        )
    if preflight.structure_error is not None:
        raise BrowserSkillPromotionApprovalError(
            f"native skill structure check failed: {preflight.structure_error}"
        )
    if preflight.lint_errors:
        raise BrowserSkillPromotionApprovalError("native skill lint reported an error")
    if preflight.security_verdict not in {"safe", "caution"}:
        raise BrowserSkillPromotionApprovalError(
            "native skill security scan quarantined the reviewed package"
        )
    if (
        preflight.policy.requirement != "not_required"
        or preflight.policy.approval != "not_applicable"
        or preflight.policy.support != "available"
        or _DIGEST.fullmatch(preflight.policy.revision) is None
    ):
        if preflight.policy.requirement == "required":
            raise BrowserSkillPromotionApprovalError(
                "native skill-write approval is required but unsupported for project-local promotion"
            )
        raise BrowserSkillPromotionApprovalError("native write policy is unavailable")



def _receipt(
    plan: Mapping[str, object],
    preflight: NativePromotionPreflight,
    reviewer_identity: str,
) -> dict[str, object]:
    source = _mapping(plan, "source")
    # `trace_id`, `trace_revision`, `trace_digest` and `fixture_digests` are the
    # shipped `/v1` names of the receipt's source binding and stay frozen, so a
    # receipt written before #1571 still reads. `receipt_source_binding` is the
    # one place a draft source fills them with its own id and content digest.
    source_binding = receipt_source_binding(source)
    binding = {
        "schema_version": PROMOTION_APPROVAL_RECEIPT_SCHEMA_VERSION,
        "operation": plan["operation"],
        "activation_id": plan["activation_id"],
        "payload_digest": plan["payload_digest"],
        "rollback_of": plan["rollback_of"],
        "base_digest": plan["base_digest"],
        "base_entry_digest": plan["base_entry_digest"],
        "previous_generation": source["previous_generation"],
        "reviewer_identity": reviewer_identity,
        "reviewed_diff_digest": plan["diff_digest"],
        "project_root": plan["project_root"],
        "project_identity": plan["project_identity"],
        "target_path": plan["target_path"],
        "trace_id": source_binding["trace_id"],
        "trace_revision": source_binding["trace_revision"],
        "trace_digest": source_binding["trace_digest"],
        "fixture_digests": source_binding["fixture_digests"],
        "generic_draft_digest": plan["generic_draft_digest"],
        "generation": plan["generation"],
        "package_digest": plan["package_digest"],
        "entry_digest": plan["entry_digest"],
        "manifest_digest": plan["manifest_digest"],
        "base_package_digest": plan["base_package_digest"],
        "native_preflight_digest": _preflight_digest(preflight),
        "policy_revision": preflight.policy.revision,
    }
    return {**binding, "receipt_id": _digest(_canonical(binding))}


def _preflight_payload(preflight: NativePromotionPreflight) -> dict[str, object]:
    return asdict(preflight)


def _preflight_digest(preflight: NativePromotionPreflight) -> str:
    return _digest(_canonical(_preflight_payload(preflight)))


def _receipt_path(root: Path, receipt_id: str) -> Path:
    if _DIGEST.fullmatch(receipt_id) is None:
        raise BrowserSkillPromotionApprovalError("promotion receipt id is invalid")
    return _safe_path(
        root,
        (".omh", "browser-skill-promotions", "receipts", f"{receipt_id}.json"),
    )


def _safe_path(root: Path, parts: tuple[str, ...]) -> Path:
    current = root
    for index, part in enumerate(parts):
        if not part or part in {".", ".."} or "/" in part or "\\" in part:
            raise BrowserSkillPromotionApprovalError("unsafe promotion receipt path")
        current = current / part
        if _lexists(current) and current.is_symlink():
            raise BrowserSkillPromotionApprovalError(
                "promotion receipt symlink escape refused"
            )
        if index < len(parts) - 1 and current.exists() and not current.is_dir():
            raise BrowserSkillPromotionApprovalError(
                "promotion receipt parent is not a directory"
            )
    return current


def _stage_file(root: Path, relative: str, text: str) -> None:
    parts = tuple(Path(relative).parts)
    if (
        not parts
        or Path(relative).is_absolute()
        or any(part in {".", ".."} for part in parts)
    ):
        raise BrowserSkillPromotionApprovalError("reviewed package has an unsafe path")
    destination = root.joinpath(*parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8", newline="")


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for parent in (path.parent, *path.parent.parents):
        if _lexists(parent) and parent.is_symlink():
            raise BrowserSkillPromotionApprovalError(
                "promotion receipt symlink escape refused"
            )
    temporary = path.with_name(f".{path.name}.{os.getpid()}-{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink():
            raise BrowserSkillPromotionApprovalError(
                "promotion receipt symlink escape refused"
            )
        os.replace(temporary, path)
    finally:
        if _lexists(temporary) and not temporary.is_symlink():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(_read_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrowserSkillPromotionApprovalError(
            "promotion approval receipt is malformed"
        ) from exc
    if not isinstance(value, dict):
        raise BrowserSkillPromotionApprovalError(
            "promotion approval receipt is not an object"
        )
    return value


def _read_bytes(path: Path) -> bytes:
    if path.is_symlink():
        raise BrowserSkillPromotionApprovalError("promotion receipt symlink escape refused")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 262144:
            raise BrowserSkillPromotionApprovalError("promotion input is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = 262145
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining == 0:
            raise BrowserSkillPromotionApprovalError("promotion input exceeds its byte bound")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _string_map(value: Mapping[str, object], key: str) -> dict[str, str]:
    mapping = _mapping(value, key)
    if not all(isinstance(name, str) and isinstance(text, str) for name, text in mapping.items()):
        raise BrowserSkillPromotionApprovalError(f"promotion {key} is malformed")
    return {str(name): str(text) for name, text in mapping.items()}


def _mapping(value: Mapping[str, object], key: str) -> dict[str, object]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise BrowserSkillPromotionApprovalError(f"promotion {key} is malformed")
    return dict(nested)


def _package_digest(package: Mapping[str, str]) -> str:
    return _digest(
        b"".join(
            name.encode("utf-8") + b"\0" + package[name].encode("utf-8")
            for name in sorted(package)
        )
    )


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
