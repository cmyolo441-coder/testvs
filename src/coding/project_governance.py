from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..system.metadata_safety import is_body_shaped_metadata_text, is_sensitive_metadata_text
from ..workflows.domain_intelligence_bounded_json import decode_bounded_json_object


PROJECT_GOVERNANCE_DISCOVERY_SCHEMA_VERSION = "project_governance_discovery/v1"
PROJECT_GOVERNANCE_PROFILE_SCHEMA_VERSION = "project_governance_profile/v1"
PROJECT_GOVERNANCE_BLOCKED_SCHEMA_VERSION = "project_governance_blocked/v1"
GOVERNANCE_DECISIONS = ("not_applicable", "accept", "decline")
_MAX_BYTES = 65_536
_CANDIDATES = (
    ("AGENTS.md", "instructions"), ("CLAUDE.md", "instructions"), ("CONTRIBUTING.md", "qa"),
    (".github/pull_request_template.md", "pr"), (".github/PULL_REQUEST_TEMPLATE.md", "pr"),
    (".github/workflows/ci.yml", "ci"), (".github/workflows/ci.yaml", "ci"),
)
_POLICIES = (
    ("tests_required", "verification", ("test", "tests"), ("no tests", "skip tests")),
    ("lint_required", "verification", ("lint",), ("no lint", "skip lint")),
    ("typecheck_required", "verification", ("typecheck", "type check", "pyright", "mypy"), ("no typecheck", "skip typecheck")),
    ("review_required", "review", ("review",), ("no review", "skip review")),
    ("dco_required", "review", ("dco", "signed-off-by"), ("no dco", "skip dco")),
)
_SENSITIVE_RE = re.compile(
    r"(?:\bsk-[A-Za-z0-9_-]{8,}|(?:api[_-]?key|aws_secret_access_key|password|token|secret)\s*[:=]\s*[^\s]+|-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----)",
    re.IGNORECASE,
)

# --- Org safety rule source (issue #802) ------------------------------------
#
# A second bounded local reader in the same idiom as `_read_sources`: closed
# field set, byte cap, per-source sha256, symlink rejection, closed reason-code
# vocabulary. Two things are new. It is bounded in TIME as well as size, and it
# is fail-closed on every failure mode -- there is no "unavailable therefore
# fine" branch, because an org rule that cannot be read must never be read as
# permission. The caller (`quality/safety_preflight.py`) turns any
# `status != "available"` into a denial.
#
# Issue #805 adds one question and deliberately keeps it outside this reader:
# whether a local attestation over the source bytes verifies. That is not the
# same question as whether the document is readable, so it lives in
# `coding/safety_rule_attestation.py` and reaches this module only as three more
# refusal codes plus the two result builders at the bottom of this section.
ORG_SAFETY_RULE_SOURCE_SCHEMA_VERSION = "omh_org_safety_rules/v1"
ORG_SAFETY_RULE_SOURCE_RESULT_SCHEMA_VERSION = "omh_org_safety_rule_source/v1"
ORG_SAFETY_RULE_SOURCE_MAX_BYTES = 16_384
ORG_SAFETY_RULE_SOURCE_TIME_BUDGET_SECONDS = 0.5
ORG_SAFETY_RULE_SOURCE_CLAIM_BOUNDARY = (
    "An org safety rule source is bounded local metadata; it is not compliance, execution, "
    "review, CI, or merge evidence."
)
ORG_SAFETY_RULE_SOURCE_REASON_CODES = (
    "org_source_available",
    "org_source_missing",
    "org_source_unsafe",
    "org_source_unreadable",
    "org_source_oversized",
    "org_source_timed_out",
    "org_source_malformed",
    "org_source_unknown_version",
    "org_source_unknown_fields",
    "org_source_unsafe_metadata",
    # The local attestation refusals (issue #805). They live in this vocabulary
    # rather than a second one because they are the same kind of answer -- this
    # source may not be activated, and here is the field that says why -- and
    # because `quality/safety_preflight` already turns every non-available code
    # in this tuple into a denial. They are NOT the attestation axis itself;
    # `coding/safety_rule_attestation.py` owns that, and it stays a separate
    # field so a valid tag over a malformed document and an invalid tag over a
    # well-formed one stay two distinguishable answers.
    "org_source_attestation_missing",
    "org_source_attestation_invalid",
    "org_source_attestation_key_unreadable",
)
_ORG_SOURCE_FIELDS = frozenset({"schema_version", "revision", "denied_remote_target_kinds", "max_target_paths"})
_ORG_SOURCE_CHUNK_BYTES = 4_096
_ORG_SOURCE_MAX_DEPTH = 4
_ORG_SOURCE_MAX_NODES = 128
_ORG_SOURCE_MAX_TOKENS = 16
_ORG_SOURCE_MAX_TOKEN_CHARS = 120
_ORG_SOURCE_MAX_TARGET_PATHS = 1_024
_ORG_SOURCE_PATH_FIELD = "org_rule_source.source_path"


def discover_project_governance(project_root: str | Path, *, decision: str = "not_applicable") -> dict[str, object]:
    if decision not in GOVERNANCE_DECISIONS:
        raise ValueError(f"unsupported governance decision: {decision}")
    root = Path(project_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("project root must be a non-symlink directory")
    resolved = root.resolve(strict=True)
    sources, policies, unsafe = _read_sources(resolved)
    root_hash = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()
    if unsafe:
        return _blocked(root_hash, sources, unsafe)
    if not sources:
        return _pending_or_default(root_hash, decision)
    if decision != "not_applicable":
        raise ValueError("governance decision is only valid when no existing governance is found")
    conflicts = _conflicts(policies)
    if conflicts:
        return _blocked(root_hash, sources, conflicts)
    return _existing(root_hash, sources, policies)


def governance_handoff_attachment(discovery: dict[str, object]) -> dict[str, dict[str, object]]:
    status = discovery.get("status")
    if status in {"existing_project_governance", "accepted_advisory_defaults"}:
        profile = discovery.get("profile")
        return {"project_governance_profile": profile} if isinstance(profile, dict) else {}
    if status == "blocked":
        blocked = discovery.get("blocked")
        return {"project_governance_blocked": blocked} if isinstance(blocked, dict) else {}
    return {}


def validate_project_governance_profile(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["project governance profile must be an object"]
    required = {"schema_version", "origin", "status", "root_identity_sha256", "sources", "expectations", "claim_boundary"}
    errors: list[str] = []
    if set(value) != required:
        errors.append("project governance profile keys are invalid")
    if value.get("schema_version") != PROJECT_GOVERNANCE_PROFILE_SCHEMA_VERSION:
        errors.append("project governance profile schema_version is invalid")
    if value.get("origin") not in {"existing_project", "advisory_default"}:
        errors.append("project governance profile origin is invalid")
    if value.get("status") != "prepared_not_observed":
        errors.append("project governance profile status is invalid")
    if not _sha(value.get("root_identity_sha256")):
        errors.append("project governance profile root identity is invalid")
    if not _sources_valid(value.get("sources")) or not _expectations_valid(value.get("expectations"), value.get("sources")):
        errors.append("project governance profile sources or expectations are invalid")
    if not _claim_boundary_valid(value.get("claim_boundary")):
        errors.append("project governance profile claim boundary is invalid")
    return errors


def validate_project_governance_blocked(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["project governance blocked marker must be an object"]
    required = {"schema_version", "status", "root_identity_sha256", "sources", "conflict_codes", "claim_boundary"}
    if set(value) != required or value.get("schema_version") != PROJECT_GOVERNANCE_BLOCKED_SCHEMA_VERSION or value.get("status") != "blocked":
        return ["project governance blocked marker is invalid"]
    if not _sha(value.get("root_identity_sha256")) or not _sources_valid(value.get("sources")):
        return ["project governance blocked marker source metadata is invalid"]
    codes = value.get("conflict_codes")
    if not isinstance(codes, list) or not codes or any(code not in _conflict_codes() for code in codes):
        return ["project governance blocked marker conflicts are invalid"]
    if not _claim_boundary_valid(value.get("claim_boundary")):
        return ["project governance blocked marker claim boundary is invalid"]
    return []


def read_org_safety_rule_source(
    source_path: str | Path,
    *,
    time_budget_seconds: float = ORG_SAFETY_RULE_SOURCE_TIME_BUDGET_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Read the opt-in org safety rule source under a size and a time bound.

    Local file only: no network, no new dependency, no model. Every failure
    mode returns `status: "unavailable"` with its own reason code and the
    offending field, so the caller can deny and say why. There is no branch
    that reports an unreadable source as permission.
    """
    result, _ = read_org_safety_rule_source_with_bytes(
        source_path, time_budget_seconds=time_budget_seconds, clock=clock
    )
    return result


def read_org_safety_rule_source_with_bytes(
    source_path: str | Path,
    *,
    time_budget_seconds: float = ORG_SAFETY_RULE_SOURCE_TIME_BUDGET_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[dict[str, object], bytes]:
    """The reader above, plus the exact bytes it read.

    The local attestation lane (issue #805) authenticates the SOURCE BYTES
    rather than the parsed document, and it has to authenticate the same read
    the rules came from: opening the file a second time would tag one read and
    then apply another. The bytes are empty exactly when the read itself failed,
    which is every case where the result is already unavailable for a
    source-side reason. A document that read cleanly but parsed badly still
    returns its bytes, so a caller can report the tag over those bytes and the
    parse failure as the two separate answers they are.
    """
    identity = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()
    if not str(source_path).strip():
        # Opted in without a configured path: still a denial, never an
        # implicit "off". Turning the org level off is a separate decision.
        return org_safety_rule_source_unavailable(identity, "org_source_missing", _ORG_SOURCE_PATH_FIELD), b""
    path = Path(source_path)
    started = clock()
    if path.is_symlink():
        return org_safety_rule_source_unavailable(identity, "org_source_unsafe", _ORG_SOURCE_PATH_FIELD), b""
    if not path.exists():
        return org_safety_rule_source_unavailable(identity, "org_source_missing", _ORG_SOURCE_PATH_FIELD), b""
    if not path.is_file():
        return org_safety_rule_source_unavailable(identity, "org_source_unsafe", _ORG_SOURCE_PATH_FIELD), b""
    chunks: list[bytes] = []
    total = 0
    try:
        with path.open("rb") as handle:
            while True:
                if clock() - started > time_budget_seconds:
                    return org_safety_rule_source_unavailable(identity, "org_source_timed_out", _ORG_SOURCE_PATH_FIELD), b""
                chunk = handle.read(_ORG_SOURCE_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > ORG_SAFETY_RULE_SOURCE_MAX_BYTES:
                    return org_safety_rule_source_unavailable(identity, "org_source_oversized", _ORG_SOURCE_PATH_FIELD), b""
                chunks.append(chunk)
    except FileNotFoundError:
        return org_safety_rule_source_unavailable(identity, "org_source_missing", _ORG_SOURCE_PATH_FIELD), b""
    except OSError:
        return org_safety_rule_source_unavailable(identity, "org_source_unreadable", _ORG_SOURCE_PATH_FIELD), b""
    if clock() - started > time_budget_seconds:
        return org_safety_rule_source_unavailable(identity, "org_source_timed_out", _ORG_SOURCE_PATH_FIELD), b""
    raw = b"".join(chunks)
    return _org_source_document(identity, raw), raw


def _org_source_document(identity: str, raw: bytes) -> dict[str, object]:
    try:
        document = decode_bounded_json_object(raw, max_depth=_ORG_SOURCE_MAX_DEPTH, max_nodes=_ORG_SOURCE_MAX_NODES)
    except ValueError:
        return org_safety_rule_source_unavailable(identity, "org_source_malformed", "org_rule_source.document")
    if document.get("schema_version") != ORG_SAFETY_RULE_SOURCE_SCHEMA_VERSION:
        return org_safety_rule_source_unavailable(identity, "org_source_unknown_version", "org_rule_source.schema_version")
    unknown = sorted(set(document) - _ORG_SOURCE_FIELDS)
    if unknown:
        return org_safety_rule_source_unavailable(identity, "org_source_unknown_fields", f"org_rule_source.{unknown[0]}")
    revision = document.get("revision")
    if not isinstance(revision, str) or not revision.strip():
        return org_safety_rule_source_unavailable(identity, "org_source_malformed", "org_rule_source.revision")
    kinds = document.get("denied_remote_target_kinds", [])
    if not _org_bounded_tokens(kinds):
        return org_safety_rule_source_unavailable(identity, "org_source_malformed", "org_rule_source.denied_remote_target_kinds")
    cap = document.get("max_target_paths")
    if cap is not None and (not isinstance(cap, int) or isinstance(cap, bool) or not 0 <= cap <= _ORG_SOURCE_MAX_TARGET_PATHS):
        return org_safety_rule_source_unavailable(identity, "org_source_malformed", "org_rule_source.max_target_paths")
    unsafe_field = _org_unsafe_metadata_field(revision, kinds)
    if unsafe_field:
        return org_safety_rule_source_unavailable(identity, "org_source_unsafe_metadata", unsafe_field)
    return org_safety_rule_source_available(
        identity,
        content_sha256=hashlib.sha256(raw).hexdigest(),
        revision=revision,
        rules={"denied_remote_target_kinds": sorted(set(kinds)), "max_target_paths": cap},
    )


def _org_bounded_tokens(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) <= _ORG_SOURCE_MAX_TOKENS
        and all(isinstance(item, str) and item.strip() and len(item) <= _ORG_SOURCE_MAX_TOKEN_CHARS for item in value)
    )


def _org_unsafe_metadata_field(revision: str, kinds: list[str]) -> str:
    candidates = [("revision", revision)]
    candidates.extend((f"denied_remote_target_kinds[{index}]", item) for index, item in enumerate(kinds))
    for name, text in candidates:
        if is_sensitive_metadata_text(text) or is_body_shaped_metadata_text(text, limit=_ORG_SOURCE_MAX_TOKEN_CHARS):
            return f"org_rule_source.{name}"
    return ""


def org_safety_rule_source_available(
    identity: str, *, content_sha256: str, revision: str, rules: dict[str, object]
) -> dict[str, object]:
    """The available result shape, built in one place so it cannot drift.

    Public for the local attestation lane (issue #805): when a new revision
    fails verification, that lane serves the last revision that did verify, out
    of local trust state rather than out of the file. The retained rule set has
    to reach the evaluator in exactly the shape the reader produces, not in a
    lookalike a second literal would drift away from.
    """
    return {
        "schema_version": ORG_SAFETY_RULE_SOURCE_RESULT_SCHEMA_VERSION,
        "status": "available",
        "reason_code": "org_source_available",
        "field": "",
        "source_identity_sha256": identity,
        "content_sha256": content_sha256,
        "revision": revision,
        "rules": dict(rules),
        "claim_boundary": ORG_SAFETY_RULE_SOURCE_CLAIM_BOUNDARY,
    }


def org_safety_rule_source_unavailable(identity: str, reason_code: str, field: str) -> dict[str, object]:
    """The unavailable result shape, naming its reason code and offending field.

    Public for the same reason. An attestation failure is a refusal of the same
    kind the reader already makes -- the document parsed, but the local tag over
    its bytes did not verify -- so it speaks this vocabulary rather than opening
    a second one. The reason code is checked because a code outside the tuple
    would reach `safety_preflight._CORRECTIONS` with no correction to render.
    """
    if reason_code not in ORG_SAFETY_RULE_SOURCE_REASON_CODES:
        raise ValueError(f"unsupported org rule source reason code: {reason_code}")
    return {
        "schema_version": ORG_SAFETY_RULE_SOURCE_RESULT_SCHEMA_VERSION,
        "status": "unavailable",
        "reason_code": reason_code,
        "field": field,
        "source_identity_sha256": identity,
        "content_sha256": "",
        "revision": "",
        "rules": {},
        "claim_boundary": ORG_SAFETY_RULE_SOURCE_CLAIM_BOUNDARY,
    }


def _read_sources(root: Path) -> tuple[list[dict[str, object]], list[tuple[str, str, str, str, str]], list[str]]:
    sources: list[dict[str, object]] = []
    policies: list[tuple[str, str, str, str, str]] = []
    unsafe: list[str] = []
    seen: set[tuple[int, int]] = set()
    for relative, category in _candidate_pairs(root):
        path = root
        escaped = False
        for part in Path(relative).parts:
            path = path / part
            if path.is_symlink():
                escaped = True
                break
        if escaped:
            unsafe.append("unsafe_source")
            continue
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file() or path.resolve().parent != (root / relative).resolve().parent:
            unsafe.append("unsafe_source")
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        data = path.read_bytes()
        if len(data) > _MAX_BYTES:
            unsafe.append("unsafe_source")
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            unsafe.append("unsafe_source")
            continue
        if _SENSITIVE_RE.search(text):
            unsafe.append("unsafe_source")
            continue
        sources.append({"category": category, "path": relative, "sha256": hashlib.sha256(data).hexdigest(), "byte_length": len(data)})
        extracted = _extract(category, relative, text)
        if any(item[4] == "both" for item in extracted):
            unsafe.append("unsafe_source")
        policies.extend(extracted)
    return sorted(sources, key=lambda item: (str(item["category"]), str(item["path"]))), policies, sorted(set(unsafe))


def _extract(category: str, path: str, text: str) -> list[tuple[str, str, str, str, str]]:
    normalized = " ".join(text.lower().split())
    result: list[tuple[str, str, str, str, str]] = []
    for identifier, expectation_category, positive, negative in _POLICIES:
        has_negative = any(_matches(normalized, term) for term in negative)
        without_negation = normalized
        for term in negative:
            without_negation = re.sub(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", " ", without_negation)
        has_positive = any(_matches(without_negation, term) for term in positive)
        if has_positive or has_negative:
            result.append((category, path, identifier, expectation_category, "both" if has_positive and has_negative else "negative" if has_negative else "positive"))
    if category == "pr":
        result.append((category, path, "pr_template_present", "pr", "positive"))
    if category == "ci":
        result.append((category, path, "ci_workflow_present", "ci", "positive"))
    return result


def _matches(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _candidate_pairs(root: Path) -> list[tuple[str, str]]:
    pairs = list(_CANDIDATES)
    for directory, category, suffixes in (
        (".github/PULL_REQUEST_TEMPLATE", "pr", (".md",)),
        (".github/workflows", "ci", (".yml", ".yaml")),
    ):
        path = root / directory
        if path.is_dir() and not path.is_symlink():
            pairs.extend((item.relative_to(root).as_posix(), category) for item in sorted(path.iterdir()) if item.suffix in suffixes)
        elif path.is_symlink():
            pairs.append((directory, category))
    return pairs


def _conflicts(policies: list[tuple[str, str, str, str, str]]) -> list[str]:
    conflicts: set[str] = set()
    for category in {item[0] for item in policies}:
        for identifier in {item[2] for item in policies if item[0] == category}:
            values = {item[4] for item in policies if item[0] == category and item[2] == identifier}
            if "positive" in values and "negative" in values:
                conflicts.add(f"{category}_policy_conflict")
    return sorted(conflicts)


def _existing(root_hash: str, sources: list[dict[str, object]], policies: list[tuple[str, str, str, str, str]]) -> dict[str, object]:
    expectations = _expectations(policies, sources, "existing_project")
    return _discovery("existing_project_governance", "not_applicable", root_hash, sources, expectations, profile=_profile("existing_project", root_hash, sources, expectations))


def _pending_or_default(root_hash: str, decision: str) -> dict[str, object]:
    if decision == "decline":
        return _discovery("declined_advisory_defaults", "declined", root_hash, [], [])
    if decision == "accept":
        expectations = _default_expectations()
        return _discovery("accepted_advisory_defaults", "accepted", root_hash, [], expectations, profile=_profile("advisory_default", root_hash, [], expectations))
    return _discovery("pending_user_decision", "pending_user_decision", root_hash, [], _default_expectations())


def _blocked(root_hash: str, sources: list[dict[str, object]], codes: list[str]) -> dict[str, object]:
    blocked = {"schema_version": PROJECT_GOVERNANCE_BLOCKED_SCHEMA_VERSION, "status": "blocked", "root_identity_sha256": root_hash, "sources": sources, "conflict_codes": codes, "claim_boundary": "Blocked governance metadata is not compliance, execution, review, CI, or merge evidence."}
    return _discovery("blocked", "blocked", root_hash, sources, [], blocked=blocked, codes=codes)


def _discovery(status: str, decision: str, root_hash: str, sources: list[dict[str, object]], expectations: list[dict[str, object]], *, profile: dict[str, object] | None = None, blocked: dict[str, object] | None = None, codes: list[str] | None = None) -> dict[str, object]:
    return {"schema_version": PROJECT_GOVERNANCE_DISCOVERY_SCHEMA_VERSION, "root_identity_sha256": root_hash, "status": status, "decision": decision, "sources": sources, "expectations": expectations, "conflict_codes": codes or [], "profile": profile, "blocked": blocked, "claim_boundary": "Discovery is local prepared metadata, not compliance, execution, review, CI, or merge evidence."}


def _profile(origin: str, root_hash: str, sources: list[dict[str, object]], expectations: list[dict[str, object]]) -> dict[str, object]:
    return {"schema_version": PROJECT_GOVERNANCE_PROFILE_SCHEMA_VERSION, "origin": origin, "status": "prepared_not_observed", "root_identity_sha256": root_hash, "sources": sources, "expectations": expectations, "claim_boundary": "Prepared governance context is not compliance, execution, review, CI, or merge evidence."}


def _expectations(policies: list[tuple[str, str, str, str, str]], sources: list[dict[str, object]], precedence: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for category, path, identifier, expectation_category, state in policies:
        if state == "positive":
            grouped.setdefault((identifier, expectation_category), []).append(path)
    return [
        {"id": identifier, "category": category, "source_paths": sorted(set(source_paths)), "precedence": precedence}
        for (identifier, category), source_paths in sorted(grouped.items())
    ]


def _default_expectations() -> list[dict[str, object]]:
    return [{"id": identifier, "category": category, "source_paths": [], "precedence": "advisory_default"} for identifier, category in (("tests_required", "verification"), ("lint_required", "verification"), ("review_required", "review"), ("dco_required", "review"))]


def _sources_valid(value: Any) -> bool:
    return isinstance(value, list) and len(value) <= 8 and all(isinstance(item, dict) and set(item) == {"category", "path", "sha256", "byte_length"} and item.get("category") in {"instructions", "qa", "pr", "ci"} and isinstance(item.get("path"), str) and not item["path"].startswith("/") and ".." not in item["path"].split("/") and _sha(item.get("sha256")) and isinstance(item.get("byte_length"), int) and 0 <= item["byte_length"] <= _MAX_BYTES for item in value)


def _expectations_valid(value: Any, sources: Any) -> bool:
    known_paths = {item["path"] for item in sources} if isinstance(sources, list) else set()
    return isinstance(value, list) and len(value) <= 16 and all(isinstance(item, dict) and set(item) == {"id", "category", "source_paths", "precedence"} and item.get("id") in {item[0] for item in _POLICIES} | {"pr_template_present", "ci_workflow_present"} and item.get("category") in {"verification", "review", "pr", "ci"} and item.get("precedence") in {"existing_project", "advisory_default"} and isinstance(item.get("source_paths"), list) and all(path in known_paths for path in item["source_paths"]) for item in value)


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _claim_boundary_valid(value: Any) -> bool:
    text = str(value).lower()
    return all(term in text for term in ("not compliance", "execution", "review", "ci", "merge"))


def _conflict_codes() -> set[str]:
    return {"instructions_policy_conflict", "qa_policy_conflict", "pr_policy_conflict", "ci_policy_conflict", "unsafe_source"}
