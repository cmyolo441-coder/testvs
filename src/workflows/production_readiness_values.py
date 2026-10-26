from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from typing import Any, Final, NoReturn, cast

READINESS_MATRIX_SCHEMA_VERSION: Final = "readiness_matrix/v1"
OBSERVED_CHECK_RESULT_SCHEMA_VERSION: Final = "observed_check_result/v1"
EXTERNAL_READINESS_EVIDENCE_SCHEMA_VERSION: Final = "external_readiness_evidence/v1"
OBSERVED_POSTCONDITION_SCHEMA_VERSION: Final = "observed_postcondition/v1"
EXTERNAL_READINESS_AUTHENTICITY_SCHEMA_VERSION: Final = "external_readiness_authenticity/v1"
EXTERNAL_READINESS_AUTHENTICITY_ALGORITHM: Final = "hmac-sha256"
READINESS_CATEGORIES: Final = (
    "build",
    "tests",
    "ci",
    "security_privacy",
    "performance",
    "observability",
    "rollback",
    "docs_support",
    "release_communication",
)
READINESS_CATEGORY_POLICY_SCHEMA_VERSION: Final = "readiness_category_policy/v1"
READINESS_CATEGORY_POLICY: Final = (
    ("build", False),
    ("tests", False),
    ("ci", True),
    ("security_privacy", False),
    ("performance", False),
    ("observability", False),
    ("rollback", False),
    ("docs_support", False),
    ("release_communication", False),
)
READINESS_EVIDENCE_STATES: Final = ("missing", "observed", "failed")
READINESS_VERDICTS: Final = ("GO", "HOLD", "BLOCK")
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
_HMAC_RE = re.compile(r"^[0-9a-f]{64}$")
_MIN_HMAC_KEY_BYTES = 32
_MAX_HMAC_KEY_BYTES = 4_096
READINESS_CANONICAL_JSON_MAX_DEPTH: Final = 16
READINESS_CANONICAL_JSON_MAX_NODES: Final = 2_048
READINESS_CANONICAL_JSON_MAX_BYTES: Final = 65_536
_CANONICAL_JSON_REJECTED: Final = object()
_READINESS_CAPTURE_ERRORS: Final = (
    "readiness_matrix signed data is not safely bounded canonical JSON",
    "external readiness evidence is unauthenticated",
    "verdict must match derived verdict HOLD",
)

@dataclass(frozen=True, slots=True)
class ValidatedReadinessArtifact:
    """Immutable authority for one accepted canonical readiness artifact."""

    canonical_bytes: bytes
    verdict: str

    def detached_copy(self) -> dict[str, Any]:
        value = json.loads(self.canonical_bytes)
        if type(value) is not dict:
            raise RuntimeError("validated readiness artifact is not an object")
        return cast(dict[str, Any], value)


@dataclass(frozen=True, slots=True)
class ReadinessValidationResult:
    """Typed outcome of capturing and validating caller-owned readiness data."""

    errors: tuple[str, ...]
    verdict: str
    artifact: ValidatedReadinessArtifact | None

    @property
    def accepted(self) -> bool:
        return self.artifact is not None and not self.errors

    def require_artifact(self) -> ValidatedReadinessArtifact:
        if self.artifact is None:
            raise ValueError("; ".join(self.errors))
        return self.artifact


class ReadinessAuthenticationError(Exception):
    """Raised when structurally unsafe data cannot be authenticated."""


class ReadinessTrustContext:
    """Caller-held HMAC state with redacted rendering and blocked standard exports.

    These supported guards do not claim resistance to arbitrary same-process introspection.
    """

    __slots__ = ("context_id", "__hmac_template", "__usable")

    def __init__(self, context_id: str, hmac_key: bytes) -> None:
        self.context_id = str(context_id)
        self.__usable = _valid_ref(self.context_id) and _MIN_HMAC_KEY_BYTES <= len(hmac_key) <= _MAX_HMAC_KEY_BYTES
        self.__hmac_template = hmac.new(hmac_key, digestmod=hashlib.sha256)

    def __repr__(self) -> str:
        return f"ReadinessTrustContext(context_id={self.context_id!r}, signer=<redacted>)"

    __str__ = __repr__

    def __format__(self, format_spec: str) -> str:
        return format(str(self), format_spec)

    def __copy__(self) -> NoReturn:
        raise TypeError("ReadinessTrustContext is opaque and non-copyable")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("ReadinessTrustContext is opaque and non-copyable")

    def __reduce__(self) -> NoReturn:
        raise TypeError("ReadinessTrustContext is opaque and non-serializable")

    def __reduce_ex__(self, protocol: object) -> NoReturn:
        raise TypeError("ReadinessTrustContext is opaque and non-serializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("ReadinessTrustContext is opaque and non-serializable")

    def _usable(self) -> bool:
        return self.__usable

    def _sign(self, payload: bytes) -> str:
        signer = self.__hmac_template.copy()
        signer.update(payload)
        return signer.hexdigest()

    def _verify(self, payload: bytes, candidate: str) -> bool:
        expected = self._sign(payload)
        return hmac.compare_digest(candidate, expected)

# Machine-consumed rollback policy. Rollback removes the new consumer and
# generated annotations; it does not rewrite persisted matrices or promote old
# operation_artifact/v1 records. Readers keep legacy bytes and evidence state.
READINESS_MATRIX_ROLLBACK_CONTRACT: Final = {
    "schema_version": "readiness_matrix_rollback/v1",
    "contract_id": READINESS_MATRIX_SCHEMA_VERSION,
    "rollback_action": "remove_consumer_and_annotations",
    "persisted_record_action": "leave_unchanged",
    "legacy_read_policy": "read_without_upgrade",
    "canonical_reliability_contract": "omh_operation_artifact/v1",
    "compatible_legacy_contracts": ["operation_artifact/v1"],
    "legacy_evidence_state_action": "preserve",
}


OBSERVED_CHECK_RESULT_KEYS: Final = {
    "schema_version",
    "check_id",
    "result",
    "observed_at",
    "evidence_ref",
    "scope",
    "category",
    "task_id",
    "revision",
    "row_id",
}


def _duplicates(values: list[str]) -> list[str]:
    return sorted({value for value in values if values.count(value) > 1})


def _valid_ref(value: object) -> bool:
    return isinstance(value, str) and bool(_REF_RE.fullmatch(value))


def _category_requires_external_observation(category: str) -> bool:
    return next((required for name, required in READINESS_CATEGORY_POLICY if name == category), False)


def _category_policy_error(category: str, requires_external: bool) -> str:
    mode = "external observation" if requires_external else "local observation"
    return f"category policy requires {mode} for {category or '<unknown>'}"


def readiness_row_id(scope: str, category: str, task_id: str, revision: str) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {"scope": scope, "category": category, "task_id": task_id, "revision": revision},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]
    return f"readiness-row-{digest}"


def _canonical_contract_ref() -> dict[str, Any]:
    return {
        "contract_id": READINESS_MATRIX_SCHEMA_VERSION,
        "enforcement_level": "executable_validated",
        "consumer_id": "parse_readiness_matrix",
        "category_policy": {
            "schema_version": READINESS_CATEGORY_POLICY_SCHEMA_VERSION,
            "categories": [
                {"category": category, "requires_external_observation": required}
                for category, required in READINESS_CATEGORY_POLICY
            ],
        },
    }


def _matrix_id(scope: str, task_id: str, revision: str, created_at: str) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {"scope": scope, "task_id": task_id, "revision": revision, "created_at": created_at},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]
    return f"readiness-{digest}"
