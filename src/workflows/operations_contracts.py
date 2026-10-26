from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Literal, Protocol, cast, overload

if TYPE_CHECKING:
    from .production_readiness import ReadinessValidationResult, validate_readiness_matrix

EnforcementLevel = Literal["executable_validated", "shared_operation_validated", "guidance_only"]
ENFORCEMENT_LEVELS: tuple[EnforcementLevel, ...] = (
    "executable_validated",
    "shared_operation_validated",
    "guidance_only",
)


@dataclass(frozen=True, slots=True)
class ArtifactContractRef:
    contract_id: str
    enforcement_level: EnforcementLevel
    consumer_id: str

    def __post_init__(self) -> None:
        if not self.contract_id.strip():
            raise ValueError("contract_id is required")
        if self.enforcement_level not in ENFORCEMENT_LEVELS:
            raise ValueError(f"unsupported enforcement_level: {self.enforcement_level}")
        if self.enforcement_level == "guidance_only" and self.consumer_id:
            raise ValueError("guidance_only contracts must not name an executable consumer")
        if self.enforcement_level != "guidance_only" and not self.consumer_id:
            raise ValueError("validated contracts require a consumer_id")


@dataclass(frozen=True, slots=True)
class OperationsArtifactContract:
    workflow_id: str
    ref: ArtifactContractRef

    @property
    def contract_id(self) -> str:
        return self.ref.contract_id

    @property
    def enforcement_level(self) -> EnforcementLevel:
        return self.ref.enforcement_level

    @property
    def consumer_id(self) -> str:
        return self.ref.consumer_id


ValidatorConsumerId = Literal[
    "validate_agent_operator_productivity_card",
    "validate_ops_service_quality_board",
    "validate_operation_artifact",
]
ReadinessConsumerId = Literal["parse_readiness_matrix"]


class ValidatorConsumer(Protocol):
    def __call__(self, record: dict[str, object]) -> list[str]: ...


class ReadinessConsumer(Protocol):
    def __call__(
        self, record: dict[str, object], *, trusted_context: object | None = None
    ) -> ReadinessValidationResult: ...


_CONSUMER_IMPORTS = {
    "validate_agent_operator_productivity_card": "omh.operator_productivity:validate_agent_operator_productivity_card",
    "validate_ops_service_quality_board": "omh.ops_service_quality:validate_ops_service_quality_board",
    "validate_operation_artifact": "omh.operations:validate_operation_artifact",
    "parse_readiness_matrix": "omh.production_readiness:parse_readiness_matrix",
}

# This is the complete operations classification. Enforcement level describes
# machine handling only; evidence state belongs to each produced artifact.
OPERATIONS_ARTIFACT_CONTRACTS = (
    OperationsArtifactContract(
        "agent-ops-review",
        ArtifactContractRef(
            "agent_operator_productivity/v1", "executable_validated", "validate_agent_operator_productivity_card"
        ),
    ),
    OperationsArtifactContract(
        "ops-observability-card",
        ArtifactContractRef("ops_service_quality_board/v1", "executable_validated", "validate_ops_service_quality_board"),
    ),
    OperationsArtifactContract(
        "reliability-review",
        ArtifactContractRef("omh_operation_artifact/v1", "shared_operation_validated", "validate_operation_artifact"),
    ),
    OperationsArtifactContract(
        "production-audit",
        ArtifactContractRef("readiness_matrix/v1", "executable_validated", "parse_readiness_matrix"),
    ),
    OperationsArtifactContract(
        "build-failure-triage", ArtifactContractRef("build_failure_triage_plan/v1", "guidance_only", "")
    ),
    OperationsArtifactContract(
        "security-safety-review", ArtifactContractRef("security_safety_review_plan/v1", "guidance_only", "")
    ),
    OperationsArtifactContract(
        "failure-signal-audit", ArtifactContractRef("failure_signal_audit_plan/v1", "guidance_only", "")
    ),
    OperationsArtifactContract("ops-review", ArtifactContractRef("ops-review", "guidance_only", "")),
    OperationsArtifactContract("deploy-and-monitor", ArtifactContractRef("deploy-and-monitor", "guidance_only", "")),
    OperationsArtifactContract("support-operations", ArtifactContractRef("support-operations", "guidance_only", "")),
)

_CONTRACT_BY_WORKFLOW = {item.workflow_id: item for item in OPERATIONS_ARTIFACT_CONTRACTS}


def artifact_contracts_for_workflow(workflow_id: str) -> tuple[ArtifactContractRef, ...]:
    item = _CONTRACT_BY_WORKFLOW.get(workflow_id)
    return (item.ref,) if item else ()


@overload
def resolve_artifact_contract_consumer(consumer_id: ReadinessConsumerId) -> ReadinessConsumer: ...


@overload
def resolve_artifact_contract_consumer(consumer_id: ValidatorConsumerId) -> ValidatorConsumer: ...


@overload
def resolve_artifact_contract_consumer(
    consumer_id: str,
) -> ValidatorConsumer | ReadinessConsumer: ...


def resolve_artifact_contract_consumer(
    consumer_id: str,
) -> ValidatorConsumer | ReadinessConsumer:
    import_path = _CONSUMER_IMPORTS.get(consumer_id)
    if import_path is None:
        raise LookupError(f"unknown artifact contract consumer: {consumer_id}")
    module_name, attribute = import_path.split(":", 1)
    consumer = getattr(import_module(module_name), attribute, None)
    if not callable(consumer):
        raise LookupError(f"artifact contract consumer is not callable: {consumer_id}")
    if consumer_id == "parse_readiness_matrix":
        return cast(ReadinessConsumer, consumer)
    return cast(ValidatorConsumer, consumer)


if TYPE_CHECKING:
    _typed_readiness_consumer: ReadinessConsumer = resolve_artifact_contract_consumer(
        "parse_readiness_matrix"
    )
    _typed_agent_validator: ValidatorConsumer = resolve_artifact_contract_consumer(
        "validate_agent_operator_productivity_card"
    )
    _typed_quality_validator: ValidatorConsumer = resolve_artifact_contract_consumer(
        "validate_ops_service_quality_board"
    )
    _typed_operation_validator: ValidatorConsumer = resolve_artifact_contract_consumer(
        "validate_operation_artifact"
    )
    _typed_direct_readiness_validator: ValidatorConsumer = validate_readiness_matrix


def validate_operations_artifact_contracts() -> list[str]:
    errors: list[str] = []
    workflow_ids = [item.workflow_id for item in OPERATIONS_ARTIFACT_CONTRACTS]
    duplicates = sorted({workflow_id for workflow_id in workflow_ids if workflow_ids.count(workflow_id) > 1})
    if duplicates:
        errors.append(f"duplicate operations workflow classifications: {', '.join(duplicates)}")
    for item in OPERATIONS_ARTIFACT_CONTRACTS:
        if item.enforcement_level == "guidance_only":
            continue
        try:
            _ = resolve_artifact_contract_consumer(item.consumer_id)
        except LookupError as exc:
            errors.append(str(exc))
    return errors
