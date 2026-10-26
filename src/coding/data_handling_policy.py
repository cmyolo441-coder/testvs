"""Gate a model chain on the declared data-handling policy of its models.

The only exclusion OMH had before this was `excluded_providers` at provider
granularity plus a per-machine reorder of `model-chains.json`. Neither is
grounded in a policy, so nothing could keep work off a model whose training or
retention behaviour nobody had read. This module is that gate, and it is a
pure function over contract data: no file, no process, no network.

Two properties are the point rather than caveats.

**Sensitivity is a declared field, never inferred.** `work_is_sensitive` is a
boolean the caller passes. Nothing here reads a user message, and nothing here
matches text -- `quality/safety_preflight.py` says in its own header why a
safety decision that depends on user wording is the failure, and this gate is
built the same way. A caller that never declares sensitivity gets its chain
back unchanged with `applied: false`, so the gate cannot fire by accident.

**Unknown is excluded, and named.** A model whose contract declares no
data-handling axis, or declares it `not_recorded`, is dropped exactly like one
whose documented default conflicts -- but under its own reason code, because
"nobody read the policy" and "the policy says the data is trained on" are
different facts and an operator acts differently on each. There is no
permitting-by-default path; silence is not permission.

What a `documented_permitting` verdict is NOT: permission from this account's
vendor agreement. The contract carries the vendor's published default for the
surface it was read from, and `DATA_HANDLING_ACCOUNT_SCOPE` says that tier,
region, and contract move it in both directions. So the verdict narrows a
chain to the models whose published default permits the work; confirming the
account's own agreement stays with the operator.
"""

from __future__ import annotations

from typing import Any, Final, Iterable, Mapping, Sequence

from .model_contracts import (
    DATA_HANDLING_ACCOUNT_SCOPE,
    MODEL_CONTRACTS,
    RETENTION_BOUNDED,
    RETENTION_NONE,
    RETENTION_NOT_RECORDED,
    TRAINING_USE_EXCLUDED,
    TRAINING_USE_NOT_RECORDED,
    model_contract,
)

DATA_HANDLING_VERDICT_SCHEMA_VERSION: Final[str] = "model_data_handling_verdict/v1"
SENSITIVE_WORK_CHAIN_SCHEMA_VERSION: Final[str] = "sensitive_work_chain/v1"

VERDICT_PERMITTING: Final[str] = "documented_permitting"
VERDICT_CONFLICTING: Final[str] = "documented_conflicting"
VERDICT_UNKNOWN: Final[str] = "unknown"
DATA_HANDLING_VERDICTS: Final[tuple[str, ...]] = (
    VERDICT_PERMITTING,
    VERDICT_CONFLICTING,
    VERDICT_UNKNOWN,
)

REASON_DOCUMENTED_DEFAULT_PERMITS: Final[str] = "documented_default_permits"
REASON_TRAINING_USE_INCLUDES: Final[str] = "documented_default_trains_on_input"
REASON_RETENTION_INDEFINITE: Final[str] = "documented_default_retains_indefinitely"
REASON_NO_DOCUMENTED_CONTRACT: Final[str] = "no_documented_contract"
REASON_AXIS_NOT_DECLARED: Final[str] = "data_handling_axis_not_declared"
REASON_POLICY_NOT_RECORDED: Final[str] = "data_handling_policy_not_recorded"

REASON_TEXT: Final[dict[str, str]] = {
    REASON_DOCUMENTED_DEFAULT_PERMITS: (
        "The contract's documented default keeps the input out of training and names a "
        "bounded retention. This is the vendor's published default, not this account's "
        "agreement."
    ),
    REASON_TRAINING_USE_INCLUDES: (
        "The contract's documented default uses the input for training, so declared-sensitive "
        "work is excluded from this model."
    ),
    REASON_RETENTION_INDEFINITE: (
        "The contract's documented default retains the input indefinitely, so declared-sensitive "
        "work is excluded from this model."
    ),
    REASON_NO_DOCUMENTED_CONTRACT: (
        "No exact, declared, or dated-snapshot contract resolves for this model, so no "
        "data-handling policy has been read for it at all. Excluded for unknown policy, not "
        "omitted."
    ),
    REASON_AXIS_NOT_DECLARED: (
        "A contract resolves for this model but declares no `data_handling` axis. Excluded for "
        "unknown policy; add the axis to the contract from the vendor's data-usage page."
    ),
    REASON_POLICY_NOT_RECORDED: (
        "The contract declares the axis with a value that was not read from any vendor page. "
        "Excluded for unknown policy; nothing here treats an unread policy as a permitting one."
    ),
}

SENSITIVE_WORK_CHAIN_CLAIM_BOUNDARY: Final[str] = (
    "A data-handling verdict reads the vendor's documented default as recorded in the model "
    f"contract on its `sources_read` date. {DATA_HANDLING_ACCOUNT_SCOPE}. A "
    "`documented_permitting` verdict is therefore not permission from this account's "
    "agreement, not evidence that a provider serves the model to this account, and not "
    "evidence that any work ran. Sensitivity is whatever the caller declared; nothing here "
    "inspects a user message to decide it."
)


def model_data_handling_verdict(
    model_id: str,
    *,
    contracts: Mapping[str, Mapping[str, object]] = MODEL_CONTRACTS,
) -> dict[str, Any]:
    """The data-handling verdict for one model id, with the reason behind it."""
    contract = _contract_for(model_id, contracts)
    if contract is None:
        return _verdict(model_id, VERDICT_UNKNOWN, REASON_NO_DOCUMENTED_CONTRACT)
    axis = contract.get("data_handling")
    if not isinstance(axis, Mapping):
        return _verdict(model_id, VERDICT_UNKNOWN, REASON_AXIS_NOT_DECLARED)
    training_use = str(axis.get("training_use", TRAINING_USE_NOT_RECORDED))
    retention = str(axis.get("retention", RETENTION_NOT_RECORDED))
    if training_use == TRAINING_USE_NOT_RECORDED or retention == RETENTION_NOT_RECORDED:
        return _verdict(
            model_id,
            VERDICT_UNKNOWN,
            REASON_POLICY_NOT_RECORDED,
            training_use=training_use,
            retention=retention,
        )
    # Both rungs must hold. Checked in this order so the reported reason is
    # the training answer when both are wrong: an operator asked to accept a
    # retention window reads differently from one told the input trains a
    # model, and only one of those is usually negotiable.
    if training_use != TRAINING_USE_EXCLUDED:
        reason = REASON_TRAINING_USE_INCLUDES
    elif retention not in (RETENTION_NONE, RETENTION_BOUNDED):
        reason = REASON_RETENTION_INDEFINITE
    else:
        return _verdict(
            model_id,
            VERDICT_PERMITTING,
            REASON_DOCUMENTED_DEFAULT_PERMITS,
            training_use=training_use,
            retention=retention,
        )
    return _verdict(
        model_id,
        VERDICT_CONFLICTING,
        reason,
        training_use=training_use,
        retention=retention,
    )


def data_handling_filtered_chain(
    model_ids: Sequence[str] | Iterable[str],
    *,
    work_is_sensitive: bool,
    contracts: Mapping[str, Mapping[str, object]] = MODEL_CONTRACTS,
) -> dict[str, Any]:
    """The chain a declared-sensitive request may use, plus what it dropped and why.

    `work_is_sensitive` is the declaration. False returns the chain unchanged
    with `applied: false` and no verdicts, because a gate that also fires on
    undeclared work would make every route depend on a policy nobody asked for.
    """
    requested = [str(model_id or "").strip() for model_id in model_ids]
    requested = [model_id for model_id in requested if model_id]
    if not work_is_sensitive:
        return {
            "schema_version": SENSITIVE_WORK_CHAIN_SCHEMA_VERSION,
            "work_is_sensitive": False,
            "applied": False,
            "chain": list(requested),
            "entries": [],
            "excluded": [],
            "summary": {
                "requested": len(requested),
                "permitted": len(requested),
                "excluded_conflicting_policy": 0,
                "excluded_unknown_policy": 0,
            },
            "reason_codes": dict(REASON_TEXT),
            "claim_boundary": SENSITIVE_WORK_CHAIN_CLAIM_BOUNDARY,
        }
    entries = [model_data_handling_verdict(model_id, contracts=contracts) for model_id in requested]
    kept = [entry["model"] for entry in entries if entry["verdict"] == VERDICT_PERMITTING]
    excluded = [entry for entry in entries if entry["verdict"] != VERDICT_PERMITTING]
    return {
        "schema_version": SENSITIVE_WORK_CHAIN_SCHEMA_VERSION,
        "work_is_sensitive": True,
        "applied": True,
        "chain": kept,
        "entries": entries,
        "excluded": excluded,
        "summary": {
            "requested": len(requested),
            "permitted": len(kept),
            # Split rather than one `excluded` count: an unread policy is a
            # gap in the contracts, a conflicting one is a property of the
            # model, and the two are closed by different work.
            "excluded_conflicting_policy": sum(1 for row in excluded if row["verdict"] == VERDICT_CONFLICTING),
            "excluded_unknown_policy": sum(1 for row in excluded if row["verdict"] == VERDICT_UNKNOWN),
        },
        "reason_codes": dict(REASON_TEXT),
        "claim_boundary": SENSITIVE_WORK_CHAIN_CLAIM_BOUNDARY,
    }


def _contract_for(
    model_id: str,
    contracts: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Resolve through the shared projection, or read a caller's own table.

    The default path is `model_contract`, so alias, declared-inheritance, and
    dated-snapshot resolution behave here exactly as they do everywhere else.
    A caller supplying its own table -- a test pinning the permitting case, an
    audit over a candidate contract set -- gets an exact-id lookup against it,
    which is enough to grade a table without teaching this module a second
    resolver.
    """
    if contracts is MODEL_CONTRACTS:
        return model_contract(model_id)
    normalized = str(model_id or "").strip().casefold()
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[1]
    return contracts.get(normalized)


def _verdict(
    model_id: str,
    verdict: str,
    reason: str,
    *,
    training_use: str = "",
    retention: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": DATA_HANDLING_VERDICT_SCHEMA_VERSION,
        "model": model_id,
        "verdict": verdict,
        "reason": reason,
        "reason_text": REASON_TEXT[reason],
        "training_use": training_use,
        "retention": retention,
    }


__all__ = [
    "DATA_HANDLING_VERDICTS",
    "DATA_HANDLING_VERDICT_SCHEMA_VERSION",
    "REASON_AXIS_NOT_DECLARED",
    "REASON_DOCUMENTED_DEFAULT_PERMITS",
    "REASON_NO_DOCUMENTED_CONTRACT",
    "REASON_POLICY_NOT_RECORDED",
    "REASON_RETENTION_INDEFINITE",
    "REASON_TEXT",
    "REASON_TRAINING_USE_INCLUDES",
    "SENSITIVE_WORK_CHAIN_CLAIM_BOUNDARY",
    "SENSITIVE_WORK_CHAIN_SCHEMA_VERSION",
    "VERDICT_CONFLICTING",
    "VERDICT_PERMITTING",
    "VERDICT_UNKNOWN",
    "data_handling_filtered_chain",
    "model_data_handling_verdict",
]
