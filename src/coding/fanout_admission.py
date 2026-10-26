"""Adaptive submission-window state for the explicit fanout bridge."""

from __future__ import annotations

from typing import Any, Mapping

from ..system.metadata_safety import redact_metadata_text
from .fanout_contracts import REPRODUCTION_UNIT_KIND

FANOUT_ADMISSION_SCHEMA_VERSION = "fanout_admission/v1"
FANOUT_ADMISSION_ADJUSTMENT_LIMIT = 32
FANOUT_ADMISSION_CLAIM_BOUNDARY = (
    "Adaptive admission records only how observed local unit process results changed this dispatch's "
    "submission window. A provider-limit status class is not provider quota truth, and admission "
    "decisions are not verification, review, CI, merge-readiness, or merge evidence. Dry-run plans "
    "are not observed execution."
)

_PROVIDER_LIMIT_FAILURE_KIND = "limit_shaped"
_PROVIDER_LIMIT_RETRY_CLASS = "transient_provider_limit"


class AdaptiveFanoutAdmission:
    """Additive-increase, multiplicative-decrease admission state."""

    def __init__(self, *, ceiling: int, dry_run: bool = False) -> None:
        self.ceiling = max(1, int(ceiling))
        self.initial_window = min(2, self.ceiling)
        self.window = self.initial_window
        self.minimum_window = self.initial_window
        self.dry_run = bool(dry_run)
        self._observed_completion_count = 0
        self._observed_clean_completion_count = 0
        self._observed_provider_pressure_count = 0
        self._adjustment_count = 0
        self._adjustments: list[dict[str, Any]] = []

    def available_slots(self, inflight: int) -> int:
        return max(0, self.window - max(0, int(inflight)))

    def observe(self, unit_id: str, result: Mapping[str, Any]) -> None:
        """Apply one observed unit-process result to the admission window."""
        if self.dry_run or not _is_observed_process_completion(result):
            return
        self._observed_completion_count += 1
        before = self.window
        if _has_provider_limit_pressure(result):
            self._observed_provider_pressure_count += 1
            self.window = max(1, before // 2)
            status_class = "provider_limit_pressure"
            action = "halve" if self.window < before else "hold_minimum"
        elif _is_clean_completion(result):
            self._observed_clean_completion_count += 1
            self.window = min(self.ceiling, before + 1)
            status_class = "clean_completion"
            action = "increase" if self.window > before else "hold_ceiling"
        else:
            return
        self.minimum_window = min(self.minimum_window, self.window)
        self._adjustment_count += 1
        if len(self._adjustments) < FANOUT_ADMISSION_ADJUSTMENT_LIMIT:
            self._adjustments.append(
                {
                    "unit_id": redact_metadata_text(str(unit_id), limit=64),
                    "status_class": status_class,
                    "action": action,
                    "window_before": before,
                    "window_after": self.window,
                }
            )

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": FANOUT_ADMISSION_SCHEMA_VERSION,
            "mode": "adaptive",
            "requested": True,
            "initial_window": self.initial_window,
            "ceiling": self.ceiling,
            "final_window": self.window,
            "minimum_window": self.minimum_window,
            "observation_status": (
                "not_observed_dry_run"
                if self.dry_run
                else (
                    "observed_local_process_results"
                    if self._observed_completion_count
                    else "no_observed_unit_results"
                )
            ),
            "observed_completion_count": self._observed_completion_count,
            "observed_clean_completion_count": self._observed_clean_completion_count,
            "observed_provider_pressure_count": self._observed_provider_pressure_count,
            "adjustment_count": self._adjustment_count,
            "adjustments": list(self._adjustments),
            "adjustments_omitted": max(
                0, self._adjustment_count - len(self._adjustments)
            ),
            "claim_boundary": FANOUT_ADMISSION_CLAIM_BOUNDARY,
        }


def reproduction_hold_released(unit: Mapping[str, Any], result: Mapping[str, Any] | None) -> bool:
    """Whether a finished dependency lets the units behind it be admitted.

    The reproduction hold: a unit that depends on a reproduction unit is
    admitted only once the dispatcher itself observed that unit's declared
    reproduction command exit nonzero. The verdict is read from the receipt's
    fields, never from output text, and nothing short of `failure_reproduced`
    observed by the dispatcher releases it -- a dry-run plan, a command that
    never ran, and a command that exited 0 all keep the hold closed.

    Every unit that is not a reproduction unit releases nothing here, so a
    split that declares no reproduction admits exactly as it did before.
    """
    if unit.get("kind") != REPRODUCTION_UNIT_KIND:
        return True
    receipt = result.get("reproduction") if isinstance(result, Mapping) else None
    return (
        isinstance(receipt, Mapping)
        and receipt.get("status") == "failure_reproduced"
        and receipt.get("observed_by") == "dispatcher"
    )


def _is_observed_process_completion(result: Mapping[str, Any]) -> bool:
    exit_code = result.get("exit_code")
    return isinstance(exit_code, int) and not isinstance(exit_code, bool)


def _is_clean_completion(result: Mapping[str, Any]) -> bool:
    return (
        result.get("status") == "completed"
        and result.get("process_succeeded") is True
        and result.get("exit_code") == 0
    )


def _has_provider_limit_pressure(result: Mapping[str, Any]) -> bool:
    if result.get("failure_kind") == _PROVIDER_LIMIT_FAILURE_KIND:
        return True
    retry = result.get("retry")
    if not isinstance(retry, Mapping):
        return False
    decisions = retry.get("decisions")
    if not isinstance(decisions, (list, tuple)):
        return False
    return any(
        isinstance(decision, Mapping)
        and decision.get("failure_class") == _PROVIDER_LIMIT_RETRY_CLASS
        for decision in decisions
    )
