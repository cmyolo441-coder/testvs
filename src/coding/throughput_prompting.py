from __future__ import annotations

from .model_contracts import contract_model_id
from .model_routing import model_family


_ULW_THROUGHPUT_WORKFLOWS = frozenset({"ultrawork"})
_BASE_THROUGHPUT_RULES = (
    "Parallel independent tool calls, repository reads, searches, and reasoning lanes in one batch whenever their inputs do not depend on each other.",
    "Keep dependency-bound work sequential; collect the upstream result before starting work that consumes it.",
)
_ADVANCED_THROUGHPUT_RULES = (
    "While delegated lanes run, continue only non-overlapping work and merge their evidence after completion is observed.",
    "For delegated work, declare one goal, an observable stop condition, and required evidence; judge completion from that evidence.",
    "Stop as soon as decisive evidence satisfies the stated criteria; do not reopen settled work without contradictory output.",
    "A delegated lane returns a distilled report — outcome, evidence pointers, open items — sized like a short briefing, never its transcript.",
)
_EVAL_BATCHING_RULE = (
    "When an eval or code-execution surface can batch independent operations, use one eval cell and run them concurrently "
    "inside it instead of emitting separate eval calls."
)


def build_throughput_overlay(
    profile: str,
    *,
    main_agent_model: str,
    recommended_workflow: str,
) -> dict[str, object]:
    family = model_family(main_agent_model) or "unknown"
    mode = "parallel_handoff"
    rules: list[str] = list(_BASE_THROUGHPUT_RULES)
    eval_strategy = ""
    if profile == "codex" and _is_gpt_sol_model(main_agent_model):
        mode = "gpt_sol_codex_handoff"
        rules.extend(_ADVANCED_THROUGHPUT_RULES)
        rules.append(_EVAL_BATCHING_RULE)
        eval_strategy = "single_cell_internal_parallel"
    elif profile == "claude-code" and family == "claude":
        mode = "claude_code_handoff"
        rules.extend(_ADVANCED_THROUGHPUT_RULES)
    elif (
        profile == "hermes"
        and family == "gpt"
        and recommended_workflow in _ULW_THROUGHPUT_WORKFLOWS
    ):
        mode = "gpt_hermes_ulw"
        rules.extend(_ADVANCED_THROUGHPUT_RULES)
    overlay: dict[str, object] = {
        "schema_version": "executor_throughput_overlay/v1",
        "status": "enabled",
        "mode": mode,
        "model_family": family,
        "rules": rules,
        "claim_boundary": "This overlay is prepared execution guidance only; it is not proof of parallel work, dispatch, verification, or completion.",
    }
    if eval_strategy:
        overlay["eval_strategy"] = eval_strategy
    return overlay


def _is_gpt_sol_model(model_id: str) -> bool:
    """Whether the model is a GPT Sol, read through its contract model id.

    The contract projection is the identity the calibration lookup uses, so
    a projected id (`gpt-6.1-sol-pro` -> `gpt-6.1-sol`) gets the same answer
    here as there.
    """
    normalized = str(model_id or "").strip().casefold().rsplit("/", 1)[-1]
    if not normalized:
        return False
    normalized = contract_model_id(normalized.split(":", 1)[0].split(None, 1)[0])
    return model_family(normalized) == "gpt" and normalized.endswith("-sol")


__all__ = ["build_throughput_overlay"]
