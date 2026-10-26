from __future__ import annotations


from ..executors import (
    CODING_EXECUTOR_HANDOFF_TARGETS,
    CODING_RUNTIME_HANDOFF_TARGETS,
    CODING_EXECUTOR_TARGETS,
    EXECUTOR_HANDOFF_SCHEMA_VERSION,
    PROMPT_HANDOFF_SCHEMA_VERSION,
    RUNTIME_HANDOFF_SCHEMA_VERSION,
)

TASK_PROMPT_CONTRACT_SCHEMA_VERSION = "executor_task_prompt_contract/v1"
TASK_PROMPT_REQUIRED_SECTIONS = ("Goal", "Do", "Don't", "Expected result", "Test")
EXECUTOR_PROMPTING_CONTRACT_SCHEMA_VERSION = "executor_prompting_contract/v1"
EXECUTOR_STEERING_DELTA_CONTRACT_SCHEMA_VERSION = "executor_steering_delta/v1"
EXECUTOR_PROMPTING_REQUIRED_SECTIONS = (
    "Goal",
    "Do",
    "Don't",
    "Known context",
    "Unknowns and decision rule",
    "Expected result",
    "Test",
    "Progress and blockers",
    "Evidence boundary",
    "Task",
)
EXECUTOR_PROMPTING_STRATEGIES = (
    "direct_change",
    "plan_backed_change",
    "risk_aware_change",
    "review_or_repair",
)
CODEX_SESSION_OBSERVATION_CONTRACT_SCHEMA_VERSION = "codex_session_observation_contract/v1"
CLAUDE_CODE_SESSION_OBSERVATION_CONTRACT_SCHEMA_VERSION = "claude_code_session_observation_contract/v1"
LOCAL_CAPABILITY_REPORT_CONTRACT_SCHEMA_VERSION = "executor_local_capability_report_contract/v1"
LOCAL_CAPABILITY_REPORT_REQUIRED_FIELDS = (
    "local_capabilities_used",
    "local_capability_evidence_refs",
    "local_capability_fallback_reason",
)
LOCAL_CAPABILITY_REPORT_CAPABILITY_FIELDS = ("name", "kind", "source", "purpose", "evidence_ref")
LOCAL_CAPABILITY_REPORT_ALLOWED_KINDS = (
    "skill",
    "workflow",
    "slash_command",
    "subagent",
    "agent",
    "mcp_tool",
    "repo_script",
    "test_harness",
    "ci_metadata",
    "runtime_template",
    "worker_lane",
    "worktree",
)
# Executor-neutral structural-search guidance carried by the coding_delegation
# capability blocks. Fanout unit prompts do not repeat it: their shared head
# carries STRUCTURAL_SEARCH_DISCIPLINE_GUIDANCE, which already says "structural
# search when available, grep otherwise". The final clause is required: no
# handoff field carries a detection result, so the executor must not infer OMH
# already checked PATH.
STRUCTURAL_SEARCH_GUIDANCE = (
    "Structural code search: if a structural search tool such as ast-grep is on PATH, prefer a "
    "structural query over line-based grep when the target is a syntactic shape rather than a "
    "string; fall back to grep when it is absent. OMH did not check - verify PATH yourself."
)
# Capped-search budget: STRUCTURAL_SEARCH_GUIDANCE says WHICH tool to prefer;
# this says HOW MUCH to search before escalating or stopping, so an executor's
# own exploration spend is bounded the same way every other handoff discipline
# is bounded. Deliberately tool- and flag-neutral (no ast-grep option, count,
# or byte cap is named) — OMH prescribes search discipline, it does not invent
# or configure tooling. Shared verbatim everywhere STRUCTURAL_SEARCH_GUIDANCE
# is shared, plus the `executor_prompting_contract/v1` payload
# (`structural_search_discipline`, below) so the prose and the versioned
# payload field can never drift apart.
STRUCTURAL_SEARCH_DISCIPLINE_CONTRACT_SCHEMA_VERSION = "structural_search_discipline/v1"
STRUCTURAL_SEARCH_DISCIPLINE_GUIDANCE = (
    "Capped structural search: bound code exploration to a few targeted queries — structural search "
    "when available, grep otherwise — before reading a full file; start with the narrowest query that "
    "names the shape or symbol you need. Escalate to a wider query or a full-file read only when a "
    "bounded pass returns nothing or stays genuinely ambiguous after a second try. Stop searching the "
    "moment the target is found; a further query for something already located spends tokens without "
    "adding information."
)
PROJECT_GOVERNANCE_PROFILE_SCHEMA_VERSION = "project_governance_profile/v1"
PROJECT_GOVERNANCE_BLOCKED_SCHEMA_VERSION = "project_governance_blocked/v1"
PRODUCT_FAMILY_TEMPLATE_SCHEMA_VERSION = "product_family_template/v1"
PRODUCT_FAMILY_CHOICES = ("web", "mobile", "desktop", "api")


__all__ = [
    "CODING_EXECUTOR_HANDOFF_TARGETS",
    "CODING_RUNTIME_HANDOFF_TARGETS",
    "CODING_EXECUTOR_TARGETS",
    "EXECUTOR_HANDOFF_SCHEMA_VERSION",
    "PROMPT_HANDOFF_SCHEMA_VERSION",
    "RUNTIME_HANDOFF_SCHEMA_VERSION",
    "TASK_PROMPT_CONTRACT_SCHEMA_VERSION",
    "TASK_PROMPT_REQUIRED_SECTIONS",
    "EXECUTOR_PROMPTING_CONTRACT_SCHEMA_VERSION",
    "EXECUTOR_STEERING_DELTA_CONTRACT_SCHEMA_VERSION",
    "EXECUTOR_PROMPTING_REQUIRED_SECTIONS",
    "EXECUTOR_PROMPTING_STRATEGIES",
    "CODEX_SESSION_OBSERVATION_CONTRACT_SCHEMA_VERSION",
    "CLAUDE_CODE_SESSION_OBSERVATION_CONTRACT_SCHEMA_VERSION",
    "LOCAL_CAPABILITY_REPORT_CONTRACT_SCHEMA_VERSION",
    "LOCAL_CAPABILITY_REPORT_REQUIRED_FIELDS",
    "LOCAL_CAPABILITY_REPORT_CAPABILITY_FIELDS",
    "LOCAL_CAPABILITY_REPORT_ALLOWED_KINDS",
    "STRUCTURAL_SEARCH_GUIDANCE",
    "STRUCTURAL_SEARCH_DISCIPLINE_CONTRACT_SCHEMA_VERSION",
    "STRUCTURAL_SEARCH_DISCIPLINE_GUIDANCE",
    "PROJECT_GOVERNANCE_PROFILE_SCHEMA_VERSION",
    "PROJECT_GOVERNANCE_BLOCKED_SCHEMA_VERSION",
    "PRODUCT_FAMILY_TEMPLATE_SCHEMA_VERSION",
    "PRODUCT_FAMILY_CHOICES",
]
