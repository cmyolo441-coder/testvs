"""The delegation protocol a delegated agent follows, owned once by the bundle.

Three rules: echo the goal back before using tools, run one bounded
verification pass, and cap re-review at two rounds. The control plane
(`omh.coding.unit_prompt_protocol`) re-exports them from here, so fanout unit
prompts and `omh coding composition-guide` read these same objects. The plugin
reads them directly, to put them in the system prompt of every Hermes
`delegate_task` child (#2174). One definition, no copy. Stdlib only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

# In two parts so the child section can make the criteria conditional
# without a second copy of the rule.
_GOAL_ECHO_RESTATE: Final[str] = "Before using tools, restate the overall goal, your deliverable, and the numbered criteria"
_GOAL_ECHO_CONFLICT: Final[str] = "If they conflict with your brief, stop and report the conflict instead of guessing."

GOAL_ECHO_PROTOCOL: Final[str] = f"{_GOAL_ECHO_RESTATE}. {_GOAL_ECHO_CONFLICT}"

VERIFICATION_STOP_PROTOCOL: Final[str] = (
    "After finishing work, run one full verification pass; verification is never skipped. "
    "A failed check blocks only a stated criterion; report other observations. Once criteria pass, "
    "STOP verification, including after unrelated edits. After two failed fix-and-verify cycles, "
    "commit what passes and report the failed criterion and output."
)

REVIEW_ROLE_PROTOCOL: Final[str] = (
    "Review discipline: a finding blocks only when it violates a stated success criterion of the reviewed "
    "work; list every other finding as non-blocking. Cap re-review at two rounds — after that, report the "
    "remaining criterion-cited blockers rather than starting another round."
)

# A system prompt section, not a `pre_tool_call` `modify` of `goal` and
# `tasks[].goal`. Hermes renders every plugin section into each agent's
# system prompt, a `delegate_task` child's included
# (`agent/system_prompt.py`, `build_system_prompt_parts`), so the section
# reaches every child of a batch and a nested child. A `modify` would also
# rewrite the goal that the parent's display, the `subagent_start` hook and
# the child's first user turn all show.
#
# It also reaches children no OMH-guided parent composed, such as the
# reviewer that `agent/review_engine.py` spawns for Hermes' own `/review`.
# So only the goal echo-back applies to every child, and it asks for the
# numbered criteria only when the goal has them; the shared
# `GOAL_ECHO_PROTOCOL` stays unconditional, because OMH's own unit prompts
# always carry criteria. The verification rule and the review rule each
# carry the condition they were written for (OMH's own unit prompts give the
# review rule to review units only). A child whose task matches neither,
# like a `/review` reviewer that has no stated success criteria, is not
# bound by them. "Other state" covers what a task changes without editing a
# file: a deploy, a migration, remote or config state.
DELEGATION_PROTOCOL_SECTION_ID: Final[str] = "omh.delegation-protocol"

DELEGATION_PROTOCOL_SECTION: Final[str] = "\n".join(
    (
        "OMH delegation protocol for this delegated task:",
        f"- {_GOAL_ECHO_RESTATE}, if the goal has them. {_GOAL_ECHO_CONFLICT}",
        f"- When your task changes files or other state: {VERIFICATION_STOP_PROTOCOL}",
        f"- When you review work against stated success criteria: {REVIEW_ROLE_PROTOCOL}",
    )
)

# Hermes builds a `delegate_task` child with this platform
# (`tools/delegate_tool.py`, `_build_child_agent`). Nothing else in Hermes
# sets it at d923a5eb48 (v0.22.2) or 8b9ac5773e. `omh doctor` reads the
# installed Hermes for it (`plugin_delegation_protocol`), because a rename
# would leave every child without the section and nothing would say so.
CHILD_PLATFORM: Final[str] = "subagent"


def delegation_protocol_section(session_info: object) -> str:
    """The protocol for a `delegate_task` child's system prompt, and "" for any other session.

    Hermes hands each section the agent's own platform in `session_info`
    (`_plugin_session_info`). It skips an empty section without charging the
    budget that all plugins' sections share, so the parent's prompt does not
    change.
    """
    platform = session_info.get("platform") if isinstance(session_info, Mapping) else None
    return DELEGATION_PROTOCOL_SECTION if platform == CHILD_PLATFORM else ""
