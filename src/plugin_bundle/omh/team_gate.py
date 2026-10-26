"""Ask the person before a team's check commands are frozen.

`omh_team` `team_start` is the moment a list of commands becomes something
OMH will later run on its own, once per returning helper. The plan record
saying "accepted" is written by the model (`todo_store`), so it cannot be the
whole approval. This escalates `team_start` to the host's human-approval gate
-- the `approve` directive `plan_stage_gate` already uses, and the only
`pre_tool_call` answer a model cannot decline -- with the exact command list
in the prompt.

Where nobody can answer (a delegated child, a programmatic platform, a
single-query process), the call is BLOCKED rather than let through. That is
the opposite fallback from `plan_stage_gate`, deliberately: an unanswered
edit prompt costs a person their automation, but an unanswered command
approval would be a new command-execution surface opened with no person
behind it.

The `[a]lways` grain is the exact command list, not the tool name: a person
who allows one team's commands forever has not allowed the next team's. The
key's digest must equal `omh.workflows.team.approval_rule_key`; this module
cannot import the engine (Hermes execs this directory without `omh` on the
path), so a test pins the two equal.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Final

TEAM_TOOL_NAME: Final = "omh_team"
TEAM_START_RULE_PREFIX: Final = "omh_team_start:"
_MAX_SHOWN_COMMANDS: Final = 8
_MAX_SHOWN_COMMAND_CHARS: Final = 200

TEAM_START_UNATTENDED_MESSAGE: Final = (
    "Starting a checked team needs a person to approve its check commands, and nobody can answer here."
)


def team_start_commands(tool_input: object) -> list[str]:
    """The command list a `team_start` call would freeze, in unit order."""
    if not isinstance(tool_input, Mapping) or tool_input.get("action") != "team_start":
        return []
    units = tool_input.get("units")
    if not isinstance(units, list):
        return []
    commands = []
    for unit in units[:_MAX_SHOWN_COMMANDS]:
        if isinstance(unit, Mapping):
            commands.append(str(unit.get("verification_command", "") or "").strip()[:_MAX_SHOWN_COMMAND_CHARS])
    return commands


def team_start_rule_key(commands: list[str]) -> str:
    serialized = json.dumps(commands, sort_keys=True, separators=(",", ":"), default=str)
    return TEAM_START_RULE_PREFIX + hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def team_start_directive(
    *, tool_name: object, tool_input: object, escalation_allowed: bool
) -> dict[str, str] | None:
    """`approve` with the frozen commands, `block` when nobody can answer, else None."""
    if str(tool_name or "") != TEAM_TOOL_NAME:
        return None
    commands = team_start_commands(tool_input)
    if not commands:
        return None
    plan_ref = tool_input.get("plan_ref") if isinstance(tool_input, Mapping) else None
    if not isinstance(plan_ref, str) or not plan_ref.strip():
        # A start without a plan reference cannot succeed -- the engine
        # refuses it before anything is frozen and hands back the accepted
        # plan's reference -- so asking the person here would only make them
        # answer the same list twice.
        return None
    if not escalation_allowed:
        return {"action": "block", "message": TEAM_START_UNATTENDED_MESSAGE}
    listed = "\n".join(f"{index}. {command}" for index, command in enumerate(commands, start=1))
    message = (
        f"Start a team of {len(commands)} helper part(s)? When each part comes back, OMH runs its check "
        "below in this workspace, as your user account with your real home folder, so it runs code the "
        f"helpers wrote:\n{listed}\n"
        "Each command must be written in the accepted plan as check: `<command>`, exactly. "
        "Approving allows exactly these commands for this team; denying starts nothing."
    )
    return {"action": "approve", "message": message, "rule_key": team_start_rule_key(commands)}
