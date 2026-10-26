"""Permanent alias routing for the four retired ULW engines (#954 stage 5).

`team`, `ultraprocess`, `ralph`, and `ultragoal` retired from the routable
surface with a window=0 maintainer decision: their natural-language cues route
permanently to `ultrawork`'s matching internal capability instead of passing
through an alias or warning release stage. The original invocation is
preserved in diagnostics (`alias_resolution.original_invocation`) with a
one-line `capability_reason` that states the Hermes-harness default in the
same words the orchestration vocabulary uses (#953).

The Codex-named cue cluster from the old `ultraprocess` trigger table is
deliberately NOT an engine alias (plan Q9): naming a coding CLI is an
owner-choice signal, not an engine-selection signal, and carrying it onto
`ultrawork` would encode Codex as the implicit default engine. Those cues
resolve through the owner-selection path instead; the in-message-naming
provenance writer (#953, PR A) records the naming as an explicit choice.

Copy here is informational migration copy, never a deprecation warning: the
intent now runs as `ulw-work`'s capability, permanently.
"""

from __future__ import annotations

from functools import lru_cache

from ..coding.orchestration_vocabulary import HERMES_HARNESS_DEFAULT_WORDING
from ..plugin_bundle.omh.name_report_cues import about_the_skill
from ..skills.catalog import (
    ULW_RETIRED_CAPABILITIES,
    retired_ulw_engine_definitions,
)
from ..skills.catalog_types import historical_skill_display_names, omh_skill_display_name
from ..plugin_bundle.omh.boundary_phrase import contains_boundary_phrase
from .visual_qa_cues import contains_cue_phrase
from .localization import normalized_phrase
from .reference_regions import executable_routing_text

ULW_ALIAS_TARGET_WORKFLOW = "ultrawork"

# The Codex-named cues from the old `ultraprocess` trigger cluster (plan Q9).
# Every entry names a coding CLI (or the coding agent as such) in the user's
# own words; per §5.3.1 naming is choosing, so these are owner-choice signals
# routed to the owner-selection surface, never to `ultrawork`.
CODEX_OWNER_CHOICE_CUES = (
    "delegate to codex",
    "send to codex",
    "codex implement",
    "codex progress tracking",
    "codex session tracking",
    "codex로 구현",
    "코덱스로 구현",
    "codex에게 맡기",
    "codex로 맡기",
    "코덱스에게 맡기",
    "코딩 에이전트에게 맡기",
)

# One informational selection line per capability, executor-neutral by
# construction; `tests/test_ulw_retirement.py` pins this table against the
# capability ids in `ULW_RETIRED_CAPABILITIES`.
_CAPABILITY_INTENT_LINES = {
    "coordinated_scope": "coordinated worker lanes with explicit ownership",
    "delivery_boundary": "one bounded plan-to-PR delivery cycle",
    "single_owner_persistence": "one owner finishing a concrete task through verification",
    "durable_checkpoint": "a durable goal ledger with checkpoints and a final gate",
}


def ulw_alias_capability_reason(retired_name: str) -> str:
    """The one-line reason shown on the status card for an alias route."""
    capability = ULW_RETIRED_CAPABILITIES[retired_name]
    intent = _CAPABILITY_INTENT_LINES[capability]
    return (
        f"This intent now runs as `ulw-work` capability `{capability}` ({intent}); "
        f"{HERMES_HARNESS_DEFAULT_WORDING}."
    )


def _invocation_stripped(text: str) -> str:
    stripped = text.strip()
    for prefix in ("$", "./", "/"):
        if stripped.startswith(prefix):
            return stripped[len(prefix):]
    return stripped


@lru_cache(maxsize=1)
def _alias_cue_index() -> tuple[tuple[str, str, str], ...]:
    """(normalized cue, raw cue, retired canonical) for every alias cue.

    Derived from the retired definitions' trigger tables plus current and
    historical display labels, minus the Codex-named owner-choice cluster --
    a trigger added to a retired contract later is covered automatically.
    """
    excluded = {normalized_phrase(cue) for cue in CODEX_OWNER_CHOICE_CUES}
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for definition in retired_ulw_engine_definitions():
        cues = [
            *definition.triggers,
            *definition.aliases,
            definition.name,
            omh_skill_display_name(definition.name),
            *historical_skill_display_names(definition.name),
        ]
        for cue in cues:
            normalized = normalized_phrase(cue)
            if not normalized or normalized in excluded or normalized in seen:
                continue
            seen.add(normalized)
            rows.append((normalized, cue, definition.name))
    # Longest cue first so containment matching prefers the most specific cue.
    rows.sort(key=lambda row: len(row[0]), reverse=True)
    return tuple(rows)


@lru_cache(maxsize=1)
def _codex_owner_choice_index() -> tuple[str, ...]:
    return tuple(normalized_phrase(cue) for cue in CODEX_OWNER_CHOICE_CUES)


@lru_cache(maxsize=1)
def _retired_name_and_alias_index() -> dict[str, str]:
    """Normalized canonical names and short aliases of the retired engines."""
    index: dict[str, str] = {}
    for definition in retired_ulw_engine_definitions():
        index[normalized_phrase(definition.name)] = definition.name
        for alias in definition.aliases:
            index[normalized_phrase(alias)] = definition.name
    return index


# The legacy `ultraprocess` session flows that resolved Codex from the message
# alone: issue-to-PR starts and session status/liveness questions. With the
# engine retired, a Codex-named sentence in one of these shapes diverts to the
# owner-selection surface instead of an engine route (plan Q9); the
# in-message-naming writer still records the naming as explicit-choice
# provenance.
_CODEX_NAME_MARKERS = ("codex", "코덱스")
_CODEX_SESSION_FLOW_CUES = (
    "세션이 살아있는지",
    "session alive",
    "세션 상태",
    "session status",
    "지금 뭐하고",
    "지금 뭐 하고",
    "뭐하고있는지",
    "뭐 하고 있는지",
    "진행상황",
    "진행 상황",
    "작업 시작해",
    "이슈 pr",
    "issue pr",
)


def resolve_codex_owner_choice_cue(message: str) -> str | None:
    """The Codex-named owner-choice cue this message matched, or None.

    Two shapes divert: the exact legacy cue invocations (whole-message match,
    invocation prefixes aside), and full sentences that name Codex inside one
    of the legacy session flows (issue-to-PR start, session status). Longer
    Codex-named sentences outside those flows keep their ordinary
    owner-resolution paths, which already record in-message naming as
    explicit-choice provenance.
    """
    message = executable_routing_text(message)
    normalized = normalized_phrase(_invocation_stripped(message))
    for index, cue in enumerate(_codex_owner_choice_index()):
        if normalized == cue:
            return CODEX_OWNER_CHOICE_CUES[index]
    if contains_cue_phrase(message, _CODEX_NAME_MARKERS) and contains_cue_phrase(
        message, _CODEX_SESSION_FLOW_CUES
    ):
        return message.strip()
    return None


def resolve_ulw_alias(message: str, *, allow_containment: bool = False) -> dict[str, str] | None:
    """Resolve a retired-engine cue to its `ulw-work` capability alias.

    Exact whole-message matching (with `$`/`./`//` invocation prefixes
    stripped) resolves the legacy cue vocabulary itself. With
    `allow_containment=True`, multi-word cues embedded in a longer message
    also resolve -- the same reachability the retired trigger tables used to
    provide through catalog scoring. Single-token cues never match by
    containment, so ordinary sentences mentioning "team" are not hijacked.
    """
    message = executable_routing_text(message)
    stripped = _invocation_stripped(message)
    normalized = normalized_phrase(stripped)
    if not normalized:
        return None
    for cue_normalized, cue, retired_name in _alias_cue_index():
        if normalized == cue_normalized:
            return _alias_resolution(message.strip(), retired_name)
    # A sigil-led invocation with a remainder ("$team split this work") is
    # still an explicit legacy invocation of the retired engine, and so is a
    # leading bare engine name or alias ("team fix one typo and ...") -- the
    # same leading-position rule `explicit_skill_name` applied while the
    # engines were routable.
    raw = message.strip()
    head_token = raw.split(None, 1)[0] if raw else ""
    if head_token:
        head_normalized = normalized_phrase(_invocation_stripped(head_token))
        sigil_led = raw[:1] == "$" or raw[:2] == "./" or raw[:1] == "/"
        if sigil_led:
            for cue_normalized, cue, retired_name in _alias_cue_index():
                if head_normalized == cue_normalized:
                    return _alias_resolution(head_token, retired_name)
        elif head_normalized in _retired_name_and_alias_index() and not _head_reports_or_asks(raw):
            return _alias_resolution(
                head_token, _retired_name_and_alias_index()[head_normalized]
            )
    if not allow_containment:
        return None
    for cue_normalized, cue, retired_name in _alias_cue_index():
        if " " not in cue_normalized and len(cue_normalized) < 6:
            continue
        # Literal aliases are phrases, not prefixes of product/profile/project
        # or identifiers. Preserve the cue table's hyphen/space equivalence,
        # but not compact substring matching; the shared boundary reader also
        # rejects filename suffixes such as make-a-pr.md.
        if contains_boundary_phrase(normalized.replace("-", " "), (cue_normalized.replace("-", " "),)):
            return _alias_resolution(cue, retired_name)
    return None


def _head_reports_or_asks(raw: str) -> bool:
    """Whether the words after a bare leading engine name report on it or ask about it ("ralph broke my build",
    "team stop", "ultragoal done?"): about the engine, not a request to run it (`name_report_cues`)."""
    parts = raw.split(None, 1)
    if len(parts) < 2 or parts[0].endswith((":", ",")):
        return False
    return about_the_skill(parts[0], parts[1])


def _alias_resolution(original_invocation: str, retired_name: str) -> dict[str, str]:
    capability = ULW_RETIRED_CAPABILITIES[retired_name]
    return {
        "original_invocation": original_invocation,
        "retired_contract_id": retired_name,
        "retired_display_name": omh_skill_display_name(retired_name),
        "target_contract_id": ULW_ALIAS_TARGET_WORKFLOW,
        "selected_capability": capability,
        "capability_reason": ulw_alias_capability_reason(retired_name),
    }
