"""Compatibility corpus for the retired ULW engines' legacy cue vocabulary.

Plan §9.6 (#954 stage 5, window=0): the corpus is the set union of, for each
of the four retired contracts, every trigger and alias, plus every historical
display label and the current display label -- derived from the catalog, not
transcribed, so a trigger added to a retired contract later is covered
automatically. Set semantics are part of the contract: the short aliases
(`ulr`/`ulg`/`ulp`) already appear inside their own trigger tuples and must
never be double-counted; the builder returns a set and the size assertion
compares against the catalog-derived expectation, so a silently shrunk (or
summed) corpus fails.

A cue is *unresolved* when its interaction yields neither a route (no
selected workflow dispatch) nor an owner-selection action (`choose_executor`
absent from the action tuple). Both outcomes count as resolved -- that is
what makes the Q9 decision testable: the Codex-named cues resolve through
owner selection, not through `ultrawork`, and the corpus proves they still
land somewhere.

`semantic_change_count` diffs against the pinned baseline fixture
`tests/fixtures/ulw_alias_baseline.json`, captured in the retirement PR
itself: after the window=0 decision the alias routing is the permanent
contract, so the baseline pins the post-retirement semantics and any later
drift is a regression.
"""

from __future__ import annotations

import hashlib
from typing import Mapping

from ..skills.catalog import (
    retired_ulw_engine_definitions,
    retired_ulw_engine_names,
)
from ..skills.catalog_types import historical_skill_display_names, omh_skill_display_name
from ..wrapper.contract import build_chat_interaction_payload

ULW_ALIAS_CORPUS_SCHEMA_VERSION = "omh_ulw_alias_corpus/v1"


def ulw_alias_corpus() -> tuple[str, ...]:
    """Every legacy cue of the four retired engines, set semantics, sorted."""
    cues: set[str] = set()
    for definition in retired_ulw_engine_definitions():
        cues.update(definition.triggers)
        cues.update(definition.aliases)
        cues.add(omh_skill_display_name(definition.name))
        cues.update(historical_skill_display_names(definition.name))
    return tuple(sorted(cues))


def expected_ulw_alias_corpus_size() -> int:
    """Catalog-derived expectation the corpus size is asserted against."""
    return len(ulw_alias_corpus())


def _observe_cue(cue: str, *, source: str) -> dict[str, object]:
    interaction = build_chat_interaction_payload(cue, source=source)
    route = interaction.get("route")
    route = route if isinstance(route, dict) else {}
    response = interaction.get("chat_response")
    response = response if isinstance(response, dict) else {}
    actions = response.get("actions")
    action_ids = [
        str(action.get("id", ""))
        for action in (actions if isinstance(actions, list) else [])
        if isinstance(action, dict)
    ]
    alias_resolution = route.get("alias_resolution")
    return {
        "cue": cue,
        "selected_workflow": str(route.get("selected_skill", "") or ""),
        "route_action": str(route.get("action", "") or ""),
        "action_ids": action_ids,
        "claim_boundary": str(response.get("claim_boundary", "") or ""),
        "alias_resolution": dict(alias_resolution) if isinstance(alias_resolution, dict) else None,
    }


def _is_resolved(observation: Mapping[str, object]) -> bool:
    if observation["route_action"] == "dispatch" and observation["selected_workflow"]:
        return True
    return "choose_executor" in tuple(observation["action_ids"])


def ulw_alias_corpus_report(
    *,
    source: str = "generic",
    baseline: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Resolve every corpus cue and diff it against the pinned baseline.

    ``baseline`` maps cue -> {selected_workflow, action_ids, claim_boundary};
    pass the parsed fixture. Without a baseline, ``semantic_change_count`` is
    reported as 0 and ``baseline_checked`` is False.
    """
    cues = ulw_alias_corpus()
    observations = [_observe_cue(cue, source=source) for cue in cues]
    unresolved = [obs["cue"] for obs in observations if not _is_resolved(obs)]
    semantic_changes: list[dict[str, object]] = []
    if baseline is not None:
        for obs in observations:
            pinned = baseline.get(str(obs["cue"]))
            if pinned is None:
                semantic_changes.append({"cue": obs["cue"], "issue": "missing_from_baseline"})
                continue
            if str(pinned.get("selected_workflow", "")) != obs["selected_workflow"]:
                semantic_changes.append(
                    {
                        "cue": obs["cue"],
                        "issue": "selected_workflow_changed",
                        "pinned": pinned.get("selected_workflow"),
                        "observed": obs["selected_workflow"],
                    }
                )
                continue
            if bool(str(pinned.get("claim_boundary", ""))) and not obs["claim_boundary"]:
                semantic_changes.append({"cue": obs["cue"], "issue": "claim_boundary_weakened"})
    korean_count = sum(
        1
        for cue in cues
        if any("가" <= character <= "힣" for character in cue)
    )
    digest = hashlib.sha256("\n".join(cues).encode("utf-8")).hexdigest()
    return {
        "schema_version": ULW_ALIAS_CORPUS_SCHEMA_VERSION,
        "source": source,
        "corpus_size": len(cues),
        "expected_corpus_size": expected_ulw_alias_corpus_size(),
        "korean_cue_count": korean_count,
        "retired_engines": list(retired_ulw_engine_names()),
        "corpus_digest": digest,
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
        "semantic_change_count": len(semantic_changes),
        "semantic_changes": semantic_changes,
        "baseline_checked": baseline is not None,
        "observations": observations,
        "claim_boundary": (
            "Corpus resolution is a routing preflight over legacy cues; it is not execution, "
            "review, CI, or merge evidence."
        ),
    }
