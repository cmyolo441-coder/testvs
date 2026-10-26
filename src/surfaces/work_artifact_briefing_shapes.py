"""Shape-source adapters for persisted ``coding_briefing/v1`` artifacts."""

from __future__ import annotations

from collections.abc import Mapping


def briefing_shape_source(
    artifact_id: str,
    session_id: str,
    briefing: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Return only structure explicitly recorded for one briefing artifact."""
    if artifact_id == "acceptance_and_verification":
        return _acceptance_source(artifact_id, session_id, briefing)
    if artifact_id == "status_brief":
        return _status_source(artifact_id, session_id, briefing)
    if artifact_id == "evidence_gaps":
        return _gaps_source(artifact_id, session_id, briefing)
    if artifact_id == "next_action":
        return _next_action_source(artifact_id, session_id, briefing)
    if artifact_id == "issue_pr_followup":
        return _followup_source(artifact_id, session_id, briefing)
    return None


def _acceptance_source(
    artifact_id: str,
    session_id: str,
    briefing: Mapping[str, object],
) -> Mapping[str, object] | None:
    summary = _mapping(briefing.get("work_summary"))
    acceptance = _strings(summary.get("acceptance_criteria"))
    verification = _strings(summary.get("verification_expected"))
    if not acceptance and not verification:
        return None
    prefix = _prefix(session_id, artifact_id)
    owner = str(summary.get("executor", ""))
    nodes: list[dict[str, object]] = []
    if acceptance:
        nodes.append(
            _node(
                "acceptance",
                f"Acceptance criteria ({len(acceptance)})",
                f"{prefix}.work_summary.acceptance_criteria",
                owner=owner,
            )
        )
    if verification:
        nodes.append(
            _node(
                "verification",
                f"Verification expected ({len(verification)})",
                f"{prefix}.work_summary.verification_expected",
                owner=owner,
            )
        )
    edges = (
        [
            _edge(
                "acceptance",
                "verification",
                "checked by",
                f"{prefix}.work_summary.verification_expected",
            )
        ]
        if acceptance and verification
        else []
    )
    return _source(
        artifact_id,
        session_id,
        briefing,
        nodes,
        edges=edges,
        bullets=[*acceptance, *verification],
    )


def _status_source(
    artifact_id: str,
    session_id: str,
    briefing: Mapping[str, object],
) -> Mapping[str, object] | None:
    progress = _mappings(briefing.get("progress"))
    if not progress:
        return None
    prefix = _prefix(session_id, artifact_id)
    owner = str(
        _mapping(briefing.get("current_state")).get(
            "selected_executor_profile", ""
        )
    )
    nodes = [
        _node(
            f"step-{index + 1}",
            str(step.get("label") or step.get("id") or f"Step {index + 1}"),
            f"{prefix}.progress[{index}]",
            state=str(step.get("state", "")),
            owner=owner,
        )
        for index, step in enumerate(progress)
    ]
    edges = [
        _edge(
            f"step-{index}",
            f"step-{index + 1}",
            "then",
            f"{prefix}.progress[{index}]",
        )
        for index in range(1, len(nodes))
    ]
    return _source(artifact_id, session_id, briefing, nodes, edges=edges)


def _gaps_source(
    artifact_id: str,
    session_id: str,
    briefing: Mapping[str, object],
) -> Mapping[str, object] | None:
    gaps = _strings(briefing.get("pending_gaps"))
    if not gaps:
        return None
    prefix = _prefix(session_id, artifact_id)
    nodes = [
        _node(
            f"gap-{index + 1}",
            f"Evidence gap {index + 1}",
            f"{prefix}.pending_gaps[{index}]",
            state="missing_evidence",
            owner="omh",
        )
        for index in range(len(gaps))
    ]
    return _source(artifact_id, session_id, briefing, nodes, bullets=gaps)


def _next_action_source(
    artifact_id: str,
    session_id: str,
    briefing: Mapping[str, object],
) -> Mapping[str, object] | None:
    action = str(briefing.get("next_action", ""))
    if not action:
        return None
    current = _mapping(briefing.get("current_state"))
    prefix = _prefix(session_id, artifact_id)
    node = _node(
        "next-action",
        "Recorded next action",
        f"{prefix}.next_action",
        state=str(current.get("lifecycle_status") or "prepared_not_observed"),
        owner=str(current.get("selected_executor_profile", "")),
    )
    return _source(artifact_id, session_id, briefing, [node])


def _followup_source(
    artifact_id: str,
    session_id: str,
    briefing: Mapping[str, object],
) -> Mapping[str, object] | None:
    run_id = str(briefing.get("run_id", ""))
    if not run_id:
        return None
    prefix = _prefix(session_id, artifact_id)
    owner = str(
        _mapping(briefing.get("current_state")).get(
            "selected_executor_profile", ""
        )
    )
    nodes = [
        _node("run", "Recorded run", f"{prefix}.run_id", owner=owner),
        _node("session", "Wrapper session", f"{prefix}.session_id", owner=owner),
        _node("next", "Recorded next action", f"{prefix}.next_action", owner=owner),
    ]
    edges = [
        _edge("run", "session", "belongs to", f"{prefix}.session_id"),
        _edge("session", "next", "continues with", f"{prefix}.next_action"),
    ]
    return _source(artifact_id, session_id, briefing, nodes, edges=edges)


def _source(
    artifact_id: str,
    session_id: str,
    briefing: Mapping[str, object],
    nodes: list[dict[str, object]],
    *,
    edges: list[dict[str, object]] | None = None,
    bullets: list[str] | None = None,
) -> dict[str, object]:
    return {
        "source_artifact_id": _prefix(session_id, artifact_id),
        "source_schema": str(briefing.get("schema_version", "")),
        "evidence_state": "prepared_not_observed",
        "nodes": nodes,
        "edges": edges or [],
        "bullets": bullets or [],
    }


def _node(
    node_id: str,
    label: str,
    source_ref: str,
    *,
    state: str = "prepared_not_observed",
    owner: str = "",
) -> dict[str, object]:
    return {
        "node_id": node_id,
        "label": label,
        "source_refs": [source_ref],
        "state": state,
        "owner": owner,
        "change": "unknown",
    }


def _edge(
    source_id: str,
    target_id: str,
    label: str,
    source_ref: str,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "target_id": target_id,
        "label": label,
        "source_refs": [source_ref],
    }


def _prefix(session_id: str, artifact_id: str) -> str:
    return f"{session_id}#{artifact_id}"


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]
