"""Demand-loaded wrapper actions for lineage-bound direction iterations."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ..system.paths import OmhPaths


def execute_design_direction_iteration_action(
    paths: OmhPaths,
    action: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Apply one local action through the public iteration and memory APIs."""
    from ..workflows.design_direction_iterations import (
        build_design_direction_iteration,
        design_direction_iteration_machine_actions,
        revise_design_direction_iteration,
        select_design_direction_iteration,
        show_design_direction_iteration,
        update_design_direction_iteration,
        write_design_direction_iteration,
    )

    if action == "prepare_design_direction_iteration":
        iteration = write_design_direction_iteration(
            paths,
            build_design_direction_iteration(
                _object(payload, "direction_set"),
                source_revision_digest=_required(payload, "source_revision_digest"),
                criteria_revision=_required(payload, "criteria_revision"),
                criteria_dimensions=_strings(payload, "criteria_dimensions"),
                score_threshold=_number(payload, "score_threshold"),
            ),
        )
    else:
        iteration_id = _required(payload, "iteration_id")
        iteration = show_design_direction_iteration(paths, iteration_id)
        if action == "show_design_direction_iteration":
            return _result(iteration, design_direction_iteration_machine_actions)

        current = _current(iteration)
        revision_digest = _required(payload, "revision_digest")
        if action != "revise_design_direction_iteration" and revision_digest != current["revision_digest"]:
            raise ValueError("stale revision identity refused")

        if action == "revise_design_direction_iteration":
            revised = revise_design_direction_iteration(
                iteration,
                parent_revision_digest=revision_digest,
                feedback_reference=_required(payload, "feedback_reference"),
                feedback_delta=_strings(payload, "feedback_delta"),
                direction_set=_object(payload, "direction_set"),
                successors=_successors(payload),
                criteria_revision=_required(payload, "criteria_revision"),
                criteria_dimensions=_strings(payload, "criteria_dimensions"),
                scores=_scores(payload),
                model_attempts=_model_attempts(payload),
            )
            iteration = update_design_direction_iteration(paths, revised)
        elif action == "select_design_direction_option":
            option_ref = _required(payload, "option_ref")
            option_refs = current.get("option_refs")
            if not isinstance(option_refs, list) or option_ref not in option_refs:
                raise ValueError("stale option reference refused")
            selected = select_design_direction_iteration(
                iteration,
                option_ref=option_ref,
                remember_this=payload.get("remember_this") is True,
            )
            iteration = update_design_direction_iteration(paths, selected)
        elif action == "request_design_direction_memory_review":
            return _request_memory_review(paths, iteration, payload)
        else:
            raise ValueError("unsupported design direction iteration action")
    return _result(iteration, design_direction_iteration_machine_actions)


def _request_memory_review(
    paths: OmhPaths,
    iteration: dict[str, object],
    payload: Mapping[str, object],
) -> dict[str, object]:
    from ..system.local_store import file_lock
    from ..workflows.memory import (
        _read_project_memory_candidates,
        capture_project_memory_candidate,
    )

    current = _current(iteration)
    promotion = _object(iteration, "memory_promotion")
    persisted = promotion.get("request")
    request = payload.get("memory_promotion_request")
    if (
        not isinstance(persisted, dict)
        or not persisted
        or not isinstance(request, Mapping)
        or dict(request) != persisted
        or persisted.get("accepted_revision_digest") != current["revision_digest"]
        or _object(iteration, "terminal").get("accepted_revision_digest") != current["revision_digest"]
    ):
        raise ValueError("persisted accepted memory-promotion request is required")

    thread_key = _required(payload, "thread_key")
    digest = str(current["revision_digest"])
    summary = f"Accepted design direction revision {digest}."

    def previous_capture() -> dict[str, object] | None:
        for candidate in _read_project_memory_candidates(paths):
            if candidate.get("source") != "design_direction_iteration" or candidate.get("source_ref") != digest:
                continue
            if candidate.get("scope") != {"kind": "thread", "ref": thread_key}:
                raise ValueError("memory request is already bound to another thread")
            return _memory_capture_projection(
                candidate_id=str(candidate["candidate_id"]),
                thread_key=thread_key,
                revision_digest=digest,
            )
        return None

    capture = previous_capture()
    if capture is None:
        with file_lock(paths.memory_dir / ".design-direction-capture.lock", private=True):
            capture = previous_capture()
            if capture is None:
                captured = capture_project_memory_candidate(
                    paths, summary, record_type="decision", scope_kind="thread",
                    scope_ref=thread_key, source="design_direction_iteration", source_ref=digest,
                    tags=("design-direction", "accepted-revision"), force_review=True,
                )
                candidate = captured.get("candidate")
                if not captured.get("captured") or not isinstance(candidate, dict):
                    return {
                        "schema_version": "design_direction_iteration_memory_review/v1",
                        "iteration_id": iteration["iteration_id"],
                        "accepted_revision_digest": digest,
                        "memory_capture": captured,
                        "next_action": "prepare_memory_new",
                        "claim_boundary": str(captured.get("claim_boundary") or ""),
                    }
                capture = _memory_capture_projection(
                    candidate_id=str(candidate["candidate_id"]),
                    thread_key=thread_key,
                    revision_digest=digest,
                )
    return {
        "schema_version": "design_direction_iteration_memory_review/v1",
        "iteration_id": iteration["iteration_id"],
        "accepted_revision_digest": digest,
        "memory_capture": capture,
        "next_action": "prepare_memory_new",
        "claim_boundary": (
            "An explicit accepted revision entered the existing review-first memory candidate path; "
            "no automatic approval, global preference, or Hermes-native memory write occurred."
        ),
    }


def _memory_capture_projection(
    *,
    candidate_id: str,
    thread_key: str,
    revision_digest: str,
) -> dict[str, object]:
    return {
        "captured": True,
        "auto_approved": False,
        "candidate": {
            "candidate_id": candidate_id,
            "scope": {"kind": "thread", "ref": thread_key},
            "source_ref": revision_digest,
        },
    }


def _result(
    iteration: dict[str, object],
    action_builder: Callable[[dict[str, object]], list[dict[str, object]]],
) -> dict[str, object]:
    return {
        "schema_version": "wrapper_design_direction_iteration_action/v1",
        "iteration": iteration,
        "machine_actions": action_builder(iteration),
        "claim_boundary": str(iteration.get("claim_boundary") or ""),
    }


def _current(iteration: Mapping[str, object]) -> dict[str, object]:
    snapshots = iteration.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots or not isinstance(snapshots[-1], dict):
        raise ValueError("iteration has no current snapshot")
    return snapshots[-1]


def _required(payload: Mapping[str, object], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _object(payload: Mapping[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _strings(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a non-empty string list")
    return tuple(value)


def _number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _successors(
    payload: Mapping[str, object],
) -> tuple[tuple[str, tuple[str, ...], str], ...]:
    value = payload.get("successors")
    if not isinstance(value, list) or not value:
        raise ValueError("successors must be a non-empty list")
    rows: list[tuple[str, tuple[str, ...], str]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 3 or not isinstance(row[1], (list, tuple)):
            raise ValueError("successors are invalid")
        rows.append((str(row[0]), tuple(str(item) for item in row[1]), str(row[2])))
    return tuple(rows)


def _scores(
    payload: Mapping[str, object],
) -> tuple[tuple[str, float, str, str, str], ...]:
    value = payload.get("scores", [])
    if not isinstance(value, list):
        raise ValueError("scores are invalid")
    if any(not isinstance(row, (list, tuple)) or len(row) != 5 for row in value):
        raise ValueError("scores are invalid")
    return tuple(
        (str(row[0]), float(row[1]), str(row[2]), str(row[3]), str(row[4]))
        for row in value
    )


def _model_attempts(
    payload: Mapping[str, object],
) -> tuple[tuple[str, str, int | None, float | None, float | None, str | None], ...]:
    value = payload.get("model_attempts", [])
    if not isinstance(value, list):
        raise ValueError("model_attempts are invalid")
    if any(not isinstance(row, (list, tuple)) or len(row) != 6 for row in value):
        raise ValueError("model_attempts are invalid")
    return tuple(
        (str(row[0]), str(row[1]), row[2], row[3], row[4], row[5])
        for row in value
    )
