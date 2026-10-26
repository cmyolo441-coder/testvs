"""Adapter between committed `critical_path_health/v1` projections and run health.

`run_health_summary/v2` widens the summary by exactly one optional section: a
`critical_path_health/v1` projection COMMITTED by the caller and embedded
as-is. This module is the single place that decides whether a dict is such a
projection and how its facts read in the run health text surface: validate,
never repair; no clock read, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from ..system.metadata_safety import require_opaque_metadata_ref
from .critical_path_health import CRITICAL_PATH_HEALTH_PRIVACY, CRITICAL_PATH_HEALTH_SCHEMA_VERSION

# A bounded surface an agent may poll: the committed section's lists are capped.
MAX_CRITICAL_PATH_SECTION_ITEMS: Final[int] = 100
# The text surface renders the stages that dominate the path, not every stage.
TOP_CRITICAL_STAGES: Final[int] = 3

_SECTION_KEYS: Final[frozenset[str]] = frozenset(
    "schema_version privacy executor model environment task_revisions metrics "
    "phase_attribution resource_attribution evidence_gaps".split()
)
_REVISION_KEYS: Final[frozenset[str]] = frozenset({"task_id", "revision"})
_METRICS_KEYS: Final[frozenset[str]] = frozenset(
    "wall_clock_ms active_ms queue_ms critical_path_ms peak_concurrency overlap_savings_ms "
    "repeated_cost_ms stale_count cleanup_tail_ms reused_task_count".split()
)
_ATTRIBUTION_KEYS: Final[frozenset[str]] = frozenset("name active_ms queue_ms repeated_cost_ms task_count".split())
_GAP_KEYS: Final[frozenset[str]] = frozenset({"task_id", "code"})


def committed_critical_path_health_errors(raw: object, label: str) -> list[str]:
    """Every reason `raw` is not a committed `critical_path_health/v1` projection."""
    if not isinstance(raw, Mapping):
        return [f"{label} must be a committed critical_path_health/v1 projection object"]
    errors = _key_set_errors(raw, _SECTION_KEYS, label)
    if errors:
        return errors
    for field, expected in (
        ("schema_version", CRITICAL_PATH_HEALTH_SCHEMA_VERSION),
        ("privacy", CRITICAL_PATH_HEALTH_PRIVACY),
    ):
        if raw.get(field) != expected:
            errors.append(f"{label}.{field} must be {expected}")
    for field in ("executor", "model", "environment"):
        errors.extend(_opaque_ref_errors(raw.get(field), f"{label}.{field}"))
    errors.extend(_revision_errors(raw.get("task_revisions"), label))
    metrics = raw.get("metrics")
    if isinstance(metrics, Mapping):
        errors.extend(_metrics_errors(metrics, f"{label}.metrics"))
    elif metrics is not None:
        errors.append(f"{label}.metrics must be an object or null")
    errors.extend(_attribution_errors(raw.get("phase_attribution"), f"{label}.phase_attribution"))
    errors.extend(_attribution_errors(raw.get("resource_attribution"), f"{label}.resource_attribution"))
    errors.extend(_gap_errors(raw.get("evidence_gaps"), label))
    gaps = raw.get("evidence_gaps")
    # The producer's invariant: metrics exist exactly when no evidence gap
    # does, so numbers alongside gaps are refused rather than rendered.
    if isinstance(gaps, list):
        if metrics is None and not gaps:
            errors.append(f"{label}.evidence_gaps must not be empty when metrics is null")
        if metrics is not None and gaps:
            errors.append(f"{label}.metrics must be null when evidence gaps are present")
        if metrics is None:
            errors += [
                f"{label}.{field} must be empty when metrics is null"
                for field in ("phase_attribution", "resource_attribution")
                if raw.get(field)
            ]
    return errors


def parse_committed_critical_path_health(raw: object) -> dict[str, object]:
    """Parse one committed projection, or raise `ValueError` with the first reason."""
    errors = committed_critical_path_health_errors(raw, "critical_path_health")
    if errors:
        raise ValueError(errors[0])
    assert isinstance(raw, Mapping)
    metrics = raw.get("metrics")
    # Rebuilt rather than shared, so no caller-held dict aliases the summary.
    return {
        "schema_version": CRITICAL_PATH_HEALTH_SCHEMA_VERSION,
        "privacy": CRITICAL_PATH_HEALTH_PRIVACY,
        "executor": raw.get("executor"),
        "model": raw.get("model"),
        "environment": raw.get("environment"),
        "task_revisions": [dict(item) for item in _items(raw.get("task_revisions"))],
        "metrics": dict(metrics) if isinstance(metrics, Mapping) else None,
        "phase_attribution": [dict(item) for item in _items(raw.get("phase_attribution"))],
        "resource_attribution": [dict(item) for item in _items(raw.get("resource_attribution"))],
        "evidence_gaps": [dict(item) for item in _items(raw.get("evidence_gaps"))],
    }


def render_critical_path_health_lines(section: Mapping[str, object]) -> list[str]:
    """The critical-path facts as the run health text surface reads them."""
    metrics = section.get("metrics")
    gaps = section.get("evidence_gaps")
    gap_list = [gap for gap in gaps if isinstance(gap, Mapping)] if isinstance(gaps, list) else []
    if not isinstance(metrics, Mapping):
        return [
            f"Critical path: unavailable (evidence gaps: {len(gap_list)})",
            f"Top critical stages: unavailable (evidence gaps: {len(gap_list)})",
            *_gap_lines(gap_list),
        ]
    words = ", ".join(f"{word} {metrics.get(name)} ms" for word, name in (("wall clock", "wall_clock_ms"), ("active", "active_ms"), ("queue", "queue_ms"), ("critical path", "critical_path_ms")))
    lines = [f"Critical path: {words}"]
    phases = section.get("phase_attribution")
    stages = [stage for stage in phases if isinstance(stage, Mapping)] if isinstance(phases, list) else []
    if not stages:
        lines.append("Top critical stages: none")
    else:
        lines.append("Top critical stages:")
        ranked = sorted(stages, key=lambda stage: (-_active_ms(stage), str(stage.get("name"))))[:TOP_CRITICAL_STAGES]
        lines += [
            f"- {stage.get('name')}: {stage.get('active_ms')} ms active across {stage.get('task_count')} tasks"
            for stage in ranked
        ]
    return [*lines, *_gap_lines(gap_list)]


def _gap_lines(gaps: list[Mapping[str, object]]) -> list[str]:
    if not gaps:
        return ["Critical path evidence gaps: none"]
    return [f"Critical path evidence gaps: {len(gaps)}", *(f"- {gap.get('task_id') or '(no task)'}: {gap.get('code')}" for gap in gaps)]


def _active_ms(stage: Mapping[str, object]) -> int:
    return stage.get("active_ms") if _is_int(stage.get("active_ms")) else 0


def _items(raw: object) -> list[Mapping[str, object]]:
    return [item for item in raw if isinstance(item, Mapping)] if isinstance(raw, list) else []


def _key_set_errors(payload: Mapping[str, object], allowed: frozenset[str], label: str) -> list[str]:
    errors = [f"{label} has an unsupported key: {key}" for key in sorted(set(payload) - allowed)]
    errors.extend(f"{label} is missing a required key: {key}" for key in sorted(allowed - set(payload)))
    return errors


def _list_errors(raw: object, label: str) -> list[str] | None:
    """The list-level reasons to refuse `raw`; `None` when it is a bounded list."""
    if not isinstance(raw, list):
        return [f"{label} must be a list"]
    if len(raw) > MAX_CRITICAL_PATH_SECTION_ITEMS:
        return [f"{label} must contain at most {MAX_CRITICAL_PATH_SECTION_ITEMS} items"]
    return None


def _opaque_ref_errors(value: object, label: str) -> list[str]:
    if not isinstance(value, str) or not value:
        return [f"{label} must be a non-empty opaque metadata reference"]
    try:
        require_opaque_metadata_ref(value, field=label)
    except ValueError:
        return [f"{label} must be a safe opaque metadata reference"]
    return []


def _revision_errors(raw: object, label: str) -> list[str]:
    item_label = f"{label}.task_revisions"
    shape = _list_errors(raw, item_label)
    if shape is not None:
        return shape
    errors: list[str] = []
    task_ids: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != _REVISION_KEYS:
            errors.append(f"{item_label}[{index}] must use exactly task_id and revision")
            continue
        for field in ("task_id", "revision"):
            errors.extend(_opaque_ref_errors(item.get(field), f"{item_label}[{index}].{field}"))
        if isinstance(item.get("task_id"), str):
            task_ids.append(str(item["task_id"]))
    if task_ids != sorted(set(task_ids)):
        errors.append(f"{item_label} must list unique task_ids in ascending order")
    return errors


def _metrics_errors(raw: Mapping[str, object], label: str) -> list[str]:
    key_errors = _key_set_errors(raw, _METRICS_KEYS, label)
    if key_errors:
        return key_errors
    # `cleanup_tail_ms` is the one metric the producer can derive negatively
    # (cleanup finishing before the last non-cleanup task), so it is held to
    # `int` rather than to `>= 0`; a committed projection carries it as-is.
    return [
        f"{label}.{name} must be {'an integer' if name == 'cleanup_tail_ms' else 'a nonnegative integer'}"
        for name in sorted(_METRICS_KEYS)
        if not _is_int(raw.get(name), minimum=None if name == "cleanup_tail_ms" else 0)
    ]


def _attribution_errors(raw: object, label: str) -> list[str]:
    shape = _list_errors(raw, label)
    if shape is not None:
        return shape
    errors: list[str] = []
    names: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != _ATTRIBUTION_KEYS:
            errors.append(f"{label}[{index}] must use exactly name, active_ms, queue_ms, repeated_cost_ms, and task_count")
            continue
        errors.extend(_opaque_ref_errors(item.get("name"), f"{label}[{index}].name"))
        errors += [
            f"{label}[{index}].{field} must be a nonnegative integer"
            for field in ("active_ms", "queue_ms", "repeated_cost_ms", "task_count")
            if not _is_int(item.get(field), minimum=0)
        ]
        if isinstance(item.get("name"), str):
            names.append(str(item["name"]))
    if names != sorted(set(names)):
        errors.append(f"{label} must list unique names in ascending order")
    return errors


def _gap_errors(raw: object, label: str) -> list[str]:
    item_label = f"{label}.evidence_gaps"
    shape = _list_errors(raw, item_label)
    if shape is not None:
        return shape
    errors: list[str] = []
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != _GAP_KEYS:
            errors.append(f"{item_label}[{index}] must use exactly task_id and code")
            continue
        task_id = item.get("task_id")
        if task_id != "":
            errors.extend(_opaque_ref_errors(task_id, f"{item_label}[{index}].task_id"))
        code = item.get("code")
        code_errors = _opaque_ref_errors(code, f"{item_label}[{index}].code")
        errors.extend(code_errors)
        if isinstance(task_id, str) and isinstance(code, str) and not code_errors:
            pairs.append((task_id, code))
    if pairs != sorted(set(pairs)):
        errors.append(f"{item_label} must list unique task and code pairs in ascending order")
    return errors


def _is_int(value: object, *, minimum: int | None = None) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    return minimum is None or value >= minimum
