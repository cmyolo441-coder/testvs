from __future__ import annotations

from copy import deepcopy
from html import escape

from .design_direction_iterations_schema import (
    MAX_MODEL_ATTEMPTS,
    MAX_SNAPSHOTS,
    TERMINAL_REASONS,
    _budget_usage,
    _criteria,
    _score_records,
    _scores_are_comparable,
    _snapshot,
    canonical_digest,
)
from .design_directions import choose_design_direction, render_design_direction_set_html, validate_design_direction_set


def revise_design_direction_iteration(
    iteration: dict[str, object],
    *,
    parent_revision_digest: str,
    feedback_reference: str,
    feedback_delta: tuple[str, ...],
    direction_set: object,
    successors: tuple[tuple[str, tuple[str, ...], str], ...],
    criteria_revision: str,
    criteria_dimensions: tuple[str, ...],
    scores: tuple[tuple[str, float, str, str, str], ...] = (),
    model_attempts: tuple[tuple[str, str, int | None, float | None, float | None, str | None], ...] = (),
) -> dict[str, object]:
    """Append one material, parent-bound revision or return its exact replay."""
    _ensure_valid(iteration)
    prior_snapshots = _record_list(iteration["snapshots"], "snapshots")
    current = prior_snapshots[-1]
    key = canonical_digest((parent_revision_digest, feedback_reference, feedback_delta))
    replay = _replay(iteration, key)
    if replay is not None:
        return replay
    if _record(iteration["terminal"], "terminal")["outcome"] != "OPEN":
        raise ValueError("iteration is terminal")
    if parent_revision_digest != current["revision_digest"]:
        raise ValueError("stale parent revision refused")
    if len(prior_snapshots) >= MAX_SNAPSHOTS:
        return _terminalized(iteration, "revision_cap_exhausted")
    used_attempts = _record(iteration["budget_usage"], "budget_usage")["model_attempts"]
    if not isinstance(used_attempts, int) or isinstance(used_attempts, bool):
        raise ValueError("model attempt usage must be an integer")
    if used_attempts >= MAX_MODEL_ATTEMPTS or used_attempts + len(model_attempts) > MAX_MODEL_ATTEMPTS:
        return _terminalized(iteration, "model_call_cap_exhausted")
    repair_count = sum(
        attempt.get("purpose") == "schema_repair"
        for snapshot in prior_snapshots
        for attempt in _record_list(snapshot["model_attempts"], "model_attempts")
    ) + sum(attempt[0] == "schema_repair" for attempt in model_attempts)
    if repair_count > 1:
        raise ValueError("schema repair is allowed at most once per iteration")
    if not isinstance(direction_set, dict) or validate_design_direction_set(direction_set):
        raise ValueError("direction_set must be a valid design_direction_set/v1")
    if direction_set == current["direction_set"]:
        raise ValueError("revision must materially differ from its parent")
    policy = iteration["policy"]
    if not isinstance(policy, dict):
        raise ValueError("iteration policy is invalid")
    threshold = policy["score_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("iteration score threshold is invalid")
    criteria = _criteria(criteria_revision, criteria_dimensions, threshold)
    parent_criteria = current["criteria"]
    if not isinstance(parent_criteria, dict):
        raise ValueError("parent criteria is invalid")
    if criteria_revision == parent_criteria["revision"] and criteria != parent_criteria:
        raise ValueError("criteria revision cannot change its dimensions or threshold")
    candidate_scores = _score_records(scores, criteria)
    comparable = criteria["digest"] == parent_criteria["digest"] and _scores_are_comparable(current["scores"], candidate_scores)
    baseline = str(current["revision_digest"]) if comparable else ""
    snapshot = _snapshot(
        index=len(prior_snapshots),
        parent_digest=parent_revision_digest,
        root_set_digest=str(iteration["root_set_digest"]),
        source_revision_digest=str(iteration["source_revision_digest"]),
        criteria=criteria,
        direction_set=direction_set,
        feedback_reference=feedback_reference,
        feedback_delta=feedback_delta,
        successors=successors,
        scores=scores,
        model_attempts=model_attempts,
        comparison_baseline=baseline,
        comparable=comparable,
    )
    _validate_transition(current, snapshot)
    result = deepcopy(iteration)
    snapshots = result["snapshots"]
    if not isinstance(snapshots, list):
        raise ValueError("iteration snapshots are invalid")
    snapshots.append(snapshot)
    result["budget_usage"] = _budget_usage(snapshots)
    if comparable and _score_mean(snapshot) <= _score_mean(current):
        result["terminal"] = _terminal("BLOCK/REVISE", "no_improvement")
    elif _score_mean(snapshot) >= threshold:
        result["terminal"] = _terminal("STOPPED", "threshold_reached")
    _ensure_valid(result)
    return result


def render_design_direction_iteration_html(iteration: dict[str, object]) -> str:
    """Render the current vocabulary preview with the full iteration trajectory visible."""
    _ensure_valid(iteration)
    current = _record_list(iteration["snapshots"], "snapshots")[-1]
    preview = dict(_record(current["direction_set"], "direction_set"))
    terminal = _record(iteration["terminal"], "terminal")
    accepted_ref = terminal["accepted_option_ref"]
    notice = None
    if isinstance(accepted_ref, str) and accepted_ref:
        preview = choose_design_direction(preview, accepted_ref.rsplit(":", 1)[1])
    elif terminal["outcome"] == "OPEN":
        notice = "Select with omh ops design-direction-iterations select using a current --option-ref from this iteration."
    else:
        notice = f"Selection is closed: {terminal['outcome']} ({terminal['reason']})."
    document = render_design_direction_set_html(preview, selection_notice=notice)
    document = document.replace("</style>", f"{_ITERATION_STYLES}</style>", 1)
    return document.replace("  <footer>", f"{_iteration_trajectory_html(iteration, current)}  <footer>", 1)


# The trajectory is visible text, not data attributes: a person reviewing the
# preview must see every revision, its parent, and why the loop stopped without
# devtools. Digests are machine-consumed identities, so they render in full and
# wrap (`overflow-wrap: anywhere`) instead of truncating or overflowing at 375px.
_ITERATION_STYLES = """
  .iter { margin-top: 2.5rem; padding-top: 1.25rem; border-top: 1px solid var(--page-hair); }
  .iter h2 { font-size: 1.0625rem; margin: 0 0 .3rem; letter-spacing: -.01em; }
  .iter dl { margin: 0 0 1rem; font-size: .75rem; line-height: 1.55; }
  .iter dt { color: var(--page-muted); font-size: .625rem; text-transform: uppercase; letter-spacing: .08em; }
  .iter dd { margin: 0 0 .45rem; }
  .iter dd ul { margin: 0; padding-left: 1.1rem; }
  /* Every identifier in the trajectory wraps, not only digests: long model,
     evaluator, and criteria ids are valid at up to 160 chars and must not
     overflow the 375px viewport. */
  .iter code { overflow-wrap: anywhere; word-break: break-all; }
  .iter-outcome { font-size: .8125rem; margin: 0 0 .35rem; }
  .iter-outcome strong { text-transform: uppercase; letter-spacing: .06em; }
  .iter-budget { color: var(--page-muted); font-size: .75rem; line-height: 1.6; margin: 0 0 1.25rem; }
  .revs { list-style: none; margin: 0; padding: 0; display: grid; gap: 1rem; }
  .rev { border: 1px solid var(--page-hair); border-radius: 10px; padding: .85rem .95rem; }
  .rev h3 { font-size: .8125rem; margin: 0 0 .6rem; }
  .rev .cur { color: var(--page-muted); font-weight: normal; font-size: .625rem;
              text-transform: uppercase; letter-spacing: .08em; }
  .iter-claim { color: var(--page-muted); font-size: .75rem; line-height: 1.6;
                max-width: 62ch; margin: 1.25rem 0 0; }
"""


def _observed(value: object) -> str:
    return "not observed" if value is None else escape(str(value))


def _latency(value: object) -> str:
    return "not observed" if value is None else f"{escape(str(value))} ms"


def _digest_code(value: object, empty: str) -> str:
    text = str(value or "")
    if not text:
        return escape(empty)
    return f"<code>{escape(text)}</code>"


def _record(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"iteration {label} is invalid")
    return value


def _record_list(value: object, label: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"iteration {label} is invalid")
    return value


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"iteration {label} is invalid")
    return value


def _row(term: str, description: str) -> str:
    return f"<dt>{escape(term)}</dt><dd>{description}</dd>"


def _revision_html(snapshot: dict[str, object], *, is_current: bool) -> str:
    rows = [
        _row("revision digest", f"<code>{escape(str(snapshot['revision_digest']))}</code>"),
        _row("parent revision", _digest_code(snapshot["parent_revision_digest"], "root revision - no parent")),
    ]
    refs = "".join(f"<li><code>{escape(ref)}</code></li>" for ref in _string_list(snapshot["option_refs"], "option references"))
    rows.append(_row("stable option references", f"<ul>{refs}</ul>"))
    options = _record_list(_record(snapshot["direction_set"], "direction_set")["options"], "direction set options")
    vocabulary = "".join(
        f"<li>Option {escape(str(option['option_id']).upper())} &middot; "
        f"hierarchy {escape(str(option['hierarchy']))} &middot; palette {escape(str(option['palette']))} &middot; "
        f"typography {escape(str(option['typography']))} &middot; layout {escape(str(option['layout']))} &middot; "
        f"signature {escape(str(option['signature_element']))} &middot; "
        f"avoids {', '.join(escape(pattern) for pattern in _string_list(option['avoid_patterns'], 'avoid patterns'))}</li>"
        for option in options
    )
    rows.append(_row("option vocabulary", f"<ul>{vocabulary}</ul>"))
    feedback = _record(snapshot["feedback"], "feedback")
    reference = str(feedback["reference"] or "")
    if reference:
        delta = ", ".join(escape(term) for term in _string_list(feedback["delta"], "feedback delta"))
        rows.append(_row("feedback", f'reference <code>{escape(reference)}</code> &middot; changed {delta}'))
    else:
        rows.append(_row("feedback", "none - root revision"))
    successors = _record_list(snapshot["successors"], "successors")
    if successors:
        ancestry = "".join(
            f"<li>{escape(str(successor['kind']))}: "
            + (", ".join(f"<code>{escape(ref)}</code>" for ref in _string_list(successor["from_option_refs"], "successor sources")) or "no parent option")
            + f" &rarr; {_digest_code(successor['to_option_ref'], 'no active option')}</li>"
            for successor in successors
        )
        rows.append(_row("option ancestry", f"<ul>{ancestry}</ul>"))
    else:
        rows.append(_row("option ancestry", "root revision - no transitions"))
    criteria = _record(snapshot["criteria"], "criteria")
    dimensions = ", ".join(escape(dimension) for dimension in _string_list(criteria["dimensions"], "criteria dimensions"))
    rows.append(_row(
        "criteria",
        f'revision <code>{escape(str(criteria["revision"]))}</code> &middot; dimensions {dimensions} &middot; '
        f'threshold {escape(str(criteria["score_threshold"]))} &middot; digest <code>{escape(str(criteria["digest"]))}</code>',
    ))
    scores = _record_list(snapshot["scores"], "scores")
    if scores:
        items = "".join(
            f"<li>{escape(str(score['dimension']))} {escape(str(score['score']))} &middot; "
            f'evidence <code>{escape(str(score["evidence_ref"]))}</code> &middot; '
            f'evaluator <code>{escape(str(score["evaluator_id"]))}</code> &middot; '
            f'rubric <code>{escape(str(score["rubric_revision"]))}</code></li>'
            for score in scores
        )
        rows.append(_row("scores", f"<ul>{items}</ul>"))
    else:
        rows.append(_row("scores", "no scores recorded"))
    comparison = _record(snapshot["score_comparison"], "score comparison")
    if comparison["comparable_with_parent"]:
        rows.append(_row(
            "score comparison",
            f'comparable with parent &middot; baseline {_digest_code(comparison["baseline_revision_digest"], "none")}',
        ))
    else:
        rows.append(_row("score comparison", "not comparable with parent - no shared baseline"))
    attempts = _record_list(snapshot["model_attempts"], "model attempts")
    if attempts:
        items = "".join(
            f"<li>{escape(str(attempt['purpose']))} &middot; model <code>{escape(str(attempt['model_id']))}</code> &middot; "
            f"tokens {_observed(attempt['tokens'])} &middot; cost {_observed(attempt['cost'])} &middot; "
            f"latency {_latency(attempt['latency_ms'])} &middot; failure {_observed(attempt['failure_ref'])}</li>"
            for attempt in attempts
        )
        rows.append(_row("model attempts", f"<ul>{items}</ul>"))
    else:
        rows.append(_row("model attempts", "no model attempts recorded"))
    usage = _record(snapshot["usage"], "usage")
    rows.append(_row(
        "revision usage",
        f"{_observed(usage['model_attempts'])} attempt(s) &middot; tokens {_observed(usage['observed_tokens'])} &middot; "
        f"cost {_observed(usage['observed_cost'])} &middot; latency {_latency(usage['observed_latency_ms'])} &middot; "
        f"failures {_observed(usage['failure_count'])}",
    ))
    marker = ' <span class="cur">current</span>' if is_current else ""
    return f'<li class="rev"><h3>Revision {escape(str(snapshot["revision_index"]))}{marker}</h3><dl>{"".join(rows)}</dl></li>'


def _iteration_trajectory_html(iteration: dict[str, object], current: dict[str, object]) -> str:
    terminal = iteration["terminal"]
    policy = iteration["policy"]
    budget = iteration["budget_usage"]
    snapshots = iteration["snapshots"]
    if not isinstance(terminal, dict) or not isinstance(policy, dict) or not isinstance(budget, dict) or not isinstance(snapshots, list):
        raise ValueError("iteration record is invalid")
    outcome = str(terminal["outcome"])
    reason = terminal["reason"]
    detail = f' &middot; reason <code>{escape(str(reason))}</code>' if reason else " &middot; no terminal reason recorded"
    accepted_ref = terminal["accepted_option_ref"]
    if accepted_ref:
        detail += f' &middot; accepted option <code>{escape(str(accepted_ref))}</code>'
    identity = "".join([
        _row("iteration", f'<code>{escape(str(iteration["iteration_id"]))}</code>'),
        _row("root set digest", f'<code>{escape(str(iteration["root_set_digest"]))}</code>'),
        _row("source revision digest", f'<code>{escape(str(iteration["source_revision_digest"]))}</code>'),
        _row(
            "current revision digest",
            f'<code>{escape(str(current["revision_digest"]))}</code> '
            f"(revision {len(snapshots) - 1} across {len(snapshots)} snapshot(s))",
        ),
    ])
    revisions: list[str] = []
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            raise ValueError("iteration snapshots are invalid")
        revisions.append(_revision_html(snapshot, is_current=snapshot is current))
    return f"""
  <section class="iter" data-design-direction-iteration="v1" data-root-set-digest="{escape(str(iteration['root_set_digest']))}" data-revision-digest="{escape(str(current['revision_digest']))}" data-terminal-outcome="{escape(outcome)}">
    <h2>Iteration trajectory</h2>
    <p class="meta">Static vocabulary and history preview. Direction-fit scores are advisory; this is not implementation, browser, accessibility, or visual-QA evidence.</p>
    <dl>{identity}</dl>
    <p class="iter-outcome">Outcome: <strong>{escape(outcome)}</strong>{detail}</p>
    <p class="iter-budget">Budget: {len(snapshots)} of {escape(str(policy['max_snapshots']))} snapshots &middot; {len(snapshots) - 1} of {escape(str(policy['max_revision_rounds']))} revision rounds &middot; {escape(str(budget['model_attempts']))} of {escape(str(policy['max_model_attempts']))} model attempts &middot; tokens {_observed(budget['observed_tokens'])} &middot; cost {_observed(budget['observed_cost'])} &middot; latency {_latency(budget['observed_latency_ms'])} &middot; failures {_observed(budget['failure_count'])} &middot; score threshold {escape(str(policy['score_threshold']))} ({escape(str(policy['no_improvement_rule']))})</p>
    <ol class="revs">{"".join(revisions)}</ol>
    <p class="iter-claim">{escape(str(iteration['claim_boundary']))}</p>
  </section>
"""


def select_design_direction_iteration(iteration: dict[str, object], *, option_ref: str, remember_this: bool = False) -> dict[str, object]:
    """Accept exactly one current stable option reference; stale previews cannot select."""
    _ensure_valid(iteration)
    if _record(iteration["terminal"], "terminal")["outcome"] != "OPEN":
        raise ValueError("iteration is terminal")
    current = _record_list(iteration["snapshots"], "snapshots")[-1]
    if option_ref not in _string_list(current["option_refs"], "option_refs"):
        raise ValueError("stale option reference refused")
    result = deepcopy(iteration)
    result["terminal"] = _terminal("ACCEPT", "accepted", str(current["revision_digest"]), option_ref)
    if remember_this:
        result["memory_promotion"] = {
            "state": "requested_for_review",
            "request": {
                "action": "memory-new",
                "review_required": True,
                "accepted_revision_digest": str(current["revision_digest"]),
                "automatic_write": False,
                "global_promotion": False,
            },
        }
    _ensure_valid(result)
    return result


def terminate_design_direction_iteration(iteration: dict[str, object], *, reason: str) -> dict[str, object]:
    """Record a non-success terminal reason where an operator stops the loop."""
    _ensure_valid(iteration)
    if _record(iteration["terminal"], "terminal")["outcome"] != "OPEN":
        raise ValueError("iteration is terminal")
    if reason not in TERMINAL_REASONS or reason in ("accepted", "threshold_reached", "no_improvement", "revision_cap_exhausted", "model_call_cap_exhausted"):
        raise ValueError("terminal reason must be an operator-observed blocker or cancellation")
    result = _terminalized(iteration, reason)
    _ensure_valid(result)
    return result


def _terminalized(iteration: dict[str, object], reason: str) -> dict[str, object]:
    result = deepcopy(iteration)
    outcome = "CANCELLED" if reason == "cancelled" else "BLOCK/REVISE"
    result["terminal"] = _terminal(outcome, reason)
    _ensure_valid(result)
    return result


def _terminal(outcome: str, reason: str, revision_digest: str | None = None, option_ref: str | None = None) -> dict[str, object]:
    return {"outcome": outcome, "reason": reason, "accepted_revision_digest": revision_digest, "accepted_option_ref": option_ref}


def _replay(iteration: dict[str, object], key: str) -> dict[str, object] | None:
    for snapshot in _record_list(iteration["snapshots"], "snapshots")[1:]:
        if snapshot["idempotency_key"] == key:
            return deepcopy(iteration)
    return None


def _validate_transition(parent: dict[str, object], child: dict[str, object]) -> None:
    parent_refs = set(_string_list(parent["option_refs"], "parent option_refs"))
    child_refs = set(_string_list(child["option_refs"], "child option_refs"))
    covered: set[str] = set()
    targets: set[str] = set()
    for successor in _record_list(child["successors"], "successors"):
        kind = successor["kind"]
        sources = _string_list(successor["from_option_refs"], "from_option_refs")
        target = successor["to_option_ref"]
        if not isinstance(sources, list) or not isinstance(target, str):
            raise ValueError("successors are invalid")
        if kind == "introduced" and sources:
            raise ValueError("introduced successor cannot have a parent")
        if kind == "combined" and len(sources) < 2:
            raise ValueError("combined successor needs at least two parents")
        if kind in ("preserved", "revised", "dropped") and len(sources) != 1:
            raise ValueError("single-parent successor needs exactly one parent")
        if kind == "dropped" and target:
            raise ValueError("dropped successor cannot target an active option")
        if kind != "dropped" and target not in child_refs:
            raise ValueError("successor target must be a current stable option reference")
        if not set(sources).issubset(parent_refs) or covered.intersection(sources):
            raise ValueError("successors must cover each parent option exactly once")
        covered.update(sources)
        if target and target in targets:
            raise ValueError("successors must target each active option exactly once")
        if target:
            targets.add(target)
    if covered != parent_refs or targets != child_refs:
        raise ValueError("successors must preserve complete option ancestry")


def _score_mean(snapshot: dict[str, object]) -> float:
    scores = snapshot["scores"]
    if not isinstance(scores, list) or not scores:
        return -1.0
    values = [float(score["score"]) for score in scores if isinstance(score, dict)]
    return sum(values) / len(values) if values else -1.0


def _ensure_valid(iteration: dict[str, object]) -> None:
    from .design_direction_iterations_validation import validate_design_direction_iteration

    errors = validate_design_direction_iteration(iteration)
    if errors:
        raise ValueError("; ".join(errors))
