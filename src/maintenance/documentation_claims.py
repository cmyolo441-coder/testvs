"""Deterministic selected-claim audit, separate from generated artifact drift."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import TypedDict

from ..catalogs.documentation_claims import documentation_claims
from .documentation_claims_model import ModelAdapter, ModelRun
from .documentation_claims_worker import run_bounded_probe


DOCUMENTATION_CLAIM_AUDIT_SCHEMA = "documentation_claim_audit/v1"
STATES = ("supported", "stale", "unresolved", "not_run")
MAX_MODEL_RUNS = 3
MAX_TIMEOUT_SECONDS = 30


class ClaimRow(TypedDict):
    id: str
    question: str
    invariant: str
    expected_fact: bool | str
    observed_fact: bool | str | None
    pages: list[str]
    anchors: list[dict[str, str]]
    mode: str
    probe: str
    risk: str
    owner: str
    advisory: bool
    state: str
    observed: bool
    evidence_class: str
    reason: str
    model_run: ModelRun | None


class DocumentationClaimReport(TypedDict):
    schema_version: str
    mode: str
    observed: bool
    ok: bool
    selection: list[str]
    claims: list[ClaimRow]
    summary: dict[str, int]
    generated_artifact_drift: dict[str, object]
    model_evaluation: dict[str, object]
    bounds: dict[str, object]
    claim_boundary: str


def documentation_claims_report(
    *, root: Path | None = None, claim_ids: tuple[str, ...] | None = None,
    enable_model: bool = False, model_adapter: ModelAdapter | None = None,
    model_run_cap: int = 1, timeout: float = 10,
) -> DocumentationClaimReport:
    """Run the reviewed default set or explicitly named claims, once each.

    No dynamic probe registration and no model invocation without both explicit
    enablement and selection. Generated equality is reported but is not semantic
    support. Unknown selections are invocation errors, not empty successful runs.
    """
    if type(model_run_cap) is not int or not 0 <= model_run_cap <= MAX_MODEL_RUNS:
        raise ValueError("model run cap must be between 0 and 3")
    if type(timeout) not in (float, int) or not math.isfinite(timeout) or not 0 < timeout <= MAX_TIMEOUT_SECONDS:
        raise ValueError("per-check timeout must be greater than 0 and at most 30 seconds")
    if enable_model and not claim_ids:
        raise ValueError("model evaluation requires explicit --claim selection")
    catalog = documentation_claims()
    known = {claim.claim_id for claim in catalog}
    if claim_ids is not None and (not claim_ids or set(claim_ids) - known):
        raise ValueError("unknown or empty documentation claim selection")
    selected = set(known if claim_ids is None else claim_ids)
    resolved_root = (root or Path.cwd()).resolve()
    rows: list[ClaimRow] = []
    runs = 0
    attempts = 0
    for claim in sorted(catalog, key=lambda item: item.claim_id):
        advisory = claim.mode == "model_assisted"
        row: ClaimRow = {
            "id": claim.claim_id, "question": claim.question, "invariant": claim.invariant,
            "expected_fact": claim.expected_fact, "observed_fact": None,
            "pages": list(claim.pages), "anchors": [asdict(anchor) for anchor in claim.anchors],
            "mode": claim.mode, "probe": claim.probe, "risk": claim.risk, "owner": claim.owner,
            "advisory": advisory, "state": "not_run", "observed": False,
            "evidence_class": "generated_artifact_drift" if claim.mode == "render_equality" else "documentation_claim",
            "reason": "not_selected", "model_run": None,
        }
        if claim.claim_id not in selected:
            rows.append(row)
            continue
        if advisory and (not enable_model or model_adapter is None or attempts >= model_run_cap):
            row["reason"] = "model_disabled" if not enable_model else "model_unavailable" if model_adapter is None else "model_run_cap"
            rows.append(row)
            continue
        if advisory:
            attempts += 1
        result = run_bounded_probe(resolved_root, claim, timeout, model_adapter if advisory else None)
        if result.get("model_started"):
            runs += 1
        if "error" in result:
            row.update(state="unresolved", reason=result["error"], model_run=result.get("model_run"),
                       observed=bool(result.get("model_started")))
        elif advisory and "model_run" in result:
            model_run = result["model_run"]
            row.update(state=model_run["state"], observed=True, model_run=model_run,
                       observed_fact=model_run["state"] == "supported" if model_run["state"] != "unresolved" else None,
                       reason="advisory_evaluation")
        elif "fact" in result:
            fact = result["fact"]
            row.update(state="supported" if fact == claim.expected_fact else "stale",
                       observed=True, observed_fact=fact, reason="fact_match" if fact == claim.expected_fact else "fact_mismatch")
        else:
            raise RuntimeError("invalid bounded probe result")
        rows.append(row)
    selected_rows = [row for row in rows if row["id"] in selected]
    deterministic = [row for row in selected_rows if not row["advisory"]]
    generated = [row for row in selected_rows if row["evidence_class"] == "generated_artifact_drift"]
    return {
        "schema_version": DOCUMENTATION_CLAIM_AUDIT_SCHEMA,
        "mode": "observed_audit", "observed": any(row["observed"] for row in selected_rows),
        "ok": all(row["state"] == "supported" for row in deterministic),
        "selection": sorted(selected), "claims": rows,
        "summary": {state: sum(row["state"] == state for row in selected_rows) for state in STATES},
        "generated_artifact_drift": {
            "evidence_class": "generated_artifact_drift",
            "state": "not_run" if not generated else "unresolved" if any(row["state"] == "unresolved" for row in generated)
                     else "stale" if any(row["state"] == "stale" for row in generated) else "supported",
            "observed": any(row["observed"] for row in generated),
            "claim_ids": [row["id"] for row in generated],
            "complete_registry_observed": False,
            "registry_command": "omh release drift --json",
        },
        "model_evaluation": {"enabled": enable_model, "runs": runs, "attempts": attempts, "run_cap": model_run_cap,
                             "hard_run_cap": MAX_MODEL_RUNS, "advisory": True, "release_blocking": False},
        "bounds": {"timeout_seconds": timeout, "network_by_default": False, "automatic_doc_edits": False},
        "claim_boundary": (
            "Selected local implementation facts only; catalog enrollment and prepared documentation work are not "
            "observed checks. Generated equality is separate, not semantic support. Model judgments are advisory, "
            "never release gates. This is not full documentation coverage, review, CI, merge, or live Hermes evidence."
        ),
    }


def format_documentation_claims(report: DocumentationClaimReport) -> str:
    lines = ["Documentation claims: " + ("PASS" if report["ok"] else "NEEDS ATTENTION")]
    for row in report["claims"]:
        lines.append(f"{row['state']} {row['id']} [{row['evidence_class']}] owner={row['owner']}")
        lines.append(f"  pages={','.join(row['pages'])} anchors=" + ",".join(f"{a['path']}:{a['symbol']}" for a in row["anchors"]))
        lines.append(f"  expected={row['expected_fact']!r} observed={row['observed_fact']!r} reason={row['reason']}")
    lines.append("Generated registry (separate, not run): omh release drift --json")
    lines.append(report["claim_boundary"])
    return "\n".join(lines)
