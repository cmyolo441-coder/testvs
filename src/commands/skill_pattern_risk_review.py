"""CLI entry for `skill_pattern_risk_review/v1`.

The workflow's own docstring said it plainly: "no production surface mints a
review yet; this module is the contract and `tests/test_skill_pattern_risk_review.py`
is its only caller." So the security step of borrowing a third-party skill
pattern existed as a library and could not be run by anyone. #1571 asks for the
step to be reachable, and this is the reachable form of it.

The command mints exactly one review over one explicitly named local directory:
it runs the same `plugin_risk_audit/v1` scan `omh ops plugin-risk-audit` runs,
cites that payload, and attaches the reviewer's four separate judgments --
usefulness, risk, confidence, native reproduction. It re-scans nothing later and
imports, installs, registers and executes nothing, exactly as the audit does not.

A reviewer decision is optional here for the reason the workflow builds it that
way: minting a review is not deciding one, and a clean scan must never read as
an approval.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..installer import OmhError
from ..workflows.plugin_risk_audit import audit_plugin_risk
from ..workflows.skill_pattern_risk_review import (
    CONFIDENCE_LEVELS,
    REVIEWER_DECISIONS,
    RISK_RESOLUTIONS,
    SkillPatternRiskReviewError,
    build_skill_pattern_risk_review,
)
from ..local_store import utc_now
from .common import _print_json


def _resolutions(values: list[str]) -> list[dict[str, str]]:
    """Parse `--resolve category=resolution:target` into resolution rows.

    A risk resolves to a native constraint OMH adopts or to an explicit
    rejection, and each names its target, so the flag carries all three parts
    rather than leaving one to be inferred.
    """
    rows: list[dict[str, str]] = []
    for value in values:
        category, separator, remainder = value.partition("=")
        resolution, target_separator, target = remainder.partition(":")
        if not separator or not target_separator or not category.strip() or not target.strip():
            raise OmhError(
                "--resolve must be given as category=resolution:target, for example "
                "network_request=rejected:reach_a_remote_endpoint, got: " + value
            )
        if resolution not in RISK_RESOLUTIONS:
            raise OmhError(f"--resolve resolution must be one of {list(RISK_RESOLUTIONS)}, got: {resolution}")
        row = {"category": category.strip(), "resolution": resolution}
        key = "native_constraint" if resolution == "native_constraint" else "prohibited_behavior"
        row[key] = target.strip()
        rows.append(row)
    return rows


def cmd_ops_skill_pattern_risk_review(args: argparse.Namespace) -> int:
    decision = (
        {"decided_by": args.decided_by, "decision": args.decision}
        if args.decided_by or args.decision
        else None
    )
    # Parse the flag before the scan. The shape of `--resolve` is decidable from
    # the argument alone, and a caller who mistyped it should be told that
    # rather than made to wait on directory I/O that will be discarded. It also
    # keeps the refusal identical everywhere: `read_static_plugin_sources`
    # refuses outright where `O_NOFOLLOW` is absent, so a scan-first order made
    # an argument error unreportable on that platform.
    resolutions = _resolutions(args.resolve)
    try:
        audit = audit_plugin_risk(Path(args.skill_root))
        review = build_skill_pattern_risk_review(
            skill_ref=args.skill_ref,
            audit=audit,
            intended_outcome=args.intended_outcome,
            procedure_steps=args.step,
            required_authority=args.authority,
            required_data=args.data,
            side_effects=args.side_effect,
            risk_resolutions=resolutions,
            confidence_level=args.confidence_level,
            confidence_basis=args.confidence_basis,
            evidence_limits=args.evidence_limit,
            safe_pattern=args.safe_pattern,
            native_constraints=args.native_constraint,
            prohibited_behaviors=args.prohibited,
            reviewer_decision=decision,
            prepared_at=args.prepared_at or utc_now(),
        )
    except (OSError, SkillPatternRiskReviewError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(review)
    return 0


def add_ops_skill_pattern_risk_review_command(ops_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    review = ops_sub.add_parser(
        "skill-pattern-risk-review",
        help=(
            "Mint one skill_pattern_risk_review/v1 over an explicit local skill directory: what the pattern "
            "is worth, what it costs, how far the evidence goes, and what OMH may reproduce natively."
        ),
        description=(
            "Runs the same bounded static plugin_risk_audit/v1 scan over one explicitly named local "
            "directory, cites that payload, and records the reviewer's usefulness, risk, confidence and "
            "native-reproduction judgments as four separate blocks. Every risk category the scan reports "
            "must resolve to a named native constraint or an explicit rejection. Nothing is imported, "
            "installed, registered or executed, and a clean scan never reads as an approval: only a "
            "recorded reviewer decision approves anything, and what it approves is building a native "
            "pattern, never adopting the source."
        ),
    )
    review.add_argument("--skill-root", required=True, help="Explicit local skill or plugin directory to scan.")
    review.add_argument("--skill-ref", required=True, help="Opaque reference for the reviewed skill.")
    review.add_argument("--intended-outcome", required=True, help="What the procedure is for, in one line.")
    review.add_argument("--step", action="append", default=[], help="One procedure step; repeat in order.")
    review.add_argument("--authority", action="append", default=[], help="One required authority; repeat per authority.")
    review.add_argument("--data", action="append", default=[], help="One kind of required data; repeat per kind.")
    review.add_argument("--side-effect", action="append", default=[], help="One observable side effect; repeat per effect.")
    review.add_argument(
        "--resolve",
        action="append",
        default=[],
        help="Resolve one audited risk category as category=resolution:target; repeat per category.",
    )
    review.add_argument("--confidence-level", choices=CONFIDENCE_LEVELS, required=True)
    review.add_argument("--confidence-basis", action="append", default=[], help="What the confidence rests on; repeat.")
    review.add_argument("--evidence-limit", action="append", default=[], help="What the evidence cannot show; repeat.")
    review.add_argument("--safe-pattern", required=True, help="The part OMH may reproduce natively, in one line.")
    review.add_argument("--native-constraint", action="append", default=[], help="One constraint the native version binds itself to; repeat.")
    review.add_argument("--prohibited", action="append", default=[], help="One behavior the native version must not copy; repeat.")
    review.add_argument("--decided-by", default="", help="Reviewer reference; requires --decision.")
    review.add_argument("--decision", choices=REVIEWER_DECISIONS, default=None, help="Reviewer decision; requires --decided-by.")
    review.add_argument("--prepared-at", default="", help="Timestamp to stamp instead of the current UTC time.")
    review.set_defaults(func=cmd_ops_skill_pattern_risk_review)
