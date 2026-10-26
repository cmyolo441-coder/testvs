"""Explicit post-composition scan; separate from metadata admission and approval."""
from __future__ import annotations

from hashlib import sha256
import re

from .handoff_risk_model import MAX_BRIEF_BYTES, ScanError, ScanInput, ScanReport, empty_report, error_report
from .handoff_risk_repository import observe_repository
from .handoff_risk_rules import RuleContext, brief_findings


def scan_handoff(inputs: ScanInput) -> ScanReport:
    """Compose bounded brief and repository observations without storing them."""
    report = empty_report(inputs.repo is not None)
    try:
        if inputs.brief is None and inputs.repo is None:
            raise ScanError("input_required")
        if not inputs.protected_branches or len(inputs.protected_branches) > 32 or any(
            not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,199}", branch)
            or ".." in branch or "//" in branch or branch.endswith(("/", ".", ".lock"))
            or any(part.startswith(".") for part in branch.split("/"))
            for branch in inputs.protected_branches
        ):
            raise ScanError("protected_branch_invalid")
        text = None
        if inputs.brief is not None:
            if len(inputs.brief) > MAX_BRIEF_BYTES:
                raise ScanError("brief_oversized")
            try:
                text = inputs.brief.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ScanError("invalid_utf8") from exc
            report["input_summary"].update(brief_bytes=len(inputs.brief), brief_sha256=sha256(inputs.brief).hexdigest())
        branch = None
        if inputs.repo is not None:
            observation = observe_repository(inputs.repo)
            branch = observation.branch
            report["findings"].extend(observation.findings)
            report["input_summary"].update(
                tracked_count=observation.tracked_count, dirty_count=observation.dirty_count,
                untracked_count=observation.untracked_count,
            )
        if text is not None:
            report["findings"].extend(brief_findings(text, RuleContext(inputs.protected_branches, branch)))
        report["findings"].sort(key=lambda item: item["id"])
        for item in report["findings"]:
            report["summary"][item["severity"]] += 1
        report["summary"]["total"] = len(report["findings"])
        report["verdict"] = "high_risk" if report["summary"]["high"] else "advisory" if report["findings"] else "clear"
        return report
    except ScanError as exc:
        return error_report(exc)
    except (OSError, ValueError):
        # Git metadata parsing and filesystem metadata access are system boundaries.
        return error_report(ScanError("repository_unreadable"))
