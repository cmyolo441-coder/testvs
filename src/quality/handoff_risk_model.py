"""Closed metadata vocabulary for explicit advisory scans (no permission authority)."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final, Literal, TypedDict

MAX_BRIEF_BYTES: Final = 65536
MAX_METADATA_BYTES: Final = 1048576
MAX_ENTRIES: Final = 10000
CLAIM_BOUNDARY: Final = (
    "Advisory deterministic signals in declared bounded inputs only. Clear is not proof "
    "of safety, permission, execution, verification, review, CI, or merge readiness. "
    "Existing preflight, approval and host policy remain authoritative."
)
FindingId = Literal[
    "destructive_git", "destructive_filesystem", "destructive_database", "test_weakening",
    "protected_branch_write", "dirty_worktree", "untracked_files", "tracked_secret_path",
]
Severity = Literal["low", "medium", "high"]
ErrorCategory = Literal[
    "input_required", "input_unreadable", "invalid_utf8", "brief_oversized",
    "repository_invalid", "repository_unreadable", "metadata_oversized",
    "repository_timeout", "command_syntax_invalid", "protected_branch_invalid",
]


@dataclass(frozen=True, slots=True)
class ScanError(Exception):
    category: ErrorCategory

    def __post_init__(self) -> None:
        # Initialize BaseException.args so inherited __str__ retains the category.
        Exception.__init__(self, self.category)


@dataclass(frozen=True, slots=True)
class ScanInput:
    brief: bytes | None = None
    repo: Path | None = None
    protected_branches: tuple[str, ...] = ("main", "master")


class Evidence(TypedDict):
    token_class: str
    sha256: str
    offset: int | None


class Finding(TypedDict):
    id: FindingId
    severity: Severity
    confidence: Literal["medium", "high"]
    evidence: Evidence
    advice_code: str


class InputSummary(TypedDict):
    brief_bytes: int | None
    brief_sha256: str | None
    repo_supplied: bool
    tracked_count: int | None
    dirty_count: int | None
    untracked_count: int | None


class Summary(TypedDict):
    low: int
    medium: int
    high: int
    total: int


class ScanReport(TypedDict):
    schema_version: Literal["handoff_risk_scan/v1"]
    status: Literal["completed", "scan_error"]
    verdict: Literal["clear", "advisory", "high_risk"] | None
    findings: list[Finding]
    summary: Summary
    input_summary: InputSummary
    error_category: ErrorCategory | None
    claim_boundary: str


@dataclass(frozen=True, slots=True)
class Signal:
    kind: FindingId
    source: bytes
    offset: int | None = None


def finding(signal: Signal) -> Finding:
    """Retain only normalized classes plus digest, never source bytes or paths."""
    medium = signal.kind in ("dirty_worktree", "untracked_files")
    return Finding(
        id=signal.kind, severity="medium" if medium else "high", confidence="high",
        evidence=Evidence(token_class=signal.kind, sha256=sha256(signal.source).hexdigest(), offset=signal.offset),
        advice_code="review_workspace_changes" if medium else "confirm_or_security_review",
    )


def empty_report(repo_supplied: bool = False) -> ScanReport:
    return ScanReport(
        schema_version="handoff_risk_scan/v1", status="completed", verdict="clear", findings=[],
        summary=Summary(low=0, medium=0, high=0, total=0),
        input_summary=InputSummary(brief_bytes=None, brief_sha256=None, repo_supplied=repo_supplied,
                                   tracked_count=None, dirty_count=None, untracked_count=None),
        error_category=None, claim_boundary=CLAIM_BOUNDARY,
    )


def error_report(error: ScanError) -> ScanReport:
    report = empty_report()
    report.update(status="scan_error", verdict=None, error_category=error.category)
    return report
