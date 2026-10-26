"""Blocked-source recovery for deep research (`research_source_recovery/v1`, issue #1527).

When a research source is blocked, the strongest safe evidence is the goal,
and the order is fixed: classify the failure, hand it once to the host's
reviewed blocked-page recovery capability when the class is one that
capability addresses, and name a retrieval gap with its reason otherwise.

Three decisions live here and nothing else:

- `plan_source_recovery` maps one failure to exactly one next step. A direct
  retrieval that succeeded gets no recovery route at all. HTTP 403, HTTP 429,
  a paywall, a WAF, or a bot wall gets one handoff to
  `blocked-page-recovery` when the host offers it and budget remains. An
  authentication-bound source, a class the capability does not address, a
  missing capability, or a spent budget gets an unresolved gap. A source
  whose one handoff is already spent gets no second one.
- `build_research_source_recovery` mints the record of what happened to one
  source: canonical URL, failure class, route, attempt count, retrieval time,
  evidence class, outcome, gap reason, and residual uncertainty.
- `research_source_recovery_errors` checks a run's records together and
  refuses a retry loop: two handoffs for the same canonical URL.

OMH performs no retrieval. The handoff is a prepared request for the host's
own capability, built only from typed fields -- it has no field that could
carry text from the blocked page -- and it never asks for a login, a payment,
a credential, or an access-control bypass. A record is retrieval metadata the
host reported, not execution, review, CI, or publication evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..system.metadata_safety import is_body_shaped_metadata_text, is_secret_value_shaped
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


RESEARCH_SOURCE_RECOVERY_SCHEMA_VERSION = "research_source_recovery/v1"

#: The host capability a recoverable failure is handed to. Hermes ships it as
#: the bundled `blocked-page-recovery` skill; another host names its own.
RECOVERY_CAPABILITY = "blocked-page-recovery"

#: The failure classes the recovery capability addresses.
RECOVERABLE_FAILURE_CLASSES = ("http_403", "http_429", "paywall", "waf", "bot_wall")

#: Every failure class a direct retrieval can be filed under. HTTP status
#: alone can be ambiguous, so an unclassifiable failure stays `unknown`.
FAILURE_CLASSES = RECOVERABLE_FAILURE_CLASSES + (
    "authentication_required",
    "missing_archive",
    "network_failure",
    "unknown",
)

ROUTES = (RECOVERY_CAPABILITY, "none")

#: A live page and a historical capture are separate evidence; `none` is a
#: source nothing was retrieved for.
EVIDENCE_CLASSES = ("live_page", "historical_capture", "none")

OUTCOMES = ("recovered", "unresolved_gap")

GAP_REASONS = (
    "authentication_required",
    "unsupported_failure_class",
    "missing_capability",
    "budget_exhausted",
    "recovery_failed",
)

#: One bounded recovery attempt per source per failure episode.
MAX_RECOVERY_ATTEMPTS = 1

RESEARCH_SOURCE_RECOVERY_KEYS = (
    "canonical_url",
    "captured_at",
    "claim_boundary",
    "evidence_class",
    "failure_class",
    "gap_reason",
    "outcome",
    "recovery_attempts",
    "residual_uncertainty",
    "retrieved_at",
    "route",
    "schema_version",
)

HANDOFF_KEYS = (
    "canonical_url",
    "capability",
    "failure_class",
    "freshness_constraint",
    "max_attempts",
    "retrieval_budget_remaining",
    "source_requirement",
)

CLAIM_BOUNDARY = (
    "A research source recovery record states what the host reported for one blocked source. OMH "
    "performed no retrieval, bypassed no login, payment, or access control, and the record is not "
    "execution, review, CI, or publication evidence."
)

MAX_URL_CHARS = 2048
MAX_LINE_CHARS = 200

_LABEL = "research_source_recovery"

_VOCABULARIES = (
    ("failure_class", FAILURE_CLASSES),
    ("route", ROUTES),
    ("evidence_class", EVIDENCE_CLASSES),
    ("outcome", OUTCOMES),
)

_STRING_FIELDS = (
    "canonical_url",
    "captured_at",
    "evidence_class",
    "failure_class",
    "gap_reason",
    "outcome",
    "residual_uncertainty",
    "retrieved_at",
    "route",
)


class ResearchSourceRecoveryError(ValueError):
    """Raised when a recovery plan or record cannot be built from what was supplied."""


def plan_source_recovery(
    *,
    canonical_url: str,
    failure_class: str,
    capability_available: bool,
    retrieval_budget_remaining: int,
    recovery_attempts_spent: int = 0,
    source_requirement: str = "",
    freshness_constraint: str = "",
) -> dict[str, Any]:
    """The one next step for one source.

    `failure_class` is empty when direct retrieval succeeded. The result's
    `action` is `cite_direct` (no recovery route), `handoff` (one prepared
    request for the recovery capability), `unresolved_gap` (a validated
    record naming the reason), or `already_attempted` (this source's one
    handoff is spent; its recorded outcome stands).
    """
    url = str(canonical_url or "").strip()
    url_errors = _url_errors(url)
    if url_errors:
        raise ResearchSourceRecoveryError(url_errors[0])
    if not failure_class:
        return {"action": "cite_direct", "handoff": None, "record": None}
    if failure_class not in FAILURE_CLASSES:
        raise ResearchSourceRecoveryError(f"{_LABEL} failure_class must be one of {', '.join(FAILURE_CLASSES)}")
    if recovery_attempts_spent >= MAX_RECOVERY_ATTEMPTS:
        return {"action": "already_attempted", "handoff": None, "record": None}
    reason = _gap_reason_before_handoff(failure_class, capability_available, retrieval_budget_remaining)
    if reason:
        record = build_research_source_recovery(
            canonical_url=url,
            failure_class=failure_class,
            outcome="unresolved_gap",
            gap_reason=reason,
            residual_uncertainty=f"no recovery attempted: {reason}",
        )
        return {"action": "unresolved_gap", "handoff": None, "record": record}
    handoff = {
        "capability": RECOVERY_CAPABILITY,
        "canonical_url": url,
        "failure_class": failure_class,
        "source_requirement": str(source_requirement or "").strip(),
        "freshness_constraint": str(freshness_constraint or "").strip(),
        "retrieval_budget_remaining": retrieval_budget_remaining,
        "max_attempts": MAX_RECOVERY_ATTEMPTS,
    }
    for key in ("source_requirement", "freshness_constraint"):
        if _line_errors(handoff[key]):
            raise ResearchSourceRecoveryError(f"{_LABEL} {key} must be one bounded line without secrets")
    return {"action": "handoff", "handoff": handoff, "record": None}


def build_research_source_recovery(
    *,
    canonical_url: str,
    failure_class: str,
    outcome: str,
    gap_reason: str = "",
    evidence_class: str = "none",
    retrieved_at: str = "",
    captured_at: str = "",
    residual_uncertainty: str = "",
) -> dict[str, Any]:
    """Mint the record of one blocked source, or refuse.

    The route and attempt count follow from the outcome and gap reason: a
    recovered source and a failed recovery each spent the one handoff, and
    every other gap spent none. Nothing is inferred beyond that.
    """
    attempted = outcome == "recovered" or gap_reason == "recovery_failed"
    record = {
        "schema_version": RESEARCH_SOURCE_RECOVERY_SCHEMA_VERSION,
        "canonical_url": str(canonical_url or "").strip(),
        "failure_class": str(failure_class or ""),
        "route": RECOVERY_CAPABILITY if attempted else "none",
        "recovery_attempts": MAX_RECOVERY_ATTEMPTS if attempted else 0,
        "retrieved_at": str(retrieved_at or "").strip(),
        "evidence_class": str(evidence_class or ""),
        "captured_at": str(captured_at or "").strip(),
        "outcome": str(outcome or ""),
        "gap_reason": str(gap_reason or ""),
        "residual_uncertainty": " ".join(str(residual_uncertainty or "").split()),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    errors = validate_research_source_recovery(record)
    if errors:
        raise ResearchSourceRecoveryError(errors[0])
    return record


def validate_research_source_recovery(record: Any) -> list[str]:
    """Every contract violation in one record, structural faults first."""
    if not isinstance(record, Mapping):
        return [f"{_LABEL} must be an object"]
    errors: list[str] = []
    extra = sorted(set(record) - set(RESEARCH_SOURCE_RECOVERY_KEYS))
    if extra:
        errors.append(f"{_LABEL} has unsupported keys: {extra}")
    missing = sorted(set(RESEARCH_SOURCE_RECOVERY_KEYS) - set(record))
    if missing:
        errors.append(f"{_LABEL} is missing keys: {missing}")
    if record.get("schema_version") != RESEARCH_SOURCE_RECOVERY_SCHEMA_VERSION:
        errors.append(f"{_LABEL} schema_version must be {RESEARCH_SOURCE_RECOVERY_SCHEMA_VERSION}")
    for key in _STRING_FIELDS:
        if not isinstance(record.get(key), str):
            errors.append(f"{_LABEL} {key} must be a string")
    attempts = record.get("recovery_attempts")
    if not isinstance(attempts, int) or isinstance(attempts, bool):
        errors.append(f"{_LABEL} recovery_attempts must be an integer")
    if errors:
        return errors
    for key, allowed in _VOCABULARIES:
        if record[key] not in allowed:
            errors.append(f"{_LABEL} {key} must be one of {', '.join(allowed)}")
    if record["claim_boundary"] != CLAIM_BOUNDARY:
        errors.append(f"{_LABEL} claim_boundary must state the record boundary")
    errors.extend(_url_errors(record["canonical_url"]))
    if _line_errors(record["residual_uncertainty"]):
        errors.append(f"{_LABEL} residual_uncertainty must be one bounded line without secrets")
    for key in ("retrieved_at", "captured_at"):
        if record[key] and not _is_iso_timestamp(record[key]):
            errors.append(f"{_LABEL} {key} must be an ISO-8601 timestamp")
    if errors:
        return errors
    errors.extend(_attempt_errors(record))
    errors.extend(_outcome_errors(record))
    return errors


def research_source_recovery_errors(records: Sequence[Any]) -> list[str]:
    """Every violation across one run's records, including retry loops.

    A source gets one handoff per failure episode. Two records that each
    spent a handoff on the same canonical URL are a retry loop, whatever
    their outcomes.
    """
    errors: list[str] = []
    attempted: set[str] = set()
    for index, record in enumerate(records):
        issues = validate_research_source_recovery(record)
        errors.extend(f"record {index}: {issue}" for issue in issues)
        if issues or record["recovery_attempts"] == 0:
            continue
        if record["canonical_url"] in attempted:
            errors.append(
                f"record {index}: {_LABEL} retries {record['canonical_url']}; "
                f"a source gets at most {MAX_RECOVERY_ATTEMPTS} recovery handoff"
            )
        attempted.add(record["canonical_url"])
    return errors


def _gap_reason_before_handoff(failure_class: str, capability_available: bool, budget: int) -> str:
    if failure_class == "authentication_required":
        return "authentication_required"
    if failure_class not in RECOVERABLE_FAILURE_CLASSES:
        return "unsupported_failure_class"
    if not capability_available:
        return "missing_capability"
    if budget <= 0:
        return "budget_exhausted"
    return ""


def _attempt_errors(record: Mapping[str, Any]) -> list[str]:
    attempts = record["recovery_attempts"]
    if attempts < 0 or attempts > MAX_RECOVERY_ATTEMPTS:
        return [f"{_LABEL} recovery_attempts must be 0 or {MAX_RECOVERY_ATTEMPTS}; a source is never retried in a loop"]
    if (attempts == 0) != (record["route"] == "none"):
        return [f"{_LABEL} route {record['route']} disagrees with recovery_attempts {attempts}"]
    if attempts and record["failure_class"] not in RECOVERABLE_FAILURE_CLASSES:
        return [f"{_LABEL} failure_class {record['failure_class']} is never handed to {RECOVERY_CAPABILITY}"]
    return []


def _outcome_errors(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    evidence = record["evidence_class"]
    reason = record["gap_reason"]
    if record["outcome"] == "recovered":
        if reason:
            errors.append(f"{_LABEL} a recovered source carries no gap_reason")
        if evidence == "none":
            errors.append(f"{_LABEL} a recovered source names its evidence_class")
        if not record["retrieved_at"]:
            errors.append(f"{_LABEL} a recovered source records when it was retrieved")
        if evidence == "historical_capture" and not record["captured_at"]:
            errors.append(f"{_LABEL} a historical capture carries its provider-reported captured_at")
        if evidence == "live_page" and record["captured_at"]:
            errors.append(f"{_LABEL} a live page carries no captured_at; a capture is never the live page")
        return errors
    if reason not in GAP_REASONS:
        errors.append(f"{_LABEL} an unresolved gap names one gap_reason: {', '.join(GAP_REASONS)}")
    if evidence != "none" or record["captured_at"]:
        errors.append(f"{_LABEL} an unresolved gap carries no evidence; it is never a partial success")
    if not record["residual_uncertainty"]:
        errors.append(f"{_LABEL} an unresolved gap states its residual_uncertainty")
    if errors:
        return errors
    if (reason == "recovery_failed") != (record["recovery_attempts"] == MAX_RECOVERY_ATTEMPTS):
        errors.append(f"{_LABEL} only recovery_failed follows a spent handoff")
    expected = _gap_reason_before_handoff(record["failure_class"], True, 1)
    if expected and reason != expected:
        errors.append(f"{_LABEL} failure_class {record['failure_class']} leaves gap_reason {expected}")
    return errors


def _url_errors(value: str) -> list[str]:
    if not value:
        return [f"{_LABEL} canonical_url is required"]
    if len(value) > MAX_URL_CHARS:
        return [f"{_LABEL} canonical_url exceeds {MAX_URL_CHARS} characters"]
    if not (value.startswith("https://") or value.startswith("http://")):
        return [f"{_LABEL} canonical_url must be an http(s) URL"]
    if any(char.isspace() or ord(char) < 32 for char in value):
        return [f"{_LABEL} canonical_url must not contain whitespace or control characters"]
    authority = value.split("://", 1)[1].split("/", 1)[0]
    if "@" in authority or not authority or is_secret_value_shaped(value):
        return [f"{_LABEL} canonical_url must not carry credentials"]
    return []


def _line_errors(value: str) -> list[str]:
    if is_body_shaped_metadata_text(value, limit=MAX_LINE_CHARS) or is_secret_value_shaped(value):
        return ["not one bounded line without secrets"]
    return []


def _is_iso_timestamp(value: str) -> bool:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        strict_fromisoformat(text)
    except ValueError:
        return False
    return True
