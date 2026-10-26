"""One language-diagnostic check over one workspace revision interval.

A language server answers a narrow question quickly: at this revision, which
positions in these files does the analyser object to? That answer is genuinely
useful right after an edit, and it is routinely overstated. "No diagnostics"
gets reported as "verified", and the reader hears compilation, tests, review,
and CI -- none of which a diagnostic pass performs.

This module records the narrow answer with the narrow label attached, so the
overstatement has nowhere to happen:

- `verdict` is a closed vocabulary whose clean member is
  `no_new_diagnostics_observed`, not `passed`. There is no verdict that reads
  as verification because none exists to select.
- `summary_label` is derived, never supplied. A caller cannot hand in prose
  that upgrades its own result.
- `language_diagnostic_supports_claim` answers every claim except
  `fresh_language_diagnostic_check` with False, for every record, including a
  perfectly clean one. The refusal is structural, not a policy string.
- A record whose diagnostics were observed at a revision other than the
  interval end is `stale_diagnostics`, and a record missing its workspace or
  either interval endpoint is `attribution_unavailable`. Neither backs the one
  claim this record can back, because a diagnostic result that cannot be
  attributed to an interval says nothing about what that interval introduced.

OMH observes nothing here. It starts no language server, opens no socket, and
reads no source file; every diagnostic on a record was supplied by a caller
that ran the provider itself. The record is metadata only -- severity, code,
workspace-relative path, and position. Diagnostic *messages* are not a field,
and an input diagnostic carrying one is refused by key name, because a message
is where a source body would arrive.

Every value is derived from the arguments alone. No clock is read: `observed_at`
is a parameter, and it is deliberately excluded from `record_id`, so the same
observation reported twice is the same record.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping

from ..coding.executors import EXECUTOR_PROFILES
from ..system.metadata_safety import require_opaque_metadata_ref

LANGUAGE_DIAGNOSTIC_EVIDENCE_SCHEMA_VERSION = "language_diagnostic_evidence/v1"
LANGUAGE_DIAGNOSTIC_EVIDENCE_PRIVACY = "metadata_only"

# Who observed the check. `wrapper` plus the canonical executor profiles, so a
# record never has to name Codex to describe a Claude Code or Hermes run.
LANGUAGE_DIAGNOSTIC_OWNERS = ("wrapper", *EXECUTOR_PROFILES)

# What the caller's provider actually did. Only `observed` can produce a usable
# result; the other three are preserved rather than collapsed into "clean",
# because a provider that never ran and a provider that found nothing are the
# two states this record exists to keep apart.
LANGUAGE_DIAGNOSTIC_CHECK_STATES = ("observed", "unsupported", "failed", "not_observed")

LANGUAGE_DIAGNOSTIC_SEVERITIES = ("error", "warning", "information", "hint")

# Derived from the revisions, never supplied. `fresh` means the diagnostics
# were observed at the interval end; `stale` means they were observed
# somewhere else; `unknown` means the caller did not say where.
LANGUAGE_DIAGNOSTIC_FRESHNESS_STATES = ("fresh", "stale", "unknown")

LANGUAGE_DIAGNOSTIC_ATTRIBUTION_STATES = ("attributable", "unavailable")

LANGUAGE_DIAGNOSTIC_VERDICTS = (
    "no_new_diagnostics_observed",
    "new_diagnostics_observed",
    "attribution_unavailable",
    "stale_diagnostics",
    "freshness_unknown",
    "provider_unsupported",
    "provider_failed",
    "not_observed",
)
# The two verdicts that describe a fresh, attributable check. Every other
# verdict names a reason the check cannot be read as a result at all.
LANGUAGE_DIAGNOSTIC_USABLE_VERDICTS = ("no_new_diagnostics_observed", "new_diagnostics_observed")

# Claims a reader might try to settle with this record.
LANGUAGE_DIAGNOSTIC_CLAIMS = (
    "fresh_language_diagnostic_check",
    "verification",
    "compilation",
    "test_execution",
    "review",
    "ci",
    "merge_readiness",
    "merge",
)
# The only one it can ever settle. Kept as a tuple rather than a bare string so
# the "everything else is False" invariant is expressible as set arithmetic.
LANGUAGE_DIAGNOSTIC_SUPPORTABLE_CLAIMS = ("fresh_language_diagnostic_check",)
LANGUAGE_DIAGNOSTIC_NOT_EVIDENCE_FOR = tuple(
    claim for claim in LANGUAGE_DIAGNOSTIC_CLAIMS if claim not in LANGUAGE_DIAGNOSTIC_SUPPORTABLE_CLAIMS
)

LANGUAGE_DIAGNOSTIC_CLAIM_BOUNDARY = (
    "A language diagnostic evidence record is one language-diagnostic check over one workspace revision "
    "interval. A clean result means no new diagnostics were observed in that check; it is not compilation, "
    "test, verification, review, CI, merge-readiness, or merge evidence."
)

LANGUAGE_DIAGNOSTIC_EVIDENCE_KEYS = (
    "attribution",
    "baseline_revision",
    "changed_path_count",
    "changed_paths",
    "check_state",
    "claim_boundary",
    "config_digest",
    "diagnostics_revision",
    "end_revision",
    "evidence_refs",
    "freshness",
    "introduced",
    "introduced_count",
    "not_evidence_for",
    "observed_at",
    "owner",
    "privacy",
    "provider",
    "record_id",
    "resolved",
    "resolved_count",
    "schema_version",
    "summary_label",
    "verdict",
    "workspace_id",
)

# The whole of one normalized diagnostic. There is no message, body, snippet,
# or fix field, and there is no room for one: an input carrying a key outside
# this set is refused by name.
LANGUAGE_DIAGNOSTIC_ITEM_KEYS = ("character", "code", "line", "path", "severity", "source")

MAX_REFERENCE_CHARS = 120
MAX_CHANGED_PATHS = 200
MAX_DIAGNOSTICS = 200
MAX_EVIDENCE_REFS = 8
MAX_SUMMARY_LABEL_CHARS = 512
MAX_POSITION = 1_000_000

_ABSOLUTE_WINDOWS_PATH = re.compile(r"^[A-Za-z]:/")


class LanguageDiagnosticEvidenceError(ValueError):
    """Raised when a language-diagnostic observation cannot become a record."""


def build_language_diagnostic_evidence(
    *,
    owner: str,
    provider: str,
    workspace_id: str = "",
    baseline_revision: str = "",
    end_revision: str = "",
    diagnostics_revision: str = "",
    check_state: str = "observed",
    config_digest: str = "",
    changed_paths: Iterable[str] = (),
    introduced: Iterable[Mapping[str, Any]] = (),
    resolved: Iterable[Mapping[str, Any]] = (),
    observed_at: str = "",
    evidence_refs: Iterable[str] = (),
) -> dict[str, Any]:
    """Build one record, or refuse.

    Nothing about the outcome is a parameter. `freshness`, `attribution`,
    `verdict`, and `summary_label` are all derived below, so a caller supplies
    what it observed and never what that observation proves.
    """
    if owner not in LANGUAGE_DIAGNOSTIC_OWNERS:
        raise LanguageDiagnosticEvidenceError(f"language_diagnostic_evidence owner is unsupported: {owner!r}")
    if check_state not in LANGUAGE_DIAGNOSTIC_CHECK_STATES:
        raise LanguageDiagnosticEvidenceError(
            f"language_diagnostic_evidence check_state is unsupported: {check_state!r}"
        )
    safe_provider = _required_ref(provider, field="language_diagnostic_evidence provider")
    safe_workspace = _optional_ref(workspace_id, field="language_diagnostic_evidence workspace_id")
    safe_baseline = _optional_ref(baseline_revision, field="language_diagnostic_evidence baseline_revision")
    safe_end = _optional_ref(end_revision, field="language_diagnostic_evidence end_revision")
    safe_diagnostics_revision = _optional_ref(
        diagnostics_revision, field="language_diagnostic_evidence diagnostics_revision"
    )
    safe_config_digest = _optional_ref(config_digest, field="language_diagnostic_evidence config_digest")
    safe_observed_at = _optional_ref(observed_at, field="language_diagnostic_evidence observed_at")

    normalized_changed_paths = _normalized_paths(changed_paths)
    normalized_introduced = _normalized_diagnostics(introduced, field="introduced")
    normalized_resolved = _normalized_diagnostics(resolved, field="resolved")
    normalized_refs = _normalized_evidence_refs(evidence_refs)

    freshness = _derive_freshness(check_state, safe_end, safe_diagnostics_revision)
    attribution = _derive_attribution(check_state, safe_workspace, safe_baseline, safe_end)
    verdict = _derive_verdict(check_state, freshness, attribution, len(normalized_introduced))
    record = {
        "schema_version": LANGUAGE_DIAGNOSTIC_EVIDENCE_SCHEMA_VERSION,
        "record_id": _record_id(
            owner,
            safe_provider,
            safe_workspace,
            safe_baseline,
            safe_end,
            safe_diagnostics_revision,
            safe_config_digest,
            check_state,
            normalized_introduced,
            normalized_resolved,
        ),
        "owner": owner,
        "provider": safe_provider,
        "config_digest": safe_config_digest,
        "check_state": check_state,
        "workspace_id": safe_workspace,
        "baseline_revision": safe_baseline,
        "end_revision": safe_end,
        "diagnostics_revision": safe_diagnostics_revision,
        "changed_paths": normalized_changed_paths,
        "changed_path_count": len(normalized_changed_paths),
        "observed_at": safe_observed_at,
        "freshness": freshness,
        "attribution": attribution,
        "introduced": normalized_introduced,
        "introduced_count": len(normalized_introduced),
        "resolved": normalized_resolved,
        "resolved_count": len(normalized_resolved),
        "verdict": verdict,
        "summary_label": _summary_label(
            verdict,
            workspace_id=safe_workspace,
            baseline_revision=safe_baseline,
            end_revision=safe_end,
            diagnostics_revision=safe_diagnostics_revision,
            introduced_count=len(normalized_introduced),
        ),
        "evidence_refs": normalized_refs,
        "privacy": LANGUAGE_DIAGNOSTIC_EVIDENCE_PRIVACY,
        "not_evidence_for": list(LANGUAGE_DIAGNOSTIC_NOT_EVIDENCE_FOR),
        "claim_boundary": LANGUAGE_DIAGNOSTIC_CLAIM_BOUNDARY,
    }
    errors = validate_language_diagnostic_evidence(record)
    if errors:
        raise LanguageDiagnosticEvidenceError(errors[0])
    return record


def validate_language_diagnostic_evidence(record: Any) -> list[str]:
    """Return every reason the payload is not a valid record.

    The derived fields are re-derived here rather than merely type-checked. A
    record that reached disk, a wrapper, or a status line with its `verdict`
    edited to a clean one is the exact failure this contract exists to prevent,
    so the check that catches it lives on the read path too.
    """
    if not isinstance(record, dict):
        return ["language_diagnostic_evidence must be an object"]
    errors: list[str] = []
    extra_keys = sorted(set(record) - set(LANGUAGE_DIAGNOSTIC_EVIDENCE_KEYS))
    if extra_keys:
        errors.append(f"language_diagnostic_evidence has unsupported keys: {extra_keys}")
    missing_keys = sorted(set(LANGUAGE_DIAGNOSTIC_EVIDENCE_KEYS) - set(record))
    if missing_keys:
        errors.append(f"language_diagnostic_evidence is missing keys: {missing_keys}")
    if missing_keys:
        return errors

    if record.get("schema_version") != LANGUAGE_DIAGNOSTIC_EVIDENCE_SCHEMA_VERSION:
        errors.append(
            f"language_diagnostic_evidence schema_version must be {LANGUAGE_DIAGNOSTIC_EVIDENCE_SCHEMA_VERSION}"
        )
    if record.get("privacy") != LANGUAGE_DIAGNOSTIC_EVIDENCE_PRIVACY:
        errors.append("language_diagnostic_evidence privacy must be metadata_only")
    if record.get("claim_boundary") != LANGUAGE_DIAGNOSTIC_CLAIM_BOUNDARY:
        errors.append("language_diagnostic_evidence claim_boundary must state the language-diagnostic boundary")
    if list(record.get("not_evidence_for") or ()) != list(LANGUAGE_DIAGNOSTIC_NOT_EVIDENCE_FOR):
        errors.append(
            "language_diagnostic_evidence not_evidence_for must list every claim this record cannot settle"
        )
    for field, vocabulary in (
        ("owner", LANGUAGE_DIAGNOSTIC_OWNERS),
        ("check_state", LANGUAGE_DIAGNOSTIC_CHECK_STATES),
        ("freshness", LANGUAGE_DIAGNOSTIC_FRESHNESS_STATES),
        ("attribution", LANGUAGE_DIAGNOSTIC_ATTRIBUTION_STATES),
        ("verdict", LANGUAGE_DIAGNOSTIC_VERDICTS),
    ):
        if record.get(field) not in vocabulary:
            errors.append(f"language_diagnostic_evidence {field} is unsupported: {record.get(field)!r}")

    errors.extend(_reference_errors(record.get("record_id"), field="record_id", required=True))
    errors.extend(_reference_errors(record.get("provider"), field="provider", required=True))
    for field in ("workspace_id", "baseline_revision", "end_revision", "diagnostics_revision", "config_digest", "observed_at"):
        errors.extend(_reference_errors(record.get(field), field=field, required=False))

    errors.extend(_path_list_errors(record.get("changed_paths")))
    for field in ("introduced", "resolved"):
        errors.extend(_diagnostic_list_errors(record.get(field), field=field))
    errors.extend(_evidence_ref_errors(record.get("evidence_refs")))
    for count_field, list_field in (
        ("changed_path_count", "changed_paths"),
        ("introduced_count", "introduced"),
        ("resolved_count", "resolved"),
    ):
        values = record.get(list_field)
        if isinstance(values, list) and record.get(count_field) != len(values):
            errors.append(f"language_diagnostic_evidence {count_field} must equal len({list_field})")

    label = record.get("summary_label")
    if not isinstance(label, str) or not label.strip():
        errors.append("language_diagnostic_evidence summary_label must be a nonblank string")
    elif len(label) > MAX_SUMMARY_LABEL_CHARS:
        errors.append(
            f"language_diagnostic_evidence summary_label must be at most {MAX_SUMMARY_LABEL_CHARS} characters"
        )

    if errors:
        return errors
    return _derivation_errors(record)


def language_diagnostic_supports_claim(record: Any, claim: str) -> bool:
    """True only for a fresh, attributable check asked about its own claim.

    Every other question -- verification, compilation, tests, review, CI,
    merge-readiness, merge -- is False for every record this module can build.
    A caller cannot reach a True by supplying a cleaner input, because no input
    selects the claim.
    """
    if claim not in LANGUAGE_DIAGNOSTIC_SUPPORTABLE_CLAIMS:
        return False
    if validate_language_diagnostic_evidence(record):
        return False
    return record["verdict"] in LANGUAGE_DIAGNOSTIC_USABLE_VERDICTS


def language_diagnostic_claim_support(record: Any) -> dict[str, Any]:
    """One reportable answer per claim, for a status surface to render."""
    return {
        "schema_version": LANGUAGE_DIAGNOSTIC_EVIDENCE_SCHEMA_VERSION,
        "supported_claims": [
            claim for claim in LANGUAGE_DIAGNOSTIC_CLAIMS if language_diagnostic_supports_claim(record, claim)
        ],
        "unsupported_claims": [
            claim for claim in LANGUAGE_DIAGNOSTIC_CLAIMS if not language_diagnostic_supports_claim(record, claim)
        ],
        "claim_boundary": LANGUAGE_DIAGNOSTIC_CLAIM_BOUNDARY,
    }


def _derive_freshness(check_state: str, end_revision: str, diagnostics_revision: str) -> str:
    if check_state != "observed":
        return "unknown"
    if not end_revision or not diagnostics_revision:
        return "unknown"
    return "fresh" if diagnostics_revision == end_revision else "stale"


def _derive_attribution(check_state: str, workspace_id: str, baseline_revision: str, end_revision: str) -> str:
    if check_state != "observed":
        return "unavailable"
    if workspace_id and baseline_revision and end_revision:
        return "attributable"
    return "unavailable"


def _derive_verdict(check_state: str, freshness: str, attribution: str, introduced_count: int) -> str:
    if check_state == "unsupported":
        return "provider_unsupported"
    if check_state == "failed":
        return "provider_failed"
    if check_state == "not_observed":
        return "not_observed"
    if attribution != "attributable":
        return "attribution_unavailable"
    if freshness == "stale":
        return "stale_diagnostics"
    if freshness != "fresh":
        return "freshness_unknown"
    return "new_diagnostics_observed" if introduced_count else "no_new_diagnostics_observed"


def _summary_label(
    verdict: str,
    *,
    workspace_id: str,
    baseline_revision: str,
    end_revision: str,
    diagnostics_revision: str,
    introduced_count: int,
) -> str:
    """The one line a human reads, derived so no caller can write it.

    Every branch names the check as a language-diagnostic check and none uses
    a word that reads as verification, so the clean branch cannot be quoted
    into a stronger claim than it makes.
    """
    interval = f"workspace {workspace_id} between {baseline_revision} and {end_revision}"
    if verdict == "no_new_diagnostics_observed":
        return (
            f"No new diagnostics were observed in a fresh language-diagnostic check of {interval}. "
            "Compilation, tests, review, and CI are separate and unobserved here."
        )
    if verdict == "new_diagnostics_observed":
        return (
            f"{introduced_count} newly introduced diagnostics were observed in a fresh language-diagnostic "
            f"check of {interval}. Compilation, tests, review, and CI are separate and unobserved here."
        )
    if verdict == "attribution_unavailable":
        return (
            "Diagnostics were observed but cannot be attributed to a workspace and revision interval, "
            "so they are not a language-diagnostic result for any interval."
        )
    if verdict == "stale_diagnostics":
        return (
            f"Diagnostics were observed at revision {diagnostics_revision}, which is not the interval end "
            f"{end_revision}, so this language-diagnostic check is stale."
        )
    if verdict == "freshness_unknown":
        return (
            "Diagnostics were observed without naming the revision they were observed at, so the freshness "
            "of this language-diagnostic check is unknown."
        )
    if verdict == "provider_unsupported":
        return "No language-diagnostic provider was available for this workspace, so no check was run."
    if verdict == "provider_failed":
        return "The language-diagnostic provider failed, so no diagnostics were observed."
    return "No language-diagnostic check was observed."


def _derivation_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    check_state = str(record["check_state"])
    end_revision = str(record["end_revision"])
    diagnostics_revision = str(record["diagnostics_revision"])
    freshness = _derive_freshness(check_state, end_revision, diagnostics_revision)
    attribution = _derive_attribution(
        check_state, str(record["workspace_id"]), str(record["baseline_revision"]), end_revision
    )
    introduced_count = int(record["introduced_count"])
    verdict = _derive_verdict(check_state, freshness, attribution, introduced_count)
    if record["freshness"] != freshness:
        errors.append(f"language_diagnostic_evidence freshness must be derived as {freshness!r}")
    if record["attribution"] != attribution:
        errors.append(f"language_diagnostic_evidence attribution must be derived as {attribution!r}")
    if record["verdict"] != verdict:
        errors.append(f"language_diagnostic_evidence verdict must be derived as {verdict!r}")
    expected_label = _summary_label(
        verdict,
        workspace_id=str(record["workspace_id"]),
        baseline_revision=str(record["baseline_revision"]),
        end_revision=end_revision,
        diagnostics_revision=diagnostics_revision,
        introduced_count=introduced_count,
    )
    if record["summary_label"] != expected_label:
        errors.append("language_diagnostic_evidence summary_label must be the derived label for its verdict")
    return errors


def _normalized_paths(values: Iterable[str]) -> list[str]:
    paths = sorted({_workspace_relative_path(value, field="changed_paths") for value in values})
    if len(paths) > MAX_CHANGED_PATHS:
        raise LanguageDiagnosticEvidenceError(
            f"language_diagnostic_evidence changed_paths must have at most {MAX_CHANGED_PATHS} entries"
        )
    return paths


def _normalized_diagnostics(values: Iterable[Mapping[str, Any]], *, field: str) -> list[dict[str, Any]]:
    normalized = [_normalized_diagnostic(value, field=field) for value in values]
    unique = {tuple(item[key] for key in LANGUAGE_DIAGNOSTIC_ITEM_KEYS): item for item in normalized}
    ordered = [unique[key] for key in sorted(unique)]
    if len(ordered) > MAX_DIAGNOSTICS:
        raise LanguageDiagnosticEvidenceError(
            f"language_diagnostic_evidence {field} must have at most {MAX_DIAGNOSTICS} entries"
        )
    return ordered


def _normalized_diagnostic(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LanguageDiagnosticEvidenceError(f"language_diagnostic_evidence {field} entries must be objects")
    extra = sorted(str(key) for key in value if str(key) not in LANGUAGE_DIAGNOSTIC_ITEM_KEYS)
    if extra:
        # `message` arrives here. It is refused rather than dropped so a caller
        # learns the record carries no source body, instead of silently
        # believing it sent one.
        raise LanguageDiagnosticEvidenceError(
            f"language_diagnostic_evidence {field} entries carry unsupported keys: {extra}; "
            f"a diagnostic is metadata only ({', '.join(LANGUAGE_DIAGNOSTIC_ITEM_KEYS)})"
        )
    severity = str(value.get("severity", "")).strip().lower()
    if severity not in LANGUAGE_DIAGNOSTIC_SEVERITIES:
        raise LanguageDiagnosticEvidenceError(
            f"language_diagnostic_evidence {field} severity is unsupported: {value.get('severity')!r}"
        )
    return {
        "severity": severity,
        "code": _optional_ref(str(value.get("code", "") or ""), field=f"language_diagnostic_evidence {field} code"),
        "path": _workspace_relative_path(value.get("path", ""), field=f"{field} path"),
        "line": _position(value.get("line", 0), field=f"{field} line"),
        "character": _position(value.get("character", 0), field=f"{field} character"),
        "source": _optional_ref(
            str(value.get("source", "") or ""), field=f"language_diagnostic_evidence {field} source"
        ),
    }


def _normalized_evidence_refs(values: Iterable[str]) -> list[str]:
    refs: list[str] = []
    for value in values:
        ref = _required_ref(value, field="language_diagnostic_evidence evidence_refs")
        if ref not in refs:
            refs.append(ref)
    if len(refs) > MAX_EVIDENCE_REFS:
        raise LanguageDiagnosticEvidenceError(
            f"language_diagnostic_evidence evidence_refs must have at most {MAX_EVIDENCE_REFS} entries"
        )
    return refs


def _workspace_relative_path(value: Any, *, field: str) -> str:
    """A workspace-relative posix path, or a refusal.

    Separators are normalized so a Windows caller and a POSIX caller recording
    the same file produce the same record, and an absolute or escaping path is
    refused because it names a machine rather than a position in the workspace
    the interval is attributed to.
    """
    if not isinstance(value, str) or not value.strip():
        raise LanguageDiagnosticEvidenceError(f"language_diagnostic_evidence {field} must be a nonblank path")
    text = value.strip().replace("\\", "/")
    if text.startswith("/") or _ABSOLUTE_WINDOWS_PATH.match(text):
        raise LanguageDiagnosticEvidenceError(
            f"language_diagnostic_evidence {field} must be workspace-relative, not absolute: {value!r}"
        )
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise LanguageDiagnosticEvidenceError(
            f"language_diagnostic_evidence {field} must stay inside the workspace: {value!r}"
        )
    return _required_ref("/".join(parts), field=f"language_diagnostic_evidence {field}")


def _position(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LanguageDiagnosticEvidenceError(
            f"language_diagnostic_evidence {field} must be an integer offset"
        )
    if value < 0 or value > MAX_POSITION:
        raise LanguageDiagnosticEvidenceError(
            f"language_diagnostic_evidence {field} must be between 0 and {MAX_POSITION}"
        )
    return value


def _required_ref(value: Any, *, field: str) -> str:
    try:
        text = require_opaque_metadata_ref(value, field=field)
    except ValueError as error:
        raise LanguageDiagnosticEvidenceError(str(error)) from error
    if len(text) > MAX_REFERENCE_CHARS:
        raise LanguageDiagnosticEvidenceError(f"{field} must be at most {MAX_REFERENCE_CHARS} characters")
    return text


def _optional_ref(value: Any, *, field: str) -> str:
    if value == "":
        return ""
    return _required_ref(value, field=field)


def _reference_errors(value: Any, *, field: str, required: bool) -> list[str]:
    if not isinstance(value, str):
        return [f"language_diagnostic_evidence {field} must be a string"]
    if not value:
        return [] if not required else [f"language_diagnostic_evidence {field} must not be empty"]
    try:
        _required_ref(value, field=f"language_diagnostic_evidence {field}")
    except LanguageDiagnosticEvidenceError as error:
        return [str(error)]
    return []


def _path_list_errors(values: Any) -> list[str]:
    if not isinstance(values, list):
        return ["language_diagnostic_evidence changed_paths must be a list"]
    errors: list[str] = []
    if len(values) > MAX_CHANGED_PATHS:
        errors.append(f"language_diagnostic_evidence changed_paths must have at most {MAX_CHANGED_PATHS} entries")
    for index, value in enumerate(values):
        try:
            normalized = _workspace_relative_path(value, field=f"changed_paths[{index}]")
        except LanguageDiagnosticEvidenceError as error:
            errors.append(str(error))
            continue
        if normalized != value:
            errors.append(f"language_diagnostic_evidence changed_paths[{index}] must be a normalized relative path")
    if values != sorted(str(value) for value in values if isinstance(value, str)) or len(set(values)) != len(values):
        errors.append("language_diagnostic_evidence changed_paths must be sorted and unique")
    return errors


def _diagnostic_list_errors(values: Any, *, field: str) -> list[str]:
    if not isinstance(values, list):
        return [f"language_diagnostic_evidence {field} must be a list"]
    errors: list[str] = []
    if len(values) > MAX_DIAGNOSTICS:
        errors.append(f"language_diagnostic_evidence {field} must have at most {MAX_DIAGNOSTICS} entries")
    keys: list[tuple[Any, ...]] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict) or set(value) != set(LANGUAGE_DIAGNOSTIC_ITEM_KEYS):
            errors.append(
                f"language_diagnostic_evidence {field}[{index}] must carry exactly "
                f"{list(LANGUAGE_DIAGNOSTIC_ITEM_KEYS)}"
            )
            continue
        try:
            normalized = _normalized_diagnostic(value, field=field)
        except LanguageDiagnosticEvidenceError as error:
            errors.append(str(error))
            continue
        if normalized != value:
            errors.append(f"language_diagnostic_evidence {field}[{index}] must be normalized")
        keys.append(tuple(normalized[key] for key in LANGUAGE_DIAGNOSTIC_ITEM_KEYS))
    # Only meaningful once every entry normalized: a list with a rejected entry
    # is short here, and reporting it as unsorted on top of the real fault would
    # bury the reason it was rejected.
    if len(keys) == len(values) and keys != sorted(set(keys)):
        errors.append(f"language_diagnostic_evidence {field} must be sorted and unique")
    return errors


def _evidence_ref_errors(values: Any) -> list[str]:
    if not isinstance(values, list):
        return ["language_diagnostic_evidence evidence_refs must be a list"]
    errors: list[str] = []
    if len(values) > MAX_EVIDENCE_REFS:
        errors.append(f"language_diagnostic_evidence evidence_refs must have at most {MAX_EVIDENCE_REFS} entries")
    if len(set(map(str, values))) != len(values):
        errors.append("language_diagnostic_evidence evidence_refs must be unique")
    for index, value in enumerate(values):
        errors.extend(_reference_errors(value, field=f"evidence_refs[{index}]", required=True))
    return errors


def _record_id(
    owner: str,
    provider: str,
    workspace_id: str,
    baseline_revision: str,
    end_revision: str,
    diagnostics_revision: str,
    config_digest: str,
    check_state: str,
    introduced: list[dict[str, Any]],
    resolved: list[dict[str, Any]],
) -> str:
    """Identity of *which* check, not of when it was reported.

    `observed_at` is excluded on purpose: the same observation recorded twice
    is one record, and a clock in the seed would make it two.
    """
    parts = [owner, provider, workspace_id, baseline_revision, end_revision, diagnostics_revision, config_digest, check_state]
    for label, items in (("introduced", introduced), ("resolved", resolved)):
        parts.append(label)
        parts.extend(
            ":".join(str(item[key]) for key in LANGUAGE_DIAGNOSTIC_ITEM_KEYS) for item in items
        )
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"langdiag-{digest}"
