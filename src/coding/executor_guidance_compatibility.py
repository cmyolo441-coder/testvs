"""Outbound guidance compatibility across supported coding owners.

`executor_guidance_compatibility/v1` answers one question: OMH already owns a
piece of guidance (the rendered executor prompt body, a workflow instruction, a
reporting rule) — will every supported coding owner receive what OMH meant?

This module is the outbound mirror of `omh.workflows.prompt_compatibility`
(`prompt_compatibility_audit/v1`), and the pair is meant to be read together.
They differ in direction and in what they are allowed to touch:

- **Inbound** (`prompt_compatibility`): reads prompt FILES someone wants to
  bring INTO OMH, and reports whether OMH can accept them — format, candidate
  command name, argument syntax, trust review, command-name collisions. Its
  input is untrusted third-party text on disk, so it hashes bytes, bounds IO,
  and refuses symlinks. Its question is "can we accept this file".
- **Outbound** (this module): reads guidance OMH ALREADY OWNS, in memory, and
  reports whether each supported coding owner can take it as written. There is
  no filesystem, no hashing, and no trust review, because the text is OMH's own.
  Its question is "does every owner receive what we meant".

A workflow can be perfectly clear in Hermes chat and still hand a selected
coding owner an instruction it cannot parse, cannot perform, silently reads as
something else, or cannot report evidence for. Those are four different
failures, so the audit keeps four separate axes and never collapses them:

- **syntax** — does the rendered guidance fit the owner's guidance form (every
  declared section present, in the shape that form carries).
- **capability** — does the owner support what the guidance asks for, read from
  that owner's `executor_capability_snapshot/v1`.
- **semantic equivalence** — does the owner receive the same instruction OMH
  meant, or does the guidance name another host's mechanics.
- **observation support** — can the owner report the lifecycle evidence the
  guidance requires, through a channel OMH can actually observe.

Boundaries:

- **Guidance text in, declared tokens out.** The guidance body is consumed by
  the leakage scanner and never returned. Only tokens from OMH's own declared
  `HOST_SPECIFIC_VOCABULARY` reach a caller, so an audit payload cannot carry a
  prompt body, a secret pasted into guidance, or arbitrary operator text.
- **Declared vocabulary, not heuristics.** Host-specific leakage is exact
  token-set membership against `HOST_SPECIFIC_VOCABULARY`. There is no fuzzy
  match, no substring scan, and no free-text guessing, so the same guidance
  always produces the same findings.
- **Unknown is a state, never availability.** `unknown` sits below `available`
  in `_STATE_RANK` and an owner row's overall state is the worst of its four
  axes, so an unknown capability can never be rendered as observed support.
  `reads_as_observed_availability` is the single predicate that decides which
  state means "observed"; exactly one state does.
- **Every supported owner gets a row.** A record missing an owner is a
  validation error, not an omission, and every row carries a portable fallback
  naming the form the owner can take when it cannot take the guidance as
  written. The fallback is the existing `coding_prompt_handoff/v1` generic
  portable prompt, not a new mechanism.
- **Executor-neutral shape.** Owner-specific content lives in named fields
  (`owner`, `label`, `guidance_form`, `observation_channel`, axis `detail`), so
  no owner — Codex least of all — is privileged by the record's structure.
- **Deterministic.** No wall clock, no IO, no network, no LLM. Two audits of
  the same inputs are byte-identical, which is what makes the payload safe to
  compare in a test or a gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Final, Iterable, Mapping, TypeAlias

from ..system.metadata_safety import require_opaque_metadata_ref
from .coding_contracts import EXECUTOR_PROMPTING_REQUIRED_SECTIONS, PROMPT_HANDOFF_SCHEMA_VERSION
from .executor_capability_snapshots import KNOWN_CAPABILITY_NAMES, LOCAL_WORKFLOW_CAPABILITY_NAME
from .executors import (
    EXECUTOR_PROFILES,
    HERMES_CODING_TEAM_STATUS_LADDER,
    executor_label,
    prompt_invocation_for_profile,
    public_executor_options,
    runtime_templates_for_profile,
)


EXECUTOR_GUIDANCE_COMPATIBILITY_SCHEMA_VERSION: Final[str] = "executor_guidance_compatibility/v1"

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

# The four axes, kept separate on purpose. Collapsing them loses the difference
# between "the owner cannot parse this" and "the owner cannot prove it ran".
GUIDANCE_COMPATIBILITY_AXES: Final[tuple[str, ...]] = (
    "syntax",
    "capability",
    "semantic_equivalence",
    "observation_support",
)

# Reuses the capability-status vocabulary of `omh_parity_matrix/v1`
# (`available` / `partial`) rather than inventing a fifth status language, plus
# the two negative states this audit needs: a declared inability (`unsupported`)
# and an absence of evidence (`unknown`).
GUIDANCE_COMPATIBILITY_STATES: Final[tuple[str, ...]] = (
    "available",
    "partial",
    "unsupported",
    "unknown",
)

# The whole of acceptance criterion 3 lives in this set. Exactly one state means
# "the owner is observed to support this"; `unknown` is not it, and neither is
# any other state. Every renderer asks this predicate instead of comparing
# strings itself.
OBSERVED_AVAILABILITY_STATES: Final[frozenset[str]] = frozenset({"available"})

# Worst-first. An owner's overall state is the minimum of its four axes under
# this rank, so one unknown axis pulls the row off `available`. `unsupported`
# ranks below `unknown` because a declared inability is actionable while an
# absence of evidence is not; both are non-available, which is what matters.
_STATE_RANK: Final[dict[str, int]] = {"unsupported": 0, "unknown": 1, "partial": 2, "available": 3}

SUPPORTED_GUIDANCE_OWNERS: Final[tuple[str, ...]] = tuple(sorted(EXECUTOR_PROFILES))

# Guidance forms and observation channels are derived from the work-owner mode
# each owner already declares in `public_executor_options()`. Deriving them
# keeps this module from growing a second, drifting owner table.
_GUIDANCE_FORMS: Final[dict[str, str]] = {
    "external_executor": "dispatch_prompt",
    "prompt_only_handoff": "pasted_prompt",
    "runtime_handoff": "skill_invocation",
}
_OBSERVATION_CHANNELS: Final[dict[str, str]] = {
    "external_executor": "observed_run_record",
    "prompt_only_handoff": "operator_reported",
    "runtime_handoff": "runtime_observation_ledger",
}

# The lifecycle events an external executor's run-backed handoff can produce as
# observed evidence. These are the three named in that profile's own
# `recommended_for` text in `public_executor_options()` ("observed
# dispatch/result/review evidence"); runtime owners derive theirs from the
# `observed_event` values their templates declare, and prompt-only owners
# observe nothing because OMH never dispatches them.
_EXTERNAL_EXECUTOR_OBSERVED_EVENTS: Final[frozenset[str]] = frozenset(
    {"worker_dispatch", "worker_result", "review"}
)

# `executor_capability_snapshot/v1` statuses projected onto the audit
# vocabulary. `prepared` is `partial` because a prepared capability is OMH's
# intent, not the host's confirmation. Anything absent or unrecognised is
# `unknown` — never `available`.
_CAPABILITY_STATUS_STATES: Final[dict[str, str]] = {
    "host_observed": "available",
    "prepared": "partial",
    "unavailable": "unsupported",
    "unknown": "unknown",
}

MAX_GUIDANCE_BYTES: Final[int] = 131_072
MAX_REQUIRED_SECTIONS: Final[int] = 64
MAX_REQUIRED_CAPABILITIES: Final[int] = 32
MAX_REQUIRED_OBSERVATIONS: Final[int] = 32
_MAX_SECTION_LENGTH: Final[int] = 120

EXECUTOR_GUIDANCE_COMPATIBILITY_CLAIM_BOUNDARY: Final[str] = (
    "The audit reads OMH-owned guidance text in memory and returns bounded per-owner compatibility "
    "metadata. It is not owner installation, host capability probing, guidance delivery, dispatch, "
    "execution, verification, review, CI, or merge evidence, and a compatibility state is never proof "
    "that the named owner received, parsed, or ran the guidance."
)

_ROOT_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema_version", "guidance_ref", "axes", "leakage", "owners", "summary", "not_observed", "claim_boundary"}
)
_OWNER_FIELDS: Final[frozenset[str]] = frozenset(
    {"owner", "label", "guidance_form", "observation_channel", "state", "axes", "fallback"}
)
_AXIS_FIELDS: Final[frozenset[str]] = frozenset({"axis", "state", "checked", "unmet", "detail"})
_LEAKAGE_FIELDS: Final[frozenset[str]] = frozenset({"status", "finding_count", "findings"})
_FINDING_FIELDS: Final[frozenset[str]] = frozenset({"token", "host", "mechanic"})
_FALLBACK_FIELDS: Final[frozenset[str]] = frozenset(
    {"required", "kind", "schema_version", "portable_profile", "portable_form", "reason"}
)
_SUMMARY_FIELDS: Final[frozenset[str]] = frozenset(
    {"owner_count", "state_counts", "fallback_required_count", "leakage_finding_count"}
)
_NOT_OBSERVED_FIELDS: Final[frozenset[str]] = frozenset(
    {"owner_capability_probe", "guidance_delivery", "owner_dispatch", "owner_execution", "evidence_report"}
)

# Keys that must never appear anywhere in the payload. The wall-clock names are
# there because this record is compared byte-for-byte; the raw-material names
# are there because the guidance body is consumed, never returned.
_FORBIDDEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "audited_at",
        "generated_at",
        "guidance_body",
        "guidance_text",
        "observed_at",
        "prompt",
        "raw",
        "raw_guidance",
        "raw_prompt",
        "recorded_at",
        "timestamp",
        "transcript",
    }
)

_FALLBACK_KIND: Final[str] = "portable_prompt_handoff"
_PORTABLE_FALLBACK_PROFILE: Final[str] = "generic"

# Token grammar. A guidance token keeps the punctuation that makes a host
# mechanic recognisable (`.claude/settings.json`, `CODEX_HOME`) and drops
# everything else, so matching is exact set membership rather than a substring
# scan over free text.
_TOKEN_SPLIT: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9._/:-]+")
_TOKEN_EDGE: Final[str] = "._/:-"


class ExecutorGuidanceCompatibilityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class HostVocabularyEntry:
    """One host-specific mechanic OMH guidance must not name by accident."""

    token: str
    host: str
    mechanic: str


# The declared leakage vocabulary. `host` is the host the mechanic belongs to:
# when it is also a supported owner, that owner reads the token natively and
# every other owner does not; when it is not (Cursor, Copilot, Windsurf, Gemini
# CLI), the token is foreign to every supported owner. Adding an entry is the
# only way to make the audit detect a new mechanic — that is the point.
HOST_SPECIFIC_VOCABULARY: Final[tuple[HostVocabularyEntry, ...]] = (
    HostVocabularyEntry("TodoWrite", "claude-code", "Claude Code built-in todo tool"),
    HostVocabularyEntry("ExitPlanMode", "claude-code", "Claude Code plan-mode tool"),
    HostVocabularyEntry("subagent_type", "claude-code", "Claude Code Task tool parameter"),
    HostVocabularyEntry("CLAUDE.md", "claude-code", "Claude Code project instruction file"),
    HostVocabularyEntry(".claude/settings.json", "claude-code", "Claude Code settings file"),
    HostVocabularyEntry(".claude/skills", "claude-code", "Claude Code skill directory"),
    HostVocabularyEntry("apply_patch", "codex", "Codex patch envelope"),
    HostVocabularyEntry("CODEX_HOME", "codex", "Codex home environment variable"),
    HostVocabularyEntry(".codex/config.toml", "codex", "Codex configuration file"),
    HostVocabularyEntry(".codex/prompts", "codex", "Codex custom prompt directory"),
    HostVocabularyEntry(".hermes/plugins", "hermes", "Hermes plugin directory"),
    HostVocabularyEntry(".cursorrules", "cursor", "Cursor legacy rule file"),
    HostVocabularyEntry(".cursor/rules", "cursor", "Cursor project rule directory"),
    HostVocabularyEntry(".github/copilot-instructions.md", "github-copilot", "GitHub Copilot instruction file"),
    HostVocabularyEntry(".windsurfrules", "windsurf", "Windsurf rule file"),
    HostVocabularyEntry("GEMINI.md", "gemini-cli", "Gemini CLI project instruction file"),
)


@dataclass(frozen=True, slots=True)
class _OwnerGuidanceProfile:
    owner: str
    label: str
    guidance_form: str
    observation_channel: str
    observable_events: frozenset[str]


def build_executor_guidance_compatibility(
    *,
    guidance_ref: str,
    guidance_text: str,
    required_sections: Iterable[str] = EXECUTOR_PROMPTING_REQUIRED_SECTIONS,
    required_capabilities: Iterable[str] = (),
    required_observations: Iterable[str] = (),
    capability_snapshots: Mapping[str, Mapping[str, JsonValue]] | None = None,
) -> JsonObject:
    """Audit one piece of OMH-owned guidance against every supported coding owner."""
    reference = _validated_guidance_ref(guidance_ref)
    text = _validated_guidance_text(guidance_text)
    sections = _validated_sections(required_sections)
    capabilities = _validated_capabilities(required_capabilities)
    observations = _validated_observations(required_observations)
    snapshots = _validated_snapshots(capability_snapshots)

    findings = _leakage_findings(text)
    rows = [
        _owner_row(
            profile,
            guidance_text=text,
            required_sections=sections,
            required_capabilities=capabilities,
            required_observations=observations,
            snapshot=snapshots.get(profile.owner),
            findings=findings,
        )
        for profile in _owner_profiles()
    ]
    payload: JsonObject = {
        "schema_version": EXECUTOR_GUIDANCE_COMPATIBILITY_SCHEMA_VERSION,
        "guidance_ref": reference,
        "axes": list(GUIDANCE_COMPATIBILITY_AXES),
        "leakage": {
            "status": "detected" if findings else "clean",
            "finding_count": len(findings),
            "findings": [
                {"token": entry.token, "host": entry.host, "mechanic": entry.mechanic} for entry in findings
            ],
        },
        "owners": list(rows),
        "summary": _summary(rows, leakage_finding_count=len(findings)),
        "not_observed": {name: {"status": "not_observed"} for name in sorted(_NOT_OBSERVED_FIELDS)},
        "claim_boundary": EXECUTOR_GUIDANCE_COMPATIBILITY_CLAIM_BOUNDARY,
    }
    errors = validate_executor_guidance_compatibility(payload)
    if errors:
        raise ExecutorGuidanceCompatibilityError("; ".join(errors))
    return payload


def validate_executor_guidance_compatibility(payload: Mapping[str, JsonValue]) -> list[str]:
    """Return every contract violation in an audit payload, or an empty list."""
    errors = _forbidden_key_errors(payload)
    errors.extend(_field_errors("audit", payload, _ROOT_FIELDS))
    if payload.get("schema_version") != EXECUTOR_GUIDANCE_COMPATIBILITY_SCHEMA_VERSION:
        errors.append(f"schema_version must be {EXECUTOR_GUIDANCE_COMPATIBILITY_SCHEMA_VERSION}")
    try:
        require_opaque_metadata_ref(payload.get("guidance_ref"), field="guidance_ref")
    except ValueError as error:
        errors.append(str(error))
    if payload.get("axes") != list(GUIDANCE_COMPATIBILITY_AXES):
        errors.append(f"axes must be exactly {', '.join(GUIDANCE_COMPATIBILITY_AXES)}")
    if payload.get("claim_boundary") != EXECUTOR_GUIDANCE_COMPATIBILITY_CLAIM_BOUNDARY:
        errors.append("claim_boundary must be the declared guidance compatibility boundary")
    errors.extend(_not_observed_errors(payload.get("not_observed")))
    leakage_count = _leakage_errors(payload.get("leakage"), errors)
    rows = payload.get("owners")
    if not isinstance(rows, list):
        errors.append("owners must be a list of owner rows")
        return errors
    errors.extend(_owner_coverage_errors(rows))
    for row in rows:
        if isinstance(row, Mapping):
            errors.extend(_owner_row_errors(row))
    errors.extend(_summary_errors(payload.get("summary"), rows, leakage_count))
    return errors


def reads_as_observed_availability(state: str) -> bool:
    """Whether a compatibility state may be rendered as observed owner support.

    The only state that may is `available`. `unknown` must reach a reader as an
    open question, so this predicate — not a string comparison at the call site
    — decides.
    """
    return state in OBSERVED_AVAILABILITY_STATES


def guidance_leakage_findings(guidance_text: str) -> tuple[HostVocabularyEntry, ...]:
    """Return the declared host mechanics named by this guidance, if any."""
    return _leakage_findings(_validated_guidance_text(guidance_text))


# --- Owner projection -----------------------------------------------------


@lru_cache(maxsize=1)
def _owner_profiles() -> tuple[_OwnerGuidanceProfile, ...]:
    profiles = [
        _OwnerGuidanceProfile(
            owner=str(option["profile"]),
            label=executor_label(str(option["profile"])),
            guidance_form=_GUIDANCE_FORMS[str(option["work_owner_mode"])],
            observation_channel=_OBSERVATION_CHANNELS[str(option["work_owner_mode"])],
            observable_events=_observable_events(
                str(option["profile"]), str(option["work_owner_mode"])
            ),
        )
        for option in public_executor_options()
    ]
    return tuple(sorted(profiles, key=lambda profile: profile.owner))


def _observable_events(owner: str, work_owner_mode: str) -> frozenset[str]:
    if work_owner_mode == "runtime_handoff":
        return frozenset(
            str(template["observed_event"])
            for template in runtime_templates_for_profile(owner)
            if template.get("observed_event")
        )
    if work_owner_mode == "external_executor":
        return _EXTERNAL_EXECUTOR_OBSERVED_EVENTS
    return frozenset()


def _owner_row(
    profile: _OwnerGuidanceProfile,
    *,
    guidance_text: str,
    required_sections: tuple[str, ...],
    required_capabilities: tuple[str, ...],
    required_observations: tuple[str, ...],
    snapshot: Mapping[str, JsonValue] | None,
    findings: tuple[HostVocabularyEntry, ...],
) -> JsonObject:
    axes = {
        "syntax": _syntax_axis(profile, guidance_text, required_sections),
        "capability": _capability_axis(profile, required_capabilities, snapshot),
        "semantic_equivalence": _semantic_axis(profile, findings),
        "observation_support": _observation_axis(profile, required_observations),
    }
    state = _worst_state(tuple(str(axis["state"]) for axis in axes.values()))
    unmet_axes = sorted(name for name, axis in axes.items() if axis["state"] != "available")
    return {
        "owner": profile.owner,
        "label": profile.label,
        "guidance_form": profile.guidance_form,
        "observation_channel": profile.observation_channel,
        "state": state,
        "axes": dict(axes),
        "fallback": _fallback(state, unmet_axes),
    }


def _syntax_axis(
    profile: _OwnerGuidanceProfile, guidance_text: str, required_sections: tuple[str, ...]
) -> JsonObject:
    headings = _section_headings(guidance_text)
    missing = tuple(section for section in required_sections if section not in headings)
    if not required_sections:
        state = "unknown"
        detail = (
            f"no guidance section is declared, so the {profile.guidance_form} form cannot be checked"
        )
    elif missing:
        state = "unsupported"
        detail = f"the {profile.guidance_form} form cannot carry guidance with a missing section"
    else:
        state = "available"
        detail = f"every declared section is present in the {profile.guidance_form} form"
    return _axis("syntax", state, required_sections, missing, detail)


def _capability_axis(
    profile: _OwnerGuidanceProfile,
    required_capabilities: tuple[str, ...],
    snapshot: Mapping[str, JsonValue] | None,
) -> JsonObject:
    states = {name: _capability_state(snapshot, name) for name in required_capabilities}
    unmet = tuple(name for name in required_capabilities if states[name] != "available")
    if not required_capabilities:
        state = "available"
        detail = f"the guidance asks {profile.owner} for no declared capability"
    else:
        state = _worst_state(tuple(states.values()))
        source = "an owner capability snapshot" if snapshot is not None else "no owner capability snapshot"
        detail = f"capability states for {profile.owner} resolved from {source}"
    return _axis("capability", state, required_capabilities, unmet, detail)


def _capability_state(snapshot: Mapping[str, JsonValue] | None, name: str) -> str:
    if snapshot is None:
        return "unknown"
    capabilities = snapshot.get("capabilities")
    if not isinstance(capabilities, Mapping):
        return "unknown"
    capability = capabilities.get(name)
    if not isinstance(capability, Mapping):
        return "unknown"
    return _CAPABILITY_STATUS_STATES.get(str(capability.get("status", "")), "unknown")


def _semantic_axis(profile: _OwnerGuidanceProfile, findings: tuple[HostVocabularyEntry, ...]) -> JsonObject:
    detected = tuple(entry.token for entry in findings)
    foreign = tuple(entry.token for entry in findings if entry.host != profile.owner)
    if not findings:
        state = "available"
        detail = f"no declared host mechanic is named, so {profile.owner} receives the intended instruction"
    elif foreign:
        state = "unsupported"
        detail = f"{profile.owner} would receive host mechanics it does not have"
    else:
        state = "partial"
        detail = f"the guidance names {profile.owner} host mechanics and is no longer executor-neutral"
    return _axis("semantic_equivalence", state, detected, foreign, detail)


def _observation_axis(profile: _OwnerGuidanceProfile, required_observations: tuple[str, ...]) -> JsonObject:
    unmet = tuple(event for event in required_observations if event not in profile.observable_events)
    if not required_observations:
        state = "available"
        detail = f"the guidance requires no lifecycle evidence from {profile.observation_channel}"
    elif not unmet:
        state = "available"
        detail = f"{profile.observation_channel} carries every required lifecycle event"
    elif len(unmet) < len(required_observations):
        state = "partial"
        detail = f"{profile.observation_channel} carries only some of the required lifecycle events"
    else:
        state = "unknown"
        detail = f"{profile.observation_channel} produces no OMH-observed evidence for the required events"
    return _axis("observation_support", state, required_observations, unmet, detail)


def _axis(name: str, state: str, checked: Iterable[str], unmet: Iterable[str], detail: str) -> JsonObject:
    return {
        "axis": name,
        "state": state,
        "checked": sorted(checked),
        "unmet": sorted(unmet),
        "detail": detail,
    }


def _fallback(state: str, unmet_axes: list[str]) -> JsonObject:
    required = not reads_as_observed_availability(state)
    invocation = prompt_invocation_for_profile(_PORTABLE_FALLBACK_PROFILE)
    if required:
        reason = (
            "the owner cannot take the guidance as written on: "
            f"{', '.join(unmet_axes)}; hand it over as the portable prompt instead"
        )
    else:
        reason = "the owner takes the guidance as written; the portable prompt stays available anyway"
    return {
        "required": required,
        "kind": _FALLBACK_KIND,
        "schema_version": PROMPT_HANDOFF_SCHEMA_VERSION,
        "portable_profile": _PORTABLE_FALLBACK_PROFILE,
        "portable_form": invocation["dispatch_text_template"],
        "reason": reason,
    }


def _summary(rows: list[JsonObject], *, leakage_finding_count: int) -> JsonObject:
    counts = {state: 0 for state in GUIDANCE_COMPATIBILITY_STATES}
    fallback_required = 0
    for row in rows:
        counts[str(row["state"])] += 1
        fallback = row["fallback"]
        if isinstance(fallback, Mapping) and fallback.get("required"):
            fallback_required += 1
    return {
        "owner_count": len(rows),
        "state_counts": dict(sorted(counts.items())),
        "fallback_required_count": fallback_required,
        "leakage_finding_count": leakage_finding_count,
    }


def _worst_state(states: tuple[str, ...]) -> str:
    if not states:
        return "unknown"
    return min(states, key=lambda state: _STATE_RANK.get(state, 0))


# --- Leakage detection ----------------------------------------------------


def _leakage_findings(guidance_text: str) -> tuple[HostVocabularyEntry, ...]:
    tokens = _guidance_tokens(guidance_text)
    matched = [entry for entry, match in _normalized_vocabulary() if match in tokens]
    return tuple(sorted(matched, key=lambda entry: (entry.host, entry.token)))


@lru_cache(maxsize=1)
def _normalized_vocabulary() -> tuple[tuple[HostVocabularyEntry, str], ...]:
    normalized: list[tuple[HostVocabularyEntry, str]] = []
    for entry in HOST_SPECIFIC_VOCABULARY:
        match = _normalized_vocabulary_token(entry.token)
        if not match:
            raise ExecutorGuidanceCompatibilityError(
                f"host vocabulary token must normalize to one token: {entry.host}"
            )
        normalized.append((entry, match))
    return tuple(normalized)


def _normalized_vocabulary_token(token: str) -> str:
    candidate = token.casefold().strip(_TOKEN_EDGE)
    if not candidate or _TOKEN_SPLIT.search(candidate):
        return ""
    return candidate


def _guidance_tokens(text: str) -> frozenset[str]:
    """Tokenize guidance into an exact-match set, with path prefixes included.

    Path prefixes matter: guidance naming `~/.claude/skills/reviewer/SKILL.md`
    is naming `.claude/skills`, and a bare set-membership test would miss it.
    Emitting each `/`-prefix keeps detection exact while still catching the
    deeper path — no substring scan is involved.
    """
    tokens: set[str] = set()
    for raw in _TOKEN_SPLIT.split(text.casefold()):
        token = raw.strip(_TOKEN_EDGE)
        if not token:
            continue
        tokens.add(token)
        parts = token.split("/")
        for index in range(1, len(parts)):
            prefix = "/".join(parts[:index]).strip(_TOKEN_EDGE)
            if prefix:
                tokens.add(prefix)
    return frozenset(tokens)


def _section_headings(guidance_text: str) -> frozenset[str]:
    """Return the heading lines of the rendered guidance.

    A heading is a whole line, never a fragment of one: `render_executor_prompt_sections`
    emits each section title on its own line above its bullets, and the trailing
    task section as `Task:`. Matching whole lines keeps a section named inside a
    bullet from counting as that section being present.
    """
    headings: set[str] = set()
    for line in guidance_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("-", "*", "#", ">")):
            continue
        headings.add(stripped)
        if stripped.endswith(":"):
            headings.add(stripped[:-1].strip())
    return frozenset(headings)


# --- Input validation -----------------------------------------------------


def _validated_guidance_ref(guidance_ref: str) -> str:
    try:
        return require_opaque_metadata_ref(guidance_ref, field="guidance_ref")
    except ValueError as error:
        raise ExecutorGuidanceCompatibilityError(str(error)) from error


def _validated_guidance_text(guidance_text: str) -> str:
    if not isinstance(guidance_text, str) or not guidance_text.strip():
        raise ExecutorGuidanceCompatibilityError("guidance text must be a nonempty string")
    if len(guidance_text.encode("utf-8")) > MAX_GUIDANCE_BYTES:
        raise ExecutorGuidanceCompatibilityError(f"guidance text exceeds {MAX_GUIDANCE_BYTES} bytes")
    return guidance_text


def _validated_sections(required_sections: Iterable[str]) -> tuple[str, ...]:
    sections = tuple(str(section).strip() for section in required_sections)
    if len(sections) > MAX_REQUIRED_SECTIONS:
        raise ExecutorGuidanceCompatibilityError(
            f"at most {MAX_REQUIRED_SECTIONS} required sections may be declared"
        )
    for section in sections:
        if not section or len(section) > _MAX_SECTION_LENGTH:
            raise ExecutorGuidanceCompatibilityError("each required section must be a nonempty bounded string")
    if len(set(sections)) != len(sections):
        raise ExecutorGuidanceCompatibilityError("required sections must be unique")
    return sections


def _validated_capabilities(required_capabilities: Iterable[str]) -> tuple[str, ...]:
    capabilities = tuple(str(name).strip() for name in required_capabilities)
    if len(capabilities) > MAX_REQUIRED_CAPABILITIES:
        raise ExecutorGuidanceCompatibilityError(
            f"at most {MAX_REQUIRED_CAPABILITIES} required capabilities may be declared"
        )
    allowed = KNOWN_CAPABILITY_NAMES | {LOCAL_WORKFLOW_CAPABILITY_NAME}
    for name in capabilities:
        if name not in allowed:
            raise ExecutorGuidanceCompatibilityError(f"unsupported required capability name: {name}")
    if len(set(capabilities)) != len(capabilities):
        raise ExecutorGuidanceCompatibilityError("required capabilities must be unique")
    return capabilities


def _validated_observations(required_observations: Iterable[str]) -> tuple[str, ...]:
    observations = tuple(str(name).strip() for name in required_observations)
    if len(observations) > MAX_REQUIRED_OBSERVATIONS:
        raise ExecutorGuidanceCompatibilityError(
            f"at most {MAX_REQUIRED_OBSERVATIONS} required observations may be declared"
        )
    for name in observations:
        if name not in HERMES_CODING_TEAM_STATUS_LADDER:
            raise ExecutorGuidanceCompatibilityError(f"unsupported required observation: {name}")
    if len(set(observations)) != len(observations):
        raise ExecutorGuidanceCompatibilityError("required observations must be unique")
    return observations


def _validated_snapshots(
    capability_snapshots: Mapping[str, Mapping[str, JsonValue]] | None,
) -> dict[str, Mapping[str, JsonValue]]:
    if capability_snapshots is None:
        return {}
    snapshots: dict[str, Mapping[str, JsonValue]] = {}
    for owner, snapshot in capability_snapshots.items():
        if owner not in SUPPORTED_GUIDANCE_OWNERS:
            raise ExecutorGuidanceCompatibilityError(f"unsupported capability snapshot owner: {owner}")
        if not isinstance(snapshot, Mapping):
            raise ExecutorGuidanceCompatibilityError(f"capability snapshot for {owner} must be a mapping")
        snapshots[owner] = snapshot
    return snapshots


# --- Payload validation ---------------------------------------------------


def _field_errors(path: str, value: Mapping[str, JsonValue], allowed: frozenset[str]) -> list[str]:
    keys = {str(key) for key in value}
    errors: list[str] = []
    missing = allowed - keys
    if missing:
        errors.append(f"{path} is missing required fields: {', '.join(sorted(missing))}")
    unexpected = keys - allowed
    if unexpected:
        errors.append(f"{path} contains unsupported fields: {', '.join(sorted(unexpected))}")
    return errors


def _forbidden_key_errors(value: JsonValue | Mapping[str, JsonValue], path: str = "audit") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text.casefold() in _FORBIDDEN_KEYS:
                errors.append(f"{path}.{key_text} is forbidden metadata")
            errors.extend(_forbidden_key_errors(item, f"{path}.{key_text}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_forbidden_key_errors(item, f"{path}[{index}]"))
    return errors


def _not_observed_errors(not_observed: JsonValue) -> list[str]:
    if not isinstance(not_observed, Mapping):
        return ["not_observed must be a mapping"]
    errors = _field_errors("not_observed", not_observed, _NOT_OBSERVED_FIELDS)
    for name, entry in not_observed.items():
        if not isinstance(entry, Mapping) or entry != {"status": "not_observed"}:
            errors.append(f"not_observed.{name} must be exactly a not_observed status")
    return errors


def _leakage_errors(leakage: JsonValue, errors: list[str]) -> int:
    if not isinstance(leakage, Mapping):
        errors.append("leakage must be a mapping")
        return 0
    errors.extend(_field_errors("leakage", leakage, _LEAKAGE_FIELDS))
    findings = leakage.get("findings")
    if not isinstance(findings, list):
        errors.append("leakage.findings must be a list")
        return 0
    declared = {entry.token for entry in HOST_SPECIFIC_VOCABULARY}
    for index, finding in enumerate(findings):
        if not isinstance(finding, Mapping):
            errors.append(f"leakage.findings[{index}] must be a mapping")
            continue
        errors.extend(_field_errors(f"leakage.findings[{index}]", finding, _FINDING_FIELDS))
        if finding.get("token") not in declared:
            errors.append(f"leakage.findings[{index}].token must come from the declared host vocabulary")
    count = leakage.get("finding_count")
    if count != len(findings):
        errors.append("leakage.finding_count must equal the number of findings")
    expected_status = "detected" if findings else "clean"
    if leakage.get("status") != expected_status:
        errors.append(f"leakage.status must be {expected_status}")
    return len(findings)


def _owner_coverage_errors(rows: list[JsonValue]) -> list[str]:
    owners = [str(row.get("owner", "")) for row in rows if isinstance(row, Mapping)]
    errors: list[str] = []
    if len(owners) != len(set(owners)):
        errors.append("owners must not repeat a coding owner")
    missing = sorted(set(SUPPORTED_GUIDANCE_OWNERS) - set(owners))
    if missing:
        errors.append(f"owners must cover every supported coding owner; missing: {', '.join(missing)}")
    unexpected = sorted(set(owners) - set(SUPPORTED_GUIDANCE_OWNERS))
    if unexpected:
        errors.append(f"owners contains unsupported coding owners: {', '.join(unexpected)}")
    if owners != sorted(owners):
        errors.append("owners must be sorted by coding owner")
    return errors


def _owner_row_errors(row: Mapping[str, JsonValue]) -> list[str]:
    owner = str(row.get("owner", "?"))
    errors = _field_errors(f"owners.{owner}", row, _OWNER_FIELDS)
    state = row.get("state")
    if state not in GUIDANCE_COMPATIBILITY_STATES:
        errors.append(f"owners.{owner}.state must be one of {', '.join(GUIDANCE_COMPATIBILITY_STATES)}")
    axes = row.get("axes")
    if not isinstance(axes, Mapping):
        errors.append(f"owners.{owner}.axes must be a mapping of the four compatibility axes")
        return errors
    errors.extend(_field_errors(f"owners.{owner}.axes", axes, frozenset(GUIDANCE_COMPATIBILITY_AXES)))
    axis_states: list[str] = []
    for name in GUIDANCE_COMPATIBILITY_AXES:
        axis = axes.get(name)
        if not isinstance(axis, Mapping):
            errors.append(f"owners.{owner}.axes.{name} must be a mapping")
            continue
        errors.extend(_field_errors(f"owners.{owner}.axes.{name}", axis, _AXIS_FIELDS))
        if axis.get("axis") != name:
            errors.append(f"owners.{owner}.axes.{name}.axis must name its own axis")
        axis_state = axis.get("state")
        if axis_state not in GUIDANCE_COMPATIBILITY_STATES:
            errors.append(f"owners.{owner}.axes.{name}.state must be a declared compatibility state")
            continue
        axis_states.append(str(axis_state))
        for field in ("checked", "unmet"):
            values = axis.get(field)
            if not isinstance(values, list) or values != sorted(str(item) for item in values):
                errors.append(f"owners.{owner}.axes.{name}.{field} must be a sorted list of strings")
        detail = axis.get("detail")
        if not isinstance(detail, str) or not detail.strip():
            errors.append(f"owners.{owner}.axes.{name}.detail must be a nonempty string")
    if len(axis_states) == len(GUIDANCE_COMPATIBILITY_AXES) and state in GUIDANCE_COMPATIBILITY_STATES:
        if any(not reads_as_observed_availability(axis_state) for axis_state in axis_states) and (
            reads_as_observed_availability(str(state))
        ):
            errors.append(
                f"owners.{owner}.state must not read as observed availability while an axis does not"
            )
        elif state != _worst_state(tuple(axis_states)):
            errors.append(f"owners.{owner}.state must be the worst of its four axis states")
    errors.extend(_fallback_errors(owner, row.get("fallback"), state))
    return errors


def _fallback_errors(owner: str, fallback: JsonValue, state: JsonValue) -> list[str]:
    if not isinstance(fallback, Mapping):
        return [f"owners.{owner}.fallback must name the portable form the owner can take"]
    errors = _field_errors(f"owners.{owner}.fallback", fallback, _FALLBACK_FIELDS)
    if fallback.get("kind") != _FALLBACK_KIND:
        errors.append(f"owners.{owner}.fallback.kind must be {_FALLBACK_KIND}")
    if fallback.get("schema_version") != PROMPT_HANDOFF_SCHEMA_VERSION:
        errors.append(f"owners.{owner}.fallback.schema_version must be {PROMPT_HANDOFF_SCHEMA_VERSION}")
    if fallback.get("portable_profile") != _PORTABLE_FALLBACK_PROFILE:
        errors.append(f"owners.{owner}.fallback.portable_profile must be {_PORTABLE_FALLBACK_PROFILE}")
    for field in ("portable_form", "reason"):
        value = fallback.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"owners.{owner}.fallback.{field} must be a nonempty string")
    required = fallback.get("required")
    if not isinstance(required, bool):
        errors.append(f"owners.{owner}.fallback.required must be a boolean")
    elif isinstance(state, str) and state in GUIDANCE_COMPATIBILITY_STATES:
        if required is reads_as_observed_availability(state):
            errors.append(
                f"owners.{owner}.fallback.required must be true exactly when the state is not available"
            )
    return errors


def _summary_errors(summary: JsonValue, rows: list[JsonValue], leakage_finding_count: int) -> list[str]:
    if not isinstance(summary, Mapping):
        return ["summary must be a mapping"]
    errors = _field_errors("summary", summary, _SUMMARY_FIELDS)
    if summary.get("owner_count") != len(rows):
        errors.append("summary.owner_count must equal the number of owner rows")
    if summary.get("leakage_finding_count") != leakage_finding_count:
        errors.append("summary.leakage_finding_count must equal the leakage finding count")
    expected_counts = {state: 0 for state in GUIDANCE_COMPATIBILITY_STATES}
    expected_fallbacks = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        state = str(row.get("state", ""))
        if state in expected_counts:
            expected_counts[state] += 1
        fallback = row.get("fallback")
        if isinstance(fallback, Mapping) and fallback.get("required") is True:
            expected_fallbacks += 1
    if summary.get("state_counts") != dict(sorted(expected_counts.items())):
        errors.append("summary.state_counts must count every declared state across the owner rows")
    if summary.get("fallback_required_count") != expected_fallbacks:
        errors.append("summary.fallback_required_count must equal the rows requiring a fallback")
    return errors


__all__ = [
    "EXECUTOR_GUIDANCE_COMPATIBILITY_CLAIM_BOUNDARY",
    "EXECUTOR_GUIDANCE_COMPATIBILITY_SCHEMA_VERSION",
    "GUIDANCE_COMPATIBILITY_AXES",
    "GUIDANCE_COMPATIBILITY_STATES",
    "HOST_SPECIFIC_VOCABULARY",
    "MAX_GUIDANCE_BYTES",
    "MAX_REQUIRED_CAPABILITIES",
    "MAX_REQUIRED_OBSERVATIONS",
    "MAX_REQUIRED_SECTIONS",
    "OBSERVED_AVAILABILITY_STATES",
    "SUPPORTED_GUIDANCE_OWNERS",
    "ExecutorGuidanceCompatibilityError",
    "HostVocabularyEntry",
    "build_executor_guidance_compatibility",
    "guidance_leakage_findings",
    "reads_as_observed_availability",
    "validate_executor_guidance_compatibility",
]
