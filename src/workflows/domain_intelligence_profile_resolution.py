from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unicodedata


MAX_DOMAIN_CONTEXT_INPUT_CODE_POINTS = 4_096
MAX_DOMAIN_CONTEXT_MATCHES = 64


@dataclass(frozen=True)
class DomainClarificationTarget:
    workflow_hint: str
    required_input: str
    question_locale: str
    question_text: str


@dataclass(frozen=True)
class DomainClarificationResolution:
    reason: str
    target: DomainClarificationTarget | None = None


def resolve_domain_clarification_target(
    profiles: tuple[dict[str, object], ...],
    message: str,
    *,
    project_root: Path,
    locale: str,
) -> DomainClarificationTarget | None:
    """Compatibility projection for callers that only need the selected target."""
    return resolve_domain_clarification_target_result(
        profiles,
        message,
        project_root=project_root,
        locale=locale,
    ).target


def resolve_domain_clarification_target_result(
    profiles: tuple[dict[str, object], ...],
    message: str,
    *,
    project_root: Path,
    locale: str,
) -> DomainClarificationResolution:
    from ..plugin_bundle.omh.project_identity import resolve_project_identity
    from ..skills import catalog

    if len(message) > MAX_DOMAIN_CONTEXT_INPUT_CODE_POINTS:
        return DomainClarificationResolution("input_too_large")

    resolution = resolve_project_identity(project_root)
    if resolution.state != "resolved":
        return DomainClarificationResolution("no_match")
    expected_scope = {
        "kind": "project",
        "ref": resolution.identity,
        "ref_authority": "operator_or_wrapper_supplied",
        "identity_claim": "not_authenticated_identity_evidence",
    }
    normalized_message = _normalize_match_text(message)
    if len(normalized_message) > MAX_DOMAIN_CONTEXT_INPUT_CODE_POINTS:
        return DomainClarificationResolution("input_too_large")
    matches: list[tuple[dict[str, object], dict[str, object]]] = []
    for profile in profiles:
        if profile.get("scope") != expected_scope:
            continue
        mappings = profile.get("vocabulary_mappings")
        if not isinstance(mappings, list):
            return DomainClarificationResolution("profile_store_unhealthy")
        for mapping in mappings:
            if not isinstance(mapping, dict):
                return DomainClarificationResolution("profile_store_unhealthy")
            if _matches_normalized_message(normalized_message, mapping.get("phrase")):
                matches.append((profile, mapping))
                if len(matches) > MAX_DOMAIN_CONTEXT_MATCHES:
                    return DomainClarificationResolution("match_overflow")
    if not matches:
        return DomainClarificationResolution("no_match")

    canonicals = {str(mapping.get("canonical")) for _profile, mapping in matches}
    if len(canonicals) != 1:
        return DomainClarificationResolution("canonical_conflict")

    routable = {
        definition.name: definition for definition in catalog.routable_definitions()
    }
    builtin_names = {definition.name for definition in catalog.builtin_definitions()}
    selected_hints: set[str] = set()
    for profile, _mapping in matches:
        hints = profile.get("workflow_hints")
        if not isinstance(hints, list):
            return DomainClarificationResolution("profile_store_unhealthy")
        if not hints:
            return DomainClarificationResolution("empty_workflow_hints")
        for hint in hints:
            if not isinstance(hint, str) or hint not in builtin_names:
                return DomainClarificationResolution("unknown_workflow_hint")
            if hint not in routable:
                exposure = catalog.surface_exposure_for_skill(hint)
                reason = (
                    "compatibility_only_workflow_hint"
                    if exposure.compatibility_alias
                    else "non_routable_workflow_hint"
                )
                return DomainClarificationResolution(reason)
            selected_hints.add(hint)
    if len(selected_hints) != 1:
        return DomainClarificationResolution("conflicting_workflow_hints")

    workflow_hint = next(iter(selected_hints))
    definition = routable[workflow_hint]
    questions = definition.expert_questions
    if not questions:
        return DomainClarificationResolution("missing_question_spec")
    question = questions[0]
    if question.required_input not in definition.required_inputs:
        return DomainClarificationResolution("missing_question_spec")
    selected_locale = (
        "ko" if isinstance(locale, str) and locale.casefold() == "ko" else "en"
    )
    return DomainClarificationResolution(
        "applied",
        DomainClarificationTarget(
            workflow_hint=workflow_hint,
            required_input=question.required_input,
            question_locale=selected_locale,
            question_text=question.question_for_locale(selected_locale),
        ),
    )


def matches_reviewed_phrase(message: object, phrase: object) -> bool:
    """Return whether a normalized literal phrase has a valid Unicode boundary match."""
    return _matches_normalized_message(_normalize_match_text(message), phrase)


def _matches_normalized_message(normalized_message: str, phrase: object) -> bool:
    normalized_phrase = _normalize_match_text(phrase)
    if not normalized_message or not normalized_phrase:
        return False

    offset = normalized_message.find(normalized_phrase)
    while offset >= 0:
        end = offset + len(normalized_phrase)
        left_valid = (
            not _is_word_like(normalized_phrase[0])
            or offset == 0
            or not _is_word_like(normalized_message[offset - 1])
        )
        right_valid = (
            not _is_word_like(normalized_phrase[-1])
            or end == len(normalized_message)
            or not _is_word_like(normalized_message[end])
        )
        if left_valid and right_valid:
            return True
        offset = normalized_message.find(normalized_phrase, offset + 1)
    return False


def _normalize_match_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value).strip()
    return " ".join(normalized.split()).casefold()


def _is_word_like(value: str) -> bool:
    return value == "_" or unicodedata.category(value)[0] in {"L", "N", "M"}
