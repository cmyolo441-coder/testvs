from collections.abc import Iterator
from dataclasses import dataclass
import re

from ..plugin_bundle.omh.domain_signals import (
    _fold_for_match,
    canonical_domain_token,
    canonical_domain_tokens,
    domain_tokens_are_locally_negated,
    DomainOperatorOverride,
    DomainRouteSignal,
    excluded_specialist_domain_skills as _excluded_specialist_domain_skills,
    specialist_domain_operator_override as _specialist_domain_operator_override,
    specialist_domain_route_signal as _specialist_domain_route_signal,
    specialist_domain_route_signals,
)
from .localization import normalized_phrase, routing_tokens
from .visual_qa_cues import contains_cue_phrase

RELEVANCE_POLICY = "shared_domain_signal/v1"

# These practitioner terms only establish whether a clarification candidate
# shares the request's domain. They are intentionally separate from the
# dispatch trigger catalog in the plugin bundle.
_RELEVANCE_ONLY_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "finance-analysis",
        (
            "dso",
            "revenue cutoff",
            "revenue recognition",
            "asc 606",
            "asc606",
            "burn multiple",
            "nrr",
            "net revenue retention",
        ),
    ),
    (
        "legal-compliance-review",
        (
            "indemnity",
            "liability cap",
            "gdpr article 35",
            "dpia",
        ),
    ),
    ("sales-development", ("meddpicc",)),
)

# The source observation classified this vocabulary as specialist-shaped, but
# it is not sufficient to nominate a catalog skill. It therefore activates the
# relevance gate while deliberately yielding no named candidate.
_UNOWNED_SPECIALIST_CUES = (
    ("rules-distill", ("four-fifths rule", "4/5 rule")),
    ("curriculum-design", ("bloom backward design", "bloom taxonomy")),
)

_DOWNSTREAM_PRD_ACTION_TOKENS = frozenset({"write", "create", "draft", "prepare"})
_DOWNSTREAM_PRD_CUES = ("prd", "product requirements document")
_DISCOVERY_EVIDENCE_CUES = ("findings", "evidence", "receipt")
_DISCOVERY_EVIDENCE_SOURCE_CUES = (
    "from these",
    "from the",
    "based on",
    "using these",
    "using the",
    "supplied",
    "provided",
)
_DISCOVERY_VALIDATION_ACTION_TOKENS = frozenset(
    {"validate", "validation", "test", "testing", "interview", "interviews"}
)
_DISCOVERY_VALIDATION_SUBJECT_CUES = (
    "customer interview",
    "customer interviews",
    "interview customers",
    "customer problem",
    "problem hypothesis",
    "this idea",
    "the idea",
    "an idea",
)
_DISCOVERY_COMPLETION_TOKENS = frozenset(
    {"already", "validated", "completed", "supplied", "provided"}
)
_COPY_ACTION_TOKENS = frozenset({"write", "create", "draft", "prepare", "rewrite"})
_COPY_DELIVERABLE_CUES = ("email", "email copy", "copy", "message copy", "push copy", "banner copy")
_COMPLETED_LIFECYCLE_TOKENS = frozenset({"approved", "planned"})
_LIFECYCLE_DESIGN_CUES = (
    "design",
    "designing",
    "experiment design",
    "journey design",
    "audience",
    "consent",
    "frequency",
    "measurement",
    "holdout",
    "treatment",
    "segment",
)


def _is_downstream_product_brief_request(message: str) -> bool:
    """Return whether supplied discovery evidence is being turned into a PRD."""
    normalized = normalized_phrase(message)
    tokens = routing_tokens(normalized)
    return (
        bool(tokens & _DOWNSTREAM_PRD_ACTION_TOKENS)
        and contains_cue_phrase(normalized, _DOWNSTREAM_PRD_CUES)
        and contains_cue_phrase(normalized, _DISCOVERY_EVIDENCE_CUES)
        and contains_cue_phrase(normalized, _DISCOVERY_EVIDENCE_SOURCE_CUES)
    )


def _is_active_product_discovery_validation_request(message: str) -> bool:
    """Return whether customer or idea validation is requested before a PRD."""
    normalized = normalized_phrase(message)
    tokens = routing_tokens(normalized)
    return (
        bool(tokens & _DISCOVERY_VALIDATION_ACTION_TOKENS)
        and "before" in tokens
        and contains_cue_phrase(normalized, _DOWNSTREAM_PRD_CUES)
        and contains_cue_phrase(normalized, _DISCOVERY_VALIDATION_SUBJECT_CUES)
        and not bool(tokens & _DISCOVERY_COMPLETION_TOKENS)
    )


def _is_completed_lifecycle_copy_request(message: str) -> bool:
    """Return whether a completed lifecycle plan needs only a copy deliverable."""
    normalized = normalized_phrase(message)
    tokens = routing_tokens(normalized)
    return (
        bool(tokens & _COPY_ACTION_TOKENS)
        and bool(tokens & _COMPLETED_LIFECYCLE_TOKENS)
        and contains_cue_phrase(normalized, _COPY_DELIVERABLE_CUES)
        and not contains_cue_phrase(normalized, _LIFECYCLE_DESIGN_CUES)
    )


def specialist_domain_route_signal(message: str) -> DomainRouteSignal | None:
    """Keep active discovery validation ahead of, and completed evidence behind, a PRD."""
    signal = _specialist_domain_route_signal(message)
    if _is_active_product_discovery_validation_request(message):
        return DomainRouteSignal(
            skill="product-discovery-validation",
            matched_cues=("active customer or idea validation", "before prd"),
        )
    if signal is not None and signal.skill == "product-discovery-validation" and _is_downstream_product_brief_request(message):
        return DomainRouteSignal(skill="product-brief", matched_cues=("supplied discovery evidence", "prd"))
    return signal


def excluded_specialist_domain_skills(message: str) -> frozenset[str]:
    """Keep the discovery-to-PRD boundary available in both directions."""
    excluded = _excluded_specialist_domain_skills(message)
    if _is_active_product_discovery_validation_request(message):
        return (excluded - {"product-discovery-validation"}) | {"product-brief"}
    if _is_downstream_product_brief_request(message):
        return (excluded - {"product-brief"}) | {"product-discovery-validation"}
    return excluded


def specialist_domain_operator_override(
    message: str,
    signal: DomainRouteSignal | None = None,
) -> DomainOperatorOverride | None:
    """Let a downstream copy task leave a completed lifecycle-design workflow."""
    if signal is None:
        signal = specialist_domain_route_signal(message)
    if signal is not None and signal.skill == "lifecycle-growth" and _is_completed_lifecycle_copy_request(message):
        return DomainOperatorOverride(skill="content-operator", matched_cues=("completed lifecycle copy",))
    return _specialist_domain_operator_override(message, signal)


@dataclass(frozen=True)
class ClarificationRelevance:
    skills: tuple[str, ...] | None
    blocked_skills: tuple[str, ...] = ()

    @property
    def applies(self) -> bool:
        return self.skills is not None


def _relevance_pattern(cue: str) -> re.Pattern[str]:
    phrase = _fold_for_match(cue)
    if phrase == "4/5 rule":
        pattern = r"4\s*/\s*5[\s_-]+rule"
    else:
        parts = tuple(part for part in re.split(r"[\s_-]+", phrase) if part)
        pattern = r"[\s_-]+".join(re.escape(part) for part in parts)
    return re.compile(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])")


_RELEVANCE_CUE_GROUPS = tuple(
    (f"cue_{index}", skill, blocked, _relevance_pattern(cue).pattern)
    for index, (skill, cue, blocked) in enumerate(
        (
            *((skill, cue, False) for skill, cues in _RELEVANCE_ONLY_CUES for cue in cues),
            *((skill, cue, True) for skill, cues in _UNOWNED_SPECIALIST_CUES for cue in cues),
        )
    )
)
_RELEVANCE_SCAN_PATTERN = re.compile(
    "|".join(
        (
            *(f"(?P<{group}>{pattern})" for group, _, _, pattern in _RELEVANCE_CUE_GROUPS),
            r"(?P<separator>[,;.!?\n]+|\band\b)",
            r"(?P<contraction>[a-z]+n['’]t)",
            r"(?P<token>[a-z0-9]+)",
        )
    )
)
_RELEVANCE_CUE_BY_GROUP = {
    group: (skill, blocked) for group, skill, blocked, _ in _RELEVANCE_CUE_GROUPS
}
_OWNED_RELEVANCE_SKILLS = tuple(skill for skill, _ in _RELEVANCE_ONLY_CUES)
_BLOCKED_RELEVANCE_SKILLS = tuple(skill for skill, _ in _UNOWNED_SPECIALIST_CUES)


def _normalized_relevance_message(message: str) -> str:
    return _fold_for_match(message)


def _full_message_relevance_scan(normalized: str) -> Iterator[re.Match[str]]:
    """Expose the one actual combined-matcher traversal for instrumentation."""
    return _RELEVANCE_SCAN_PATTERN.finditer(normalized)


def _scan_normalized_relevance_message(normalized: str) -> ClarificationRelevance:
    """Derive all relevance outcomes in one combined full-message scan."""
    positive_owned: set[str] = set()
    positive_blocked: set[str] = set()
    clause_tokens: list[str] = []
    saw_positive_cue = False
    for match in _full_message_relevance_scan(normalized):
        group = match.lastgroup
        if group == "separator":
            clause_tokens.clear()
            continue
        if group in {"token", "contraction"}:
            clause_tokens.append(canonical_domain_token(match.group()))
            continue
        if group is None:
            continue
        skill, blocked = _RELEVANCE_CUE_BY_GROUP[group]
        if not domain_tokens_are_locally_negated(tuple(clause_tokens), len(clause_tokens)):
            saw_positive_cue = True
            (positive_blocked if blocked else positive_owned).add(skill)
        clause_tokens.extend(canonical_domain_tokens(match.group()))

    skills = tuple(skill for skill in _OWNED_RELEVANCE_SKILLS if skill in positive_owned)
    blocked_skills = tuple(
        skill for skill in _BLOCKED_RELEVANCE_SKILLS if skill in positive_blocked
    )
    return ClarificationRelevance(skills if saw_positive_cue else None, blocked_skills)


def classify_clarification_relevance(message: str) -> ClarificationRelevance:
    """Classify request relevance with one fold and one full-message scan."""
    return _scan_normalized_relevance_message(_normalized_relevance_message(message))


def clarification_relevance_skills(message: str) -> tuple[str, ...] | None:
    """Compatibility projection for callers that do not retain classification."""
    return classify_clarification_relevance(message).skills


__all__ = (
    "ClarificationRelevance",
    "DomainOperatorOverride",
    "DomainRouteSignal",
    "RELEVANCE_POLICY",
    "clarification_relevance_skills",
    "classify_clarification_relevance",
    "excluded_specialist_domain_skills",
    "specialist_domain_operator_override",
    "specialist_domain_route_signal",
    "specialist_domain_route_signals",
)
