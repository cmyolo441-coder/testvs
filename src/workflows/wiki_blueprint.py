"""Turn a wiki request into a startable structure instead of a fresh improvisation.

The wiki skill used to ask for a durable fact and a source before it would do
anything, which is exactly what someone standing up a wiki does not have yet:
their problem is that nothing is divided, not that a sentence needs a home. This
module answers the structural question deterministically - audience scale and
the kinds of knowledge that repeat select an organization model, the destination
classifier picks the store shape - so the same request produces the same
blueprint instead of whatever the model improvises that turn.

Nothing here creates, writes to, or migrates a store. OMH does not host the
wiki; the user's own store does, and the blueprint says so in its own boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Final

from ..catalogs.awesome_hermes_agent import AwesomeHermesCoverage, awesome_hermes_coverage
from .knowledge_connections import KnowledgeConnectionOptions, build_knowledge_connection_intent
from .wiki_patterns import (
    MULTI_WRITER_AUDIENCES,
    WikiPattern,
    wiki_agent_reader_rules,
    wiki_operation_rules,
    wiki_pattern,
    wiki_patterns,
)


WIKI_BLUEPRINT_SCHEMA_VERSION: Final = "wiki_blueprint/v1"

# A blueprint someone can act on today. Past this, the list becomes a backlog
# nobody starts.
SEED_PAGE_CAP: Final = 10

UNKNOWN_AUDIENCE: Final = "unknown"
_AUDIENCE_ALIASES: Final = {
    "personal": "personal",
    "solo": "personal",
    "me": "personal",
    "myself": "personal",
    "individual": "personal",
    "개인": "personal",
    "혼자": "personal",
    "small group": "small_group",
    "small_group": "small_group",
    "small team": "small_group",
    "소그룹": "small_group",
    "team": "team",
    "팀": "team",
    "org": "organization",
    "organization": "organization",
    "company": "organization",
    "전사": "organization",
    "조직": "organization",
}

# Knowledge types map to the model that serves them; order is the priority when a
# request names several.
_KNOWLEDGE_TYPE_MODELS: Final = (
    ("decision", "Decision log (ADR)"),
    ("adr", "Decision log (ADR)"),
    ("결정", "Decision log (ADR)"),
    ("onboarding", "Diátaxis"),
    ("how-to", "Diátaxis"),
    ("procedure", "Diátaxis"),
    ("runbook", "Diátaxis"),
    ("troubleshooting", "Diátaxis"),
    ("절차", "Diátaxis"),
    ("code", "Docs-as-code"),
    ("api", "Docs-as-code"),
    ("codebase", "Docs-as-code"),
    ("research", "Zettelkasten / evergreen notes"),
    ("idea", "Zettelkasten / evergreen notes"),
    ("리서치", "Zettelkasten / evergreen notes"),
    ("glossary", "Map of Content (MOC)"),
    ("term", "Map of Content (MOC)"),
    ("용어", "Map of Content (MOC)"),
    ("project", "PARA"),
    ("task", "PARA"),
    ("프로젝트", "PARA"),
)

_AUDIENCE_DEFAULT_MODELS: Final = {
    "personal": "PARA",
    "small_group": "Map of Content (MOC)",
    "team": "Diátaxis",
    "organization": "Diátaxis",
    UNKNOWN_AUDIENCE: "Map of Content (MOC)",
}

# Only a folder the user actually asked for. `local_markdown_folder` is the
# classifier's default when no destination was named at all, so treating it as a
# repository signal would hand docs-as-code - review-gated and team-first - to a
# solo user who said nothing about where the wiki lives.
# Stable paths, an enumerable index, and it works at every scale.
_AGENT_SAFE_DEFAULT_MODEL: Final = "Map of Content (MOC)"

_REPO_DESTINATION_KINDS: Final = {"markdown_folder"}

UNKNOWN_DESTINATION_KIND: Final = "unknown_external_destination"

# Vault wording the shared classifier reads only from an explicit target. Bare
# "vault" is left out on purpose: a secrets vault is not a knowledge vault.
_VAULT_TERMS: Final = ("obsidian", "옵시디언", "markdown vault", "마크다운 볼트")

# Wording that means a machine is one of the readers.
_AGENT_READER_TERMS: Final = (
    "agent",
    "assistant",
    "hermes",
    "claude",
    "codex",
    "llm",
    " ai ",
    "에이전트",
    "어시스턴트",
    "에이아이",
)

# Kinds the classifier only returns when the request actually identified a store,
# by message text or by option. `local_markdown_folder` is missing on purpose: it
# is what comes back when nothing was said.
_CLASSIFIED_DESTINATION_KINDS: Final = {
    "markdown_vault",
    "notion_knowledge_base",
    "google_document_store",
    "database",
    "markdown_folder",
}

# Terms that mark an upstream entry as knowledge-structure material rather than a
# storage backend.
_KNOWLEDGE_ITEM_TERMS: Final = (
    "wiki",
    "knowledge base",
    "obsidian",
    "markdown vault",
    "second brain",
    "notes",
    "documentation",
)


@dataclass(frozen=True, slots=True)
class WikiBlueprintRequest:
    """What the design interview collected, with every field allowed to be absent."""

    text: str = ""
    audience_scale: str = ""
    maintenance_owner: str = ""
    knowledge_types: tuple[str, ...] = ()
    # Who reads it, which is a different question from how many people write it.
    # Left empty, the message is scanned: people say "Hermes will read this" long
    # before anyone thinks to answer an interview field.
    readers: tuple[str, ...] = ()
    connection: KnowledgeConnectionOptions = field(default_factory=KnowledgeConnectionOptions)


def build_wiki_blueprint(request: WikiBlueprintRequest) -> dict[str, object]:
    audience = normalize_audience(request.audience_scale)
    destination = build_knowledge_connection_intent(request.text, options=_destination_options(request))
    agent_readers = has_agent_readers(request)
    primary, alternative = select_models(
        audience=audience,
        knowledge_types=request.knowledge_types,
        destination_kind=str(destination["kind"]),
        agent_readers=agent_readers,
    )
    # More than one writer, not "big". Two people already need agreed naming, a
    # canonical link target, and a named owner.
    shared = audience in MULTI_WRITER_AUDIENCES
    owner = request.maintenance_owner.strip()
    return {
        "schema_version": WIKI_BLUEPRINT_SCHEMA_VERSION,
        "status": "prepared",
        "audience_scale": audience,
        "shared_audience": shared,
        "agent_readers": agent_readers,
        "agent_reader_rules": _agent_reader_payload(agent_readers),
        "destination": destination,
        "organization_model": _model_payload(primary),
        "alternative_model": _model_payload(alternative),
        "skeleton": list(primary.skeleton),
        "entry_points": _entry_points(primary, shared=shared),
        "conventions": _rules_payload(("Naming", "Linking", "Entry point"), shared=shared),
        "maintenance": {
            "owner": owner or "unmaintained",
            "owner_known": bool(owner),
            "rules": _rules_payload(("Ownership", "Update cadence", "Duplication", "Retirement", "Access"), shared=shared),
        },
        "seed_page_cap": SEED_PAGE_CAP,
        "ecosystem_candidates": ecosystem_candidates(),
        "missing_facts": _missing_facts(
            audience=audience,
            owner=owner,
            knowledge_types=request.knowledge_types,
            destination=destination,
        ),
        "next_action": (
            "Confirm the model and skeleton with the user, then create the entry point and seed pages in their own "
            "store."
        ),
        "claim_boundary": (
            "A wiki blueprint is prepared design context. It is not evidence that a store was created, a vault or "
            "workspace was written to, a connector ran, a migration happened, or that any page exists."
        ),
    }


def _destination_options(request: WikiBlueprintRequest) -> KnowledgeConnectionOptions:
    """Read a vault named in the message as the destination it plainly is.

    The shared classifier resolves Notion, Google Drive, databases, and markdown
    folders from message text, but Obsidian only from an explicit option, so
    "structure my Obsidian vault" fell through to the unnamed-folder default.
    Research-department depends on that asymmetry - a passing "using Obsidian if
    possible" in a recurring-ops request must not become a store commitment - so
    the promotion happens here, where naming a vault *is* choosing where the wiki
    lives, instead of in the classifier both lanes share.
    """
    if request.connection.knowledge_store.strip() or request.connection.storage.strip():
        return request.connection
    haystack = request.text.casefold()
    for term in _VAULT_TERMS:
        if term in haystack:
            return replace(request.connection, knowledge_store=term)
    return request.connection


def normalize_audience(value: str) -> str:
    normalized = str(value or "").strip().casefold()
    if not normalized:
        return UNKNOWN_AUDIENCE
    return _AUDIENCE_ALIASES.get(normalized, UNKNOWN_AUDIENCE)


def has_agent_readers(request: WikiBlueprintRequest) -> bool:
    """Whether an agent is among the readers, from the interview field or the message."""
    for reader in request.readers:
        if _contains_any(str(reader or "").casefold(), _AGENT_READER_TERMS):
            return True
    return _contains_any(request.text.casefold(), _AGENT_READER_TERMS)


def select_models(
    *,
    audience: str,
    knowledge_types: tuple[str, ...],
    destination_kind: str,
    agent_readers: bool = False,
) -> tuple[WikiPattern, WikiPattern]:
    """Pick a primary model and a distinct alternative, most specific signal first."""
    ranked = _ranked_model_names(
        audience=audience,
        knowledge_types=knowledge_types,
        destination_kind=destination_kind,
        agent_readers=agent_readers,
    )
    resolved = [pattern for pattern in (wiki_pattern(name) for name in ranked) if pattern is not None]
    primary = resolved[0] if resolved else wiki_patterns()[0]
    alternative = next((pattern for pattern in resolved[1:] if pattern.name != primary.name), None)
    if alternative is None:
        # Falling back to the first catalog entry hands an organization-wide wiki
        # a personal-scale model as its second option. Prefer one that fits the
        # audience, and only then take whatever is left.
        remaining = [pattern for pattern in wiki_patterns() if pattern.name != primary.name]
        alternative = next(
            (pattern for pattern in remaining if _suits_audience(pattern.name, audience)),
            remaining[0],
        )
    return primary, alternative


def wiki_ecosystem_coverage() -> tuple[AwesomeHermesCoverage, ...]:
    """Upstream entries that are about building a knowledge base, not storing bytes.

    The `wiki` surface alone is too wide: coverage rules attach it to every
    memory-provider plugin, and a vector store is not a wiki pattern someone can
    copy. The second filter keeps entries whose own text is about wikis, notes,
    or documentation, which is what a person designing a structure can learn
    from. `vault` is deliberately qualified - an HSM secret vault is not a
    knowledge vault.
    """
    return tuple(
        coverage
        for coverage in sorted(awesome_hermes_coverage(), key=lambda coverage: coverage.item.id)
        if "wiki" in coverage.omh_surfaces and _knowledge_oriented(coverage)
    )


def ecosystem_candidates() -> list[dict[str, str]]:
    return [
        {
            "id": coverage.item.id,
            "name": coverage.item.name,
            "url": coverage.item.url,
            "coverage_status": coverage.status,
        }
        for coverage in wiki_ecosystem_coverage()
    ]


def _knowledge_oriented(coverage: AwesomeHermesCoverage) -> bool:
    haystack = f"{coverage.item.name} {coverage.item.summary}".casefold()
    return any(term in haystack for term in _KNOWLEDGE_ITEM_TERMS)


def _ranked_model_names(
    *,
    audience: str,
    knowledge_types: tuple[str, ...],
    destination_kind: str,
    agent_readers: bool = False,
) -> list[str]:
    names: list[str] = []
    for knowledge_type in knowledge_types:
        normalized = str(knowledge_type or "").strip().casefold()
        if not normalized:
            continue
        for term, model in _KNOWLEDGE_TYPE_MODELS:
            if term in normalized and model not in names:
                names.append(model)
    if destination_kind in _REPO_DESTINATION_KINDS and "Docs-as-code" not in names:
        names.append("Docs-as-code")
    default_model = _AUDIENCE_DEFAULT_MODELS.get(audience, _AUDIENCE_DEFAULT_MODELS[UNKNOWN_AUDIENCE])
    if default_model not in names:
        names.append(default_model)
    if agent_readers and not any(
        _suits_audience(name, audience) and _suits_agent_readers(name) for name in names
    ):
        # Demotion needs something to promote. Without this, a solo user whose
        # agent reads the wiki chooses between PARA, which relocates pages by
        # design, and a model built for a team - one fails each axis.
        names.append(_AGENT_SAFE_DEFAULT_MODEL)
    # What the knowledge is about cannot outrank who the wiki is for. A solo
    # developer documenting a repository matches docs-as-code on content and
    # fails it on audience: the review step is friction with no reviewer. Models
    # outside the audience drop behind the ones inside it, and behind the
    # audience default, rather than disappearing - they stay available as the
    # alternative, where the audience note explains the cost.
    # Both fit tests at once, audience first. Applying them as two successive
    # re-sorts let the later one undo the earlier: a solo user documenting code
    # for an agent had docs-as-code demoted for audience, then promoted straight
    # back because the model it lost to relocates pages. Sorting is stable, so
    # signal order survives inside a tier.
    return sorted(
        names,
        key=lambda name: (
            0 if _suits_audience(name, audience) else 1,
            0 if not agent_readers or _suits_agent_readers(name) else 1,
        ),
    )


def _suits_agent_readers(model_name: str) -> bool:
    pattern = wiki_pattern(model_name)
    return pattern is None or pattern.suits_agent_readers


def _agent_reader_payload(agent_readers: bool) -> list[dict[str, str]]:
    if not agent_readers:
        return []
    return [
        {"topic": rule.topic, "rule": rule.rule, "failure_if_skipped": rule.failure_if_skipped}
        for rule in wiki_agent_reader_rules()
    ]


def _contains_any(haystack: str, terms: tuple[str, ...]) -> bool:
    return any(term in haystack for term in terms)


def _suits_audience(model_name: str, audience: str) -> bool:
    if audience == UNKNOWN_AUDIENCE:
        return True
    pattern = wiki_pattern(model_name)
    return pattern is None or audience in pattern.suits_audiences


def _model_payload(pattern: WikiPattern) -> dict[str, object]:
    return {
        "name": pattern.name,
        "rationale": pattern.one_line,
        "fits_when": list(pattern.fits_when),
        "breaks_when": list(pattern.breaks_when),
        "skeleton": list(pattern.skeleton),
        "audience_note": pattern.audience_note,
    }


def _entry_points(pattern: WikiPattern, *, shared: bool) -> list[str]:
    if shared:
        return ["home page answering 'what is here and where do I start'", f"one index per {pattern.skeleton[0]} section"]
    return ["one index note listing live areas"]


def _rules_payload(topics: tuple[str, ...], *, shared: bool) -> list[dict[str, str]]:
    selected = {topic.casefold() for topic in topics}
    return [
        {
            "topic": rule.topic,
            "rule": rule.shared if shared else rule.personal,
            "failure_if_skipped": rule.failure_if_skipped,
        }
        for rule in wiki_operation_rules()
        if rule.topic.casefold() in selected
    ]


def _missing_facts(
    *,
    audience: str,
    owner: str,
    knowledge_types: tuple[str, ...],
    destination: dict[str, object],
) -> list[str]:
    missing: list[str] = []
    if audience == UNKNOWN_AUDIENCE:
        missing.append("audience scale (personal, small group, team, or organization)")
    if not owner:
        missing.append("maintenance owner and review cadence")
    if not [item for item in knowledge_types if str(item or "").strip()]:
        missing.append("knowledge types the wiki must hold")
    # Asking again for something the message already said is the round-trip this
    # skill exists to remove. "Set up a wiki in Notion" names the store even
    # though no `--knowledge-store` option was passed, so only a named-but-
    # unverified target or a bare default stays open.
    if destination["kind"] == UNKNOWN_DESTINATION_KIND:
        missing.append("confirmation that the named destination exists")
    elif str(destination["kind"]) not in _CLASSIFIED_DESTINATION_KINDS and not destination["explicit_target"]:
        missing.append("destination store or confirmation that it exists")
    return missing
