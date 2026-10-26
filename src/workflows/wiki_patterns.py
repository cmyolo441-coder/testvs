"""Wiki organization patterns and operating rules, as data.

Someone standing up a wiki has to answer "how is this divided and who keeps it
alive" before a single page is worth writing, and the answer splits on audience:
a personal vault and a team wiki fail in different ways, so the same structure
cannot serve both. These tables carry the reusable part of that answer so the
skill proposes a model with its breaking conditions instead of improvising one
per conversation.

They live outside `SKILL.md` because a session that never mentions a wiki should
not pay for them; the wiki skill points at the rendered references and loads
them only when structure is actually being decided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# Two different axes, and conflating them gives a pair of people the operating
# rules of a solo vault. Model fit asks how big the corpus and its readership
# are, so a small group still suits the personal-scale models. Operating rules
# ask how many people write, and the answer stops being "one" at two: naming has
# to be agreed, links need a canonical target, and sections need an owner.
PERSONAL_AUDIENCES: Final = ("personal", "small_group")
SHARED_AUDIENCES: Final = ("team", "organization")
MULTI_WRITER_AUDIENCES: Final = ("small_group", "team", "organization")
AUDIENCE_SCALES: Final = PERSONAL_AUDIENCES + SHARED_AUDIENCES


@dataclass(frozen=True, slots=True)
class WikiPattern:
    """One organization model, with the conditions that make it fail."""

    name: str
    one_line: str
    fits_when: tuple[str, ...]
    breaks_when: tuple[str, ...]
    skeleton: tuple[str, ...]
    audience_note: str
    # Audiences this model is a primary recommendation for. A model outside its
    # audience is still offered as the alternative rather than dropped, because
    # the audience note explains what it would cost.
    suits_audiences: tuple[str, ...] = AUDIENCE_SCALES
    # Whether the model holds up when an agent is one of the readers. The test is
    # path stability: an agent cites a page by its location, so a model that
    # moves pages as a matter of routine breaks every citation it made.
    suits_agent_readers: bool = True


@dataclass(frozen=True, slots=True)
class WikiAgentReaderRule:
    """One requirement that appears once an agent reads the wiki, not only people."""

    topic: str
    rule: str
    failure_if_skipped: str


@dataclass(frozen=True, slots=True)
class WikiOperationRule:
    """One operating decision, answered separately for solo and shared wikis."""

    topic: str
    personal: str
    shared: str
    failure_if_skipped: str


_PATTERNS: Final = (
    WikiPattern(
        "PARA",
        "Split by actionability: projects, areas, resources, archive.",
        (
            "Most pages exist to support work that is currently moving.",
            "The same person both writes and acts on the notes.",
            "Finished work should stop competing with live work for attention.",
        ),
        (
            "Reference knowledge outlives the project that produced it and gets archived with it.",
            "Several people disagree about whether something is still a live area.",
        ),
        ("projects/", "areas/", "resources/", "archive/"),
        "Strong for personal and small-group vaults; needs an explicit owner per area before a team can trust it.",
        suits_audiences=PERSONAL_AUDIENCES,
        # Moving a page from projects/ to archive/ as work finishes is the whole
        # mechanic, so paths churn by design and an agent's citations rot.
        suits_agent_readers=False,
    ),
    WikiPattern(
        "Zettelkasten / evergreen notes",
        "One idea per page, densely linked, titles written as claims.",
        (
            "The value is in connections between ideas, not in a filing location.",
            "The same concepts keep resurfacing across unrelated work.",
            "The author rereads and revises pages rather than appending to them.",
        ),
        (
            "Pages are procedural steps that must be followed in order.",
            "Readers need one canonical answer instead of a network to traverse.",
            "Nobody has time to revise pages, so links rot into a dead graph.",
        ),
        ("notes/", "index or entry note", "link conventions"),
        "Best solo. In a team it needs a curator, or two people write two competing notes on one idea.",
        suits_audiences=PERSONAL_AUDIENCES,
    ),
    WikiPattern(
        "Diátaxis",
        "Split by reader intent: tutorial, how-to, reference, explanation.",
        (
            "Readers arrive with different needs against the same subject.",
            "New people must be onboarded without a guide sitting next to them.",
            "Mixing conceptual background into task steps is already hurting.",
        ),
        (
            "The corpus is small enough that four quadrants leave most of them empty.",
            "The subject is decisions and history rather than a product someone uses.",
        ),
        ("tutorials/", "how-to/", "reference/", "explanation/"),
        "Made for shared and public documentation; overkill for a personal vault.",
        suits_audiences=SHARED_AUDIENCES,
    ),
    WikiPattern(
        "Map of Content (MOC)",
        "Hand-curated entry pages that route to everything below them.",
        (
            "Search alone does not tell readers what exists.",
            "The structure has to change without moving or renaming files.",
            "A few high-traffic topics deserve a curated front door.",
        ),
        (
            "Nobody maintains the maps, so they silently drift from the pages.",
            "The page count is small enough that a flat list already works.",
        ),
        ("maps/", "topic pages", "one root entry map"),
        "Works at both scales; in a team, name an owner per map or the maps go stale first.",
    ),
    WikiPattern(
        "Docs-as-code",
        "Markdown in the repository, changed by pull request, reviewed like code.",
        (
            "Knowledge describes a codebase and goes stale when the code changes.",
            "Review and history matter as much as the content.",
            "Contributors already live in the repository.",
        ),
        (
            "Contributors do not use git, so the review step blocks writing entirely.",
            "The knowledge is not about a single repository.",
        ),
        ("docs/", "docs/adr/", "docs/runbooks/"),
        "Team-first. For personal use the review step is friction with no reviewer.",
        suits_audiences=SHARED_AUDIENCES,
    ),
    WikiPattern(
        "Decision log (ADR)",
        "Append-only records of what was decided, why, and what was rejected.",
        (
            "The same arguments keep getting relitigated.",
            "New people ask why something is the way it is.",
            "Context behind a choice matters longer than the choice itself.",
        ),
        (
            "Used as the only structure, so day-to-day reference has nowhere to live.",
            "Records are edited in place, which destroys the history that justified the log.",
        ),
        ("decisions/", "one file per decision", "status field"),
        "Valuable at every scale, and the single highest-return page type for a team.",
    ),
)


_OPERATION_RULES: Final = (
    WikiOperationRule(
        "Entry point",
        "One index note listing live areas; search covers the rest.",
        "One README or home page that answers 'what is here and where do I start'.",
        "Readers land in a flat file list, conclude nothing is here, and ask in chat instead.",
    ),
    WikiOperationRule(
        "Naming",
        "Whatever the author will recognize later; consistency matters more than the scheme.",
        "One written rule, since two people will otherwise pick two schemes for one topic.",
        "The same subject accumulates three pages under three names and none of them is complete.",
    ),
    WikiOperationRule(
        "Linking",
        "Link freely; a dangling link is a to-do, not an error.",
        "Link to canonical pages only, so readers do not have to guess which copy is current.",
        "Knowledge fragments into islands that only their author can navigate.",
    ),
    WikiOperationRule(
        "Ownership",
        "Implicitly the author; say so rather than pretending it is managed.",
        "A named owner per section, because shared ownership means nobody updates it.",
        "Pages age with no one responsible, and readers stop trusting all of them equally.",
    ),
    WikiOperationRule(
        "Update cadence",
        "On touch: update the page when the work touches it again.",
        "A stated review interval per section, plus a last-reviewed date on the page.",
        "A confidently wrong page outlives the truth and costs more than no page.",
    ),
    WikiOperationRule(
        "Duplication",
        "Merge on sight; the author is the only one holding the map.",
        "Search before writing, and merge into the canonical page rather than adding a near-copy.",
        "Two pages disagree and readers cannot tell which one to believe.",
    ),
    WikiOperationRule(
        "Retirement",
        "Archive rather than delete; keep it out of the search path.",
        "Mark superseded with a pointer to the replacement, and keep the record.",
        "Deleted context makes old decisions unexplainable; kept-but-unmarked context misleads.",
    ),
    WikiOperationRule(
        "Access",
        "Not a question until someone else needs to read it.",
        "Decide who can read and who can write before the first sensitive page exists.",
        "Either the wiki holds secrets it should not, or the people who need it cannot open it.",
    ),
)


_AGENT_READER_RULES: Final = (
    WikiAgentReaderRule(
        "Stable page identity",
        "Give each page a path that survives reorganization, and redirect rather than move.",
        "An agent cites a page by its location, so every earlier answer points at a file that is no longer there.",
    ),
    WikiAgentReaderRule(
        "One topic per page",
        "Split a page that answers more than one question.",
        "Retrieval returns whole pages, so a page covering five topics drags four irrelevant ones into context.",
    ),
    WikiAgentReaderRule(
        "Machine-readable header",
        "Put title, one-line summary, and last-reviewed date in front matter on every page.",
        "An agent cannot infer scope or freshness from visual layout the way a person skimming can.",
    ),
    WikiAgentReaderRule(
        "Self-contained pages",
        "Avoid 'see above', 'as mentioned', and meaning that depends on the neighbouring page.",
        "Retrieval delivers one page without its siblings, so the missing context is silently filled in wrong.",
    ),
    WikiAgentReaderRule(
        "Enumerable index",
        "Keep a listing file an agent can read, not only a visual home page.",
        "The agent cannot discover what exists and answers from whatever it happened to match.",
    ),
)

AGENT_READER: Final = "agent"
HUMAN_READER: Final = "human"


def wiki_agent_reader_rules() -> tuple[WikiAgentReaderRule, ...]:
    return _AGENT_READER_RULES


def wiki_patterns() -> tuple[WikiPattern, ...]:
    return _PATTERNS


def wiki_operation_rules() -> tuple[WikiOperationRule, ...]:
    return _OPERATION_RULES


def wiki_pattern(name: str) -> WikiPattern | None:
    normalized = name.strip().casefold()
    for pattern in _PATTERNS:
        if pattern.name.casefold() == normalized:
            return pattern
    return None
