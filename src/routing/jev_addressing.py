"""Which `jev-*` skill a message invokes, when it addresses Jev at all.

A message reaches a Jev skill only by ADDRESSING Jev -- asking it, using it,
having it do something, or naming a Jev skill. Mentioning Jev is not enough:
"review the jev plugin PR", "debug why the jev plugin keeps looping", and "is
jev a good coding model?" are about Jev and keep their ordinary owner. That is
why every `jev-*` skill holds its own trigger tokens back and this module is the
only way in.

Precedence, most specific first (fixed, so an ask never depends on scoring):

1. A Jev skill's name in its invocation form (a sigil, or opening the
   message, or after "use").
2. The message addresses Jev (and is not a negation, a configuration, or a
   choice among options) and carries one skill's own phrase, then a cue for
   one skill's job, checked in `_CUE_ORDER`.
3. The message addresses Jev and the ordinary route would have gone, at
   `high` confidence, to a partner workflow in `JEV_SIBLING_BY_PARTNER`: the
   sibling takes it.
4. The message addresses Jev: `jev-ask`, ahead of the generic `ask` skill.

Returns None for everything else, and the router proceeds as if this module
did not exist.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from typing import TYPE_CHECKING, Final

from .localization import normalized_phrase, phrase_is_spoken

if TYPE_CHECKING:
    from .query import RoutingQuery

JEV_ASK: Final = "jev-ask"
JEV_ROUTE: Final = "jev-route"
JEV_FAILURE_TRIAGE: Final = "jev-failure-triage"
JEV_REVIEW_GATE: Final = "jev-review-gate"
JEV_ACTION_CHECK: Final = "jev-action-check"
JEV_DONE_CHECK: Final = "jev-done-check"
JEV_SKILL_NAMES: Final = (JEV_ASK, JEV_ROUTE, JEV_FAILURE_TRIAGE, JEV_REVIEW_GATE, JEV_ACTION_CHECK, JEV_DONE_CHECK)

# Phrases that address Jev as the one being asked to do something. Closed
# (critic C4): a sentence that only talks about Jev is not here on purpose.
# "run ... through jev" is spelled out whole, because a bare "through jev"
# also reads "go through jev's pricing page".
JEV_ADDRESSING_PHRASES: Final = (
    "ask jev",
    "use jev",
    "have jev",
    "let jev",
    "run it through jev",
    "run this through jev",
    "run that through jev",
    "run this diff through jev",
    "run the diff through jev",
)
# Imperatives that address Jev only when they open the message ("jev check
# this command") or follow "run a"/"do a". "the jev check in doctor" is the
# doctor line, not a request.
JEV_OPENING_IMPERATIVES: Final = ("jev check", "jev score", "jev question")
_IMPERATIVE_LEADS: Final = ("", "run a ", "do a ", "please ", "run ", "do ")
# Sentences that contain an addressing phrase and still do not address Jev:
# configuration ("use jev as my router model"), negation ("don't ask jev"),
# a choice among options (F10: "use jev or claude"), or a description of
# someone's use ("we use jev in production").
JEV_NOT_ADDRESSING_PHRASES: Final = (
    "use jev as",
    "jev as my",
    "jev as the",
    "install jev",
    "install the jev",
    "set up jev",
    "set up the jev",
    "configure jev",
    "configure the jev",
    "don't use jev",
    "do not use jev",
    "never use jev",
    "don't ask jev",
    "do not ask jev",
    "never ask jev",
    "no need to ask jev",
    "without jev",
    "ask jev later",
    "should we use jev",
    "should i use jev",
    "whether to use jev",
    "we use jev",
    "they use jev",
    "you use jev",
    "i use jev",
)
# "jev or claude", "gpt or jev": Jev named among options is a choice, which
# `decide` or research owns, never an ask.
_JEV_AMONG_OPTIONS: Final = re.compile(r"(?<![\w-])jev\s+or\s|\sor\s+jev(?![\w-])")

# 1. Each skill's own phrases. A phrase that already addresses Jev and names
# the job wins over every cue below.
JEV_SKILL_PHRASES: Final[dict[str, tuple[str, ...]]] = {
    JEV_ROUTE: ("ask jev which workflow", "jev pick the workflow"),
    JEV_ACTION_CHECK: ("ask jev if this command is safe", "jev action check", "jev risk check"),
    JEV_DONE_CHECK: ("ask jev if this is done", "jev done check", "jev evidence check"),
    JEV_FAILURE_TRIAGE: ("ask jev if this failure is transient", "jev failure triage"),
    JEV_REVIEW_GATE: ("ask jev to review this diff", "jev review gate"),
}

# 2. Cues for each skill's job, consulted only once the message addresses Jev.
_CUE_ORDER: Final = (JEV_ROUTE, JEV_DONE_CHECK, JEV_FAILURE_TRIAGE, JEV_ACTION_CHECK, JEV_REVIEW_GATE)
JEV_SKILL_CUES: Final[dict[str, tuple[str, ...]]] = {
    # Multi-word job phrases only: a bare "command", "patch", "review", or
    # "is complete" also reads "the command palette docs", "the patch notes",
    # or "this doc section is complete", which are free-form jev-ask
    # questions, not a preset's `state`.
    JEV_ROUTE: ("which workflow", "what workflow", "which omh workflow", "which skill fits"),
    JEV_ACTION_CHECK: (
        "command is safe",
        "command safe",
        "safe to run",
        "before i run",
        "before running",
        "rm -rf",
        "risky to run",
        "this write safe",
    ),
    JEV_DONE_CHECK: (
        "task is done",
        "task is finished",
        "task is complete",
        "work is done",
        "work is finished",
        "really done",
        "completion claim",
        "claim is supported",
    ),
    JEV_FAILURE_TRIAGE: (
        "failing test",
        "test failure",
        "build failure",
        "this failure",
        "the failure",
        "keeps failing",
        "failed with",
        "build broke",
        "run that broke",
        "broke the build",
        "error output",
        "transient",
        "flaky",
        "triage",
    ),
    JEV_REVIEW_GATE: (
        "review this diff",
        "review the diff",
        "review my diff",
        "this diff",
        "the diff",
        "review this pr",
        "review the pr",
        "review my pr",
        "review this pull request",
        "review this change",
    ),
}

# 3. Partner workflow -> Jev sibling, for a message that addresses Jev. No
# partner maps to `jev-action-check`: its preset judges a command about to
# run, and a command-operator or security owner also takes tracebacks and
# audits ("have jev look at this pytest traceback"); it is reached by its own
# phrases and cues only.
JEV_SIBLING_BY_PARTNER: Final[dict[str, str]] = {
    "build-failure-triage": JEV_FAILURE_TRIAGE,
    "agent-debug": JEV_FAILURE_TRIAGE,
    "code-review": JEV_REVIEW_GATE,
    "verification-gate": JEV_DONE_CHECK,
}

_NAME_TOKEN: Final = re.compile(r"[^\s,.;:!?()\[\]{}'\"`]+")
_NAME_SIGILS: Final = ("./", "$", "/", "@")
_NAME_LEADS: Final = frozenset({"use", "run", "invoke"})


def _named_skill(message: str) -> str | None:
    """A Jev skill's canonical or display name in its INVOCATION form.

    A sigil (`/omh-jev-ask`, `$omh-jev-ask`) anywhere, or a bare name that
    opens the message or follows "use"/"run"/"invoke". A bare name elsewhere
    is a mention: "fix the bug in omh-jev-review-gate" is maintenance on the
    skill, and it keeps its ordinary owner like "fix the bug in
    omh-code-review" does.
    """
    tokens = _NAME_TOKEN.findall(message)
    for index, raw in enumerate(tokens):
        token = raw
        sigiled = False
        for sigil in _NAME_SIGILS:
            if token.startswith(sigil):
                token = token[len(sigil):]
                sigiled = True
                break
        if token.startswith("omh-"):
            token = token[len("omh-"):]
        if token not in JEV_SKILL_NAMES:
            continue
        if sigiled or index == 0 or tokens[index - 1] in _NAME_LEADS:
            return token
    return None


def _spoken(normalized: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase_is_spoken(normalized, normalized_phrase(phrase)) for phrase in phrases)


def _declines_jev(normalized: str) -> bool:
    """A configuration, a negation, a description, or Jev among options."""
    return _spoken(normalized, JEV_NOT_ADDRESSING_PHRASES) or bool(_JEV_AMONG_OPTIONS.search(normalized))


def addresses_jev(message: str) -> bool:
    normalized = normalized_phrase(message)
    if _declines_jev(normalized):
        return False
    if _spoken(normalized, JEV_ADDRESSING_PHRASES) or normalized.startswith("jev,"):
        return True
    return any(
        normalized.startswith(lead + imperative)
        for lead in _IMPERATIVE_LEADS
        for imperative in JEV_OPENING_IMPERATIVES
    )


def jev_addressed_skill(message: str, names: Collection[str], query: RoutingQuery | None = None) -> str | None:
    """The `jev-*` skill this message invokes, or None. See the module docstring.

    `query`, when given, is `message` already prepared (`query.raw == message`);
    the partner scorer reads it instead of rebuilding the prep chain.
    """
    if "jev" not in message.casefold():
        return None
    normalized = normalized_phrase(message)
    named = _named_skill(normalized)
    if named and named in names:
        return named
    if _declines_jev(normalized):
        return None
    # A skill's own phrase names the job and the addressee at once ("jev done
    # check"), so it needs no separate addressing phrase.
    for skill, phrases in JEV_SKILL_PHRASES.items():
        if skill in names and _spoken(normalized, phrases):
            return skill
    if not addresses_jev(message):
        return None
    for skill in _CUE_ORDER:
        if skill in names and _spoken(normalized, JEV_SKILL_CUES[skill]):
            return skill
    from .recommend import confident_scored_field_winner

    partner = confident_scored_field_winner(query or message)
    sibling = JEV_SIBLING_BY_PARTNER.get(partner)
    if sibling and sibling in names:
        return sibling
    sibling = _lexical_sibling(message, names)
    if sibling:
        return sibling
    return JEV_ASK if JEV_ASK in names else None


# A sibling picked by word overlap alone sends the user's data to a preset
# that judges one kind of thing, so it needs both an absolute score (several
# content words shared with that sibling's situations and triggers, not one)
# and a clear lead over the next sibling. Anything less stays with `jev-ask`,
# the preset that assumes nothing about the question.
JEV_LEXICAL_SIBLING_FLOOR: Final = 9.0
JEV_LEXICAL_SIBLING_MARGIN: Final = 5.0


def _lexical_sibling(message: str, names: Collection[str]) -> str | None:
    """The one non-ask sibling this message describes, ranked on catalog text.

    `jev` itself is dropped from the query: every sibling carries it, so it
    only raises all five scores together.
    """
    from .lexical_shortlist import lexical_ranking

    ranked = [
        (skill, score)
        for skill, score in lexical_ranking(message, frozenset({"jev"}))
        if skill in JEV_SKILL_NAMES and skill != JEV_ASK and skill in names
    ]
    if not ranked or ranked[0][1] < JEV_LEXICAL_SIBLING_FLOOR:
        return None
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if ranked[0][1] - runner_up < JEV_LEXICAL_SIBLING_MARGIN:
        return None
    return ranked[0][0]


__all__ = [
    "JEV_ADDRESSING_PHRASES",
    "JEV_NOT_ADDRESSING_PHRASES",
    "JEV_OPENING_IMPERATIVES",
    "JEV_SIBLING_BY_PARTNER",
    "JEV_SKILL_CUES",
    "JEV_SKILL_NAMES",
    "JEV_SKILL_PHRASES",
    "addresses_jev",
    "jev_addressed_skill",
]
