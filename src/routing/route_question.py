"""One typed question shape for a route OMH cannot decide on its own.

A deterministic router either has enough signal or it does not. When it does
not, the thing it needs answered is not "run the skill" -- it is two different
questions, and keeping them apart is the whole point of this module:

- **Which** workflow, if any, is the best fit among the candidates. That is a
  relative judgment over a shortlist, so it is one Choice with an explicit
  ``none`` option. A Choice always returns one of its options, so without that
  option the answer "no workflow applies" cannot be expressed at all.
- **Whether** each candidate actually fits. That is an absolute judgment about
  one workflow, so it is one yes/no question per candidate, each answerable
  without reference to the others.

The two carry no structural invariant between them: a Choice over options and
one yes/no per option answer different questions, and an answerer may pick a
candidate in the Choice while saying no to every fit. That is not a
contradiction to resolve here. This module builds the questions; a scorer
decides what a set of answers means.

Nothing in this module performs I/O, reads configuration, or names any
answerer. The same block is built for a live undecidable route and for an
offline corpus item, which is why it lives here rather than beside either
consumer.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from ..plugin_bundle.omh.route_question_mode import ROUTE_QUESTION_MODE_OFF

ROUTE_QUESTION_SCHEMA_VERSION = "route_question/v1"

# The question ids. `ROUTE_CHOICE_KEY` is the single relative choice;
# `FIT_QUESTION_PREFIX` + skill name is the absolute per-candidate question.
ROUTE_CHOICE_KEY = "route_choice"
FIT_QUESTION_PREFIX = "fits::"

# The option that lets a Choice answer "no workflow applies". A Choice returns
# one of its options and nothing else, so an answerer without this option is
# forced to name a workflow it may have just rejected.
NO_WORKFLOW_OPTION = "none"
NO_WORKFLOW_DESCRIPTION = "No OMH workflow applies; answer the request directly."

# Flat, editorial defaults. They are deliberately not derived from the router's
# own confidence band: the question exists precisely because that band was too
# low to act on, so scaling the acceptance bar with it would set the loosest
# bar exactly where the router is least sure. Flat keeps the bar the same
# whatever produced the question, and an operator raises it for a surface where
# a wrong dispatch costs more.
FITS_DISPATCH_THRESHOLD = 0.8
FITS_CLARIFY_THRESHOLD = 0.5

# Catalog descriptions carry this product prefix. It is stripped from every
# option and question so the prefix cannot become a cue of its own.
_DESCRIPTION_PREFIX = "[omh] "

_CHOICE_INSTRUCTIONS = (
    "Which OMH workflow is the best fit for this request? "
    "This is a relative choice among the listed options: pick the closest fit, "
    f"or `{NO_WORKFLOW_OPTION}` when the request asks for none of them."
)

_CLAIM_BOUNDARY = (
    "A route question is a prepared question about one request, not a routing "
    "decision, an execution, or evidence that any workflow ran. An unanswered "
    "question changes nothing: the deterministic route stays in force."
)


# Why a built question has nothing to decide, in the order they are tested.
# `route_question_decline_reason` in `omh.routing.chat` is the one producer;
# these are the only values it returns besides "".
DECLINE_ACKNOWLEDGEMENT = "acknowledgement"
DECLINE_ONE_WORD_REPLY = "one_word_reply"
DECLINE_NO_CANDIDATE = "no_candidate"
DECLINE_SINGLE_CANDIDATE = "single_candidate"
ROUTE_QUESTION_DECLINE_REASONS = (
    DECLINE_ACKNOWLEDGEMENT,
    DECLINE_ONE_WORD_REPLY,
    DECLINE_NO_CANDIDATE,
    DECLINE_SINGLE_CANDIDATE,
)


def route_question_candidate_count(question: Mapping[str, Any] | None) -> int:
    """How many workflows a question's Choice offers, `none` not counted.

    Read from the question block itself, because that block is what an
    answerer is shown: a count taken from anywhere else could describe a
    shortlist the question does not carry.
    """
    questions = question.get("questions") if isinstance(question, Mapping) else None
    choice = questions.get(ROUTE_CHOICE_KEY) if isinstance(questions, Mapping) else None
    options = choice.get("options") if isinstance(choice, Mapping) else None
    if not isinstance(options, Mapping):
        return 0
    return sum(1 for option in options if option != NO_WORKFLOW_OPTION)


def apply_route_question_mode(route: dict[str, Any], mode: str) -> dict[str, Any]:
    """The route a surface hands out, given the configured mode.

    `off` withholds the question; every other mode leaves the route exactly
    as the router built it. `shadow` is today's behaviour, and `unknown` -- a
    mode the process could not read -- is deliberately not treated as `off`:
    an unreadable switch changes nothing the surface does, and the record
    that carries `unknown` is where the operator finds out it was unreadable.
    The route is modified in place and returned, so a caller that owns a
    fresh payload does not pay for a copy.
    """
    if mode == ROUTE_QUESTION_MODE_OFF:
        route.pop("route_question", None)
    return route


def clean_skill_description(value: object) -> str:
    """Strip the catalog's product prefix from a skill description."""
    text = str(value or "").strip()
    if text.startswith(_DESCRIPTION_PREFIX):
        text = text[len(_DESCRIPTION_PREFIX):].strip()
    return text


def _candidate_rows(candidates: Iterable[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """Return (skill, description) pairs in candidate order, first mention wins."""
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for candidate in candidates or ():
        if not isinstance(candidate, Mapping):
            continue
        skill = str(candidate.get("skill") or "").strip()
        if not skill or skill == NO_WORKFLOW_OPTION or skill in seen:
            continue
        seen.add(skill)
        rows.append((skill, clean_skill_description(candidate.get("description"))))
    return rows


def message_digest(message: str) -> str:
    """The one producer of the message digest every route surface reports.

    `routing_record_payload` publishes this same value under `message_sha256`,
    and so does the chat interaction payload. It is named here because three
    things now have to agree on it -- a corpus item, the question digest built
    from it, and a recorded answer joined back by it -- and a fourth inline
    `hashlib` call is how they would stop agreeing. A test pins this against
    the routing record's own field.
    """
    return hashlib.sha256(str(message).encode("utf-8")).hexdigest()


def normalized_route_candidates(candidates: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """The candidate rows a question is built from, as the question carries them.

    A caller that wants to show the same shortlist it asked about -- a prompt,
    a corpus item -- reads it from here rather than rebuilding it, so what is
    shown and what was digested cannot diverge.
    """
    return [{"skill": skill, "description": description} for skill, description in _candidate_rows(candidates)]


def fit_question_key(skill: str) -> str:
    """The absolute per-candidate question id for one skill."""
    return f"{FIT_QUESTION_PREFIX}{skill}"


def fit_question_skill(key: str) -> str:
    """The skill a fit question id names, or "" when the id is not one."""
    text = str(key or "")
    if not text.startswith(FIT_QUESTION_PREFIX):
        return ""
    return text[len(FIT_QUESTION_PREFIX):]


def route_question_digest(*, message_sha256: str, candidates: Iterable[Mapping[str, Any]]) -> str:
    """The digest that joins one recorded answer back to one question.

    The message digest is in it, not only the shortlist. A shortlist is the
    router's top few skills, and unrelated requests share one constantly: over
    the two shipped routing corpora a digest built from the questions alone
    names a group of hundreds of requests rather than a request. An answer
    recorded against such a digest can only be joined to one of them, so a
    reader that takes the first charges the answerer with whatever the others
    expected. A digest without the message is that defect, which is why the
    message is required rather than optional here.
    """
    rows = _candidate_rows(candidates)
    canonical = json.dumps(
        {
            "message_sha256": str(message_sha256),
            "candidates": [{"skill": skill, "description": description} for skill, description in rows],
            "schema_version": ROUTE_QUESTION_SCHEMA_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_route_question_from_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    message_sha256: str,
    reasons: Iterable[str] = (),
    digest: str = "",
) -> dict[str, object]:
    """Build the typed question for one undecidable route.

    `candidates` are mappings carrying `skill` and `description`; a candidate
    without a skill name, or one named `none`, is dropped, and a repeated skill
    keeps its first description. An empty candidate list is legal and yields a
    Choice whose only option is `none` -- that is the honest shape when the
    router found nothing, and it still lets an answerer disagree by fitting
    nothing.

    `message_sha256` identifies the request this question was built for. It has
    no default: a question digest that does not carry it identifies a shortlist
    instead of a question, and `route_question_digest` says what that costs a
    reader joining answers back.

    `reasons` are the router's own words for why the route was undecidable and
    are carried through unchanged. `digest` joins an answer back to the exact
    question it answered; when it is empty the digest is derived from the
    message and the candidates, so every block has one.
    """
    if not str(message_sha256 or "").strip():
        raise ValueError("a route question must name the message it was built for")
    rows = _candidate_rows(candidates)
    options: dict[str, str] = {skill: description for skill, description in rows}
    options[NO_WORKFLOW_OPTION] = NO_WORKFLOW_DESCRIPTION
    questions: dict[str, object] = {
        ROUTE_CHOICE_KEY: {
            "type": "choice",
            "instructions": _CHOICE_INSTRUCTIONS,
            "options": options,
        }
    }
    for skill, description in rows:
        detail = f" `{skill}`: {description}" if description else ""
        questions[fit_question_key(skill)] = {
            "type": "noul",
            "instructions": (
                f"Does this request ask for the work `{skill}` does?{detail} "
                "Answer for this workflow alone, independently of the others."
            ),
        }
    return {
        "schema_version": ROUTE_QUESTION_SCHEMA_VERSION,
        "question_digest": str(digest)
        or route_question_digest(
            message_sha256=message_sha256,
            candidates=[{"skill": skill, "description": description} for skill, description in rows],
        ),
        "questions": questions,
        "thresholds": {
            "fits_dispatch": FITS_DISPATCH_THRESHOLD,
            "fits_clarify": FITS_CLARIFY_THRESHOLD,
        },
        "reasons": [str(reason) for reason in reasons if str(reason).strip()],
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def build_route_question_for_candidate_handoff(
    candidate_handoff: Mapping[str, Any],
    *,
    message: str,
) -> dict[str, object]:
    """Build the question for an undecidable route, from the handoff alone.

    This is the whole call shape, in one place, because two callers have to
    produce byte-identical inputs or the thing the digest exists for stops
    working: the live route attaches a question to an undecidable route, and
    the offline corpus projects the same message, and an answer recorded on
    one is scored against the other. Two call sites assembling the same
    arguments by hand is how they drift, and the drift is silent -- the digests
    simply stop matching and every recorded answer lands as unmatched.

    The candidates come from the handoff rather than from `route`'s public
    `recommendations`, which is a compacted projection that drops
    `description`; a question built from it asks about the same skills with
    none of the text that lets an answerer judge them. The reasons are the
    handoff's own machine codes, not the route's prose reason.

    `limit` has no place here. The handoff decided its own shortlist, and a
    corpus that cut it further would ask a different question than the live
    route asks.
    """
    rows = candidate_handoff.get("candidates") if isinstance(candidate_handoff, Mapping) else None
    candidates = [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []
    raw_reasons = candidate_handoff.get("reasons") if isinstance(candidate_handoff, Mapping) else None
    reasons = [str(reason) for reason in raw_reasons] if isinstance(raw_reasons, list) else []
    return build_route_question_from_candidates(
        candidates,
        message_sha256=message_digest(message),
        reasons=reasons,
    )


__all__ = [
    "DECLINE_ACKNOWLEDGEMENT",
    "DECLINE_NO_CANDIDATE",
    "DECLINE_ONE_WORD_REPLY",
    "DECLINE_SINGLE_CANDIDATE",
    "ROUTE_QUESTION_DECLINE_REASONS",
    "apply_route_question_mode",
    "route_question_candidate_count",
    "FITS_CLARIFY_THRESHOLD",
    "FITS_DISPATCH_THRESHOLD",
    "FIT_QUESTION_PREFIX",
    "NO_WORKFLOW_DESCRIPTION",
    "NO_WORKFLOW_OPTION",
    "ROUTE_CHOICE_KEY",
    "ROUTE_QUESTION_SCHEMA_VERSION",
    "build_route_question_for_candidate_handoff",
    "build_route_question_from_candidates",
    "clean_skill_description",
    "fit_question_key",
    "fit_question_skill",
    "message_digest",
    "normalized_route_candidates",
    "route_question_digest",
]
