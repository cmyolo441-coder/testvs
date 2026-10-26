"""Deterministic segmentation of a compound outcome request (issue #816).

A compound request asks for more than one outcome in one message: "research the
payment providers, write the migration plan, and implement the winner". The
router scores one message against one skill and returns the single best match,
so handing it that whole sentence returns `research` and silently drops the
other two outcomes. Composing a multi-step workflow needs the parts.

Why a positional split rather than `contains_cue_phrase`
--------------------------------------------------------

`contains_cue_phrase` answers whether a declared phrase is present. It cannot
say where, and it is deliberately unanchored: it matches `and` inside
`understand`. A splitter needs both properties it lacks -- the offsets, and word
boundaries so a connector living inside a word is not treated as a separator.
So the connector table is declared here the way every other cue table in this
package is declared, and it is matched with a boundary-anchored pattern built
from that same table rather than with ad hoc `in` checks scattered through the
code. The two token helpers keep the parts they own: `routing_tokens` decides
whether a fragment carries enough signal to be a segment at all, and
`normalized_phrase` collapses fragments that differ only by case or Unicode
form.

Splitting is deliberately generous and the verdict is deliberately strict.
Over-splitting costs one extra cached recommendation lookup per fragment;
under-splitting loses an outcome the user asked for. Whether the request is
actually compound is not decided here -- that needs the capability families the
fragments resolve to, which lives in `omh.workflows.workflow_composition`.

Language
--------

The connectors are English only, following the Routing Language Policy in
`docs/DIRECTION.md`: English is the precision target, and per-language trigger
tables do not scale to the languages a global product must serve. A Korean or
Japanese compound request therefore reads as one segment here, which surfaces
downstream as `not_compound` -- a visible gap rather than a wrong composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re

from .domain_signals import DomainRouteSignal, specialist_domain_route_signals
from .localization import normalized_phrase, routing_tokens


# Sequencing and coordination cues that separate one requested outcome from the
# next. Kept to cues that join clauses; words that merely sequence inside a
# clause ("next sprint", "plus one") are left out because they separate nothing.
COMPOUND_SEGMENT_CONNECTORS = (
    "after that",
    "afterwards",
    "and also",
    "and then",
    "and",
    "as well as",
    "before that",
    "finally",
    "followed by",
    "once that is done",
    "then",
)

# Punctuation that separates clauses. A newline counts because chat surfaces
# turn a bulleted ask into one message with embedded newlines.
_PUNCTUATION_CLASS = r"[,;\n]"

# Longest connector first: a single alternation tries its branches left to
# right at each position, so `and` listed before `and then` would consume the
# `and` and strand `then` at the head of the following segment.
_CONNECTOR_PATTERN = re.compile(
    "|".join(
        [
            _PUNCTUATION_CLASS,
            *(
                rf"\b{re.escape(phrase)}\b"
                for phrase in sorted(COMPOUND_SEGMENT_CONNECTORS, key=len, reverse=True)
            ),
        ]
    ),
    re.IGNORECASE,
)

# Trailing/leading punctuation and dashes carry no routing signal, and leaving
# them on changes the normalized key two identical fragments hash to.
_SEGMENT_TRIM = " \t\r\n.!?:-–—"


@dataclass(frozen=True)
class CompoundRequestSegments:
    """The outcome fragments one request stated, in the order it stated them.

    `segments` holds at most one entry for a single-outcome request and is
    empty only when the request carries no routable token at all. `connectors`
    records which declared separators were actually matched, so a reader can
    see why the request split the way it did.
    """

    segments: tuple[str, ...]
    connectors: tuple[str, ...]

    @property
    def segmented(self) -> bool:
        return len(self.segments) > 1


def compound_request_segments(message: str) -> CompoundRequestSegments:
    """Split one request into the outcome fragments it asked for."""
    return _compound_request_segments_cached(message.strip())


def distinct_complete_domain_signals(message: str) -> tuple[DomainRouteSignal, ...]:
    """Return distinct specialist domains expressed as complete segments."""
    segments = compound_request_segments(message)
    if not segments.segmented:
        return ()
    signals: list[DomainRouteSignal] = []
    seen: set[str] = set()
    for segment in segments.segments:
        for signal in specialist_domain_route_signals(segment):
            if signal.skill in seen:
                continue
            seen.add(signal.skill)
            signals.append(signal)
    return tuple(signals)


@lru_cache(maxsize=2048)
def _compound_request_segments_cached(text: str) -> CompoundRequestSegments:
    if not text:
        return CompoundRequestSegments((), ())
    connectors = tuple(
        sorted(
            {
                folded
                for match in _CONNECTOR_PATTERN.finditer(text)
                if (folded := normalized_phrase(match.group(0)).strip())
            }
        )
    )
    fragments: list[str] = []
    seen: set[str] = set()
    for raw in _CONNECTOR_PATTERN.split(text):
        candidate = raw.strip(_SEGMENT_TRIM)
        if not candidate or not routing_tokens(candidate):
            continue
        key = normalized_phrase(candidate)
        if key in seen:
            continue
        seen.add(key)
        fragments.append(candidate)
    if len(fragments) > 1:
        return CompoundRequestSegments(tuple(fragments), connectors)
    # One surviving fragment, or none: the connectors that matched did not
    # separate two token-bearing asks, so report the request whole rather than
    # a truncated piece of it.
    whole = text.strip(_SEGMENT_TRIM)
    return CompoundRequestSegments((whole,) if routing_tokens(whole) else (), connectors)
