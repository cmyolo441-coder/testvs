"""One message's routing stages, prepared once.

`RoutingQuery.from_message` runs the router's text-prep chain in order and
keeps every stage by name, so a reader takes the stage it needs instead of
re-deriving the chain at its call site:

1. `raw` -- the message as received.
2. `executable` -- `executable_routing_text(raw)`: quoted and reference
   regions removed. `has_strong_named_catalog_owner` reads it for the
   negated-name check.
3. `scrubbed` -- `scrub_diagnostic_status_text(executable)`: diagnostic status
   lines removed. The scorer passes it to `explicit_skill_invocation` and to
   `_score_definition` as `routing_query`.
4. `routing_text` -- `prepare_routing_text(_strip_path_like_fragments(scrubbed))`:
   path-like fragments neutralized, then locale aliases applied.
5. `normalized` -- `normalized_phrase(routing_text.scoring_text)`: what every
   phrase trigger, guard, and offers-itself predicate matches against.
6. `tokens` -- the router's `_tokens(normalized)`: the token set the scorer
   and the guards intersect with. Frozen here; the scorer mutates sets it
   derives from its token argument, so callers hand it `set(tokens)`.

Readers that need an earlier stage alone (prepare-only or executable-only
call sites in `chat.py`, `playbooks.py`, `route_plan.py`) are different
stages, not copies of this chain.
"""

from __future__ import annotations

from dataclasses import dataclass

from .intent import scrub_diagnostic_status_text
from .localization import RoutingText, normalized_phrase, prepare_routing_text
from .policy import everyday_sense_phrase_unanchored
from .reference_regions import executable_routing_text
from ..plugin_bundle.omh.hermes_session_commands import LOOP_TIMER_NONE, loop_timer_verdict


@dataclass(frozen=True)
class RoutingQuery:
    raw: str
    executable: str
    scrubbed: str
    routing_text: RoutingText
    normalized: str
    tokens: frozenset[str]
    # What a loop on a timer in this text is (`hermes_session_commands`), read
    # on `executable`: `normalized` joins every whitespace run, which erases
    # the block ends a messaging gateway's own prefixes close with.
    loop_timer: str = LOOP_TIMER_NONE
    # Whether the masker removed quoted, code, link or relayed text from the
    # message: here (`executable` differs from `raw`), or before, when the
    # caller masked the text it passes as `raw` and says so. What the removed
    # text named is unread, so a predicate that reads where data goes must
    # not prove a place inside (`recommend._SHIELDED_OFFERS_ITSELF`).
    shielded: bool = False

    @classmethod
    def from_message(cls, text: str, *, shielded: bool = False) -> RoutingQuery:
        # `recommend.py` imports this module, so its helpers load late.
        from .recommend import _strip_path_like_fragments, _tokens

        executable = executable_routing_text(text)
        scrubbed = scrub_diagnostic_status_text(executable)
        routing_text = prepare_routing_text(_strip_path_like_fragments(scrubbed))
        normalized = normalized_phrase(routing_text.scoring_text)
        return cls(
            raw=text,
            executable=executable,
            scrubbed=scrubbed,
            routing_text=routing_text,
            normalized=normalized,
            tokens=frozenset(_tokens(normalized)),
            loop_timer=loop_timer_verdict(executable),
            shielded=shielded or executable != text,
        )

    @classmethod
    def coerce(cls, value: str | RoutingQuery) -> RoutingQuery:
        if isinstance(value, RoutingQuery):
            return value
        return cls.from_message(value)

    def offers_itself_withheld(self, skill: str) -> bool:
        """True when `skill` has an offers-itself precondition and this query fails it.

        `_score_definition` drops such a skill before scoring; a reader that
        ranks the catalog another way asks here so it drops the same skill.
        """
        from .recommend import _offers_itself_predicate

        offers_itself = _offers_itself_predicate(skill, shielded=self.shielded)
        if offers_itself is None:
            return False
        return not offers_itself(self.normalized, set(self.tokens))

    def everyday_sense_withheld(self, skill: str) -> bool:
        """True when `skill` shares only an everyday-English phrase with this query.

        `_score_definition` drops such a skill before scoring (see
        `EVERYDAY_SENSE_PHRASES` in `policy.py`); a reader that ranks the
        catalog another way, or matches the phrase on a fast path, asks here
        so it drops the same skill.
        """
        return everyday_sense_phrase_unanchored(skill, self.normalized)
