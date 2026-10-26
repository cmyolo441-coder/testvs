"""Whether a route answer's own numbers agree with each other.

Each probability in an answer is validated on its own when it is recorded
(`route_answer_store`). That says nothing about whether the answer, read as a
whole, is one distribution over the question it claims to answer. Three shapes
are refused here, and an answer in any of them is an ``invalid_answer``:

- ``coverage`` -- the Choice distribution omits an option the question offered.
  A distribution over some of the options is a distribution over a different
  question.
- ``mass`` -- the Choice probabilities do not sum to one within
  `PROBABILITY_MASS_TOLERANCE`.
- ``argmax`` -- `route_choice` is not the option with the highest probability,
  or is not in the distribution at all. An answerer that picks one option
  while putting more mass on another has said two different things.

An ``invalid_answer`` means "no opinion". It is counted, so its rate is
readable, and it is never scored as agreeing or disagreeing with the router.
It is never a reason for the caller to stop: the deterministic route was in
force before the answer and stays in force after it.

An answer that carries no Choice probabilities has no distribution to
contradict and is not checked. The field is optional on purpose: an invented
number is worse than an absent one.

Stdlib only; the core scorer imports this module so the live record and the
offline score apply one rule.
"""

from __future__ import annotations

from typing import Final, Iterable, Mapping

INVALID_ANSWER_VERDICT: Final = "invalid_answer"
INVALID_ANSWER_COVERAGE: Final = "coverage"
INVALID_ANSWER_MASS: Final = "mass"
INVALID_ANSWER_ARGMAX: Final = "argmax"
INVALID_ANSWER_REASONS: Final = (INVALID_ANSWER_COVERAGE, INVALID_ANSWER_MASS, INVALID_ANSWER_ARGMAX)

# How far the Choice probabilities may sum from one. Two decimal places is the
# precision an answerer reporting rounded percentages can honestly hit; a
# distribution further off than this is not a rounding artefact.
PROBABILITY_MASS_TOLERANCE: Final = 0.01
# Ties at the maximum are not a contradiction; float noise between them is not
# either.
_ARGMAX_TOLERANCE: Final = 1e-9


def invalid_answer_reasons(
    route_choice: object,
    probabilities: Mapping[str, object] | None,
    *,
    options: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """The contradictions in one answer, in `INVALID_ANSWER_REASONS` order.

    `options` are the Choice options of the question the answer is joined to.
    Without them coverage cannot be judged and is not reported; mass and argmax
    need only the answer itself.
    """
    if not isinstance(probabilities, Mapping) or not probabilities:
        return ()
    numbers: dict[str, float] = {}
    for key, value in probabilities.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            # A non-number is a malformed row, which the readers already
            # report; it is not this module's verdict to give.
            return ()
        numbers[str(key)] = float(value)
    reasons: list[str] = []
    if options is not None and any(str(option) not in numbers for option in options):
        reasons.append(INVALID_ANSWER_COVERAGE)
    if abs(sum(numbers.values()) - 1.0) > PROBABILITY_MASS_TOLERANCE:
        reasons.append(INVALID_ANSWER_MASS)
    choice = str(route_choice or "").strip()
    if choice not in numbers or numbers[choice] < max(numbers.values()) - _ARGMAX_TOLERANCE:
        reasons.append(INVALID_ANSWER_ARGMAX)
    return tuple(reasons)


__all__ = [
    "INVALID_ANSWER_ARGMAX",
    "INVALID_ANSWER_COVERAGE",
    "INVALID_ANSWER_MASS",
    "INVALID_ANSWER_REASONS",
    "INVALID_ANSWER_VERDICT",
    "PROBABILITY_MASS_TOLERANCE",
    "invalid_answer_reasons",
]
