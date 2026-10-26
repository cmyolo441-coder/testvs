"""One shape for every percentage OMH reports about its own corpora.

A percentage printed without its denominator is a claim nobody can check, and
the two ways OMH got that wrong are both silent:

- ``max(1, count)`` in a divisor turns an empty corpus into a confident
  ``0.0%``. That is a stronger claim than the truth: 0% asserts that every
  case was measured and every case failed, while an empty corpus measured
  nothing at all. This module reports ``percent = None`` with a named basis
  instead, the same way the live-lane receipt reports ``pass_rate = None``
  when nothing was graded.
- A numerator assembled from more than one outcome bucket reads as if it came
  from one. ``covered_percent`` folding model-selection handoffs in with
  resolved routes is fair, but only when the payload says so; explaining it in
  a text formatter leaves every JSON consumer guessing.

So a reported rate carries four things beside the number: what the numerator
summed, what the denominator counted, which observations were excluded before
either was taken, and whether a percentage exists at all.

This is a reporting shape, not a statistic. It computes nothing a caller could
not compute; it refuses to let a caller report the result without saying what
it divided.
"""

from __future__ import annotations

from dataclasses import dataclass

REPORTED_RATE_SCHEMA_VERSION = "omh_reported_rate/v1"

# The basis recorded when the denominator is zero. Named rather than blank so a
# reader can tell "nothing was measured" from "the measurement came out at 0%".
NO_OBSERVATIONS_BASIS = "no_observations"
OBSERVATIONS_BASIS = "observed_cases"


@dataclass(frozen=True, slots=True)
class ReportedRate:
    """A percentage that carries its own denominator and exclusions."""

    numerator: int
    denominator: int
    numerator_of: tuple[str, ...]
    denominator_of: str
    excluded: tuple[str, ...]
    percent: float | None

    @property
    def basis(self) -> str:
        return OBSERVATIONS_BASIS if self.denominator else NO_OBSERVATIONS_BASIS

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": REPORTED_RATE_SCHEMA_VERSION,
            "percent": self.percent,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "numerator_of": list(self.numerator_of),
            "denominator_of": self.denominator_of,
            "excluded": list(self.excluded),
            "basis": self.basis,
        }


def reported_rate(
    *,
    numerator: int,
    denominator: int,
    numerator_of: tuple[str, ...] | list[str],
    denominator_of: str,
    excluded: tuple[str, ...] | list[str] = (),
    digits: int = 1,
) -> ReportedRate:
    """Build a rate that names what it divided.

    `numerator_of` lists the outcome buckets summed into the numerator, so a
    composed numerator cannot be mistaken for a single bucket. `denominator_of`
    names what was counted. `excluded` names every observation class dropped
    before either count was taken -- an exclusion nobody named is an exclusion
    nobody can argue with.

    A zero denominator yields `percent = None`, never `0.0`.
    """
    if numerator < 0 or denominator < 0:
        raise ValueError("reported rate counts must be non-negative")
    if numerator > denominator:
        raise ValueError("reported rate numerator cannot exceed its denominator")
    names = tuple(str(name) for name in numerator_of if str(name).strip())
    if not names:
        raise ValueError("reported rate must name what its numerator counts")
    if not str(denominator_of).strip():
        raise ValueError("reported rate must name what its denominator counts")
    percent = round(numerator * 100 / denominator, digits) if denominator else None
    return ReportedRate(
        numerator=numerator,
        denominator=denominator,
        numerator_of=names,
        denominator_of=str(denominator_of),
        excluded=tuple(str(name) for name in excluded if str(name).strip()),
        percent=percent,
    )


def meets_target(rate: ReportedRate, target_percent: float) -> bool:
    """Return whether a rate clears a target, with an unmeasured rate failing.

    An unmeasured rate is not a met target. The alternative -- treating "no
    cases" as a pass -- is how an empty corpus silently reports success.
    """
    return rate.percent is not None and rate.percent >= target_percent


def format_reported_rate(rate: ReportedRate) -> str:
    """Render a rate as a line that carries its denominator."""
    numerator_of = " + ".join(rate.numerator_of)
    if rate.percent is None:
        head = f"unmeasured ({NO_OBSERVATIONS_BASIS})"
    else:
        head = f"{rate.percent}%"
    line = f"{head} — {rate.numerator}/{rate.denominator} {rate.denominator_of} ({numerator_of})"
    if rate.excluded:
        line += f"; excluded: {', '.join(rate.excluded)}"
    return line


def reported_rate_shape_errors(value: object) -> tuple[str, ...]:
    """Return why a serialized rate payload is not a well-formed reported rate."""
    expected = {
        "schema_version", "percent", "numerator", "denominator",
        "numerator_of", "denominator_of", "excluded", "basis",
    }
    if not isinstance(value, dict) or set(value) != expected:
        return ("reported rate keys are not closed",)
    errors: list[str] = []
    if value.get("schema_version") != REPORTED_RATE_SCHEMA_VERSION:
        errors.append("reported rate schema_version is invalid")
    for field in ("numerator", "denominator"):
        count = value.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            errors.append(f"reported rate {field} must be a non-negative integer")
    names = value.get("numerator_of")
    if not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names):
        errors.append("reported rate numerator_of must name at least one counted bucket")
    denominator_of = value.get("denominator_of")
    if not isinstance(denominator_of, str) or not denominator_of:
        errors.append("reported rate denominator_of must name what was counted")
    excluded = value.get("excluded")
    if not isinstance(excluded, list) or not all(isinstance(name, str) and name for name in excluded):
        errors.append("reported rate excluded must be a list of named exclusions")
    percent = value.get("percent")
    denominator = value.get("denominator")
    if denominator == 0:
        if percent is not None:
            errors.append("reported rate over an empty denominator must report percent null, never a number")
        if value.get("basis") != NO_OBSERVATIONS_BASIS:
            errors.append(f"reported rate over an empty denominator must state basis {NO_OBSERVATIONS_BASIS}")
    else:
        if isinstance(percent, bool) or not isinstance(percent, (int, float)):
            errors.append("reported rate percent must be a number when the denominator is non-empty")
        if value.get("basis") != OBSERVATIONS_BASIS:
            errors.append(f"reported rate over a non-empty denominator must state basis {OBSERVATIONS_BASIS}")
    return tuple(errors)


__all__ = [
    "NO_OBSERVATIONS_BASIS",
    "OBSERVATIONS_BASIS",
    "REPORTED_RATE_SCHEMA_VERSION",
    "ReportedRate",
    "format_reported_rate",
    "meets_target",
    "reported_rate",
    "reported_rate_shape_errors",
]
