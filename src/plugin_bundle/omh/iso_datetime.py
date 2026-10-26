"""``datetime.fromisoformat`` that refuses hour 24 on every supported Python.

CPython 3.14 reads ``2026-10-24T24:00:00`` as midnight of the next day, while
3.11 to 3.13 raise ``hour must be in 0..23``. It is the only input 3.13 and
3.14 parse differently, and the only one any of them accepts where another
refuses. (3.13 also started keeping a sub-second-only UTC offset such as
``+00:00:00.5`` that 3.11 and 3.12 drop; that moves a value by under a second
and is left alone here.) Hermes' gateway loads this bundle on 3.14 and
``omh`` runs on whatever Python its venv was built with, so without this guard
one stored stamp can be valid to one process and invalid to the other.

Stdlib only: Hermes imports the bundle without ``omh`` on its path, and the
``omh`` package imports this module from the bundle.
"""

from __future__ import annotations

from datetime import date, datetime, time


def strict_fromisoformat(text: str) -> datetime:
    """``datetime.fromisoformat(text)``, refusing hour 24 as 3.11 to 3.13 do."""
    parsed = datetime.fromisoformat(text)
    # Hour 24 is accepted only as 24:00:00 exactly, and it moves the date one
    # day past the one the text names. A real midnight starts with its own date.
    if parsed.time() == time(0) and not text.startswith(_date_spellings(parsed.date())):
        raise ValueError("hour must be in 0..23")
    return parsed


def _date_spellings(day: date) -> tuple[str, ...]:
    """Every way ``fromisoformat`` accepts ``day`` written as the start of a text."""
    week_year, week, weekday = day.isocalendar()
    spellings = (
        f"{day.year:04d}-{day.month:02d}-{day.day:02d}",
        f"{day.year:04d}{day.month:02d}{day.day:02d}",
        f"{week_year:04d}-W{week:02d}-{weekday}",
        f"{week_year:04d}W{week:02d}{weekday}",
    )
    if weekday == 1:
        # A week without a weekday (``2026-W43``) names its Monday.
        spellings += (f"{week_year:04d}-W{week:02d}", f"{week_year:04d}W{week:02d}")
    return spellings
