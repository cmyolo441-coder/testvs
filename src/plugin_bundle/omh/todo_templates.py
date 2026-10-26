"""Named phase templates a plan record can be stamped with.

A code story has a shape the person already knows: the change is described,
built, reviewed, tested, fixed, then explained three ways and closed. Nothing
made that shape the DEFAULT shape of a plan. Each session invented its phases
again, and a phase nobody thought of that day cost nothing to leave out --
which is the same thing as not having the shape at all.

A template is the shape written down once, in this module, and stamped onto
the record by name. ``build_todo_record`` fills the phases from here when a
writer asks for the template and sends no items, and holds every later write
to the same coverage: every template phase present, each one first appearing
in template order, each item inside one of them. First appearance, not
position, so a plan that returns to an earlier phase for one more task is not
refused for it. The stamp is a record field (``template``), so a
reader answers "is this a story plan" by reading a field rather than by
matching the phase labels back out of prose -- the rule this repository
already applies to every other plan judgement (`recorded_blocked_reason`).

What the template deliberately does NOT do is decide when it applies. No
detector here reads a person's message: the stamp arrives because a writer
passed ``template`` on an ``omh_todo`` call, and a plan declared without one
is exactly the plan it was before this module existed.

**Skipping a phase.** A change with no UI owes no manual test guide, so a
phase must be droppable -- and dropping it must cost something, or the shape
is advisory again. The cost is that the phase cannot LEAVE the list: the
coverage rule below refuses a stamped plan that is missing one, naming it. A
phase this change does not need is still declared, marked ``done`` and
carrying a ``blocked_reason`` that says why it produced nothing.

``done`` rather than a fourth item state, for the reason `todo_store` records
where it chose a ``blocked_reason`` field over a ``blocked`` state: the
counts, the HUD projection and the widget keep reading three states, and a
record carrying a state value an installed reader does not know renders
nothing at all. ``done`` is also the only state that lets the plan finish --
``next_open_item`` walks past done items, while a ``pending`` item carrying a
reason is the plan's own stop criterion and would halt the run at the skipped
phase with every later phase unreached.

What this module ENFORCES and what it merely ASKS FOR are different, and the
line is worth stating because the enforcement is the narrower of the two.
Enforced: the phase cannot leave the list -- ``template_coverage_error`` reads
only the phase labels, and a stamped plan missing one is refused by name.
Asked for: that the phase left standing carries a reason. Nothing checks it,
and nothing can, because a phase that was genuinely worked is also ``done``
with no reason, so "a done phase needs a reason" is not a rule any validator
can hold. The reason is carried by the tool description and by the refusal
text; writing it is the writer's own discipline.

The same boundary one step further out: no record can see a phase marked done
by a writer that simply did not do it. OMH observes no work; it observes
declarations. What the template buys is a phase that is impossible to forget
and impossible to remove without the removal being refused -- and that is the
whole of what a record can enforce.
"""
from __future__ import annotations

from typing import Any, Iterable

# The ten phases of a code story, in delivery order, with the item text each
# phase is declared with when the writer asks for the template and sends no
# items of its own. Labels are fixed English: they are the record's vocabulary
# and every coverage message quotes them back.
CODE_STORY_TEMPLATE = "code-story"

_CODE_STORY_PHASES: tuple[tuple[str, str], ...] = (
    ("I. Story", "Write the story: what changes, for whom, and what done looks like"),
    ("II. Implement", "Implement the change"),
    ("III. Review", "Review the change against this repository's standards"),
    ("IV. QA", "Run the tests and gates that prove the change"),
    ("V. Fix", "Fix what review and QA found"),
    ("VI. Manual test guide", "Write the manual test guide a person can follow"),
    ("VII. Deep guide", "Write the deep guide: how the change works and why"),
    ("VIII. ELI5", "Explain the change in plain language"),
    ("IX. Quiz", "Write the comprehension quiz for the change"),
    ("X. Close", "Close the story: land the change and report the observed evidence"),
)

TODO_TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    CODE_STORY_TEMPLATE: _CODE_STORY_PHASES,
}

# What a refusal tells the writer to do instead of dropping the phase. One
# sentence, because it is appended to a message that already names the phase.
SKIP_INSTRUCTION = (
    "keep it and mark it state=done with a blocked_reason saying why it does not apply"
)


def template_names() -> tuple[str, ...]:
    return tuple(sorted(TODO_TEMPLATES))


def template_phase_labels(name: str) -> tuple[str, ...]:
    """The phase labels ``name`` declares, in delivery order."""
    return tuple(label for label, _ in TODO_TEMPLATES[name])


def template_items(name: str) -> list[dict[str, Any]]:
    """One pending item per phase of ``name``, ready for ``validate_todo_items``."""
    return [
        {"text": text, "state": "pending", "phase": label}
        for label, text in TODO_TEMPLATES[name]
    ]


def effective_item_phases(items: Iterable[dict[str, Any]]) -> list[str]:
    """Each item's phase, with subtasks inheriting the section above them.

    The same walk `runtime_reader._todo_summary` makes when it decides which
    phase an item renders under: a named phase opens a section, a nested item
    (``depth`` above 0) continues the one it sits in, and a top-level item
    with no phase closes the section. Written here as well because the
    coverage check has to agree with what the HUD will show -- a plan whose
    items validate into phases the panel then draws differently would be a
    stamp that means one thing to the store and another to the person.
    """
    inherited = ""
    resolved: list[str] = []
    for item in items:
        name = str(item.get("phase", "") or "")
        if name:
            inherited = name
        elif int(item.get("depth", 0) or 0) == 0:
            inherited = ""
        resolved.append(inherited)
    return resolved


def template_coverage_error(name: str, items: list[dict[str, Any]]) -> str:
    """Why ``items`` do not satisfy template ``name``, or ``""``.

    Order is checked over FIRST APPEARANCES, not positions: the phases are
    de-duplicated in the order they are first named and that sequence must
    equal the template's. So a plan that comes back to an earlier phase for
    one more task passes, which is the behaviour a run wants -- returning to
    `III. Review` after `X. Close` opened is a plan doing its job, not a plan
    declaring its phases wrongly.

    Returns a message rather than raising so the one caller
    (``build_todo_record``) keeps raising the store's own error type, and so
    this module stays readable by a test that wants the sentence.
    """
    required = template_phase_labels(name)
    resolved = effective_item_phases(items)
    if any(not phase for phase in resolved):
        loose = next(
            index for index, phase in enumerate(resolved, start=1) if not phase
        )
        return (
            f"todo template {name!r} needs every item inside one of its phases; "
            f"item {loose} names none. Give it a phase, or nest it under one with depth."
        )
    declared: list[str] = []
    for phase in resolved:
        if phase not in declared:
            declared.append(phase)
    unknown = [phase for phase in declared if phase not in required]
    if unknown:
        return (
            f"todo template {name!r} does not have the phase(s) {', '.join(repr(p) for p in unknown)}; "
            f"its phases are {', '.join(repr(p) for p in required)}."
        )
    missing = [phase for phase in required if phase not in declared]
    if missing:
        return (
            f"todo template {name!r} is missing the phase(s) "
            f"{', '.join(repr(p) for p in missing)}. A phase this change does not need is "
            f"not dropped: {SKIP_INSTRUCTION}."
        )
    if declared != list(required):
        return (
            f"todo template {name!r} declares its phases out of order "
            f"({', '.join(repr(p) for p in declared)}); the order is "
            f"{', '.join(repr(p) for p in required)}."
        )
    return ""
