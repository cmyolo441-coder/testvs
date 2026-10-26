"""Plural design directions, and a static preview a person can actually look at.

`design_orchestration/v1` can hold exactly one hierarchy, one palette, one
typography, one layout, and one signature element, and its validator re-derives
the whole object and rejects anything not byte-identical - so it is closed
against extension by construction. Every CLI flag behind it is required, which
means the command cannot be invoked until the answer is already known. It can
record a decision; it can never be the thing that elicits one.

This module is the plural sibling: 2-4 directions offered together with a
`chosen_option` slot that starts empty. `render_design_direction_set_html`
turns that into one self-contained HTML file with the options side by side.

On the boundary: this writes a file, and nothing else. No socket is bound, no
port is opened, no browser is launched, no network call is made - the operator
opens the file themselves. That is the same shape as every other artifact OMH
writes, and it is deliberately not the shape of the brainstorm server this idea
came from, which binds a port and shells out to `open`.
"""

from __future__ import annotations

from copy import deepcopy
from html import escape
from typing import Any, Final

from .design_orchestration import (
    DESIGN_AUDIENCES,
    DESIGN_AVOID_PATTERNS,
    DESIGN_HIERARCHIES,
    DESIGN_LAYOUTS,
    DESIGN_MODES,
    DESIGN_PALETTES,
    DESIGN_PLATFORMS,
    DESIGN_PRIMARY_TASKS,
    DESIGN_SIGNATURE_ELEMENTS,
    DESIGN_SURFACES,
    DESIGN_TYPOGRAPHIES,
    _build_context_reference,
    _context_descriptors,
    _require_choice,
)

DESIGN_DIRECTION_SET_SCHEMA_VERSION: Final = "design_direction_set/v1"
# Four is the ceiling because a person comparing options holds about that many
# at once, and every option past the fourth is picked less often than the noise
# in the first three. Two is the floor because one option is not a choice - it
# is `design_orchestration/v1`, which already exists.
DESIGN_DIRECTION_OPTION_IDS: Final = ("a", "b", "c", "d")
_UNCHOSEN: Final = ""


def _build_direction_option(descriptor: tuple[str, str, str, str, str, str, tuple[str, ...]]) -> dict[str, object]:
    option_id, hierarchy, palette, typography, layout, signature_element, avoid_patterns = descriptor
    _require_choice(option_id, DESIGN_DIRECTION_OPTION_IDS, "option id")
    _require_choice(hierarchy, DESIGN_HIERARCHIES, "hierarchy")
    _require_choice(palette, DESIGN_PALETTES, "palette")
    _require_choice(typography, DESIGN_TYPOGRAPHIES, "typography")
    _require_choice(layout, DESIGN_LAYOUTS, "layout")
    _require_choice(signature_element, DESIGN_SIGNATURE_ELEMENTS, "signature_element")
    if not 1 <= len(avoid_patterns) <= len(DESIGN_AVOID_PATTERNS):
        raise ValueError("avoid_patterns must contain between one and six values")
    for pattern in avoid_patterns:
        _require_choice(pattern, DESIGN_AVOID_PATTERNS, "avoid_pattern")
    if len(set(avoid_patterns)) != len(avoid_patterns):
        raise ValueError("avoid_patterns must be unique")
    return {
        "option_id": option_id,
        "hierarchy": hierarchy,
        "palette": palette,
        "typography": typography,
        "layout": layout,
        "signature_element": signature_element,
        "avoid_patterns": list(avoid_patterns),
    }


def build_design_direction_set(
    *,
    surface: str,
    audience: str,
    primary_task: str,
    platform: str,
    mode: str,
    context_references: tuple[tuple[str, str, str], ...],
    options: tuple[tuple[str, str, str, str, str, str, tuple[str, ...]], ...],
    chosen_option: str = _UNCHOSEN,
) -> dict[str, object]:
    _require_choice(surface, DESIGN_SURFACES, "surface")
    _require_choice(audience, DESIGN_AUDIENCES, "audience")
    _require_choice(primary_task, DESIGN_PRIMARY_TASKS, "primary_task")
    _require_choice(platform, DESIGN_PLATFORMS, "platform")
    _require_choice(mode, DESIGN_MODES, "mode")
    if not 1 <= len(context_references) <= 5:
        raise ValueError("context_references must contain between one and five descriptors")
    references = [_build_context_reference(descriptor) for descriptor in context_references]
    reference_ids = [str(reference["reference_id"]) for reference in references]
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("context reference_id values must be unique")
    if not 2 <= len(options) <= len(DESIGN_DIRECTION_OPTION_IDS):
        raise ValueError("options must contain between two and four directions")
    built = [_build_direction_option(descriptor) for descriptor in options]
    option_ids = [str(option["option_id"]) for option in built]
    if len(set(option_ids)) != len(option_ids):
        raise ValueError("option_id values must be unique")
    if option_ids != list(DESIGN_DIRECTION_OPTION_IDS[: len(option_ids)]):
        raise ValueError("option_id values must be the leading ids in order, starting at a")
    # Two directions that differ in nothing are one direction presented twice,
    # which makes the choice a coin flip and the record of it meaningless.
    comparable = {tuple(sorted((key, str(value)) for key, value in option.items() if key != "option_id"))
                  for option in built}
    if len(comparable) != len(built):
        raise ValueError("options must differ from each other in at least one direction value")
    if chosen_option != _UNCHOSEN and chosen_option not in option_ids:
        raise ValueError("chosen_option must be empty or name one of the offered options")
    return {
        "schema_version": DESIGN_DIRECTION_SET_SCHEMA_VERSION,
        "status": "prepared_not_observed",
        "intent": {
            "surface": surface,
            "audience": audience,
            "primary_task": primary_task,
            "platform": platform,
            "mode": mode,
        },
        "context_references": references,
        "options": built,
        "chosen_option": chosen_option,
        "choice_status": "chosen" if chosen_option else "awaiting_choice",
        "preview": {
            "format": "static_html",
            "self_contained": True,
            "server_bound": False,
            "browser_launched": False,
            "rendered_observed": False,
        },
        "stop_conditions": [
            "No opaque project, user, or Hermes context reference is available.",
            "Fewer than two materially different directions can be described for this surface.",
            "The choice needs rendered evidence of the real product rather than a vocabulary preview.",
        ],
        "claim_boundary": (
            "This prepared design direction set records only bounded design intent, opaque context references, two to four "
            "direction vocabularies, and which one was chosen. The static preview is a rendering of that vocabulary, not of the "
            "product: it is not implementation, browser verification, accessibility or visual QA, review, CI, deployment, or "
            "merge evidence, and a recorded choice is not evidence that anyone looked at it."
        ),
    }


def _option_descriptors(value: object) -> tuple[tuple[str, str, str, str, str, str, tuple[str, ...]], ...] | None:
    if not isinstance(value, list):
        return None
    descriptors: list[tuple[str, str, str, str, str, str, tuple[str, ...]]] = []
    for option in value:
        if not isinstance(option, dict) or set(option) != {
            "option_id", "hierarchy", "palette", "typography", "layout", "signature_element", "avoid_patterns"
        }:
            return None
        avoid_patterns = option.get("avoid_patterns")
        if not isinstance(avoid_patterns, list) or not all(isinstance(item, str) for item in avoid_patterns):
            return None
        scalars = [option.get(key) for key in
                   ("option_id", "hierarchy", "palette", "typography", "layout", "signature_element")]
        if not all(isinstance(item, str) for item in scalars):
            return None
        descriptors.append((*scalars, tuple(avoid_patterns)))  # type: ignore[arg-type]
    return tuple(descriptors)


def validate_design_direction_set(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["design direction set must be an object"]
    required = {
        "schema_version",
        "status",
        "intent",
        "context_references",
        "options",
        "chosen_option",
        "choice_status",
        "preview",
        "stop_conditions",
        "claim_boundary",
    }
    if set(value) != required:
        return ["design direction set keys are invalid"]
    intent = value.get("intent")
    if not isinstance(intent, dict) or set(intent) != {"surface", "audience", "primary_task", "platform", "mode"}:
        return ["design direction set intent is invalid"]
    if not all(isinstance(item, str) for item in intent.values()):
        return ["design direction set intent is invalid"]
    descriptors = _context_descriptors(value.get("context_references"))
    if descriptors is None:
        return ["design direction set context references are invalid"]
    options = _option_descriptors(value.get("options"))
    if options is None:
        return ["design direction set options are invalid"]
    chosen_option = value.get("chosen_option")
    if not isinstance(chosen_option, str):
        return ["design direction set chosen_option is invalid"]
    try:
        expected = build_design_direction_set(
            surface=intent["surface"],
            audience=intent["audience"],
            primary_task=intent["primary_task"],
            platform=intent["platform"],
            mode=intent["mode"],
            context_references=descriptors,
            options=options,
            chosen_option=chosen_option,
        )
    except ValueError as exc:
        return [str(exc)]
    return [] if value == expected else ["design direction set values are invalid"]


def compact_design_direction_set(value: Any) -> dict[str, object]:
    if not isinstance(value, dict) or validate_design_direction_set(value):
        return {}
    return deepcopy(value)


def choose_design_direction(value: Any, option_id: str) -> dict[str, object]:
    """Record a choice against an existing set, leaving the offer untouched.

    Rebuilds rather than mutating so the result goes through the same validator
    as a fresh set: a stored artifact that has drifted cannot be laundered into
    a valid one by writing a choice onto it.
    """
    issues = validate_design_direction_set(value)
    if issues:
        raise ValueError(issues[0])
    intent = value["intent"]
    descriptors = _context_descriptors(value["context_references"])
    options = _option_descriptors(value["options"])
    if descriptors is None or options is None:  # pragma: no cover - validator already proved these
        raise ValueError("design direction set is invalid")
    return build_design_direction_set(
        surface=intent["surface"],
        audience=intent["audience"],
        primary_task=intent["primary_task"],
        platform=intent["platform"],
        mode=intent["mode"],
        context_references=descriptors,
        options=options,
        chosen_option=option_id,
    )


# --- static preview -------------------------------------------------------
#
# The vocabulary is rendered as the thing it names, not as a table of the words.
# A person choosing between `editorial_serif` and `utilitarian_mono` is choosing
# between two typefaces, so the preview has to set them.

_PALETTE_TOKENS: Final = {
    "restrained_neutral": ("#f7f7f5", "#1b1b19", "#6b6b63", "#e2e2dc", "#3d5a4c"),
    "contextual_accent": ("#fbfaf8", "#191821", "#6a6780", "#e6e3ee", "#4b3fa7"),
    "high_contrast": ("#ffffff", "#000000", "#3a3a3a", "#d0d0d0", "#b4001f"),
}
_TYPE_STACKS: Final = {
    "editorial_serif": ("'Iowan Old Style','Palatino Linotype',Palatino,Georgia,serif", "0"),
    "system_sans": ("system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',sans-serif", "-0.011em"),
    "utilitarian_mono": ("ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace", "-0.02em"),
}
_HIERARCHY_ORDER: Final = {
    "task_first": ("action", "evidence", "content"),
    "evidence_first": ("evidence", "action", "content"),
    "content_first": ("content", "action", "evidence"),
}
_BLOCK_LABELS: Final = {
    "action": "Primary action",
    "evidence": "Evidence",
    "content": "Content",
}
_SIGNATURE_LABELS: Final = {
    "evidence_rail": "Evidence rail",
    "decision_map": "Decision map",
    "progress_trace": "Progress trace",
    "none": "",
}


def _block_html(kind: str) -> str:
    label = escape(_BLOCK_LABELS[kind])
    if kind == "action":
        return f'<div class="blk act"><span>{label}</span><em class="btn">Continue</em></div>'
    if kind == "evidence":
        return (
            f'<div class="blk ev"><span>{label}</span>'
            '<i class="bar w80"></i><i class="bar w55"></i><i class="bar w66"></i></div>'
        )
    return (
        f'<div class="blk ct"><span>{label}</span>'
        '<i class="line w90"></i><i class="line w75"></i><i class="line w84"></i><i class="line w40"></i></div>'
    )


def _mockup_html(option: dict[str, object]) -> str:
    order = _HIERARCHY_ORDER[str(option["hierarchy"])]
    blocks = "".join(_block_html(kind) for kind in order)
    signature = _SIGNATURE_LABELS[str(option["signature_element"])]
    rail = f'<div class="rail"><span>{escape(signature)}</span></div>' if signature else ""
    layout = str(option["layout"])
    if layout == "split_panel":
        return f'<div class="mock split">{rail}<div class="col">{blocks}</div></div>'
    if layout == "editorial_grid":
        return f'<div class="mock grid">{blocks}{rail}</div>'
    return f'<div class="mock single">{blocks}{rail}</div>'


def _option_card_html(option: dict[str, object], chosen: str) -> str:
    option_id = str(option["option_id"])
    background, ink, muted, hairline, accent = _PALETTE_TOKENS[str(option["palette"])]
    family, tracking = _TYPE_STACKS[str(option["typography"])]
    is_chosen = option_id == chosen
    avoided = ", ".join(escape(str(pattern)) for pattern in option["avoid_patterns"])  # type: ignore[union-attr]
    facts = "".join(
        f"<dt>{escape(key)}</dt><dd>{escape(str(option[key]))}</dd>"
        for key in ("hierarchy", "palette", "typography", "layout", "signature_element")
    )
    return f"""
  <section class="opt{' chosen' if is_chosen else ''}" style="
      --bg:{background}; --ink:{ink}; --muted:{muted}; --hair:{hairline}; --accent:{accent};
      --family:{family}; --tracking:{tracking};">
    <header class="opt-head">
      <span class="tag">Option {escape(option_id.upper())}</span>
      {'<span class="chosen-mark">chosen</span>' if is_chosen else ''}
    </header>
    <div class="canvas">{_mockup_html(option)}</div>
    <dl class="facts">{facts}</dl>
    <p class="avoid"><span>avoids</span> {avoided}</p>
  </section>"""


def render_design_direction_set_html(value: Any, *, selection_notice: str | None = None) -> str:
    """One self-contained HTML document. No external request of any kind."""
    issues = validate_design_direction_set(value)
    if issues:
        raise ValueError(issues[0])
    intent = value["intent"]
    chosen = str(value["chosen_option"])
    cards = "".join(_option_card_html(option, chosen) for option in value["options"])
    summary = " &middot; ".join(
        escape(f"{key.replace('_', ' ')}: {intent[key]}")
        for key in ("surface", "audience", "primary_task", "platform", "mode")
    )
    choice_line = (
        f"Option {escape(chosen.upper())} is recorded as chosen."
        if chosen
        else "No option is chosen yet. Record one with "
        "<code>omh ops design-directions … --choose &lt;id&gt;</code>."
    )
    if selection_notice is not None:
        choice_line = escape(selection_notice)
    reference_count = len(value["context_references"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Design directions &mdash; {escape(str(intent['surface']))}</title>
<style>
  :root {{
    --page: #fdfdfc; --page-ink: #17170f; --page-muted: #6f6f64; --page-hair: #e4e4dd;
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --page: #131312; --page-ink: #f2f2ec; --page-muted: #9b9b90; --page-hair: #2e2e2a; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 2.5rem 1.5rem 4rem; background: var(--page); color: var(--page-ink); }}
  .wrap {{ max-width: 1180px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 .4rem; letter-spacing: -.02em; }}
  .meta {{ color: var(--page-muted); font-size: .8125rem; margin: 0 0 .35rem; }}
  .choice {{ font-size: .8125rem; margin: 0 0 2rem; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .95em; }}
  .opts {{ display: grid; gap: 1.25rem; grid-template-columns: repeat(auto-fit, minmax(255px, 1fr)); }}
  /* Column + growing canvas so the fact lists bottom-align across options.
     Without it a shorter mockup pulls its own facts up and the eye compares
     rows that are not the same row. */
  .opt {{ border: 1px solid var(--page-hair); border-radius: 10px; overflow: hidden; background: var(--bg);
          color: var(--ink); font-family: var(--family); letter-spacing: var(--tracking);
          display: flex; flex-direction: column; }}
  .opt > .canvas {{ flex: 1; }}
  .opt.chosen {{ outline: 2px solid var(--accent); outline-offset: -1px; }}
  .opt-head {{ display: flex; justify-content: space-between; align-items: center;
               padding: .6rem .85rem; border-bottom: 1px solid var(--hair); }}
  .tag {{ font-size: .6875rem; text-transform: uppercase; letter-spacing: .09em; color: var(--muted); }}
  .chosen-mark {{ font-size: .6875rem; text-transform: uppercase; letter-spacing: .09em; color: var(--accent); }}
  .canvas {{ padding: 1rem .85rem; min-height: 250px; }}
  .mock {{ display: flex; flex-direction: column; gap: .6rem; }}
  .mock.split {{ flex-direction: row-reverse; align-items: stretch; gap: .55rem; }}
  .mock.split .col {{ display: flex; flex-direction: column; gap: .6rem; flex: 1; }}
  .mock.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: .55rem; }}
  .blk {{ border: 1px solid var(--hair); border-radius: 6px; padding: .55rem .6rem; }}
  .blk span {{ display: block; font-size: .625rem; text-transform: uppercase;
               letter-spacing: .08em; color: var(--muted); margin-bottom: .4rem; }}
  .btn {{ display: inline-block; font-style: normal; font-size: .75rem; padding: .3rem .8rem;
          border-radius: 5px; background: var(--accent); color: var(--bg); }}
  .bar {{ display: block; height: .5rem; border-radius: 2px; background: var(--accent); opacity: .28; margin-bottom: .28rem; }}
  .line {{ display: block; height: .3rem; border-radius: 2px; background: var(--ink); opacity: .17; margin-bottom: .3rem; }}
  .w90 {{ width: 90%; }} .w84 {{ width: 84%; }} .w80 {{ width: 80%; }}
  .w75 {{ width: 75%; }} .w66 {{ width: 66%; }} .w55 {{ width: 55%; }} .w40 {{ width: 40%; }}
  .rail {{ border: 1px dashed var(--accent); border-radius: 6px; padding: .5rem .45rem;
           min-width: 4.6rem; display: flex; align-items: center; justify-content: center; }}
  .rail span {{ font-size: .625rem; text-transform: uppercase; letter-spacing: .07em;
                color: var(--accent); text-align: center; line-height: 1.35; }}
  .facts {{ display: grid; grid-template-columns: auto 1fr; gap: .12rem .6rem; margin: 0;
            padding: .7rem .85rem; border-top: 1px solid var(--hair); font-size: .6875rem; }}
  .facts dt {{ color: var(--muted); }}
  .facts dd {{ margin: 0; }}
  .avoid {{ margin: 0; padding: .55rem .85rem .8rem; font-size: .6875rem; color: var(--muted); }}
  .avoid span {{ text-transform: uppercase; letter-spacing: .08em; }}
  footer {{ margin-top: 2.25rem; padding-top: 1rem; border-top: 1px solid var(--page-hair);
            color: var(--page-muted); font-size: .75rem; line-height: 1.6; max-width: 62ch; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Design directions</h1>
  <p class="meta">{summary} &middot; {reference_count} opaque context reference(s)</p>
  <p class="choice">{choice_line}</p>
  <div class="opts">{cards}
  </div>
  <footer>{escape(str(value['claim_boundary']))}</footer>
</div>
</body>
</html>
"""
