"""Who could answer a route question on THIS machine, in order of specificity.

A route question is typed so that something other than the deterministic
router can answer it. This module reports what that something could be here,
and nothing else. It is a report, not a recommendation: OMH does not call a
third-party plugin, does not ask the host model to call one, and the presence
of a rung is never a reason to prefer it.

The rungs, most specific first:

* ``omh_jev_ask`` -- OMH's own opt-in Jev tool could answer here: a route
  resolves (a ``TYPESAFE_API_KEY``, or an OpenRouter key plus the operator
  setting). It appears only then, with status ``available``. The tool itself
  still refuses unless the person asked for Jev in the current turn.
* ``jev_plugin`` -- a Jev-class plugin is on this machine. It appears only
  when one was detected, and its ``status`` is the strongest tier OMH can
  prove: ``installed`` (a manifest declares a Jev-class tool, or the plugin
  is a known one), ``enabled`` (Hermes' config also lists the plugin),
  ``observed`` (a Jev-class tool call reached dispatch on this install at
  least once). Jev-class tools are ``jev_``-prefixed, plus the exact
  ``nerve_`` names the renamed ``nerve`` lineage declares.
* ``main_model`` -- the host model reading this payload. Always available,
  because a model that can read the question can answer it.
* ``none`` -- leave the question unanswered. Always available, and it is not
  a failure: an unanswered question changes nothing and the deterministic
  route stays in force.

Every tier under-claims by construction. A config shape this module cannot
read reports ``installed`` rather than ``enabled``; an unreadable plugins
directory drops the ``jev_plugin`` rung entirely rather than inventing one.
That direction is deliberate -- a missing rung costs a reader an option it
could have had, while an invented one is a claim about a machine.

Stdlib and intra-bundle imports only: Hermes loads this directory with its own
interpreter, which has no reason to have the `omh` package on its path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .jev_ask_store import ROUTE_NONE, route_available
from .jev_sidekick import classify_plugin
from .runtime_reader import _yaml_list_values, installed_plugin_advertised_tools
from .todo_store import strip_control_characters
from .tool_bursts import jev_tool_observed_at

OMH_JEV_ASK_ANSWERER = "omh_jev_ask"
JEV_PLUGIN_ANSWERER = "jev_plugin"
MAIN_MODEL_ANSWERER = "main_model"
NO_ANSWERER = "none"

STATUS_INSTALLED = "installed"
STATUS_ENABLED = "enabled"
STATUS_OBSERVED = "observed"
STATUS_AVAILABLE = "available"

# How many plugin names and tool names one rung carries. A machine with more
# Jev-class plugins than this has a bigger problem than a truncated list, and
# the rung is a hint surface rather than an inventory -- `omh doctor --json`
# carries the full posture.
MAX_LADDER_ITEMS = 8

# Every string in this rung is written by whoever published the plugin, and the
# rung is serialized into the host model's context on every undecidable route.
# The manifest reader bounds a file at 256 KiB, which bounds nothing useful
# here: eight names of that size is megabytes of someone else's text per call.
# So each item is control-stripped and cut to a name-sized bound, matching what
# the record store next door does to a skill name. This is not a claim that the
# remaining text is safe to obey -- it is third-party data either way, and the
# `claim_boundary` beside it says so -- only that a manifest cannot spend the
# turn's context or smuggle an escape sequence through a field OMH echoes.
MAX_LADDER_ITEM_CHARS = 80

# The Hermes config is read once per ladder, not once per detected plugin, and
# only this far in. It is a host file rather than a third-party one, but it is
# still an unbounded read on a path OMH does not own.
MAX_CONFIG_BYTES = 256 * 1024

_JEV_CLAIM_BOUNDARY = (
    "A plugin name, an enabled name, or one observed tool call is what this "
    "machine declares, not evidence that Jev was served, that the plugin "
    "answered anything, or that OMH called it. OMH never calls a third-party "
    "plugin."
)
_OMH_JEV_ASK_CLAIM_BOUNDARY = (
    "Available means a route resolves here, not that Jev was served. The ask "
    "runs only after the user asks for Jev in this turn, and it sends the "
    "user's message off the machine."
)
_MAIN_MODEL_CLAIM_BOUNDARY = (
    "An answer from the model reading this payload is a routing judgment it "
    "declares about itself, not an observation OMH made."
)
_NO_ANSWERER_CLAIM_BOUNDARY = (
    "Leaving the question unanswered changes nothing: the deterministic route "
    "stays in force."
)


def answerer_ladder(hermes_home: object = "", omh_home: object = "") -> list[dict[str, Any]]:
    """The answerers available on this machine, most specific first.

    Both homes are optional strings or paths. Without a Hermes home no plugin
    can be detected, and without an OMH home no tool call can have been
    observed; each absence costs its own tier and never the whole ladder.
    """
    rungs: list[dict[str, Any]] = []
    ask_rung = _omh_jev_ask_rung(omh_home)
    if ask_rung is not None:
        rungs.append(ask_rung)
    jev_rung = _jev_plugin_rung(hermes_home, omh_home)
    if jev_rung is not None:
        rungs.append(jev_rung)
    rungs.append(
        {
            "answerer": MAIN_MODEL_ANSWERER,
            "status": STATUS_AVAILABLE,
            "claim_boundary": _MAIN_MODEL_CLAIM_BOUNDARY,
        }
    )
    rungs.append(
        {
            "answerer": NO_ANSWERER,
            "status": STATUS_AVAILABLE,
            "claim_boundary": _NO_ANSWERER_CLAIM_BOUNDARY,
        }
    )
    return rungs


def _omh_jev_ask_rung(omh_home: object) -> dict[str, Any] | None:
    """The `omh_jev_ask` rung, or None when no route resolves or no OMH home was named."""
    home = str(omh_home or "").strip()
    if not home:
        return None
    route = route_available(Path(home))
    if route == ROUTE_NONE:
        return None
    return {
        "answerer": OMH_JEV_ASK_ANSWERER,
        "status": STATUS_AVAILABLE,
        "route": route,
        "claim_boundary": _OMH_JEV_ASK_CLAIM_BOUNDARY,
    }


def _jev_plugin_rung(hermes_home: object, omh_home: object) -> dict[str, Any] | None:
    """The `jev_plugin` rung, or None when no Jev-class plugin was detected.

    The raw directory names decide the tier, because the host's config lists a
    plugin under the name it is installed as; the bounded copies are what the
    rung echoes. Keeping the two apart means a hostile name costs its publisher
    the `enabled` tier rather than costing OMH a correct reading.
    """
    detected = _detected_jev_plugins(hermes_home)
    if not detected:
        return None
    names = sorted(detected)
    tools = sorted({tool for name in names for tool in detected[name]})
    status = STATUS_INSTALLED
    if set(names) & _enabled_plugin_names(hermes_home):
        status = STATUS_ENABLED
    if _jev_tool_observed(omh_home):
        status = STATUS_OBSERVED
    return {
        "answerer": JEV_PLUGIN_ANSWERER,
        "status": status,
        "plugins": _bounded_items(names),
        "tools": _bounded_items(tools),
        "claim_boundary": _JEV_CLAIM_BOUNDARY,
    }


def _bounded_items(values: list[str]) -> list[str]:
    """Third-party strings, control-stripped, cut to a name, and capped in count.

    An item that is nothing but control characters strips to empty and is
    dropped rather than echoed as `""`: it names no plugin and no tool, and a
    blank row in the list would read as one that exists.
    """
    bounded: list[str] = []
    for value in values:
        item = strip_control_characters(value)[:MAX_LADDER_ITEM_CHARS]
        if item:
            bounded.append(item)
        if len(bounded) >= MAX_LADDER_ITEMS:
            break
    return bounded


def _detected_jev_plugins(hermes_home: object) -> dict[str, list[str]]:
    """Installed plugin directory name -> the Jev-class tools its manifest declares.

    The classifier decides what counts as Jev-class, so a known plugin under a
    name that declares no tool at all is detected too, and the tool-prefix
    signal keeps a renamed one from disappearing.
    """
    home = _home_path(hermes_home)
    if home is None:
        return {}
    try:
        declarations = installed_plugin_advertised_tools(home)
    except OSError:
        # An unreadable plugins directory is the absence of a reading, not the
        # absence of a plugin. The rung is dropped rather than reported at a
        # tier that was not proved.
        return {}
    detected: dict[str, list[str]] = {}
    for name, tools in declarations.items():
        record = classify_plugin(name, tools)
        if record is None:
            continue
        detected[name] = [str(tool) for tool in record["jev_tools"]]
    return detected


def _enabled_plugin_names(hermes_home: object) -> frozenset[str]:
    """The plugin names Hermes' own config enables, or an empty set.

    The bundle cannot import core's `config_adapter`, so this reuses the list
    reader the HUD already has, applied to the `plugins:` block alone rather
    than to the whole file -- an `enabled:` key nested under some other
    plugin's settings is a different key with the same name.

    Read once per ladder rather than once per detected plugin: the answer is a
    property of the file, not of the name being asked about, and re-reading it
    up to eight times made an unbounded read into eight of them.

    Every way this can fail yields the empty set, which reports each plugin one
    tier lower than it sits. That is the safe direction, and it has to be the
    WIDE direction too: the read used to catch `OSError` alone, so a config
    with one non-UTF-8 byte raised `UnicodeDecodeError` -- a `ValueError`, not
    an `OSError` -- out through `build_chat_route_hint_payload` and failed
    `omh chat route-hint` outright. The ladder is allowed to cost itself; it is
    not allowed to cost the payload it rides on.

    A shape this reader cannot follow (an inline `plugins: {enabled: [omh]}`,
    an anchor) reads as absent for the same reason. Core's `config_adapter`
    has followed the flow mapping since #1814 and this reader has not, because
    the bundle cannot import it and a second scanner for one tier is the wrong
    trade. The divergence costs a routing tier and never a report: this answer
    picks which hint to offer, and nothing on this path states whether Hermes
    enables a plugin.
    """
    home = _home_path(hermes_home)
    if home is None:
        return frozenset()
    try:
        with (home / "config.yaml").open("rb") as handle:
            raw = handle.read(MAX_CONFIG_BYTES)
        text = raw.decode("utf-8")
    except (OSError, ValueError):
        return frozenset()
    block = _plugins_block(text)
    if not block:
        return frozenset()
    enabled = set(_yaml_list_values(block, "enabled"))
    return frozenset(enabled - set(_yaml_list_values(block, "disabled")))


def _plugins_block(config_text: str) -> str:
    """The lines under a top-level `plugins:` key, or "".

    Sliced by indentation so the list reader below sees one block. The last
    `plugins:` key wins, the way a YAML loader resolves a duplicate mapping
    key.
    """
    lines = config_text.splitlines()
    start = -1
    for index, line in enumerate(lines):
        if line.strip().split("#", 1)[0].rstrip() == "plugins:" and not line[:1].isspace():
            start = index
    if start < 0:
        return ""
    block: list[str] = []
    for line in lines[start + 1:]:
        if line.strip() and not line[:1].isspace():
            break
        block.append(line)
    return "\n".join(block)


def _jev_tool_observed(omh_home: object) -> bool:
    """Whether this install has ever dispatched a Jev-class tool call.

    The empty home is refused rather than passed down. `jev_tool_observed_at("")`
    falls back to the ambient `default_omh_home()`, so a caller that named a
    Hermes home and no OMH home would have been answered from the machine's
    real ledger while being told nothing was read -- the docstring above
    promised the absence costs its own tier, and it has to be true.
    """
    home = str(omh_home or "").strip()
    if not home:
        return False
    try:
        return jev_tool_observed_at(home) > 0
    except OSError:
        return False


def _home_path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    return Path(text)


__all__ = [
    "JEV_PLUGIN_ANSWERER",
    "MAIN_MODEL_ANSWERER",
    "MAX_LADDER_ITEMS",
    "MAX_LADDER_ITEM_CHARS",
    "NO_ANSWERER",
    "OMH_JEV_ASK_ANSWERER",
    "STATUS_AVAILABLE",
    "STATUS_ENABLED",
    "STATUS_INSTALLED",
    "STATUS_OBSERVED",
    "answerer_ladder",
]
