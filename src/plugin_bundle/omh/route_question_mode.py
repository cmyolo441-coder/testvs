"""Which mode the route question runs in, and where that value came from.

The route question (`route_question/v1`) is a shadow surface: the router
attaches it to an undecidable route, something answers it, and the answer is
recorded and measured without changing the route. This module names that mode
and reads it from the operator's own OMH config, so every record the surface
writes can say which mode produced it.

The value is read from ``<omh_home>/routing/route-question.json``::

    {"mode": "off"}

Three things this module will not do, each for a reason:

- **It will not turn a read failure into a default.** A file that exists and
  cannot be read, parsed, or understood yields ``unknown`` together with the
  reason, never ``shadow``. A surface that silently falls back to its default
  when its switch cannot be read is a switch nobody can trust, and the record
  is where a reader would otherwise find out. A file that does not exist is a
  different fact -- the operator never set a mode -- and yields the documented
  default with ``mode_source`` saying so.
- **It will not accept ``on``.** ``on`` is the deciding mode, in which an
  answer would change the route. Nothing applies an answer yet, so a config
  naming it is reported as ``unknown`` with ``on_not_available`` rather than
  recorded as a mode the surface is not actually in.
- **It will not take the mode from the model.** Nothing here reads a tool
  argument or a payload; the only input is a file under the OMH home.

Stdlib only: the plugin bundle loads this file under Hermes' interpreter, and
the core package imports it from here so the two cannot disagree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

ROUTE_QUESTION_MODE_OFF: Final = "off"
ROUTE_QUESTION_MODE_SHADOW: Final = "shadow"
ROUTE_QUESTION_MODE_UNKNOWN: Final = "unknown"
# The deciding mode. Named so a config that asks for it is refused by name.
ROUTE_QUESTION_MODE_ON: Final = "on"

# The values an operator may configure today.
CONFIGURABLE_ROUTE_QUESTION_MODES: Final = (ROUTE_QUESTION_MODE_OFF, ROUTE_QUESTION_MODE_SHADOW)
# Every value a record can carry.
RECORDED_ROUTE_QUESTION_MODES: Final = (
    ROUTE_QUESTION_MODE_OFF,
    ROUTE_QUESTION_MODE_SHADOW,
    ROUTE_QUESTION_MODE_UNKNOWN,
)
# Today's behaviour, and what an operator who never wrote the file gets.
DEFAULT_ROUTE_QUESTION_MODE: Final = ROUTE_QUESTION_MODE_SHADOW

MODE_SOURCE_CONFIG: Final = "omh_config"
MODE_SOURCE_DEFAULT: Final = "default"
# A record built by a caller that never read the config at all.
MODE_SOURCE_NOT_READ: Final = "not_read"

MODE_ERROR_UNREADABLE: Final = "unreadable"
MODE_ERROR_TOO_LARGE: Final = "too_large"
MODE_ERROR_NOT_JSON: Final = "not_json"
MODE_ERROR_NOT_AN_OBJECT: Final = "not_an_object"
MODE_ERROR_MISSING_MODE: Final = "missing_mode"
MODE_ERROR_ON_NOT_AVAILABLE: Final = "on_not_available"
MODE_ERROR_UNSUPPORTED_VALUE: Final = "unsupported_value"

ROUTE_QUESTION_CONFIG_DIRNAME: Final = "routing"
ROUTE_QUESTION_CONFIG_FILENAME: Final = "route-question.json"
MAX_ROUTE_QUESTION_CONFIG_BYTES: Final = 4096


def route_question_config_path(omh_home: Path | str) -> Path:
    return Path(omh_home) / ROUTE_QUESTION_CONFIG_DIRNAME / ROUTE_QUESTION_CONFIG_FILENAME


def unread_route_question_mode() -> dict[str, str]:
    """The reading a record carries when nobody read the config."""
    return {"mode": ROUTE_QUESTION_MODE_UNKNOWN, "mode_source": MODE_SOURCE_NOT_READ}


def read_route_question_mode(omh_home: Path | str) -> dict[str, str]:
    """``{"mode", "mode_source"}``, plus ``mode_error`` when the mode is unknown."""
    path = route_question_config_path(omh_home)
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_ROUTE_QUESTION_CONFIG_BYTES + 1)
    except FileNotFoundError:
        return {"mode": DEFAULT_ROUTE_QUESTION_MODE, "mode_source": MODE_SOURCE_DEFAULT}
    except OSError:
        return _unknown(MODE_ERROR_UNREADABLE)
    if len(raw) > MAX_ROUTE_QUESTION_CONFIG_BYTES:
        return _unknown(MODE_ERROR_TOO_LARGE)
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return _unknown(MODE_ERROR_NOT_JSON)
    if not isinstance(document, dict):
        return _unknown(MODE_ERROR_NOT_AN_OBJECT)
    if "mode" not in document:
        return _unknown(MODE_ERROR_MISSING_MODE)
    value = document["mode"]
    if value == ROUTE_QUESTION_MODE_ON:
        return _unknown(MODE_ERROR_ON_NOT_AVAILABLE)
    if not isinstance(value, str) or value not in CONFIGURABLE_ROUTE_QUESTION_MODES:
        return _unknown(MODE_ERROR_UNSUPPORTED_VALUE)
    return {"mode": value, "mode_source": MODE_SOURCE_CONFIG}


def route_question_mode_fields(reading: object) -> dict[str, str]:
    """The closed subset of a reading a record may carry.

    A reading handed in by a caller is re-checked here, so a record can only
    ever say one of the recorded modes and one of the named sources.
    """
    if not isinstance(reading, dict):
        return unread_route_question_mode()
    mode = reading.get("mode")
    source = reading.get("mode_source")
    if mode not in RECORDED_ROUTE_QUESTION_MODES or source not in (
        MODE_SOURCE_CONFIG,
        MODE_SOURCE_DEFAULT,
        MODE_SOURCE_NOT_READ,
    ):
        return unread_route_question_mode()
    fields = {"mode": str(mode), "mode_source": str(source)}
    error = reading.get("mode_error")
    if mode == ROUTE_QUESTION_MODE_UNKNOWN and error in _MODE_ERRORS:
        fields["mode_error"] = str(error)
    return fields


_MODE_ERRORS: Final = (
    MODE_ERROR_UNREADABLE,
    MODE_ERROR_TOO_LARGE,
    MODE_ERROR_NOT_JSON,
    MODE_ERROR_NOT_AN_OBJECT,
    MODE_ERROR_MISSING_MODE,
    MODE_ERROR_ON_NOT_AVAILABLE,
    MODE_ERROR_UNSUPPORTED_VALUE,
)


def _unknown(error: str) -> dict[str, str]:
    return {"mode": ROUTE_QUESTION_MODE_UNKNOWN, "mode_source": MODE_SOURCE_CONFIG, "mode_error": error}


__all__ = [
    "CONFIGURABLE_ROUTE_QUESTION_MODES",
    "DEFAULT_ROUTE_QUESTION_MODE",
    "MODE_SOURCE_CONFIG",
    "MODE_SOURCE_DEFAULT",
    "MODE_SOURCE_NOT_READ",
    "RECORDED_ROUTE_QUESTION_MODES",
    "ROUTE_QUESTION_MODE_OFF",
    "ROUTE_QUESTION_MODE_ON",
    "ROUTE_QUESTION_MODE_SHADOW",
    "ROUTE_QUESTION_MODE_UNKNOWN",
    "read_route_question_mode",
    "route_question_config_path",
    "route_question_mode_fields",
    "unread_route_question_mode",
]
