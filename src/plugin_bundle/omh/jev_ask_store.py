"""Where `omh_jev_ask` finds its route and records what it did -- never a key or any text.

Three things live here, all local:

* The operator setting `<omh_home>/jev/settings.json`. `{"openrouter_route":
  true}` is the only way the OpenRouter route becomes available: an
  `OPENROUTER_API_KEY` on its own is an OpenRouter account, not a request to
  send OMH's questions through it, and treating it as one would make every
  OpenRouter user a Jev user. The same file is the home for the route-question
  answerer opt-in proposed in #1816, so one Jev egress has one consent file.
  OMH never writes it. It is a local-trust opt-in, not a control against the
  host model: anything with write access to `<omh_home>` -- the operator, and
  a model with a file tool -- can write it. It chooses which of the person's
  own keys an ask may use; it cannot send an ask, which still needs this
  turn's consent (`jev_consent`).
* Route availability. The key VALUE is read only at call time, through the
  host's profile-scoped reader (`agent.secret_scope.get_secret`) when Hermes
  is present and `os.environ` only when that import does not exist. A host
  that refuses the read (`UnscopedSecretError` under the multiplexing gateway)
  makes the route unresolvable; OMH never falls back to another source.
* The ask ledger `<omh_home>/jev/asks.jsonl`: one metadata-only line per
  attempted ask, every status included. It carries hashes, counts, statuses,
  usage, and cost, and never the key, `state`, question text, or the reply.
  `omh doctor` reads it for the last status. It is a record, not evidence:
  a model with a file tool can append a row, so `omh_route_answer` confirms
  an `omh_jev_ask` provenance claim against `answered_ask` -- the asks this
  process itself answered -- and never against the file.

Stdlib and intra-bundle imports only.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from .awareness_delivery import _awareness_delivery_lock

JEV_DIRNAME: Final = "jev"
SETTINGS_FILENAME: Final = "settings.json"
LEDGER_FILENAME: Final = "asks.jsonl"
OPENROUTER_SETTING_KEY: Final = "openrouter_route"
LEDGER_SCHEMA_VERSION: Final = "omh_jev_ask_record/v1"
MAX_SETTINGS_BYTES: Final = 4096
MAX_LEDGER_BYTES: Final = 512 * 1024
MAX_LEDGER_LINES: Final = 512
_LEDGER_LOCK_SECONDS: Final = 2.0
MAX_REMEMBERED_ASKS: Final = 256

_answered_lock = threading.Lock()
# ask_id -> the ledger record of an ask this process sent and Jev answered.
_answered_asks: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

ROUTE_TYPESAFE: Final = "typesafe"
ROUTE_OPENROUTER: Final = "openrouter"
ROUTE_NONE: Final = "none"
# route id -> key variable name. The URLs live in `jev_ask_client.ROUTES`,
# which only the tool may import; `tests/test_jev_ask_tool.py` pins that the
# two tables name the same routes and the same variables.
ROUTE_KEY_NAMES: Final[dict[str, str]] = {
    ROUTE_TYPESAFE: "TYPESAFE_API_KEY",
    ROUTE_OPENROUTER: "OPENROUTER_API_KEY",
}


class KeyUnresolvable(RuntimeError):
    """The host refused to read the key for this profile."""


def settings_path(omh_home: Path) -> Path:
    return Path(omh_home) / JEV_DIRNAME / SETTINGS_FILENAME


def ledger_path(omh_home: Path) -> Path:
    return Path(omh_home) / JEV_DIRNAME / LEDGER_FILENAME


def openrouter_route_enabled(omh_home: Path) -> bool:
    """True only for a readable settings file whose `openrouter_route` is the literal `true`."""
    path = settings_path(omh_home)
    try:
        if path.is_symlink():
            return False
        with path.open("rb") as handle:
            raw = handle.read(MAX_SETTINGS_BYTES + 1)
    except OSError:
        return False
    if len(raw) > MAX_SETTINGS_BYTES:
        return False
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False
    return isinstance(data, dict) and data.get(OPENROUTER_SETTING_KEY) is True


def read_key(name: str) -> str:
    """The key value for one variable name, or "" when it is not set.

    Raises KeyUnresolvable when the host's scoped reader refuses. The value is
    returned to the one caller that sends it and is never stored.
    """
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret
    except ImportError:
        return str(os.environ.get(name, "") or "")
    try:
        value = get_secret(name)
    except UnscopedSecretError as error:
        raise KeyUnresolvable(f"the host did not resolve {name} for this profile") from error
    return str(value or "")


def resolve_route(requested: str, omh_home: Path) -> tuple[str, str]:
    """(route, key variable name) for a requested route, or (ROUTE_NONE, "") when none resolves.

    `auto` prefers the first-party route. A forced route whose key is absent
    does not fall back. Raises KeyUnresolvable from `read_key`.
    """
    candidates: list[str]
    if requested == ROUTE_TYPESAFE:
        candidates = [ROUTE_TYPESAFE]
    elif requested == ROUTE_OPENROUTER:
        candidates = [ROUTE_OPENROUTER]
    else:
        candidates = [ROUTE_TYPESAFE, ROUTE_OPENROUTER]
    for route in candidates:
        if route == ROUTE_OPENROUTER and not openrouter_route_enabled(omh_home):
            continue
        name = ROUTE_KEY_NAMES[route]
        if read_key(name):
            return route, name
    return ROUTE_NONE, ""


def route_available(omh_home: Path) -> str:
    """Which route an ask would take right now, for `check_fn`; never raises."""
    try:
        route, _ = resolve_route("auto", omh_home)
    except KeyUnresolvable:
        return ROUTE_NONE
    except Exception:  # noqa: BLE001 - classified: availability probe fails closed to "none", and the tool stays hidden
        return ROUTE_NONE
    return route


def route_available_by_names(omh_home: Path, env_names: set[str] | frozenset[str]) -> str:
    """The same answer from variable NAMES only, for `omh doctor`; reads no value."""
    if ROUTE_KEY_NAMES[ROUTE_TYPESAFE] in env_names:
        return ROUTE_TYPESAFE
    if ROUTE_KEY_NAMES[ROUTE_OPENROUTER] in env_names and openrouter_route_enabled(omh_home):
        return ROUTE_OPENROUTER
    return ROUTE_NONE


def append_ledger_record(omh_home: Path, record: Mapping[str, Any]) -> bool:
    """Append one metadata-only record, keeping the newest MAX_LEDGER_LINES; False when unwritable."""
    path = ledger_path(omh_home)
    try:
        for ancestor in (path, path.parent):
            if ancestor.is_symlink():
                return False
        with _awareness_delivery_lock(path, timeout_seconds=_LEDGER_LOCK_SECONDS):
            lines = _ledger_lines(path)
            lines.append(json.dumps(dict(record), sort_keys=True, ensure_ascii=True))
            lines = lines[-MAX_LEDGER_LINES:]
            # A random, exclusively created name: a predictable one let a
            # planted symlink redirect this write and its chmod at any file.
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            replaced = False
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    try:
                        os.fchmod(handle.fileno(), 0o600)
                    except (AttributeError, OSError):
                        pass  # mkstemp already created it 0600; best-effort as before
                    handle.write("\n".join(lines) + "\n")
                os.replace(temporary, path)
                replaced = True
            finally:
                if not replaced:
                    try:
                        os.unlink(temporary)
                    except OSError:
                        pass
    except OSError:
        return False
    return True


def read_ledger_records(omh_home: Path) -> list[dict[str, Any]]:
    """Every parseable record, oldest first; an unreadable ledger reads as empty."""
    records: list[dict[str, Any]] = []
    for line in _ledger_lines(ledger_path(omh_home)):
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("schema_version") == LEDGER_SCHEMA_VERSION:
            records.append(data)
    return records


def remember_answered_ask(record: Mapping[str, Any]) -> None:
    """Keep an answered ask's record in this process; only `omh_jev_ask` calls this."""
    ask_id = str(record.get("ask_id") or "")
    if not ask_id or record.get("status") != "answered":
        return
    with _answered_lock:
        _ = _answered_asks.pop(ask_id, None)
        _answered_asks[ask_id] = dict(record)
        while len(_answered_asks) > MAX_REMEMBERED_ASKS:
            _ = _answered_asks.popitem(last=False)


def answered_ask(ask_id: str) -> dict[str, Any] | None:
    """The record of an ask THIS process sent and Jev answered, else None.

    Process-local on purpose: the ledger file is writable by anything with
    access to `<omh_home>`, so a row there proves nothing. A restart forgets
    every ask, which can only refuse a claim.
    """
    wanted = str(ask_id or "").strip()
    if not wanted:
        return None
    with _answered_lock:
        record = _answered_asks.get(wanted)
        return dict(record) if record is not None else None


def forget_answered_asks() -> None:
    """Test seam: forget every answered ask."""
    with _answered_lock:
        _answered_asks.clear()


def ledger_summary(omh_home: Path) -> dict[str, Any]:
    records = read_ledger_records(omh_home)
    return {
        "ledger_calls": len(records),
        "last_status": str(records[-1].get("status", "")) if records else "",
    }


def _ledger_lines(path: Path) -> list[str]:
    try:
        if path.is_symlink() or not path.is_file():
            return []
        with path.open("rb") as handle:
            raw = handle.read(MAX_LEDGER_BYTES + 1)
    except OSError:
        return []
    text = raw[-MAX_LEDGER_BYTES:].decode("utf-8", errors="replace")
    return [line for line in text.splitlines() if line.strip().startswith("{")]


__all__ = [
    "KeyUnresolvable",
    "LEDGER_SCHEMA_VERSION",
    "OPENROUTER_SETTING_KEY",
    "ROUTE_KEY_NAMES",
    "ROUTE_NONE",
    "ROUTE_OPENROUTER",
    "ROUTE_TYPESAFE",
    "answered_ask",
    "append_ledger_record",
    "forget_answered_asks",
    "ledger_path",
    "ledger_summary",
    "openrouter_route_enabled",
    "read_key",
    "read_ledger_records",
    "remember_answered_ask",
    "resolve_route",
    "route_available",
    "route_available_by_names",
    "settings_path",
]
