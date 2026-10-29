"""Central type-guard utilities for provider/data boundaries
(deep-reliability audit, worker 13/20).

Why: malformed provider entries caused ``AttributeError: 'str' object has no
attribute 'get'`` (and siblings) whenever a provider returned an
unexpected shape — a usage string instead of a dict, a tool_calls list
containing strings, an SSE event that parsed to a JSON array instead of
an object, content blocks where a plain string was assumed.

These five tiny functions are the systemic fix: apply them at every
place where untrusted provider data enters our code (client.py's SSE
loop and JSON parser, and the TUI panel ingest).
They never raise; they never try to be clever (``ensure_dict`` on a
string returns the default — it deliberately does NOT attempt JSON
parsing, that is a different function's job); they just hand back a
value of the promised type so downstream ``.get`` / iteration / str
calls cannot blow up.

Contract:
- ``ensure_dict`` / ``ensure_list`` / ``ensure_int`` / ``ensure_bool``:
  return the value unchanged when it is exactly the promised type,
  else the default (and a fresh empty container / zero / False when
  the default itself is the wrong type).
- ``ensure_str``: returns the value when it is a ``str``; coerces via
  ``str()`` for ``int``/``float``/``bool``; returns the default for
  ``dict``/``list`` (and other containers, ``None``, unknown objects)
  — never a Python repr of a container. ``bytes``/``bytearray`` are
  decoded (UTF-8, errors replaced) rather than rejected, since provider
  wire data often arrives as bytes.

Only stdlib. No imports from other fullagent modules (avoids import
cycles) — safe to import at module level anywhere.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ensure_dict", "ensure_list", "ensure_str", "ensure_int",
           "ensure_bool"]


def ensure_dict(v: Any, default: dict | None = None) -> dict:
    """Return ``v`` if it is a dict, else a safe dict default.

    A string is NEVER JSON-parsed here — use ``parse_tool_arguments``
    (tools.py) when you actually want string -> dict conversion.
    """
    if isinstance(v, dict):
        return v
    if isinstance(default, dict):
        return default
    return {}


def ensure_list(v: Any, default: list | None = None) -> list:
    """Return ``v`` if it is a list, else a safe list default.

    Tuples/sets are NOT coerced (callers using index access would break
    on a silently-coerced tuple in subtle ways); pass an explicit list
    if you need one.
    """
    if isinstance(v, list):
        return v
    if isinstance(default, list):
        return default
    return []


def ensure_str(v: Any, default: str = "") -> str:
    """Return ``v`` as a str.

    ``str`` passes through; ``int``/``float``/``bool`` are coerced via
    ``str()`` (``5`` -> ``"5"``, ``True`` -> ``"True"``); ``dict``/``list``
    (and other containers, ``None``, unknown objects) yield ``default``
    — never a repr like ``"{'type': 'text'}"``. ``bytes``/``bytearray``
    are decoded UTF-8 with errors replaced. Never raises.
    """
    if isinstance(v, str):
        return v
    if isinstance(v, (bool, int, float)):
        try:
            return str(v)
        except Exception:  # noqa: BLE001 — never let coercion crash a turn
            pass
    elif isinstance(v, (bytes, bytearray)):
        try:
            return bytes(v).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — never let decoding crash a turn
            pass
    # dict, list, tuple, set, None, and unknown objects: default.
    if isinstance(default, str):
        return default
    return ""


def ensure_int(v: Any, default: int = 0) -> int:
    """Return ``v`` if it is an int (bools included), else a safe int
    default. No coercion of strings/floats — that is a different
    function's job. Never raises.
    """
    if isinstance(v, int):
        return v
    if isinstance(default, int) and not isinstance(default, bool):
        return default
    return 0


def ensure_bool(v: Any, default: bool = False) -> bool:
    """Return ``v`` if it is a bool, else a safe bool default. No
    coercion of truthy/falsy values (``1``, ``"true"``) — explicit is
    better than surprising at a data boundary. Never raises.
    """
    if isinstance(v, bool):
        return v
    if isinstance(default, bool):
        return default
    return False


# ---------------------------------------------------------------------------
# Self-test (offline: no API key needed).  Run: python -m fullagent.typeguards
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _passed = 0
    _failed: list[str] = []

    def check(name: str, got: Any, want: Any) -> None:
        global _passed
        if got == want and type(got) is type(want):
            _passed += 1
        else:
            _failed.append(f"{name}: got {got!r} ({type(got).__name__}), "
                           f"want {want!r} ({type(want).__name__})")

    # ---- ensure_dict ----
    check("dict/ok", ensure_dict({"a": 1}), {"a": 1})
    check("dict/None", ensure_dict(None), {})
    check("dict/str-never-parsed", ensure_dict('{"a": 1}'), {})
    check("dict/list", ensure_dict([1, 2]), {})
    check("dict/int", ensure_dict(5), {})
    check("dict/custom-default", ensure_dict(None, {"d": 1}), {"d": 1})
    check("dict/bad-default", ensure_dict("x", "nope"), {})  # type: ignore[arg-type]

    # ---- ensure_list ----
    check("list/ok", ensure_list([1, "a"]), [1, "a"])
    check("list/None", ensure_list(None), [])
    check("list/str-not-coerced", ensure_list("abc"), [])
    check("list/tuple-not-coerced", ensure_list((1, 2)), [])
    check("list/dict", ensure_list({"a": 1}), [])
    check("list/custom-default", ensure_list(None, [9]), [9])
    check("list/bad-default", ensure_list(1, "nope"), [])  # type: ignore[arg-type]

    # ---- ensure_str (coercion contract) ----
    check("str/ok", ensure_str("hello"), "hello")
    check("str/empty", ensure_str(""), "")
    check("str/None", ensure_str(None), "")
    check("str/int-coerced", ensure_str(42), "42")
    check("str/float-coerced", ensure_str(3.7), "3.7")
    check("str/bool-coerced", ensure_str(True), "True")
    check("str/dict-default", ensure_str({"a": 1}), "")
    check("str/list-default", ensure_str(["a"]), "")
    check("str/tuple-default", ensure_str((1,)), "")
    check("str/dict-custom-default", ensure_str({"a": 1}, "dflt"), "dflt")
    check("str/bytes", ensure_str(b"hi"), "hi")
    check("str/object-default", ensure_str(object()), "")
    check("str/bad-default", ensure_str(1, 5), "1")  # type: ignore[arg-type]

    # ---- ensure_int (strict) ----
    check("int/ok", ensure_int(42), 42)
    check("int/bool-is-int", ensure_int(True), True)
    check("int/None", ensure_int(None), 0)
    check("int/str-not-coerced", ensure_int("42"), 0)
    check("int/float-not-coerced", ensure_int(3.7), 0)
    check("int/custom-default", ensure_int("xx", -1), -1)
    check("int/bad-default", ensure_int(None, "nope"), 0)  # type: ignore[arg-type]

    # ---- ensure_bool (strict) ----
    check("bool/true", ensure_bool(True), True)
    check("bool/false", ensure_bool(False), False)
    check("bool/None", ensure_bool(None), False)
    check("bool/int-not-coerced", ensure_bool(1), False)
    check("bool/str-not-coerced", ensure_bool("true"), False)
    check("bool/custom-default", ensure_bool(None, True), True)
    check("bool/bad-default", ensure_bool(1, "nope"), False)  # type: ignore[arg-type]

    # ---- the reported crash shapes: provider-shaped garbage ----
    check("crash/event-array", ensure_dict(["not", "a", "dict"]).get("x"), None)
    check("crash/usage-str", ensure_dict("123 tokens").get("prompt_tokens"), None)
    check("crash/tc-str", ensure_dict("call_1").get("index", 0), 0)
    check("crash/content-list", ensure_str([{"type": "text"}]), "")
    check("crash/content-int", ensure_str(7), "7")

    print(f"typeguards self-test: {_passed} passed, {len(_failed)} failed")
    for f in _failed:
        print("  FAIL:", f)
    raise SystemExit(1 if _failed else 0)
