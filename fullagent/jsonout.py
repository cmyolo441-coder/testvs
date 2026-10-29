"""Machine-readable JSON output mode for headless turns.

Renders a completed :class:`fullagent.agent.Turn` as a single-line JSON
object so scripts and other tools can consume agent runs without scraping
the pretty terminal output.

Public API:
    - :func:`register` -- attach helpers to an agent (duck-typed, minimal).
    - :func:`turn_to_json` -- turn -> plain ``dict`` (all values JSON-safe).
    - :func:`emit_json` -- print the dict as one JSON line, nothing else.

Integration (wired by the coordinator, NOT in this file — do not edit
``__main__.py`` here):

    1. In ``fullagent/__main__.py``'s ``main()``, add to the argparse block::

           _p.add_argument("--output-format", choices=["text", "json"],
                           default="text",
                           help="headless turn output format (text|json)")

       Placement: next to the existing ``--resume`` / ``--continue``
       arguments; the parsed namespace must be passed into ``_headless``.

    2. In ``_headless``, after a turn completes (where the pretty printer
       would run)::

           from . import jsonout
           if ns.output_format == "json":
               jsonout.emit_json(jsonout.turn_to_json(
                   turn, getattr(agent, "session_id", None),
                   extra={"agent": agent}))
           else:
               ...existing pretty printing...

Schema emitted by :func:`turn_to_json`::

    {
      "session_id": str | null,
      "model": str,
      "user": str,
      "assistant": str,
      "tool_calls": [
        {"name": str, "args": {...}, "result_truncated": str, "status": str}
      ],
      "usage": {"input_tokens": int, "output_tokens": int},
      "cost_usd": float | null,
      "error": str | null,
      "duration_s": float,
      ...extra keys merged last...
    }

All field access is duck-typed with getattr guards, so fake or older
Turn objects never raise.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

RESULT_TRUNCATE = 2000  # chars of tool result kept in result_truncated

_SCHEMA_KEYS = (
    "session_id", "model", "user", "assistant", "tool_calls",
    "usage", "cost_usd", "error", "duration_s",
)


def _as_dict(value: Any) -> Dict[str, Any]:
    """Best-effort conversion of args-like payloads to a JSON-safe dict."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    try:
        return dict(value)
    except Exception:
        return {"value": value}


def _json_safe(value: Any) -> Any:
    """Coerce anything into a JSON-serialisable value (never raises)."""
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(_json_safe(k)): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _usage(turn: Any) -> Dict[str, int]:
    # Delegate to the canonical extractor: ad-hoc int() coercion here
    # used to pass negative values straight through (and any future
    # usage-shape weirdness would need fixing in two places).
    from .client import safe_token_counts
    in_tok, out_tok = safe_token_counts(getattr(turn, "usage", None))
    return {"input_tokens": in_tok, "output_tokens": out_tok}


def _tool_calls(turn: Any) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    tools = getattr(turn, "tools", None) or []
    for ev in tools:
        result = getattr(ev, "result", "")
        result = result if isinstance(result, str) else str(result)
        calls.append({
            "name": getattr(ev, "name", ""),
            "args": _json_safe(_as_dict(getattr(ev, "args", {}) or {})),
            "result_truncated": result[:RESULT_TRUNCATE],
            "status": getattr(ev, "status", "unknown") or "unknown",
        })
    return calls


def _cost_usd(turn: Any, extra: Optional[Dict[str, Any]]) -> Optional[float]:
    """Turn-level cost if present, else fold of the agent's event log."""
    direct = getattr(turn, "cost_usd", None)
    if direct is not None:
        try:
            return float(direct)
        except (TypeError, ValueError):
            pass
    extra = extra or {}
    agent = extra.get("agent")
    if agent is not None:
        # The agent's event log folds cost.incurred events into a running
        # total; fullagent ships with free-tier providers so this is
        # usually 0.0 but non-zero on paid keys.
        try:
            from .kernel import fold
            return float(fold(agent.log).cost_usd)
        except Exception:
            pass
    return None


def turn_to_json(turn: Any, session_id: Optional[str],
                 extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Render a completed turn as a JSON-safe dict.

    ``turn`` is duck-typed against the :class:`Turn` dataclass in
    ``fullagent.agent`` (``user_text``, ``assistant_text``, ``model_id``,
    ``tools`` of ``ToolEvent`` ``(name, args, result, status)``,
    ``usage`` dict with ``prompt_tokens``/``completion_tokens``,
    ``error``, ``duration``). ``extra``, when given, is merged into the
    result (reserved key ``agent`` is consumed internally for cost and
    never emitted).
    """
    if session_id is None:
        session_id = getattr(turn, "session_id", None)
    error = getattr(turn, "error", "") or None
    try:
        duration = float(getattr(turn, "duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    obj: Dict[str, Any] = {
        "session_id": session_id,
        "model": getattr(turn, "model_id", "") or "",
        "user": getattr(turn, "user_text", "") or "",
        "assistant": getattr(turn, "assistant_text", "") or "",
        "tool_calls": _tool_calls(turn),
        "usage": _usage(turn),
        "cost_usd": _cost_usd(turn, extra),
        "error": error,
        "duration_s": duration,
    }
    if extra:
        merged = dict(extra)
        merged.pop("agent", None)  # internal-only, never emitted
        obj.update({k: _json_safe(v) for k, v in merged.items()})
    # Top-level fields come straight from the turn; coerce them too so the
    # documented "all values JSON-safe" contract holds for fakes/older Turns.
    return _json_safe(obj)


def emit_json(obj: Dict[str, Any]) -> None:
    """Print one JSON object as a single line; no other output."""
    print(json.dumps(obj, ensure_ascii=False))


def register(agent: Any) -> None:
    """Attach JSON-output helpers to an agent (duck-typed, minimal).

    Sets ``agent.turn_to_json`` and ``agent.emit_json`` so headless code
    can call them without importing this module.
    """
    agent.turn_to_json = turn_to_json
    agent.emit_json = emit_json


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import io
    from contextlib import redirect_stdout
    from types import SimpleNamespace

    # Fake Turn mirroring fullagent.agent.Turn / ToolEvent shapes
    calls = [
        SimpleNamespace(name="read", args={"path": "/tmp/a.txt"},
                        result="ok", status="done"),
        SimpleNamespace(name="write", args={"path": "/tmp/b.txt"},
                        result="x" * 5000, status="error"),
    ]
    turn = SimpleNamespace(
        user_text="sum these",
        assistant_text="done",
        model_id="deepseek-v4.1-flash-free",
        tools=calls,
        usage={"prompt_tokens": 120, "completion_tokens": 45},
        error="",
        duration=3.25,
    )

    obj = turn_to_json(turn, "sess-1")
    for key in _SCHEMA_KEYS:
        assert key in obj, f"missing schema key: {key}"
    assert obj["session_id"] == "sess-1"
    assert obj["model"] == "deepseek-v4.1-flash-free"
    assert obj["user"] == "sum these"
    assert obj["assistant"] == "done"
    assert obj["usage"] == {"input_tokens": 120, "output_tokens": 45}
    assert obj["error"] is None
    assert obj["duration_s"] == 3.25
    assert obj["cost_usd"] is None  # no tracker in self-test
    assert len(obj["tool_calls"]) == 2
    tc0 = obj["tool_calls"][0]
    assert tc0["name"] == "read" and tc0["args"] == {"path": "/tmp/a.txt"}
    assert tc0["status"] == "done"
    tc1 = obj["tool_calls"][1]
    assert len(tc1["result_truncated"]) == 2000, \
        f"truncation wrong: {len(tc1['result_truncated'])}"
    assert tc1["status"] == "error"

    # Missing attributes never raise; everything defaults sanely
    sparse = turn_to_json(SimpleNamespace(), None)
    for key in _SCHEMA_KEYS:
        assert key in sparse, f"sparse missing: {key}"
    assert sparse["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert sparse["tool_calls"] == []

    # Cost from turn attribute wins when present
    costy = turn_to_json(SimpleNamespace(cost_usd="0.25"), "s")
    assert costy["cost_usd"] == 0.25

    # emit_json: single line, parseable, matches turn_to_json
    buf = io.StringIO()
    with redirect_stdout(buf):
        emit_json(obj)
    line = buf.getvalue()
    assert line.count("\n") == 1, "emit_json must be a single line"
    assert json.loads(line) == obj

    # register attaches helpers
    fake = SimpleNamespace()
    register(fake)
    assert fake.turn_to_json is turn_to_json
    assert fake.emit_json is emit_json

    print("jsonout self-test PASSED")
