"""Cost tracking — token usage and $ estimates per session.

Claude Code-style: every model completion burns input/output tokens; this
module accumulates them per session and converts them to USD using a
per-model price table, exposed via ``/cost`` in the TUI.

Public API:
    - :func:`register` -- attach ``agent.cost_tracker`` (a
      :class:`CostTracker`) for ``agent.session_id``.
    - :func:`handle_cost` -- TUI entry point ``(ui, arg) -> None``; prints
      session totals (turns, in/out tokens, estimated $) plus a per-model
      breakdown when more than one model was used.

Wiring: the coordinator must add ONE call in ``agent.py`` where per-turn
usage is known — see ``register``'s docstring for the exact hook point.

``python3 -m fullagent.costtrack`` runs the built-in self-test.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, Optional

COST_DIR = os.path.join(os.environ.get("FULLAGENT_HOME") or os.path.join(os.path.expanduser("~"), ".fullagent"), "cost")

_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]")

# ---------------------------------------------------------------------------
# Price table
# ---------------------------------------------------------------------------
# Per model id: (input USD per 1M tokens, output USD per 1M tokens).
#
# IMPORTANT: these are APPROXIMATE public prices, not billing-grade data.
# They exist only to give a rough spend estimate in the TUI. Prices change;
# treat any non-zero number here as "illustrative, could be stale".
# Models whose entry is (0.0, 0.0) are either on a free tier or have unknown
# pricing — "pricing unknown" is reported in that case (see
# :func:`cost_of`).
PRICING: Dict[str, tuple[float, float]] = {
    "deepseek-v4.1-flash-free": (0.0, 0.0),
    "minimax-m3.1-flash-free": (0.0, 0.0),
}

USAGE = "usage:\n  /cost        show session token usage and estimated cost\n  /cost reset  clear this session's counters"


def _safe_session_id(session_id: Optional[str]) -> str:
    sid = (session_id or "default").strip() or "default"
    sid = _FILENAME_RE.sub("_", sid)
    return sid[:64]


def _persist_path(session_id: Optional[str]) -> str:
    return os.path.join(COST_DIR, _safe_session_id(session_id) + ".json")


def pricing_for(model_id: Optional[str]) -> tuple[float, float]:
    """Return ``(input_per_1M, output_per_1M)``; unknown models get (0,0)."""
    return PRICING.get(str(model_id or ""), (0.0, 0.0))


def is_pricing_known(model_id: Optional[str]) -> bool:
    return str(model_id or "") in PRICING


def cost_of(model_id: Optional[str], in_tokens: int,
            out_tokens: int) -> float:
    """USD estimate for a turn; 0.0 when pricing is unknown."""
    in_rate, out_rate = pricing_for(model_id)
    return (in_tokens / 1_000_000) * in_rate + \
           (out_tokens / 1_000_000) * out_rate


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class CostTracker:
    """Per-session token/cost accumulator, persisted best-effort to disk."""

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = _safe_session_id(session_id)
        self._lock = threading.Lock()
        # model_id -> {"turns": int, "in": int, "out": int}
        self._by_model: Dict[str, Dict[str, int]] = {}
        self.load()

    # -- persistence ----------------------------------------------------
    def load(self) -> None:
        path = _persist_path(self.session_id)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                cleaned: Dict[str, Dict[str, int]] = {}
                for mid, row in data.items():
                    if not isinstance(mid, str) or not isinstance(row, dict):
                        continue
                    try:
                        cleaned[mid] = {
                            "turns": max(0, int(row.get("turns", 0) or 0)),
                            "in": max(0, int(row.get("in", 0) or 0)),
                            "out": max(0, int(row.get("out", 0) or 0)),
                        }
                    except (TypeError, ValueError):
                        # one hand-edited/corrupt row must not discard the
                        # whole session (int(list) raised TypeError here)
                        continue
                self._by_model = cleaned
        except (OSError, ValueError, TypeError):
            self._by_model = {}

    def save(self) -> None:
        try:
            os.makedirs(COST_DIR, exist_ok=True)
            path = _persist_path(self.session_id)
            # atomic write: a crash mid-dump (or two writers racing) must
            # never leave truncated JSON that load() silently resets to $0
            tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(self._by_model, fh, ensure_ascii=False,
                              indent=2)
                os.replace(tmp, path)
            except OSError:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError:
            pass  # persistence is best-effort; never break a turn

    # -- accumulation ---------------------------------------------------
    def add_turn(self, model_id: Optional[str], in_tokens: int,
                 out_tokens: int) -> None:
        """Record one completed turn (or completion). Never raises."""
        try:
            mid = str(model_id or "unknown")
            with self._lock:
                row = self._by_model.get(mid)
                if row is None:
                    row = self._by_model[mid] = {"turns": 0, "in": 0, "out": 0}
                row["turns"] += 1
                row["in"] += max(0, int(in_tokens or 0))
                row["out"] += max(0, int(out_tokens or 0))
                self.save()
        except Exception:
            pass

    def reset(self) -> None:
        """Clear this session's counters (persisted as empty)."""
        with self._lock:
            self._by_model = {}
            self.save()

    # -- reporting ------------------------------------------------------
    def totals(self) -> Dict[str, Any]:
        """Session totals: turns, in/out tokens, estimated USD + breakdown."""
        with self._lock:
            rows = {mid: dict(r) for mid, r in self._by_model.items()}
        turns = sum(r["turns"] for r in rows.values())
        in_tokens = sum(r["in"] for r in rows.values())
        out_tokens = sum(r["out"] for r in rows.values())
        cost = sum(cost_of(mid, r["in"], r["out"]) for mid, r in rows.items())
        breakdown = {
            mid: {
                "turns": r["turns"],
                "in_tokens": r["in"],
                "out_tokens": r["out"],
                "cost_usd": round(cost_of(mid, r["in"], r["out"]), 4),
                "pricing_known": is_pricing_known(mid),
            }
            for mid, r in sorted(rows.items())
        }
        return {
            "turns": turns,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
            "cost_usd": round(cost, 4),
            "by_model": breakdown,
        }


# The tracker for module-level helpers; set by register().
_tracker: Optional[CostTracker] = None
_tracker_lock = threading.Lock()


def _get_tracker(session_id: Optional[str] = None) -> CostTracker:
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = CostTracker(session_id)
        return _tracker


def get_cost_tracker() -> CostTracker:
    """Current session's tracker (module-level)."""
    return _get_tracker()


# ---------------------------------------------------------------------------
# TUI handler
# ---------------------------------------------------------------------------

def _fmt_num(n: int) -> str:
    return f"{n:,}"


def handle_cost(ui, arg: str) -> None:
    """TUI ``/cost`` handler: print session token usage and $ estimate."""
    arg = (arg or "").strip()
    if arg in ("-h", "--help", "help"):
        ui.print_info(USAGE)
        return
    try:
        agent = getattr(ui, "agent", None)
        tracker = getattr(agent, "cost_tracker", None) if agent else None
        if tracker is None:
            tracker = _get_tracker()
        if arg == "reset":
            tracker.reset()
            ui.print_info("cost counters reset for this session")
            return
        totals = tracker.totals()
        lines = [
            "session cost",
            f"  turns:     {_fmt_num(totals['turns'])}",
            f"  input:     {_fmt_num(totals['in_tokens'])} tokens",
            f"  output:    {_fmt_num(totals['out_tokens'])} tokens",
            f"  estimated: ${totals['cost_usd']:.2f}",
        ]
        if len(totals["by_model"]) > 1:
            lines.append("  by model:")
            for mid, b in totals["by_model"].items():
                note = "" if b["pricing_known"] else " (pricing unknown)"
                lines.append(
                    f"    {mid}: {_fmt_num(b['turns'])} turns, "
                    f"{_fmt_num(b['in_tokens'])} in / "
                    f"{_fmt_num(b['out_tokens'])} out, "
                    f"${b['cost_usd']:.2f}{note}")
        ui.print_info("\n".join(lines))
    except Exception as e:  # noqa: BLE001 — cost must never crash the TUI
        ui.print_error(f"cost failed: {e}")


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def register(agent: Any) -> None:
    """Attach a per-session CostTracker as ``agent.cost_tracker``.

    WIRE-IN HOOK POINT for the coordinator — in ``agent.py``:
        method ``_emit_cost(self, usage)`` at agent.py:1559 (called from the
        turn loop at agent.py:1065, right after each ``_complete`` returns).
        That is the single place where per-turn ``usage`` (prompt_tokens /
        completion_tokens) and the current model (``self.model.id``) are both
        known. Add one line there, e.g.::

            tracker = getattr(self, "cost_tracker", None)
            if tracker is not None:
                tracker.add_turn(self.model.id, tin, tout)

    ``register`` itself is duck-typed: a plain object with ``session_id``
    is enough, and it never raises.
    """
    global _tracker
    try:
        session_id = getattr(agent, "session_id", None)
        tracker = CostTracker(session_id)
        agent.cost_tracker = tracker
        with _tracker_lock:
            _tracker = tracker
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import tempfile
    from types import SimpleNamespace

    # Isolate the persist dir for the self-test.
    tmp = tempfile.mkdtemp(prefix="cost_selftest_")
    global COST_DIR
    COST_DIR = tmp

    failures = []

    def check(name, cond):
        print(("PASS" if cond else "FAIL") + " " + name)
        if not cond:
            failures.append(name)

    # 1. known free-route pricing
    check("known pricing math",
          cost_of("deepseek-v4.1-flash-free", 500_000, 250_000) == 0.0)
    check("pricing_for known", pricing_for("deepseek-v4.1-flash-free") == (0.0, 0.0))
    check("is_pricing_known true", is_pricing_known("deepseek-v4.1-flash-free"))

    # 2. unknown model handling
    check("unknown -> (0,0)", pricing_for("nope/mystery-9") == (0.0, 0.0))
    check("unknown not known", not is_pricing_known("nope/mystery-9"))
    check("unknown cost 0", cost_of("nope/mystery-9", 1_000_000, 5) == 0.0)

    # 3. accumulation + totals
    agent = SimpleNamespace(session_id="cost-selftest")
    register(agent)
    check("register attaches tracker",
          isinstance(getattr(agent, "cost_tracker", None), CostTracker))
    t = agent.cost_tracker
    t.add_turn("deepseek-v4.1-flash-free", 500_000, 250_000)
    t.add_turn("deepseek-v4.1-flash-free", 500_000, 250_000)
    t.add_turn("nope/mystery-9", 1_000_000, 0)        # $0.00, unknown pricing
    tot = t.totals()
    check("totals turns", tot["turns"] == 3)
    check("totals in", tot["in_tokens"] == 2_000_000)
    check("totals out", tot["out_tokens"] == 500_000)
    check("totals cost", tot["cost_usd"] == 0.0)
    check("breakdown two models", len(tot["by_model"]) == 2)
    check("breakdown flags unknown",
          not tot["by_model"]["nope/mystery-9"]["pricing_known"])
    check("breakdown flags known",
          tot["by_model"]["deepseek-v4.1-flash-free"]["pricing_known"])

    # 4. persistence round-trip with the temp dir
    path = os.path.join(tmp, "cost-selftest.json")
    check("persist file written", os.path.exists(path))
    again = CostTracker("cost-selftest")
    tot2 = again.totals()
    check("reload totals match", tot2["turns"] == 3 and
          tot2["cost_usd"] == 0.0)

    # 5. reset
    t.reset()
    tot3 = t.totals()
    check("reset zeroes", tot3["turns"] == 0 and tot3["cost_usd"] == 0.0)

    # 6. corrupted file never crashes
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{not valid json")
    bad = CostTracker("cost-selftest")
    check("corrupt file -> empty", bad.totals()["turns"] == 0)

    # 7. handler output
    class FakeUI:
        def __init__(self, agent=None):
            self.agent = agent
            self.out = []
            self.errs = []

        def print_info(self, s):
            self.out.append(s)

        def print_error(self, s):
            self.errs.append(s)

    t.add_turn("deepseek-v4.1-flash-free", 1_000_000, 0)
    t.add_turn("nope/mystery-9", 100, 200)
    ui = FakeUI(agent)
    handle_cost(ui, "")
    panel = "\n".join(ui.out)
    check("handler turns line", "turns:" in panel and "2" in panel)
    check("handler estimated $", "estimated:" in panel and "$0.00" in panel)
    check("handler per-model breakdown", "by model:" in panel)
    check("handler unknown note",
          "pricing unknown" in panel or "nope/mystery-9" not in panel)

    # unknown model breakdown note
    t.add_turn("nope/mystery-9", 10, 10)
    ui2 = FakeUI(agent)
    handle_cost(ui2, "")
    panel2 = "\n".join(ui2.out)
    check("handler marks unknown pricing",
          "nope/mystery-9" in panel2 and "pricing unknown" in panel2)

    ui3 = FakeUI(agent)
    handle_cost(ui3, "--help")
    check("handler --help usage", any("usage:" in s for s in ui3.out))

    ui4 = FakeUI(agent)
    handle_cost(ui4, "reset")
    check("handler reset", t.totals()["turns"] == 0)
    check("handler reset message",
          any("reset" in s for s in ui4.out))

    ui5 = FakeUI(None)
    handle_cost(ui5, "")
    check("handler no-agent no crash", bool(ui5.out))

    # 8. importable through the package path used by the feature-modules tuple
    import importlib
    mod = importlib.import_module("fullagent.costtrack")
    check("module importable as fullagent.costtrack", mod is not None)
    check("register callable", callable(mod.register))
    check("handle_cost callable", callable(mod.handle_cost))

    print(("PASS" if not failures else "FAIL") +
          f" self-test ({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
