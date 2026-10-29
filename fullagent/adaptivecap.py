"""Adaptive per-turn tool-call cap (permanent-fix sprint, worker 8/20).

Why: the fixed 25-call cap is dumb — fast calls (<5s avg) could safely do
more, and slow calls (>15s avg) should stop earlier to respect the user's
time. This tracker keeps a rolling (exponentially-weighted) average of
per-call wall time plus a running total of tool-call time for the turn.
After every finished tool call the agent asks it ``note(duration_s)``; a
non-None return is the stop message, which says WHY the turn stopped
(call-count cap vs time budget) and that "continue" resumes it.

Scheduler yields are exempt: ``note(..., waiting_on_background=True)``
for TaskCollect/TaskOutput waits on already-running subagents. Those
waits are not tool execution, so they consume neither the call cap nor
the time budget — a parent turn must stay alive while its own parallel
workers finish.

Tiers (tunables in config.py), evaluated live after every call:
  avg < FAST_CALL_AVG_THRESHOLD_S  -> ADAPTIVE_CAP_FAST calls, TURN_TIME_BUDGET_S
  avg > SLOW_CALL_AVG_THRESHOLD_S  -> ADAPTIVE_CAP_SLOW calls, TURN_TIME_BUDGET_SLOW_S
  otherwise                        -> MAX_TOOL_CALLS_PER_TURN calls, TURN_TIME_BUDGET_S
The turn stops at whichever of the call cap or the time budget hits
first. The tier can move mid-turn: one slow call does not permanently
punish the rest of the turn, and a turn that slows down gets reined in.

Compatibility: stop messages keep the "Turn stopped after" prefix and a
"tool call(s)" mention so continueresume._is_capped_stop() still
recognises them as capped stops, and agent.py sets turn.cap_stopped so
the worker-7/20 compaction still fires.

Only stdlib. Thread-safe (cheap lock; _finish_one replays sequentially
today, but parallel batches make the guarantee worth having).
"""

from __future__ import annotations

import math
import threading

from . import config

__all__ = ["AdaptiveTurnCap"]

_EMA_ALPHA = 0.5  # rolling-average weight of the newest call duration


class AdaptiveTurnCap:
    """Per-turn tool-call cap that adapts to observed call speed.

    ``note(duration_s) -> str | None`` — call once per finished tool
    call with its wall time (ev.duration). Returns None while the turn
    may continue, or the stop message when the cap trips. All tunables
    default to config.py values and can be overridden for tests.
    """

    def __init__(self,
                 base_cap: int | None = None,
                 fast_cap: int | None = None,
                 slow_cap: int | None = None,
                 fast_threshold_s: float | None = None,
                 slow_threshold_s: float | None = None,
                 budget_s: float | None = None,
                 slow_budget_s: float | None = None,
                 ema_alpha: float = _EMA_ALPHA):
        self.base_cap = base_cap if base_cap is not None \
            else config.MAX_TOOL_CALLS_PER_TURN
        self.fast_cap = fast_cap if fast_cap is not None \
            else config.ADAPTIVE_CAP_FAST
        self.slow_cap = slow_cap if slow_cap is not None \
            else config.ADAPTIVE_CAP_SLOW
        self.fast_threshold_s = fast_threshold_s if fast_threshold_s is not None \
            else config.FAST_CALL_AVG_THRESHOLD_S
        self.slow_threshold_s = slow_threshold_s if slow_threshold_s is not None \
            else config.SLOW_CALL_AVG_THRESHOLD_S
        self.budget_s = budget_s if budget_s is not None \
            else config.TURN_TIME_BUDGET_S
        self.slow_budget_s = slow_budget_s if slow_budget_s is not None \
            else config.TURN_TIME_BUDGET_SLOW_S
        self.ema_alpha = ema_alpha
        self._lock = threading.Lock()
        self.calls = 0
        self.total_time = 0.0
        self._avg: float | None = None
        self._stopped = False
        self._stop_message: str | None = None

    # -- internals ----------------------------------------------------

    def _limits(self) -> tuple[int, float, str]:
        """(call_cap, time_budget_s, tier_name) for the current average."""
        avg = self._avg if self._avg is not None else 0.0
        if avg < self.fast_threshold_s:
            return self.fast_cap, self.budget_s, "fast"
        if avg > self.slow_threshold_s:
            return self.slow_cap, self.slow_budget_s, "slow"
        return self.base_cap, self.budget_s, "normal"

    @staticmethod
    def _clean_duration(duration_s) -> float:
        try:
            d = float(duration_s)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(d) or d < 0:
            return 0.0
        return d

    # -- public API ---------------------------------------------------

    def status(self) -> dict:
        """Snapshot for tests / debugging / status lines."""
        with self._lock:
            cap, budget, tier = self._limits()
            return {
                "calls": self.calls,
                "avg_call_s": round(self._avg, 3)
                if self._avg is not None else 0.0,
                "total_tool_time_s": round(self.total_time, 3),
                "call_cap": cap,
                "time_budget_s": budget,
                "tier": tier,
                "stopped": self._stopped,
            }

    def note(self, duration_s: float, *,
             waiting_on_background: bool = False) -> str | None:
        """Record one finished tool call. Returns the stop message once
        the cap trips, else None. After tripping, keeps returning the
        same message without counting further (idempotent latch).

        ``waiting_on_background=True`` marks a scheduler-yield call (a
        TaskCollect/TaskOutput wait on already-running work). Such a
        call is not tool execution: it neither consumes the call cap nor
        the time budget. Without this, eight parallel subagents could
        exhaust the 120s slow budget while they are still working and
        stop the parent turn before it can collect their reports."""
        d = self._clean_duration(duration_s)
        if waiting_on_background:
            return None
        with self._lock:
            if self._stopped:
                return self._stop_message
            self.calls += 1
            self.total_time += d
            if self._avg is None:
                self._avg = d
            else:
                self._avg += self.ema_alpha * (d - self._avg)
            cap, budget, _tier = self._limits()
            assert self._avg is not None
            avg = self._avg
            noun = "tool call" if self.calls == 1 else "tool calls"
            reasons: list[str] = []
            if self.calls >= cap:
                reasons.append(
                    f"call cap of {cap} reached "
                    f"(avg call time {avg:.1f}s)")
            if self.total_time >= budget:
                reasons.append(
                    f"tool-time budget of {budget:.0f}s exhausted "
                    f"({self.total_time:.0f}s of tool time, "
                    f"avg {avg:.1f}s/call)")
            if reasons:
                self._stopped = True
                # "Turn stopped after" prefix + "tool calls" mention keep
                # continueresume._is_capped_stop() matching, so the
                # "continue" flow and cap compaction keep working.
                self._stop_message = (
                    f"Turn stopped after {self.calls} {noun} — "
                    + " and ".join(reasons)
                    + ". Say 'continue' to resume from where it left off.")
                return self._stop_message
            return None


if __name__ == "__main__":
    # self-test — run with: python -m fullagent.adaptivecap
    # Simulates fast / slow / mixed call-time profiles and proves the
    # cap adapts instead of always stopping at 25.
    failures: list[str] = []

    def check(label, cond, detail=""):
        print(f"{'PASS' if cond else 'FAIL'}: {label}"
              + (f" [{detail}]" if detail else ""))
        if not cond:
            failures.append(label)

    def drive(durations, **kw):
        """Feed durations through a fresh cap; return (stop_at, message).

        Scenarios run against the CLASSIC small caps (explicit overrides)
        so the arithmetic stays deterministic no matter what marathon
        scale the shipped config defaults to.
        """
        kw.setdefault("base_cap", 25)
        kw.setdefault("fast_cap", 40)
        kw.setdefault("slow_cap", 12)
        kw.setdefault("budget_s", 180.0)
        kw.setdefault("slow_budget_s", 120.0)
        cap = AdaptiveTurnCap(**kw)
        msg = None
        for i, d in enumerate(durations, 1):
            msg = cap.note(d)
            if msg is not None:
                return i, msg, cap.status()
        return None, None, cap.status()

    # 0. shipped defaults are marathon-scale (long-run mode): a long
    # autonomous session must not hit an artificial mid-task stop.
    check("default config is marathon-scale",
          config.MAX_TOOL_CALLS_PER_TURN >= 200
          and config.ADAPTIVE_CAP_FAST >= 400
          and config.ADAPTIVE_CAP_SLOW >= 120
          and config.TURN_TIME_BUDGET_S >= 3600
          and config.TURN_TIME_BUDGET_SLOW_S >= 1800,
          f"{config.MAX_TOOL_CALLS_PER_TURN}/{config.TURN_TIME_BUDGET_S}")

    FAST = config.ADAPTIVE_CAP_FAST          # 400
    SLOW = config.ADAPTIVE_CAP_SLOW          # 120
    BASE = config.MAX_TOOL_CALLS_PER_TURN    # 200
    BUDGET = config.TURN_TIME_BUDGET_S       # 3600
    SLOW_BUDGET = config.TURN_TIME_BUDGET_SLOW_S  # 1800

    # 1. fast calls (2s avg): must sail PAST the old dumb 25 and stop at
    # the fast cap of 40 — with a call-count reason, not a budget reason.
    stop_at, msg, st = drive([2.0] * 60)
    check("fast 2s calls do not stop at the old 25",
          stop_at is not None and stop_at > 25, f"stopped at {stop_at}")
    check("fast 2s calls stop at the fast cap 40",
          stop_at == 40, f"stopped at {stop_at}")
    check("fast stop cites the call cap",
          msg is not None and "call cap of 40 reached" in msg, msg or "")
    check("fast stop invites continue",
          msg is not None and "continue" in msg.lower(), "")
    check("fast tier reported", st["tier"] == "fast", st["tier"])

    # 2. slow calls (20s avg): must stop at the 120s budget (call 6),
    # well before the slow 12-call cap — time is the binding limit.
    stop_at, msg, st = drive([20.0] * 60)
    check("slow 20s calls stop at the 120s budget, not at 25",
          stop_at is not None and stop_at == 6, f"stopped at {stop_at}")
    check("slow stop cites the time budget",
          msg is not None and "tool-time budget of 120s exhausted" in msg,
          msg or "")
    check("slow tier reported", st["tier"] == "slow", st["tier"])

    # 3. normal calls (10s avg): base tier — 25 calls AND 180s budget,
    # budget binds first at call 18 (180/10).
    stop_at, msg, st = drive([10.0] * 60)
    check("normal 10s calls stop before the 25-call cap via budget",
          stop_at == 18, f"stopped at {stop_at}")
    check("normal stop cites the time budget",
          msg is not None and "tool-time budget of 180s exhausted" in msg,
          msg or "")
    check("normal tier reported", st["tier"] == "normal", st["tier"])

    # 4. mixed: one slow call (60s) then fast calls (1s) — the cap must
    # ADAPT back up to the fast tier (40 calls) instead of staying
    # punished. Total ~99s < 180s budget, so the count cap binds.
    stop_at, msg, st = drive([60.0] + [1.0] * 60)
    check("mixed slow-then-fast adapts back up to the 40-call fast cap",
          stop_at == 40, f"stopped at {stop_at}")
    check("mixed stop cites the call cap",
          msg is not None and "call cap of 40 reached" in msg, msg or "")

    # 5. mixed the other way: fast calls then slow ones — the cap must
    # tighten: with the average >15s the slow 12-call cap binds (12
    # calls hit while only 60s of the 120s budget is spent).
    stop_at, msg, st = drive([1.0] * 10 + [25.0] * 60)
    check("fast-then-slow tightens to the 12-call slow cap",
          stop_at == 12, f"stopped at {stop_at}")
    check("fast-then-slow cites the call cap",
          msg is not None and "call cap of 12 reached" in msg, msg or "")
    check("fast-then-slow reports the slow tier",
          st["tier"] == "slow", st["tier"])
    # 5b. the slow BUDGET also binds: 20s calls from the start spend
    # the 120s budget at call 6, long before any call cap.
    stop_at, msg, st = drive([20.0] * 60)
    check("slow budget binds before the slow call cap",
          stop_at == 6 and "tool-time budget of 120s exhausted" in msg,
          f"stopped at {stop_at}")

    # 6. single huge call: one 200s call exceeds the 180s budget alone.
    stop_at, msg, st = drive([200.0])
    check("single 200s call trips the budget immediately",
          stop_at == 1, f"stopped at {stop_at}")
    check("singular 'tool call' grammar", "1 tool call —" in (msg or ""),
          msg or "")

    # 7. poison durations never crash the tracker.
    cap = AdaptiveTurnCap()
    for bad in (-3.0, None, "junk", float("nan"), float("inf")):
        r = cap.note(bad)
        check(f"bad duration {bad!r} treated as 0, no stop", r is None,
              f"calls={cap.calls}")
    check("tracker still functional after poison input",
          cap.note(1.0) is None and cap.calls == 6, "")

    # 8. idempotent latch: after tripping, note() keeps returning the
    # same message without counting further.
    cap = AdaptiveTurnCap(base_cap=2, fast_cap=2, slow_cap=2)
    cap.note(1.0)
    m1 = cap.note(1.0)
    m2 = cap.note(1.0)
    check("stops at custom base_cap=2", m1 is not None, m1 or "")
    check("latch: same message, no extra counting",
          m2 == m1 and cap.calls == 2, f"calls={cap.calls}")

    # 8b. background waits are scheduler yields, not tool work: repeated
    # 90s TaskCollect waits while 8 subagents are still running must not
    # consume the call cap or trip the time budget. Real tool calls after
    # them still count normally.
    cap = AdaptiveTurnCap(base_cap=3, fast_cap=3, slow_cap=3,
                          budget_s=0.5, slow_budget_s=0.5)
    waits = [cap.note(90.0, waiting_on_background=True) for _ in range(8)]
    check("background waits never stop the turn",
          all(m is None for m in waits), waits)
    check("background waits consume no calls or tool time",
          cap.calls == 0 and cap.total_time == 0.0,
          f"calls={cap.calls}, total={cap.total_time}")
    check("real tool after background waits still counts",
          cap.note(0.1) is None and cap.calls == 1, f"calls={cap.calls}")
    check("real slow tool after background waits still trips budget",
          cap.note(1.0) is not None, f"calls={cap.calls}")

    # 9. boundary tiers: exactly 5.0 -> normal (not fast); exactly
    # 15.0 -> normal (not slow).
    cap = AdaptiveTurnCap()
    cap.note(5.0)
    check("avg == 5.0 is the normal tier", cap.status()["tier"] == "normal",
          cap.status()["tier"])
    cap = AdaptiveTurnCap()
    cap.note(15.0)
    check("avg == 15.0 is the normal tier", cap.status()["tier"] == "normal",
          cap.status()["tier"])

    # 10. message compatibility with the continue/resume flow.
    from .continueresume import _is_capped_stop
    _, m_fast, _ = drive([2.0] * 60)
    _, m_slow, _ = drive([20.0] * 60)
    check("fast stop recognised as capped by continueresume",
          _is_capped_stop(m_fast), "")
    check("slow stop recognised as capped by continueresume",
          _is_capped_stop(m_slow), "")

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        raise SystemExit(1)
    print("ADAPTIVE CAP SELF-TEST PASS")
