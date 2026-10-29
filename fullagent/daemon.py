"""DAEMON — Mission Control (autonomous long-running missions).

Missions that outlive a single turn: the daemon owns a mission record in
the event log, advances it one TICK at a time, checkpoints progress after
every tick, and can be resumed from the last checkpoint after any restart.
A mission is a queue of steps; each tick executes the next pending step
through a caller-supplied executor, records the outcome, and checkpoints.

Hard rules (mechanical):
  * All mission state lives in the event log — daemon.mission /
    daemon.checkpoint / daemon.tick / daemon.wake / daemon.done. A daemon
    object keeps no authoritative state; resume() rebuilds everything from
    the fold, so a crash loses at most one in-flight tick.
  * A step that fails is retried up to max_retries, then the mission is
    BLOCKED (never silently skipped) and says exactly where it stopped.
  * Self-wake: wake_conditions are deterministic predicates over the fold
    (e.g. "a verdict failed", "budget event sealed"). due() reports which
    conditions currently hold — the scheduler (or a human) decides when to
    actually run the next tick. The daemon never sleeps in-process.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .kernel import EventLog, fold
from ._foundation import get_logger

_log = get_logger("daemon")

MISSION_STATES = ("RUNNING", "BLOCKED", "DONE", "ABANDONED")
STEP_STATES = ("PENDING", "RUNNING", "DONE", "FAILED", "SKIPPED")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Step:
    id: str
    task: str
    state: str = "PENDING"
    attempts: int = 0
    result: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "task": self.task, "state": self.state,
                "attempts": self.attempts, "result": self.result[:300]}


@dataclass
class Mission:
    mission_id: str
    statement: str
    steps: list[Step] = field(default_factory=list)
    state: str = "RUNNING"
    checkpoint_seq: int = -1     # event seq of the last checkpoint
    ticks: int = 0

    def pending(self) -> Step | None:
        for s in self.steps:
            if s.state == "PENDING":
                return s
        return None

    def progress(self) -> float:
        if not self.steps:
            return 1.0
        done = sum(1 for s in self.steps if s.state == "DONE")
        return done / len(self.steps)


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

class Daemon:
    """Advance missions one tick at a time over the event log.

    `executor` runs one step: executor(step.task) -> str result. A result
    starting with 'ERROR:' counts as a failed attempt."""

    def __init__(self, log: EventLog, executor=None,
                 max_retries: int = 2) -> None:
        self.log = log
        self.executor = executor
        self.max_retries = max(0, int(max_retries))
        # BUG FIX (races): mission ids embed log.head(), and the
        # head-read + seal-append were not atomic — two concurrent
        # start() calls minted the SAME mission id, and resume() then
        # merged both missions' events into one corrupted record.
        # _start_lock makes id minting + sealing atomic.
        self._start_lock = threading.Lock()
        # BUG FIX (races): two concurrent tick() calls on the same
        # mission both resumed the same pending step and executed it
        # TWICE (double side effects, double-counted attempts). Each
        # mission gets its own lock; tick() takes it non-blocking and
        # the loser reports instead of running. Non-blocking (not
        # blocking) so a hung executor can never deadlock a second
        # caller, and an executor that calls back into tick() gets an
        # error dict instead of a deadlock.
        self._locks_lock = threading.Lock()
        self._tick_locks: dict[str, threading.Lock] = {}

    # -- mission lifecycle -----------------------------------------------------

    def _tick_lock(self, mission_id: str) -> threading.Lock:
        with self._locks_lock:
            return self._tick_locks.setdefault(mission_id,
                                               threading.Lock())

    def start(self, statement: str, tasks: list[str]) -> Mission:
        """Seal a new mission. Step ids are M1..Mn."""
        # SPEED-EXPOSURE FIX: the old millisecond-clock id could collide
        # when missions are started back-to-back (now that everything
        # runs faster); seq is strictly monotonic, so this never can.
        # ...as long as the head-read and the seal are atomic: without
        # the lock, two threads starting missions at once read the same
        # head and minted duplicate ids.
        with self._start_lock:
            mission_id = f"mission-{self.log.head() + 1}"
            steps = [Step(id=f"M{i + 1}", task=t)
                     for i, t in enumerate(tasks)]
            self.log.append("daemon.mission",
                            {"mission_id": mission_id, "statement": statement,
                             "steps": [s.to_dict() for s in steps],
                             "state": "RUNNING"},
                            actor="daemon")
        return Mission(mission_id, statement, steps)

    def tick(self, mission_id: str) -> dict:
        """Execute the next pending step, seal the outcome, checkpoint.

        Returns {mission_id, step, state, result, progress}. A step that
        exhausts its retries BLOCKS the mission — it is never skipped.
        If another tick is already in flight for this mission, this call
        reports {"error": "tick already in progress"} instead of
        executing the step a second time."""
        lock = self._tick_lock(mission_id)
        if not lock.acquire(blocking=False):
            return {"mission_id": mission_id,
                    "error": "tick already in progress for this mission"}
        try:
            return self._tick(mission_id)
        finally:
            lock.release()

    def _tick(self, mission_id: str) -> dict:
        """One serialized tick — see tick() for the contract."""
        m = self.resume(mission_id)
        if m is None:
            return {"mission_id": mission_id, "error": "no such mission"}
        if m.state != "RUNNING":
            return {"mission_id": mission_id, "state": m.state,
                    "error": f"mission is {m.state}, not RUNNING"}
        step = m.pending()
        if step is None:
            self._close(m, "DONE")
            return {"mission_id": mission_id, "state": "DONE",
                    "progress": 1.0}

        step.attempts += 1
        result = ""
        if self.executor is not None:
            try:
                result = str(self.executor(step.task))
            except Exception as e:
                result = f"ERROR: {type(e).__name__}: {e}"
        else:
            result = "ERROR: no executor attached"

        failed = result.startswith("ERROR:")
        step.state = "FAILED" if failed else "DONE"
        step.result = result
        m.ticks += 1

        self.log.append("daemon.tick",
                        {"mission_id": mission_id, "step": step.to_dict(),
                         "failed": failed, "attempt": step.attempts},
                        actor="daemon")

        if failed and step.attempts > self.max_retries:
            # retries exhausted — the mission blocks here, visibly.
            # _patch_state already seals the checkpoint; a second one
            # would just be a duplicate event in the log.
            step.state = "FAILED"
            self._patch_state(m, "BLOCKED")
            return {"mission_id": mission_id, "step": step.id,
                    "state": "BLOCKED", "result": result,
                    "progress": round(m.progress(), 3)}

        if failed:
            step.state = "PENDING"  # retry on the next tick
        if m.pending() is None and not failed:
            # last step done — seal daemon.done directly; a checkpoint
            # immediately before it would be a redundant event
            self._close(m, "DONE")
            return {"mission_id": mission_id, "step": step.id,
                    "state": "DONE", "result": result, "progress": 1.0}
        self._checkpoint(m)
        return {"mission_id": mission_id, "step": step.id,
                "state": "RUNNING", "result": result,
                "progress": round(m.progress(), 3)}

    def abandon(self, mission_id: str, reason: str = "") -> bool:
        # BUG FIX (race): abandoning while a tick is mid-flight let the
        # tick seal RUNNING step state AFTER us and resurrect the
        # mission. Take the mission's tick lock (non-blocking — a hung
        # executor must not wedge abandon); on contention report False
        # so the caller retries once the tick lands.
        lock = self._tick_lock(mission_id)
        if not lock.acquire(blocking=False):
            _log.warning("abandon deferred: tick in flight for %s",
                         mission_id)
            return False
        try:
            m = self.resume(mission_id)
            if m is None or m.state == "DONE":
                return False
            self._patch_state(m, "ABANDONED")
            self.log.append("daemon.done",
                            {"mission_id": mission_id, "state": "ABANDONED",
                             "reason": reason}, actor="human")
            return True
        finally:
            lock.release()

    # -- checkpoints + resume ----------------------------------------------------

    def _checkpoint(self, m: Mission) -> None:
        ev = self.log.append("daemon.checkpoint",
                             {"mission_id": m.mission_id,
                              "steps": [s.to_dict() for s in m.steps],
                              "ticks": m.ticks, "state": m.state},
                             actor="daemon")
        m.checkpoint_seq = ev.seq

    def _patch_state(self, m: Mission, state: str) -> None:
        m.state = state
        self.log.append("daemon.checkpoint",
                        {"mission_id": m.mission_id,
                         "steps": [s.to_dict() for s in m.steps],
                         "ticks": m.ticks, "state": state},
                        actor="daemon")

    def _close(self, m: Mission, state: str) -> None:
        # the done event carries the final step states, so resume()
        # rebuilds correctly without a redundant checkpoint first
        m.state = state
        self.log.append("daemon.done",
                        {"mission_id": m.mission_id, "state": state,
                         "ticks": m.ticks,
                         "steps": [s.to_dict() for s in m.steps],
                         "progress": round(m.progress(), 3)},
                        actor="daemon")

    def _mission_index(self) -> dict[str, list[dict]]:
        """One pass over the fold's daemon events -> {mission_id: events}.

        SPEED: resume()/missions() used to re-scan every daemon event per
        mission (O(missions x events)); the index is built once and each
        mission replays only its own events."""
        index: dict[str, list[dict]] = {}
        for e in fold(self.log).daemon_events:
            mid = e.get("mission_id")
            if mid:
                index.setdefault(mid, []).append(e)
        return index

    def resume(self, mission_id: str) -> Mission | None:
        """Rebuild a mission purely from the fold — the daemon's crash
        recovery. Latest checkpoint wins; ticks replay step states."""
        return self._resume_from(self._mission_index().get(mission_id))

    def _resume_from(self, evs: list[dict] | None) -> Mission | None:
        if not evs:
            return None
        base = None
        for e in evs:
            if e.get("type") == "daemon.mission":
                base = e
                break  # mission ids are unique; first seal is the base
        if base is None:
            return None
        mission_id = str(base.get("mission_id", ""))
        steps = [Step(id=s["id"], task=s["task"], state=s.get("state") or "PENDING",
                      attempts=int(s.get("attempts", 0)),
                      result=s.get("result", ""))
                 for s in base.get("steps") or []]
        m = Mission(mission_id, str(base.get("statement", "")), steps,
                    state="RUNNING")
        # replay checkpoints and ticks in sealed order
        for e in evs:
            if e.get("type") == "daemon.checkpoint":
                m.ticks = int(e.get("ticks", m.ticks))
                m.state = str(e.get("state", m.state))
                by_id = {s["id"]: s for s in e.get("steps") or []}
                for s in m.steps:
                    if s.id in by_id:
                        s.state = by_id[s.id].get("state", s.state)
                        s.attempts = int(by_id[s.id].get("attempts",
                                                         s.attempts))
                        s.result = by_id[s.id].get("result", s.result)
            elif e.get("type") == "daemon.done":
                m.state = str(e.get("state", m.state))
                # the done event carries the terminal step states (sealed
                # instead of a redundant final checkpoint)
                by_id = {s["id"]: s for s in e.get("steps") or []}
                for s in m.steps:
                    if s.id in by_id:
                        s.state = by_id[s.id].get("state", s.state)
                        s.attempts = int(by_id[s.id].get("attempts",
                                                         s.attempts))
                        s.result = by_id[s.id].get("result", s.result)
        return m

    def missions(self) -> list[dict]:
        """One summary row per mission, newest first."""
        index = self._mission_index()
        # newest first: mission ids embed the head seq at creation
        # ("mission-<seq>"), so reverse-sealed-order == newest first
        ids = [mid for mid, evs in index.items()
               if any(e.get("type") == "daemon.mission" for e in evs)]
        rows = []
        for mid in reversed(ids):
            m = self._resume_from(index[mid])
            if m is None:
                continue
            rows.append({"mission_id": mid, "statement": m.statement,
                         "state": m.state, "ticks": m.ticks,
                         "progress": round(m.progress(), 3),
                         "steps": len(m.steps)})
        return rows

    # -- self-wake -----------------------------------------------------------------

    def wake_conditions(self) -> list[str]:
        """Deterministic predicates over the fold that justify waking the
        daemon for another tick. The daemon itself never sleeps or polls —
        this only REPORTS what currently holds."""
        st = fold(self.log)
        reasons: list[str] = []
        running = [r for r in self.missions() if r["state"] == "RUNNING"]
        if running:
            reasons.append(f"{len(running)} mission(s) RUNNING with "
                           "pending steps")
        failed_verdicts = sum(1 for v in st.verdicts if v.get("passed") is False)
        if failed_verdicts:
            reasons.append(f"{failed_verdicts} failed verdict(s) to react to")
        if st.budget_events:
            reasons.append("budget event sealed — re-evaluate missions")
        return reasons

    def due(self) -> bool:
        return bool(self.wake_conditions())

    def format_status(self) -> str:
        rows = self.missions()
        lines = ["DAEMON — mission control"]
        if not rows:
            lines.append("  no missions")
        for r in rows[:8]:
            bar = int(r["progress"] * 12)
            lines.append(f"  {r['mission_id']}  [{r['state']:<9}] "
                         f"{'█' * bar}{'░' * (12 - bar)} "
                         f"{r['progress']:.0%}  {r['ticks']} ticks  "
                         f"{r['statement'][:36]}")
        wakes = self.wake_conditions()
        if wakes:
            lines.append("  wake: " + "; ".join(wakes))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "daemon.jsonl")

        results = {"step two": "ERROR: transient failure"}

        def executor(task: str) -> str:
            return results.get(task, f"OK: {task}")

        d = Daemon(log, executor, max_retries=1)

        # start a 3-step mission
        m = d.start("ship the parser", ["step one", "step two", "step three"])
        assert m.mission_id and len(m.steps) == 3

        # tick 1: step one done
        r = d.tick(m.mission_id)
        assert r["state"] == "RUNNING" and r["step"] == "M1", r

        # tick 2: step two fails (attempt 1) -> stays RUNNING for retry
        r = d.tick(m.mission_id)
        assert r["result"].startswith("ERROR:"), r
        m2 = d.resume(m.mission_id)
        assert m2.state == "RUNNING", m2.state

        # tick 3: transient failure cleared -> step two done
        results["step two"] = "OK: recovered"
        r = d.tick(m.mission_id)
        assert r["step"] == "M2" and not r["result"].startswith("ERROR:"), r

        # tick 4: step three done -> mission DONE
        r = d.tick(m.mission_id)
        assert r["state"] == "DONE" and r["progress"] == 1.0, r

        # resume rebuilds the finished mission from the fold alone
        m3 = d.resume(m.mission_id)
        assert m3.state == "DONE" and m3.progress() == 1.0
        assert all(s.state == "DONE" for s in m3.steps)

        # a permanently failing step BLOCKS the mission after retries
        d2 = Daemon(log, lambda t: "ERROR: permanent", max_retries=1)
        m4 = d2.start("doomed", ["only step"])
        d2.tick(m4.mission_id)          # attempt 1 fails -> retry
        r = d2.tick(m4.mission_id)      # attempt 2 fails -> BLOCKED
        assert r["state"] == "BLOCKED", r
        assert d2.resume(m4.mission_id).state == "BLOCKED"

        # abandon works on a blocked mission
        assert d2.abandon(m4.mission_id, "giving up")
        assert d2.resume(m4.mission_id).state == "ABANDONED"

        # wake conditions reflect running missions + failed verdicts
        d3 = Daemon(log, executor)
        m5 = d3.start("long mission", ["a", "b"])
        assert d3.due() is True
        assert any("RUNNING" in w for w in d3.wake_conditions())

        # missions() lists newest first with progress
        rows = d3.missions()
        assert rows[0]["mission_id"] == m5.mission_id
        assert "DAEMON" in d3.format_status()

        # everything is sealed in the ledger
        evs = fold(log).daemon_events
        types = {e["type"] for e in evs}
        assert {"daemon.mission", "daemon.checkpoint", "daemon.tick",
                "daemon.done"} <= types

        # concurrent start() calls mint UNIQUE mission ids (the head-read
        # + seal used to race and mint duplicates)
        import threading as _th
        d4 = Daemon(log, executor)
        ids: list[str] = []
        _start_barrier = _th.Barrier(8)

        def _start_one(i: int) -> None:
            _start_barrier.wait()
            ids.append(d4.start(f"race {i}", ["s"]).mission_id)

        _starters = [_th.Thread(target=_start_one, args=(i,))
                       for i in range(8)]
        for t in _starters:
            t.start()
        for t in _starters:
            t.join()
        assert len(set(ids)) == 8, ids

        # concurrent tick() calls on one mission execute the step ONCE —
        # the loser reports instead of double-running side effects
        ran = _th.Event()
        calls = {"n": 0}
        _tick_barrier = _th.Barrier(4)

        def slow_executor(task: str) -> str:
            calls["n"] += 1
            ran.wait(5)
            return "OK: slow"

        d5 = Daemon(log, slow_executor)
        m6 = d5.start("tick race", ["only"])
        tick_results: list[dict] = []

        def _tick_one() -> None:
            _tick_barrier.wait()
            tick_results.append(d5.tick(m6.mission_id))

        _tickers = [_th.Thread(target=_tick_one) for _ in range(4)]
        for t in _tickers:
            t.start()
        ran.wait(2)  # let the winner get deep into the executor...
        ran.set()     # ...then release it
        for t in _tickers:
            t.join()
        assert calls["n"] == 1, calls
        assert sum(1 for r in tick_results
                   if r.get("error") == "tick already in progress "
                   "for this mission") == 3, tick_results

    print("DAEMON SELF-TEST PASS")
