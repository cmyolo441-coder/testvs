"""CORTEX — orchestration (§13).

The plan is a typed DAG; every node carries a predicate, a risk class, a
path set (the write-exclusivity key), and cost estimates. Three hard
mechanisms live here, all rung 1 (pure Python, free, deterministic):

  * Write-exclusivity (invariant I7): two WRITE nodes with overlapping
    path sets can never be scheduled concurrently. Reads fan out, writes
    serialise (§16.1).
  * Hierarchical budget governor (§13.5): a run budget with slices per
    subtree. Breaching any axis PAUSES — never silently kills — and emits
    budget.event. A child operation cannot consume its parent's budget.
  * Loop / thrash / oscillation detectors (§13.4): exact-repeat tool
    calls, file-content A-B-A-B oscillation, and cost-slope breaches are
    detected from the event log and sealed as loop.alert events.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

from .kernel import EventLog, fold
from ._foundation import get_logger

_log = get_logger("cortex")

NODE_KINDS = ("READ", "WRITE", "EXEC", "VERIFY", "ASK", "RESEARCH",
              "REFACTOR")
NODE_STATUSES = ("PENDING", "RUNNING", "PASSED", "FAILED", "SKIPPED",
                 "ROLLED_BACK", "BRANCHED")
RISK_LEVELS = ("SAFE", "GUARDED", "DESTRUCTIVE", "IRREVERSIBLE")


def _canonical_args(name: str, args: dict) -> str:
    try:
        payload = json.dumps({"name": name, "args": args}, sort_keys=True,
                             ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        # circular references or exotic objects would otherwise raise and
        # take down the loop detector — fall back to a repr-based hash
        try:
            items = sorted((str(k), repr(v))
                           for k, v in (args or {}).items())
        except Exception:
            items = [("unhashable", repr(type(args)))]
        payload = repr((name, items))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Plan node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    id: str
    goal: str
    kind: str = "READ"
    predicate: dict | None = None       # PredicateSpec (§19)
    depends_on: list[str] = field(default_factory=list)
    path_set: list[str] = field(default_factory=list)  # write-exclusivity
    risk: str = "SAFE"
    clause_id: str | None = None        # attribution (§38.1)
    est_cost_usd: float = 0.0
    est_steps: int = 1
    status: str = "PENDING"
    attempts: int = 0

    def to_dict(self) -> dict:
        return {"id": self.id, "goal": self.goal, "kind": self.kind,
                "predicate": self.predicate, "depends_on": self.depends_on,
                "path_set": self.path_set, "risk": self.risk,
                "clause_id": self.clause_id, "est_cost_usd": self.est_cost_usd,
                "est_steps": self.est_steps, "status": self.status,
                "attempts": self.attempts}


# ---------------------------------------------------------------------------
# Plan DAG
# ---------------------------------------------------------------------------

class Plan:
    """A typed DAG over the event log. Nodes are sealed as plan.node events;
    status changes as plan.node.status. The frontier is a pure fold."""

    def __init__(self, log: EventLog) -> None:
        self.log = log

    def add(self, node: Node) -> Node:
        if node.kind not in NODE_KINDS:
            raise ValueError(f"node kind must be one of {NODE_KINDS}")
        if node.risk not in RISK_LEVELS:
            raise ValueError(f"risk must be one of {RISK_LEVELS}")
        self.log.append("plan.node", node.to_dict(), actor="sovereign",
                        correlation_id=node.clause_id)
        return node

    def set_status(self, node_id: str, status: str) -> None:
        if status not in NODE_STATUSES:
            raise ValueError(f"status must be one of {NODE_STATUSES}")
        self.log.append("plan.node.status", {"id": node_id,
                                             "status": status},
                        actor="kernel")

    def nodes(self) -> dict[str, dict]:
        return fold(self.log).nodes

    def frontier(self) -> list[dict]:
        """Nodes whose depends_on are all PASSED and that are PENDING.
        This is the parallel frontier the scheduler draws from (§16.5)."""
        nodes = self.nodes()
        out: list[dict] = []
        for n in nodes.values():
            if n.get("status") != "PENDING":
                continue
            deps = n.get("depends_on") or []
            if all(nodes.get(d, {}).get("status") == "PASSED" for d in deps):
                out.append(n)
        return out

    def eligible(self, max_parallel: int = 8) -> list[dict]:
        """The frontier filtered by write-exclusivity (I7): no two WRITE
        nodes with overlapping path sets, and no more than max_parallel.
        Deterministic ordering keeps the selection reproducible.

        I7 holds ACROSS scheduler ticks, not just within one call: WRITE
        nodes already RUNNING seed the lock set, so a second overlapping
        write scheduled on the next tick can never run concurrently.
        A WRITE with an empty path_set has unknown scope and is treated
        conservatively — it serialises against every other write."""
        frontier = sorted(self.frontier(), key=lambda n: n.get("id", ""))
        # seed locks from writes already in flight (cross-tick I7)
        locked_paths: set[str] = set()
        lock_all = False
        for n in self.nodes().values():
            if n.get("status") == "RUNNING" and n.get("kind") == "WRITE":
                ps = set(n.get("path_set") or [])
                if ps:
                    locked_paths |= ps
                else:
                    lock_all = True
        chosen: list[dict] = []
        for n in frontier:
            if len(chosen) >= max_parallel:
                break
            if n.get("kind") == "WRITE":
                paths = set(n.get("path_set") or [])
                if lock_all or (paths & locked_paths) \
                        or (not paths and locked_paths):
                    continue  # would overlap an already-scheduled write
                if paths:
                    locked_paths |= paths
                else:
                    lock_all = True
            chosen.append(n)
        return chosen


# ---------------------------------------------------------------------------
# Budget governor (§13.5)
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    """Run budget. Defaults are UNLIMITED on every axis: the run never
    pauses for spend — the governor machinery stays (events, /budget,
    per-slice caps when a caller passes explicit numbers), but nothing
    stops unless a human sets a limit with /budget set."""
    max_usd: float = math.inf
    max_steps: int = 1_000_000_000
    max_tokens: int = 1_000_000_000_000
    max_files: int = 100_000_000
    slices: dict[str, float] = field(default_factory=dict)  # subtree -> fraction


class BudgetGovernor:
    """Hierarchical budget over the event log. Every check is a fold; every
    breach is a budget.event that PAUSES the run (never silently kills).

    Spend is SESSION-SCOPED: the fold starts at the latest session.start,
    so a fresh session always starts with a fresh budget. Without this the
    counter accumulates across sessions and, once max_steps is crossed,
    every future turn is paused forever — even a trivial "hi"."""

    def __init__(self, log: EventLog, budget: Budget | None = None) -> None:
        self.log = log
        self.budget = budget or Budget()
        self.baseline_seq = -1  # reset() anchor: ignore events <= this
        self._last_reason = ""  # dedupe budget.event spam while paused
        # (branch, head, start): where the latest session.start scan ended
        self._session_cache: tuple[str, int, int] | None = None

    def _session_start(self) -> int:
        """seq of the latest session.start (-1 if none) — where the current
        session's spend begins. -1 (not 0) so seq-0 events still count.

        Cached against (branch, head). Steady-state appends only move the
        head forward, so we rescan just the delta appended since the
        cached head (O(delta), reversed, stopping at the cached head)
        instead of walking the whole session backwards on every spend() —
        the old code did the full walk on every call, and enforce()
        runs per turn-loop iteration, so this was O(session) per turn.
        A rewind seals a kernel.rewind MARKER (the head moves forward
        while the chain moves back), so a rewind marker in the delta, a
        backwards-moving head, or a branch switch all fall back to the
        full backward scan."""
        branch = self.log.branch
        head = self.log.head()
        cached = self._session_cache
        if cached is not None:
            c_branch, c_head, c_start = cached
            if c_branch == branch and c_head == head:
                return c_start
            if c_branch == branch and head > c_head:
                start = c_start
                full_rescan = False
                # reversed: newest first, stop at the cached head
                for ev in reversed(self.log.events()):
                    if ev.seq <= c_head:
                        break
                    if ev.type == "kernel.rewind":
                        full_rescan = True
                        break
                    if ev.type == "session.start":
                        start = max(start, ev.seq)
                if not full_rescan:
                    self._session_cache = (branch, head, start)
                    return start
            # head moved backwards, the branch switched, or a rewind
            # marker appeared in the delta: the cached scan no longer
            # covers this chain
        start = -1
        for ev in reversed(self.log.events()):
            if ev.type == "session.start":
                start = ev.seq
                break
        self._session_cache = (branch, head, start)
        return start

    def spend(self) -> dict:
        st = fold(self.log,
                  from_seq=max(self.baseline_seq, self._session_start()))
        return {"usd": st.cost_usd, "steps": st.tool_calls,
                "tokens": st.tokens_in + st.tokens_out,
                "files": len(st.files_touched)}

    def reset(self) -> None:
        """Forget the current spend — the budget restarts from now. Sealed
        as a budget.event so the extension is never invisible."""
        self.baseline_seq = self.log.head()
        self._last_reason = ""
        self.log.append("budget.event",
                        {"kind": "reset", "spend": {"usd": 0.0, "steps": 0,
                                                    "tokens": 0, "files": 0}},
                        actor="human")

    def set_limit(self, axis: str, value) -> str:
        """Raise/lower one budget axis: steps | usd | tokens | files."""
        axis = axis.strip().lower()
        limits = {"steps": ("max_steps", int), "usd": ("max_usd", float),
                  "tokens": ("max_tokens", int), "files": ("max_files", int)}
        if axis not in limits:
            raise ValueError("axis must be one of: "
                             + ", ".join(sorted(limits)))
        attr, cast = limits[axis]
        try:
            value = cast(value)
        except (TypeError, ValueError):
            raise ValueError(f"{axis} budget must be a number, "
                             f"got {value!r}")
        setattr(self.budget, attr, value)
        self._last_reason = ""
        self.log.append("budget.event",
                        {"kind": "limit", "axis": axis, "value": value},
                        actor="human")
        return f"{axis} budget set to {value}"

    def check(self) -> tuple[bool, str]:
        """Return (ok, reason). A breach on ANY axis pauses the run."""
        ok, reason, _ = self._check()
        return ok, reason

    def _check(self) -> tuple[bool, str, dict]:
        """check() plus the spend dict — the breach path reuses the same
        fold instead of paying for a second one."""
        s = self.spend()
        b = self.budget
        if s["usd"] > b.max_usd:
            return False, f"USD budget exceeded: ${s['usd']:.4f} > ${b.max_usd}", s
        if s["steps"] > b.max_steps:
            return False, f"step budget exceeded: {s['steps']} > {b.max_steps}", s
        if s["tokens"] > b.max_tokens:
            return False, f"token budget exceeded: {s['tokens']} > {b.max_tokens}", s
        if s["files"] > b.max_files:
            return False, f"file budget exceeded: {s['files']} > {b.max_files}", s
        return True, "", s

    def enforce(self) -> bool:
        """Check and, on breach, seal a budget.event (pause). Returns True
        if the run may continue. While paused, only the FIRST breach per
        reason is sealed — no event spam on every loop iteration."""
        ok, reason, spend = self._check()
        if ok:
            self._last_reason = ""
            return True
        if reason != self._last_reason:
            self._last_reason = reason
            self.log.append("budget.event",
                            {"kind": "exceeded", "reason": reason,
                             "spend": spend},
                            actor="kernel")
        return False

    def slice_for(self, subtree: str) -> float:
        """USD slice for a subtree — a hard cap a child operation cannot borrow
        past (§13.5)."""
        frac = self.budget.slices.get(subtree, 0.0)
        return self.budget.max_usd * frac


# ---------------------------------------------------------------------------
# Loop / thrash / oscillation detectors (§13.4)
# ---------------------------------------------------------------------------

class LoopDetector:
    """Detects wasted motion from the event log. All checks are rung 1."""

    def __init__(self, log: EventLog, repeat_threshold: int = 3,
                 window: int = 10) -> None:
        self.log = log
        self.repeat_threshold = repeat_threshold
        self.window = window
        self._last_sig = ""  # dedupe loop.alert spam while repeating

    def exact_repeat(self) -> str | None:
        """hash(tool, canonical_args) seen repeat_threshold times within
        the last `window` tool.call events -> returns the signature.

        Scans backwards and stops after `window` tool calls — the hot
        path is O(window), not O(log size)."""
        counts: dict[str, int] = {}
        seen = 0
        for ev in reversed(self.log.events()):
            if ev.type != "tool.call":
                continue
            seen += 1
            if seen > self.window:
                break
            sig = _canonical_args(ev.data.get("name", ""),
                                  ev.data.get("args") or {})
            counts[sig] = counts.get(sig, 0) + 1
            if counts[sig] >= self.repeat_threshold:
                return sig
        return None

    def oscillation(self, path: str, hashes: list[str]) -> bool:
        """File content hash flipping A-B-A-B -> True (hard stop, present
        both versions to the human)."""
        if len(hashes) < 4:
            return False
        a, b, c, d = hashes[-4:]
        return a == c and b == d and a != b

    def detect(self) -> list[dict]:
        """Run all detectors, seal loop.alert events, return the alerts.
        While the same repetition persists, only the first detection is
        sealed — no alert spam on every turn."""
        alerts: list[dict] = []
        sig = self.exact_repeat()
        if sig and sig != self._last_sig:
            self._last_sig = sig
            alert = {"kind": "exact_repeat", "signature": sig,
                     "action": "force REFLECT with the repetition as evidence"}
            alerts.append(alert)
            self.log.append("loop.alert", alert, actor="kernel")
        elif not sig:
            self._last_sig = ""
        return alerts


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "cortex.jsonl")

        # -- plan DAG + frontier + write-exclusivity (I7) --------------------
        plan = Plan(log)
        plan.add(Node("n1", "read config", kind="READ"))
        plan.add(Node("n2", "write auth", kind="WRITE",
                      path_set=["src/auth.py"], depends_on=["n1"]))
        plan.add(Node("n3", "write auth tests", kind="WRITE",
                      path_set=["src/auth.py"], depends_on=["n1"]))
        plan.add(Node("n4", "write docs", kind="WRITE",
                      path_set=["docs.md"], depends_on=["n1"]))

        # n1 is the only PENDING node with all deps passed
        assert [n["id"] for n in plan.frontier()] == ["n1"]
        plan.set_status("n1", "PASSED")

        # n2 and n3 overlap on src/auth.py -> only one may be scheduled
        elig = plan.eligible()
        ids = [n["id"] for n in elig]
        assert "n2" in ids and "n4" in ids, ids
        assert not ("n2" in ids and "n3" in ids), \
            f"overlapping writes scheduled together: {ids}"

        # -- budget governor (I8) ---------------------------------------------
        gov = BudgetGovernor(log, Budget(max_usd=0.001, max_steps=1000))
        log.append("cost.incurred", {"usd": 0.5, "tokens_in": 10,
                                     "tokens_out": 5})
        ok, reason = gov.check()
        assert not ok and "USD" in reason, (ok, reason)
        assert gov.enforce() is False
        budget_events = fold(log).budget_events
        assert budget_events and budget_events[-1]["kind"] == "exceeded"

        # slices: a child operation's hard cap
        gov2 = BudgetGovernor(log, Budget(max_usd=10.0,
                                          slices={"analysis": 0.05}))
        assert abs(gov2.slice_for("analysis") - 0.5) < 1e-9

        # -- loop detector (§13.4) ---------------------------------------------
        log2 = EventLog(Path(td) / "loops.jsonl")
        det = LoopDetector(log2, repeat_threshold=3, window=10)
        for _ in range(2):
            log2.append("tool.call", {"name": "read_file",
                                      "args": {"path": "x.py"}})
        assert det.exact_repeat() is None
        log2.append("tool.call", {"name": "read_file",
                                  "args": {"path": "x.py"}})
        sig = det.exact_repeat()
        assert sig is not None
        alerts = det.detect()
        assert alerts and alerts[0]["kind"] == "exact_repeat"
        assert fold(log2).loop_alerts

        # oscillation A-B-A-B
        assert det.oscillation("f.py", ["h1", "h2", "h1", "h2"]) is True
        assert det.oscillation("f.py", ["h1", "h2", "h1", "h3"]) is False
        assert det.oscillation("f.py", ["h1", "h2", "h1"]) is False

    print("CORTEX SELF-TEST PASS")
