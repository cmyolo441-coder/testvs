"""ROUTER — the cost brain (smart model routing).

Every subtask is routed to the CHEAPEST model that is still capable of
doing it well. Routing is deterministic Python (rung 1): a multi-axis
difficulty classifier scores the task, a capability/cost table scores the
models, and the cheapest model whose capability clears the task's
difficulty wins. A quality guardrail escalates to a stronger model when a
cheap one would be out of its depth.

Nothing here calls a model. Every decision is sealed as a 'router.decision'
event, so the routing history is replayable and the savings are auditable:
the fold can always answer "what did we spend, and what would always using
the strongest model have cost?"

Design (pure Python, stdlib only):
  * Difficulty axes — code density, tool fan-out, reasoning keywords,
                      length, ambiguity. Each axis is a cheap heuristic.
  * Capability table — per-model capability score + input/output cost.
  * Guardrail       — if the task needs tools/reasoning a candidate lacks,
                      it is skipped; if nothing cheap clears the bar, the
                      strongest capable model is chosen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .kernel import EventLog, fold
from ._foundation import get_logger

_log = get_logger("router")

# ---------------------------------------------------------------------------
# Capability / cost table
# ---------------------------------------------------------------------------
# capability: 0.0..1.0 — how strong the model is on hard tasks.
# cost_in/out: relative $ per 1k tokens (free tiers are 0). These are
# relative units used to compare routes, not billing truth.
MODEL_TABLE: dict[str, dict[str, Any]] = {
    "deepseek-v4.1-flash-free": {"capability": 0.95, "cost_in": 0.0,
                                 "cost_out": 0.0, "tools": True,
                                 "reasoning": True},
    "minimax-m3.1-flash-free": {"capability": 0.9, "cost_in": 0.0,
                                "cost_out": 0.0, "tools": True,
                                "reasoning": True},
}

# the strongest model in the DEFAULT table — the escalation ceiling when no
# custom table is in play (kept for backwards compatibility)
_STRONGEST = max(MODEL_TABLE, key=lambda m: MODEL_TABLE[m]["capability"])

# ---------------------------------------------------------------------------
# Difficulty classification (rung 1)
# ---------------------------------------------------------------------------

_REASONING_WORDS = (
    "prove", "derive", "theorem", "optimize", "optimise", "algorithm",
    "complexity", "architect", "design", "refactor", "debug", "diagnose",
    "root cause", "trade-off", "tradeoff", "reason", "why", "analyse",
    "analyze", "security", "vulnerab",
)
_TOOL_WORDS = (
    "run", "execute", "build", "test", "install", "git", "grep", "search",
    "file", "read", "write", "edit", "command", "shell", "compile",
)
_CODE_RE = re.compile(r"```|def |class |import |=>|::|\{\{|\$\{")


@dataclass
class Difficulty:
    """The multi-axis difficulty score of one task (0.0 .. 1.0)."""
    score: float = 0.0
    needs_tools: bool = False
    needs_reasoning: bool = False
    axes: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"score": round(self.score, 3),
                "needs_tools": self.needs_tools,
                "needs_reasoning": self.needs_reasoning,
                "axes": self.axes}


def classify(text: str) -> Difficulty:
    """Score a task's difficulty across cheap heuristic axes."""
    # harden against non-string callers (None, bytes) — never a crash path
    if not isinstance(text, str):
        text = str(text)
    low = text.lower()
    words = max(1, len(text.split()))

    code_density = min(1.0, len(_CODE_RE.findall(text)) / 6.0)
    reasoning_hits = sum(1 for w in _REASONING_WORDS if w in low)
    tool_hits = sum(1 for w in _TOOL_WORDS if w in low)
    reasoning = min(1.0, reasoning_hits / 3.0)
    tooling = min(1.0, tool_hits / 4.0)
    length = min(1.0, words / 120.0)

    # weighted blend — reasoning and code weigh most
    score = (0.34 * reasoning + 0.26 * code_density + 0.20 * tooling
             + 0.12 * length + 0.08 * min(1.0, reasoning_hits / 5.0))
    score = max(0.05, min(1.0, score))
    return Difficulty(
        score=score,
        needs_tools=tool_hits >= 2,
        needs_reasoning=reasoning_hits >= 2,
        axes={"reasoning": round(reasoning, 3),
              "code": round(code_density, 3),
              "tooling": round(tooling, 3),
              "length": round(length, 3)},
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

@dataclass
class RouteChoice:
    model_id: str
    difficulty: float
    reason: str
    escalated: bool = False
    est_cost: float = 0.0
    est_tokens: int = 1500  # token estimate the cost was computed from

    def to_dict(self) -> dict:
        return {"model": self.model_id, "difficulty": round(self.difficulty, 3),
                "reason": self.reason, "escalated": self.escalated,
                "est_cost": round(self.est_cost, 6),
                "est_tokens": self.est_tokens}


class Router:
    """Route tasks to the cheapest capable model. All decisions are sealed
    into the event log; state is recovered by folding."""

    def __init__(self, log: EventLog,
                 table: dict[str, dict] | None = None) -> None:
        self.log = log
        # `table or MODEL_TABLE` treated an explicitly EMPTY table as
        # "use the production catalog": the empty-table degradation below
        # (`if not self.table`) was dead code and an empty custom table
        # silently routed to real models. Only None means "default".
        self.table = table if table is not None else MODEL_TABLE
        # BUGFIX: the escalation ceiling belongs to THIS router's table.
        # The old module-global _STRONGEST leaked across Router instances
        # with custom tables — the `escalated` flag and savings() both
        # compared against the wrong model.
        self._strongest = (max(self.table,
                               key=lambda m: self.table[m]["capability"])
                           if self.table else _STRONGEST)

    def _capable(self, model_id: str, diff: Difficulty) -> bool:
        spec = self.table.get(model_id)
        if spec is None:
            return False
        if diff.needs_tools and not spec.get("tools"):
            return False
        if diff.needs_reasoning and not spec.get("reasoning"):
            return False
        return True

    def _cost(self, model_id: str, est_tokens: int) -> float:
        spec = self.table.get(model_id, {})
        # rough split: 2/3 prompt, 1/3 completion
        tin = est_tokens * 2 // 3
        tout = est_tokens - tin
        return (tin / 1000.0) * spec.get("cost_in", 0.0) + \
               (tout / 1000.0) * spec.get("cost_out", 0.0)

    def choose(self, task: str, est_tokens: int = 1500,
               prefer: str | None = None) -> RouteChoice:
        """Pick the cheapest model whose capability clears the task.

        `prefer` pins a model when the caller (or the user) insists; it is
        still capability-checked, and if it cannot do the job the router
        escalates and says so."""
        diff = classify(task)
        # Quality guardrail: require a capability MARGIN above the task's
        # difficulty, so a hard task never sits right at a cheap model's
        # ceiling. This is what forces genuine escalation.
        required = max(0.30, min(0.97, diff.score + 0.25))

        if prefer and prefer in self.table:
            spec = self.table[prefer]
            if self._capable(prefer, diff) and spec["capability"] >= required:
                choice = RouteChoice(prefer, diff.score,
                                     f"pinned '{prefer}' is capable",
                                     escalated=False,
                                     est_cost=self._cost(prefer, est_tokens),
                                     est_tokens=est_tokens)
                self._seal(task, diff, choice)
                return choice
            # pinned model can't do it — escalate past it

        candidates = [m for m in self.table
                      if self._capable(m, diff)
                      and self.table[m]["capability"] >= required]
        escalated = False
        if not candidates:
            # nothing cheap clears the bar -> strongest capable model
            candidates = [m for m in self.table if self._capable(m, diff)]
            escalated = True
        if not candidates:
            # even the fallback capability check cleared nothing (custom
            # table without a tool-capable model, or an empty table) —
            # degrade to the table's best instead of raising ValueError
            if not self.table:
                choice = RouteChoice(self._strongest, diff.score,
                                     "no models in routing table — "
                                     f"defaulting to '{self._strongest}'",
                                     escalated=True, est_cost=0.0,
                                     est_tokens=est_tokens)
                self._seal(task, diff, choice)
                return choice
            candidates = list(self.table)
            escalated = True
        if escalated:
            best = max(candidates,
                       key=lambda m: (self.table[m]["capability"],
                                      -self._cost(m, est_tokens)))
        else:
            # cheapest first; break ties toward higher capability
            best = min(candidates,
                       key=lambda m: (self._cost(m, est_tokens),
                                      -self.table[m]["capability"]))
        spec = self.table[best]
        reason = (f"difficulty {diff.score:.2f} -> cheapest capable "
                  f"'{best}' (cap {spec['capability']:.2f})")
        choice = RouteChoice(best, diff.score, reason,
                             escalated=escalated,
                             est_cost=self._cost(best, est_tokens),
                             est_tokens=est_tokens)
        self._seal(task, diff, choice)
        return choice

    def _seal(self, task: str, diff: Difficulty, choice: RouteChoice) -> None:
        self.log.append("router.decision",
                        {"task": task[:200], **choice.to_dict(),
                         "axes": diff.axes},
                        actor="router")

    # -- projections (pure folds) -------------------------------------------

    def decisions(self) -> list[dict]:
        return fold(self.log).router_decisions

    def savings(self) -> dict:
        """What we spent vs what always using the strongest model would have
        cost. The dividend of smart routing, made auditable."""
        decs = self.decisions()
        if not decs:
            return {"routed": 0, "est_spent": 0.0, "strongest_cost": 0.0,
                    "saved": 0.0}
        spent = sum(float(d.get("est_cost", 0.0)) for d in decs)
        # strongest_cost must use each decision's OWN token estimate —
        # the old hardcoded 1500 made `saved` wrong whenever choose()
        # was called with a non-default est_tokens (1500 fallback keeps
        # pre-fix events comparable).
        strongest = sum(self._cost(self._strongest,
                                   int(d.get("est_tokens", 1500)))
                        for d in decs)
        return {"routed": len(decs), "est_spent": round(spent, 6),
                "strongest_cost": round(strongest, 6),
                "saved": round(max(0.0, strongest - spent), 6)}

    def format_status(self) -> str:
        s = self.savings()
        decs = self.decisions()
        lines = ["ROUTER — the cost brain",
                 f"  routed {s['routed']} task(s)   est spent "
                 f"${s['est_spent']:.4f}   saved ${s['saved']:.4f} "
                 f"vs always-strongest"]
        for d in decs[-6:]:
            esc = " ⤒" if d.get("escalated") else ""
            lines.append(f"    {d.get('model', '?'):<26} "
                         f"diff {d.get('difficulty', 0):.2f}{esc}  "
                         f"${d.get('est_cost', 0):.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "router.jsonl")
        # Synthetic capabilities exercise generic routing independently of
        # the singleton production catalog.
        r = Router(log, table={
            "basic": {"capability": 0.55, "cost_in": 0.0,
                      "cost_out": 0.0, "tools": False,
                      "reasoning": False},
            "advanced": {"capability": 0.95, "cost_in": 0.01,
                         "cost_out": 0.02, "tools": True,
                         "reasoning": True},
        })

        # a trivial chat task routes to a free, capable model
        easy = r.choose("say hello in one word")
        assert easy.difficulty < 0.5, easy
        assert r.table[easy.model_id]["cost_in"] == 0.0, easy

        # a hard reasoning+code task escalates to a strong, reasoning-capable
        # model — strictly stronger than what the trivial task got
        hard = r.choose("Prove the algorithm's complexity and refactor the "
                        "class to optimize it, then debug the root cause.\n"
                        "```python\ndef f(x):\n    return x\n```")
        assert hard.difficulty > easy.difficulty, (easy, hard)
        assert r.table[hard.model_id]["reasoning"] is True, hard
        assert (r.table[hard.model_id]["capability"]
                > r.table[easy.model_id]["capability"]), (easy, hard)

        # a tool-heavy task never lands on a no-tool model
        tooly = r.choose("run the tests, then grep the log file and read it")
        assert r.table[tooly.model_id]["tools"] is True, tooly

        # a pinned model is respected when capable
        pinned = r.choose("summarize this", prefer="basic")
        assert pinned.model_id == "basic", pinned

        # a pinned model that can't do the job is escalated past
        pinned_hard = r.choose(
            "run the build and execute the test suite",
            prefer="basic")  # no tool support
        assert pinned_hard.model_id != "basic", pinned_hard
        assert r.table[pinned_hard.model_id]["tools"] is True

        # decisions are sealed and foldable
        assert len(r.decisions()) == 5
        s = r.savings()
        assert s["routed"] == 5 and s["saved"] >= 0.0

    print("ROUTER SELF-TEST PASS")
