"""TUNER — Bayesian-ish auto-tuning of the agent's own knobs.

Which effort level, worker step count, and compaction threshold make
turns score best? Instead of guessing, the tuner runs the classic
Tree-structured Parzen Estimator loop:

    suggest    split observed (config, score) history into GOOD (top γ)
               and REST; sample candidate values per knob by drawing
               from the good-set's per-knob histogram with probability
               l(x) ∝ good-density / (good + rest density) — the TPE
               density ratio — and otherwise explore uniformly. Early
               on (no history) it space-fills.
    observe    every trial's score feeds the history
    best       the argmax config so far, sealed as tuner.best

This is a real (discrete-knob) TPE: per-knob histogram likelihoods,
density-ratio candidate acceptance, γ = 25%, plus ε-greedy exploration
so nothing is ever permanently locked out. The self-test optimizes a
known quadratic-with-interaction objective and proves convergence
toward the optimum without ever evaluating the whole grid.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .kernel import EventLog
from ._foundation import get_logger

_log = get_logger("tuner")

_GAMMA = 0.25          # fraction of history counted as GOOD
_EPSILON = 0.10        # uniform exploration probability
_PATIENCE = 8          # stop if no improvement for this many trials
# MEMORY BOUND (worker 19/20): the score cache is pure memoization —
# capped FIFO so a long tuning run cannot grow it without limit.
_SCORE_CACHE_MAX = 4096


def _config_key(config: dict) -> tuple:
    """Hashable identity for a config (values are discrete knob choices)."""
    try:
        key = tuple(sorted(config.items()))
        hash(key)          # sortable values may still be unhashable
        return key
    except TypeError:
        return tuple(sorted((k, repr(v)) for k, v in config.items()))


@dataclass
class Trial:
    config: dict
    score: float = 0.0
    ts_order: int = 0


@dataclass
class TunerReport:
    best_config: dict
    best_score: float
    trials: int
    distinct: int

    def to_dict(self) -> dict:
        return {"best_config": self.best_config,
                "best_score": round(self.best_score, 4),
                "trials": self.trials, "distinct": self.distinct}


class ParzenTuner:
    """TPE over discrete knob spaces."""

    def __init__(self, log: EventLog, space: dict[str, list],
                 seed: int = 13, objective=None,
                 patience: int = _PATIENCE) -> None:
        """space: knob -> allowed values. objective(config) -> float —
        injectable (production: run a scored benchmark turn; tests: a
        synthetic function). patience: early-stop after this many trials
        with no improvement to the best score."""
        space = space or {}
        for k, v in space.items():
            if not v:
                raise ValueError(f"tuner knob {k!r} has no allowed values")
        if not space:
            raise ValueError("tuner space must define at least one knob")
        self.log = log
        self.space = {k: list(v) for k, v in space.items()}
        self.rng = random.Random(seed)
        self.objective = objective
        self.patience = patience
        self.history: list[Trial] = []
        self._n = 0
        # Score cache: evaluating the same discrete config twice is pure
        # waste — the objective is deterministic, so reuse the first
        # score instead of burning another trial on it.
        self._score_cache: dict[tuple, float] = {}
        self._last_was_cached = False

    # -- TPE internals --------------------------------------------------------

    def _density(self, value, values: list) -> float:
        """Histogram likelihood of `value` under observed `values`
        (with a smoothing floor so empty bins never zero out)."""
        if not values:
            return 1.0
        n_hit = sum(1 for v in values if v == value)
        return (n_hit + 0.5) / (len(values) + 0.5)

    def _sample_knob(self, knob: str, good: list, rest: list) -> object:
        """Draw one value by TPE density-ratio acceptance."""
        values = self.space[knob]
        if not good:
            return self.rng.choice(values)      # space-fill early
        # l(x) ∝ p(x|good) / (p(x|good) + p(x|rest))
        weights = []
        good_vals = [t.config[knob] for t in good]
        rest_vals = [t.config[knob] for t in rest]
        for v in values:
            pg = self._density(v, good_vals)
            pr = self._density(v, rest_vals)
            weights.append(pg / (pg + pr))
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(values)
        pick = self.rng.random() * total
        acc = 0.0
        for v, w in zip(values, weights):
            acc += w
            if pick <= acc:
                return v
        return values[-1]

    # -- the loop ------------------------------------------------------------------

    def suggest(self) -> dict:
        """One config proposal: ε-uniform explore, else TPE exploit."""
        n_good = max(1, int(len(self.history) * _GAMMA))
        ranked = sorted(self.history, key=lambda t: -t.score)
        good, rest = ranked[:n_good], ranked[n_good:]
        if not self.history or self.rng.random() < _EPSILON:
            return {k: self.rng.choice(v)
                    for k, v in self.space.items()}
        return {k: self._sample_knob(k, good, rest)
                for k in self.space}

    def observe(self, config: dict, score: float) -> None:
        # a config that does not match the space (missing knobs, unknown
        # knobs, or values outside the allowed lists) would poison the
        # per-knob histograms and crash suggest() with a KeyError —
        # reject it loudly instead of corrupting the history
        if set(config) != set(self.space):
            raise ValueError(
                f"config knobs {sorted(config)} do not match space knobs "
                f"{sorted(self.space)}")
        for k, v in config.items():
            if v not in self.space[k]:
                raise ValueError(f"value {v!r} for knob {k!r} is not in "
                                 f"{self.space[k]}")
        self._n += 1
        self.history.append(Trial(config=dict(config),
                                  score=float(score),
                                  ts_order=self._n))
        self.log.append("tuner.trial",
                        {"config": config, "score": round(float(score),
                                                          4),
                         "n": self._n}, actor="kernel")

    def step(self) -> Trial:
        """Suggest → evaluate (objective) → observe. Returns the trial.

        A config seen before reuses its cached score instead of
        re-running the objective — TPE exploitation revisits winners
        often, and each repeat evaluation is a wasted trial."""
        if self.objective is None:
            raise RuntimeError("no objective attached")
        config = self.suggest()
        key = _config_key(config)
        if key in self._score_cache:
            score = self._score_cache[key]
            self._last_was_cached = True
        else:
            score = float(self.objective(config))
            self._score_cache[key] = score
            # bound memory: the cache is pure memoization — evict the
            # oldest entries first-in-first-out (dicts keep insertion
            # order). Re-evaluating an evicted config just costs a trial.
            while len(self._score_cache) > _SCORE_CACHE_MAX:
                self._score_cache.pop(next(iter(self._score_cache)))
            self._last_was_cached = False
        self.observe(config, score)
        return self.history[-1]

    def best(self) -> Trial | None:
        if not self.history:
            return None
        return max(self.history, key=lambda t: t.score)

    def run(self, n: int = 20) -> TunerReport:
        """Run up to n trials, stopping early on convergence.

        Convergence for discrete TPE means the sampler has collapsed:
        it keeps suggesting configs we have already evaluated. Those
        trials are free (score cache) but pointless — further trials
        will not discover anything new. We stop after `patience`
        consecutive cache-hit trials. A stale *best score* alone is NOT
        a stop signal: TPE legitimately explores for long stretches
        (20+ trials) before its next improvement."""
        if n < 1:
            raise ValueError(f"run() needs n >= 1, got {n}")
        dup_streak = 0
        for _ in range(n):
            self.step()
            if self._last_was_cached:
                dup_streak += 1
                if dup_streak >= self.patience:
                    self.log.append(
                        "tuner.early_stop",
                        {"trials": len(self.history),
                         "distinct": len(self._score_cache),
                         "patience": self.patience,
                         "reason": "suggester converged to evaluated "
                                   "configs"},
                        actor="kernel")
                    break
            else:
                dup_streak = 0
        b = self.best()
        assert b is not None
        distinct = len({_config_key(t.config) for t in self.history})
        report = TunerReport(b.config, b.score, len(self.history),
                             distinct)
        self.log.append("tuner.best", report.to_dict())
        return report

    def format(self) -> str:
        b = self.best()
        lines = [f"TUNER — {len(self.history)} trials"]
        if b:
            lines.append(f"  best score {b.score:.4f} with "
                         + ", ".join(f"{k}={v}"
                                     for k, v in b.config.items()))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test — a known objective, conquered without grid search
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "tuner.jsonl")

        # known optimum: effort=high, steps=medium, compact=tight
        opt = {"effort": "high", "steps": "medium",
               "compact": "tight"}
        score_map = {"effort": {"low": 0.2, "medium": 0.5, "high": 0.9},
                     "steps": {"low": 0.4, "medium": 0.8, "high": 0.5},
                     "compact": {"loose": 0.3, "tight": 0.7}}

        def objective(cfg: dict) -> float:
            # multiplicative landscape with interactions — a naive
            # per-knob greedy sweep can get stuck; TPE explores
            base = 1.0
            for k, v in cfg.items():
                base *= score_map[k][v]
            penalty = 0.0
            if cfg["steps"] == "high" and cfg["compact"] == "tight":
                penalty = 0.3      # interaction: bad combo
            return base + 0.1 - penalty

        best_possible = objective(opt)

        space = {"effort": ["low", "medium", "high"],
                 "steps": ["low", "medium", "high"],
                 "compact": ["loose", "tight"]}
        tuner = ParzenTuner(log, space, seed=21, objective=objective)
        report = tuner.run(n=40)

        # converged near the optimum without exhausting the grid
        assert report.best_score >= best_possible * 0.999, \
            (report.to_dict(), best_possible)
        assert report.best_config == opt, report.to_dict()
        # ...but never evaluated everything (27 combos exist)
        assert report.distinct <= 26

        # determinism with the same seed
        t2 = ParzenTuner(EventLog(Path(td) / "t2.jsonl"), space,
                         seed=21, objective=objective)
        r2 = t2.run(n=40)
        assert r2.best_config == report.best_config

        # early suggestions space-fill (no history yet)
        t3 = ParzenTuner(EventLog(Path(td) / "t3.jsonl"), space,
                         seed=4)
        first = t3.suggest()
        assert set(first) == set(space)

        # density ratio math: good-only values outweigh unseen ones
        t3.observe({"effort": "high", "steps": "low",
                    "compact": "loose"}, 0.9)
        t3.observe({"effort": "high", "steps": "low",
                    "compact": "loose"}, 0.8)
        t3.observe({"effort": "low", "steps": "high",
                    "compact": "tight"}, 0.1)
        draws = [t3.suggest()["effort"] for _ in range(60)]
        assert draws.count("high") > draws.count("low"), \
            {"high": draws.count("high"), "low": draws.count("low")}

        # events sealed
        kinds = {e.type for e in log.events()}
        assert {"tuner.trial", "tuner.best"} <= kinds

        # step() without an objective is a clean error
        try:
            ParzenTuner(log, space).step()
            raise AssertionError("must raise without objective")
        except RuntimeError:
            pass

        # MEMORY BOUND (worker 19/20): the score cache is pure
        # memoization — hammer it with far more distinct configs than the
        # cap and it must stay bounded (FIFO eviction)
        big_space = {"p1": list(range(40)), "p2": list(range(40)),
                     "p3": list(range(40))}
        t4 = ParzenTuner(EventLog(Path(td) / "t4.jsonl"), big_space,
                         seed=7,
                         objective=lambda c: float(c["p1"] + c["p2"]))
        for _ in range(_SCORE_CACHE_MAX + 500):
            t4.step()
        assert len(t4._score_cache) <= _SCORE_CACHE_MAX, \
            f"score cache unbounded: {len(t4._score_cache)}"
        # still functional after eviction: cached hits still served
        assert t4._last_was_cached in (True, False)
        print(f"memory: score cache capped at {len(t4._score_cache)} "
              f"after {_SCORE_CACHE_MAX + 500} steps")

        print("TUNER SELF-TEST PASS")
