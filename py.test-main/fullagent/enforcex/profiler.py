"""profiler — Nanosecond Execution Profiler, ASCII Flamegraph & Bottleneck Advisor.

Self-test: ``python3 -m fullagent.enforcex.profiler``
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterator, List, Sequence

__all__ = [
    "SpanRecord",
    "StageStats",
    "ProfileReport",
    "ExecutionProfiler",
    "percentile",
]

_MAX_ELAPSED_MS = 1.0e9  # ~11.5 days; keeps all aggregates finite


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except Exception:
        return default
    if not math.isfinite(f):
        return default
    return f


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_name(value: Any) -> str:
    if value is None:
        return "span"
    if isinstance(value, str):
        return value or "span"
    try:
        return str(value) or "span"
    except Exception:
        return "span"


def percentile(sorted_vals: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile. Inputs are sanitised and sorted
    defensively; non-finite values/pcts degrade to 0. Never raises."""
    try:
        vals = sorted(_finite_float(v, 0.0) for v in (sorted_vals or ()))
    except Exception:
        return 0.0
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])
    p = _finite_float(pct, 0.0)
    p = max(0.0, min(100.0, p))
    rank = (p / 100.0) * (len(vals) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(vals[low])
    weight = rank - low
    return float(vals[low] * (1.0 - weight) + vals[high] * weight)


@dataclass
class SpanRecord:
    name: str
    elapsed_ms: float
    depth: int = 0
    parent: str = ""
    ok: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "elapsed_ms": round(self.elapsed_ms, 4),
            "depth": self.depth,
            "parent": self.parent,
            "ok": self.ok,
        }


@dataclass
class StageStats:
    name: str
    count: int = 0
    total_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "errors": self.errors,
        }


@dataclass
class ProfileReport:
    total_spans: int = 0
    total_ms: float = 0.0
    stages: Dict[str, StageStats] = field(default_factory=dict)
    hotspots: List[str] = field(default_factory=list)
    flamegraph_ascii: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_spans": self.total_spans,
            "total_ms": round(self.total_ms, 3),
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "hotspots": list(self.hotspots),
            "flamegraph_ascii": self.flamegraph_ascii,
        }


class ExecutionProfiler:
    def __init__(self, max_spans: int = 2048) -> None:
        self.max_spans = max(64, int(max_spans))
        self._lock = threading.RLock()
        self._spans: List[SpanRecord] = []
        self._tls = threading.local()

    def _stack(self) -> List[str]:
        st = getattr(self._tls, "stack", None)
        if st is None:
            st = []
            self._tls.stack = st
        return st

    @contextlib.contextmanager
    def span(self, name: str) -> Iterator[None]:
        stack = self._stack()
        parent = stack[-1] if stack else ""
        depth = len(stack)
        stack.append(str(name or "span"))
        t0 = time.perf_counter_ns()
        ok = True
        try:
            yield
        except Exception:
            ok = False
            raise
        finally:
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            stack.pop()
            rec = SpanRecord(name=str(name or "span"), elapsed_ms=elapsed_ms, depth=depth, parent=parent, ok=ok)
            with self._lock:
                self._spans.append(rec)
                if len(self._spans) > self.max_spans:
                    del self._spans[:-self.max_spans]

    def record(self, name: str, elapsed_ms: float, *, depth: int = 0, ok: bool = True) -> None:
        with self._lock:
            self._spans.append(
                SpanRecord(name=str(name or "span"), elapsed_ms=max(0.0, float(elapsed_ms)), depth=depth, ok=ok)
            )
            if len(self._spans) > self.max_spans:
                del self._spans[:-self.max_spans]

    def reset(self) -> None:
        with self._lock:
            self._spans.clear()

    def report(self, width: int = 40) -> ProfileReport:
        with self._lock:
            spans = list(self._spans)
        if not spans:
            return ProfileReport()

        by_name: Dict[str, List[SpanRecord]] = {}
        for s in spans:
            by_name.setdefault(s.name, []).append(s)

        stages: Dict[str, StageStats] = {}
        for name, items in by_name.items():
            vals = sorted(x.elapsed_ms for x in items)
            tot = sum(vals)
            errs = sum(1 for x in items if not x.ok)
            stages[name] = StageStats(
                name=name,
                count=len(vals),
                total_ms=tot,
                mean_ms=tot / len(vals),
                min_ms=vals[0],
                max_ms=vals[-1],
                p50_ms=percentile(vals, 50.0),
                p95_ms=percentile(vals, 95.0),
                p99_ms=percentile(vals, 99.0),
                errors=errs,
            )

        root_total = sum(s.elapsed_ms for s in spans if s.depth == 0) or sum(s.elapsed_ms for s in spans) or 0.001
        sorted_stages = sorted(stages.values(), key=lambda x: -x.total_ms)
        hotspots = [
            f"{st.name} ({st.total_ms:.2f}ms total, p95={st.p95_ms:.2f}ms, {st.total_ms / root_total * 100:.1f}%)"
            for st in sorted_stages[:3]
        ]

        lines: List[str] = []
        bar_w = max(10, int(width))
        max_stage_ms = max(st.total_ms for st in sorted_stages) or 1.0
        for st in sorted_stages[:12]:
            filled = int(round((st.total_ms / max_stage_ms) * bar_w))
            bar = "█" * max(1, filled) + "░" * max(0, bar_w - max(1, filled))
            lines.append(f"{st.name:<22} |{bar}| {st.total_ms:7.2f}ms (n={st.count})")

        return ProfileReport(
            total_spans=len(spans),
            total_ms=root_total,
            stages=stages,
            hotspots=hotspots,
            flamegraph_ascii="\n".join(lines),
        )


def _self_test() -> int:
    prof = ExecutionProfiler()
    with prof.span("pipeline"):
        with prof.span("injection_scan"):
            time.sleep(0.002)
        with prof.span("gate_eval"):
            time.sleep(0.004)
    rep = prof.report()
    assert rep.total_spans == 3
    assert "pipeline" in rep.stages and "gate_eval" in rep.stages
    assert rep.stages["gate_eval"].total_ms >= 1.0
    assert "█" in rep.flamegraph_ascii
    assert len(rep.hotspots) >= 2

    print("profiler._self_test: OK PASS (nested spans, percentiles, ASCII flamegraph)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
