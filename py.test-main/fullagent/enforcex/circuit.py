"""circuit — Adaptive 3-State Circuit Breaker, Token-Bucket Rate Limiter & Bulkhead Governor.

Self-test: ``python3 -m fullagent.enforcex.circuit``
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

__all__ = [
    "CircuitOpenError",
    "RateLimitExceededError",
    "CircuitMetrics",
    "CircuitBreaker",
    "TokenBucketLimiter",
    "Bulkhead",
]


def _as_int(value: Any, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError, OverflowError):
        return max(minimum, default)


def _as_float(value: Any, default: float, minimum: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return max(minimum, default)
    if not math.isfinite(f):
        return max(minimum, default)
    return max(minimum, f)


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the circuit breaker is OPEN."""


class RateLimitExceededError(RuntimeError):
    """Raised when a token bucket rate limit is exceeded."""


@dataclass
class CircuitMetrics:
    name: str
    state: str = "CLOSED"
    total_calls: int = 0
    successes: int = 0
    failures: int = 0
    rejected: int = 0
    consecutive_failures: int = 0
    last_failure_ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "total_calls": self.total_calls,
            "successes": self.successes,
            "failures": self.failures,
            "rejected": self.rejected,
            "consecutive_failures": self.consecutive_failures,
            "last_failure_ts": self.last_failure_ts,
        }


class CircuitBreaker:
    def __init__(
        self,
        name: str = "default",
        *,
        failure_threshold: int = 3,
        recovery_timeout_s: float = 5.0,
        half_open_successes: int = 1,
        half_open_max_calls: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = str(name or "default")
        self.failure_threshold = _as_int(failure_threshold, 3, 1)
        self.recovery_timeout_s = _as_float(recovery_timeout_s, 5.0, 0.01)
        self.half_open_successes = _as_int(half_open_successes, 1, 1)
        self.half_open_max_calls = _as_int(half_open_max_calls, 1, 1)
        self._clock = clock
        self._lock = threading.RLock()
        self._state = "CLOSED"
        self._total = 0
        self._successes = 0
        self._failures = 0
        self._rejected = 0
        self._consec_failures = 0
        self._half_open_ok = 0
        self._half_open_inflight = 0
        self._opened_at = 0.0
        self._last_failure_ts = 0.0

    @property
    def state(self) -> str:
        with self._lock:
            self._check_cooldown()
            return self._state

    def _check_cooldown(self) -> None:
        if self._state == "OPEN":
            if self._clock() - self._opened_at >= self.recovery_timeout_s:
                self._state = "HALF_OPEN"
                self._half_open_ok = 0
                self._half_open_inflight = 0

    def allow_request(self) -> bool:
        with self._lock:
            self._check_cooldown()
            return self._state in ("CLOSED", "HALF_OPEN")

    def record_success(self) -> None:
        with self._lock:
            self._successes += 1
            self._consec_failures = 0
            if self._state == "HALF_OPEN":
                self._half_open_ok += 1
                if self._half_open_ok >= self.half_open_successes:
                    self._state = "CLOSED"
                    self._half_open_inflight = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._consec_failures += 1
            now = self._clock()
            self._last_failure_ts = now
            if self._state == "HALF_OPEN" or self._consec_failures >= self.failure_threshold:
                self._state = "OPEN"
                self._opened_at = now
                self._half_open_inflight = 0

    def call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        fallback: Optional[Callable[[], Any]] = None,
        **kwargs: Any,
    ) -> Any:
        half_open_ticket = False
        with self._lock:
            self._total += 1
            self._check_cooldown()
            if self._state == "OPEN":
                self._rejected += 1
                if fallback is not None:
                    return fallback()
                raise CircuitOpenError(f"Circuit '{self.name}' is OPEN")
            if self._state == "HALF_OPEN":
                # Only a bounded number of concurrent probes in HALF_OPEN.
                if self._half_open_inflight >= self.half_open_max_calls:
                    self._rejected += 1
                    if fallback is not None:
                        return fallback()
                    raise CircuitOpenError(f"Circuit '{self.name}' is HALF_OPEN (probe in flight)")
                self._half_open_inflight += 1
                half_open_ticket = True

        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.record_failure()
            if fallback is not None:
                return fallback()
            raise
        else:
            self.record_success()
            return result
        finally:
            if half_open_ticket:
                with self._lock:
                    self._half_open_inflight = max(0, self._half_open_inflight - 1)

    def reset(self) -> None:
        with self._lock:
            self._state = "CLOSED"
            self._consec_failures = 0
            self._half_open_ok = 0
            self._half_open_inflight = 0
            self._opened_at = 0.0

    def metrics(self) -> CircuitMetrics:
        with self._lock:
            self._check_cooldown()
            return CircuitMetrics(
                name=self.name,
                state=self._state,
                total_calls=self._total,
                successes=self._successes,
                failures=self._failures,
                rejected=self._rejected,
                consecutive_failures=self._consec_failures,
                last_failure_ts=self._last_failure_ts,
            )


class TokenBucketLimiter:
    def __init__(
        self,
        rate_per_sec: float = 10.0,
        burst: int = 20,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rate = _as_float(rate_per_sec, 10.0, 0.001)
        self.capacity = float(_as_int(burst, 20, 1))
        self._tokens = self.capacity
        self._clock = clock
        self._last_ts = self._clock()
        self._lock = threading.Lock()

    def try_acquire(self, tokens: float = 1.0) -> bool:
        try:
            need = float(tokens)
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(need) or need < 0.0:
            return False
        with self._lock:
            now = self._clock()
            elapsed = max(0.0, now - self._last_ts)
            self._last_ts = now
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            if self._tokens >= need:
                self._tokens -= need
                return True
            return False


class Bulkhead:
    def __init__(self, max_concurrent: int = 4, timeout_s: float = 5.0) -> None:
        self.max_concurrent = _as_int(max_concurrent, 4, 1)
        self.timeout_s = _as_float(timeout_s, 5.0, 0.01)
        self._sem = threading.BoundedSemaphore(self.max_concurrent)
        self._lock = threading.Lock()
        self.active = 0
        self.rejected = 0

    def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        acquired = self._sem.acquire(timeout=self.timeout_s)
        if not acquired:
            with self._lock:
                self.rejected += 1
            raise RateLimitExceededError("Bulkhead concurrency limit reached")
        with self._lock:
            self.active += 1
        try:
            return fn(*args, **kwargs)
        finally:
            with self._lock:
                self.active -= 1
            self._sem.release()


def _self_test() -> int:
    now_box = [100.0]
    cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout_s=10.0, clock=lambda: now_box[0])
    assert cb.state == "CLOSED"

    for _ in range(2):
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
        except ValueError:
            pass
    assert cb.state == "OPEN"
    assert cb.call(lambda: "live", fallback=lambda: "fallback_val") == "fallback_val"
    assert cb.metrics().rejected == 1

    now_box[0] += 11.0
    assert cb.state == "HALF_OPEN"
    assert cb.call(lambda: "recovered") == "recovered"
    assert cb.state == "CLOSED"

    tb = TokenBucketLimiter(rate_per_sec=2.0, burst=2, clock=lambda: now_box[0])
    assert tb.try_acquire(1.0) and tb.try_acquire(1.0)
    assert not tb.try_acquire(1.0)
    now_box[0] += 1.0
    assert tb.try_acquire(1.0)

    bh = Bulkhead(max_concurrent=2, timeout_s=0.05)
    assert bh.run(lambda x: x + 1, 41) == 42

    # Malformed constructor args are coerced instead of raising
    cb_bad = CircuitBreaker(None, failure_threshold=None, recovery_timeout_s=float("nan"),
                            half_open_successes="x", half_open_max_calls=-4)  # type: ignore[arg-type]
    assert cb_bad.state == "CLOSED" and cb_bad.failure_threshold >= 1
    tb_bad = TokenBucketLimiter(rate_per_sec=float("inf"), burst=None)  # type: ignore[arg-type]
    assert tb_bad.capacity >= 1.0 and math.isfinite(tb_bad.rate)
    bh_bad = Bulkhead(max_concurrent=0, timeout_s=None)  # type: ignore[arg-type]
    assert bh_bad.max_concurrent >= 1

    # last_failure_ts tracks the most recent failure, even while already OPEN
    cb_ts = CircuitBreaker("ts", failure_threshold=10, clock=lambda: now_box[0])
    try:
        cb_ts.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
    except ValueError:
        pass
    now_box[0] += 5.0
    try:
        cb_ts.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
    except ValueError:
        pass
    last_failure = now_box[0]
    now_box[0] += 5.0
    assert cb_ts.metrics().last_failure_ts == last_failure

    # Token bucket rejects malformed/negative requests and caps refill at burst
    tb2 = TokenBucketLimiter(rate_per_sec=1000.0, burst=2, clock=lambda: now_box[0])
    now_box[0] += 3600.0
    assert tb2.try_acquire(2.0) is True
    assert tb2.try_acquire(1.0) is False
    assert tb2.try_acquire(float("nan")) is False
    assert tb2.try_acquire(-1.0) is False
    assert tb2.try_acquire("bad") is False  # type: ignore[arg-type]

    # HALF_OPEN permits a bounded number of concurrent probes
    gate_started, gate_release = threading.Event(), threading.Event()

    def _blocking_probe() -> str:
        gate_started.set()
        assert gate_release.wait(2.0)
        return "ok"

    cb_probe = CircuitBreaker("probe", failure_threshold=1, recovery_timeout_s=4.0,
                              half_open_max_calls=1, clock=lambda: now_box[0])
    try:
        cb_probe.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
    except ValueError:
        pass
    now_box[0] += 5.0
    assert cb_probe.state == "HALF_OPEN"
    probe_out: list = []
    probe_err: list = []

    def _run_probe() -> None:
        try:
            probe_out.append(cb_probe.call(_blocking_probe))
        except Exception as exc:  # pragma: no cover - failure diagnostics
            probe_err.append(exc)

    t_probe = threading.Thread(target=_run_probe, daemon=True)
    t_probe.start()
    assert gate_started.wait(2.0)
    assert cb_probe.call(lambda: "second", fallback=lambda: "fallback") == "fallback"
    gate_release.set()
    t_probe.join(2.0)
    assert not probe_err and probe_out == ["ok"] and cb_probe.state == "CLOSED"

    # A failed HALF_OPEN probe re-opens the circuit
    cb_hf = CircuitBreaker("hf", failure_threshold=1, recovery_timeout_s=1.0,
                           clock=lambda: now_box[0])
    try:
        cb_hf.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
    except ValueError:
        pass
    now_box[0] += 2.0
    assert cb_hf.state == "HALF_OPEN"
    try:
        cb_hf.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
    except ValueError:
        pass
    assert cb_hf.state == "OPEN"
    cb_hf.reset()
    assert cb_hf.allow_request() and cb_hf.state == "CLOSED"

    # Bulkhead rejects when saturated instead of queueing forever
    bh2 = Bulkhead(max_concurrent=1, timeout_s=0.05)
    bh_started, bh_release = threading.Event(), threading.Event()

    def _hold_slot() -> str:
        bh_started.set()
        assert bh_release.wait(2.0)
        return "done"

    t_bh = threading.Thread(target=lambda: bh2.run(_hold_slot), daemon=True)
    t_bh.start()
    assert bh_started.wait(2.0)
    try:
        bh2.run(lambda: "overflow")
        raise AssertionError("expected bulkhead rejection")
    except RateLimitExceededError:
        pass
    bh_release.set()
    t_bh.join(2.0)
    assert bh2.active == 0 and bh2.rejected == 1

    print("circuit._self_test: OK PASS (CLOSED->OPEN->HALF_OPEN->CLOSED, token bucket, bulkhead)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
