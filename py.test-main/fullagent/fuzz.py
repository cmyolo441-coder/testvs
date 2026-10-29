"""FUZZ — real property-based fuzzing.

Feeds a function a stream of generated inputs — random, boundary, and
mutated — and watches for crashes, hangs and invariant violations. When
something breaks, the engine SHRINKS the failing input to a minimal
reproducer, which is the difference between "it crashed somewhere" and
"here is the exact smallest input that breaks it."

Design (pure stdlib, deterministic under a seed):
  * Generators produce typed values (int, str, list, dict, bytes, None)
    with a bias toward boundaries (0, -1, empty, huge, unicode).
  * A property is a callable under test; any raised exception is a crash.
    An optional invariant callable can assert post-conditions.
  * Shrinking: for a failing input, repeatedly try simpler variants
    (shorter strings, smaller numbers, dropped elements) and keep the
    smallest one that still fails.
  * Runs are sealed as fuzz.run / fuzz.crash / fuzz.shrunk events.
"""

from __future__ import annotations

import random
import string
import threading
from dataclasses import dataclass, field
from typing import Callable

from .kernel import EventLog, fold
from ._foundation import get_logger

_log = get_logger("fuzz")

MAX_SHRINK_STEPS = 60
DEFAULT_CALL_TIMEOUT = 5.0   # per-invocation hang guard (seconds)
SHRINK_PROBE_TIMEOUT = 1.0   # shorter probe while shrinking hangs


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

_BOUNDARY_INTS = (0, 1, -1, 2, -2, 2**7, -2**7, 2**15, 2**31, -2**31,
                  2**63, -2**63)
_BOUNDARY_STRS = ("", " ", "\n", "\t", "\x00", "a", "abc", "A" * 64,
                  "é", "😀", "'\"\\", "<script>", "../../../etc/passwd",
                  "%s%s%s", "{0}", "$(rm -rf /)", "SELECT * FROM t;--")


class Generator:
    """Typed random input generation with a boundary bias."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def integer(self) -> int:
        if self.rng.random() < 0.35:
            return self.rng.choice(_BOUNDARY_INTS)
        return self.rng.randint(-10**6, 10**6)

    def text(self) -> str:
        if self.rng.random() < 0.35:
            return self.rng.choice(_BOUNDARY_STRS)
        n = self.rng.randint(0, 40)
        alphabet = string.printable
        return "".join(self.rng.choice(alphabet) for _ in range(n))

    def blob(self) -> bytes:
        if self.rng.random() < 0.3:
            return b""
        return self.rng.randbytes(self.rng.randint(1, 32))

    def lst(self) -> list:
        n = self.rng.randint(0, 8)
        return [self.any(depth=1) for _ in range(n)]

    def dct(self) -> dict:
        n = self.rng.randint(0, 5)
        return {self.text()[:8]: self.any(depth=1) for _ in range(n)}

    def any(self, depth: int = 0) -> object:
        if depth > 2:
            return self.integer()
        choice = self.rng.random()
        if choice < 0.25:
            return self.integer()
        if choice < 0.5:
            return self.text()
        if choice < 0.6:
            return None
        if choice < 0.7:
            return self.rng.random()
        if choice < 0.85:
            return self.lst()
        if choice < 0.92:
            return self.blob()
        return self.dct()

    def args_for(self, nargs: int) -> tuple:
        return tuple(self.any() for _ in range(nargs))


# ---------------------------------------------------------------------------
# Shrinking
# ---------------------------------------------------------------------------

def _simpler(variant: object, gen: Generator) -> list:
    """Yield simpler variants of a failing value for shrinking."""
    out: list = []
    if isinstance(variant, str):
        if variant:
            out.append("")
            out.append(variant[0])
            out.append(variant[:len(variant) // 2])
            out.append(variant[len(variant) // 2:])
            for i in range(min(len(variant), 8)):
                out.append(variant[:i] + variant[i + 1:])
    elif isinstance(variant, int):
        if variant != 0:
            out.append(0)
            if variant > 0:
                out.append(variant - 1)
            elif variant < 0:
                out.append(variant + 1)
            if variant // 2 != variant:
                out.append(variant // 2)
    elif isinstance(variant, list):
        if variant:
            out.append([])
            out.append(variant[:len(variant) // 2])
            for i in range(min(len(variant), 6)):
                out.append(variant[:i] + variant[i + 1:])
    elif isinstance(variant, dict):
        if variant:
            out.append({})
            keys = list(variant)
            for k in keys[:4]:
                d = dict(variant)
                del d[k]
                out.append(d)
    elif isinstance(variant, float):
        out.append(0.0)
    elif isinstance(variant, bytes):
        if variant:
            out.append(b"")
            out.append(variant[:len(variant) // 2])
    return out


# ---------------------------------------------------------------------------
# Fuzz engine
# ---------------------------------------------------------------------------

@dataclass
class Crash:
    args: tuple
    error: str
    shrunk_args: tuple = field(default_factory=tuple)
    shrunk_error: str = ""
    iterations: int = 0
    kind: str = "exception"  # "exception" | "hang" | "invariant"


@dataclass
class FuzzReport:
    target: str
    iterations: int = 0
    crashes: int = 0
    invariant_failures: int = 0
    first_crash: Crash | None = None
    ok: bool = True

    def to_dict(self) -> dict:
        fc = None
        if self.first_crash:
            fc = {"args": repr(self.first_crash.args)[:200],
                  "error": self.first_crash.error[:200],
                  "kind": self.first_crash.kind,
                  "shrunk_args": repr(self.first_crash.shrunk_args)[:200],
                  "shrunk_error": self.first_crash.shrunk_error[:200]}
        return {"target": self.target, "iterations": self.iterations,
                "crashes": self.crashes,
                "invariant_failures": self.invariant_failures,
                "first_crash": fc, "ok": self.ok}


def _call_guarded(target: Callable, args: tuple,
                  timeout: float) -> tuple[bool, object, str, str]:
    """Invoke target(*args) with a hang guard.

    Returns (ok, result, error, kind): ok=True on clean return (kind
    is ""); ok=False with error set on exception (kind "exception")
    OR on hang (kind "hang", the call exceeded `timeout`). The kind is
    returned explicitly — sniffing the error text for "TimeoutError"
    misclassified a target that genuinely *raised* TimeoutError as a
    hang. The worker is a daemon thread — on timeout the fuzz run
    moves on while the stuck call keeps running in the background
    (Python cannot kill threads), which is exactly the "hang" finding
    the docstring promises.
    """
    box: dict = {}

    def run() -> None:
        try:
            box["result"] = target(*args)
        except BaseException as e:  # noqa: BLE001 — any crash is a
            # finding, including KeyboardInterrupt / SystemExit /
            # GeneratorExit. (Catching only Exception silently
            # reported those as clean passes.)
            box["error"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return False, None, f"TimeoutError: hung for >{timeout:g}s", "hang"
    if "error" in box:
        return False, None, box["error"], "exception"
    return True, box.get("result"), "", ""


class Fuzzer:
    """Property-based fuzzing over the event log.

    `target(*args)` is the callable under test. `invariant(result, *args)`
    is an optional post-condition; returning False (or raising) counts as a
    failure. Neither is ever modified — the fuzzer only observes."""

    def __init__(self, log: EventLog, seed: int = 0) -> None:
        self.log = log
        self.gen = Generator(seed)

    def fuzz(self, target: Callable, iterations: int = 200, nargs: int = 1,
             invariant: Callable | None = None, name: str = "",
             timeout: float = DEFAULT_CALL_TIMEOUT) -> FuzzReport:
        # fail fast on bad input — a negative iteration count or a
        # non-callable target is a caller bug, not a fuzz finding
        if not callable(target):
            raise ValueError("target must be callable")
        if iterations is None or int(iterations) < 0:
            raise ValueError(f"iterations must be >= 0, got {iterations!r}")
        if nargs is None or int(nargs) < 0:
            raise ValueError(f"nargs must be >= 0, got {nargs!r}")
        if timeout is None or float(timeout) <= 0:
            raise ValueError(f"timeout must be positive, got {timeout!r}")
        iterations, nargs, timeout = int(iterations), int(nargs), float(timeout)
        report = FuzzReport(target=name or getattr(target, "__name__",
                                                   "target"))
        self.log.append("fuzz.run",
                        {"target": report.target, "iterations": iterations,
                         "nargs": nargs, "timeout": timeout}, actor="fuzzer")
        for i in range(iterations):
            report.iterations = i + 1
            args = self.gen.args_for(nargs)
            ok, result, error, kind = _call_guarded(target, args, timeout)
            if not ok:
                self._record_crash(report, target, args, error, kind,
                                   i + 1, timeout)
                continue
            if invariant is not None:
                try:
                    inv_ok = invariant(result, *args)
                except Exception as e:
                    self._record_crash(
                        report, target, args,
                        f"invariant raised {type(e).__name__}: {e}",
                        "invariant", i + 1, timeout, invariant)
                    continue
                if not inv_ok:
                    # a returned-False invariant is a finding too: it
                    # gets shrunk and logged like every other kind
                    # (previously it bypassed _record_crash entirely —
                    # no shrink, no fuzz.crash/fuzz.shrunk events)
                    self._record_crash(
                        report, target, args, "invariant returned False",
                        "invariant", i + 1, timeout, invariant,
                        as_invariant_failure=True)
        report.ok = report.crashes == 0 and report.invariant_failures == 0
        return report

    def _record_crash(self, report: FuzzReport, target: Callable,
                      args: tuple, error: str, kind: str, iteration: int,
                      timeout: float, invariant: Callable | None = None,
                      as_invariant_failure: bool = False) -> None:
        """Single choke point for every crash kind: count it, shrink it,
        remember the first, seal the event."""
        if as_invariant_failure:
            report.invariant_failures += 1
        else:
            report.crashes += 1
        crash = Crash(args=args, error=error, kind=kind,
                      iterations=iteration)
        self._shrink(crash, target, timeout, invariant)
        if report.first_crash is None:
            report.first_crash = crash
        self.log.append("fuzz.crash",
                        {"target": report.target, "kind": kind,
                         "args": repr(args)[:200], "error": error[:200]},
                        actor="fuzzer")

    def _shrink(self, crash: Crash, target: Callable,
                timeout: float, invariant: Callable | None = None) -> None:
        """Reduce the failing args to a minimal reproducer.

        A simplification only counts if it reproduces the SAME failure:
        the same exception type for exceptions, another hang for hangs,
        and the same invariant outcome (returned False, or raised the
        same exception type) for invariant failures. The old code never
        re-ran the invariant while shrinking, so invariant crashes
        could not shrink and were misreported as "(no longer
        reproduces)". Hang probes use a short timeout — waiting the
        full budget per candidate would make shrinking take forever."""
        want_type = crash.error.split(":")[0]
        probe = SHRINK_PROBE_TIMEOUT if crash.kind == "hang" else timeout
        # for invariant-raised crashes, remember WHICH exception type
        # the invariant raised so shrinking matches the same failure
        want_inv_exc: str | None = None
        if crash.kind == "invariant" and \
                crash.error.startswith("invariant raised "):
            want_inv_exc = crash.error[len("invariant raised "):].split(
                ":")[0]

        def reproduces(candidate: list) -> bool:
            ok, result, error, kind = _call_guarded(target,
                                                    tuple(candidate), probe)
            if crash.kind == "hang":
                return kind == "hang"
            if crash.kind == "invariant":
                if not ok or invariant is None:
                    # the target crashing is a DIFFERENT failure —
                    # shrinking must not drift onto another bug
                    return False
                try:
                    inv_ok = invariant(result, *candidate)
                except Exception as e:
                    return want_inv_exc is not None and \
                        type(e).__name__ == want_inv_exc
                return not inv_ok and want_inv_exc is None
            return not ok and error.split(":")[0] == want_type

        current = list(crash.args)
        for _ in range(MAX_SHRINK_STEPS):
            improved = False
            for idx, val in enumerate(current):
                for simpler in _simpler(val, self.gen):
                    candidate = list(current)
                    candidate[idx] = simpler
                    if reproduces(candidate):
                        current = candidate
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        crash.shrunk_args = tuple(current)
        if reproduces(list(current)):
            # re-run once more to capture the shrunk failure's text
            ok, result, error, _ = _call_guarded(target, tuple(current),
                                                 probe)
            if crash.kind == "invariant" and ok and invariant is not None:
                try:
                    if invariant(result, *current):
                        crash.shrunk_error = "(no longer reproduces)"
                    else:
                        crash.shrunk_error = "invariant returned False"
                except Exception as e:
                    crash.shrunk_error = \
                        f"invariant raised {type(e).__name__}: {e}"
            else:
                crash.shrunk_error = error
        else:
            crash.shrunk_error = "(no longer reproduces)"
        self.log.append("fuzz.shrunk",
                        {"args": repr(crash.shrunk_args)[:200],
                         "error": crash.shrunk_error[:200]},
                        actor="fuzzer")

    # -- projections -----------------------------------------------------------

    def runs(self) -> list[dict]:
        return fold(self.log).fuzz_events

    def format_status(self) -> str:
        evs = self.runs()
        runs = [e for e in evs if e["type"] == "fuzz.run"]
        crashes = [e for e in evs if e["type"] == "fuzz.crash"]
        lines = ["FUZZ", f"  runs {len(runs)}   crashes {len(crashes)}"]
        for c in crashes[-5:]:
            lines.append(f"    ⚠ {c.get('error', '')[:60]}  "
                         f"args {c.get('args', '')[:40]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "fuzz.jsonl")
        fz = Fuzzer(log, seed=42)

        # a robust function survives fuzzing
        report = fz.fuzz(lambda x: 1, iterations=100, nargs=1,
                         name="always_ok")
        assert report.ok and report.crashes == 0, report.to_dict()
        assert report.iterations == 100

        # a function that crashes on empty string is found + shrunk
        def crashy(s):
            if isinstance(s, str) and len(s) == 0:
                raise ValueError("empty!")
            return s

        report = fz.fuzz(crashy, iterations=300, nargs=1, name="crashy")
        assert report.crashes >= 1, report.to_dict()
        fc = report.first_crash
        assert fc is not None and "empty" in fc.error
        # shrinking finds the minimal reproducer: the empty string
        assert fc.shrunk_args == ("",), fc.shrunk_args
        assert "empty" in fc.shrunk_error

        # an integer crash on zero is shrunk to 0
        def divvy(n):
            if not isinstance(n, int):
                return None
            return 100 // n  # ZeroDivisionError when n == 0

        report = fz.fuzz(divvy, iterations=300, nargs=1, name="divvy")
        assert report.crashes >= 1
        fc = report.first_crash
        assert fc.shrunk_args == (0,), fc.shrunk_args

        # invariant violations are caught (sorted output must be sorted)
        def bad_sort(xs):
            if isinstance(xs, list) and len(xs) > 3:
                return list(reversed(sorted(xs)))  # wrong on purpose
            return sorted(xs) if isinstance(xs, list) else []

        def is_sorted(result, *args):
            return isinstance(result, list) and \
                all(result[i] <= result[i + 1]
                    for i in range(len(result) - 1))

        report = fz.fuzz(bad_sort, iterations=300, nargs=1,
                         invariant=is_sorted, name="bad_sort")
        assert report.invariant_failures >= 1 or report.crashes >= 1, \
            report.to_dict()

        # deterministic under the same seed
        fz2 = Fuzzer(EventLog(Path(td) / "fuzz2.jsonl"), seed=42)
        r1 = fz2.fuzz(crashy, iterations=50, nargs=1, name="crashy")
        fz3 = Fuzzer(EventLog(Path(td) / "fuzz3.jsonl"), seed=42)
        r2 = fz3.fuzz(crashy, iterations=50, nargs=1, name="crashy")
        assert r1.crashes == r2.crashes

        # events are sealed
        evs = fz.runs()
        types = {e["type"] for e in evs}
        assert {"fuzz.run", "fuzz.crash", "fuzz.shrunk"} <= types
        assert "FUZZ" in fz.format_status()

    print("FUZZ SELF-TEST PASS")
