"""Advanced Parallel Execution Engine for Independent Read-Only Tool Calls.

Executes maximal runs of independent read-only tool calls concurrently in a
bounded, adaptive ThreadPoolExecutor while preserving strict model-emitted
transcript order and read-after-write barriers:

* ``PARALLEL_READ_TOOLS`` is strictly gated to ``SPECULATIVE_TOOLS``
  (read_file, list_dir, file_info, search_files, glob_files).
* ``dispatch_block()`` walks tool-call blocks left to right, executing
  consecutive read batches concurrently while flushing before and after any
  mutating or interactive tool call.
* Records structured ``BatchMetrics`` (wall time, summed latency, speedup factor,
  peak concurrency) for telemetry and diagnostics.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from .cancelguard import TurnCancelled
from .speculate import SPECULATIVE_TOOLS

#: Tools that may run concurrently — the Speculator's read-only whitelist.
PARALLEL_READ_TOOLS = frozenset(SPECULATIVE_TOOLS)

#: Cap on fan-out: enough to hide I/O latency, small enough to stay polite.
PARALLEL_MAX_WORKERS = 8


@dataclass(frozen=True)
class BatchMetrics:
    """Telemetry from a parallel tool-call batch execution."""

    batch_size: int
    workers_used: int
    wall_time_s: float
    sum_duration_s: float
    speedup: float


_METRICS_LOCK = threading.Lock()
_LAST_BATCH_METRICS: BatchMetrics | None = None


def get_last_batch_metrics() -> BatchMetrics | None:
    """Return telemetry from the most recent parallel read batch."""
    with _METRICS_LOCK:
        return _LAST_BATCH_METRICS


def is_parallelizable(tool_name: str, *, hooks_enabled: bool = False) -> bool:
    """True only for structurally read-only tools when interactive hooks are off."""
    if hooks_enabled:
        return False
    return tool_name in PARALLEL_READ_TOOLS


def plan_batches(
    tool_names: list[str], *, hooks_enabled: bool = False
) -> list[list[int]]:
    """Partition call indices into parallel batches of >= 2 consecutive read calls."""
    batches: list[list[int]] = []
    i, n = 0, len(tool_names)
    while i < n:
        if not is_parallelizable(tool_names[i], hooks_enabled=hooks_enabled):
            i += 1
            continue
        j = i
        while j < n and is_parallelizable(
            tool_names[j], hooks_enabled=hooks_enabled
        ):
            j += 1
        if j - i >= 2:
            batches.append(list(range(i, j)))
        i = j
    return batches


def _run_batch(
    execute: Callable[..., None],
    batch: list[tuple[dict, Any]],
    *,
    approve: Any,
    on_status: Callable[[str], None],
    causation_id: str | None,
    on_tool_output: Callable[[str, str], None] | None,
) -> BatchMetrics:
    """Run one batch of ToolEvents concurrently and record BatchMetrics."""
    global _LAST_BATCH_METRICS
    workers = max(1, min(PARALLEL_MAX_WORKERS, len(batch)))
    durations: list[float] = [0.0] * len(batch)

    def _run(indexed_item: tuple[int, tuple[dict, Any]]) -> None:
        idx, (_, ev) = indexed_item
        t0 = time.perf_counter()
        try:
            execute(
                ev,
                approve,
                on_status,
                causation_id=causation_id,
                on_tool_output=on_tool_output,
            )
        except TurnCancelled:
            # Esc/Ctrl+C during a parallel read batch must abort the turn.
            # Swallowing it left ev.status "running" and the turn continuing.
            raise
        except Exception as exc:
            # Never leave a phantom "running" event behind: seal it as an
            # error so the transcript/log stay consistent (worker 3 fix).
            try:
                ev.status = "error"
                ev.result = f"ERROR: {type(exc).__name__}: {exc}"
            except Exception:
                pass
        finally:
            durations[idx] = time.perf_counter() - t0

    t_start = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="paratool"
    ) as ex:
        list(ex.map(_run, enumerate(batch)))
    wall_s = max(1e-6, time.perf_counter() - t_start)
    sum_s = sum(durations)
    metrics = BatchMetrics(
        batch_size=len(batch),
        workers_used=workers,
        wall_time_s=wall_s,
        sum_duration_s=sum_s,
        speedup=round(sum_s / wall_s, 2) if wall_s > 0 else 1.0,
    )
    with _METRICS_LOCK:
        _LAST_BATCH_METRICS = metrics
    return metrics


def dispatch_block(
    execute: Callable[..., None],
    finish: Callable[[dict, Any], bool],
    pending: list[tuple[dict, Any]],
    *,
    approve: Any,
    on_status: Callable[[str], None],
    causation_id: str | None = None,
    on_tool_output: Callable[[str, str], None] | None = None,
    hooks_enabled: bool = False,
) -> None:
    """Execute one model-emitted tool-call block in deterministic order."""
    i, n = 0, len(pending)
    while i < n:
        if is_parallelizable(
            pending[i][1].name, hooks_enabled=hooks_enabled
        ):
            j = i
            while j < n and is_parallelizable(
                pending[j][1].name, hooks_enabled=hooks_enabled
            ):
                j += 1
            if j - i >= 2:
                _run_batch(
                    execute,
                    pending[i:j],
                    approve=approve,
                    on_status=on_status,
                    causation_id=causation_id,
                    on_tool_output=on_tool_output,
                )
                for tc_b, ev_b in pending[i:j]:
                    if finish(tc_b, ev_b):
                        return
                i = j
                continue
        tc, ev = pending[i]
        execute(
            ev,
            approve,
            on_status,
            causation_id=causation_id,
            on_tool_output=on_tool_output,
        )
        if finish(tc, ev):
            return
        i += 1


# ---------------------------------------------------------------------------
# Self-test — proves parallel speedup, result ordering, metrics, and write safety.
# Run:  python3 -m fullagent.parallel_tools
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from .tools import read_file as _real_read_file
    from .tools import write_file as _real_write_file

    LAT = 0.25

    def _ev(name: str, args: dict) -> SimpleNamespace:
        return SimpleNamespace(
            name=name, args=args, result="", status="running", duration=0.0
        )

    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(5):
            p = Path(td) / f"f{i}.txt"
            p.write_text(f"content-of-file-{i}\n" * 20)
            paths.append(str(p))

        def _timed_execute(
            ev, approve, on_status, causation_id=None, on_tool_output=None
        ):
            t0 = time.time()
            time.sleep(LAT)
            ev.result = _real_read_file(**ev.args)
            ev.status = "done"
            ev.duration = time.time() - t0

        def _sequential(pending):
            t0 = time.time()
            for _, ev in pending:
                _timed_execute(ev, None, lambda s: None)
            return time.time() - t0

        seq_pending = [
            ({"id": f"c{i}"}, _ev("read_file", {"path": p}))
            for i, p in enumerate(paths)
        ]
        t_seq = _sequential(seq_pending)

        par_pending = [
            ({"id": f"c{i}"}, _ev("read_file", {"path": p}))
            for i, p in enumerate(paths)
        ]
        finished: list[str] = []
        t0 = time.time()
        dispatch_block(
            _timed_execute,
            lambda tc, ev: finished.append(tc["id"]) or False,
            par_pending,
            approve=None,
            on_status=lambda s: None,
        )
        t_par = time.time() - t0

        for i, (_, ev) in enumerate(par_pending):
            assert f"content-of-file-{i}" in ev.result
            assert ev.result == seq_pending[i][1].result
        assert finished == [f"c{i}" for i in range(5)]
        speedup = t_seq / t_par
        m = get_last_batch_metrics()
        assert m is not None and m.batch_size == 5 and m.speedup >= 3.0
        print(
            f"[1] 5 reads  sequential={t_seq:.2f}s  "
            f"parallel={t_par:.2f}s  speedup={speedup:.1f}x (metrics={m.speedup}x)"
        )
        assert speedup >= 3.0

        names = [
            "read_file",
            "write_file",
            "read_file",
            "read_file",
            "run_command",
            "glob_files",
        ]
        batches = plan_batches(names)
        assert batches == [[2, 3]]
        assert plan_batches(["read_file"]) == []
        assert plan_batches(["write_file", "edit_file"]) == []
        assert plan_batches(names, hooks_enabled=True) == []
        print(
            "[2] plan_batches: writes flush batches, mutating tools "
            "never parallelize, hooks force sequential — OK"
        )

        order: list[str] = []
        state = {"inflight": 0, "peak": 0}
        order_lock = threading.Lock()

        def _rec_execute(
            ev, approve, on_status, causation_id=None, on_tool_output=None
        ):
            with order_lock:
                state["inflight"] += 1
                state["peak"] = max(state["peak"], state["inflight"])
                cur = state["inflight"]
            try:
                time.sleep(0.15)
                with order_lock:
                    order.append(
                        f"{ev.name}:{ev.args.get('path', '')}@{cur}"
                    )
                ev.result, ev.status = "ok", "done"
            finally:
                with order_lock:
                    state["inflight"] -= 1

        mixed = [
            ({"id": "c0"}, _ev("read_file", {"path": "a"})),
            ({"id": "c1"}, _ev("write_file", {"path": "b", "content": "x"})),
            ({"id": "c2"}, _ev("read_file", {"path": "c"})),
            ({"id": "c3"}, _ev("read_file", {"path": "d"})),
        ]
        transcript: list[str] = []
        dispatch_block(
            _rec_execute,
            lambda tc, ev: transcript.append(tc["id"]) or False,
            mixed,
            approve=None,
            on_status=lambda s: None,
        )

        kinds = [o.split("@")[0].split(":")[0] for o in order]
        assert kinds == ["read_file", "write_file", "read_file", "read_file"]
        assert transcript == ["c0", "c1", "c2", "c3"]
        assert order.count("write_file:b@1") == 1
        assert state["peak"] >= 2
        print(
            f"[3] mixed block exec order={kinds} transcript={transcript} "
            f"write ran once @inflight=1, reads overlapped "
            f"(peak={state['peak']}) — OK"
        )

        target = str(Path(td) / "w.txt")
        Path(target).write_text("old\n")

        def _real_execute(
            ev, approve, on_status, causation_id=None, on_tool_output=None
        ):
            if ev.name == "write_file":
                ev.result = _real_write_file(**ev.args)
            else:
                ev.result = _real_read_file(**ev.args)
            ev.status = "done"

        w_pending = [
            ({"id": "c0"}, _ev("read_file", {"path": paths[0]})),
            ({"id": "c1"}, _ev("read_file", {"path": paths[1]})),
            (
                {"id": "c2"},
                _ev("write_file", {"path": target, "content": "new-data\n"}),
            ),
            ({"id": "c3"}, _ev("read_file", {"path": target})),
            ({"id": "c4"}, _ev("read_file", {"path": paths[2]})),
        ]
        seen: list[str] = []
        dispatch_block(
            _real_execute,
            lambda tc, ev: seen.append(ev.result) or False,
            w_pending,
            approve=None,
            on_status=lambda s: None,
        )
        assert "new-data" in w_pending[3][1].result
        assert "new-data" not in w_pending[0][1].result
        print("[4] read-after-write sees the write (real handlers) — OK")

    print(
        "\nparatools self-test: ALL PASS "
        f"(whitelist={sorted(PARALLEL_READ_TOOLS)}, "
        f"max_workers={PARALLEL_MAX_WORKERS})"
    )
