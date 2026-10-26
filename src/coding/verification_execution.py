"""Dispatch-scoped concurrency and resource policy for verification checks."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import threading
from collections.abc import Callable
from typing import TypeVar

from .verification_plan import VerificationNode

_T = TypeVar("_T")


class VerificationExecutionGate:
    """Bound every verification process in one fanout dispatch.

    The gate is deliberately created by the dispatcher, then shared by every
    unit plan and its post-integration wave. Its one executor applies the
    policy width across plans; stateful checks additionally acquire the lock
    for their declared resource class, regardless of which producer requested
    them.
    """

    def __init__(self, max_workers: int) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max(1, max_workers))
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def submit(self, node: VerificationNode, run: Callable[[], _T]) -> Future[_T]:
        """Queue one planned check under the dispatch-wide resource policy."""
        return self._submit(node.safety, node.resource_class, run)

    def submit_legacy(self, run: Callable[[], _T]) -> Future[_T]:
        """Queue one compatibility command as a stateful local-CPU check."""
        return self._submit("stateful", "local_cpu", run)

    def shutdown(self) -> None:
        """Wait for the dispatch's submitted verification work to finish."""
        self._pool.shutdown(wait=True)

    def _submit(self, safety: str, resource_class: str, run: Callable[[], _T]) -> Future[_T]:
        return self._pool.submit(self._run, safety, resource_class, run)

    def _run(self, safety: str, resource_class: str, run: Callable[[], _T]) -> _T:
        if safety == "read_only":
            return run()
        with self._locks_guard:
            lock = self._locks.setdefault(resource_class, threading.Lock())
        with lock:
            return run()
