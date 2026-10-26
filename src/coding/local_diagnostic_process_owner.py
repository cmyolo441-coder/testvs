"""Owned process-tree lifecycle for local diagnostic providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import subprocess
from threading import Event, Lock
import time
from typing import BinaryIO, Protocol

from ._hermes_child_process import terminate_process_group
from .local_diagnostic_windows_job import CtypesWindowsJobApi


_CREATE_SUSPENDED = 0x00000004
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_TERMINATE_GRACE_SECONDS = 1.0
# Correctness bound, not a performance budget: how long _await_empty_job may
# keep polling before it gives up on the Job Object's kill-on-close finishing
# asynchronously. A 1.0s shared deadline (the old behavior) can be exhausted
# by ordinary scheduling delay on a loaded CI runner, not just a stuck child.
# Same class of defect as #1599 (fixed in #1604 via
# timeout=2 -> _THREAD_DEADLINE_SECONDS = 30); see #1644.
_JOB_EMPTY_DEADLINE_SECONDS = 30.0

WINDOWS_OWNED_PROCESS_FLAGS = _CREATE_SUSPENDED | _CREATE_NEW_PROCESS_GROUP


class ProcessTreeOwner(Protocol):
    """One idempotent authority that reaps the provider's complete tree."""

    def terminate(self, first_signal: int) -> bool: ...


class WindowsJobApi(Protocol):
    def create_kill_on_close_job(self) -> int: ...

    def assign_process(self, job: int, process_handle: int) -> bool: ...

    def resume_process(self, process_id: int) -> bool: ...

    def active_processes(self, job: int) -> int | None: ...

    def terminate_job(self, job: int) -> bool: ...

    def close_job(self, job: int) -> None: ...


class PosixProcessTreeOwner:
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process
        self._lock = Lock()
        self._result: bool | None = None

    def terminate(self, first_signal: int) -> bool:
        with self._lock:
            if self._result is None:
                _signals, self._result = terminate_process_group(
                    self._process,
                    _TERMINATE_GRACE_SECONDS,
                    first_signal,
                )
            return self._result


class WindowsJobObjectOwner:
    """Kill-on-close Job Object assigned before the provider can execute."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        api: WindowsJobApi,
        job: int,
    ) -> None:
        self._process = process
        self._api = api
        self._job = job
        self._lock = Lock()
        self._result: bool | None = None
        # Diagnostic only -- never read by terminate()'s own control flow, and
        # never changes what terminate() returns. Distinguishes the two waits
        # a bare `False` result collapses together (#1644): the Job Object
        # never reporting empty vs. the leader process handle not reporting
        # exit within the grace period once the job did empty.
        self.termination_diagnostic: str | None = None

    @classmethod
    def attach(
        cls,
        process: subprocess.Popen[bytes],
        *,
        api: WindowsJobApi | None = None,
    ) -> WindowsJobObjectOwner:
        windows_api = api or CtypesWindowsJobApi()
        job = windows_api.create_kill_on_close_job()
        process_handle = int(getattr(process, "_handle"))
        if not windows_api.assign_process(job, process_handle):
            windows_api.close_job(job)
            raise OSError("diagnostic process could not enter its Windows Job Object")
        if not windows_api.resume_process(process.pid):
            windows_api.terminate_job(job)
            windows_api.close_job(job)
            raise OSError("diagnostic process could not resume inside its Windows Job Object")
        return cls(process, api=windows_api, job=job)

    def terminate(self, first_signal: int) -> bool:
        del first_signal
        with self._lock:
            if self._result is not None:
                return self._result
            active = self._api.active_processes(self._job)
            terminated = active == 0
            if active is not None and active > 0:
                terminated = self._api.terminate_job(self._job)
            empty = terminated and _await_empty_job(self._api, self._job)
            self.termination_diagnostic = None if empty else "job_did_not_empty"
            try:
                self._process.wait(timeout=_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                reaped = False
                if empty:
                    self.termination_diagnostic = "process_wait_timed_out"
            else:
                reaped = True
            self._api.close_job(self._job)
            self._result = empty and reaped
            return self._result


def start_owned_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout: int | BinaryIO | None,
    stderr: int | BinaryIO | None,
) -> tuple[subprocess.Popen[bytes], ProcessTreeOwner]:
    """Start suspended on Windows, bind ownership, then allow execution."""
    windows = os.name == "nt"
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=dict(env),
        stdout=stdout,
        stderr=stderr,
        start_new_session=not windows,
        creationflags=WINDOWS_OWNED_PROCESS_FLAGS if windows else 0,
    )
    if not windows:
        return process, PosixProcessTreeOwner(process)
    try:
        return process, WindowsJobObjectOwner.attach(process)
    except BaseException:  # cleanup boundary: a suspended child must never escape
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            pass
        raise


def _await_empty_job(api: WindowsJobApi, job: int) -> bool:
    deadline = time.monotonic() + _JOB_EMPTY_DEADLINE_SECONDS
    while True:
        active = api.active_processes(job)
        if active is None:
            return False
        if active == 0:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        Event().wait(min(0.01, remaining))
