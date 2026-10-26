"""Private process and output helpers for isolated Hermes children."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import ctypes
from dataclasses import dataclass
import os
import signal
import subprocess
from threading import Event, Thread, current_thread, main_thread
import time
from typing import BinaryIO, Final

from ..system.metadata_safety import is_sensitive_metadata_text
from ..system.output_truncation import truncation_notice, truncation_record

_FORCE_KILL_SIGNAL: Final = getattr(signal, "SIGKILL", signal.SIGTERM)
MAX_CAPTURE_BYTES: Final = 16_384
_READ_CHUNK_BYTES: Final = 64 * 1024


@dataclass(frozen=True, slots=True)
class BoundedStreamCapture:
    """A hard-capped stream prefix plus non-content truncation metadata."""

    data: bytes
    truncated: bool
    # Total bytes the drainer read before discarding, counted rather than
    # retained, so a truncation record can state the real original size instead
    # of reporting it as unknown. Zero means "not counted by this producer",
    # which the record renders as unknown rather than as an empty stream.
    seen_bytes: int = 0


class PipeDrainer:
    """Drain one pipe to EOF while retaining no more than the byte cap."""

    def __init__(self, pipe: BinaryIO, *, name: str, limit_bytes: int = MAX_CAPTURE_BYTES) -> None:
        self._pipe = pipe
        # Per-instance so one caller can read a larger document without
        # raising the cap for every child this module drains. The default is
        # the shared cap; only a caller that states a reason passes more.
        self._limit = max(0, int(limit_bytes))
        self._parts: list[bytes] = []
        self._retained = 0
        self._seen = 0
        self._truncated = False
        self._error = False
        self.done = Event()
        self.thread = Thread(target=self._run, name=name, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def capture(self) -> BoundedStreamCapture:
        return BoundedStreamCapture(
            b"".join(self._parts), self._truncated or self._error, self._seen
        )

    def close(self) -> None:
        try:
            self._pipe.close()
        except OSError:
            pass

    def _run(self) -> None:
        try:
            while True:
                chunk = self._pipe.read(_READ_CHUNK_BYTES)
                if not chunk:
                    break
                # Counted, never retained: this is what lets the truncation
                # record name the real original size at zero memory cost.
                self._seen += len(chunk)
                remaining = self._limit - self._retained
                if remaining > 0:
                    kept = chunk[:remaining]
                    self._parts.append(kept)
                    self._retained += len(kept)
                if len(chunk) > remaining:
                    self._truncated = True
        except (OSError, ValueError):
            # A main-thread emergency close is itself represented as truncated;
            # partial diagnostics remain safe and bounded.
            self._error = True
        finally:
            self.close()
            self.done.set()


def start_pipe_drainers(
    process: subprocess.Popen[bytes], *, limit_bytes: int = MAX_CAPTURE_BYTES
) -> tuple[PipeDrainer, PipeDrainer]:
    """Start both readers before the child can fill either OS pipe."""
    if process.stdout is None or process.stderr is None:
        raise ValueError("Hermes child output pipes are required")
    stdout = PipeDrainer(process.stdout, name=f"hermes-stdout-{process.pid}", limit_bytes=limit_bytes)
    stderr = PipeDrainer(process.stderr, name=f"hermes-stderr-{process.pid}", limit_bytes=limit_bytes)
    stdout.start()
    stderr.start()
    return stdout, stderr


def write_stdin_and_close(process: subprocess.Popen[bytes], prompt: str) -> tuple[Thread, Event]:
    """Write stdin off-thread so a non-reading child cannot defeat the timeout."""
    done = Event()

    def write() -> None:
        try:
            if process.stdin is not None:
                process.stdin.write(prompt.encode("utf-8"))
                process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            done.set()

    thread = Thread(target=write, name=f"hermes-stdin-{process.pid}", daemon=True)
    thread.start()
    return thread, done


def await_pipe_threads(
    drainers: tuple[PipeDrainer, PipeDrainer],
    stdin_thread: tuple[Thread, Event],
    *,
    timeout: float,
) -> bool:
    """Await exact pipe/thread completion signals within one monotonic budget."""
    deadline = time.monotonic() + max(0.05, timeout)
    threads = [(stdin_thread[0], stdin_thread[1]), *((item.thread, item.done) for item in drainers)]
    complete = True
    for thread, done in threads:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not done.wait(remaining):
            complete = False
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        complete = complete and not thread.is_alive()
    if not complete:
        if stdin_thread[0].is_alive():
            # Closing stdin unblocks a child that stopped reading before exit.
            # Popen's pipe object may already have been closed by the writer.
            pass
        for drainer in drainers:
            if drainer.thread.is_alive():
                drainer.close()
        for thread, _done in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        complete = all(not thread.is_alive() for thread, _done in threads)
    return complete


def terminate_process_group(
    process: subprocess.Popen[bytes], grace: float, first_signal: int
) -> tuple[tuple[int, ...], bool]:
    """Signal the process group, reap its leader, and prove group exit."""
    if os.name == "nt":
        return _terminate_windows_process_tree(process, grace, first_signal)
    sent: list[int] = []
    if group_absent(process.pid):
        return (), _reap_process(process, 0.05)
    for signum in (first_signal, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
            sent.append(signum)
        except ProcessLookupError:
            break
        except PermissionError:
            try:
                process.send_signal(signum)
                sent.append(signum)
            except ProcessLookupError:
                pass
        _reap_process(process, max(0.05, grace))
        if group_absent(process.pid):
            break
    reaped = _reap_process(process, max(0.05, grace))
    return tuple(sent), reaped and _await_group_absent(process.pid, max(0.1, grace))


def _terminate_windows_process_tree(
    process: subprocess.Popen[bytes],
    grace: float,
    first_signal: int,
) -> tuple[tuple[int, ...], bool]:
    if process.poll() is not None:
        return (), True
    sent = [first_signal]
    try:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=max(1.0, grace + 1.0),
        )
        sent.append(_FORCE_KILL_SIGNAL)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            sent.append(_FORCE_KILL_SIGNAL)
        except OSError:
            pass
    return tuple(sent), _reap_process(process, max(0.1, grace))


def _await_group_absent(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not group_absent(pid):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        Event().wait(min(0.01, remaining))
    return True


def _reap_process(process: subprocess.Popen[bytes], timeout: float) -> bool:
    if process.poll() is not None:
        return True
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return True


def group_absent(pid: int) -> bool:
    if os.name == "nt":
        return process_absent(pid)
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def process_absent(pid: int) -> bool:
    """Return whether a process ID no longer identifies a live process."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return ctypes.get_last_error() in {87, 1168}
    try:
        exit_code = ctypes.c_uint32()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value != 259
    finally:
        kernel32.CloseHandle(handle)


def capture_truncation_record(
    captured: BoundedStreamCapture | bytes | str | object,
    *,
    source: str,
) -> dict[str, object]:
    """The `omh_output_truncation/v1` record for one bounded stream capture.

    Nothing spills here, and that is not an omission: the drainer enforces the
    cap while reading, so the discarded bytes never existed in this process to
    write anywhere. The record says exactly that (`content_not_retained`)
    rather than implying a pointer that could never be produced.
    """
    raw, truncated, seen = _capture_parts(captured)
    return truncation_record(
        source=source,
        reason_code="capture_cap" if truncated else "not_truncated",
        limit_bytes=MAX_CAPTURE_BYTES,
        kept_bytes=len(raw),
        original_bytes=seen if seen >= len(raw) else None,
        keep="head",
        spill_status="content_not_retained" if truncated else "not_attempted",
    )


def bounded_redacted_output(
    captured: BoundedStreamCapture | bytes | str | object,
    *,
    secrets: Iterable[str],
    source: str = "hermes child stream capture",
) -> str:
    """Redact a byte-bounded diagnostic and attach a content-free continuation notice."""
    raw, truncated, _seen = _capture_parts(captured)
    value = raw.decode("utf-8", errors="replace")
    known_secrets = sorted({item for item in secrets if len(item) >= 4}, key=len, reverse=True)
    for secret in known_secrets:
        value = value.replace(secret, "[redacted]")
        if truncated:
            # A byte cap can bisect a credential. Remove the longest retained
            # prefix so truncation never turns redaction into a partial leak.
            overlap = _suffix_prefix_overlap(value, secret)
            if overlap:
                value = value[:-overlap] + "[redacted]"
    value = "".join(
        "[redacted]\n" if is_sensitive_metadata_text(line) else line
        for line in value.splitlines(keepends=True)
    )
    if not truncated:
        return value
    notice = truncation_notice(capture_truncation_record(captured, source=source))
    return f"{value}\n{notice}\n"


def _capture_parts(
    captured: BoundedStreamCapture | bytes | str | object,
) -> tuple[bytes, bool, int]:
    """Normalize the three capture shapes to (retained bytes, truncated, bytes seen)."""
    if isinstance(captured, BoundedStreamCapture):
        return captured.data, captured.truncated, captured.seen_bytes
    if isinstance(captured, bytes):
        return captured[:MAX_CAPTURE_BYTES], len(captured) > MAX_CAPTURE_BYTES, len(captured)
    raw_value = str(captured or "").encode("utf-8", errors="replace")
    return raw_value[:MAX_CAPTURE_BYTES], len(raw_value) > MAX_CAPTURE_BYTES, len(raw_value)


def _suffix_prefix_overlap(value: str, secret: str) -> int:
    """Find a cap-bisected secret in linear time and bounded memory."""
    limit = min(len(value), len(secret) - 1)
    if limit <= 0:
        return 0
    prefix = secret[:limit]
    failure = [0] * len(prefix)
    matched = 0
    for index in range(1, len(prefix)):
        while matched and prefix[index] != prefix[matched]:
            matched = failure[matched - 1]
        if prefix[index] == prefix[matched]:
            matched += 1
        failure[index] = matched
    matched = 0
    for character in value[-limit:]:
        while matched and character != prefix[matched]:
            matched = failure[matched - 1]
        if character == prefix[matched]:
            matched += 1
            if matched == len(prefix):
                return matched
    return matched


def block_relay_signals() -> object | None:
    """Block relay signals across spawn until process-group handlers exist."""
    if current_thread() is not main_thread() or not hasattr(signal, "pthread_sigmask"):
        return None
    return signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})


def restore_signal_mask(mask: object | None) -> None:
    if mask is not None:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)  # type: ignore[arg-type]


def install_signal_relays(process: subprocess.Popen[bytes]) -> dict[int, object]:
    """Relay parent terminal signals into the isolated child process group."""
    if current_thread() is not main_thread():
        return {}
    old: dict[int, object] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old[signum] = signal.getsignal(signum)

        def relay(received: int, _frame: object) -> None:
            try:
                os.killpg(process.pid, received)
            except ProcessLookupError:
                pass
            if received == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(128 + received)

        signal.signal(signum, relay)
    return old


def restore_signal_handlers(handlers: Mapping[int, object]) -> None:
    for signum, handler in handlers.items():
        signal.signal(signum, handler)  # type: ignore[arg-type]


def merge_signals(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
    return first + tuple(item for item in second if item not in first)
