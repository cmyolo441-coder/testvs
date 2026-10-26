"""Hard-capped pipe draining for local diagnostic subprocesses."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread
from typing import BinaryIO


_READ_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class BoundedDiagnosticCapture:
    """A retained output prefix plus whether any bytes were discarded."""

    data: bytes
    truncated: bool


class DiagnosticPipeDrainer:
    """Drain to EOF while retaining no more than the configured byte cap."""

    def __init__(
        self,
        pipe: BinaryIO,
        *,
        max_bytes: int,
        name: str,
    ) -> None:
        self._pipe = pipe
        self._max_bytes = max_bytes
        self._parts: list[bytes] = []
        self._retained = 0
        self._truncated = False
        self._error = False
        self.done = Event()
        self.thread = Thread(target=self._run, name=name, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def finish(self, timeout: float) -> BoundedDiagnosticCapture:
        complete = self.done.wait(timeout)
        if not complete:
            self.close()
        self.thread.join(timeout=timeout)
        return BoundedDiagnosticCapture(
            data=b"".join(self._parts),
            truncated=(
                self._truncated
                or self._error
                or not complete
                or self.thread.is_alive()
            ),
        )

    def close(self) -> None:
        try:
            self._pipe.close()
        except OSError:
            pass

    def _run(self) -> None:
        try:
            while chunk := self._pipe.read(_READ_CHUNK_BYTES):
                remaining = self._max_bytes - self._retained
                if remaining > 0:
                    kept = chunk[:remaining]
                    self._parts.append(kept)
                    self._retained += len(kept)
                if len(chunk) > remaining:
                    self._truncated = True
        except (OSError, ValueError):
            self._error = True
        finally:
            self.close()
            self.done.set()
