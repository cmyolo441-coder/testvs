"""Bounded producer engine for normalized metadata and supported native adapters.

Importing this module does not collect anything. Only constructing ActivityObserver
starts its owned worker; plugin registration never does so on unsupported hosts.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
import json
from threading import Condition, Event, Thread
from time import monotonic
from typing import TypedDict, assert_never, final

try:  # The receipt sink and the shared atomic writer are core-owned.
    # This engine produces normalized intervals; admission and storage stay in
    # the OMH package, so a copy of either here would be a second
    # implementation of the same rules. Binding them has to stay OPTIONAL all
    # the same: Hermes' loaders exec every top-level file of the bundle -- the
    # memory-provider lane does it eagerly -- and keep the half-initialized
    # module in `sys.modules` when one raises, which turned "this host has no
    # observer" into an ImportError on every tool call (#1623). Construction
    # is where the absence is reported; `register` in the bundle's
    # __init__.py turns that into the `observer_setup_failed` status.
    from omh.system.paths import OmhPaths
    from omh.system.local_store import atomic_write_json
    from omh.workflows.session_activity_receipts import ingest_session_activity_receipt
except ImportError:  # pragma: no cover - standalone plugin hosts have no omh package.
    _CORE_AVAILABLE = False
else:
    _CORE_AVAILABLE = True

from .activity_observer_events import ActivityEvent, EventError, EventValue, Kind, OpaqueRef, opaque_ref, parse_event
from .activity_observer_state import JSON, Room

__all__ = ["ActivityObserver", "opaque_ref"]


class ObserverStatus(TypedDict):
    readiness: str
    compatibility: str
    dropped: int
    gapped: int
    rejected: int
    write_failed: int
    last_outcome: str


@final
class ActivityObserver:
    """Single-owner aggregation with bounded nonblocking submission.

    The condition protects queue/control counters only, never aggregation or I/O.
    Call close and check its result before disposing the invocation's state home.
    """
    def __init__(self, paths: OmhPaths, profile_ref: OpaqueRef, *, on_status: Callable[[ObserverStatus], None] | None = None) -> None:
        if not _CORE_AVAILABLE:
            # Before the worker exists: a host with no receipt store has
            # nothing for this engine to produce into, and a thread that
            # discovered that later would have nowhere to report it.
            raise ImportError("the omh package is not importable, so no session activity can be recorded")
        self.paths, self.profile_ref = paths, profile_ref
        self._on_status = on_status
        self.checkpoint_path = paths.runtime_dir / f"group-activity-{profile_ref[7:]}.json"
        self._condition = Condition()
        self._pending: deque[ActivityEvent | Event] = deque()
        self._rooms: dict[tuple[OpaqueRef, OpaqueRef, OpaqueRef], Room] = {}
        self._terminal: dict[tuple[OpaqueRef, OpaqueRef, OpaqueRef], tuple[float, Room]] = {}
        self._closing = False
        self._done = Event()
        self._status: ObserverStatus = {"readiness": "local_engine", "compatibility": "normalized_input_only",
            "dropped": 0, "gapped": 0, "rejected": 0, "write_failed": 0, "last_outcome": "not_observed"}
        self._worker = Thread(target=self._run, name="omh-group-activity", daemon=True)
        self._worker.start()

    def enqueue(self, raw: Mapping[str, EventValue]) -> bool:
        """Parse and enqueue only; never wait for aggregation, checkpoint, or receipt I/O."""
        try:
            event = parse_event(raw, self.profile_ref)
        except EventError as exc:
            self.reject_event(exc.code)
            return False
        with self._condition:
            if self._closing or len(self._pending) >= 1024:
                self._status["dropped"] = min(10**9, self._status["dropped"] + 1)
                self._status["last_outcome"] = "queue_dropped" if not self._closing else "observer_closed"
                return False
            self._pending.append(event)
            self._condition.notify()
        return True

    def status(self) -> ObserverStatus:
        with self._condition:
            return self._status.copy()

    def flush(self, timeout: float = 10) -> bool:
        """Await a queue barrier subscribed before enqueue, not timing-based polling."""
        barrier = Event()
        with self._condition:
            if self._closing:
                return self._done.is_set()
            if not self._condition.wait_for(lambda: len(self._pending) < 1024, timeout):
                return False
            self._pending.append(barrier)
            self._condition.notify()
        return barrier.wait(timeout)

    def close(self, timeout: float = 10) -> bool:
        """Bounded unload drain. A timeout reports loss, never blocks a host indefinitely."""
        with self._condition:
            self._closing = True
            self._condition.notify()
        if not self._done.wait(timeout):
            with self._condition:
                self._status["dropped"] = min(10**9, self._status["dropped"] + 1)
                self._status["last_outcome"] = "drain_timeout"
            return False
        self._worker.join()
        return True

    def reject_event(self, reason: str) -> None:
        """Record a bounded adapter/parser category without retaining rejected metadata."""
        with self._condition:
            self._status["rejected"] = min(10**9, self._status["rejected"] + 1)
            self._status["last_outcome"] = reason

    def _failed_write(self) -> None:
        with self._condition:
            self._status["write_failed"] = min(10**9, self._status["write_failed"] + 1)
            self._status["last_outcome"] = "write_failed"

    def _recover(self) -> None:
        try:
            with self.checkpoint_path.open("rb") as handle:
                raw = handle.read(131073)
            if len(raw) > 131072:
                raise EventError("invalid_checkpoint")
            data: JSON = json.loads(raw)
            if not isinstance(data, dict) or set(data) != {"schema", "rooms"} or data["schema"] != "omh_group_activity_checkpoint/v1":
                raise EventError("invalid_checkpoint")
            records = data["rooms"]
            if not isinstance(records, list) or len(records) > 64:
                raise EventError("invalid_checkpoint")
            recovered = [Room.recover(record, self.profile_ref) for record in records]
            self._rooms = {(room.profile, room.room, room.session): room for room in recovered}
        except FileNotFoundError:
            return
        except (EventError, ValueError, UnicodeError):
            self.reject_event("invalid_checkpoint")
            # Missing trust in a checkpoint permanently lowers subsequent intervals.
            with self._condition:
                self._status["dropped"] += 1
        except OSError:
            self._failed_write()
            with self._condition:
                self._status["dropped"] += 1

    def _checkpoint(self) -> None:
        try:
            atomic_write_json(self.checkpoint_path, {"schema": "omh_group_activity_checkpoint/v1",
                "rooms": [room.checkpoint() for room in self._rooms.values()]}, private=True)
        except OSError:
            self._failed_write()
            for room in self._rooms.values():
                room.partial = True
        self._publish_status()

    def _publish_status(self) -> None:
        if self._on_status is not None:
            try:
                self._on_status(self.status())
            except OSError:
                self._failed_write()

    def _emit(self, room: Room, final: bool) -> None:
        room.partial |= bool(self.status()["dropped"])
        try:
            result = ingest_session_activity_receipt(self.paths, room.receipt(final), expected_profile_ref=self.profile_ref)
            with self._condition:
                self._status["last_outcome"] = result["outcome"]
        except (OSError, ValueError):
            self._failed_write()

    def _aggregate(self, event: ActivityEvent) -> None:
        now = monotonic()
        self._terminal = {key: value for key, value in self._terminal.items() if now - value[0] < 86400}
        terminal = self._terminal.get(event.scope)
        if terminal is not None:
            if not terminal[1].is_replay(event):
                self.reject_event("stale_after_final")
            return  # Admission remains authoritative after this bounded cache expires.
        room = self._rooms.get(event.scope)
        if room is None:
            if len(self._rooms) >= 64:
                with self._condition:
                    self._status["dropped"] = min(10**9, self._status["dropped"] + 1)
                    self._status["last_outcome"] = "room_capacity"
                return
            room = Room.start(event)
            self._rooms[event.scope] = room
        try:
            outcome = room.accept(event)
        except EventError as exc:
            self.reject_event(exc.code)
            return
        match outcome:
            case "duplicate":
                return
            case "gap":
                with self._condition:
                    self._status["gapped"] = min(10**9, self._status["gapped"] + 1)
            case "accepted":
                pass
            case unreachable:
                assert_never(unreachable)
        match event.kind:
            case Kind.SESSION_END:
                self._emit(room, True)
                del self._rooms[event.scope]
                if len(self._terminal) >= 64:
                    del self._terminal[next(iter(self._terminal))]
                self._terminal[event.scope] = (now, room)
            case Kind.SESSION_START | Kind.MEMBER_START | Kind.MEMBER_COMPLETE | Kind.TOOL_CALL | Kind.TOOL_ERROR | Kind.COMPACTION | Kind.ACTIVITY:
                pass
            case unreachable:
                assert_never(unreachable)
        self._checkpoint()

    def _run(self) -> None:
        try:
            self._recover()
            self._publish_status()
            while True:
                with self._condition:
                    # Wake at the earliest retention deadline, not a polling interval.
                    expiry = min((stamp + 86400 for stamp, _ in self._terminal.values()), default=None)
                    timeout = max(0, expiry - monotonic()) if expiry is not None else None
                    _ = self._condition.wait_for(lambda: bool(self._pending) or self._closing, timeout)
                    if self._closing and not self._pending:
                        break
                    item = self._pending.popleft() if self._pending else None
                    self._condition.notify_all()
                now = monotonic()
                self._terminal = {key: value for key, value in self._terminal.items() if now - value[0] < 86400}
                match item:
                    case None:
                        continue
                    case Event():
                        self._publish_status()
                        item.set()
                    case ActivityEvent():
                        try:
                            self._aggregate(item)
                        except (OSError, ValueError):
                            with self._condition:
                                self._status["dropped"] = min(10**9, self._status["dropped"] + 1)
                                self._status["last_outcome"] = "aggregation_failed"
                    case unreachable:
                        assert_never(unreachable)
            if self._rooms:
                for room in self._rooms.values():
                    self._emit(room, False)
                self._checkpoint()
        finally:
            self._publish_status()
            self._done.set()
