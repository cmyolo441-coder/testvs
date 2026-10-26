"""Bounded room aggregates and receipt projection; no event journal."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Literal, TypeAlias, assert_never

from .activity_observer_events import ActivityEvent, EventError, Kind, OpaqueRef, parse_ref, parse_stamp

JSON: TypeAlias = str | int | float | bool | None | list["JSON"] | dict[str, "JSON"]
COUNTERS = ("tool_calls", "tool_errors", "compaction_boundaries")


@dataclass(slots=True)
class Room:
    """Mutable aggregate: one worker owns all counter and identity mutations."""
    profile: OpaqueRef
    room: OpaqueRef
    session: OpaqueRef
    started: str
    ended: str
    sequence: int = -1
    partial: bool = True
    counting: bool = True
    counts: dict[str, int] = field(default_factory=dict)
    seen: dict[OpaqueRef, str] = field(default_factory=dict)
    members: set[OpaqueRef] = field(default_factory=set)
    turns: set[tuple[OpaqueRef, OpaqueRef]] = field(default_factory=set)

    @classmethod
    def start(cls, event: ActivityEvent) -> Room:
        return cls(event.profile_ref, event.room_ref, event.session_ref,
                   event.observed_at, event.observed_at,
                   partial=event.kind != Kind.SESSION_START or event.sequence != 0)

    def is_replay(self, event: ActivityEvent) -> bool:
        return self.seen.get(event.event_ref) == event.identity_digest

    def accept(self, event: ActivityEvent) -> Literal["duplicate", "gap", "accepted"]:
        """Reject conflicts/staleness, retain dedupe keys, stop counting at capacity."""
        fingerprint = event.identity_digest
        previous = self.seen.get(event.event_ref)
        if previous is not None:
            if self.is_replay(event):
                return "duplicate"
            self.partial = True
            raise EventError("identity_conflict")
        if event.sequence <= self.sequence or event.observed_at < self.ended:
            self.partial = True
            raise EventError("stale_event")
        gap = event.sequence != self.sequence + 1
        self.partial |= gap
        if len(self.seen) >= 4096:
            self.counting = False
            self.partial = True
        else:
            self.seen[event.event_ref] = fingerprint
        if event.member_ref is not None and event.turn_ref is not None:
            pair = (event.member_ref, event.turn_ref)
            if ((event.member_ref not in self.members and len(self.members) >= 256)
                    or (pair not in self.turns and len(self.turns) >= 256)):
                self.counting = False
                self.partial = True
            else:
                self.members.add(event.member_ref)
                self.turns.add(pair)
        self.sequence, self.ended = event.sequence, event.observed_at
        metric = ""
        match event.kind:
            case Kind.TOOL_CALL:
                metric = "tool_calls"
            case Kind.TOOL_ERROR:
                metric = "tool_errors"
            case Kind.COMPACTION:
                metric = "compaction_boundaries"
            case Kind.SESSION_START:
                self.partial |= len(self.seen) != 1
            case Kind.SESSION_END | Kind.MEMBER_START | Kind.MEMBER_COMPLETE | Kind.ACTIVITY:
                pass
            case unreachable:
                assert_never(unreachable)
        if metric and self.counting:
            self.counts[metric] = self.counts.get(metric, 0) + 1
        return "gap" if gap else "accepted"

    def checkpoint(self) -> dict[str, JSON]:
        return {"profile": self.profile, "room": self.room, "session": self.session,
                "started": self.started, "ended": self.ended, "sequence": self.sequence,
                "counts": {name: value for name, value in self.counts.items()}}

    @classmethod
    def recover(cls, raw: JSON, profile: OpaqueRef) -> Room:
        """Recovery intentionally forgets exactness; prior sequence prevents replay inflation."""
        if not isinstance(raw, dict) or set(raw) != {"profile", "room", "session", "started", "ended", "sequence", "counts"}:
            raise EventError("invalid_checkpoint")
        refs: list[OpaqueRef] = []
        for name in ("profile", "room", "session"):
            value = raw[name]
            if not isinstance(value, str):
                raise EventError("invalid_checkpoint")
            refs.append(parse_ref(value))
        if refs[0] != profile:
            raise EventError("cross_profile")
        start, end, sequence, counts = raw["started"], raw["ended"], raw["sequence"], raw["counts"]
        if not isinstance(start, str) or not isinstance(end, str) or not isinstance(sequence, int) or isinstance(sequence, bool):
            raise EventError("invalid_checkpoint")
        if not 0 <= sequence <= 10**9 or parse_stamp(start) > parse_stamp(end) or not isinstance(counts, dict):
            raise EventError("invalid_checkpoint")
        values: dict[str, int] = {}
        for name, value in counts.items():
            if name not in COUNTERS or not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 10**9:
                raise EventError("invalid_checkpoint")
            values[name] = value
        return cls(refs[0], refs[1], refs[2], start, end, sequence=sequence, partial=True, counts=values)

    def receipt(self, final: bool) -> dict[str, JSON]:
        # Core receipt admission remains the sole receipt persistence authority.
        from omh.workflows.session_activity_receipts import CLAIM_BOUNDARY, METRIC_NAMES
        scope = "|".join((self.profile, self.room, self.session))
        session = "sha256:" + hashlib.sha256(scope.encode()).hexdigest()
        identity = f"{scope}|{'final' if final else 'partial'}|{self.sequence}"
        coverage = "full_session" if final and not self.partial else "from_producer_load"
        metrics: dict[str, JSON] = {}
        for name in METRIC_NAMES:
            value = self.counts.get(name)
            metrics[name] = {"value": value, "availability": "observed" if value is not None else "unavailable",
                             "measurement": ("exact" if coverage == "full_session" else "floor") if value is not None else ""}
        return {"schema_version": "session_activity_receipt/v1", "privacy": "metadata_only",
                "claim_boundary": CLAIM_BOUNDARY,
                "receipt_id": "sha256:" + hashlib.sha256(identity.encode()).hexdigest(),
                "profile_ref": self.profile, "session_ref": session,
                "producer": {"kind": "observer_plugin", "ref": "omh-group-activity", "version": "1"},
                "observed_at": self.ended,
                "observed_interval": {"started_at": self.started, "ended_at": self.ended, "coverage": coverage},
                "boundary": {"kind": "session_end" if final else "process_exit", "final": final, "sequence": self.sequence},
                "metrics": metrics, "skills": {"exposed_names": [], "activated_names": [],
                    "exposed_names_truncated": False, "activated_names_truncated": False, "name_form": "digest"},
                "model_refs": [], "evidence_refs": []}
