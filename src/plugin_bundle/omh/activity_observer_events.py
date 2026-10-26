"""Closed internal metadata boundary, not a Hermes callback contract."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import hmac
import re
from typing import Final, NewType, TypeAlias
from .iso_datetime import strict_fromisoformat

OpaqueRef = NewType("OpaqueRef", str)
EventValue: TypeAlias = str | int | None
SCHEMA: Final = "omh_group_activity_event/v1"
KEYS: Final = frozenset({"schema", "profile_ref", "session_ref", "room_ref", "member_ref",
                        "turn_ref", "kind", "event_ref", "sequence", "observed_at"})
REF: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
STAMP: Final = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")


class Kind(StrEnum):
    SESSION_START = "session_start"
    MEMBER_START = "member_start"
    MEMBER_COMPLETE = "member_complete"
    TOOL_CALL = "tool_call"
    TOOL_ERROR = "tool_error"
    COMPACTION = "compaction"
    SESSION_END = "session_end"
    ACTIVITY = "activity"  # Observed metadata with no justified metric mapping.


class EventError(ValueError):
    """A bounded category, never the untrusted input."""
    def __init__(self, code: str) -> None:
        self.code: str = code
        super().__init__(code)


def opaque_ref(key: bytes, identity: str) -> OpaqueRef:
    """Derive topology at an explicit adapter boundary with a profile-local key.

    Callers own key creation/storage; this function never retains the input.
    Normalized events must already contain these digests, never platform IDs.
    """
    if not key or len(identity) > 4096:
        raise EventError("invalid_identity")
    return OpaqueRef("sha256:" + hmac.new(key, identity.encode(), hashlib.sha256).hexdigest())


def parse_ref(value: EventValue) -> OpaqueRef:
    if not isinstance(value, str) or len(value) != 71 or not REF.fullmatch(value):
        raise EventError("invalid_identity")
    return OpaqueRef(value)


def parse_stamp(value: EventValue) -> str:
    if not isinstance(value, str) or len(value) != 20 or not STAMP.fullmatch(value):
        raise EventError("invalid_time")
    try:
        _ = strict_fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise EventError("invalid_time") from None
    return value


@dataclass(frozen=True, slots=True)
class ActivityEvent:
    profile_ref: OpaqueRef
    session_ref: OpaqueRef
    room_ref: OpaqueRef
    member_ref: OpaqueRef | None
    turn_ref: OpaqueRef | None
    kind: Kind
    event_ref: OpaqueRef
    sequence: int
    observed_at: str

    @property
    def scope(self) -> tuple[OpaqueRef, OpaqueRef, OpaqueRef]:
        return self.profile_ref, self.room_ref, self.session_ref

    @property
    def identity_digest(self) -> str:
        # Collector arrival time can change on replay; producer facts cannot.
        facts = (self.scope, self.member_ref, self.turn_ref, self.kind, self.event_ref, self.sequence)
        return hashlib.sha256(repr(facts).encode()).hexdigest()


def parse_event(raw: Mapping[str, EventValue], profile: OpaqueRef) -> ActivityEvent:
    """Copy a bounded, closed shape into an immutable value before enqueue."""
    if len(raw) != len(KEYS) or set(raw) != KEYS or raw["schema"] != SCHEMA:
        raise EventError("invalid_shape")
    own_profile = parse_ref(raw["profile_ref"])
    if own_profile != profile:
        raise EventError("cross_profile")
    raw_kind = raw["kind"]
    if not isinstance(raw_kind, str) or len(raw_kind) > 32:
        raise EventError("unknown_kind")
    try:
        kind = Kind(raw_kind)
    except ValueError:
        raise EventError("unknown_kind") from None
    sequence = raw["sequence"]
    if not isinstance(sequence, int) or isinstance(sequence, bool) or not 0 <= sequence <= 10**9:
        raise EventError("invalid_sequence")
    member = parse_ref(raw["member_ref"]) if raw["member_ref"] is not None else None
    turn = parse_ref(raw["turn_ref"]) if raw["turn_ref"] is not None else None
    boundary = kind in {Kind.SESSION_START, Kind.SESSION_END}
    if boundary != (member is None and turn is None) or (not boundary and (member is None or turn is None)):
        raise EventError("invalid_identity")
    return ActivityEvent(own_profile, parse_ref(raw["session_ref"]), parse_ref(raw["room_ref"]),
                         member, turn, kind, parse_ref(raw["event_ref"]), sequence,
                         parse_stamp(raw["observed_at"]))
