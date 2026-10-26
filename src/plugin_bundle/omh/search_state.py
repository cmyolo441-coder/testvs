"""Bound latest search state to host dispatch context, never model identity labels.

This process-local store retains digests, operation flags and a numeric clock floor.
It never retains raw identifiers, capabilities, queries, plans or bodies, or evicts a binding
(to make room by forgetting one would permit replay); capacity fails closed.
After restart a prior assessment cannot resume without its original binding.
A fresh plan after restart is not cross-process budget or permission evidence.
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import timezone
from .iso_datetime import strict_fromisoformat

MAX_SESSIONS = 256
MAX_REQUESTS_PER_SESSION = 32
_lock = threading.Lock()
_sessions: dict[str, dict[str, dict]] = {}
_BINDING = ("grant_id", "source_kind", "provider_id", "principal_id", "scope_id", "tool_name")


class HostSearchContextError(ValueError):
    """The host did not provide a bounded request/session binding."""


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _host_key(kwargs: dict) -> tuple[str, str]:
    # Standard Hermes registry dispatch kwargs: session_id, task_id, user_task.
    # task_id owns browser/terminal resources, not the search request. Neither
    # it nor anything in tool args/observation is an identity override here.
    session, request = kwargs.get("session_id"), kwargs.get("user_task")
    if (type(session) is not str or not session.strip() or len(session) > 512
            or type(request) is not str or not request.strip() or len(request) > 131072):
        raise HostSearchContextError("host_context_unavailable")
    return _digest(session), _digest(request)


def _binding(capability: dict) -> str:
    return _digest([capability[key] for key in _BINDING])


def _operations(capability: dict) -> int:
    if capability["availability"] != "available":
        return 0
    return sum(flag for operation, flag in (("search", 1), ("read", 2))
               if operation in capability["operations"])


def _clock(raw: str) -> int:
    # Same comparison semantics/precision as the validated producer timestamps;
    # integer UTC microseconds avoid float rounding and retain no raw token.
    parsed = strict_fromisoformat(raw.strip().replace("Z", "+00:00"))
    stamp = (parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None
             else parsed.astimezone(timezone.utc))
    return ((stamp.toordinal() * 86400 + stamp.hour * 3600 + stamp.minute * 60 + stamp.second)
            * 1000000 + stamp.microsecond)


def _merge_data(data: dict, record: dict, merge) -> dict:
    # Validate the entire supplied current context before reading/restricting it.
    # The real fusion producer still validates every candidate on the next call.
    context = data["scope"]
    merge(candidate_sets=[], scope=context)
    if _clock(context["capabilities"]["checked_at"]) < record["clock_floor"]:
        raise ValueError("invalid_time:/payload/scope/capabilities/checked_at")
    effective = []
    for index, cap in enumerate(context["capabilities"]["capabilities"]):
        old = record["capabilities"].get(_digest(cap["capability_id"]))
        if old is not None and old[0] != _binding(cap):
            raise ValueError("scope_mismatch:/payload/scope/capabilities/capabilities/" + str(index))
        operations = [operation for operation, flag in (("search", 1), ("read", 2))
                      if old is not None and old[1] & flag and operation in cap["operations"]]
        if operations:
            effective.append(dict(cap, operations=operations))
        else:
            # No new-ID/unavailable-at-plan/empty-intersection authority. Keep
            # the supplied binding for candidate validation, but deny access;
            # never emit an invalid empty operations array or candidate bodies.
            availability = "denied" if cap["availability"] == "available" else cap["availability"]
            effective.append(dict(cap, availability=availability))
    snapshot = dict(context["capabilities"], capabilities=effective)
    return dict(data, scope=dict(context, capabilities=snapshot))


def anchored_search(action: str, data: dict, kwargs: dict, functions: dict, render) -> str:
    session, request = _host_key(kwargs)
    with _lock:
        records = _sessions.get(session, {})
        record = records.get(request)
        if action == "plan":
            # Even a new caller request_id or an identical rebuilt plan cannot
            # reset a budget for the same host request. Failed builds claim no slot.
            if record is not None:
                raise ValueError("invalid_state:/payload/spec/request_id")
            if ((session not in _sessions and len(_sessions) >= MAX_SESSIONS)
                    or len(records) >= MAX_REQUESTS_PER_SESSION):
                raise HostSearchContextError("host_state_capacity")
            result = functions[action](**data)
            request_id = _digest(result["spec"]["request_id"])
            if any(row["request_id"] == request_id for row in records.values()):
                raise ValueError("invalid_state:/payload/spec/request_id")
            pending = {"request_id": request_id, "plan": _digest(result),
                       "scope": _digest(result["spec"]["scope"]), "state": None,
                       "clock_floor": _clock(result["capabilities"]["checked_at"]),
                       "capabilities": {_digest(cap["capability_id"]): (_binding(cap), _operations(cap))
                                        for cap in result["capabilities"]["capabilities"]}}
            response = render(result)
            # Only a complete, bounded, serialized response can claim a slot.
            records[request] = pending
            _sessions[session] = records
            return response
        if record is None:
            raise ValueError("invalid_state:/payload/plan")
        if action == "assess":
            if _digest(data["plan"]) != record["plan"]:
                raise ValueError("invalid_state:/payload/plan")
            previous = data["observations"]["previous_state"]
            if (None if previous is None else _digest(previous)) != record["state"]:
                raise ValueError("invalid_state:/payload/observations/previous_state")
        elif _digest(data["scope"]["scope"]) != record["scope"]:
            raise ValueError("scope_mismatch:/payload/scope")
        if action == "merge":
            data = _merge_data(data, record, functions[action])
        result = functions[action](**data)
        pending_state = _digest(result["state"]) if action == "assess" else None
        # Rendering is part of this locked transaction, including the public
        # observation envelope. A failure leaves the exact delivered anchor.
        response = render(result)
        if action == "assess":
            record["state"] = pending_state
        return response
