"""OpenAI-compatible streaming chat client (requests + manual SSE parsing)."""

from __future__ import annotations

import hashlib
import json
import queue
import random
import re
import threading
import time
from email.utils import parsedate_to_datetime
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator

# `requests` (~1.4s cold: pulls urllib3/chardet/email) is only needed when an
# actual provider HTTP call is made — never at CLI startup / import time.
# A transparent proxy stands in at module level so every existing
# `requests.*` reference below keeps working unchanged; the real import
# fires on first attribute access and then replaces the proxy in globals.
class _LazyRequests:
    def __getattr__(self, name: str):
        import requests as _real
        globals()["requests"] = _real
        return getattr(_real, name)


requests = _LazyRequests()  # type: ignore[assignment]

from . import config
from ._foundation import get_logger
from .config import Effort, Model, Provider
# TurnCancelled lives in .cancelguard (a leaf module with no import
# cycles); imported here so `from .client import TurnCancelled` keeps
# working for every existing consumer.
from .cancelguard import (TurnCancelled, run_cancellable,
                          sleep_cancellable)
# StallWatcher is a stdlib-only leaf module (no import cycles) — the
# client is its only consumer.
from .stallwatch import StallWatcher
from . import typeguards as _typeguards  # central provider-boundary guards
_log = get_logger("client")
_CONTENT_PLACEHOLDER = " "
HISTORY_MAX_MESSAGE_CHARS = 100_000
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
MAX_RETRIES = 3
RATE_LIMIT_BASE_WAIT = 2.0
RATE_LIMIT_MAX_WAIT = 60.0
# Fail fast on TCP/TLS connect; the (long) read budget stays untouched for
# slow streams. A dead provider hangs 10s here, not the full 300s timeout.
CONNECT_TIMEOUT = 10.0
# Stall watchdog for SSE streams: if no event AND no keepalive comment
# arrives for this long, the stream is dead â abort it instead of waiting
# out the full read budget. requests' read timeout is per socket read, so
# a server dribbling one byte just under the read budget could otherwise
# hold the call open forever; this closes that hole.
STREAM_STALL_TIMEOUT = 60.0

class APIError(Exception):
    def __init__(self, message: str, status: int | None = None,
                 retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class APITimeoutError(APIError):
    """A model call exceeded its time budget (connect, read, stall, or the
    total per-call deadline). Subclasses APIError so every existing
    `except APIError` handler keeps working unchanged â the timeout just
    arrives with a clearer type and message instead of hanging forever."""


# -- fast HTTP ---------------------------------------------------------------
# ONE pooled session for every request: the TCP+TLS handshake to the
# provider happens once and is reused for every model call, tool turn and
# follow-up. A turn that makes a dozen model calls saves a full handshake
# on each — this is the single biggest latency cut.
_SESSION: requests.Session | None = None
_SESSION_LOCK = threading.Lock()


def _http() -> requests.Session:
    global _SESSION
    # Double-checked locking: prewarm_connection runs on a background
    # thread, so two threads can otherwise build a pooled Session each and
    # silently drop one (leaking its pool/sockets until GC).
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                s = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=8, pool_maxsize=16, max_retries=0)
                s.mount("https://", adapter)
                s.mount("http://", adapter)
                _SESSION = s
    return _SESSION


def shared_session() -> requests.Session:
    """Public accessor for the one pooled, keep-alive Session used by every
    model call. Ad-hoc HTTP (web tools, sink posts) should use this instead
    of one-shot ``requests.get/post(...)`` calls — a one-shot call pays a
    fresh TCP+TLS handshake every time, while this session reuses its pooled
    connection (HTTP keep-alive). Thread-safe: requests sessions multiplex
    a thread-safe urllib3 pool, and construction is double-checked-locked."""
    return _http()


def _timeouts(timeout: float) -> tuple[float, float]:
    """Split (connect, read) timeouts. Connect fails fast; the read budget
    stays long because a healthy stream can legitimately take minutes."""
    return (min(CONNECT_TIMEOUT, timeout), timeout)


def _sleep_capped_cancellable(delay: float, deadline: float,
                              should_cancel: Callable[[], bool] | None) -> None:
    """Retry backoff that honours Esc/Ctrl+C AND never sleeps past the
    call's total deadline — the budget belongs to the request, not to the
    wait between attempts."""
    remaining = deadline - time.monotonic()
    if remaining > 0:
        sleep_cancellable(min(delay, remaining), should_cancel)


def _backoff(attempt: int) -> float:
    """Retry delay: fast first retry, then escalate, with jitter so a fleet
    of agents doesn't thundering-herd a recovering provider."""
    base = 0.5 if attempt == 0 else 2.0 * attempt
    return base * random.uniform(0.8, 1.2)


def _parse_retry_after(value: Any) -> float | None:
    """Parse Retry-After as seconds or an HTTP date; invalid values vanish."""
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        return max(0.0, min(float(raw), 3600.0))
    except ValueError:
        try:
            when = parsedate_to_datetime(raw)
            if when.tzinfo is None:
                return None
            return max(0.0, min(when.timestamp() - time.time(), 3600.0))
        except (TypeError, ValueError, OverflowError):
            return None


def _rate_limit_delay(error: APIError, attempt: int) -> float:
    """Prefer the server's wait hint; otherwise use capped exponential backoff."""
    if error.retry_after is not None:
        return max(0.0, min(float(error.retry_after), 3600.0))
    base = min(RATE_LIMIT_MAX_WAIT,
               RATE_LIMIT_BASE_WAIT * (2 ** min(attempt, 8)))
    return min(RATE_LIMIT_MAX_WAIT,
               base + random.uniform(0.0, min(2.0, base * 0.2)))


def _is_rate_limited(error: APIError) -> bool:
    if error.status == 429:
        return True
    message = str(error).lower().replace("_", " ").replace("-", " ")
    return ("rate limit" in message or "too many requests" in message
            or "throttl" in message)


# Errors that can NEVER heal by retrying: an unknown model id, bad auth, a
# malformed request. Retrying those burned ~14s of backoff (2+4+8) before
# the UI showed the real error — indistinguishable from "the app is slow".
_PERMANENT_STATUSES = {400, 401, 403, 404, 405, 422}


def _is_permanent_error(error: APIError) -> bool:
    """True when retrying is pointless: the request itself is wrong.

    Gateways answer a retired/dead model id with 503 ``model_not_found``
    or "No available channel for model ...", which is permanent even
    though 503 is normally transient."""
    if error.status in _PERMANENT_STATUSES:
        return True
    message = str(error).lower()
    return ("model_not_found" in message
            or "no available channel" in message
            or "invalid api key" in message
            or "unauthorized" in message)


def _wait_for_rate_limit(error: APIError, attempt: int,
                         should_cancel: Callable[[], bool] | None,
                         on_status: Callable[[str], None] | None = None) -> None:
    delay = _rate_limit_delay(error, attempt)
    if on_status:
        try:
            on_status(f"Rate limit reached (429); retrying in "
                      f"{delay:.0f}s (Ctrl+C to cancel)")
        except Exception:
            pass
    sleep_cancellable(delay, should_cancel)


def _base_headers(provider: Provider, stream: bool) -> dict[str, str]:
    """The per-request headers. Built from one place so every call path
    (streaming, blocking, overflow-retry) sends the identical set."""
    headers = {"Content-Type": "application/json"}
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"
    if stream:
        headers["Accept"] = "text/event-stream"
    return headers


# -- prewarm: skip redundant warmups -----------------------------------------
# The warmup's value is the pooled TCP+TLS connection, not the /models body
# (which nobody reads). Warming the same provider twice within the TTL is a
# wasted round-trip, so remember per-base_url warm times.
_warm_at: dict[str, float] = {}
_PREWARM_TTL = 120.0


def prewarm_connection(provider) -> None:
    """SPEED: establish the TCP+TLS connection to the provider at startup
    (or model switch), so the first model call doesn't pay the handshake
    cost. Runs in a background thread — never blocks the UI. The connection
    sits in the pool ready for the first real request. Warmups for the same
    provider within the TTL are skipped (no redundant API calls)."""
    if not provider.api_key:
        return  # do not spend requests on /models warmup without a key
    def _warm():
        key = provider.base_url
        now = time.monotonic()
        if now - _warm_at.get(key, 0.0) < _PREWARM_TTL:
            return
        _warm_at[key] = now
        try:
            url = provider.base_url.rstrip("/") + "/models"
            warm_headers = ({"Authorization": f"Bearer {provider.api_key}"}
                            if provider.api_key else {})
            _http().get(url, timeout=_timeouts(5), headers=warm_headers)
        except Exception:
            pass  # prewarming is best-effort, never a crash path

    threading.Thread(target=_warm, daemon=True, name="prewarm").start()


# NOTE: TurnCancelled was defined here; it now lives in .cancelguard
# (imported above) — a single class object, so `except TurnCancelled`
# catches it no matter which module raised it.


@dataclass
class ToolCallDelta:
    id: str = ""
    # Stream-hot path: tool arguments can be megabytes (e.g. a whole file
    # in write_file args) arriving in thousands of chunks. Appending to a
    # list and joining once is O(n); `+=` on a str is O(n^2) and made the
    # stream loop visibly slower as the arguments grew.
    name_parts: list = field(default_factory=list)
    arguments_parts: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return "".join(self.name_parts)

    @property
    def arguments(self) -> str:
        return "".join(self.arguments_parts)


@dataclass
class StreamResult:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict | None = None
    model: str = ""

    def __post_init__(self) -> None:
        # Providers and cassette replays can hand us wrong-typed values
        # (usage as a string, tool_calls as ["str"], content as an int,
        # model as a dict). Coerce at construction so the declared types
        # always hold — one bad field must never become an
        # AttributeError/TypeError downstream (e.g. turn.usage.get(...)
        # in tui.py, tc["function"] in agent.py). Never raises.
        self.normalize()

    def normalize(self) -> "StreamResult":
        """Coerce every field to its declared type, in place.

        Never raises. Tool-call entries go through the single
        normalize_tool_call() canonicalizer so a malformed entry is
        dropped or repaired, never left as a crash vector.
        """
        # STRICT str-or-empty (not ensure_str's str()-ification):
        # non-string junk must become "", never "123"/"{'m': 1}".
        self.content = _strict_str(self.content)
        self.reasoning = _strict_str(self.reasoning)
        self.model = _strict_str(self.model)
        self.finish_reason = (None if self.finish_reason is None
                              else _strict_str(self.finish_reason))
        # usage: dict-or-None only. A non-empty string is truthy and
        # would sail past `if usage` guards straight into usage.get().
        self.usage = self.usage if isinstance(self.usage, dict) else None
        normed: list[dict] = []
        for tc in _typeguards.ensure_list(self.tool_calls):
            canon = normalize_tool_call(tc)
            if canon is not None:
                normed.append(canon)
        self.tool_calls = normed
        return self


_tc_fallback_seq = [0]


def normalize_tool_call(tc: Any) -> dict | None:
    """Canonicalize ANY value into a valid OpenAI-style tool_call dict.

    The tool_calls pipeline (client parsing -> sequential tool execution ->
    agent.py history) has crashed repeatedly with
    `AttributeError: 'str' object has no attribute 'get'` because
    providers (and old history/cassettes) can hand us strings, None,
    ints, partial dicts, or dicts whose ``function`` is a JSON string
    instead of an object. Every creation site and every iteration site
    must run values through this function and skip None results.

    Returns ``{"id": str, "type": "function",
    "function": {"name": str, "arguments": str}}`` or None when the
    value is unrecoverable (not a dict at all, or no usable function).
    ``arguments`` is always returned as a JSON string — a dict/list
    value is json.dumps-ed so downstream ``fn["arguments"]`` handling
    sees one canonical shape.
    """
    if not isinstance(tc, dict):
        return None
    raw_fn = tc.get("function")
    if isinstance(raw_fn, str):
        # Some providers serialise the whole function object as a JSON
        # string: {"function": "{\\"name\\": \\"read\\", ...}"}. Parse it;
        # if it is not JSON, treat the string itself as the tool name
        # (e.g. {"function": "read"}) rather than dropping the call.
        try:
            parsed = json.loads(raw_fn)
            raw_fn = parsed if isinstance(parsed, dict) else {
                "name": raw_fn, "arguments": "{}"}
        except (ValueError, TypeError):
            raw_fn = {"name": raw_fn, "arguments": "{}"}
    if not isinstance(raw_fn, dict):
        return None
    name = raw_fn.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = raw_fn.get("arguments", "")
    if isinstance(args, str):
        arguments = args
    else:
        try:
            arguments = json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            arguments = str(args)
    tc_id = tc.get("id")
    if not isinstance(tc_id, str) or not tc_id:
        _tc_fallback_seq[0] += 1
        tc_id = f"call_auto_{_tc_fallback_seq[0]}"
    return {"id": tc_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments}}


def _coerce_token_count(value: Any) -> int:
    """Coerce one usage value to a non-negative int. Never raises.

    Accepts ints (incl. bools), floats (truncated toward zero), and
    numeric strings ("42", "3.7", " 12 "). Anything else — None,
    garbage strings, inf/nan, containers — yields 0. Note: the central
    typeguards.ensure_int deliberately does NOT coerce strings/floats,
    so usage counting carries its own coercion here.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        try:
            return max(0, int(value))
        except (OverflowError, ValueError):
            return 0
    if isinstance(value, str):
        s = value.strip()
        if s:
            try:
                return max(0, int(s))
            except (ValueError, OverflowError):
                try:
                    return max(0, int(float(s)))
                except (ValueError, OverflowError):
                    pass
    return 0


def safe_token_counts(usage: Any) -> tuple[int, int]:
    """Safely extract (prompt_tokens, completion_tokens) from ANY usage shape.

    Providers are inconsistent: usage may be a dict, None, a raw string,
    a number, or a malformed dict (string/None values, unexpected keys).
    This function never raises and always returns non-negative ints.
    Also honours the Anthropic-style input_tokens/output_tokens aliases.
    """
    usage = _typeguards.ensure_dict(usage)
    if not usage:
        return (0, 0)
    prompt = None
    for key in ("prompt_tokens", "input_tokens"):
        if key in usage:
            prompt = usage[key]
            break
    completion = None
    for key in ("completion_tokens", "output_tokens"):
        if key in usage:
            completion = usage[key]
            break
    return (_coerce_token_count(prompt), _coerce_token_count(completion))


def usage_tokens(usage: Any) -> tuple[int, int]:
    """Historical name for safe_token_counts (kept: tests and older
    call sites import it)."""
    return safe_token_counts(usage)


def assistant_message(content: str | None, tool_calls: list[dict],
                      reasoning: str = "") -> dict:
    """Build an assistant message that is valid for thinking-aware
    backends.

    Reasoning-capable backends can validate the *history*: any
    assistant turn that carried tool_calls must also carry
    reasoning_content when the model was invoked with thinking enabled
    (reasoning_effort=low). If the history was built while reasoning
    was stripped, the next request fails with
    'messages[N].reasoning_content is required for thinking tool-call
    history'.

    This helper guarantees the invariant: tool-call turns always carry
    reasoning_content (real reasoning when available, else an empty
    string), so history never becomes invalid regardless of the
    provider's suppression mode.
    """
    msg: dict[str, Any] = {"role": "assistant"}
    # OpenAI spec: content is nullable when tool_calls are present. Preserve
    # any string the backend gave us — empty/whitespace text from a reasoning
    # model that spent the budget on thinking is a legitimate reply, not the
    # same as "no content at all". Using truthiness here was turning real
    # "" / " " replies into None and corrupting conversation history.
    # HARDEN (malformed-provider audit): content / tool_calls / reasoning
    # can arrive as dicts, lists, or bare strings from sloppy providers —
    # storing them raw lets a downstream tc.get() / str op raise
    # AttributeError: 'str' object has no attribute ... . Coerce every
    # input; this function never raises.
    if content is None:
        msg["content"] = None
    elif isinstance(content, str):
        msg["content"] = content
    elif isinstance(content, (dict, list)):
        try:
            msg["content"] = json.dumps(content, ensure_ascii=False)
        except (TypeError, ValueError):
            msg["content"] = str(content)
    else:
        msg["content"] = _typeguards.ensure_str(content)
    # Only real dict tool calls survive: strings/scalars are dropped, a
    # bare dict becomes a one-element list, [1, 2, "x"] yields [].
    calls: list[dict] = []
    _tcs = [tool_calls] if isinstance(tool_calls, dict) else tool_calls
    if isinstance(_tcs, (list, tuple)):
        for tc in _tcs:
            # One canonical shape: full normalization, not just a dict
            # check — {"function": "name-string"} would sail past
            # isinstance() and crash fn.get("name") downstream with
            # AttributeError: 'str' object has no attribute 'get'.
            norm = normalize_tool_call(tc)
            if norm is not None:
                calls.append(norm)
    if calls:
        msg["tool_calls"] = calls
        # A few OpenAI-compatible gateways reject null/empty content on an
        # assistant tool-call turn even though the protocol permits it.
        if not isinstance(msg.get("content"), str) or not msg["content"]:
            msg["content"] = _CONTENT_PLACEHOLDER
    # Always carry reasoning_content when reasoning exists OR when the
    # turn carried tool_calls (history validation requires it).  Set
    # both keys for maximum provider compatibility.
    _rs = reasoning
    if _rs is not None and not isinstance(_rs, str):
        if isinstance(_rs, (dict, list)):
            try:
                _rs = json.dumps(_rs, ensure_ascii=False)
            except (TypeError, ValueError):
                _rs = str(_rs)
        else:
            _rs = _typeguards.ensure_str(_rs)
    if _rs:
        msg["reasoning_content"] = _rs
        # some providers also accept "reasoning"
        msg["reasoning"] = _rs
    elif calls:
        msg["reasoning_content"] = ""
    return msg


@dataclass(frozen=True)
class HistoryDiagnostic:
    code: str
    index: int
    severity: str
    detail: str


def validate_history(history: Any) -> list[HistoryDiagnostic]:
    """Describe malformed chat history without mutating it or raising."""
    if not isinstance(history, list):
        return [HistoryDiagnostic("history_not_a_list", -1, "error",
                                  "history must be a list")]
    out: list[HistoryDiagnostic] = []
    known_roles = {"system", "user", "assistant", "tool"}
    calls: dict[str, int] = {}
    call_order: list[tuple[int, str]] = []
    tool_rows: list[tuple[int, str | None]] = []
    for index, message in enumerate(history):
        if not isinstance(message, dict):
            out.append(HistoryDiagnostic("non_dict_message", index,
                                         "error", "message is not an object"))
            continue
        role = message.get("role")
        if not isinstance(role, str) or role not in known_roles:
            out.append(HistoryDiagnostic("unknown_role", index, "error",
                                         f"unknown role: {role!r}"))
        try:
            size = len(json.dumps(message, ensure_ascii=False, default=str))
        except Exception:
            size = HISTORY_MAX_MESSAGE_CHARS + 1
        if size > HISTORY_MAX_MESSAGE_CHARS:
            out.append(HistoryDiagnostic(
                "oversize_message", index, "warning",
                f"serialized message is {size} characters"))
        if role == "assistant":
            raw_calls = message.get("tool_calls")
            if raw_calls is None:
                raw_calls = []
            elif not isinstance(raw_calls, (list, tuple)):
                raw_calls = [raw_calls]
            if raw_calls:
                content = message.get("content")
                if not isinstance(content, str) or not content:
                    out.append(HistoryDiagnostic(
                        "empty_content_with_tool_calls", index, "warning",
                        "assistant tool-call message has empty content"))
            for call in raw_calls:
                if not isinstance(call, dict) or not isinstance(call.get("id"), str) \
                        or not call.get("id"):
                    out.append(HistoryDiagnostic(
                        "tool_call_missing_id", index, "error",
                        "assistant tool call has no non-empty string id"))
                    continue
                call_id = call["id"]
                if call_id in calls:
                    out.append(HistoryDiagnostic(
                        "duplicate_tool_call_id", index, "error",
                        f"tool call id {call_id!r} was already issued"))
                else:
                    calls[call_id] = index
                    call_order.append((index, call_id))
        elif role == "tool":
            tid = message.get("tool_call_id")
            if not isinstance(tid, str) or not tid:
                out.append(HistoryDiagnostic(
                    "tool_response_missing_id", index, "error",
                    "tool response has no non-empty string tool_call_id"))
                tool_rows.append((index, None))
            else:
                tool_rows.append((index, tid))
    seen_responses: set[str] = set()
    for index, tid in tool_rows:
        if tid is None:
            continue
        if tid in seen_responses:
            out.append(HistoryDiagnostic(
                "duplicate_tool_response_id", index, "error",
                f"tool call id {tid!r} has more than one response"))
            continue
        seen_responses.add(tid)
        owner = calls.get(tid)
        if owner is None:
            # Distinguish an out-of-order response from an id never issued.
            future = any(call_id == tid and call_index > index
                         for call_index, call_id in call_order)
            code = "tool_response_out_of_order" if future else "orphan_tool_response"
            out.append(HistoryDiagnostic(
                code, index, "error",
                f"tool response {tid!r} has no preceding assistant call"))
        elif owner >= index:
            out.append(HistoryDiagnostic(
                "tool_response_out_of_order", index, "error",
                f"tool response {tid!r} precedes its assistant call"))
    for index, call_id in call_order:
        if call_id not in seen_responses:
            out.append(HistoryDiagnostic(
                "orphan_tool_call", index, "error",
                f"tool call {call_id!r} has no response"))
    return out


def _repair_role_sequence(messages: list, ) -> list[str]:
    """Repair role ordering and tool-call adjacency in place.

    Invalid messages are dropped, adjacent assistant messages are merged,
    and valid tool replies are moved directly after their issuing turn.
    Tool replies without an issuing call are removed.
    """
    repairs: list[str] = []
    if not isinstance(messages, list):
        return ["history is not a list"]
    valid: list[dict] = []
    for message in messages:
        if (not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant", "tool"}):
            repairs.append("dropped invalid role or non-dict message")
            continue
        valid.append(message)
    merged: list[dict] = []
    for message in valid:
        if message.get("role") == "assistant" and merged \
                and merged[-1].get("role") == "assistant":
            previous = merged[-1]
            left, right = previous.get("content"), message.get("content")
            parts = [str(x) for x in (left, right) if x not in (None, "")]
            previous["content"] = "\n".join(parts) if parts else None
            calls = previous.get("tool_calls") or []
            other = message.get("tool_calls") or []
            if isinstance(calls, dict):
                calls = [calls]
            if isinstance(other, dict):
                other = [other]
            previous["tool_calls"] = list(calls) + list(other)
            for key in ("reasoning_content", "reasoning"):
                if message.get(key) and not previous.get(key):
                    previous[key] = message[key]
            repairs.append("merged consecutive assistant messages")
        else:
            merged.append(message)

    owners: dict[str, int] = {}
    for index, message in enumerate(merged):
        if message.get("role") != "assistant":
            continue
        raw_calls = message.get("tool_calls") or []
        if isinstance(raw_calls, dict):
            raw_calls = [raw_calls]
        for call in raw_calls if isinstance(raw_calls, (list, tuple)) else []:
            if isinstance(call, dict) and isinstance(call.get("id"), str) and call["id"]:
                owners.setdefault(call["id"], index)
    tools_by_owner: dict[int, list[dict]] = {}
    for message_index, message in enumerate(merged):
        if message.get("role") != "tool":
            continue
        tid = message.get("tool_call_id")
        owner = owners.get(tid) if isinstance(tid, str) else None
        if owner is None:
            if not any(m.get("role") in {"system", "user"}
                       for m in merged[:message_index]):
                repairs.append("dropped leading tool response")
            else:
                repairs.append("dropped orphan tool response")
        else:
            tools_by_owner.setdefault(owner, []).append(message)
    repaired: list[dict] = []
    for index, message in enumerate(merged):
        if message.get("role") == "tool":
            continue
        repaired.append(message)
        if message.get("role") == "assistant":
            tool_rows = tools_by_owner.get(index, [])
            if tool_rows:
                repaired.extend(tool_rows)
    if repaired != merged:
        # Reordering is a repair, even when all the same dict objects survive.
        before_ids = [id(m) for m in merged]
        after_ids = [id(m) for m in repaired]
        if before_ids != after_ids and not any("reordered" in r for r in repairs):
            repairs.append("reordered tool responses after issuing assistant")
    if not any(m.get("role") == "user" for m in repaired):
        insert_at = 1 if repaired and repaired[0].get("role") == "system" else 0
        repaired.insert(insert_at, {"role": "user", "content": "(continue)"})
        repairs.append("inserted placeholder user message")
    if repaired and repaired[0].get("role") not in {"system", "user"}:
        repaired.insert(0, {"role": "user", "content": "(continue)"})
        repairs.append("inserted placeholder user message")
    messages[:] = repaired
    return repairs


def _sanitize_messages(messages: list[dict]) -> bool:
    """Canonicalize tool history and preserve only well-paired calls.

    The operation is in-place, defensive, and returns whether any repair
    was made. Orphan/duplicate calls and responses are removed before the
    payload reaches a provider.
    """
    if not isinstance(messages, list):
        return False
    fixed = False
    issued: dict[str, tuple[int, dict]] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        raw = message.get("tool_calls")
        if raw is None:
            continue
        items = raw if isinstance(raw, (list, tuple)) else [raw]
        normalized: list[dict] = []
        for call in items:
            if not isinstance(call, dict):
                fixed = True
                continue
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id:
                fixed = True
                continue
            canon = normalize_tool_call(call)
            if canon is None or call_id in issued:
                fixed = True
                continue
            normalized.append(canon)
            issued[call_id] = (index, canon)
        if len(normalized) != len(items):
            fixed = True
        if normalized:
            if message.get("tool_calls") != normalized:
                message["tool_calls"] = normalized
                fixed = True
        elif "tool_calls" in message:
            message.pop("tool_calls", None)
            fixed = True
            # Remove an assistant turn made empty by dropping its only call.
            if not message.get("content"):
                message["_sanitize_empty_tool_turn"] = True

    seen_responses: set[str] = set()
    kept_responses: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        tid = message.get("tool_call_id")
        owner = issued.get(tid) if isinstance(tid, str) else None
        if owner is None or owner[0] >= index or tid in seen_responses:
            message["_sanitize_drop_tool"] = True
            fixed = True
            continue
        seen_responses.add(tid)
        kept_responses.append(tid)
    missing = set(issued) - seen_responses
    if missing:
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            raw = message.get("tool_calls") or []
            if isinstance(raw, dict):
                raw = [raw]
            kept = [call for call in raw if isinstance(call, dict)
                    and call.get("id") not in missing]
            if len(kept) != len(raw):
                fixed = True
                if kept:
                    message["tool_calls"] = kept
                else:
                    message.pop("tool_calls", None)
                    if not message.get("content"):
                        message["_sanitize_empty_tool_turn"] = True
    if fixed:
        cleaned = []
        for message in messages:
            if not isinstance(message, dict):
                cleaned.append(message)
                continue
            if message.pop("_sanitize_drop_tool", False):
                continue
            if message.pop("_sanitize_empty_tool_turn", False):
                continue
            cleaned.append(message)
        messages[:] = cleaned
    # Ensure all surviving tool-call turns satisfy both provider quirks.
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant" \
                or not message.get("tool_calls"):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            message["content"] = _coerce_text(content) or _CONTENT_PLACEHOLDER
            fixed = True
        elif not content:
            message["content"] = _CONTENT_PLACEHOLDER
            fixed = True
        if message.get("reasoning_content") is None and message.get("reasoning") is None:
            message["reasoning_content"] = ""
            fixed = True
    return fixed


def _is_reasoning_content_error(err: Exception) -> bool:
    """True if an APIError is the reasoning_content validation."""
    msg = str(err).lower()
    return "reasoning_content" in msg and "required" in msg


# -- token-limit pre-check (worker 17/20) -------------------------------------
# ISSUE: requests that exceed the model's context window were sent as-is
# and only failed ~44s later with invalid_request_error. This pre-check
# runs inside build_payload — the single funnel for both send paths —
# and prunes history FIRST when input + requested max_tokens cannot fit
# the per-model window (config.MODEL_CONTEXT_WINDOWS). A request we can
# prove won't fit is never sent.
def _prune_messages_view(messages: list[dict], window: int) -> list:
    """agent.prune_messages via lazy import (agent.py is heavy and must
    not be imported at client import time). Falls back to a minimal
    local pruner that keeps the system prompt plus the tail."""
    try:
        from .agent import prune_messages
        return prune_messages(messages, window=window)
    except Exception:
        out = []
        rest = list(messages)
        if rest and isinstance(rest[0], dict) \
                and rest[0].get("role") == "system":
            out.append(rest[0])
            rest = rest[1:]
        out.extend(rest[-window:] if window > 0 else rest)
        return out


def _irreducible_core(messages: list[dict]) -> list:
    """The smallest history still worth sending: system prompt (if any),
    the first user message (task framing) and the latest user message."""
    core: list = []
    rest = [m for m in messages if isinstance(m, dict)]
    if rest and rest[0].get("role") == "system":
        core.append(rest[0])
        rest = rest[1:]
    users = [m for m in rest if m.get("role") == "user"]
    if users:
        core.append(users[0])
        if len(users) > 1:
            core.append(users[-1])
    elif rest:
        core.append(rest[-1])
    return core


def _ensure_window_fit(model: Model, messages: list[dict],
                       tools: list[dict] | None, effort: Effort) -> int:
    """Estimate input tokens; prune history FIRST when the request cannot
    fit the model's context window. Returns the estimated input tokens
    for the (possibly pruned) history.

    Pruning is in place (``messages[:] = ...``) so the caller's retry /
    overflow logic sees the same history that was actually sent. Raises
    APIError only when even the irreducible core cannot fit — the message
    carries "context window" so is_context_overflow routes it through the
    existing shrink-and-retry path instead of a doomed send.
    """
    window = effective_window(model)
    requested = effort.max_tokens or 0
    # When max_tokens is unset the backend chooses the completion length;
    # still reserve room for a minimal completion so a huge input alone
    # can never go out unflagged.
    completion = requested if requested else _MIN_COMPLETION_TOKENS

    def _fits(n: int) -> bool:
        margin = _context_margin(window) + n // 32
        return n + margin + completion <= window

    n = _input_tokens(model, messages, tools)
    if _fits(n):
        return n
    before_msgs, before_n = len(messages), n
    _log.warning(
        "token pre-check: ~%d input tokens + %d max_tokens would exceed "
        "the %s context window (%d) — pruning history first",
        n, completion, model.label, window)
    # Progressively tighter sliding windows (reuses agent.prune_messages:
    # keeps the system prompt + first user message + the tail).
    for w in (20, 10, 4):
        cand = _prune_messages_view(messages, window=w)
        n = _input_tokens(model, cand, tools)
        if _fits(n):
            messages[:] = cand
            _log.warning(
                "token pre-check: pruned history %d -> %d messages "
                "(~%d -> ~%d input tokens); request now fits the window",
                before_msgs, len(messages), before_n, n)
            return n
    # Last resort: the irreducible core.
    core = _irreducible_core(messages)
    n = _input_tokens(model, core, tools)
    if core and _fits(n):
        messages[:] = core
        _log.warning(
            "token pre-check: pruned history to the irreducible core "
            "(%d messages, ~%d input tokens); request now fits the window",
            len(messages), n)
        return n
    raise APIError(
        f"conversation is too large for {model.label} "
        f"(~{n:,} input tokens vs {window:,} context window) — start a "
        f"new session (/new), rewind (/rewind), or switch to a "
        f"larger-context model (Ctrl+T)")


def build_payload(model: Model, effort: Effort, messages: list[dict],
                  tools: list[dict] | None, stream: bool = True) -> dict:
    # Diagnose the original history before repairing it so malformed
    # sessions produce actionable logs rather than silently losing context.
    diagnostics = validate_history(messages)
    for diagnostic in diagnostics:
        _log.warning("history validation %s at message %s: %s",
                     diagnostic.code, diagnostic.index, diagnostic.detail)
        if diagnostic.index >= 0 and diagnostic.index < len(messages):
            _log.debug("history validation offender: %r",
                       messages[diagnostic.index])
    _repair_role_sequence(messages)
    # Sanitize history before it hits the wire — fixes stale sessions that
    # were built before reasoning_content was preserved.
    _sanitize_messages(messages)
    # TOKEN-LIMIT PRE-CHECK (worker 17/20): estimate input tokens BEFORE
    # anything goes on the wire; if input + requested max_tokens cannot
    # fit the model's context window, prune history FIRST (and log it)
    # instead of sending a request we can prove the backend will reject
    # with invalid_request_error after ~44s.
    input_tokens = _ensure_window_fit(model, messages, tools, effort)
    payload: dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "stream": stream,
        "temperature": effort.temperature,
    }
    if effort.max_tokens:
        # The request must fit in the window: input + max_tokens <= window,
        # otherwise the backend rejects it wholesale.
        # The input estimate is computed ONCE and shared by the clamp and
        # the hard invariant below — previously the whole conversation was
        # JSON-serialized up to 4 extra times per request.
        windowed = _window_max_tokens(model, effort, messages, tools,
                                      input_tokens)
        payload["max_tokens"] = _clamp_max_tokens(model.provider, windowed)
        # HARD INVARIANT — the last mechanical gate before the wire. No
        # request may leave with estimated input + max_tokens over the
        # window. If any earlier layer drifted, clamp again here rather
        # than send a doomed request.
        window = effective_window(model)
        if input_tokens + payload["max_tokens"] > window:
            payload["max_tokens"] = max(
                _MIN_COMPLETION_TOKENS, window - input_tokens - CONTEXT_MARGIN)
    if tools and model.supports_tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    # REAL THINKING: request actual chain-of-thought from the model when
    # the selected effort asks for it (reasoning_effort is set on medium+
    # efforts). "none" stays the default so cheap/fast efforts never pay
    # the hidden-CoT latency. The streamed reasoning reaches the TUI via
    # client.py's on_reasoning callback (reasoning_content field).
    if model.supports_reasoning:
        payload["reasoning_effort"] = effort.reasoning_effort or "none"
    return payload


# -- fast token-count cache -------------------------------------------------
# estimate_tokens() re-serialized the whole payload with json.dumps on EVERY
# call (~40 ms per MB of history). That is fine once per request, but it is
# called repeatedly on identical content: _input_tokens() runs it twice per
# send, ctxmeter + statuscmd + the TUI border re-read the same message list,
# and the agent compaction loops call it in tight while loops.
#
# Two tiers:
#   Tier 1 (identity): the same Python object, verified unmutated via a
#   cheap O(#messages) structural fingerprint, returns its cached count
#   with ZERO serialization and ZERO hashing (~microseconds). The
#   fingerprint is re-computed on every lookup and compared to the stored
#   one, so even id() reuse after GC is safe: a different object either
#   mismatches (recount) or matches (same serialized length -> same
#   count). Any real edit the agent performs (append/pop/in-place
#   truncate) changes the fingerprint and recounts.
#   Tier 2 (digest): blake2b(serialized) -> count for everything else —
#   fresh but identical objects, non-message shapes. The correctness net.
# Both tiers are keyed on the calibration epoch: when the backend teaches
# us a new chars/token ratio, stale counts are never served.
#
# Approximation note: tier 1 treats two messages as identical when their
# fingerprint matches (lengths + head/tail of every string field). A
# pathological in-place edit that keeps every field length AND head/tail
# while changing JSON-escapable characters mid-string could shift the true
# serialized length by a few chars (< 1 token); the 8K+ context margins
# dwarf that, and any length-changing edit recounts exactly.
_TOKEN_CACHE_MAX = 1024          # tier-2 digest entries
_ID_CACHE_MAX = 128             # tier-1 identity entries
_token_cache: "OrderedDict[tuple, int]" = OrderedDict()
_id_cache: "OrderedDict[int, tuple]" = OrderedDict()
_token_cache_lock = threading.Lock()


def _str_sig(s: Any) -> tuple | None:
    """O(1) signature of a string field: (length, head, tail)."""
    if s is None:
        return (0, "", "")
    if not isinstance(s, str):
        return None
    return (len(s), s[:16], s[-16:])


def _message_fingerprint(obj: Any) -> tuple | None:
    """Cheap structural fingerprint for a list of message dicts.

    O(#messages), O(1) per field — no serialization, no content hashing.
    Covers every string field of every message (plus tool-call payloads),
    so any edit that could change the serialized length is detected.
    Returns None for anything that is not a plain list of dicts (those
    use the digest tier only).
    """
    if not isinstance(obj, list):
        return None
    fp: list = [len(obj)]
    for m in obj:
        if not isinstance(m, dict):
            return None
        item: list = []
        for k, v in m.items():
            if k == "tool_calls" and isinstance(v, list):
                tc: list = []
                for t in v:
                    if not isinstance(t, dict):
                        return None
                    parts: list = []
                    for tk, tv in t.items():
                        if tk == "function" and isinstance(tv, dict):
                            fsig = tuple((fk, _str_sig(fv))
                                         for fk, fv in tv.items())
                            if any(s is None for _, s in fsig):
                                return None
                            parts.append((tk, fsig))
                        else:
                            s = _str_sig(tv)
                            if s is None:
                                return None
                            parts.append((tk, s))
                    tc.append(tuple(parts))
                item.append((k, tuple(tc)))
            else:
                s = _str_sig(v)
                if s is None:
                    return None
                item.append((k, s))
        fp.append(tuple(item))
    return tuple(fp)


# -- raw token estimation (worker 17/20) --------------------------------------
# The old estimator was a flat chars/3.2 division. The raw estimate is now
# segment-aware instead: tiktoken's cl100k_base BPE when the package
# happens to be installed (a good generic stand-in for the unknown
# tokenizers behind these gateways), else a BPE-aware heuristic that beats
# chars/4 by counting ASCII words (~1.32 tok/word), CJK chars (~1.0
# tok/char — common CJK chars are often a single token; rare ones cost
# more and the learned correction absorbs the difference) and leftover
# punctuation/whitespace/JSON framing (~1 tok per 2.5 chars) separately. Either raw estimate is then pulled toward the
# backend's REAL counts by the per-model learned correction multiplier
# below, so the very first request of a session is already sane and every
# later one tracks reality.
#
# tiktoken is OPTIONAL and imported lazily (it is heavy; module import
# must stay fast). Everything here never raises.
_tiktoken_enc = None
_tiktoken_unavailable = False


def _tiktoken_count(text: str) -> int | None:
    """Token count via tiktoken cl100k_base, or None when tiktoken is not
    installed or fails. Never raises."""
    global _tiktoken_enc, _tiktoken_unavailable
    if _tiktoken_unavailable:
        return None
    try:
        if _tiktoken_enc is None:
            with _token_cache_lock:
                if _tiktoken_enc is None and not _tiktoken_unavailable:
                    import tiktoken
                    _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        return len(_tiktoken_enc.encode(text))
    except Exception:
        _tiktoken_unavailable = True
        return None


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
            or 0x3040 <= o <= 0x30FF or 0xF900 <= o <= 0xFAFF
            or 0xAC00 <= o <= 0xD7AF or 0xFF00 <= o <= 0xFFEF)


def _word_tokens(w: str) -> float:
    """Token estimate for one whitespace-free run. Short runs are usually
    a single BPE token (~1.32 with the trailing space); hash/base64/uuid-
    like runs (mixed digits+uppercase) tokenize ~2 chars/token; other
    long runs (URLs, paths) ~3 chars/token past the first 8."""
    alnum = 0
    mixed = 0
    for ch in w:
        if ch.isalnum():
            alnum += 1
            if ch.isdigit() or "A" <= ch <= "Z":
                mixed += 1
    if alnum >= 4 and mixed / alnum >= 0.4:
        return max(2.0, len(w) / 2.0)
    if len(w) <= 8:
        return 1.32
    return 1.32 + (len(w) - 8) / 3.0


def _heuristic_token_count(text: str) -> int:
    """BPE-aware fallback when tiktoken is unavailable. Deliberately a
    touch generous: for the pre-send fit check it is safer to over-count
    (prune early) than to under-count (send a doomed request and wait ~44s
    for invalid_request_error)."""
    if not text:
        return 0
    words = _WORD_RE.findall(text)
    word_chars = 0
    word_tokens = 0.0
    for w in words:
        word_chars += len(w)
        word_tokens += _word_tokens(w)
    cjk = 0
    spaces = 0
    for ch in text:
        if _is_cjk(ch):
            cjk += 1
        elif ch.isspace():
            spaces += 1
    # Punctuation / JSON framing (~1 tok per 2.5 chars); whitespace mostly
    # merges into neighbouring BPE tokens, so it counts far less.
    punct = len(text) - word_chars - cjk - spaces
    return max(1, int(word_tokens + cjk * 1.0 + punct / 2.5 + spaces / 8.0))


def _raw_token_estimate(text: str) -> int:
    """Best raw token estimate for a string, before per-model correction."""
    n = _tiktoken_count(text)
    if n is not None:
        return max(1, n)
    return _heuristic_token_count(text)


def _token_cache_key(payload: str, model_id: str) -> tuple[str, str, int]:
    digest = hashlib.blake2b(payload.encode("utf-8"),
                             digest_size=16).hexdigest()
    return (digest, _cal_key(model_id), _calibration_epoch)


def _count_payload(payload: str, model_id: str = "") -> int:
    """Token count for an already-serialized payload, via the content cache.

    Cache hit: O(1) dict lookup. Miss: one cheap blake2b + the raw
    estimate (tiktoken if available, else the BPE-aware heuristic) times
    the per-model learned correction, so displayed counts track the
    backend's real tokenizer after the first few responses.
    """
    key = _token_cache_key(payload, model_id)
    with _token_cache_lock:
        hit = _token_cache.get(key)
        if hit is not None:
            _token_cache.move_to_end(key)
            return hit
    count = max(1, int(_raw_token_estimate(payload)
                       * _correction_factor(model_id)))
    with _token_cache_lock:
        _token_cache[key] = count
        _token_cache.move_to_end(key)
        while len(_token_cache) > _TOKEN_CACHE_MAX:
            _token_cache.popitem(last=False)  # evict oldest
    return count


def _count_uncached(obj: Any, model_id: str,
                    fingerprint: tuple | None) -> int:
    """Serialize, count via the digest tier, and populate tier 1."""
    try:
        payload = json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        payload = str(obj)
        fingerprint = None  # not the object we fingerprinted; stay safe
    count = _count_payload(payload, model_id)
    if fingerprint is not None:
        # NOTE: plain lists do not support weakref; id() keying is safe
        # here because the fingerprint is re-validated on every lookup
        # (see the tier-1 comment above).
        with _token_cache_lock:
            _id_cache[id(obj)] = (fingerprint, _calibration_epoch, count)
            _id_cache.move_to_end(id(obj))
            while len(_id_cache) > _ID_CACHE_MAX:
                _id_cache.popitem(last=False)
    return count


def estimate_tokens_fast(text: str, model_id: str = "") -> int:
    """Hot-path token estimate for plain text (raw estimate x per-model
    correction, cached).

    Strings are immutable and the count depends only on ``len(text)`` for
    a given calibration, so the key ``(id, len, model bucket, epoch)``
    is exact with no hashing at all: repeated counts of the same string
    are ~microseconds. Non-string input falls back to
    :func:`estimate_tokens`.
    """
    if not isinstance(text, str):
        return estimate_tokens(text, model_id)
    key = (id(text), len(text), _cal_key(model_id), _calibration_epoch)
    with _token_cache_lock:
        hit = _token_cache.get(key)
        if hit is not None:
            _token_cache.move_to_end(key)
            return hit
    count = max(1, int(_raw_token_estimate(text)
                       * _correction_factor(model_id)))
    with _token_cache_lock:
        _token_cache[key] = count
        _token_cache.move_to_end(key)
        while len(_token_cache) > _TOKEN_CACHE_MAX:
            _token_cache.popitem(last=False)
    return count

def estimate_tokens(obj: Any, model_id: str = "") -> int:
    """Deterministic token estimate for any JSON-serialisable object.

    Starts from a conservative ~3.2 chars/token baseline, then corrects
    itself with the REAL chars/token ratio learned from every backend
    response (usage.prompt_tokens vs the prompt we actually sent). The
    backend's own tokenizer is the ground truth — once a few responses
    have landed, this estimate tracks reality instead of a fixed guess,
    which is what keeps long sessions from ever overflowing the window.

    Calibration is PER MODEL (tokenizers differ between models) and is
    persisted to disk, so the very first request of a fresh process on a
    big project already uses the learned ratio.

    Speed: the count for an unchanged object is served from the identity
    cache (~microseconds, no serialization); mutated or new content pays
    one serialization and is then cached by content digest."""
    # Tier 1: same object, verified unmutated -> free.
    fingerprint = _message_fingerprint(obj)
    if fingerprint is not None:
        with _token_cache_lock:
            entry = _id_cache.get(id(obj))
            if entry is not None:
                fp, epoch, count = entry
                if epoch == _calibration_epoch and fp == fingerprint:
                    _id_cache.move_to_end(id(obj))
                    return count
    # Tier 2 (inside _count_uncached): serialize once, digest-cache.
    return _count_uncached(obj, model_id, fingerprint)


# -- learned tokenizer calibration (schema 2: correction multipliers) ---------
# The backend rejects a request when input + max_tokens exceeds the model's
# context window, and it counts input with ITS OWN tokenizer. The raw
# estimate (tiktoken when installed, else the BPE-aware heuristic) only
# approximates it, so we learn a per-model correction multiplier from
# every response and keep a safety margin on top.
#
# Enterprise hardening:
#   * calibration is keyed by model id — switching models mid-session can
#     never poison the estimate (each model has its own tokenizer);
#   * calibration AND learned context windows persist to disk, so a restart
#     on a huge project starts already calibrated;
#   * the context window itself is learned: when a backend error reports a
#     smaller window than configured, we remember it for that model.

_ESTIMATOR_SCHEMA = 2
# NOTE (worker 17/20): schema 2 replaced the old chars/token ratios
# (schema 1). Old persisted samples are ignored on load, never mixed in.
_BASELINE_CORRECTION = 1.0
_MIN_CORRECTION = 0.05           # ignore absurdly cheap usage reports
_MAX_CORRECTION = 20.0          # ignore absurdly expensive ones
_CORRECTION_SAMPLES_MAX = 8     # rolling window of recent measurements
_correction_samples: dict[str, list[float]] = {}  # model id -> multipliers
_learned_windows: dict[str, int] = {}         # model id -> real window
_calibration_loaded = False
_CALIBRATION_FILE = config.APP_DIR / "calibration.json"

# Bumped every time the learned correction changes. The token-count
# cache keys on this epoch so a freshly learned multiplier instantly
# invalidates stale counts (displayed numbers keep tracking reality).
_calibration_epoch = 0


def _bump_calibration_epoch() -> None:
    global _calibration_epoch
    _calibration_epoch += 1


def _cal_key(model_id: str) -> str:
    """Calibration bucket key. Unknown/empty model ids share one bucket."""
    return model_id or "_default"


def _load_calibration() -> None:
    """Restore persisted calibration once per process (best effort)."""
    global _calibration_loaded
    if _calibration_loaded:
        return
    _calibration_loaded = True
    try:
        data = json.loads(_CALIBRATION_FILE.read_text())
    except (OSError, ValueError):
        return
    # Schema 2: per-model correction multipliers. Schema-1 "ratios"
    # (chars/token) are deliberately NOT migrated — different units.
    if data.get("schema") == _ESTIMATOR_SCHEMA:
        corrections = data.get("corrections") or {}
        for key, samples in corrections.items():
            if isinstance(samples, list):
                clean = [float(s) for s in samples
                         if isinstance(s, (int, float))
                         and _MIN_CORRECTION <= float(s) <= _MAX_CORRECTION]
                if clean:
                    _correction_samples[key] = clean[-_CORRECTION_SAMPLES_MAX:]
    windows = data.get("windows") or {}
    for key, win in windows.items():
        if isinstance(win, int) and win > 0:
            _learned_windows[key] = win
    if _correction_samples:
        # Disk state changed the learned multipliers -> cached counts
        # computed under the old ones must go.
        _bump_calibration_epoch()


def _save_calibration() -> None:
    """Persist calibration so the next process starts already tuned."""
    try:
        config.ensure_dirs()
        # Atomic write (tmp + rename): concurrent savers from worker
        # threads can otherwise interleave and tear the JSON, silently
        # losing all calibration on the next load.
        tmp = _CALIBRATION_FILE.with_name(_CALIBRATION_FILE.name + ".tmp")
        tmp.write_text(json.dumps({
            "schema": _ESTIMATOR_SCHEMA,
            "corrections": _correction_samples,
            "windows": _learned_windows,
        }))
        tmp.replace(_CALIBRATION_FILE)
    except OSError:
        pass  # persistence is an optimisation, never a failure path


def _correction_factor(model_id: str = "") -> float:
    """Current per-model correction multiplier: the mean of recent
    actual/raw measurements when we have any, else 1.0 (the raw
    estimate is used as-is)."""
    _load_calibration()
    samples = _correction_samples.get(_cal_key(model_id)) \
        or _correction_samples.get("_default")
    if not samples:
        return _BASELINE_CORRECTION
    return sum(samples) / len(samples)


def learn_token_correction(raw_tokens: float, actual_tokens: int,
                           model_id: str = "") -> None:
    """Record one real (raw estimate, actual) measurement from a backend
    response. Called after every completion whose usage reports
    prompt_tokens. (worker 17/20: renamed from learn_token_ratio — the
    first argument is now the RAW ESTIMATED TOKEN COUNT, not chars.)"""
    if raw_tokens <= 0 or actual_tokens <= 0:
        return
    mult = actual_tokens / raw_tokens
    # ignore implausible outliers (broken usage reporting)
    if not (_MIN_CORRECTION <= mult <= _MAX_CORRECTION):
        return
    key = _cal_key(model_id)
    samples = _correction_samples.setdefault(key, [])
    samples.append(mult)
    if len(samples) > _CORRECTION_SAMPLES_MAX:
        del samples[: len(samples) - _CORRECTION_SAMPLES_MAX]
    _bump_calibration_epoch()  # new ground truth -> drop cached counts
    _save_calibration()


def learn_context_window(model_id: str, window: int) -> None:
    """Remember the real context window a backend reported for a model.
    We only ever shrink the configured window — a backend that reports a
    smaller window is authoritative for that deployment."""
    if window <= 0 or not model_id:
        return
    key = _cal_key(model_id)
    known = _learned_windows.get(key)
    if known is None or window < known:
        _learned_windows[key] = window
        _save_calibration()


def effective_window(model: Model) -> int:
    """The context window to plan against: the configured value, capped by
    anything the backend has actually reported for this model."""
    _load_calibration()
    learned = _learned_windows.get(_cal_key(model.id))
    if learned is None:
        return model.context_window
    return min(model.context_window, learned)


def _prompt_payload_text(messages: list[dict],
                         tools: list[dict] | None) -> str:
    """Serialized form of exactly what we send as the prompt. The raw
    token estimator and the calibration learner both work from this
    text (not just its length) so the estimate tracks the backend's
    real tokenizer."""
    try:
        return json.dumps({"messages": messages, "tools": tools or []},
                          ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(messages) + str(tools or [])


# Safety headroom between the estimated input and the window. Scales with
# input size: the bigger the prompt, the bigger the absolute tokenizer
# error can be, so the margin grows instead of staying a fixed constant.
CONTEXT_MARGIN = 8_192
_MIN_COMPLETION_TOKENS = 1_024    # never clamp max_tokens below this


def _context_margin(window: int) -> int:
    """Fixed safety margin, scaled down for small windows so the fit
    check stays meaningful (an 8k margin on an 8k window could never
    fit anything). Unchanged for all production windows (>= 128k)."""
    return min(CONTEXT_MARGIN, window // 8)


def _input_tokens(model: Model, messages: list[dict],
                  tools: list[dict] | None) -> int:
    """Estimated prompt tokens for the exact request about to be sent.
    Computed once per attempt and shared by the clamp and the hard
    invariant, so the conversation isn't re-serialized for every check."""
    n = estimate_tokens(messages, model.id)
    if tools and model.supports_tools:
        n += estimate_tokens(tools, model.id)
    return n


def _window_max_tokens(model: Model, effort: Effort, messages: list[dict],
                       tools: list[dict] | None,
                       input_tokens: int | None = None) -> int:
    """Clamp the requested max_tokens so that input + max_tokens fits in
    the model's context window. Backends reject the whole request when
    the sum exceeds the window (e.g. 'maximum context length of 262144
    tokens'). The input size is estimated with the BPE-aware raw estimate
    times the learned per-model correction (see estimate_tokens) plus a
    size-scaled margin, so the clamp stays correct even as the
    conversation grows for hours. If the input alone overflows the
    window, raise a clear, recoverable error instead of sending a doomed
    request."""
    requested = effort.max_tokens or 0
    if not requested:
        return 0
    window = effective_window(model)
    if input_tokens is None:
        input_tokens = _input_tokens(model, messages, tools)
    # margin grows with the prompt: ~1 extra token of headroom per 32
    # estimated input tokens, on top of the fixed floor
    margin = _context_margin(window) + input_tokens // 32
    if input_tokens + margin + _MIN_COMPLETION_TOKENS > window:
        raise APIError(
            f"conversation is too large for {model.label} "
            f"(~{input_tokens:,} input tokens vs {window:,} "
            f"context window) — start a new session (/new), rewind "
            f"(/rewind), or switch to a larger-context model (Ctrl+T)")
    headroom = window - input_tokens - margin
    return max(_MIN_COMPLETION_TOKENS, min(requested, headroom))


# FullAgent asks for 200k output tokens everywhere, but each backend has its
# own hard ceiling. Clamp at send time so the request is never rejected for
# an oversized max_tokens; providers without a known cap pass through.
_MAX_TOKENS_CAP: dict[str, int] = {
    "kios": 131_072,
}


def _clamp_max_tokens(provider_key: str, value: int) -> int:
    cap = _MAX_TOKENS_CAP.get(provider_key)
    if cap is None:
        return value
    return min(value, cap)


# Yielded by _iter_sse_events for SSE comment lines (": ping" keepalives).
# Not a real event â dict-checking consumers skip it, but the stall
# watchdog treats it as proof the stream is alive.
_KEEPALIVE = object()
_SSE_QUEUE_BATCH_SIZE = 32


def _iter_sse_events(resp: requests.Response) -> Iterator[dict]:
    """Yield parsed JSON data objects from an SSE stream.

    Proper SSE framing: an event ends at a blank line, and multiple
    `data:` lines inside one event are joined with newline before the
    JSON parse (the old code treated every `data:` line as a complete
    event, so a multi-line data field was split into fragments that
    each failed to parse and were silently dropped). Lines are decoded
    from bytes with errors replaced, so one corrupt chunk can never
    crash the turn with a raw UnicodeDecodeError.
    """
    data_lines: list[str] = []

    def _dispatch() -> Any:
        """Parse the accumulated event. Returns "done", a parsed object,
        or None when there is nothing (or nothing parseable) to emit."""
        text = "\n".join(data_lines).strip()
        del data_lines[:]
        if not text:
            return None
        if text == "[DONE]":
            return "done"
        try:
            parsed = json.loads(text)
        except ValueError:
            return None
        # Provider boundary: a JSON array/string/number is a valid JSON
        # value but not a valid SSE event object.  The declared
        # Iterator[dict] contract promises dicts — yielding a non-dict
        # would hand it to every consumer as if it were an event object.
        # Drop it here instead of crashing downstream.
        if not isinstance(parsed, dict):
            return None
        return parsed

    for raw in resp.iter_lines():
        line = raw.decode("utf-8", errors="replace")
        if line.startswith("\ufeff"):
            line = line.lstrip("\ufeff")
        if not line.strip():
            # blank line: end of event — dispatch what accumulated
            ev = _dispatch()
            if ev == "done":
                return
            if ev is not None:
                yield ev
            continue
        if line.startswith(":"):  # comment / keepalive
            yield _KEEPALIVE  # type: ignore[misc] â sentinel, skipped by consumers
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
    # stream ended without a trailing blank line — flush the last event
    ev = _dispatch()
    if ev is not None and ev != "done":
        yield ev


def _response_socket(resp: requests.Response):
    """Best-effort lookup of the live socket behind a streaming response.

    Walks the private layers (requests -> urllib3 -> http.client ->
    BufferedReader -> SocketIO -> socket), verifying each hop, because
    the exact wrapping varies by requests/urllib3 version and by whether
    the producer is currently inside recv(). Returns None when the
    socket cannot be found (caller falls back to a plain close)."""
    try:
        import socket as _socket
    except ImportError:  # pragma: no cover — stdlib always present
        return None
    try:
        raw = getattr(resp, "raw", None)
        # Path 1 (observed): urllib3.HTTPResponse._fp (http.client.HTTPResponse)
        #   -> .fp (BufferedReader) -> .raw (SocketIO) -> ._sock
        fp = getattr(getattr(raw, "_fp", None), "fp", None)
        inner = getattr(fp, "raw", None)
        sock = getattr(inner, "_sock", None)
        if isinstance(sock, _socket.socket):
            return sock
        # Path 2: SocketIO directly under http.client.HTTPResponse
        sock = getattr(fp, "_sock", None)
        if isinstance(sock, _socket.socket):
            return sock
        # Path 3: urllib3 connection object
        conn = getattr(raw, "_connection", None)
        sock = getattr(conn, "sock", None)
        if isinstance(sock, _socket.socket):
            return sock
    except Exception:  # noqa: BLE001 — lookup is best-effort
        pass
    return None


def _abort_response(resp: requests.Response) -> None:
    """Unblock a response whose socket is stuck in recv() in another
    thread, then close it. Best-effort: never raises.

    WHY shutdown() first: the SSE producer thread blocks in socket recv()
    while holding the socket file's read lock. A plain resp.close() from
    this thread then blocks on that lock until the producer's read times
    out — so a stall watchdog (or Esc) that just calls close() would hang
    for the full read budget instead of failing fast. shutdown() makes
    the blocked recv() return immediately, the producer releases the
    lock, and close() proceeds."""
    try:
        sock = _response_socket(resp)
        if sock is not None:
            try:
                import socket as _socket
                sock.shutdown(_socket.SHUT_RDWR)
            except OSError:
                pass  # already closed/reset by the peer — fine
    except Exception:  # noqa: BLE001 — abort is best-effort
        pass
    try:
        resp.close()
    except Exception:  # noqa: BLE001 — abort is best-effort
        pass


def _iter_sse_events_cancellable(
        resp: requests.Response,
        should_cancel: Callable[[], bool] | None,
        stall_timeout: float = STREAM_STALL_TIMEOUT) -> Iterator[dict]:
    """Yield SSE events, honouring should_cancel even while the provider
    stalls. A stalled stream blocks inside iter_lines until the (long)
    read timeout — without this, Esc/Ctrl+C does nothing for up to
    DEFAULT_TIMEOUT (300s) while no chunks arrive. A daemon producer
    thread keeps consuming; this thread polls with a short timeout and
    checks cancellation between polls. Producer exceptions (e.g.
    ChunkedEncodingError on a mid-stream disconnect) are re-raised here
    with their original type so the retry layer still sees them.

    STALL WATCHDOG: requests' read timeout is per socket read, so a
    provider dribbling one byte just under the read budget could hold the
    stream open forever. If no event and no keepalive arrives for
    `stall_timeout` seconds, the response is closed (unblocking the
    producer's socket read) and an APITimeoutError is raised — the call
    fails fast instead of hanging."""
    q: queue.Queue = queue.Queue()
    _END = object()

    def _produce() -> None:
        batch: list[Any] = []
        error: Exception | None = None
        batch_lock = threading.Lock()
        timer_token = None
        timer_handle = None

        def _flush_due(token) -> None:
            nonlocal batch, timer_token, timer_handle
            with batch_lock:
                if token is not timer_token:
                    return
                timer_token = None
                timer_handle = None
                if batch:
                    q.put(batch)
                    batch = []

        try:
            for event in _iter_sse_events(resp):
                with batch_lock:
                    batch.append(event)
                    if len(batch) >= _SSE_QUEUE_BATCH_SIZE:
                        if timer_handle is not None:
                            timer_handle.cancel()
                        timer_token = None
                        timer_handle = None
                        q.put(batch)
                        batch = []
                    elif timer_handle is None:
                        token = object()
                        timer_token = token
                        timer_handle = threading.Timer(
                            0.002, _flush_due, args=(token,))
                        timer_handle.daemon = True
                        timer_handle.start()
        except Exception as e:  # noqa: BLE001 — re-raised in the consumer
            error = e
        finally:
            with batch_lock:
                if timer_handle is not None:
                    timer_handle.cancel()
                timer_token = None
                timer_handle = None
                if batch:
                    q.put(batch)
                    batch = []
                if error is not None:
                    q.put(error)
                q.put(_END)

    thread = threading.Thread(target=_produce, daemon=True,
                              name="sse-producer")
    thread.start()
    last_data = time.monotonic()
    pending = iter(())
    while True:
        try:
            item = next(pending)
        except StopIteration:
            try:
                packet = q.get(timeout=0.25)
            except queue.Empty:
                if should_cancel is not None and should_cancel():
                    # Abort the response: unblocks the producer's socket
                    # read (a plain close here or in the finally would block
                    # on the socket read lock until the read times out).
                    # The finally in _chat_stream_once then closes an
                    # already-dead socket.
                    _abort_response(resp)
                    raise TurnCancelled()
                if time.monotonic() - last_data > stall_timeout:
                    # Abort (not just close): the producer thread is blocked
                    # in socket recv() holding the read lock — a plain
                    # resp.close() would block on that lock until the read
                    # times out. _abort_response shuts the socket down first
                    # so the recv returns immediately and close() proceeds.
                    _abort_response(resp)
                    raise APITimeoutError(
                        f"stream stalled: no data for {stall_timeout:g}s — "
                        f"the provider stopped sending; /retry to resend")
                continue
            if isinstance(packet, list):
                pending = iter(packet)
                continue
            item = packet
        if item is _END:
            return
        if isinstance(item, Exception):
            raise item
        if should_cancel is not None and should_cancel():
            _abort_response(resp)
            raise TurnCancelled()
        # Any event or keepalive comment proves the stream is alive.
        last_data = time.monotonic()
        if item is _KEEPALIVE:
            continue
        yield item


def _with_error_type(msg: str, err: dict) -> str:
    """Prefix the provider's machine-readable error type (e.g.
    ``invalid_request_error``) when the message doesn't already carry it.

    Downstream recovery keys on the type — without it a 400 is
    indistinguishable from any other 400 and the turn dies as a dead
    error box. Never raises; purely cosmetic."""
    try:
        t = err.get("type") or err.get("code")
    except Exception:  # noqa: BLE001 — defensive: err shape is untrusted
        return msg
    if isinstance(t, str) and t.strip() and t.strip().lower() not in msg.lower():
        return f"{t.strip()}: {msg}"
    return msg


def _extract_error_message(body: str, status: int | None = None) -> str:
    """Best-effort human-readable message from an error response body.

    `status` feeds the last-resort fallback: a blank/whitespace-only body
    used to produce a BLANK APIError message, and agent.py displays
    str(error) with no status attached — the user got an empty error box
    with no hint that e.g. a 503 had happened. Never raises.
    """
    def _blank_fallback() -> str:
        if status is not None:
            return (f"provider returned HTTP {status} with an empty "
                    f"error body")
        return "provider returned an empty error body"

    try:
        obj = json.loads(body)
    except ValueError:
        if body.strip():
            return body[:500]
        return _blank_fallback()
    if not isinstance(obj, dict):
        # valid JSON but not an object (array/string/number) — no shape to dig
        return body[:500]
    err = obj.get("error")
    if isinstance(err, dict):
        return _with_error_type(str(err.get("message") or err), err)
    if isinstance(err, str):
        return err
    message = obj.get("message")
    if isinstance(message, str):
        return _with_error_type(message, obj)
    return body[:500]


def _error_obj_message(err: Any) -> str:
    """Coerce an already-parsed ``error`` payload to a string. Never raises.

    Providers send the error as a dict ({"message": ...}), a bare string,
    or occasionally some other shape (a list, a number). Every
    error-handling site funnels through here so APIError always carries
    a str message — no shape can leak through and crash str formatting
    or .lower() calls downstream."""
    if isinstance(err, dict):
        return str(err.get("message") or err)
    if isinstance(err, str):
        return err
    return str(err)


def _strict_str(value: Any) -> str:
    """str-or-empty: keep real strings, map every other shape to "".

    Stricter than _typeguards.ensure_str (which str()-ifies ints: 5 ->
    "5"): a non-string content/reasoning/model/finish_reason is
    provider junk, and surfacing it as "123" would corrupt display,
    history and downstream string ops. Never raises."""
    return value if isinstance(value, str) else ""


def _coerce_text(value: Any) -> str:
    """Coerce any content/reasoning shape to str. Never raises.

    Plain strings pass through; Anthropic-style content blocks
    ([{"type": "text", "text": "hi"}]) are joined on their text parts;
    every other shape (None, int, dict, ...) becomes "". Providers
    (including Kilo's gateway) have been observed sending
    non-string content."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts = []
        for item in value:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    texts.append(t)
        return "".join(texts)
    return ""


def _coerce_arguments(value: Any) -> str:
    """Coerce any tool-call arguments shape to a JSON string. Never raises.

    Downstream parse_tool_arguments() does json.loads on this, so a
    dict/list value is json.dumps-ed (str() would produce a Python repr
    with single quotes, which json.loads rejects)."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


# -- context-overflow detection + self-healing ------------------------------
# Backends reject a request outright when input + max_tokens exceeds the
# context window, and the error message carries the REAL token counts
# (e.g. "maximum context length of 262144 tokens ... 67440 tokens from the
# input messages and 195680 tokens for the completion"). We parse those
# numbers, learn the true chars/token ratio from them, re-clamp max_tokens
# to the actual headroom, and retry — so a long session heals itself
# instead of dying with the error.

_OVERFLOW_MARKERS = ("context length", "context_length", "context window",
                     "too many tokens", "maximum context",
                     "reduce the number of tokens")


def is_context_overflow(message: str) -> bool:
    """True when an API error message is a context-window overflow."""
    low = message.lower()
    return any(m in low for m in _OVERFLOW_MARKERS)


def _parse_overflow(message: str) -> dict | None:
    """Extract real token counts from a backend overflow error. Returns a
    dict with any of: window, input_tokens, completion_tokens, total."""
    if not is_context_overflow(message):
        return None
    low = message.lower()
    info: dict[str, int] = {}

    def _num(pattern: str) -> int | None:
        m = re.search(pattern, low)
        return int(m.group(1).replace(",", "")) if m else None

    window = (_num(r"maximum context length (?:of|is)\s+([\d,]+)\s+tokens")
              or _num(r"context (?:length|window) (?:of|is)\s+([\d,]+)\s+tokens"))
    if window:
        info["window"] = window
    inp = (_num(r"([\d,]+)\s+tokens?\s+from the input")
           or _num(r"([\d,]+)\s+tokens?\s+(?:in|from)\s+(?:the\s+)?(?:input|prompt|messages)")
           or _num(r"(?:input|prompt|messages)\s+(?:resulted in|is|are)\s+([\d,]+)\s+tokens"))
    if inp:
        info["input_tokens"] = inp
    comp = _num(r"([\d,]+)\s+tokens?\s+for (?:the )?completion")
    if comp:
        info["completion_tokens"] = comp
    total = (_num(r"a total of\s+([\d,]+)\s+tokens")
             or _num(r"requested (?:a total of )?([\d,]+)\s+tokens"))
    if total:
        info["total"] = total
    return info or {}


def _fit_max_tokens_from_actual(model: Model, info: dict,
                                effort: Effort) -> int | None:
    """max_tokens that provably fits, computed from the backend's OWN
    counts. None means the input alone overflows — the caller must shrink
    the conversation, not the completion budget."""
    if info.get("window"):
        learn_context_window(model.id, info["window"])
    window = effective_window(model)
    actual_input = info.get("input_tokens")
    if not actual_input:
        total, comp = info.get("total"), info.get("completion_tokens")
        if total and comp:
            actual_input = total - comp
    if not actual_input:
        return None
    margin = max(4_096, window // 64)
    headroom = window - actual_input - margin
    if headroom < _MIN_COMPLETION_TOKENS:
        return None
    requested = effort.max_tokens or headroom
    return min(requested, headroom)


def _learn_from_usage(usage: Any, sent_text: str | int | None = None,
                     model_id: str = "", *,
                     sent_chars: int | None = None) -> None:
    """Calibrate from provider usage without trusting either input shape.

    Production callers pass the serialized prompt text. Older callers may
    provide its character count positionally or as ``sent_chars``; for that
    form use a bounded, conservative four-characters-per-token estimate.
    Malformed usage and prompt metadata are ignored rather than raised.
    """
    prompt_tokens, _ = safe_token_counts(usage)
    if prompt_tokens <= 0:
        return
    if isinstance(sent_text, str):
        raw_tokens = _raw_token_estimate(sent_text)
    else:
        char_count = _coerce_token_count(
            sent_chars if sent_chars is not None else sent_text)
        if char_count <= 0:
            return
        raw_tokens = max(1, (char_count + 3) // 4)
    learn_token_correction(raw_tokens, prompt_tokens, model_id)


def shrink_tool_outputs(messages: list[dict], keep: int = 1,
                        max_chars: int = 400) -> bool:
    """In-place overflow shrinker for any message list. Used as the
    on_overflow callback by isolated chat consumers whose own tool
    loop can bloat their context — the same protection the main agent
    gets. Three escalating passes, all pairing-safe:
      1. truncate the OLDEST tool results to short summaries (keeping the
         newest `keep` verbatim),
      2. if nothing was stale, truncate the newest tool results too,
      3. if still nothing shrank, drop the oldest complete turn unit
         (user message + its assistant reply + tool results) so
         tool_call / tool-response pairing is never broken.
    Returns True if anything actually shrank."""
    tool_idx = [i for i, m in enumerate(messages)
                if m.get("role") == "tool"]
    stale = tool_idx if keep <= 0 else tool_idx[:-keep]

    def _truncate(indices: list[int]) -> bool:
        shrank = False
        for i in indices:
            content = str(messages[i].get("content") or "")
            if len(content) > max_chars:
                messages[i]["content"] = (
                    content[:max_chars]
                    + f"\n[… truncated — {len(content):,} chars originally]")
                shrank = True
        return shrank

    if _truncate(stale):
        return True
    if _truncate(tool_idx):          # even the newest results, if desperate
        return True

    # drop the oldest complete turn unit (never a lone message)
    start = 1 if messages and messages[0].get("role") == "system" else 0
    end = None
    for j in range(start + 1, len(messages)):
        if messages[j].get("role") == "user":
            end = j
            break
    if end is not None and end > start:
        del messages[start:end]
        return True
    return False


def _check_api_key(provider: Provider, model: Model | None = None) -> None:
    """Require credentials. Kios ships an embedded key (zero-config), so
    this only fires if the user explicitly cleared it."""
    if not provider.api_key:
        env_name = f"{provider.key.upper()}_API_KEY"
        raise APIError(
            f"no API key configured for {provider.name}. Set the "
            f"{env_name} environment variable or save the key in "
            f"{config.APP_DIR / (provider.key + '_api_key')} and restart.", status=401)


def chat_stream(provider: Provider, model: Model, effort: Effort,
                messages: list[dict], tools: list[dict] | None,
                on_token: Callable[[str], None] | None = None,
                on_reasoning: Callable[[str], None] | None = None,
                on_tool_start: Callable[[str], None] | None = None,
                on_tool_args: Callable[[str, str], None] | None = None,
                should_cancel: Callable[[], bool] | None = None,
                on_overflow: Callable[[], bool] | None = None,
                timeout: float = config.DEFAULT_TIMEOUT,
                on_status: Callable[[str], None] | None = None) -> StreamResult:
    """Send a streaming chat completion request; calls callbacks as tokens
    arrive; returns the fully accumulated result.

    on_status (optional): stream-time status notices that are NOT model
    output — e.g. the stall warning when the stream is open but no tokens
    arrive. It is passed through the retry layer unchanged, never marks
    the turn as "output emitted", and defaults to None (warnings then
    fall back to the on_reasoning path so direct callers still see them).

    Context-overflow recovery — three escalating layers, so a long session
    on a huge project never dies with a context-length error:
      1. pre-flight clamp with the learned per-model chars/token ratio;
      2. on rejection, parse the backend's REAL token counts, recalibrate,
         re-clamp max_tokens, retry (up to OVERFLOW_RETRIES times);
      3. if the input itself no longer fits, invoke on_overflow (the
         caller shrinks the conversation — e.g. emergency compaction) and
         retry with the shrunken messages.
    The payload invariant is re-asserted before every send: the request
    that goes out always satisfies input + max_tokens <= window."""
    _check_api_key(provider, model)
    url = provider.base_url.rstrip("/") + "/chat/completions"
    headers = _base_headers(provider, stream=True)
    current_effort = effort
    shrinks_used = 0
    # invalid_request_error auto-recovery (worker 18/20): exactly one
    # prune-and-retry per call, enforced by maybe_recover_invalid_request.
    invalid_retried = False

    # Tracks whether ANY output reached the UI during the current attempt.
    # A retry replays the WHOLE request — once tokens/tool-args have been
    # shown, retrying would duplicate the completion on screen, so
    # post-output failures surface immediately instead of being retried.
    streamed = {"out": False}

    def _mark_emitted(cb):
        if cb is None:
            return None

        def _wrapped(piece: str) -> None:
            streamed["out"] = True
            cb(piece)

        return _wrapped

    def _mark_emitted_args(cb):
        if cb is None:
            return None

        def _wrapped(name: str, chunk: str) -> None:
            streamed["out"] = True
            cb(name, chunk)

        return _wrapped

    on_token_w = _mark_emitted(on_token)
    on_reasoning_w = _mark_emitted(on_reasoning)
    on_tool_args_w = _mark_emitted_args(on_tool_args)

    for overflow_attempt in range(OVERFLOW_RETRIES + OVERFLOW_SHRINKS + 1):
        sent_text = _prompt_payload_text(messages, tools)
        try:
            payload = build_payload(model, current_effort, messages, tools,
                                    stream=True)
        except APIError as e:
            # pre-flight refusal (input already over the window). Give the
            # caller a chance to shrink the conversation and retry.
            if (is_context_overflow(str(e)) and on_overflow is not None
                    and shrinks_used < OVERFLOW_SHRINKS and on_overflow()):
                shrinks_used += 1
                continue
            raise
        try:
            result = _chat_stream_with_retries(
                url, headers, payload, on_token_w, on_reasoning_w,
                on_tool_start, on_tool_args_w, should_cancel, timeout,
                on_status=on_status)
            _learn_from_usage(result.usage, sent_text, model.id)
            return result
        except APIError as e:
            # reasoning_content validation — heal history and retry once,
            # but NEVER after output already streamed (see `streamed`)
            if _is_reasoning_content_error(e):
                if streamed["out"]:
                    raise
                if _sanitize_messages(messages):
                    continue
                # already sanitized but provider still rejects — try
                # stripping reasoning entirely and retry with sanitized copy
                # (some providers accept empty string, some need removal)
                # we already sanitized, so just raise with clearer guidance
                raise APIError(
                    f"{e} — history was sanitized but provider still "
                    f"rejects. Try /new or /rewind to clear the stale "
                    f"thinking turn.") from e
            if streamed["out"]:
                # _chat_stream_with_retries only raises post-output for
                # non-retryable failures; healing or shrinking now would
                # replay the request on top of the partial output
                raise
            # invalid_request_error auto-recovery (worker 18/20): diagnose
            # + log a full payload summary, prune the oldest half of tool
            # pairs, and retry ONCE with the cleaned history. The helper
            # raises a helpful error (never a bare "invalid request") when
            # the single retry is exhausted or nothing was prunable.
            # Imported lazily: errorrecovery imports from .client at call
            # time, so a top-level import would be circular.
            from . import errorrecovery as _errrec
            if _errrec.maybe_recover_invalid_request(
                    e, messages, tools, on_status, invalid_retried):
                invalid_retried = True
                continue
            healed = _heal_overflow(model, current_effort, e,
                                    overflow_attempt, sent_text)
            if healed is not None:
                current_effort = healed
                continue
            # clamping cannot help — the input itself is too big. Let the
            # caller shrink the conversation, then retry.
            if (is_context_overflow(str(e)) and on_overflow is not None
                    and shrinks_used < OVERFLOW_SHRINKS and on_overflow()):
                shrinks_used += 1
                continue
            raise

    # retries exhausted — surface the last overflow error
    raise APIError(
        f"conversation is too large for {model.label} even after "
        f"re-clamping — start a new session (/new) or rewind (/rewind)")


OVERFLOW_RETRIES = 2   # re-clamp-and-retry attempts (input still fits)
OVERFLOW_SHRINKS = 2   # shrink-the-conversation-and-retry attempts


def _heal_overflow(model: Model, effort: Effort, error: APIError,
                   attempt: int, sent_text: str = "") -> Effort | None:
    """Turn a backend overflow error into a tighter Effort, or None when
    the input itself no longer fits (nothing a clamp can fix)."""
    if not is_context_overflow(str(error)):
        return None
    if attempt >= OVERFLOW_RETRIES:
        return None
    info = _parse_overflow(str(error)) or {}
    # the backend just told us the real input size — calibrate the
    # per-model estimator with it so every later clamp is accurate
    if info.get("input_tokens") and sent_text:
        learn_token_correction(_raw_token_estimate(sent_text),
                               info["input_tokens"], model.id)
    fitted = _fit_max_tokens_from_actual(model, info, effort)
    if fitted is None:
        return None
    if effort.max_tokens is not None and fitted >= effort.max_tokens:
        return None  # no tightening left to do
    return replace(effort, max_tokens=fitted)


def _chat_stream_with_retries(
        url: str, headers: dict, payload: dict,
        on_token: Callable[[str], None] | None,
        on_reasoning: Callable[[str], None] | None,
        on_tool_start: Callable[[str], None] | None,
        on_tool_args: Callable[[str, str], None] | None,
        should_cancel: Callable[[], bool] | None,
        timeout: float,
        on_status: Callable[[str], None] | None = None) -> StreamResult:
    """The plain retry loop (rate limits, timeouts, connection errors).

    A retry restarts the WHOLE request — once any token has already been
    streamed to the UI a restart would replay (duplicate) the completion on
    top of the partial output, so mid-output failures surface immediately
    instead of being retried.

    Transient non-rate-limit failures share a per-call deadline. HTTP 429
    responses retry until success or cancellation, using Retry-After when
    provided and capped backoff otherwise; each resumed request receives a
    fresh per-attempt timeout."""
    last_error: Exception | None = None
    attempt = 0
    emitted = {"out": False}
    deadline = time.monotonic() + timeout

    def _tok(piece: str) -> None:
        emitted["out"] = True
        if on_token:
            on_token(piece)

    def _reason(piece: str) -> None:
        emitted["out"] = True
        if on_reasoning:
            on_reasoning(piece)

    def _targs(name: str, chunk: str) -> None:
        # tool-call arguments are already flowing to the UI — a retry now
        # would replay them on top of the live write, so mark as emitted
        emitted["out"] = True
        if on_tool_args:
            on_tool_args(name, chunk)

    rate_attempt = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # The whole per-call budget is spent — fail fast instead of
            # starting another attempt with a fresh full timeout.
            raise APITimeoutError(
                f"request timed out after {timeout:g}s") from last_error
        try:
            return _chat_stream_once(url, headers, payload,
                                     _tok, _reason, on_tool_start,
                                     _targs, should_cancel,
                                     min(timeout, remaining),
                                     on_status=on_status)
        except TurnCancelled:
            raise
        except APIError as e:
            last_error = e
            if _is_rate_limited(e) and not emitted["out"]:
                _wait_for_rate_limit(e, rate_attempt, should_cancel, on_status)
                rate_attempt += 1
                attempt = 0
                deadline = time.monotonic() + timeout
                continue
            if (e.status in RETRY_STATUSES and not emitted["out"]
                    and not _is_permanent_error(e)
                    and attempt < MAX_RETRIES - 1):
                _sleep_capped_cancellable(_backoff(attempt), deadline,
                                          should_cancel)
                attempt += 1
                continue
            raise
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < MAX_RETRIES - 1 and not emitted["out"]:
                # CANCEL: backoff sleeps honour Esc — the old bare sleep() ignored
                # the cancel flag for the whole backoff window
                _sleep_capped_cancellable(_backoff(attempt), deadline,
                                          should_cancel)
                attempt += 1
                continue
            if emitted["out"]:
                raise APITimeoutError(
                    f"stream stalled: no data for {timeout:g}s after output "
                    f"began — /retry to resend") from e
            raise APITimeoutError(
                f"request timed out after {timeout:g}s") from e
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            # ChunkedEncodingError is the typical MID-STREAM disconnect
            # (urllib3 ProtocolError / IncompleteRead) — without catching
            # it here it bypasses the retry budget entirely
            last_error = e
            if attempt < MAX_RETRIES - 1 and not emitted["out"]:
                # CANCEL: backoff sleeps honour Esc — the old bare sleep() ignored
                # the cancel flag for the whole backoff window
                _sleep_capped_cancellable(_backoff(attempt), deadline,
                                          should_cancel)
                attempt += 1
                continue
            raise APIError(f"connection failed: {e}") from e
        except requests.exceptions.RequestException as e:
            # Anything else requests can raise (TooManyRedirects,
            # InvalidURL, ...): not retryable, but surface it as a clean
            # APIError instead of a raw requests exception.
            raise APIError(f"request failed: {e}") from e
    raise APIError(str(last_error))


def _sse_error_status(err: Any) -> int | None:
    """Best-effort HTTP status for an SSE-embedded error object, so the
    retry layer can treat a streamed 429 like a real 429 (and never retry
    a streamed 401). OpenAI-style error objects carry a numeric `code`;
    some gateways send it as a string or a "rate_limit_exceeded" slug."""
    if not isinstance(err, dict):
        return None
    code = err.get("code")
    if code is None:
        code = err.get("status")
    if isinstance(code, bool):
        return None
    if isinstance(code, int):
        return code if 100 <= code < 600 else None
    if isinstance(code, str):
        low = code.strip().lower()
        if low.isdigit():
            n = int(low)
            return n if 100 <= n < 600 else None
        if "rate" in low and "limit" in low:
            return 429
    return None


def _chat_stream_once(url: str, headers: dict, payload: dict,
                      on_token: Callable[[str], None] | None,
                      on_reasoning: Callable[[str], None] | None,
                      on_tool_start: Callable[[str], None] | None,
                      on_tool_args: Callable[[str, str], None] | None,
                      should_cancel: Callable[[], bool] | None,
                      timeout: float,
                      on_status: Callable[[str], None] | None = None
                      ) -> StreamResult:
    result = StreamResult()
    # Stream-hot path: content/reasoning/tool-args arrive in thousands of
    # small chunks. `result.content += piece` per chunk is O(n^2) string
    # copying that visibly slows the loop on long outputs; append to lists
    # and join once at the end (O(n)). Callbacks still fire per chunk, so
    # display latency is unchanged.
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tc_acc: dict[int, ToolCallDelta] = {}
    announced_tools: set[int] = set()

    # CANCEL: the initial POST used to block with no cancel checks —
    # requests' timeout does NOT cover DNS resolution, and a stalled
    # connect/TLS handshake ignored Esc entirely. Run it on a worker and
    # poll, so Esc interrupts within ~0.25s even mid-connect.
    def _send() -> "requests.Response":
        r = _http().post(url, headers=headers, json=payload,
                         stream=True, timeout=_timeouts(timeout))
        if r.status_code != 200:
            # error bodies are small, but a hostile server could still
            # stall .text until the read timeout — keep it guarded too
            body = r.text
            retry_after = _parse_retry_after(r.headers.get("Retry-After"))
            r.close()  # release the pooled connection on the error path
            raise APIError(_extract_error_message(body, r.status_code),
                           status=r.status_code, retry_after=retry_after)
        return r
    resp = run_cancellable(_send, should_cancel, name="model request")

    # The stop event is created BEFORE the try so the finally below can
    # always stop the background stall-timer thread — an exception out
    # of the event loop (SSE-embedded error event, Esc cancel, stall
    # timeout, mid-stream disconnect) used to leak the daemon thread,
    # which then kept firing on_status/on_reasoning callbacks on a dead
    # stream every 5s.
    import threading as _th
    _stall_stop = _th.Event()

    try:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "text/event-stream" not in ctype:
            # Some providers return a non-streamed JSON body even when
            # stream=true was requested — parse it like blocking mode
            # instead of silently dropping the whole completion.
            # CANCEL: .json() reads the whole body — a stalled server
            # could hold this until the (long) read timeout; guard it.
            try:
                data = run_cancellable(resp.json, should_cancel,
                                       name="model response")
            except ValueError as e:
                raise APIError(
                    f"provider returned invalid JSON "
                    f"(status {resp.status_code}): {str(e)[:200]}",
                    status=resp.status_code) from e
            except BaseException:
                # CANCEL/read timeout: the abandoned worker may still be
                # blocked in recv() holding the socket read lock — shut the
                # socket down first, or the finally's resp.close() blocks
                # until the read timeout (same hazard as the SSE path).
                _abort_response(resp)
                raise
            # NB: the isinstance check must happen BEFORE data.get — a
            # provider returning a JSON array/string here used to escape
            # as a raw AttributeError instead of a clean APIError.
            model_id = data.get("model") if isinstance(data, dict) else None
            return _result_from_json(data, str(model_id or ""))
        # Every stream goes through the cancellable iterator: it honours
        # Esc/Ctrl+C mid-stall AND aborts the stream when no data arrives
        # for STREAM_STALL_TIMEOUT (the plain iterator has no watchdog —
        # a dribbling provider could hold the call open forever).
        events = _iter_sse_events_cancellable(resp, should_cancel,
                                              STREAM_STALL_TIMEOUT)
        # STALL WATCH: the hard watchdog above only fires when NO data at
        # all arrives — a stream dribbling reasoning chunks (the 512s
        # silent hang: UI stuck on "reasoning...", zero content tokens)
        # never trips it. Warn the user instead of hanging silently.
        # NB: only VISIBLE output (content tokens, tool-call activity)
        # resets the timer — reasoning pieces alone still trigger the
        # warning, which is exactly the incident being fixed.
        stall = StallWatcher()
        # The stall check MUST run on a background timer, not inside the
        # event loop — if the stream stalls (no events), the loop never
        # iterates and the check never fires. This was the 140s bug.
        # (_stall_stop is created before the try; the finally stops it on
        # every exit path.)

        def _stall_check() -> None:
            warning = stall.check()
            if warning is None:
                return
            if on_status is not None:
                try:
                    on_status(warning)
                except Exception:
                    pass
            elif on_reasoning is not None:
                try:
                    on_reasoning(warning)
                except Exception:
                    pass

        def _stall_timer() -> None:
            while not _stall_stop.wait(5.0):  # check every 5s
                _stall_check()

        _timer = _th.Thread(target=_stall_timer, daemon=True)
        _timer.start()

        def _stall_check_inline() -> None:
            warning = stall.check()
            if warning is None:
                return
            # Same never-crash rule as the background _stall_check above:
            # a raising user callback must not kill the stream loop.
            if on_status is not None:
                try:
                    on_status(warning)
                except Exception:
                    pass
            elif on_reasoning is not None:
                # fallback so direct client users (no on_status) still see
                # it: on_reasoning is the existing dim-notice path. Via the
                # retry wrapper this also marks emitted["out"] — intended:
                # the user has seen activity, so a later failure must not
                # retry-and-replay on top of it.
                try:
                    on_reasoning(warning)
                except Exception:
                    pass

        for event in events:
            _stall_check_inline()
            if should_cancel is not None and should_cancel():
                _stall_stop.set()
                raise TurnCancelled()
            # Canonicalize untrusted provider events before touching nested
            # choice/delta fields. Non-object SSE payloads are ignored here;
            # the same malformed shape in a final JSON response is an APIError.
            try:
                event = normalize_provider_response(event, for_stream=True)
            except APIError:
                continue
            if event.get("model"):
                # HARDEN: model must stay a str — a non-str model
                # (int/dict from a sloppy provider) would violate the
                # StreamResult field type (no normalize() backstop here).
                result.model = _typeguards.ensure_str(event["model"])
            if event.get("usage"):
                # Some providers return usage as a string or malformed type.
                # Only accept dicts to prevent AttributeError downstream.
                _u = event["usage"]
                if isinstance(_u, dict):
                    result.usage = _u
            if event.get("error"):
                err = event["error"]
                # HARDEN: message may be missing or a non-str (dict/list
                # from a sloppy provider) — APIError must get a real str.
                _raw_msg = err.get("message") if isinstance(err, dict) else err
                msg = (_raw_msg if isinstance(_raw_msg, str) and _raw_msg
                       else str(err))
                if isinstance(err, dict):
                    msg = _with_error_type(msg, err)
                raise APIError(msg, status=_sse_error_status(err))
            # HARDEN: choices may be a string or other non-list shape —
            # indexing a string yields a str whose .get() raises
            # AttributeError (the reported malformed-call crash). Coerce every
            # level; a malformed entry is skipped, never fatal.
            choices = _typeguards.ensure_list(event.get("choices"))
            if not choices:
                continue
            choice = _typeguards.ensure_dict(choices[0])
            if not choice:
                continue
            delta = _typeguards.ensure_dict(choice.get("delta"))

            if choice.get("finish_reason"):
                _fr = choice["finish_reason"]
                result.finish_reason = (None if _fr is None
                                        else _typeguards.ensure_str(_fr))

            # HARDEN: content may arrive as blocks (list/dict) where a
            # str was assumed — a non-str would raise TypeError at
            # "".join(content_parts) below.
            piece = _typeguards.ensure_str(delta.get("content"))
            if piece:
                content_parts.append(piece)
                stall.token_received()  # visible token — reset stall timer
                if on_token:
                    on_token(piece)

            # HARDEN: same str-only rule as content — a non-str would
            # raise TypeError at "".join(reasoning_parts) below.
            reasoning = _typeguards.ensure_str(delta.get("reasoning_content")
                                               or delta.get("reasoning"))
            if reasoning:
                reasoning_parts.append(reasoning)
                # NB: reasoning does NOT reset the stall timer — it is not
                # a visible token, and a reasoning-only stream is exactly
                # the silent-hang incident this fixes.
                if on_reasoning:
                    on_reasoning(reasoning)

            tool_calls_raw = delta.get("tool_calls")
            # HARDEN: a string tool_calls would iterate char-by-char and
            # crash on char.get(); only accept a list, and skip non-dict
            # entries individually (never let one bad entry kill the turn).
            if not isinstance(tool_calls_raw, list):
                tool_calls_raw = []
            for tc in tool_calls_raw:
                if not isinstance(tc, dict):
                    continue
                idx = tc.get("index", 0)
                # HARDEN: index may be a string/None; int() can raise —
                # default to 0 so sorted(tc_acc) later can't TypeError on
                # mixed-type keys.
                try:
                    idx = int(idx)
                except (TypeError, ValueError, OverflowError):
                    # OverflowError: int(float('inf')) from a sloppy
                    # provider's {"index": 1e999}.
                    idx = 0
                acc = tc_acc.setdefault(idx, ToolCallDelta())
                _tc_id = tc.get("id")
                # HARDEN: ToolCallDelta.id is declared str — only accept
                # real non-empty strings.
                if isinstance(_tc_id, str) and _tc_id:
                    acc.id = _tc_id
                fn = tc.get("function")
                # HARDEN: function as a string would crash fn.get below.
                if not isinstance(fn, dict):
                    fn = {}
                if fn.get("name"):
                    # HARDEN: a tool name must be a str — a non-str name
                    # (int/dict from a sloppy provider) is dropped instead
                    # of polluting the accumulated name (it could never
                    # dispatch to a real tool anyway).
                    _name = fn["name"]
                    if isinstance(_name, str) and _name:
                        acc.name_parts.append(_name)
                        stall.token_received()  # visible activity
                        if idx not in announced_tools and on_tool_start:
                            announced_tools.add(idx)
                            on_tool_start(acc.name)
                if fn.get("arguments"):
                    _args = fn["arguments"]
                    if isinstance(_args, str):
                        pass
                    elif isinstance(_args, (dict, list)):
                        # json.dumps, not str(): downstream parses this
                        # as JSON — a Python repr would be unparseable.
                        try:
                            _args = json.dumps(_args, ensure_ascii=False)
                        except (TypeError, ValueError):
                            _args = str(_args)
                    else:
                        _args = str(_args)
                    acc.arguments_parts.append(_args)
                    stall.token_received()  # visible activity
                    if on_tool_args:
                        on_tool_args(acc.name, _args)
    finally:
        # Stop the stall timer on EVERY exit path (normal return, error
        # event, cancel, stall timeout, disconnect) — without this an
        # exception out of the loop leaked the daemon thread, which kept
        # firing on_status/on_reasoning callbacks on a dead stream.
        _stall_stop.set()
        resp.close()

    result.content = "".join(content_parts)
    result.reasoning = "".join(reasoning_parts)

    for idx in sorted(tc_acc):
        acc = tc_acc[idx]
        # HARDEN: a nameless accumulated call can never dispatch to a
        # real tool — drop it instead of emitting {"name": ""}, which
        # only wastes a turn on a bogus "tool '' not available" call.
        if not acc.name:
            continue
        result.tool_calls.append({
            "id": acc.id or f"call_{idx}",
            "type": "function",
            "function": {"name": acc.name, "arguments": acc.arguments},
        })

    # Post-loop: the stall timer is stopped by the finally above.
    return result


def normalize_provider_response(data: Any, provider: str = "",
                                for_stream: bool = False) -> dict:
    """Return a defensive, canonical OpenAI-compatible response object.

    Gateways vary widely in the shape of ``choices``, usage, content and
    tool calls. This boundary copies rather than mutates input, normalizes
    every nested value, and raises APIError (never AttributeError) for a
    non-object response envelope.
    """
    if not isinstance(data, dict):
        label = f" from {provider}" if provider else ""
        raise APIError(f"provider{label} returned unexpected response shape "
                       f"({type(data).__name__}): {str(data)[:200]}")
    model = _strict_str(data.get("model"))
    usage = data.get("usage")
    usage = dict(usage) if isinstance(usage, dict) else None
    raw_choices = data.get("choices")
    if isinstance(raw_choices, dict):
        raw_choices = [raw_choices]
    if not isinstance(raw_choices, list):
        raw_choices = []

    def normalize_message(raw: Any, streaming: bool) -> dict:
        source = raw if isinstance(raw, dict) else {}
        result = {
            "content": _coerce_text(source.get("content")),
            "reasoning": _coerce_text(source.get("reasoning_content")
                                       or source.get("reasoning")),
            "tool_calls": [],
        }
        # Preserve the spelling used by the provider as well as the
        # canonical short key consumed by the stream loop.
        result["reasoning_content"] = result["reasoning"]
        raw_calls = source.get("tool_calls")
        if not isinstance(raw_calls, list):
            raw_calls = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function")
            if isinstance(function, str):
                try:
                    decoded = json.loads(function)
                    function = decoded if isinstance(decoded, dict) else {
                        "name": function}
                except (ValueError, TypeError):
                    function = {"name": function}
            if not isinstance(function, dict):
                function = {}
            if streaming:
                index = raw_call.get("index", 0)
                try:
                    index = int(index)
                except (TypeError, ValueError, OverflowError):
                    index = 0
                result["tool_calls"].append({
                    "index": index,
                    "id": _strict_str(raw_call.get("id")),
                    "type": "function",
                    "function": {
                        "name": _strict_str(function.get("name")),
                        "arguments": _coerce_arguments(
                            function.get("arguments")),
                    },
                })
            else:
                normalized = normalize_tool_call(raw_call)
                if normalized is not None:
                    result["tool_calls"].append(normalized)
        return result

    choices: list[dict] = []
    for raw_choice in raw_choices:
        choice = raw_choice if isinstance(raw_choice, dict) else {}
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            finish_reason = _strict_str(finish_reason)
        normalized_choice = {
            "index": choice.get("index", 0),
            "finish_reason": finish_reason,
            "message": normalize_message(choice.get("message"), False),
            "delta": normalize_message(choice.get("delta"), True),
        }
        choices.append(normalized_choice)
    normalized = {"model": model, "usage": usage, "choices": choices}
    # Preserve provider error envelopes and useful top-level metadata.
    if "error" in data:
        normalized["error"] = data["error"]
    for key in ("id", "created", "object"):
        if key in data:
            normalized[key] = data[key]
    return normalized


def _result_from_json(data: dict, model_id: str) -> StreamResult:
    """Parse a non-streamed chat.completion JSON body into a StreamResult
    (shared by blocking mode and the stream-mode JSON fallback)."""
    # Providers occasionally return a JSON array/string instead of the
    # chat.completion object — without this guard `data.get` raises a raw
    # AttributeError that bypasses every error handler upstream.
    # Provider boundary (typeguards): central entry guard; the explicit
    # shape check keeps the APIError contract for non-dict payloads.
    if not isinstance(data, dict):
        raise APIError(
            f"provider returned unexpected JSON shape "
            f"({type(data).__name__}): {str(data)[:200]}")
    data = normalize_provider_response(data, for_stream=False)
    # HARDEN (provider shapes): some gateways return
    # HTTP 200 with an error ENVELOPE — {"error": {"message": ...,
    # "code": 429}} or {"error": "bare string"} — instead of an error
    # status. Without this check the error silently becomes an EMPTY
    # StreamResult (content="", no tool calls) and the agent loops on
    # ghost turns. Every error shape is normalized to a clean APIError
    # with a string message, same contract as the streaming event path.
    if data.get("error"):
        err = data["error"]
        _envelope_msg = _error_obj_message(err)
        if isinstance(err, dict):
            _envelope_msg = _with_error_type(_envelope_msg, err)
        raise APIError(_envelope_msg,
                       status=_sse_error_status(err))
    result = StreamResult(
        model=(_strict_str(data.get("model"))
               or _typeguards.ensure_str(model_id)),
        usage=(data.get("usage") if isinstance(
            data.get("usage"), dict) else None))
    # HARDEN: choices may be a string ("gateway blew up"), a single dict,
    # or a list containing non-dicts — indexing a string yields a str
    # whose .get() raises AttributeError: 'str' object has no attribute
    # 'get' (the reported malformed-call crash). Coerce every level; malformed
    # entries are skipped, never fatal.
    raw_choices = data.get("choices")
    if isinstance(raw_choices, dict):
        # some gateways send a single choice object instead of a list
        raw_choices = [raw_choices]
    choices = _typeguards.ensure_list(raw_choices)
    choice = choices[0] if choices else None
    choice = choice if isinstance(choice, dict) else {}
    if choice:
        # HARDEN: message may itself be a string ("just a string") —
        # .get() on it would raise AttributeError the same way.
        msg = choice.get("message")
        msg = msg if isinstance(msg, dict) else {}
        # HARDEN: content as Anthropic-style blocks ([{"type": "text",
        # "text": "hi"}]) or a bare int must never reach the str-typed
        # field (TypeError downstream) — blocks join on text parts.
        result.content = _coerce_text(msg.get("content"))
        result.reasoning = _coerce_text(msg.get("reasoning_content")
                                       or msg.get("reasoning"))
        # HARDEN: finish_reason as a non-str (int/dict from a sloppy
        # provider) would violate the StreamResult field type — strict
        # str-or-empty, matching StreamResult.normalize().
        _fr = choice.get("finish_reason")
        result.finish_reason = (None if _fr is None else _strict_str(_fr))
        # Canonical tool-call normalization: strings/None/ints are
        # dropped, function-as-JSON-string is parsed, dict arguments
        # become JSON strings, missing ids get unique fallbacks, and
        # nameless entries are dropped. One helper, one shape — a
        # malformed entry must never reach the turn loop as a raw dict
        # (AttributeError: 'str' object has no attribute ...).
        for tc in _typeguards.ensure_list(msg.get("tool_calls")):
            norm = normalize_tool_call(tc)
            if norm is not None:
                result.tool_calls.append(norm)
    return result


def _post_blocking(url: str, headers: dict, payload: dict,
                   timeout: float,
                   should_cancel: Callable[[], bool] | None = None,
                   on_status: Callable[[str], None] | None = None) -> dict:
    """POST with the same retry policy as the streaming path (rate limits,
    timeouts, connection errors). Blocking calls have no partial output,
    so every attempt is replay-safe — unlike the stream path there is no
    emitted-output guard here.

    CANCEL: the old version blocked with zero cancel checks — a hung
    provider held the turn for the full timeout (300s) per attempt with
    Esc doing nothing. The POST, the body reads and the backoff sleeps
    are all cancel-guarded now; Esc raises TurnCancelled within ~0.25s.

    The response is ALWAYS closed (try/finally): leaking it would pin a
    pooled connection and eventually starve the session pool.

    Transient non-rate-limit failures share one deadline. HTTP 429 responses
    retry until success or cancellation, and each resumed request receives a
    fresh per-attempt timeout."""
    last_error: Exception | None = None
    deadline = time.monotonic() + timeout
    attempt = 0
    rate_attempt = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise APITimeoutError(
                f"request timed out after {timeout:g}s") from last_error
        try:
            resp = run_cancellable(
                lambda: _http().post(url, headers=headers, json=payload,
                                     timeout=_timeouts(min(timeout, remaining))),
                should_cancel, name="model request")
        except TurnCancelled:
            raise
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                _sleep_capped_cancellable(_backoff(attempt), deadline,
                                          should_cancel)
                attempt += 1
                continue
            raise APITimeoutError(
                f"request timed out after {timeout:g}s") from e
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                _sleep_capped_cancellable(_backoff(attempt), deadline,
                                          should_cancel)
                attempt += 1
                continue
            raise APIError(f"connection failed: {e}") from e
        except requests.exceptions.RequestException as e:
            # Same contract as the streaming path: anything else requests
            # can raise (TooManyRedirects, InvalidURL, InvalidHeader, ...)
            # surfaces as a clean APIError, never a raw requests
            # exception.
            raise APIError(f"request failed: {e}") from e
        try:
            if resp.status_code != 200:
                body = run_cancellable(lambda: resp.text, should_cancel,
                                       name="error body")
                retry_after = _parse_retry_after(
                    resp.headers.get("Retry-After"))
                raise APIError(_extract_error_message(body, resp.status_code),
                               status=resp.status_code,
                               retry_after=retry_after)
            try:
                data = run_cancellable(resp.json, should_cancel,
                                       name="model response")
            except TurnCancelled:
                raise
            except ValueError as e:
                raise APIError(
                    f"provider returned invalid JSON "
                    f"(status {resp.status_code}): {str(e)[:200]}",
                    status=resp.status_code) from e
            if not isinstance(data, dict):
                raise APIError(
                    f"provider returned unexpected JSON shape "
                    f"({type(data).__name__}): {str(data)[:200]}",
                    status=resp.status_code)
            return data
        except TurnCancelled:
            raise
        except APIError as e:
            if _is_rate_limited(e):
                _wait_for_rate_limit(e, rate_attempt, should_cancel, on_status)
                rate_attempt += 1
                attempt = 0
                deadline = time.monotonic() + timeout
                continue
            if (e.status in RETRY_STATUSES and not _is_permanent_error(e)
                    and attempt < MAX_RETRIES - 1):
                _sleep_capped_cancellable(_backoff(attempt), deadline,
                                          should_cancel)
                attempt += 1
                continue
            raise
        finally:
            resp.close()
    raise APIError(str(last_error))  # unreachable, keeps type checkers calm


def chat_blocking(provider: Provider, model: Model, effort: Effort,
                  messages: list[dict], tools: list[dict] | None,
                  on_overflow: Callable[[], bool] | None = None,
                  timeout: float = config.DEFAULT_TIMEOUT,
                  should_cancel: Callable[[], bool] | None = None,
                  on_status: Callable[[str], None] | None = None
                  ) -> StreamResult:
    """Non-streaming fallback (used when a provider rejects stream=true).

    Carries the same three-layer context-overflow recovery as chat_stream:
    pre-flight clamp, re-clamp-and-retry on rejection, and (when the input
    itself is too big) caller-driven shrink-and-retry via on_overflow."""
    _check_api_key(provider, model)
    url = provider.base_url.rstrip("/") + "/chat/completions"
    headers = _base_headers(provider, stream=False)
    current_effort = effort
    shrinks_used = 0

    for overflow_attempt in range(OVERFLOW_RETRIES + OVERFLOW_SHRINKS + 1):
        sent_text = _prompt_payload_text(messages, tools)
        try:
            payload = build_payload(model, current_effort, messages, tools,
                                    stream=False)
        except APIError as e:
            # pre-flight refusal (input already over the window). Give the
            # caller a chance to shrink the conversation and retry.
            if (is_context_overflow(str(e)) and on_overflow is not None
                    and shrinks_used < OVERFLOW_SHRINKS and on_overflow()):
                shrinks_used += 1
                continue
            raise
        try:
            data = _post_blocking(url, headers, payload, timeout,
                                  should_cancel=should_cancel,
                                  on_status=on_status)
        except APIError as err:
            if _is_reasoning_content_error(err):
                if _sanitize_messages(messages):
                    continue
                raise APIError(
                    f"{err} — history was sanitized but provider still "
                    f"rejects. Try /new or /rewind.") from err
            healed = _heal_overflow(model, current_effort, err,
                                    overflow_attempt, sent_text)
            if healed is not None:
                current_effort = healed
                continue
            if (is_context_overflow(str(err)) and on_overflow is not None
                    and shrinks_used < OVERFLOW_SHRINKS and on_overflow()):
                shrinks_used += 1
                continue
            raise
        result = _result_from_json(data, model.id)
        _learn_from_usage(result.usage, sent_text, model.id)
        return result

    raise APIError(
        f"conversation is too large for {model.label} even after "
        f"re-clamping — start a new session (/new) or rewind (/rewind)")


def chat_with_retry(provider: Provider, model: Model, effort: Effort,
                    messages: list[dict], tools: list[dict] | None,
                    timeout: float, should_cancel=None,
                    on_status: Callable[[str], None] | None = None) -> StreamResult:
    """Use blocking chat; keep retrying rate limits until success/cancel."""
    rate_attempt = 0
    while True:
        try:
            return chat_blocking(
                provider, model, effort, messages, tools,
                on_overflow=lambda: shrink_tool_outputs(messages),
                timeout=timeout, should_cancel=should_cancel,
                on_status=on_status)
        except APIError as err:
            if not _is_rate_limited(err):
                raise
            _wait_for_rate_limit(err, rate_attempt, should_cancel, on_status)
            rate_attempt += 1


# ---------------------------------------------------------------------------
# Self-test: prove connection reuse (run: python -m fullagent.client)
# Spins up a local HTTPS server with a self-signed cert, counts accepted
# TCP connections server-side (1 accept = 1 TLS handshake), then compares:
#   BEFORE: 3 one-shot requests.get() calls  -> 3 handshakes
#   AFTER:  3 pooled shared_session() calls   -> 1 handshake
# Also stress-tests thread-safe singleton construction from 8 threads.
# ---------------------------------------------------------------------------
def _self_test_lazy_imports() -> None:
    """Prove the import-time fix: `requests` (~1.4s cold) must NOT be in
    sys.modules after importing this module, yet must resolve on first
    real use through the _LazyRequests proxy.
    Hermetic: runs in a FRESH interpreter via subprocess, so it passes no
    matter how many HTTP-touching self-tests ran before it in this process
    (e.g. the connection-reuse test above imports requests for real)."""
    import subprocess as _sp
    import sys as _sys
    from pathlib import Path as _Path

    repo_root = str(_Path(__file__).resolve().parent.parent)
    code = (
        "import sys, fullagent.client as c; "
        "assert 'requests' not in sys.modules, 'requests leaked at import'; "
        "assert 'urllib3' not in sys.modules, 'urllib3 leaked at import'; "
        "s = c.shared_session(); "
        "assert 'requests' in sys.modules, 'lazy import did not fire'; "
        "rq = sys.modules['requests']; "
        "assert isinstance(s, rq.Session), type(s); "
        "assert s is c.shared_session() is c.shared_session(), 'singleton broken'; "
        "print('lazy-import: requests absent at import, resolves on first use')"
    )
    r = _sp.run([_sys.executable, "-c", code], capture_output=True,
                text=True, cwd=repo_root, timeout=120)
    out = (r.stdout + r.stderr).strip()
    if out:
        print(out)
    assert r.returncode == 0, f"lazy-import self-test failed:\n{out}"


if __name__ == "__main__":  # dead block (a later __main__ wins); kept as reference
    import ssl
    import subprocess
    import tempfile
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn

    try:
        import urllib3 as _u3
        _u3.disable_warnings(_u3.exceptions.InsecureRequestWarning)
    except Exception:
        pass

    class _H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive across requests

        def do_GET(self):
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.server.served += 1

        def log_message(self, *a):
            pass

    class _CountingServer(ThreadingMixIn, HTTPServer):
        # Threaded: keep-alive connections stay open in handler threads, so
        # the serve loop keeps polling and shutdown() never deadlocks.
        daemon_threads = True

        def __init__(self, *a, ctx=None, **k):
            self._ctx = ctx
            self.accepted = 0
            self.served = 0
            self._alock = threading.Lock()
            super().__init__(*a, **k)

        def get_request(self):
            conn, addr = self.socket.accept()
            conn = self._ctx.wrap_socket(conn, server_side=True)
            with self._alock:
                self.accepted += 1
            return conn, addr

    def _mk_server():
        d = tempfile.mkdtemp()
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048",
             "-keyout", f"{d}/key.pem", "-out", f"{d}/cert.pem",
             "-days", "1", "-nodes", "-subj", "/CN=127.0.0.1"],
            capture_output=True, check=True)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(f"{d}/cert.pem", f"{d}/key.pem")
        srv = _CountingServer(("127.0.0.1", 0), _H, ctx=ctx)
        return srv

    def _handshake_count_for(make_calls):
        srv = _mk_server()
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            url = f"https://127.0.0.1:{srv.server_port}/"
            start = time.monotonic()
            make_calls(url)
            elapsed = time.monotonic() - start
            time.sleep(0.2)
            return srv.accepted, srv.served, elapsed
        finally:
            srv.shutdown()

    def _one_shot_calls(url):
        for _ in range(3):
            r = requests.get(url, verify=False, timeout=(5, 10))
            r.text

    def _pooled_calls(url):
        s = shared_session()
        for _ in range(3):
            r = s.get(url, verify=False, timeout=(5, 10))
            r.text

    print("== connection-reuse self-test ==")
    acc1, served1, t1 = _handshake_count_for(_one_shot_calls)
    print(f"BEFORE (one-shot requests.get x3): {acc1} TLS handshakes, "
          f"{served1} requests served, {t1:.2f}s")
    acc2, served2, t2 = _handshake_count_for(_pooled_calls)
    print(f"AFTER  (pooled shared_session x3): {acc2} TLS handshake(s), "
          f"{served2} requests served, {t2:.2f}s")
    assert served1 == 3 and served2 == 3, "server must see all 3 requests"
    assert acc1 == 3, f"one-shot calls should open 3 connections, got {acc1}"
    assert acc2 == 1, f"pooled calls should reuse 1 connection, got {acc2}"
    print(f"handshakes: 3 -> 1 ({acc1 - acc2} saved); "
          f"time {t1:.2f}s -> {t2:.2f}s")

    # thread-safety: 8 threads racing to build the singleton must all get
    # the same session, and concurrent requests must all succeed.
    _SESSION = None
    seen, errors = [], []
    srv = _mk_server()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"https://127.0.0.1:{srv.server_port}/"

    def _worker():
        try:
            s = shared_session()
            seen.append(id(s))
            r = s.get(url, verify=False, timeout=(5, 10))
            assert r.status_code == 200 and r.text == "ok"
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=60)
    srv.shutdown()

    assert not errors, f"thread errors: {errors}"
    assert len(seen) == 8 and len(set(seen)) == 1, \
        f"all threads must share one session, got {len(set(seen))} distinct"
    print("thread-safety: 8 threads -> 1 shared session, 8/8 requests OK")
    print("PASS: connection reuse verified (3 handshakes -> 1)")


if __name__ == "__main__":
    # -- token-count + pre-check self-test (worker 17/20) ---------------------
    # Proves: (1) repeated counts of the SAME object are ~free (identity
    # tier: no serialization, no hashing) vs the uncached path;
    # (2) counts are unchanged by the cache (same math as before);
    # (3) mutations (append/pop/in-place truncate, tool-call edits)
    # recount correctly; (4) calibration updates invalidate the cache;
    # (5) the approximation stays within a sane bound of a chars/4-style
    # real-tokenizer guess; (6) the raw estimator beats flat chars/4
    # (tiktoken when installed, else the BPE-aware heuristic);
    # (7) MODEL_CONTEXT_WINDOWS holds researched values with a 128k
    # default; (8) build_payload prunes history FIRST when input +
    # max_tokens would exceed the window, logs it, and never sends a
    # request it can prove won't fit.
    import tempfile as _tempfile
    import time as _t
    from pathlib import Path as _Path

    _self_test_lazy_imports()  # import-time perf: requests must stay lazy

    # Hermetic calibration: the test learns corrections, so point the
    # calibration file at a temp dir — never touch the user's real one,
    # and stay deterministic across reruns.
    _CALIBRATION_FILE = (_Path(_tempfile.mkdtemp(prefix="toktest_"))
                         / "calibration.json")
    _calibration_loaded = False
    _correction_samples.clear()
    _learned_windows.clear()
    _token_cache.clear()
    _id_cache.clear()

    fails: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(("PASS" if cond else "FAIL"), "-", name)
        if not cond:
            fails.append(name)

    MID = "selftest-model-xyz"

    def _raw_path(obj, model_id=MID):  # uncached: same math, no cache
        p = json.dumps(obj, ensure_ascii=False, default=str)
        return max(1, int(_raw_token_estimate(p)
                          * _correction_factor(model_id)))

    # -- fixtures: ~100KB prompt and ~1MB history (the SAME objects are
    #    reused across calls, exactly like agent.messages) ----------------
    line = "def foo(bar):\n    return bar + 1  # some python code\n"
    prompt100k = [{"role": "user", "content": line * 1800}]
    hist1m = [{"role": "user", "content": line * 1800} for _ in range(10)]
    size100k = len(json.dumps(prompt100k, ensure_ascii=False))
    size1m = len(json.dumps(hist1m, ensure_ascii=False))
    print(f"fixtures: prompt={size100k // 1024}KB history={size1m // 1024}KB")

    def _bench(fn, iters=20):
        t0 = _t.perf_counter()
        for _ in range(iters):
            fn()
        return (_t.perf_counter() - t0) / iters * 1000.0

    # -- 1. identical results with/without cache -------------------------
    check("count == raw path (100KB)",
          estimate_tokens(prompt100k, MID) == _raw_path(prompt100k))
    check("count == raw path (1MB)",
          estimate_tokens(hist1m, MID) == _raw_path(hist1m))

    # -- 2. speed: uncached path vs identity-tier repeated counts ---------
    t_old_100k = _bench(lambda: _raw_path(prompt100k))
    estimate_tokens(prompt100k, MID)          # warm: miss populates tier 1
    t_hit_100k = _bench(lambda: estimate_tokens(prompt100k, MID), 500)
    print(f"100KB: raw path {t_old_100k:.2f} ms/call, "
          f"identity hit {t_hit_100k * 1000:.1f} us/call")
    check("100KB repeated count >= 50x faster",
          t_hit_100k * 50 < t_old_100k)

    t_old_1m = _bench(lambda: _raw_path(hist1m))
    estimate_tokens(hist1m, MID)             # warm
    t_hit_1m = _bench(lambda: estimate_tokens(hist1m, MID), 500)
    print(f"1MB:   raw path {t_old_1m:.2f} ms/call, "
          f"identity hit {t_hit_1m * 1000:.1f} us/call")
    check("1MB repeated count >= 50x faster", t_hit_1m * 50 < t_old_1m)
    check("1MB hit under 1ms", t_hit_1m < 1.0)

    big_text = line * 1800
    estimate_tokens_fast(big_text, MID)     # warm
    t_fast = _bench(lambda: estimate_tokens_fast(big_text, MID), 500)
    print(f"fast text path hit: {t_fast * 1000:.1f} us/call")
    check("fast path hit under 50us", t_fast < 0.05)

    # -- 3. mutations recount correctly ------------------------------------
    msgs = [{"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there"}]
    n0 = estimate_tokens(msgs, MID)
    msgs.append({"role": "user", "content": "another question here"})
    n1 = estimate_tokens(msgs, MID)
    check("append changes count", n1 > n0)
    msgs.pop()
    n2 = estimate_tokens(msgs, MID)
    check("pop restores original count", n2 == n0)
    msgs[0]["content"] = "hello world, this is a much longer message now"
    n3 = estimate_tokens(msgs, MID)
    check("in-place content edit changes count", n3 > n0)
    check("in-place edit count is exact", n3 == _raw_path(msgs))
    msgs2 = [{"role": "assistant", "content": "",
              "tool_calls": [{"id": "1", "type": "function",
                              "function": {"name": "read",
                                           "arguments": '{"p": "x"}'}}]}]
    a = estimate_tokens(msgs2, MID)
    msgs2[0]["tool_calls"][0]["function"]["arguments"] = '{"p": "x' * 500
    b = estimate_tokens(msgs2, MID)
    check("tool-call arg growth changes count", b > a)
    check("tool-call count exact", b == _raw_path(msgs2))

    # -- 4. calibration learn invalidates the cache -------------------------
    before = estimate_tokens(prompt100k, MID)
    raw = _raw_token_estimate(json.dumps(prompt100k, ensure_ascii=False))
    learn_token_correction(raw_tokens=raw, actual_tokens=raw // 2,
                           model_id=MID)  # backend says: half the raw guess
    after = estimate_tokens(prompt100k, MID)
    check("learned correction changes the count (epoch bump)",
          after < before)
    check("learned count ~= half of raw", abs(after - before // 2) <= 2)
    check("learned count still sane", after > 0)
    _correction_samples.clear()  # hermetic: don't leak into later checks
    _bump_calibration_epoch()

    # -- 5. approximation within sane bounds ---------------------------------
    guess = size100k / 4.0
    now = estimate_tokens(prompt100k, MID)
    check("within 4x of chars/4 guess", guess / 4.0 <= now <= guess * 4.0)
    check("never more tokens than chars", now <= size100k)
    check("never fewer than chars/16", now >= size100k / 16.0)

    # -- 6. raw estimator quality --------------------------------------------
    _samples = {
        "prose": ("The quick brown fox jumps over the lazy dog. " * 40),
        "code": ("def foo(bar):\n    return bar + 1  # comment\n" * 40),
        "json": ('{"role": "user", "content": "hello world"}' * 60),
        "cjk": ("中文测试内容" * 60),
    }
    for _name, _text in _samples.items():
        _h = _heuristic_token_count(_text)
        _tk = _tiktoken_count(_text)
        if _tk is not None:
            check(f"heuristic within 2x of tiktoken ({_name})",
                  _tk / 2.0 <= _h <= _tk * 2.0)
            print(f"  {_name}: heuristic={_h} tiktoken={_tk}")
        else:
            _g = len(_text) / 4.0
            check(f"heuristic within 4x of chars/4 ({_name})",
                  _g / 4.0 <= _h <= _g * 4.0)
            print(f"  {_name}: heuristic={_h} chars/4={_g:.0f} (no tiktoken)")
    check("tiktoken path never raises", True)

    # -- 7. caches stay bounded ----------------------------------------------
    for i in range(_TOKEN_CACHE_MAX + 200):
        estimate_tokens_fast(f"unique-content-{i}", MID)
    check("digest cache evicts oldest (bounded)",
          len(_token_cache) <= _TOKEN_CACHE_MAX)
    for i in range(_ID_CACHE_MAX + 20):
        estimate_tokens([{"role": "user", "content": f"m{i}"}], MID)
    check("identity cache bounded", len(_id_cache) <= _ID_CACHE_MAX)

    # -- 8. degenerate inputs never crash --------------------------------------
    for weird in ("", [], {}, None, 0):
        try:
            n = estimate_tokens(weird, MID)
            check(f"weird {weird!r} -> >=1", n >= 1)
        except Exception as e:  # noqa: BLE001
            check(f"weird {weird!r} no crash ({e})", False)

    # -- 9. per-model window table ----------------------------------------------
    check("DeepSeek V4.1 Flash Free window is 1048576",
          config.model_context_window("deepseek-v4.1-flash-free") == 1_048_576)
    check("unknown model defaults to 128k",
          config.model_context_window("nope-xyz") == 128_000)
    check("every shipped model has a sane window",
          all(m.context_window >= 128_000 for m in config.MODELS))
    check("table covers every shipped model",
          all(m.id in config.MODEL_CONTEXT_WINDOWS for m in config.MODELS))

    # -- 10. pre-check: prune FIRST, never send a doomed request ------------------
    _tiny = Model(id="selftest-tiny", provider="kios", label="Tiny",
                  context_window=8_000)
    _eff = Effort(key="t", label="T", color="", max_tokens=1_000,
                  temperature=0.0, reasoning_effort=None, description="")
    _sys = {"role": "system", "content": "sys"}
    _hist = [_sys] + [{"role": "user", "content": f"question {i} " + "x" * 500}
                      for i in range(90)]
    _hist.append({"role": "user", "content": "LATEST-USER-MESSAGE"})
    _n_before = _input_tokens(_tiny, _hist, None)
    check("fixture actually overflows the tiny window",
          _n_before + _context_margin(8_000) + _n_before // 32
          + 1_000 > 8_000)
    _payload = build_payload(_tiny, _eff, _hist, None, stream=False)
    _n_after = _input_tokens(_tiny, _payload["messages"], None)
    check("build_payload pruned instead of raising", True)
    check("pruned history is smaller", len(_payload["messages"]) < 92)
    check("system prompt survives the prune",
          _payload["messages"][0].get("role") == "system")
    check("latest user message survives the prune",
          _payload["messages"][-1].get("content") == "LATEST-USER-MESSAGE")
    check("request provably fits: input + max_tokens <= window",
          _n_after + _payload["max_tokens"] <= 8_000)
    check("input estimate returned sane", _n_after > 0)

    # -- 11. unshrinkable payload raises pre-send (never a doomed send) ------------
    _huge = [_sys, {"role": "user", "content": "LATEST " + "z" * 60_000}]
    try:
        build_payload(_tiny, _eff, _huge, None, stream=False)
        check("unfittable payload raises APIError", False)
    except APIError as e:
        check("unfittable payload raises APIError", True)
        check("error recognised as context overflow (on_overflow path)",
              is_context_overflow(str(e)))

    print()
    if fails:
        print(f"{len(fails)} FAILURES")
        raise SystemExit(1)
    print("ALL TOKEN-COUNT SELF-TESTS PASS")



if __name__ == "__main__":
    # -- timeout fail-fast self-test ---------------------------------------
    # Proves a hung model endpoint fails fast with a clear APITimeoutError
    # instead of hanging forever. Spins local TCP servers:
    #   1. black hole (accepts the connection, never responds) ->
    #      chat_blocking raises APITimeoutError in ~timeout seconds, not
    #      MAX_RETRIES x DEFAULT_TIMEOUT (900s+) of silence;
    #   2. stalled SSE (headers sent, then silence) -> the stall watchdog
    #      aborts the stream (exercised with a 2s watchdog);
    #   3. healthy SSE (keepalive comment + event + [DONE]) -> still
    #      completes (regression guard for the watchdog path).
    # Run:  python3 -m fullagent.client
    _self_test_lazy_imports()  # import-time perf: requests must stay lazy

    import socket as _socket

    _OLD_WORST = MAX_RETRIES * config.DEFAULT_TIMEOUT  # 900s+ of silence

    def _st_serve(_handler):
        srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        srv.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)
        port = srv.getsockname()[1]

        def _read_headers(conn):
            conn.settimeout(10)
            data = b""
            try:
                while b"\r\n\r\n" not in data:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                if b"\r\n\r\n" in data:
                    head, rest = data.split(b"\r\n\r\n", 1)
                    cl = 0
                    for line in head.split(b"\r\n")[1:]:
                        if line.lower().startswith(b"content-length:"):
                            try:
                                cl = int(line.split(b":", 1)[1].strip())
                            except ValueError:
                                cl = 0
                    while len(rest) < cl:
                        chunk = conn.recv(min(65536, cl - len(rest)))
                        if not chunk:
                            break
                        rest += chunk
            except OSError:
                pass

        def _loop():
            while True:
                try:
                    conn, _ = srv.accept()
                except OSError:
                    return
                _read_headers(conn)
                threading.Thread(target=_handler, args=(conn,),
                                 daemon=True).start()

        threading.Thread(target=_loop, daemon=True).start()
        return srv, port

    def _st_black_hole(conn):
        try:
            time.sleep(3600)  # accept, then never respond — not even headers
        except OSError:
            pass
        finally:
            conn.close()

    def _st_stalled_sse(conn):
        try:
            conn.sendall(b"HTTP/1.1 200 OK\r\n"
                         b"Content-Type: text/event-stream\r\n"
                         b"Cache-Control: no-cache\r\n"
                         b"Connection: keep-alive\r\n\r\n")
            time.sleep(3600)  # headers, then silence forever
        except OSError:
            pass
        finally:
            conn.close()

    def _st_healthy_sse(conn):
        try:
            conn.sendall(b"HTTP/1.1 200 OK\r\n"
                         b"Content-Type: text/event-stream\r\n"
                         b"Connection: close\r\n\r\n"
                         b": ping\n\n"
                         b'data: {"model":"t","choices":[{"delta":{"content":"hi"},'
                         b'"finish_reason":"stop"}]}\n\n'
                         b"data: [DONE]\n\n")
        except OSError:
            pass
        finally:
            conn.close()

    def _st_mk(base_url):
        return (Provider(key="t", name="t", base_url=base_url,
                         api_key="dummy", color="red"),
                Model(id="t", provider="t", label="t"),
                Effort(key="t", label="T", color="red", max_tokens=None,
                       temperature=0.0, reasoning_effort=None,
                       description="t"))

    _st_failures: list[str] = []
    _st_msgs = [{"role": "user", "content": "hi"}]

    def _st_check(name, fn, expect_timeout, limit):
        t0 = time.monotonic()
        try:
            out = fn()
        except APITimeoutError as e:
            dt = time.monotonic() - t0
            ok = expect_timeout and dt < limit
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: "
                  f"APITimeoutError after {dt:.1f}s — {e}")
            if not ok:
                _st_failures.append(name)
            return
        except Exception as e:  # noqa: BLE001 — any other error is a failure
            dt = time.monotonic() - t0
            print(f"[FAIL] {name}: unexpected {type(e).__name__} "
                  f"after {dt:.1f}s — {e}")
            _st_failures.append(name)
            return
        dt = time.monotonic() - t0
        if expect_timeout or getattr(out, "content", "") != "hi":
            print(f"[FAIL] {name}: unexpectedly succeeded "
                  f"({getattr(out, 'content', out)!r}) after {dt:.1f}s")
            _st_failures.append(name)
        else:
            print(f"[PASS] {name}: healthy stream -> {out.content!r} "
                  f"in {dt:.1f}s")

    print("== timeout fail-fast self-test ==")
    print(f"BEFORE: hung endpoint cost up to {MAX_RETRIES} x "
          f"{config.DEFAULT_TIMEOUT:g}s = {_OLD_WORST:g}s+ of silence")
    print(f"AFTER:  connect <={CONNECT_TIMEOUT:g}s, stalled SSE "
          f"<={STREAM_STALL_TIMEOUT:g}s, total per call <={config.DEFAULT_TIMEOUT:g}s")
    print()

    _srv1, _p1 = _st_serve(_st_black_hole)
    _pr, _mo, _ef = _st_mk(f"http://127.0.0.1:{_p1}")
    _st_check("black-hole blocking (timeout=6)",
              lambda: chat_blocking(_pr, _mo, _ef, _st_msgs, None, timeout=6),
              True, 60)
    _srv1.close()

    _saved_stall = STREAM_STALL_TIMEOUT
    globals()["STREAM_STALL_TIMEOUT"] = 2.0
    _srv2, _p2 = _st_serve(_st_stalled_sse)
    _pr, _mo, _ef = _st_mk(f"http://127.0.0.1:{_p2}")
    try:
        _st_check("stalled SSE stream (watchdog=2s)",
                  lambda: chat_stream(_pr, _mo, _ef, _st_msgs, None,
                                      timeout=30),
                  True, 20)
    finally:
        globals()["STREAM_STALL_TIMEOUT"] = _saved_stall
        _srv2.close()

    _srv3, _p3 = _st_serve(_st_healthy_sse)
    _pr, _mo, _ef = _st_mk(f"http://127.0.0.1:{_p3}")
    _st_check("healthy SSE stream",
              lambda: chat_stream(_pr, _mo, _ef, _st_msgs, None, timeout=30),
              False, 30)
    _srv3.close()

    print()
    if _st_failures:
        print(f"SELF-TEST FAILED: {', '.join(_st_failures)}")
        raise SystemExit(1)
    print("ALL TIMEOUT SELF-TESTS PASS")
