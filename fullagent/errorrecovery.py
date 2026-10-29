"""Automatic recovery for provider ``invalid_request_error`` (400).

Worker 18/20 — ERROR RECOVERY.

Before this module, an ``invalid_request_error`` from the provider surfaced
as a dead error box and the whole turn was wasted. ``chat_stream`` now calls
:func:`maybe_recover_invalid_request` from its ``APIError`` handler, which
implements a strict 4-step policy:

1. **diagnose** — capture and log a full payload summary (message count,
   role histogram, sizes, largest messages) plus the first error-severity
   diagnostic from :func:`fullagent.client.validate_history` (worker 12's
   validator) when it is available;
2. **prune** — aggressively drop the oldest half of tool-call/tool-result
   pairs from the history (never the system prompt, never the newest turn),
   then re-pair any strays left behind;
3. **retry ONCE** with the cleaned history (hard cap of 1 auto-retry —
   recovery never loops);
4. **helpful error** — if the retry still fails (or there was nothing safe
   left to prune), raise an :class:`APIError` that explains what was wrong
   and suggests ``/reset`` or rephrasing. Never a bare "invalid request".

The raised error keeps ``status=400`` so the agent's existing 400 fallbacks
(no-tools retry, non-stream retry) still trigger on it. Provider errors do
not switch models automatically; model selection remains under user control.

Design notes:

* No import-time dependency on ``.client`` — the validator and
  ``APIError`` are imported lazily inside the functions (``client.py``
  imports this module from inside ``chat_stream``), so there is no import
  cycle.
* ``prune_history_aggressive`` mutates the list in place. ``chat_stream``
  passes the pruned *view* of the conversation (``agent._prune_for_model``
  builds a fresh list), so the canonical session history is untouched —
  only the retried request sees the cleaned history.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ._foundation import get_logger

_log = get_logger("errorrecovery")

# Hard cap: at most this many automatic invalid_request retries per call.
# Enforced by chat_stream via the `already_retried` flag; kept here as the
# single source of truth for the policy.
MAX_INVALID_REQUEST_RETRIES = 1


def is_invalid_request_error(err: BaseException) -> bool:
    """True when `err` is the provider's 400 invalid-request rejection.

    Context-overflow errors are excluded — they have their own recovery
    ladder in ``chat_stream`` (re-clamp / shrink / compact).
    """
    from .client import APIError, is_context_overflow
    if not isinstance(err, APIError):
        return False
    status = getattr(err, "status", None)
    if status is not None and status != 400:
        return False
    msg = str(err).lower()
    if is_context_overflow(str(err)):
        return False
    return ("invalid_request_error" in msg
            or "invalid request" in msg
            or "invalidrequest" in msg)


def _message_size(m: Any) -> int:
    try:
        return len(json.dumps(m, ensure_ascii=False, default=str))
    except Exception:  # noqa: BLE001 — never let sizing break recovery
        try:
            return len(str(m))
        except Exception:  # noqa: BLE001
            return 0


def payload_summary(messages: list[dict],
                    tools: list[dict] | None = None) -> str:
    """One-line summary of the rejected payload for logs and diagnostics.

    Includes message count, role histogram, total chars, the three largest
    messages, and the first error-severity diagnostic from worker 12's
    :func:`validate_history` when it is importable. Never raises.
    """
    try:
        n = len(messages) if isinstance(messages, list) else 0
        roles: dict[str, int] = {}
        total = 0
        largest: list[tuple[int, int, str]] = []
        if isinstance(messages, list):
            for i, m in enumerate(messages):
                role = (m.get("role") if isinstance(m, dict)
                        else f"<{type(m).__name__}>")
                role = str(role)
                roles[role] = roles.get(role, 0) + 1
                size = _message_size(m)
                total += size
                largest.append((size, i, role))
        largest.sort(key=lambda t: t[0], reverse=True)
        top = ", ".join(f"#{i} {r} {s:,}ch"
                        for s, i, r in largest[:3]) or "none"

        first_invalid = "none"
        try:
            from .client import validate_history
            diags = validate_history(messages) or []
            errs = [d for d in diags
                    if getattr(d, "severity", "") == "error"]
            pick = errs[0] if errs else (diags[0] if diags else None)
            if pick is not None:
                first_invalid = (f"{pick.code} at messages[{pick.index}]: "
                                 f"{pick.detail}")
        except Exception as e:  # noqa: BLE001 — validator optional
            first_invalid = f"validator unavailable ({type(e).__name__})"

        tools_n = len(tools) if isinstance(tools, list) else 0
        return (f"messages={n} roles={roles} total_chars={total:,} "
                f"tools={tools_n} largest=[{top}] "
                f"first_invalid=[{first_invalid}]")
    except Exception as e:  # noqa: BLE001 — summary must never break recovery
        return f"<payload summary failed: {type(e).__name__}: {e}>"


def _repair_tool_pairing(messages: list[dict]) -> int:
    """Drop strays so tool_call/tool-response pairing is never broken.

    Removes assistant ``tool_calls`` entries with no matching tool
    response, and tool messages whose id was never issued by a preceding
    assistant message. Returns the number of messages/entries fixed.
    """
    fixed = 0
    if not isinstance(messages, list):
        return 0
    # ids that have a tool response
    answered: set[str] = set()
    for m in messages:
        if (isinstance(m, dict) and m.get("role") == "tool"
                and m.get("tool_call_id")):
            answered.add(str(m["tool_call_id"]))
    # ids issued by assistant messages (in order)
    issued: list[str] = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls")
        if not tcs:
            continue
        if not isinstance(tcs, (list, tuple)):
            tcs = [tcs]
        kept = [tc for tc in tcs
                if isinstance(tc, dict)
                and str(tc.get("id", "")) in answered]
        if len(kept) != len(list(tcs if isinstance(tcs, list) else [tcs])):
            fixed += 1
        if kept:
            m["tool_calls"] = kept
        else:
            m.pop("tool_calls", None)
        for tc in kept:
            issued.append(str(tc.get("id", "")))
    issued_set = set(issued)
    # drop tool responses that were never issued
    drop = [i for i, m in enumerate(messages)
            if isinstance(m, dict) and m.get("role") == "tool"
            and str(m.get("tool_call_id", "")) not in issued_set]
    for i in sorted(drop, reverse=True):
        del messages[i]
    return fixed + len(drop)


def prune_history_aggressive(messages: list[dict]) -> int:
    """Drop the oldest half of tool-call/tool-result pairs, in place.

    A "pair" is an assistant message carrying ``tool_calls`` plus the tool
    messages that answer it. The system prompt (index 0) and the newest
    pair (the current turn) are never removed. When the history has no
    tool pairs at all, falls back to dropping the oldest user-delimited
    turn unit (again, never the newest). Returns the number of messages
    removed; 0 means there was nothing safe left to prune.
    """
    if not isinstance(messages, list) or len(messages) < 4:
        return 0
    start = (1 if messages and isinstance(messages[0], dict)
             and messages[0].get("role") == "system" else 0)

    # pair groups: assistant-with-tool_calls + following tool messages
    groups: list[list[int]] = []
    i = start
    while i < len(messages):
        m = messages[i]
        if (isinstance(m, dict) and m.get("role") == "assistant"
                and m.get("tool_calls")):
            grp = [i]
            j = i + 1
            while (j < len(messages) and isinstance(messages[j], dict)
                   and messages[j].get("role") == "tool"):
                grp.append(j)
                j += 1
            groups.append(grp)
            i = j
        else:
            i += 1

    drop_idx: list[int] = []
    if len(groups) >= 2:
        # oldest half of the pairs; the newest pair is always kept
        n_drop = max(1, len(groups) // 2)
        for grp in groups[:n_drop]:
            # never drop the newest group even if the math says otherwise
            if grp is groups[-1]:
                continue
            drop_idx.extend(grp)
    else:
        # no tool pairs — drop oldest user-delimited turn units instead
        units: list[list[int]] = []
        cur: list[int] = []
        for k in range(start, len(messages)):
            m = messages[k]
            if isinstance(m, dict) and m.get("role") == "user" and cur:
                units.append(cur)
                cur = []
            cur.append(k)
        if cur:
            units.append(cur)
        if len(units) >= 2:
            n_drop = max(1, (len(units) - 1) // 2)
            for u in units[:n_drop]:
                drop_idx.extend(u)

    if not drop_idx:
        return 0
    removed = len(set(drop_idx))
    for k in sorted(set(drop_idx), reverse=True):
        del messages[k]
    _repair_tool_pairing(messages)
    return removed


def helpful_invalid_request_error(err: BaseException,
                                  messages: list[dict] | None,
                                  tools: list[dict] | None,
                                  pruned: int,
                                  retried: bool):
    """Build the user-facing error for an unrecoverable invalid request.

    Always explains what was wrong and what to do — never a bare
    "invalid request". Keeps ``status=400`` so the agent's existing 400
    fallbacks (no-tools retry, non-stream retry) still trigger on it.
    """
    from .client import APIError
    provider_msg = str(err).strip().replace("\n", " ")
    if len(provider_msg) > 280:
        provider_msg = provider_msg[:280] + "…"
    if retried and pruned > 0:
        tried = (f"I dropped the {pruned} oldest stale message(s) from the "
                 f"request and retried once — the provider still rejected it.")
    elif retried:
        tried = ("I retried once with a cleaned-up request — the provider "
                 "still rejected it.")
    else:
        tried = ("There was nothing safe left to trim from the request, so "
                 "no retry was attempted.")
    return APIError(
        "The provider rejected the request as invalid "
        "(invalid_request_error) — your message wasn't processed, so "
        "nothing was lost.\n"
        f"Provider said: {provider_msg or '(no detail)'}\n"
        f"Diagnostics: {payload_summary(messages or [], tools)}\n"
        f"What I tried: {tried}\n"
        "What you can do: /reset (or /new) for a clean history, or rephrase "
        "your last message more simply. If this repeats right after a tool "
        "call, /rewind to before that tool turn.",
        status=400)


def maybe_recover_invalid_request(err: BaseException,
                                  messages: list[dict],
                                  tools: list[dict] | None = None,
                                  on_status: Callable[[str], None] | None = None,
                                  already_retried: bool = False) -> bool:
    """``chat_stream`` recovery hook for ``invalid_request_error``.

    Returns ``True`` when the caller should rebuild the payload and retry
    (``continue`` the send loop) — history has been pruned in place and a
    user-visible notice was emitted. Returns ``False`` when `err` is not
    an invalid-request rejection (caller falls through to normal handling).
    Raises the helpful error when the single auto-retry is exhausted or
    there was nothing safe left to prune.
    """
    if not is_invalid_request_error(err):
        return False
    summary = payload_summary(messages, tools)
    _log.warning("invalid_request_error — payload summary: %s", summary)
    _log.warning("invalid_request_error — provider message: %.500s", str(err))
    if already_retried:
        # cap = 1 auto-retry: never loop, surface the helpful error
        raise helpful_invalid_request_error(err, messages, tools,
                                            pruned=0, retried=True)
    pruned = (prune_history_aggressive(messages)
              if isinstance(messages, list) else 0)
    if pruned <= 0:
        raise helpful_invalid_request_error(err, messages, tools,
                                            pruned=0, retried=False)
    if on_status is not None:
        try:
            on_status("⚠ provider rejected the request — dropped "
                      f"{pruned} stale message(s), retrying…")
        except Exception:  # noqa: BLE001 — status display must not break retry
            pass
    _log.warning("invalid_request_error — pruned %d message(s), "
                 "retrying once with cleaned history", pruned)
    return True


# ---------------------------------------------------------------------------
# Self-test: fake provider that rejects the first request with
# invalid_request_error (400) then accepts; plus a provider that always
# rejects, plus unit checks for prune + summary. Run with:
#   python -m fullagent.errorrecovery
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import socket as _socket
    import threading as _threading

    from .client import APIError, Provider, chat_stream
    try:
        # worker 12's validator — optional ("if present"); the sprint is
        # mid-refactor and it may be temporarily unimportable
        from .client import validate_history as _vh
    except ImportError:
        _vh = None
    validate_history = _vh
    from .config import Effort, Model

    _fails: list[str] = []

    def _check(name: str, cond: bool, extra: str = "") -> None:
        print(("PASS" if cond else "FAIL"), "-", name,
              (f"({extra})" if extra and not cond else ""))
        if not cond:
            _fails.append(name)

    def _mk_provider(base_url: str):
        return (Provider(key="t", name="t", base_url=base_url, api_key="x",
                         color="red"),
                Model(id="t", provider="t", label="t"),
                Effort(key="t", label="T", color="red", max_tokens=None,
                       temperature=0.0, reasoning_effort=None,
                       description="t"))

    def _tool_pair(i: int) -> list[dict]:
        tid = f"call_{i}"
        return [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": tid, "type": "function",
                             "function": {"name": "read",
                                          "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": tid,
             "content": f"tool result {i} " + "x" * 200},
        ]

    def _history(n_pairs: int) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": "sys"}]
        for i in range(n_pairs):
            msgs.append({"role": "user", "content": f"do task {i}"})
            msgs.extend(_tool_pair(i))
        msgs.append({"role": "user", "content": "final question"})
        return msgs

    class _FakeProvider:
        """Raw-socket fake OpenAI-compatible server.

        mode="recover": first POST -> 400 invalid_request_error, rest ->
        healthy SSE. mode="always400": every POST -> 400.
        Records every request body for assertions.
        """

        def __init__(self, mode: str):
            self.mode = mode
            self.bodies: list[dict] = []
            self.count = 0
            self._srv = _socket.socket()
            self._srv.setsockopt(_socket.SOL_SOCKET,
                                 _socket.SO_REUSEADDR, 1)
            self._srv.bind(("127.0.0.1", 0))
            self._srv.listen(8)
            self.port = self._srv.getsockname()[1]
            self._stop = False
            self._t = _threading.Thread(target=self._loop, daemon=True)
            self._t.start()

        def _read_request(self, conn) -> bytes:
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            head, _, rest = data.partition(b"\r\n\r\n")
            length = 0
            for line in head.decode("latin1").split("\r\n"):
                if line.lower().startswith("content-length:"):
                    length = int(line.split(":", 1)[1].strip())
            while len(rest) < length:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                rest += chunk
            return rest[:length]

        def _loop(self):
            while not self._stop:
                try:
                    conn, _ = self._srv.accept()
                except OSError:
                    return
                try:
                    with conn:
                        raw = self._read_request(conn)
                        self.count += 1
                        try:
                            self.bodies.append(json.loads(raw.decode()))
                        except Exception:  # noqa: BLE001
                            self.bodies.append({})
                        if self.mode == "always400" or self.count == 1:
                            body = json.dumps({
                                "error": {
                                    "message": "Invalid 'messages[3]."
                                               "tool_calls[0]': tool call "
                                               "is malformed",
                                    "type": "invalid_request_error",
                                    "code": "invalid_request_error",
                                }}).encode()
                            conn.sendall(
                                b"HTTP/1.1 400 Bad Request\r\n"
                                b"Content-Type: application/json\r\n"
                                b"Content-Length: " + str(len(body)).encode()
                                + b"\r\nConnection: close\r\n\r\n" + body)
                        else:
                            conn.sendall(
                                b"HTTP/1.1 200 OK\r\n"
                                b"Content-Type: text/event-stream\r\n"
                                b"Connection: close\r\n\r\n"
                                b'data: {"model":"t","choices":[{"delta":'
                                b'{"content":"recovered!"},'
                                b'"finish_reason":"stop"}]}\n\n'
                                b"data: [DONE]\n\n")
                except OSError:
                    pass

        def close(self):
            self._stop = True
            try:
                _socket.create_connection(("127.0.0.1", self.port),
                                          timeout=1).close()
            except OSError:
                pass
            self._srv.close()

    print("== errorrecovery self-test ==")
    print(f"policy: max {MAX_INVALID_REQUEST_RETRIES} auto-retry, "
          "then helpful error\n")

    # -- 1. fake provider rejects once, then accepts: auto-recovery ------
    srv = _FakeProvider("recover")
    pr, mo, ef = _mk_provider(f"http://127.0.0.1:{srv.port}")
    notices: list[str] = []
    msgs = _history(4)  # system + 4 tool-pair turns + final user
    n_before = len(msgs)
    try:
        res = chat_stream(pr, mo, ef, msgs, None, timeout=20,
                          on_status=notices.append)
        _check("recover: got reply after auto-retry",
               res.content == "recovered!", f"content={res.content!r}")
    except Exception as e:  # noqa: BLE001
        _check("recover: got reply after auto-retry", False,
               f"{type(e).__name__}: {e}")
        res = None
    _check("recover: exactly 2 provider requests (1 retry cap)",
           srv.count == 2, f"count={srv.count}")
    if len(srv.bodies) == 2:
        m1 = len(srv.bodies[0].get("messages", []))
        m2 = len(srv.bodies[1].get("messages", []))
        _check("recover: retried request had pruned history",
               m2 < m1, f"{m1} -> {m2}")
        roles2 = [m.get("role") for m in srv.bodies[1]["messages"]]
        _check("recover: system prompt + newest turn preserved",
               roles2[0] == "system"
               and srv.bodies[1]["messages"][-1]["content"]
               == "final question",
               f"roles={roles2}")
    else:
        _check("recover: retried request had pruned history", False,
               f"bodies={len(srv.bodies)}")
        _check("recover: system prompt + newest turn preserved", False, "")
    _check("recover: user-visible retry notice emitted",
           any("retrying" in n for n in notices), f"notices={notices}")
    _check("recover: history actually pruned in place",
           len(msgs) < n_before, f"{n_before} -> {len(msgs)}")
    srv.close()

    # -- 2. fake provider always rejects: helpful error, still capped ----
    srv2 = _FakeProvider("always400")
    pr2, mo2, ef2 = _mk_provider(f"http://127.0.0.1:{srv2.port}")
    msgs2 = _history(2)
    try:
        chat_stream(pr2, mo2, ef2, msgs2, None, timeout=20)
        _check("always400: raised APIError", False, "no exception")
        err_text, err_status = "", None
    except APIError as e:
        err_text, err_status = str(e), e.status
        _check("always400: raised APIError", True)
    except Exception as e:  # noqa: BLE001
        _check("always400: raised APIError", False,
               f"wrong type {type(e).__name__}")
        err_text, err_status = "", None
    _check("always400: status stays 400 (agent fallbacks intact)",
           err_status == 400, f"status={err_status}")
    _check("always400: message explains + suggests /reset",
           "/reset" in err_text and "rephrase" in err_text,
           err_text[:160])
    _check("always400: never a bare 'invalid request'",
           "what you can do" in err_text.lower()
           and len(err_text) > 200, f"len={len(err_text)}")
    _check("always400: retry capped at 1 (2 requests total)",
           srv2.count == 2, f"count={srv2.count}")
    srv2.close()

    # -- 3. non-invalid errors are untouched ------------------------------
    _check("non-invalid error passes through maybe_recover",
           maybe_recover_invalid_request(APIError("boom", status=500),
                                         []) is False)
    _check("overflow-shaped error not treated as invalid_request",
           is_invalid_request_error(
               APIError("maximum context length exceeded", status=400))
           is False)

    # -- 4. unit: prune drops oldest half of tool pairs -------------------
    h = _history(4)
    dropped = prune_history_aggressive(h)
    remaining_pairs = sum(1 for m in h
                          if isinstance(m, dict)
                          and m.get("role") == "assistant"
                          and m.get("tool_calls"))
    _check("prune: oldest half of pairs dropped (4 -> 2)",
           dropped == 4 and remaining_pairs == 2,
           f"dropped={dropped} pairs={remaining_pairs}")
    _check("prune: newest turn + system kept",
           h[0].get("role") == "system"
           and h[-1].get("content") == "final question"
           and h[-2].get("tool_call_id") == "call_3",
           f"tail={[ (m.get('role'), m.get('tool_call_id')) for m in h[-3:]]}")

    # -- 5. unit: summary flags the first invalid entry -------------------
    poisoned = _history(1)
    poisoned.insert(2, {"role": "assistant", "content": "x",
                        "tool_calls": [{"id": "orphan_1", "type": "function",
                                        "function": {"name": "f",
                                                     "arguments": "{}"}}]})
    s = payload_summary(poisoned, [{"type": "function"}])
    _check("summary: counts roles + sizes",
           "messages=" in s and "'assistant': 2" in s, s[:120])
    if validate_history is None:
        print("SKIP - summary: validator's first invalid diagnostic "
              "(validate_history not importable right now)")
    else:
        _check("summary: surfaces validator's first invalid diagnostic",
               "orphan_tool_call" in s, s[-160:])
    _check("summary: never raises on junk",
           isinstance(payload_summary([None, "x", {}]), str))

    # -- 6. unit: nothing prunable -> helpful error immediately -----------
    tiny = [{"role": "user", "content": "hi"}]
    try:
        maybe_recover_invalid_request(
            APIError("invalid_request_error: bad", status=400), tiny)
        _check("tiny history: helpful error raised", False, "no raise")
    except APIError as e:
        _check("tiny history: helpful error raised",
               "/reset" in str(e), str(e)[:120])

    print()
    if _fails:
        print(f"SELF-TEST FAILED: {', '.join(_fails)}")
        raise SystemExit(1)
    print("errorrecovery self-test: ALL PASS")
