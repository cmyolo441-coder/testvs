"""Tiny streaming HTTP client on top of urllib.

Only what a chat completion needs: POST JSON, read server-sent events line by
line, and translate HTTP failures into :class:`ProviderError` subclasses the
router can act on.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Iterator, Mapping, Optional, Tuple

from ..errors import AuthError, ProviderError, RateLimitError


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Treat redirects as errors so a wrong base URL fails fast."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _error_body(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", "replace")
    except Exception:  # pragma: no cover - body already consumed
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()[:400]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)[:400]
        if isinstance(error, str):
            return error[:400]
        if payload.get("message"):
            return str(payload["message"])[:400]
    return raw.strip()[:400]


def map_http_error(exc: urllib.error.HTTPError, provider: str) -> ProviderError:
    status = int(getattr(exc, "code", 0) or 0)
    body = _error_body(exc)
    retry_after: Optional[float] = None
    header = exc.headers.get("Retry-After") if exc.headers else None
    if header:
        try:
            retry_after = float(header)
        except ValueError:
            retry_after = None

    detail = f"{provider} returned HTTP {status}" + (f": {body}" if body else "")
    if status in {401, 403}:
        return AuthError(detail, hint="check the API key for this provider")
    if status == 429:
        return RateLimitError(detail, retry_after=retry_after)
    if status in {400, 404, 405, 422}:
        return ProviderError(detail, status=status, retryable=False)
    return ProviderError(detail, status=status, retry_after=retry_after, retryable=True)


def post_sse(
    url: str,
    payload: Mapping[str, object],
    headers: Mapping[str, str],
    *,
    timeout: float = 120.0,
    provider: str = "provider",
) -> Iterator[Tuple[Optional[str], str]]:
    """POST JSON and yield ``(event, data)`` pairs from the SSE stream."""
    body = json.dumps(payload).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": "omh/1.0",
        **headers,
    }
    request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
    try:
        response = _OPENER.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise map_http_error(exc, provider) from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise ProviderError(
            f"cannot reach {urllib.parse.urlsplit(url).netloc}: {reason}",
            retryable=True,
        ) from exc

    event: Optional[str] = None
    try:
        with response:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
                if not line:
                    event = None
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event = line[6:].strip()
                    continue
                if line.startswith("data:"):
                    yield event, line[5:].strip()
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
        raise ProviderError(f"stream interrupted: {exc}", retryable=True) from exc


def sse_json(events: Iterator[Tuple[Optional[str], str]], provider: str) -> Iterator[Tuple[Optional[str], Dict[str, object]]]:
    """Decode SSE data payloads, tolerating keep-alives and bad frames."""
    for event, data in events:
        if not data:
            continue
        if data == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"{provider} sent a malformed stream frame: {data[:120]}", retryable=False
            ) from exc
        if isinstance(payload, dict) and isinstance(payload.get("error"), (dict, str)):
            error = payload["error"]
            message = error.get("message") if isinstance(error, dict) else error
            raise ProviderError(f"{provider} error: {message}", retryable=False)
        if isinstance(payload, dict):
            yield event, payload
