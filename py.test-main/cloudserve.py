#!/usr/bin/env python3
"""FullAgent Cloud Server.

Run your FullAgent on a VPS; drive it from anywhere (e.g. your phone)
with the tiny `fac` client. ALL compute happens here on the server —
the client only sends your prompt and streams the reply back.

    python3 cloudserve.py [--port 18791] [--host 127.0.0.1]

Endpoints (token auth required on all):
    GET  /health                        -> {"ok": true, ...}
    POST /chat  {"prompt": "..."}       -> text/event-stream (SSE)
        events: token | reasoning | tool | status | done | error

Auth:  Authorization: Bearer <token>
       (or ?token=<token> for convenience)
       The token auto-generates on first run and lives at
       ~/.fullagent-cloud/token (mode 600). Anyone holding the
       server URL + token can run commands as the server user —
       keep both private.

Security model: binds 127.0.0.1 by default; expose it with a tunnel
(cloudflared) or VPN. Tool calls are AUTO-APPROVED — this is your
personal server, not a multi-user service.

Conversation state persists in FullAgent's normal event log, so a
restart keeps your history. One turn runs at a time (a second
concurrent POST gets 429).
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import queue
import secrets
import socketserver
import sys
import threading
import urllib.parse

CLOUD_DIR = os.path.expanduser("~/.fullagent-cloud")
TOKEN_FILE = os.path.join(CLOUD_DIR, "token")

_agent = None
_agent_lock = threading.Lock()
_turn_lock = threading.Lock()


def get_token() -> str:
    os.makedirs(CLOUD_DIR, exist_ok=True)
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            tok = f.read().strip()
        if tok:
            return tok
    tok = secrets.token_urlsafe(32)
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(tok + "\n")
    os.chmod(TOKEN_FILE, 0o600)
    return tok


def _ensure_working_model(cfg) -> str:
    """Fresh machines default to stealth/union-alpha (kilo), which has no
    embedded key — every turn would fail with "no API key". Fall back to
    a free model whose key is embedded, so the server works zero-config.
    Override with FULLAGENT_CLOUD_MODEL=<model-id>."""
    from fullagent.config import MODELS, PROVIDERS
    by_id = {m.id: m for m in MODELS}

    def has_key(model_id: str) -> bool:
        m = by_id.get(model_id)
        return bool(m and PROVIDERS.get(m.provider)
                    and PROVIDERS[m.provider].api_key)

    wanted = os.environ.get("FULLAGENT_CLOUD_MODEL", "").strip()
    if wanted and has_key(wanted):
        cfg.model_id = wanted
        return wanted
    current = getattr(cfg, "model_id", "")
    if has_key(current):
        return current
    for mid in ("deepseek-v4.1-flash-free", "minimax-m3.1-flash-free"):
        if has_key(mid):
            cfg.model_id = mid
            return mid
    return current  # last resort: leave it, turns will report the key error


def get_agent():
    """Build (once) the shared Agent. Fail fast at startup."""
    global _agent
    with _agent_lock:
        if _agent is None:
            from fullagent import config
            from fullagent.agent import Agent
            from fullagent.config import Config

            config.ensure_dirs()
            cfg = Config.load()
            model = _ensure_working_model(cfg)
            _agent = Agent(cfg)
            print(f"[cloud] model: {model}", flush=True)
        return _agent


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "FullAgentCloud/1.0"

    def log_message(self, fmt, *args):  # quieter logs
        msg = fmt % args
        # redact query-string tokens (e.g. ?token=...) from request logs
        if "token=" in msg:
            import re
            msg = re.sub(r"token=[^ &]+", "token=***", msg)
        sys.stderr.write("[cloud] " + msg + "\n")

    # -- helpers ------------------------------------------------------
    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _path(self) -> str:
        return urllib.parse.urlparse(self.path).path

    def _authorized(self) -> bool:
        tok = get_token()
        if not tok:
            return False

        def _eq(candidate: str, expected: str) -> bool:
            try:
                return (len(candidate) == len(expected)
                        and secrets.compare_digest(candidate, expected))
            except TypeError:  # non-ASCII header/query value
                return False

        auth = self.headers.get("Authorization", "")
        if _eq(auth, "Bearer " + tok):
            return True
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        cand = qs.get("token", [""])[0]
        return _eq(cand, tok)

    # -- routes -------------------------------------------------------
    def do_GET(self):
        if self._path() == "/health":
            if not self._authorized():
                return self._send_json(401, {"ok": False, "error": "unauthorized"})
            return self._send_json(200, {"ok": True, "service": "fullagent-cloud"})
        return self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self._path() != "/chat":
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if 0 < n <= 65536:
                    self.rfile.read(n)
            except Exception:
                pass
            return self._send_json(404, {"ok": False, "error": "not found"})
        if not self._authorized():
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if 0 < n <= 65536:
                    self.rfile.read(n)
            except Exception:
                pass
            return self._send_json(401, {"ok": False, "error": "unauthorized"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            return self._send_json(400, {"ok": False, "error": "invalid Content-Length"})
        if length < 0 or length > 10 * 1024 * 1024:
            return self._send_json(413, {"ok": False, "error": "payload too large or negative length"})
        try:
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw_body or b"{}")
        except Exception:
            return self._send_json(400, {"ok": False, "error": "bad json"})
        if not isinstance(data, dict):
            return self._send_json(400, {"ok": False, "error": "json body must be an object"})
        raw_prompt = data.get("prompt")
        if raw_prompt is not None and not isinstance(raw_prompt, str):
            return self._send_json(400, {"ok": False, "error": "prompt must be a string"})
        prompt = (raw_prompt or "").strip()
        if not prompt:
            return self._send_json(400, {"ok": False, "error": "empty prompt"})
        if not _turn_lock.acquire(blocking=False):
            return self._send_json(
                429, {"ok": False, "error": "another turn is running, try again"})
        try:
            self._stream_chat(prompt)
        finally:
            _turn_lock.release()

    # -- SSE streaming ------------------------------------------------
    def _stream_chat(self, prompt: str) -> None:
        q: "queue.Queue[tuple]" = queue.Queue()
        DONE = object()
        cancelled = threading.Event()

        def _run():
            try:
                agent = get_agent()

                def _noop(*a, **k):
                    pass

                turn = agent.run_turn(
                    prompt,
                    on_token=lambda t: q.put(("token", t)),
                    on_reasoning=lambda r: q.put(("reasoning", r)),
                    on_tool_call=lambda e: q.put(
                        ("tool", getattr(e, "name", "?"))),
                    on_tool_update=_noop,
                    on_status=lambda s: q.put(("status", s)),
                    approve=lambda tool, args: True,
                    should_cancel=cancelled.is_set,
                )
                q.put(("done", json.dumps({
                    "error": turn.error or "",
                    "model": turn.model_id,
                    "duration": round(turn.duration or 0, 2),
                })))
            except Exception as e:  # noqa: BLE001 — report, don't crash
                q.put(("error", f"{type(e).__name__}: {e}"))
            finally:
                q.put((DONE, ""))

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def _write(payload: bytes) -> bool:
            try:
                self.wfile.write(payload)
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError):
                return False

        try:
            while True:
                try:
                    etype, payload = q.get(timeout=15)
                except queue.Empty:
                    if not _write(b": ping\n\n"):  # keepalive
                        return
                    continue
                if etype is DONE:
                    return
                line = "data: %s\n\n" % json.dumps(
                    {"type": etype, "text": payload}, ensure_ascii=False)
                if not _write(line.encode("utf-8")):
                    return
        finally:
            cancelled.set()
            worker.join(timeout=5.0)
            # SSE has no Content-Length: the client only sees EOF when we
            # close. Without this the connection stays open (keep-alive)
            # and the client hangs after DONE.
            self.close_connection = True


class ThreadedServer(socketserver.ThreadingMixIn,
                     http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    try:
        default_port = int(os.environ.get("FULLAGENT_CLOUD_PORT", "18791"))
    except (TypeError, ValueError):
        # a typo'd env var must not crash startup (argparse would also
        # raise before --help could even run)
        default_port = 18791
    if not 0 <= default_port <= 65535:
        default_port = 18791
    # an empty $FULLAGENT_CLOUD_HOST made the server bind 0.0.0.0 (all
    # interfaces); empty/malformed means "use the safe loopback default"
    default_host = (
        os.environ.get("FULLAGENT_CLOUD_HOST", "").strip() or "127.0.0.1")
    ap = argparse.ArgumentParser(description="FullAgent Cloud Server")
    ap.add_argument("--port", type=int, default=default_port)
    ap.add_argument("--host", default=default_host)
    args = ap.parse_args()

    tok = get_token()
    print(f"[cloud] token: {tok}", flush=True)
    get_agent()  # fail fast if the agent can't initialise
    print(f"[cloud] agent ready; listening on {args.host}:{args.port}",
          flush=True)

    with ThreadedServer((args.host, args.port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
