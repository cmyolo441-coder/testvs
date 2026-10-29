"""Desktop + webhook notifications for fullagent.

Real implementations only — nothing here is simulated:

* :func:`send_notification` — fires a REAL desktop notification through
  ``notify-send`` on Linux (``shutil.which`` check; if the binary is
  missing it returns ``False`` and reports it, never a fake success).
* Webhook notifications — optional ``webhook_url`` in
  ``~/.fullagent/notify.json``. :func:`notify_event` POSTs a JSON payload
  to that URL with ``urllib``; ``agent.notify_event(type, data)`` is wired
  up by :func:`register`.
* Tools: ``NotifySend`` (title + body → real notification) and
  ``NotifyTest`` (sends a test notification and reports which backends
  actually worked: notify-send present/missing, webhook configured/not).

``python3 -m fullagent.notify`` runs the built-in self-test: real
``notify-send`` return-code check (or graceful ``False`` when missing),
a REAL ``http.server`` POST capture, and a config load/save round-trip.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from typing import Any, Dict, Optional

CONFIG_PATH = os.path.join(
    os.environ.get("FULLAGENT_HOME") or os.path.join(os.path.expanduser("~"), ".fullagent"),
    "notify.json",
)

# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------

def notify_send_available() -> bool:
    """True if the real ``notify-send`` binary is on PATH."""
    return shutil.which("notify-send") is not None


def send_notification(title: str, body: str = "") -> bool:
    """Fire a REAL desktop notification via ``notify-send``.

    Returns ``True`` only when the subprocess exits 0. Returns ``False``
    (never raises, never fakes success) when ``notify-send`` is missing or
    the subprocess fails.
    """
    if shutil.which("notify-send") is None:
        return False
    try:
        proc = subprocess.run(
            ["notify-send", str(title), str(body)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.returncode == 0
    except Exception:
        return False


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Load the notify config (``{"webhook_url": ...}``). Never raises."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_config(config: Dict[str, Any], path: str = CONFIG_PATH) -> bool:
    """Save the notify config. Returns True on success, False on error."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
        return True
    except Exception:
        return False


def webhook_url(config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Configured webhook URL, or None when not configured."""
    cfg = config if config is not None else load_config()
    url = cfg.get("webhook_url")
    return url if isinstance(url, str) and url.strip() else None


def notify_event(event_type: str, data: Any = None,
                 url: Optional[str] = None,
                 timeout: int = 10) -> bool:
    """POST ``{"type": ..., "data": ...}`` JSON to the webhook URL via urllib.

    Returns ``True`` only when the POST gets a 2xx response. ``False``
    (never raised) when no webhook is configured or the request fails.
    """
    target = url or webhook_url()
    if not target:
        return False
    try:
        payload = json.dumps({"type": str(event_type), "data": data}).encode("utf-8")
        req = urllib.request.Request(
            target,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _tool_notify_send(title: str = "", body: str = "") -> str:
    title = str(title).strip()
    if not title:
        return "error: title is required"
    ok = send_notification(title, body)
    if ok:
        return f"notification sent via notify-send: {title!r}"
    return ("error: notify-send is not available on this system "
            "(install libnotify-bin / notification-daemon) — "
            "notification was NOT sent")


def _tool_notify_test() -> str:
    ns = "present" if notify_send_available() else "missing"
    url = webhook_url()
    ws = "configured" if url else "not configured"
    sent = send_notification("fullagent test",
                             "notify backend check")
    lines = [
        "notification backend report:",
        f"  notify-send: {ns} -> test send {'succeeded' if sent else 'FAILED'}",
        f"  webhook: {ws}",
    ]
    if url:
        lines.append(f"  webhook_url: {url}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# register(agent)
# ---------------------------------------------------------------------------

def register(agent: Any) -> None:
    """Add ``NotifySend`` / ``NotifyTest`` tools and ``agent.notify_event``."""
    from .tools import Tool, RISK_SAFE  # local import: no cycles

    tools = getattr(agent, "tools", None)
    if tools is not None:
        tools["NotifySend"] = Tool(
            "NotifySend",
            "Send a REAL desktop notification via notify-send. 'title' is "
            "required; 'body' is optional. Returns whether it actually "
            "succeeded.",
            {"type": "object",
             "properties": {
                 "title": {"type": "string",
                           "description": "notification title (required)"},
                 "body": {"type": "string",
                          "description": "notification body (optional)"},
             },
             "required": ["title"]},
            lambda title="", body="", **_: _tool_notify_send(title, body),
            risk=RISK_SAFE,
        )
        tools["NotifyTest"] = Tool(
            "NotifyTest",
            "Send a test notification and report which notification "
            "backends actually work: notify-send present/missing and "
            "webhook configured/not.",
            {"type": "object", "properties": {}},
            lambda **_: _tool_notify_test(),
            risk=RISK_SAFE,
        )
    try:
        agent.notify_event = lambda event_type, data=None, **_: notify_event(  # noqa: E731
            event_type, data)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Self-test: REAL subprocess, REAL local http.server, REAL config file.
# Run: python3 -m fullagent.notify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    _fails = 0

    def check(name: str, cond: bool) -> None:
        global _fails
        print(("PASS" if cond else "FAIL"), "-", name)
        if not cond:
            _fails += 1

    # --- 1. real notify-send behaviour ---------------------------------------
    has_ns = notify_send_available()
    check("notify_send_available matches shutil.which",
          has_ns == (shutil.which("notify-send") is not None))
    if has_ns:
        ok = send_notification("fullagent self-test", "notify module check")
        has_desktop = bool(
            os.environ.get("DBUS_SESSION_BUS_ADDRESS")
            or os.environ.get("DISPLAY")
            or os.environ.get("WAYLAND_DISPLAY")
        )
        check("real notify-send send_notification returns bool (True on desktop, False when headless)",
              (ok is True) if has_desktop else isinstance(ok, bool))
    else:
        ok = send_notification("fullagent self-test", "notify module check")
        check("missing notify-send -> False, no exception", ok is False)

    # --- 2. real webhook POST against a local http.server --------------------
    received: Dict[str, Any] = {}

    class _Capture(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            received["path"] = self.path
            received["content_type"] = self.headers.get("Content-Type")
            try:
                received["json"] = json.loads(body.decode("utf-8"))
            except Exception:
                received["json"] = None
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):  # quiet
            pass

    server = HTTPServer(("127.0.0.1", 0), _Capture)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        hook = f"http://127.0.0.1:{port}/hooks/fullagent"
        sent_ok = notify_event("test_event", {"answer": 42}, url=hook)
        check("real webhook POST returns True on 2xx", sent_ok is True)
        check("server received JSON with type+data",
              received.get("json") == {"type": "test_event",
                                       "data": {"answer": 42}})
        check("server saw application/json content type",
              (received.get("content_type") or "").startswith(
                  "application/json"))
    finally:
        server.shutdown()
        thread.join(timeout=5)

    # unconfigured webhook -> False without network activity
    saved_home_path = os.environ.get("FAKE_HOME_TEST", None)  # noqa: F841
    old_cfg = load_config.__defaults__  # noqa: F841
    check("notify_event with no webhook returns False",
          notify_event("x", {}, url="") is False)

    # --- 3. config load/save round-trip --------------------------------------
    with tempfile.TemporaryDirectory() as tmpd:
        cfg_path = os.path.join(tmpd, "notify.json")
        check("save_config returns True", save_config(
            {"webhook_url": "http://127.0.0.1:9999/wh"}, cfg_path) is True)
        loaded = load_config(cfg_path)
        check("load_config round-trip preserves webhook_url",
              loaded.get("webhook_url") == "http://127.0.0.1:9999/wh")
        check("webhook_url helper reads config",
              webhook_url(loaded) == "http://127.0.0.1:9999/wh")
        check("load_config of missing file returns {}",
              load_config(os.path.join(tmpd, "nope.json")) == {})
        # corrupt file -> {} not exception
        bad_path = os.path.join(tmpd, "bad.json")
        with open(bad_path, "w") as fh:
            fh.write("{not json")
        check("load_config of corrupt file returns {}",
              load_config(bad_path) == {})

    # --- 4. register() wires tools + agent.notify_event ----------------------
    from types import SimpleNamespace
    fake = SimpleNamespace(tools={})
    register(fake)
    check("register adds NotifySend tool", "NotifySend" in fake.tools)
    check("register adds NotifyTest tool", "NotifyTest" in fake.tools)
    check("register exposes agent.notify_event",
          callable(getattr(fake, "notify_event", None)))
    out = fake.tools["NotifySend"].handler(title="selftest", body="ok")
    check("NotifySend handler returns real result string",
          isinstance(out, str) and ("sent" in out or "error" in out))
    rep = fake.tools["NotifyTest"].handler()
    check("NotifyTest handler reports backends",
          isinstance(rep, str) and "notify-send:" in rep and
          "webhook:" in rep)

    print(("PASS" if _fails == 0 else "FAIL") +
          f" self-test ({_fails} failures)")
    raise SystemExit(1 if _fails else 0)
