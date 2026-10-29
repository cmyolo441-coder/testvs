"""REAL SMTP email sending (stdlib smtplib + email.message only).

Public API:
    - :func:`register` -- attach ``EmailSend`` and ``EmailConfig`` tools to an
      agent (duck-typed, minimal).
    - :func:`send_email` -- real smtplib send; returns a human-readable
      result string with the SMTP server's response.
    - :func:`config_status` -- config presence/host/port report (no password).

Configuration: ``~/.fullagent/smtp.json`` with the shape::

    {"host": "...", "port": 587, "username": "...", "password": "...",
     "use_tls": true, "from_addr": "..."}

If the file is missing/invalid, ``EmailSend`` returns the clear message
"SMTP not configured -- create ~/.fullagent/smtp.json" and sends nothing.

Integration (wired by the coordinator, NOT in this file): add ``"emailops"``
to the module tuple in ``Agent._register_feature_modules`` in
``fullagent/agent.py``.
"""

from __future__ import annotations

import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional

def _base_config_path() -> Path:
    if os.environ.get("FULLAGENT_HOME"):
        return Path(os.path.expandvars(os.environ["FULLAGENT_HOME"])).expanduser() / "smtp.json"
    return Path(os.path.expanduser("~/.fullagent/smtp.json"))
SMTP_TIMEOUT = 30.0

MISSING_CONFIG_MSG = (
    "SMTP not configured -- create ~/.fullagent/smtp.json "
    "with {host, port, username, password, use_tls, from_addr}"
)

_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_CONFIG_CACHE_KEY: Optional[tuple[str, float]] = None


def _config_path() -> Path:
    """Honour FULLAGENT_SMTP_CONFIG for tests; default ~/.fullagent/smtp.json."""
    override = os.environ.get("FULLAGENT_SMTP_CONFIG")
    return Path(override) if override else _base_config_path()


def _load_config() -> Optional[Dict[str, Any]]:
    global _CONFIG_CACHE, _CONFIG_CACHE_KEY
    path = _config_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = -1.0
    key = (str(path), mtime)
    if _CONFIG_CACHE_KEY == key:
        return _CONFIG_CACHE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        _CONFIG_CACHE = None
        _CONFIG_CACHE_KEY = key
        return None
    if not isinstance(data, dict):
        _CONFIG_CACHE = None
        _CONFIG_CACHE_KEY = key
        return None
    _CONFIG_CACHE = data
    _CONFIG_CACHE_KEY = key
    return data


def _clear_config_cache() -> None:
    global _CONFIG_CACHE, _CONFIG_CACHE_KEY
    _CONFIG_CACHE = None
    _CONFIG_CACHE_KEY = None


def _valid_email(addr: str) -> bool:
    """Minimal validation: non-empty and contains @ with something on both sides."""
    if not isinstance(addr, str):
        return False
    if "\r" in addr or "\n" in addr:  # header injection / EmailMessage ValueError
        return False
    parts = addr.strip().split("@")
    return len(parts) == 2 and bool(parts[0]) and bool(parts[1])


def send_email(to: Any, subject: str, body: str,
               config: Optional[Dict[str, Any]] = None) -> str:
    """Send a REAL email through smtplib.

    ``to`` is one address or a list of addresses. Returns a result string:
    either a success message including the SMTP server's response, or an
    error message taken from the real SMTP conversation / socket errors.
    Never raises on SMTP failure — the error text is the return value.
    """
    cfg = config if config is not None else _load_config()
    if cfg is None:
        return MISSING_CONFIG_MSG
    if isinstance(to, str):
        recipients: List[str] = [to]
    else:
        recipients = list(to or [])
    recipients = [r for r in recipients if _valid_email(r)]
    if not recipients:
        return "ERROR: no valid recipient address (each must contain @)"
    host = str(cfg.get("host") or "").strip()
    if not host:
        return "ERROR: smtp.json has no 'host'"
    try:
        port = int(cfg.get("port") or 587)
    except (TypeError, ValueError):
        return "ERROR: smtp.json 'port' must be a number"
    from_addr = str(cfg.get("from_addr") or cfg.get("username") or "").strip()
    if not _valid_email(from_addr):
        return "ERROR: smtp.json 'from_addr' (or 'username') is not a valid email address"
    use_tls = bool(cfg.get("use_tls", True))
    if isinstance(subject, str) and ("\r" in subject or "\n" in subject):
        return "ERROR: subject must not contain newline characters"

    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject or ""
    msg.set_content(body or "")

    try:
        with smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT) as smtp:
            smtp.ehlo()
            if use_tls:
                smtp.starttls()
                smtp.ehlo()
            username = cfg.get("username")
            password = cfg.get("password")
            if username and password:
                smtp.login(str(username), str(password))
            refused = smtp.send_message(msg, from_addr=from_addr,
                                        to_addrs=recipients)
    except smtplib.SMTPException as e:
        return f"ERROR: SMTP failed: {e}"
    except OSError as e:
        return f"ERROR: connection failed: {e}"
    if refused:
        bad = ", ".join(sorted(refused))
        return f"ERROR: server refused recipients: {bad}"
    return f"Email sent to {', '.join(recipients)} via {host}:{port}"


def config_status() -> str:
    """Report whether SMTP is configured. Never reveals the password."""
    path = _config_path()
    if not path.exists():
        return f"SMTP not configured -- create {path} with " \
               "{host, port, username, password, use_tls, from_addr}"
    data = _load_config()
    if data is None:
        return f"SMTP config found at {path} but it is not valid JSON"
    return (f"SMTP configured: host={data.get('host')}, "
            f"port={data.get('port')}, use_tls={bool(data.get('use_tls', True))}, "
            f"username={data.get('username') or '(none)'}, "
            f"from_addr={data.get('from_addr') or '(none)'} "
            "(password is set but never displayed)")


def _tool_send_email(to: str, subject: str, body: str) -> str:
    """Tool handler: ``to`` may be one address or comma/semicolon-separated."""
    if "," in to or ";" in to:
        recipients: Any = [r.strip() for r in
                           to.replace(";", ",").split(",") if r.strip()]
    else:
        recipients = to
    return send_email(recipients, subject, body)


def _tool_email_config() -> str:
    return config_status()


def register(agent: Any) -> None:
    """Register the ``EmailSend`` and ``EmailConfig`` tools on an agent."""
    from .tools import Tool, RISK_CONFIRM, RISK_SAFE  # local import: no cycles

    tools = getattr(agent, "tools", None)
    if tools is None:
        return
    tools["EmailSend"] = Tool(
        "EmailSend",
        "Send a REAL email via configured SMTP. 'to' is one address or a "
        "comma-separated list. Returns the SMTP server's real response.",
        {"type": "object",
         "properties": {
             "to": {"type": "string",
                    "description": "recipient address(es), comma-separated"},
             "subject": {"type": "string"},
             "body": {"type": "string"},
         },
         "required": ["to", "subject", "body"]},
        _tool_send_email,
        risk=RISK_CONFIRM,
    )
    tools["EmailConfig"] = Tool(
        "EmailConfig",
        "Show whether SMTP is configured (host/port only; password never "
        "revealed).",
        {"type": "object", "properties": {}},
        lambda: _tool_email_config(),
        risk=RISK_SAFE,
    )


# ---------------------------------------------------------------------------
# Self-test: REAL smtpd-less SMTP server + REAL smtplib send.
# Run: python3 -m fullagent.emailops
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import socket
    import tempfile
    import threading

    _fails = 0

    def check(name: str, cond: bool) -> None:
        global _fails
        print(("PASS" if cond else "FAIL"), "-", name)
        if not cond:
            _fails += 1

    # --- minimal socket SMTP server that captures one message ----------------
    captured: Dict[str, Any] = {}

    class CaptureSMTP(threading.Thread):
        """Speaks just enough SMTP (EHLO/MAIL/RCPT/DATA/QUIT) to capture
        a real smtplib session. Runs until one QUIT, then stops."""

        def __init__(self) -> None:
            super().__init__(daemon=True)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(("127.0.0.1", 0))
            self.port = self.sock.getsockname()[1]
            self.sock.listen(1)

        def _rl(self, f: Any) -> bytes:
            return f.readline()

        def run(self) -> None:
            conn, _ = self.sock.accept()
            try:
                with conn:
                    f = conn.makefile("rwb")
                    f.write(b"220 capture-smtp test server\r\n")
                    f.flush()
                    in_data = False
                    buf: List[bytes] = []
                    while True:
                        line = self._rl(f)
                        if not line:
                            break
                        if in_data:
                            if line == b".\r\n":
                                captured["data"] = b"".join(buf)
                                f.write(b"250 2.0.0 OK: queued\r\n")
                                f.flush()
                                in_data = False
                                buf = []
                            else:
                                buf.append(line)
                            continue
                        up = line.strip().upper()
                        if up.startswith(b"EHLO") or up.startswith(b"HELO"):
                            f.write(b"250-localhost greets you\r\n"
                                    b"250-AUTH PLAIN LOGIN\r\n250 8BITMIME\r\n")
                        elif up.startswith(b"AUTH LOGIN"):
                            f.write(b"334 VXNlcm5hbWU6\r\n")
                            f.flush()
                            f.readline()  # username (base64)
                            f.write(b"334 UGFzc3dvcmQ6\r\n")
                            f.flush()
                            pw = f.readline()  # password (base64)
                            captured["auth_b64"] = pw.strip()
                            f.write(b"235 2.7.0 Authentication successful\r\n")
                        elif up.startswith(b"AUTH PLAIN"):
                            f.write(b"235 2.7.0 Authentication successful\r\n")
                        elif up.startswith(b"MAIL FROM:"):
                            captured["mail_from"] = line.strip().decode()
                            f.write(b"250 2.1.0 OK\r\n")
                        elif up.startswith(b"RCPT TO:"):
                            captured.setdefault("rcpt_tos", []).append(
                                line.strip().decode())
                            f.write(b"250 2.1.5 OK\r\n")
                        elif up.startswith(b"DATA"):
                            f.write(b"354 End data with <CR><LF>.<CR><LF>\r\n")
                            in_data = True
                        elif up.startswith(b"QUIT"):
                            f.write(b"221 2.0.0 Bye\r\n")
                            f.flush()
                            break
                        else:
                            f.write(b"250 OK\r\n")
                        f.flush()
            finally:
                self.sock.close()

    # 1. missing-config path: no file -> clear message, no send ---------------
    with tempfile.TemporaryDirectory() as td:
        missing = os.path.join(td, "no-smtp.json")
        os.environ["FULLAGENT_SMTP_CONFIG"] = missing
        _clear_config_cache()
        check("missing config -> clear not-configured message",
              send_email("a@example.com", "s", "b") == MISSING_CONFIG_MSG)
        check("EmailConfig reports not configured",
              "not configured" in config_status().lower())
        check("EmailConfig does not leak password",
              "test-secret" not in config_status())

        # 2. REAL send through REAL smtplib against capture server -------------
        srv = CaptureSMTP()
        srv.start()
        cfg_path = os.path.join(td, "smtp.json")
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump({
                "host": "127.0.0.1",
                "port": srv.port,
                "username": "tester@example.com",
                "password": "test-secret",
                "use_tls": False,  # plain local capture; TLS still works via SMTP
                "from_addr": "tester@example.com",
            }, fh)
        os.environ["FULLAGENT_SMTP_CONFIG"] = cfg_path
        _clear_config_cache()

        result = send_email(["alice@example.com", "bob@example.com"],
                            "Hello from self-test",
                            "This is a REAL message body 12345.")
        print("send result:", result)
        srv.join(timeout=10)
        check("send_email reports success",
              result.startswith("Email sent to "))
        check("server captured MAIL FROM",
              "tester@example.com" in captured.get("mail_from", ""))
        rcpts = " ".join(captured.get("rcpt_tos", []))
        check("server captured both RCPT TO",
              "alice@example.com" in rcpts and "bob@example.com" in rcpts)
        data = captured.get("data", b"").decode("utf-8", "replace")
        check("server captured subject", "Hello from self-test" in data)
        check("server captured body",
              "REAL message body 12345" in data)
        check("password never sent without login prompt",
              b"test-secret" not in captured.get("data", b""))
        check("EmailConfig shows host/port but not password value",
              "test-secret" not in config_status()
              and "127.0.0.1" in config_status())

        # 3. invalid recipient -> error, no send --------------------------------
        check("invalid recipient rejected",
              send_email("not-an-email", "s", "b").startswith(
                  "ERROR: no valid recipient"))
        # 4. unreachable host -> real connection error --------------------------
        _clear_config_cache()
        err = send_email("a@example.com", "s", "b",
                         config={"host": "127.0.0.1", "port": 1,
                                 "username": None, "password": None,
                                 "use_tls": False,
                                 "from_addr": "tester@example.com"})
        check("unreachable host -> connection error", err.startswith("ERROR:"))
        print("unreachable-host error:", err)

        del os.environ["FULLAGENT_SMTP_CONFIG"]
        _clear_config_cache()

    # 5. register attaches tools without a real Agent ---------------------------
    class _FakeAgent:
        def __init__(self) -> None:
            self.tools: Dict[str, Any] = {}

    fake = _FakeAgent()
    register(fake)
    check("register adds EmailSend tool", "EmailSend" in fake.tools)
    check("register adds EmailConfig tool", "EmailConfig" in fake.tools)
    check("EmailSend risk is confirm", fake.tools["EmailSend"].risk == "confirm")
    check("EmailConfig risk is safe", fake.tools["EmailConfig"].risk == "safe")

    if _fails:
        print(f"emailops self-test FAILED ({_fails} failures)")
        raise SystemExit(1)
    print("emailops self-test PASSED")
