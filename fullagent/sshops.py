"""SSH remote execution — real remote commands via the system OpenSSH client.

Two tools:

* ``SshRun`` — run a command on a remote host over SSH.
* ``SshTest`` — test SSH connectivity to a host (``ssh ... echo OK``).

No paramiko, no third-party dependency: the system ``ssh`` binary is used
through :mod:`subprocess` (no shell, argument list only). Every call uses
``-o BatchMode=yes`` so SSH never blocks on a password/key-passphrase
prompt, and ``-o ConnectTimeout=...`` so unreachable hosts fail fast
instead of hanging. Stdout, stderr and the exit code are always captured;
a connection is never faked — when it fails you get the real error plus
guidance that key-based authentication must be set up.

Requires key-based auth on the target host (``ssh-copy-id user@host`` or
an ssh-agent with the key). If keys are not set up, the tools fail fast
with a clear message saying so — they never prompt and never hang.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

from .tools import RISK_CONFIRM, RISK_SAFE, Tool

_CONNECT_TIMEOUT_TEST = 5      # seconds for SshTest
_CONNECT_TIMEOUT_RUN = 10      # seconds for SshRun
_MAX_RUN_SECONDS = 120        # hard ceiling per SshRun invocation

_AUTH_HINT = (
    "SSH authentication failed (no keys accepted / no agent). "
    "This tool requires key-based auth: run `ssh-copy-id [user@]host` "
    "or start an ssh-agent and add your key (`ssh-add`), then retry."
)

_CONNECTION_HINT = (
    "SSH connection failed. Verify the host and port are reachable and "
    "that sshd is running; common causes are connection refused, a DNS "
    "failure, or a network timeout."
)


def _ssh_binary() -> Optional[str]:
    return shutil.which("ssh")


def _validate_target(host: Any, user: Any, port: Any) -> str:
    """Validate user-supplied host/user/port; return the ssh target string.

    Raises ValueError with a human-readable message on bad input.
    """
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host is required (non-empty string)")
    host = host.strip()
    if any(c.isspace() for c in host):
        raise ValueError(f"invalid host {host!r}: must not contain whitespace")
    if host.startswith("-"):
        raise ValueError(f"invalid host {host!r}: must not start with '-'")
    if user is not None and user != "":
        if not isinstance(user, str):
            raise ValueError("user must be a string")
        user = user.strip()
        if not user or any(c.isspace() for c in user) or user.startswith("-"):
            raise ValueError(f"invalid user {user!r}")
        host = f"{user}@{host}"
    if port is not None and port != "":
        try:
            port_n = int(port)
        except (TypeError, ValueError):
            raise ValueError(f"invalid port {port!r}: must be a number")
        if not 1 <= port_n <= 65535:
            raise ValueError(f"invalid port {port_n}: out of range 1-65535")
        return host, port_n
    return host, None


def _base_args(connect_timeout: int) -> List[str]:
    return [
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={connect_timeout}",
    ]


def _is_auth_failure(exit_code: int, stderr: str) -> bool:
    text = stderr.lower()
    return exit_code == 255 and (
        "permission denied" in text
        or "no more authentication" in text
        or "publickey" in text and "authentication failed" in text
    )


def _format_result(host: str, command: str, completed: subprocess.CompletedProcess) -> str:
    out = completed.stdout.strip()
    err = completed.stderr.strip()
    code = completed.returncode
    if code == 0:
        return (f"[{host}] exit 0\n"
                f"$ {command}\n{out}" if out else f"[{host}] exit 0 — no output")
    hint = (f"\n{_AUTH_HINT}" if _is_auth_failure(code, err)
            else f"\n{_CONNECTION_HINT}" if code == 255 else "")
    return (f"[{host}] ERROR: ssh exited with code {code}\n"
            f"$ {command}\n"
            f"stdout: {out or '(empty)'}\n"
            f"stderr: {err or '(empty)'}"
            f"{hint}")


def _run_ssh(host: str, user: Any, port: Any, remote_command: List[str],
             connect_timeout: int, total_timeout: int) -> str:
    binary = _ssh_binary()
    if binary is None:
        return ("ERROR: no ssh client found on PATH — "
                "install OpenSSH to use the Ssh tools.")
    try:
        target, port_n = _validate_target(host, user, port)
    except ValueError as e:
        return f"ERROR: {e}"
    argv = [binary, *_base_args(connect_timeout)]
    if port_n is not None:
        argv += ["-p", str(port_n)]
    argv += ["--", target, *remote_command]
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True,
            timeout=total_timeout)
    except subprocess.TimeoutExpired:
        return (f"[{host}] ERROR: ssh timed out after {total_timeout}s — "
                f"host unreachable or remote command hung.")
    except OSError as e:
        return f"[{host}] ERROR: failed to start ssh: {e}"
    return _format_result(host, " ".join(remote_command), completed)


def ssh_run(host: str, command: str, user: Optional[str] = None,
            port: Optional[Any] = None) -> str:
    """Run ``command`` on ``host`` via ssh (real OpenSSH subprocess)."""
    if not isinstance(command, str) or not command.strip():
        return "ERROR: command is required (non-empty string)"
    return _run_ssh(host, user, port, [command],
                    _CONNECT_TIMEOUT_RUN, _MAX_RUN_SECONDS)


def ssh_test(host: str, user: Optional[str] = None,
             port: Optional[Any] = None) -> str:
    """Test SSH connectivity: ``ssh [user@]host echo OK`` (real)."""
    result = _run_ssh(host, user, port, ["echo", "OK"],
                      _CONNECT_TIMEOUT_TEST, _CONNECT_TIMEOUT_TEST + 15)
    if result.startswith("[") and "exit 0" in result.splitlines()[0]:
        return f"[{host}] OK — ssh connectivity works (key-based auth in place)"
    return result


_SSHRUN_PARAMETERS = {
    "type": "object",
    "properties": {
        "host": {"type": "string",
                 "description": "Hostname or IP to connect to"},
        "command": {"type": "string",
                    "description": "Shell command to run on the remote host"},
        "user": {"type": "string",
                 "description": "Remote user (default: current user)"},
        "port": {"type": "integer",
                 "description": "SSH port (default: 22)"},
    },
    "required": ["host", "command"],
}

_SSHTEST_PARAMETERS = {
    "type": "object",
    "properties": {
        "host": {"type": "string",
                 "description": "Hostname or IP to test connectivity to"},
        "user": {"type": "string",
                 "description": "Remote user (default: current user)"},
        "port": {"type": "integer",
                 "description": "SSH port (default: 22)"},
    },
    "required": ["host"],
}

_SSHRUN_DESCRIPTION = (
    "Run a shell command on a remote host over SSH (real OpenSSH, no "
    "paramiko). Requires key-based auth on the host. Fails fast with a "
    "clear error if keys are missing — never prompts for a password. "
    "Args: host (required), command (required), user, port."
)

_SSHTEST_DESCRIPTION = (
    "Test SSH connectivity to a host by running `echo OK` over SSH "
    "(real OpenSSH). Reports success only when the connection truly "
    "succeeds. Args: host (required), user, port."
)


def build_tools() -> List[Tool]:
    return [
        Tool("SshRun", _SSHRUN_DESCRIPTION, _SSHRUN_PARAMETERS,
             ssh_run, risk=RISK_CONFIRM),
        Tool("SshTest", _SSHTEST_DESCRIPTION, _SSHTEST_PARAMETERS,
             ssh_test, risk=RISK_SAFE),
    ]


def register(agent: Any) -> None:
    """Register the SshRun and SshTest tools on an agent's tool dict."""
    for tool in build_tools():
        agent.tools[tool.name] = tool


if __name__ == "__main__":
    def check(name: str, cond: bool, detail: str = "") -> None:
        print(("PASS" if cond else "FAIL"), "-", name,
              (f"({detail})" if detail else ""))
        if not cond:
            raise SystemExit(f"self-test failed: {name}")

    # 1. Tools build correctly
    tools = build_tools()
    by_name = {t.name: t for t in tools}
    check("builds SshRun + SshTest", set(by_name) == {"SshRun", "SshTest"})
    check("SshRun is confirm-risk", by_name["SshRun"].risk == RISK_CONFIRM)

    # 2. Argument validation (no ssh invoked)
    r = ssh_run("", "echo hi")
    check("empty host rejected", r.startswith("ERROR:") and "host is required" in r)
    r = ssh_run("localhost", "")
    check("empty command rejected", r.startswith("ERROR:") and "command is required" in r)
    r = ssh_run("-oProxyCommand=evil", "echo hi")
    check("dash-host rejected", "ERROR:" in r)
    r = ssh_run("localhost", "echo hi", port=99999)
    check("bad port rejected", "ERROR:" in r and "port" in r)
    r = ssh_test("")
    check("sshtest empty host rejected", r.startswith("ERROR:"))
    check("openai_schema serializes",
          by_name["SshRun"].openai_schema()["function"]["name"] == "SshRun")

    if _ssh_binary() is None:
        print("SKIP — no ssh client on PATH; remaining tests need real ssh")
        raise SystemExit(0)

    # 3. Real behavior: localhost. Either sshd+keys work (verify real
    #    output) or the connection fails cleanly with a helpful message —
    #    never a hang and never a faked success.
    t0 = time.time()
    r = ssh_run("localhost", "echo hello-ssh-test")
    elapsed = time.time() - t0
    check("localhost ssh returns quickly (<30s)", elapsed < 30,
          f"{elapsed:.1f}s")
    if "exit 0" in r and "hello-ssh-test" in r:
        check("localhost ssh real output verified", True,
              repr(r.splitlines()[0]))
    else:
        check("localhost failure is honest, not fake",
              "ERROR" in r and "ssh" in r.lower())
        helpful = ("key" in r.lower() or "permission denied" in r.lower()
                   or "connection refused" in r.lower()
                   or "timed out" in r.lower()
                   or "host key" in r.lower()
                   or "could not resolve" in r.lower())
        check("localhost failure message is helpful", helpful,
              repr(r[:120]))

    # 4. Invalid host fails fast (< 15s), real timeout not a hang
    bad = "no-such-host-xyz.invalid"
    t0 = time.time()
    r = ssh_test(bad)
    elapsed = time.time() - t0
    check("invalid host fails fast (<15s)", elapsed < 15, f"{elapsed:.1f}s")
    check("invalid host reports error, not fake OK",
          "ERROR" in r and "OK — ssh connectivity works" not in r)

    # 5. register() wires into an agent-like object
    class FakeAgent:
        def __init__(self):
            self.tools: Dict[str, Any] = {}
    fa = FakeAgent()
    register(fa)
    check("register adds both tools",
          set(fa.tools) == {"SshRun", "SshTest"})
    check("registered handlers are callable",
          callable(fa.tools["SshRun"].handler))

    print("sshops self-test: all checks passed")
