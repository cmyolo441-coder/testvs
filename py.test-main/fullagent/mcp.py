"""MCP (Model Context Protocol) client support for fullagent.

Reads server configs from ``~/.fullagent/mcp.json``::

    {"servers": {
         "name": {"command": "...", "args": [...], "env": {...}}},
     "timeout": 30}

Missing file (or bad JSON) = zero servers, never a crash.

JSON-RPC 2.0 over stdio, newline-delimited messages (no Content-Length
framing needed). For each configured server we do a fast discovery pass
at register time (short ``DISCOVERY_TIMEOUT``) and expose every MCP tool
as ``agent.tools["mcp__<server>__<tool>"]``. The stdio connection itself
is established lazily on first tool call and cached per server; a dead
server yields the clean string ``ERROR: MCP server '<name>' unavailable``
from the tool handler — never an exception to the model.

Public API for the agent and the TUI:
    - :func:`register` -- attach ``agent.mcp`` (an :class:`MCPManager`)
      and register the discovered per-tool entries in ``agent.tools``.
    - :func:`handle_mcp` -- TUI ``/mcp`` command: ``/mcp`` lists
      configured servers with connection status and tool counts,
      ``/mcp <server>`` lists that server's tools.

``python3 -m fullagent.mcp`` runs the built-in self-test.
"""

from __future__ import annotations

import json
import os
import re
import select
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

from .tools import Tool, RISK_CONFIRM

CONFIG_PATH = os.path.join(os.environ.get("FULLAGENT_HOME") or os.path.join(os.path.expanduser("~"), ".fullagent"), "mcp.json")
DEFAULT_TIMEOUT = 30.0          # seconds, for connect + tool calls
DISCOVERY_TIMEOUT = 5.0         # seconds, for register-time tool discovery
MAX_LINE_BYTES = 16 * 1024 * 1024

_NAME_RE = re.compile(r"[^A-Za-z0-9_]")


def _sanitize(name: str) -> str:
    """Make a server/tool name safe for a Tool registry key."""
    return _NAME_RE.sub("_", str(name)) or "unnamed"


class MCPError(Exception):
    """Anything that went wrong talking to an MCP server."""


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def load_config(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Return ``{server_name: server_cfg}``; empty dict on any failure."""
    path = path or CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    servers = data.get("servers")
    if not isinstance(servers, dict):
        return {}
    top_timeout = _as_timeout(data.get("timeout"), DEFAULT_TIMEOUT)
    out: Dict[str, Dict[str, Any]] = {}
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            continue
        command = cfg.get("command")
        if not command or not isinstance(command, str):
            continue  # needs a command to spawn
        args = cfg.get("args")
        env = cfg.get("env")
        out[str(name)] = {
            "command": command,
            "args": list(args) if isinstance(args, list) else [],
            "env": {str(k): str(v) for k, v in env.items()}
            if isinstance(env, dict) else {},
            "timeout": _as_timeout(cfg.get("timeout"), top_timeout),
        }
    return out


def _as_timeout(value: Any, fallback: float) -> float:
    try:
        t = float(value)
    except (TypeError, ValueError):
        return fallback
    if t <= 0:
        return fallback
    return min(t, 300.0)


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 over stdio
# ---------------------------------------------------------------------------

class _MCPClient:
    """One stdio connection to an MCP server (not thread-shared)."""

    def __init__(self, name: str, cfg: Dict[str, Any]) -> None:
        self.name = name
        self.cfg = cfg
        self.timeout = cfg["timeout"]
        self._proc: Optional[subprocess.Popen] = None
        self._next_id = 0
        self.alive = False

    # -- lifecycle ------------------------------------------------------
    def connect(self, timeout: Optional[float] = None) -> None:
        """Spawn the server and run the MCP initialize handshake."""
        if self.alive:
            return
        timeout = self.timeout if timeout is None else timeout
        env = dict(os.environ)
        env.update(self.cfg["env"])
        try:
            proc = subprocess.Popen(
                [self.cfg["command"]] + [str(a) for a in self.cfg["args"]],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env=env)
        except (OSError, ValueError) as e:
            raise MCPError(f"cannot spawn MCP server '{self.name}': {e}")
        self._proc = proc
        try:
            result = self._request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "fullagent", "version": "1.0"},
            }, timeout=timeout)
            if not isinstance(result, dict):
                raise MCPError("bad initialize response")
            # Acknowledge; notifications get no reply and no id.
            self._send({"jsonrpc": "2.0",
                        "method": "notifications/initialized"})
        except Exception:
            self.close()
            raise
        self.alive = True

    def close(self) -> None:
        proc, self._proc = self._proc, None
        self.alive = False
        if proc is not None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass

    # -- protocol ---------------------------------------------------------
    def _send(self, obj: Dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        try:
            self._proc.stdin.write(line.encode("utf-8"))
            self._proc.stdin.flush()
        except (OSError, BrokenPipeError) as e:
            raise MCPError(f"write to MCP server '{self.name}' failed: {e}")

    # stash for bytes read past the first newline
    _rest: bytes = b""

    def _read_line(self, timeout: float) -> Optional[str]:
        """Read one stdout line within timeout; None on timeout/EOF.

        Bytes read past the first newline are stashed in ``_rest`` so no
        interleaved message is ever lost.
        """
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        deadline = time.monotonic() + timeout
        buf = bytearray(getattr(self, "_rest", b""))
        self._rest = b""
        while True:
            if b"\n" in buf:
                line, _, rest = buf.partition(b"\n")
                self._rest = bytes(rest)
                return line.decode("utf-8", "replace")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._rest = bytes(buf)  # keep partial for the next call
                return None
            try:
                r, _, _ = select.select([proc.stdout], [], [], remaining)
            except (OSError, ValueError):
                self._rest = bytes(buf)
                return None
            if not r:
                self._rest = bytes(buf)
                return None
            chunk = (proc.stdout.read1(4096) if hasattr(proc.stdout, "read1")
                     else proc.stdout.read(4096))
            if not chunk:
                self._rest = bytes(buf)
                return None  # EOF
            buf += chunk
            if len(buf) > MAX_LINE_BYTES:
                raise MCPError("MCP server line exceeded size limit")

    def _request(self, method: str, params: Dict[str, Any],
                 timeout: Optional[float] = None) -> Any:
        timeout = self.timeout if timeout is None else timeout
        if self._proc is None or self._proc.poll() is not None:
            raise MCPError(f"MCP server '{self.name}' process exited")
        self._next_id += 1
        rid = self._next_id
        self._send({"jsonrpc": "2.0", "id": rid,
                    "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError(
                    f"MCP server '{self.name}' timed out on '{method}'")
            line = self._read_line(remaining)
            if line is None:
                raise MCPError(
                    f"MCP server '{self.name}' timed out on '{method}'")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue  # stray log line on stdout; ignore
            if not isinstance(msg, dict):
                continue
            if msg.get("id") != rid:
                continue  # notification or unrelated response
            if "error" in msg and msg["error"] is not None:
                err = msg["error"]
                detail = err.get("message") if isinstance(err, dict) else err
                raise MCPError(
                    f"MCP server '{self.name}' error on '{method}': {detail}")
            return msg.get("result")

    # -- MCP methods -------------------------------------------------------
    def list_tools(self) -> List[Dict[str, Any]]:
        result = self._request("tools/list", {})
        tools = result.get("tools") if isinstance(result, dict) else None
        return [t for t in tools
                if isinstance(t, dict) and t.get("name")] if isinstance(
                    tools, list) else []

    def call_tool(self, tool_name: str,
                  arguments: Dict[str, Any]) -> Dict[str, Any]:
        result = self._request("tools/call", {"name": tool_name,
                                              "arguments": arguments})
        return result if isinstance(result, dict) else {}


# ---------------------------------------------------------------------------
# manager (attached to the agent as agent.mcp)
# ---------------------------------------------------------------------------

class MCPManager:
    """Owns configs + cached stdio clients for all MCP servers."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = config_path or CONFIG_PATH
        self.servers: Dict[str, Dict[str, Any]] = load_config(
            self.config_path)
        self._clients: Dict[str, _MCPClient] = {}
        self._discovered: Dict[str, List[Dict[str, Any]]] = {}
        self._errors: Dict[str, str] = {}
        self._lock = threading.Lock()

    def server_names(self) -> List[str]:
        return sorted(self.servers)

    def ensure_client(self, name: str,
                      timeout: Optional[float] = None) -> _MCPClient:
        """Lazily connect (cached) — raises MCPError when unavailable."""
        with self._lock:
            cfg = self.servers.get(name)
            if cfg is None:
                raise MCPError(f"unknown MCP server '{name}'")
            client = self._clients.get(name)
            if client is not None and client.alive:
                return client
            client = _MCPClient(name, cfg)
            try:
                client.connect(timeout=timeout)
            except Exception as e:
                self._errors[name] = str(e)[:200]
                raise MCPError(str(e))
            self._errors.pop(name, None)
            self._clients[name] = client
            return client

    def discover(self, name: str) -> List[Dict[str, Any]]:
        """Connect (short timeout) and cache this server's tool list."""
        client = self.ensure_client(name, timeout=DISCOVERY_TIMEOUT)
        tools = client.list_tools()
        with self._lock:
            self._discovered[name] = tools
        return tools

    def discovered_tools(self, name: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._discovered.get(name, []))

    def last_error(self, name: str) -> Optional[str]:
        with self._lock:
            return self._errors.get(name)

    def is_connected(self, name: str) -> bool:
        with self._lock:
            client = self._clients.get(name)
            return bool(client is not None and client.alive)

    def status(self, name: str) -> str:
        if self.is_connected(name):
            return "connected"
        err = self.last_error(name)
        if err:
            return f"unreachable ({err})"
        if name in self._discovered:
            return "discovered"
        return "not connected"

    def shutdown(self) -> None:
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for c in clients:
            try:
                c.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tool schema + result translation
# ---------------------------------------------------------------------------

def _translate_schema(input_schema: Any) -> Dict[str, Any]:
    """MCP inputSchema is already JSON Schema — pass through safely."""
    fallback = {"type": "object", "properties": {}}
    if not isinstance(input_schema, dict):
        return fallback
    out: Dict[str, Any] = {"type": "object"}
    props = input_schema.get("properties")
    out["properties"] = props if isinstance(props, dict) else {}
    required = input_schema.get("required")
    if isinstance(required, list):
        out["required"] = [str(r) for r in required if r]
    return out


def _format_result(result: Dict[str, Any]) -> str:
    if not result:
        return "MCP tool returned an empty result."
    parts: List[str] = []
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "text":
                parts.append(str(item.get("text", "")))
            elif itype == "image":
                parts.append("[image content omitted]")
            elif itype == "resource":
                parts.append("[resource content omitted]")
    text = "\n".join(p for p in parts if p).strip()
    if not text:
        text = json.dumps(result, ensure_ascii=False)[:4000]
    if result.get("isError"):
        return "Error: " + text
    return text


def _make_tool(manager: MCPManager, server: str,
               spec: Dict[str, Any]) -> Tool:
    tool_name = str(spec.get("name", "unnamed"))
    description = str(spec.get("description") or
                      f"MCP tool '{tool_name}' on server '{server}'")
    description = f"[MCP:{server}] {description}"

    def _handler(**kwargs: Any) -> str:
        try:
            client = manager.ensure_client(server)
            result = client.call_tool(tool_name, kwargs)
            return _format_result(result)
        except Exception:
            return f"ERROR: MCP server '{server}' unavailable"

    return Tool(
        name=f"mcp__{_sanitize(server)}__{_sanitize(tool_name)}",
        description=description,
        parameters=_translate_schema(spec.get("inputSchema")),
        handler=_handler,
        # Remote MCP tools can have arbitrary side effects; they must go
        # through the approval gate like shell/edit tools.
        risk=RISK_CONFIRM,
    )


# ---------------------------------------------------------------------------
# register + TUI
# ---------------------------------------------------------------------------

def register(agent: Any) -> None:
    """Wire MCP tools into an agent (duck-typed). Guarded: never raises.

    Discovery is eager-but-fast (``DISCOVERY_TIMEOUT`` per server) so tool
    names can be registered up front; the actual stdio connections stay
    lazy — every tool handler calls ``ensure_client`` which connects on
    first use and caches the client afterwards.
    """
    try:
        manager = MCPManager()
    except Exception:
        return
    try:
        agent.mcp = manager
    except Exception:
        pass
    tools = getattr(agent, "tools", None)
    if not isinstance(tools, dict):
        return
    for name in manager.server_names():
        try:
            specs = manager.discover(name)
        except Exception:
            continue  # dead server at startup: zero tools, no crash
        for spec in specs:
            try:
                tool = _make_tool(manager, name, spec)
                if tool.name not in tools:
                    tools[tool.name] = tool
            except Exception:
                continue


def _manager_from_ui(ui: Any) -> Optional[MCPManager]:
    agent = getattr(ui, "agent", None)
    manager = getattr(agent, "mcp", None) if agent is not None else None
    if isinstance(manager, MCPManager):
        return manager
    try:
        return MCPManager()
    except Exception:
        return None


def handle_mcp(ui: Any, arg: str) -> None:
    """TUI handler for ``/mcp``. Prints via ui.print_info / ui.print_error.

    ``/mcp``            list configured servers, status, and tool counts.
    ``/mcp <server>``   list that server's tools.
    """
    manager = _manager_from_ui(ui)
    if manager is None:
        ui.print_error("MCP support not initialised")
        return
    text = (arg or "").strip()
    if not text:
        names = manager.server_names()
        if not names:
            ui.print_info(
                "no MCP servers configured — add them to "
                "~/.fullagent/mcp.json")
            return
        lines = [f"mcp servers ({len(names)}):"]
        for name in names:
            cfg = manager.servers[name]
            n_tools = len(manager.discovered_tools(name))
            lines.append(
                f"  {name:<20} {manager.status(name):<28} "
                f"{n_tools} tool(s)  [{cfg['command']} "
                f"{' '.join(cfg['args'])}]".rstrip())
        ui.print_info("\n".join(lines))
        return
    name = text.split(None, 1)[0]
    if name not in manager.servers:
        ui.print_error(
            f"unknown MCP server '{name}' — configured: "
            + (", ".join(manager.server_names()) or "(none)"))
        return
    try:
        tools = manager.discover(name)
    except Exception as e:
        ui.print_error(f"ERROR: MCP server '{name}' unavailable ({e})")
        return
    if not tools:
        ui.print_info(f"mcp server '{name}': no tools exposed")
        return
    lines = [f"mcp server '{name}' tools ({len(tools)}):"]
    for t in tools:
        desc = str(t.get("description", "")).strip().split("\n")[0]
        if len(desc) > 80:
            desc = desc[:77] + "..."
        lines.append(f"  mcp__{_sanitize(name)}__{_sanitize(t['name'])}"
                     + (f" — {desc}" if desc else ""))
    ui.print_info("\n".join(lines))


# ---------------------------------------------------------------------------
# self-test: `python3 -m fullagent.mcp`  →  PASS
# ---------------------------------------------------------------------------

_FAKE_SERVER_SRC = '''\
import json, sys

TOOLS = [{
    "name": "echo",
    "description": "Echo back the text argument",
    "inputSchema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}]

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()

for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except ValueError:
        continue
    if not isinstance(msg, dict):
        continue
    method = msg.get("method")
    rid = msg.get("id")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid,
              "result": {"protocolVersion": "2024-11-05",
                         "capabilities": {},
                         "serverInfo": {"name": "fake", "version": "0"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        params = msg.get("params") or {}
        text = (params.get("arguments") or {}).get("text", "")
        send({"jsonrpc": "2.0", "id": rid,
              "result": {"content": [{"type": "text",
                                      "text": "echo: " + str(text)}]}})
    elif rid is not None:
        send({"jsonrpc": "2.0", "id": rid,
              "error": {"code": -32601, "message": "unknown method"}})
'''


class _FakeAgent:
    def __init__(self):
        self.session_id = "mcp-selftest"
        self.tools = {}


class _FakeUI:
    def __init__(self, agent):
        self.agent = agent
        self.infos: list = []
        self.errors: list = []

    def print_info(self, text, color=None):
        self.infos.append(text)

    def print_error(self, text):
        self.errors.append(text)


def _selftest() -> None:
    import tempfile

    tmp = tempfile.mkdtemp(prefix="mcp_selftest_")
    server_path = os.path.join(tmp, "fake_mcp_server.py")
    with open(server_path, "w", encoding="utf-8") as fh:
        fh.write(_FAKE_SERVER_SRC)
    config_path = os.path.join(tmp, "mcp.json")
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump({"servers": {
            "fake": {"command": "python3", "args": [server_path]},
            "dead": {"command": "definitely-not-a-real-binary-xyz"},
        }, "timeout": 10}, fh)

    global CONFIG_PATH
    old_config_path = CONFIG_PATH
    CONFIG_PATH = config_path
    try:
        # --- missing file = zero servers, no crash ---------------------
        mgr = MCPManager(config_path=os.path.join(tmp, "nope.json"))
        assert mgr.server_names() == [], mgr.server_names()

        # --- register wires the fake server's tools --------------------
        agent = _FakeAgent()
        register(agent)
        assert isinstance(agent.mcp, MCPManager)
        assert "mcp__fake__echo" in agent.tools, sorted(agent.tools)
        assert "mcp__dead__" not in "".join(agent.tools), "dead server"

        tool = agent.tools["mcp__fake__echo"]
        assert tool.description.startswith("[MCP:fake]"), tool.description
        params = tool.parameters
        assert params["type"] == "object", params
        assert "text" in params["properties"], params
        assert params["required"] == ["text"], params

        # --- call round-trip -------------------------------------------
        out = tool.handler(text="hello mcp")
        assert out == "echo: hello mcp", repr(out)

        # --- second call reuses the cached (lazy) connection ------------
        out = tool.handler(text="again")
        assert out == "echo: again", repr(out)
        assert agent.mcp.is_connected("fake")

        # --- dead server yields a clean ERROR string, never raises -----
        dead_mgr = MCPManager(config_path=config_path)
        dead_tool = _make_tool(dead_mgr, "dead",
                               {"name": "x", "description": "x",
                                "inputSchema": {}})
        out = dead_tool.handler()
        assert out == "ERROR: MCP server 'dead' unavailable", repr(out)

        # --- schema fallback for missing inputSchema --------------------
        fallback = _translate_schema(None)
        assert fallback == {"type": "object", "properties": {}}, fallback
        fallback = _translate_schema("junk")
        assert fallback["properties"] == {}, fallback

        # --- TUI: /mcp lists servers -----------------------------------
        ui = _FakeUI(agent)
        handle_mcp(ui, "")
        assert len(ui.infos) == 1, (ui.infos, ui.errors)
        listing = ui.infos[0]
        assert "fake" in listing and "dead" in listing, listing
        assert "1 tool(s)" in listing, listing
        assert "unreachable" in listing, listing  # dead server status

        # --- TUI: /mcp fake lists tools --------------------------------
        ui = _FakeUI(agent)
        handle_mcp(ui, "fake")
        assert "mcp__fake__echo" in ui.infos[0], ui.infos
        assert "Echo back" in ui.infos[0], ui.infos

        # --- TUI: unknown server ----------------------------------------
        ui = _FakeUI(agent)
        handle_mcp(ui, "nope")
        assert ui.errors and "unknown MCP server" in ui.errors[0], ui.errors

        # --- TUI with no agent attached still works ----------------------
        ui = _FakeUI(None)
        ui.agent = None
        handle_mcp(ui, "")
        assert ui.infos and "fake" in ui.infos[0], (ui.infos, ui.errors)

        # --- real config file missing on a fresh manager ------------------
        real_missing = MCPManager.__new__(MCPManager)
        real_missing.servers = load_config(os.path.join(tmp, "missing.json"))
        assert real_missing.servers == {}

        agent.mcp.shutdown()
    finally:
        CONFIG_PATH = old_config_path

    print("PASS")


if __name__ == "__main__":
    _selftest()
