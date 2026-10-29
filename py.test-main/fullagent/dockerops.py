"""Docker CLI integration (``dockerops`` feature module).

Real docker CLI wrapper via ``subprocess`` — no mocks, no simulations.
Exposes five tools to the agent:

    docker_ps       list running containers (``docker ps``)
    docker_run      run a container (``docker run``)
    docker_logs     fetch container logs (``docker logs``)
    docker_stop     stop a container (``docker stop``)
    docker_images   list local images (``docker images``)

Availability: every handler first checks ``shutil.which("docker")`` and
``docker info``. When docker is missing (or the daemon is down) the tools
return a clear "docker not available" message instead of raising or
faking success. All subprocess calls have a 30s timeout and capture
stderr, surfacing real errors in the returned text.

``python3 -m fullagent.dockerops`` runs the built-in self-test: if docker
is available it runs the real ``hello-world`` image end-to-end and cleans
up; otherwise it verifies every tool returns the not-available message.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from typing import Optional

DOCKER_TIMEOUT_S = 30

NOT_AVAILABLE = (
    "docker not available: the docker CLI was not found on this machine "
    "or the docker daemon is not running. Install Docker "
    "(https://docs.docker.com/get-docker/) and start the daemon "
    "(e.g. `sudo systemctl start docker`), then try again."
)


# ---------------------------------------------------------------------------
# availability + low-level runner
# ---------------------------------------------------------------------------

_availability_cache: Optional[bool] = None


def docker_available() -> bool:
    """True when the docker CLI exists and the daemon answers ``docker info``.

    The result is cached for the process lifetime — availability is not
    expected to change while the agent runs.
    """
    global _availability_cache
    if _availability_cache is not None:
        return _availability_cache
    if shutil.which("docker") is None:
        _availability_cache = False
        return False
    try:
        proc = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=DOCKER_TIMEOUT_S,
        )
        _availability_cache = proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        _availability_cache = False
    return _availability_cache


def _run_docker(*args: str) -> str:
    """Run ``docker <args>`` and return a readable result string.

    Non-zero exits surface the real stderr instead of raising.
    """
    try:
        proc = subprocess.run(
            ["docker", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=DOCKER_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"error: docker {' '.join(args)} timed out after {DOCKER_TIMEOUT_S}s"
    except OSError as e:
        return f"error: failed to launch docker: {e}"
    out = proc.stdout.strip()
    err = proc.stderr.strip()
    if proc.returncode == 0:
        return out if out else "(no output)"
    detail = err or out or f"exit code {proc.returncode}"
    return f"error: docker {' '.join(args)} failed:\n{detail}"


def _require_docker() -> Optional[str]:
    """Return the not-available message when docker is unusable, else None."""
    return None if docker_available() else NOT_AVAILABLE


# ---------------------------------------------------------------------------
# tool handlers
# ---------------------------------------------------------------------------

def _docker_ps(**kwargs) -> str:
    """List running containers."""
    msg = _require_docker()
    if msg:
        return msg
    show_all = bool(kwargs.get("all", False))
    args = ["ps", "--format",
            "{{.ID}}\t{{.Image}}\t{{.Names}}\t{{.Status}}\t{{.Ports}}"]
    if show_all:
        args.append("-a")
    out = _run_docker(*args)
    if out.startswith("error") or out == "(no output)":
        return "No running containers." if out == "(no output)" else out
    lines = ["CONTAINER", "========="]
    for row in out.splitlines():
        parts = row.split("\t")
        cid = parts[0][:12] if len(parts) > 0 else "?"
        image = parts[1] if len(parts) > 1 else "?"
        name = parts[2] if len(parts) > 2 else "?"
        status = parts[3] if len(parts) > 3 else ""
        ports = parts[4] if len(parts) > 4 else ""
        desc = f"{name} ({image}) — {cid} — {status}"
        if ports:
            desc += f" — ports: {ports}"
        lines.append(desc)
    return "\n".join(lines)


def _docker_run(**kwargs) -> str:
    """Run a container from an image."""
    msg = _require_docker()
    if msg:
        return msg
    image = str(kwargs.get("image", "")).strip()
    if not image:
        return "error: 'image' is required (e.g. docker_run(image='hello-world'))"
    command = str(kwargs.get("command", "")).strip()
    name = str(kwargs.get("name", "")).strip()
    detach = bool(kwargs.get("detach", False))
    argv = ["run"]
    if detach:
        argv.append("-d")
    else:
        argv.append("--rm")
    if name:
        argv += ["--name", name]
    argv.append(image)
    if command:
        argv += shlex.split(command)
    out = _run_docker(*argv)
    if out.startswith("error"):
        return out
    if detach:
        return f"container started: {out.strip()[:64]}" + (
            f" (name: {name})" if name else "")
    return out


def _docker_logs(**kwargs) -> str:
    """Fetch a container's logs."""
    msg = _require_docker()
    if msg:
        return msg
    container = str(kwargs.get("container", "")).strip()
    if not container:
        return "error: 'container' is required (name or id)"
    try:
        tail = int(kwargs.get("tail", 50))
    except (TypeError, ValueError):
        return "error: 'tail' must be an integer"
    if tail < 1:
        tail = 1
    out = _run_docker("logs", "--tail", str(tail), container)
    if out.startswith("error"):
        return out
    if out == "(no output)":
        return f"No logs for container '{container}' (it may have produced none)."
    return f"logs for '{container}' (last {tail} lines):\n{out}"


def _docker_stop(**kwargs) -> str:
    """Stop a running container."""
    msg = _require_docker()
    if msg:
        return msg
    container = str(kwargs.get("container", "")).strip()
    if not container:
        return "error: 'container' is required (name or id)"
    out = _run_docker("stop", container)
    if out.startswith("error"):
        return out
    return f"container '{container}' stopped."


def _docker_images(**kwargs) -> str:
    """List local images."""
    msg = _require_docker()
    if msg:
        return msg
    out = _run_docker(
        "images", "--format",
        "{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.CreatedSince}}")
    if out.startswith("error"):
        return out
    if out == "(no output)":
        return "No local docker images."
    lines = ["IMAGE (repo:tag) — id — size — created"]
    for row in out.splitlines():
        parts = row.split("\t")
        repo = parts[0] if len(parts) > 0 else "?"
        iid = parts[1][:12] if len(parts) > 1 else "?"
        size = parts[2] if len(parts) > 2 else "?"
        created = parts[3] if len(parts) > 3 else "?"
        lines.append(f"{repo} — {iid} — {size} — {created}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def register(agent) -> None:
    """Attach the docker tools to an agent."""
    from .tools import RISK_CONFIRM, RISK_SAFE, Tool  # local: no import cycle

    tools = getattr(agent, "tools", None)
    if tools is None:
        raise AttributeError("agent has no 'tools' registry")

    tools["docker_ps"] = Tool(
        "docker_ps",
        "List running docker containers (name, image, status, ports). "
        "Returns a readable list; pass all=true to include stopped ones.",
        {"type": "object",
         "properties": {"all": {"type": "boolean",
                                "description": "include stopped containers"}},
         "required": []},
        _docker_ps, risk=RISK_SAFE)

    tools["docker_run"] = Tool(
        "docker_run",
        "Run a docker container from an image. Pulls the image if needed. "
        "Set detach=true to run in the background and get a container id; "
        "otherwise runs attached (--rm) and returns its output.",
        {"type": "object",
         "properties": {
             "image": {"type": "string",
                       "description": "image to run, e.g. 'hello-world'"},
             "command": {"type": "string",
                         "description": "command inside the container"},
             "name": {"type": "string",
                      "description": "optional container name"},
             "detach": {"type": "boolean",
                        "description": "run in background (default false)"}},
         "required": ["image"]},
        _docker_run, risk=RISK_CONFIRM)

    tools["docker_logs"] = Tool(
        "docker_logs",
        "Fetch the logs of a docker container (by name or id). "
        "Optional tail controls how many lines (default 50).",
        {"type": "object",
         "properties": {
             "container": {"type": "string",
                           "description": "container name or id"},
             "tail": {"type": "integer",
                      "description": "number of log lines (default 50)"}},
         "required": ["container"]},
        _docker_logs, risk=RISK_SAFE)

    tools["docker_stop"] = Tool(
        "docker_stop",
        "Stop a running docker container (by name or id).",
        {"type": "object",
         "properties": {"container": {"type": "string",
                                       "description": "container name or id"}},
         "required": ["container"]},
        _docker_stop, risk=RISK_CONFIRM)

    tools["docker_images"] = Tool(
        "docker_images",
        "List local docker images (repo:tag, id, size, age).",
        {"type": "object", "properties": {}, "required": []},
        _docker_images, risk=RISK_SAFE)


# ---------------------------------------------------------------------------
# Self-test: python3 -m fullagent.dockerops
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    failures = []

    def check(label, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))
        if not cond:
            failures.append(label)

    if docker_available():
        print("docker IS available — running real end-to-end test\n")
        # 1. real run: hello-world, attached, auto-removed
        out = _docker_run(image="hello-world")
        check("docker_run hello-world", "Hello from Docker!" in out,
              repr(out[:80]))
        # 2. real images list shows hello-world
        imgs = _docker_images()
        check("docker_images lists hello-world", "hello-world" in imgs)
        # 3. real ps works
        ps = _docker_ps()
        check("docker_ps returns text", isinstance(ps, str) and len(ps) > 0)
        # 4. real run detached + logs + stop + cleanup
        cname = "fullagent-dockerops-selftest"
        started = _docker_run(image="alpine", command="echo selftest-ok",
                              name=cname, detach=True)
        if started.startswith("container started"):
            # docker logs can be empty for a moment while the container
            # starts writing on a busy runner — poll briefly instead of
            # failing the whole gate on a scheduling race
            logs = ""
            deadline = time.time() + 10.0
            while time.time() < deadline:
                logs = _docker_logs(container=cname, tail=20)
                if "selftest-ok" in logs:
                    break
                time.sleep(0.25)
            check("docker_logs detached container", "selftest-ok" in logs,
                  repr(logs[:80]))
            stopped = _docker_stop(container=cname)
            check("docker_stop container", "stopped" in stopped)
            subprocess.run(["docker", "rm", "-f", cname],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           timeout=DOCKER_TIMEOUT_S)
        else:
            check("docker_run detached alpine", False, started)
        # 5. real error surfacing
        bad = _docker_stop(container="no-such-container-xyz")
        check("real errors surfaced", bad.startswith("error"), repr(bad[:60]))
    else:
        print("docker NOT available — verifying graceful degradation\n")
        check("docker_available() is False", docker_available() is False)
        cases = [
            ("docker_ps", _docker_ps()),
            ("docker_run", _docker_run(image="hello-world")),
            ("docker_logs", _docker_logs(container="x", tail=10)),
            ("docker_stop", _docker_stop(container="x")),
            ("docker_images", _docker_images()),
        ]
        for name, result in cases:
            ok = (isinstance(result, str)
                  and "docker not available" in result
                  and not result.startswith("error: '"))
            check(f"{name} returns not-available message", ok,
                  repr(result[:70]))

    # registration never explodes, even without docker
    class _FakeAgent:
        tools = {}

    try:
        register(_FakeAgent())
        registered = sorted(_FakeAgent.tools.keys())
        check("register() attaches 5 tools",
              registered == ["docker_images", "docker_logs", "docker_ps",
                             "docker_run", "docker_stop"],
              str(registered))
    except Exception as e:  # noqa: BLE001
        check("register() attaches 5 tools", False, f"raised {e!r}")

    print()
    if failures:
        print(f"{len(failures)} self-test failure(s): {failures}")
        sys.exit(1)
    print("dockerops self-test: all checks passed")
