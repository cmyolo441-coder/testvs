"""`/doctor` command — health diagnostics.

Quick health check of the agent's runtime: Python version, provider auth
availability (presence only — values are never printed), provider transport
connectivity (short-timeout pings or local CLI checks), the app state dir,
disk space, and core dependencies.

Two entry points (duck-typed, no imports of ``.agent``/``.tui`` so there
are no import cycles):

    - :func:`register` — attach a :class:`Doctor` to an agent as
      ``agent.doctor``; ``agent.doctor.run()`` returns the checks as
      ``list[dict]`` with ``{"name", "ok", "detail"}`` items.
    - :func:`handle_doctor` — the TUI handler: ``handle_doctor(ui, arg)``
      prints a ✓/✗ report plus a one-line summary.

``python3 -m fullagent.doctor`` runs the built-in self-test (the network
check is mocked — no real network traffic).
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import urllib.request
import urllib.error
from importlib import metadata
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import PROVIDERS, _provider_api_key

CONNECT_TIMEOUT = 5.0  # seconds per connectivity probe
DISK_WARN_BYTES = 1 * 1024 * 1024 * 1024  # warn when < 1 GB free


# ---------------------------------------------------------------------------
# network probe — a module-level hook so tests can mock it without touching
# real network state.
# ---------------------------------------------------------------------------

def _fetch_url(url: str, timeout: float = CONNECT_TIMEOUT
               ) -> Tuple[bool, str, float]:
    """GET ``url`` with a short timeout.

    Returns ``(reachable, status_or_error, latency_ms)`` and NEVER raises.
    Any HTTP response (even 4xx/5xx) counts as reachable — the host and
    the network path work; we are only probing connectivity, not auth.
    """
    start = time.perf_counter()
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "fullagent-doctor/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = f"HTTP {resp.status}"
        reachable = True
    except urllib.error.HTTPError as exc:
        status = f"HTTP {exc.code}"
        reachable = True
    except Exception as exc:  # connection refused, timeout, DNS, TLS…
        status = f"{type(exc).__name__}: {exc}"
        reachable = False
    latency_ms = (time.perf_counter() - start) * 1000.0
    return reachable, status, latency_ms


# Injected probe (default: the real one above). Self-tests replace it.
_probe: Callable[[str, float], Tuple[bool, str, float]] = _fetch_url


# ---------------------------------------------------------------------------
# individual checks — each returns {"name", "ok", "detail"}
# ---------------------------------------------------------------------------

def _result(name: str, ok: bool, detail: str) -> Dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _check_python_version() -> Dict[str, Any]:
    ok = sys.version_info >= (3, 9)
    detail = f"{sys.version_info.major}.{sys.version_info.minor}." \
             f"{sys.version_info.micro} (>= 3.9 required)"
    return _result("python", ok, detail)


def _check_provider_keys() -> List[Dict[str, Any]]:
    """One auth check per provider.

    Uses the same lookup as the rest of the app
    (env var > key file > embedded fallback). Only presence is reported —
    key VALUES never appear.
    """
    results = []
    for key in PROVIDERS:
        try:
            present = bool(_provider_api_key(key))
        except Exception as exc:
            results.append(_result(f"api-key:{key}", False,
                                  f"lookup failed: {type(exc).__name__}"))
            continue
        results.append(_result(f"api-key:{key}", present,
                               "set" if present else "missing"))
    return results


def _check_connectivity() -> List[Dict[str, Any]]:
    """Probe the configured HTTP providers."""
    results = []
    for key, provider in PROVIDERS.items():
        url = provider.base_url.rstrip("/") + "/models"
        try:
            reachable, status, latency_ms = _probe(url, CONNECT_TIMEOUT)
        except Exception as exc:  # a misbehaving probe must not raise
            reachable, status, latency_ms = (
                False, f"probe error: {type(exc).__name__}: {exc}", 0.0)
        if reachable:
            detail = f"reachable ({status}, {latency_ms:.0f} ms)"
        else:
            detail = f"unreachable ({status}, {latency_ms:.0f} ms)"
        results.append(_result(f"net:{key}", reachable, detail))
    return results


def _check_app_dir() -> Dict[str, Any]:
    app_dir = os.environ.get("FULLAGENT_HOME") or os.path.join(
        os.path.expanduser("~"), ".fullagent")
    exists = os.path.isdir(app_dir)
    writable = os.access(app_dir, os.W_OK) if exists else False
    ok = exists and writable
    detail = f"{app_dir} exists and is writable" if ok else \
        f"{app_dir} exists={exists} writable={writable}"
    return _result("app-dir", ok, detail)


def _check_disk() -> Dict[str, Any]:
    home = os.path.expanduser("~")
    try:
        usage = shutil.disk_usage(home)
        free_gb = usage.free / (1024 ** 3)
        ok = usage.free >= DISK_WARN_BYTES
        detail = f"{free_gb:.1f} GB free on home dir" + (
            "" if ok else " (WARNING: below 1 GB threshold)")
    except OSError as exc:
        return _result("disk", False, f"could not read disk usage: {exc}")
    return _result("disk", ok, detail)


def _check_dependencies() -> Dict[str, Any]:
    missing = []
    versions = []
    for pkg in ("prompt_toolkit", "rich"):
        try:
            versions.append(f"{pkg} {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            missing.append(pkg)
    if missing:
        installed_str = f" (installed: {', '.join(versions)})" if versions else ""
        return _result("deps", False,
                       "missing: " + ", ".join(missing) + installed_str)
    return _result("deps", True, ", ".join(versions) + " installed")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def run_checks() -> List[Dict[str, Any]]:
    """Run every health check; each item is {"name", "ok", "detail"}."""
    results: List[Dict[str, Any]] = [_check_python_version()]
    results += _check_provider_keys()
    results += _check_connectivity()
    results.append(_check_app_dir())
    results.append(_check_disk())
    results.append(_check_dependencies())
    return results


class Doctor:
    """Duck-typed agent attachment: ``agent.doctor.run()`` → checks."""

    def run(self) -> List[Dict[str, Any]]:
        return run_checks()


def register(agent: Any) -> None:
    """Attach a Doctor to an agent (duck-typed, additive)."""
    agent.doctor = Doctor()


def handle_doctor(ui: Any, arg: str) -> None:
    """TUI handler for ``/doctor``: print a ✓/✗ report with a summary.

    ``ui`` is the TUI object (duck-typed: ``print_info``/``print_error``).
    ``arg`` is ignored (accepted for the standard handler signature).
    """
    results = run_checks()
    passed = sum(1 for r in results if r["ok"])
    for r in results:
        mark = "✓" if r["ok"] else "✗"
        line = f"{mark} {r['name']}: {r['detail']}"
        if r["ok"]:
            ui.print_info(line)
        else:
            ui.print_error(line)
    summary = f"{passed}/{len(results)} checks passing"
    if passed == len(results):
        ui.print_info(summary)
    else:
        ui.print_error(summary)


# ---------------------------------------------------------------------------
# self-test: `python3 -m fullagent.doctor`  →  PASS
# ---------------------------------------------------------------------------

class _FakeUI:
    def __init__(self):
        self.infos: List[str] = []
        self.errors: List[str] = []

    def print_info(self, text: str, color: Optional[str] = None) -> None:
        self.infos.append(text)

    def print_error(self, text: str) -> None:
        self.errors.append(text)


def _selftest() -> None:
    global _probe
    real_probe = _probe

    # 1. every check returns the {"name", "ok", "detail"} shape
    _probe = lambda url, timeout: (True, "HTTP 200", 12.0)  # noqa: E731
    results = run_checks()
    _probe = real_probe
    assert results, "no checks returned"
    for r in results:
        assert set(r.keys()) == {"name", "ok", "detail"}, r
        assert isinstance(r["ok"], bool), r
        assert isinstance(r["detail"], str), r

    # 2. key-presence check never leaks key VALUES
    os.environ["FAKEPROV_API_KEY"] = "sk-test-secret-123456"
    try:
        keyed = _check_provider_keys()
        names = {r["name"] for r in keyed}
        assert {f"api-key:{k}" for k in PROVIDERS} == names, names
        for r in keyed:
            assert r["detail"] in ("set", "missing"), r
            assert "secret" not in r["detail"], r
            assert "sk-test" not in r["detail"], r
    finally:
        del os.environ["FAKEPROV_API_KEY"]

    # 3. mocked connectivity: reachable path formats latency
    _probe = lambda url, timeout: (True, "HTTP 404", 87.0)  # noqa: E731
    try:
        conns = _check_connectivity()
    finally:
        _probe = real_probe
    assert len(conns) == len(PROVIDERS), conns
    assert all(c["ok"] for c in conns), conns
    assert "87 ms" in conns[0]["detail"] and "reachable" in conns[0]["detail"]

    # 4. mocked connectivity: unreachable path reports cleanly
    _probe = lambda url, timeout: (False, "URLError: refused", 5.0)  # noqa: E731
    try:
        conns = _check_connectivity()
    finally:
        _probe = real_probe
    assert not any(c["ok"] for c in conns), conns
    assert "unreachable" in conns[0]["detail"]

    # 5. the real probe fails closed against a dead localhost port —
    #    no network, no hang, no raise
    ok, status, ms = _fetch_url("http://127.0.0.1:9/nope", timeout=2.0)
    assert ok is False, (ok, status)
    assert ms < 2000.0, ms

    # 6. a raising probe is also swallowed by _check_connectivity
    def _boom(url, timeout):
        raise RuntimeError("probe exploded")
    _probe = _boom
    try:
        conns = _check_connectivity()
    finally:
        _probe = real_probe
    assert not any(c["ok"] for c in conns), conns

    # 7. individual checks sanity
    assert _check_python_version()["ok"] is True  # we run on 3.9+ here
    disk = _check_disk()
    assert "free" in disk["detail"].lower(), disk
    deps = _check_dependencies()
    assert "prompt_toolkit" in deps["detail"] and "rich" in deps["detail"], \
        deps
    assert _check_app_dir()["name"] == "app-dir"

    # 8. handle_doctor renders ✓/✗ lines + "N/M checks passing" summary
    ui = _FakeUI()
    handle_doctor(ui, "")
    all_lines = ui.infos + ui.errors
    body, summary = all_lines[:-1], all_lines[-1]
    passed_lines = sum(1 for line in body if line.startswith("✓"))
    assert summary == f"{passed_lines}/{len(body)} checks passing", summary
    assert any("✓" in line for line in body), body
    failing = [line for line in body if line.startswith("✗")]
    if failing:
        assert all(line in ui.errors for line in failing), failing
        assert all(line in ui.infos
                   for line in body if line.startswith("✓")), body

    # 9. register() attaches a doctor with run()
    class _FakeAgent:
        pass
    agent = _FakeAgent()
    register(agent)
    assert hasattr(agent, "doctor")
    assert isinstance(agent.doctor.run(), list)

    print("PASS")


if __name__ == "__main__":
    _selftest()
