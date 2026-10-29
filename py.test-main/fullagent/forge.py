"""FORGE — environment reproducibility (§18).

The most ignored failure mode in the category: the agent fixes a bug that
only exists in its own environment. The Forge hashes everything that can
change behaviour into an EnvironmentDigest, seals it as an event, and
detects mid-session drift.

  * digest()  — os, python, deps, locale, toolchain versions -> one hash.
  * probe()   — digest + env.digest event; every observation is implicitly
                stamped with the environment that produced it.
  * drift()   — compare the latest digests; any material change marks
                earlier evidence stale (the Judge re-runs stale proofs).
"""

from __future__ import annotations

import functools
import hashlib
import json
import locale
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .kernel import EventLog, fold
from ._foundation import get_logger

_log = get_logger("forge")

# toolchains whose versions can change behaviour
_PROBED_TOOLS = ("git", "node", "docker", "gcc", "make")


@functools.lru_cache(maxsize=1)
def _tool_versions() -> dict[str, str]:
    """Probe toolchain versions once per process: five subprocess spawns
    on every digest() call is pure waste — versions don't change
    mid-session often enough to matter, and digest() is called on every
    PERCEIVE."""
    versions: dict[str, str] = {}
    for tool in _PROBED_TOOLS:
        exe = shutil.which(tool)
        if not exe:
            continue
        try:
            out = subprocess.run([exe, "--version"], capture_output=True,
                                 text=True, timeout=5, errors="replace")
            first = (out.stdout or out.stderr or "").splitlines()
            versions[tool] = first[0][:80] if first else "?"
        except (OSError, subprocess.TimeoutExpired):
            versions[tool] = "?"
    return versions


def reprobe_tools() -> dict[str, str]:
    """Clear the cached toolchain versions and re-probe (e.g. after a
    mid-session toolchain install that drift() should notice)."""
    _tool_versions.cache_clear()
    return _tool_versions()


def _lockfile_hash(cwd: Path) -> str | None:
    """Hash the resolved dependency tree if a lockfile exists."""
    for name in ("uv.lock", "poetry.lock", "requirements.txt",
                 "Pipfile.lock", "package-lock.json"):
        p = cwd / name
        if p.is_file():
            try:
                return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            except OSError:
                return None
    return None


class Forge:
    """Environment digest + drift detection over the event log."""

    def __init__(self, log: EventLog, cwd: str | Path | None = None) -> None:
        self.log = log
        if cwd is None:
            try:
                cwd = os.getcwd()
            except OSError:
                # the process's cwd was deleted out from under it
                # (unusual env) — fall back to home instead of crashing
                # probe() at startup
                cwd = os.path.expanduser("~")
        self.cwd = Path(cwd).resolve()

    def digest(self) -> dict:
        """Compute the EnvironmentDigest (§18.1)."""
        env_allowlist = ("PATH", "VIRTUAL_ENV", "PYTHONPATH", "LANG",
                         "FULLAGENT_HOME")
        record = {
            "os": platform.system(),
            "arch": platform.machine(),
            "kernel": platform.release(),
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "locale": str(locale.getlocale()),
            "encoding": getattr(sys.stdout, "encoding", "") or "",
            "cwd": str(self.cwd),
            "lockfile_hash": _lockfile_hash(self.cwd),
            "env": {k: os.environ.get(k, "") for k in env_allowlist},
            # copy: _tool_versions() is a process-wide lru_cache and
            # EventLog.append stores data by reference — handing out the
            # cached dict would let any later mutation of the sealed
            # event poison every future digest
            "tools": dict(_tool_versions()),
            "case_sensitive": os.name != "nt",
        }
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False,
                             default=str)
        record["digest"] = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return record

    def probe(self) -> dict:
        """Digest + seal an env.digest event. Called at PERCEIVE and
        periodically by the Librarian role."""
        record = self.digest()
        self.log.append("env.digest", record, actor="librarian")
        return record

    def drift(self) -> dict | None:
        """Compare the two most recent digests. Returns a delta dict when
        the environment changed materially, else None. Observations
        recorded before the change are stale; the Judge re-runs proofs
        whose evidence predates a material change (§18.4)."""
        digests = fold(self.log).env_digests
        if len(digests) < 2:
            return None
        prev, cur = digests[-2], digests[-1]
        if prev.get("digest") == cur.get("digest"):
            return None
        # every digest input that can change the hash must be able to
        # show up here — otherwise a real change (locale, LANG, python
        # implementation, ...) yields a delta that claims NOTHING
        # changed while the digests differ
        changed = {k for k in ("os", "arch", "kernel", "python",
                               "implementation", "locale", "encoding",
                               "lockfile_hash", "cwd", "case_sensitive")
                   if prev.get(k) != cur.get(k)}
        env_changed = {k for k in set(prev.get("env", {})) |
                       set(cur.get("env", {}))
                       if prev.get("env", {}).get(k) !=
                       cur.get("env", {}).get(k)}
        if env_changed:
            changed.add("env:" + ",".join(sorted(env_changed)))
        tools_changed = {t for t in set(prev.get("tools", {})) |
                         set(cur.get("tools", {}))
                         if prev.get("tools", {}).get(t) !=
                         cur.get("tools", {}).get(t)}
        if tools_changed:
            changed.add("tools:" + ",".join(sorted(tools_changed)))
        return {"from": prev.get("digest"), "to": cur.get("digest"),
                "changed": sorted(changed)}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        log = EventLog(Path(td) / "forge.jsonl")
        forge = Forge(log, cwd=td)

        d1 = forge.probe()
        assert d1["digest"] and len(d1["digest"]) == 16
        assert d1["python"] and d1["os"]

        # identical environment -> identical digest, no drift
        d2 = forge.probe()
        assert d1["digest"] == d2["digest"]
        assert forge.drift() is None

        # a material change (lockfile appears) -> drift detected
        (Path(td) / "requirements.txt").write_text("requests==2.32.0\n")
        d3 = forge.probe()
        assert d3["digest"] != d1["digest"]
        delta = forge.drift()
        assert delta is not None, "drift must be detected"
        assert any("lockfile" in c for c in delta["changed"]), delta

        # digests are sealed in the log
        assert len(fold(log).env_digests) == 3

    print("FORGE SELF-TEST PASS")
