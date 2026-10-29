"""sysprobe — Deep System & OS Forensics, Resource Telemetry & Security Hygiene Engine.

Self-test: ``python3 -m fullagent.enforcex.sysprobe``
"""

from __future__ import annotations

import gc
import os
import platform
import resource
import stat
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "MemoryTelemetry",
    "DiskTelemetry",
    "ProcessTelemetry",
    "SocketEntry",
    "HygieneFinding",
    "SystemHealthReport",
    "probe_memory",
    "probe_disk",
    "probe_process",
    "probe_listening_sockets",
    "audit_environment_hygiene",
    "collect_system_health",
]


@dataclass
class MemoryTelemetry:
    total_mb: float = 0.0
    available_mb: float = 0.0
    used_mb: float = 0.0
    used_pct: float = 0.0
    swap_total_mb: float = 0.0
    swap_free_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_mb": round(self.total_mb, 2),
            "available_mb": round(self.available_mb, 2),
            "used_mb": round(self.used_mb, 2),
            "used_pct": round(self.used_pct, 2),
            "swap_total_mb": round(self.swap_total_mb, 2),
            "swap_free_mb": round(self.swap_free_mb, 2),
        }


@dataclass
class DiskTelemetry:
    path: str = "."
    total_gb: float = 0.0
    free_gb: float = 0.0
    used_gb: float = 0.0
    used_pct: float = 0.0
    inodes_free: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "total_gb": round(self.total_gb, 3),
            "free_gb": round(self.free_gb, 3),
            "used_gb": round(self.used_gb, 3),
            "used_pct": round(self.used_pct, 2),
            "inodes_free": self.inodes_free,
        }


@dataclass
class ProcessTelemetry:
    pid: int = 0
    ppid: int = 0
    uid: int = -1
    rss_mb: float = 0.0
    open_fds: int = 0
    fd_soft_limit: int = 0
    active_threads: int = 0
    thread_names: List[str] = field(default_factory=list)
    gc_counts: Tuple[int, int, int] = (0, 0, 0)
    python_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "ppid": self.ppid,
            "uid": self.uid,
            "rss_mb": round(self.rss_mb, 2),
            "open_fds": self.open_fds,
            "fd_soft_limit": self.fd_soft_limit,
            "active_threads": self.active_threads,
            "thread_names": list(self.thread_names),
            "gc_counts": list(self.gc_counts),
            "python_version": self.python_version,
        }


@dataclass
class SocketEntry:
    local_ip: str
    local_port: int
    state: str = "LISTEN"
    ipv6: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "local_ip": self.local_ip,
            "local_port": self.local_port,
            "state": self.state,
            "ipv6": self.ipv6,
        }


@dataclass
class HygieneFinding:
    code: str
    severity: str
    message: str
    evidence: str = ""
    remediation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": "system_hygiene",
            "rule_id": self.code,
            "severity": self.severity,
            "message": self.message,
            "evidence": self.evidence,
            "remediation": self.remediation,
        }


@dataclass
class SystemHealthReport:
    ts: float = 0.0
    hostname: str = ""
    os_name: str = ""
    arch: str = ""
    cpu_count: int = 1
    load_avg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    memory: MemoryTelemetry = field(default_factory=MemoryTelemetry)
    disk: DiskTelemetry = field(default_factory=DiskTelemetry)
    process: ProcessTelemetry = field(default_factory=ProcessTelemetry)
    listening_sockets: List[SocketEntry] = field(default_factory=list)
    findings: List[HygieneFinding] = field(default_factory=list)
    health_score: float = 100.0
    status: str = "healthy"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "hostname": self.hostname,
            "os_name": self.os_name,
            "arch": self.arch,
            "cpu_count": self.cpu_count,
            "load_avg": [round(x, 3) for x in self.load_avg],
            "memory": self.memory.to_dict(),
            "disk": self.disk.to_dict(),
            "process": self.process.to_dict(),
            "listening_sockets": [s.to_dict() for s in self.listening_sockets],
            "findings": [f.to_dict() for f in self.findings],
            "health_score": round(self.health_score, 2),
            "status": self.status,
        }


def probe_memory() -> MemoryTelemetry:
    info: Dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split(":")
                if len(parts) == 2:
                    k = parts[0].strip()
                    v_parts = parts[1].strip().split()
                    if v_parts and v_parts[0].isdigit():
                        info[k] = int(v_parts[0])
    except OSError:
        return MemoryTelemetry()

    total_kb = info.get("MemTotal", 0)
    avail_kb = info.get("MemAvailable", info.get("MemFree", 0))
    used_kb = max(0, total_kb - avail_kb)
    total_mb = total_kb / 1024.0
    avail_mb = avail_kb / 1024.0
    used_mb = used_kb / 1024.0
    used_pct = (used_kb / total_kb * 100.0) if total_kb > 0 else 0.0
    swap_total = info.get("SwapTotal", 0) / 1024.0
    swap_free = info.get("SwapFree", 0) / 1024.0
    return MemoryTelemetry(
        total_mb=total_mb,
        available_mb=avail_mb,
        used_mb=used_mb,
        used_pct=used_pct,
        swap_total_mb=swap_total,
        swap_free_mb=swap_free,
    )


def probe_disk(path: str = ".") -> DiskTelemetry:
    target = str(path or ".")
    try:
        st = os.statvfs(target)
        total_b = st.f_blocks * st.f_frsize
        avail_b = st.f_bavail * st.f_frsize
        used_b = max(0, total_b - avail_b)
        gb = 1024.0 ** 3
        pct = (used_b / total_b * 100.0) if total_b > 0 else 0.0
        return DiskTelemetry(
            path=target,
            total_gb=total_b / gb,
            free_gb=avail_b / gb,
            used_gb=used_b / gb,
            used_pct=pct,
            inodes_free=int(st.f_favail),
        )
    except Exception:
        return DiskTelemetry(path=target)


def probe_process() -> ProcessTelemetry:
    pid = os.getpid()
    ppid = os.getppid() if hasattr(os, "getppid") else 0
    uid = os.getuid() if hasattr(os, "getuid") else -1
    rss_mb = 0.0
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = (ru / (1024.0 * 1024.0)) if sys.platform == "darwin" else (ru / 1024.0)
    except Exception:
        pass
    open_fds = 0
    try:
        fd_dir = Path("/proc/self/fd")
        if fd_dir.is_dir():
            open_fds = len(list(fd_dir.iterdir()))
    except Exception:
        pass
    soft_lim = 0
    try:
        soft_lim, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception:
        pass
    threads = threading.enumerate()
    tnames = sorted({t.name for t in threads})[:24]
    gc_c = gc.get_count()
    return ProcessTelemetry(
        pid=pid,
        ppid=ppid,
        uid=uid,
        rss_mb=rss_mb,
        open_fds=open_fds,
        fd_soft_limit=int(soft_lim) if soft_lim > 0 else 0,
        active_threads=len(threads),
        thread_names=tnames,
        gc_counts=(int(gc_c[0]), int(gc_c[1]), int(gc_c[2])),
        python_version=sys.version.split()[0],
    )


def _decode_proc_ip(hex_ip: str, ipv6: bool = False) -> str:
    try:
        if not ipv6 and len(hex_ip) == 8:
            b = bytes.fromhex(hex_ip)
            return f"{b[3]}.{b[2]}.{b[1]}.{b[0]}"
        if ipv6 and len(hex_ip) == 32:
            if hex_ip == "0" * 32:
                return "::"
            if hex_ip == "0" * 24 + "01000000" or hex_ip.endswith("00000001"):
                return "::1"
            return "::"
    except Exception:
        pass
    return "0.0.0.0"


def _fallback_listening_sockets(max_entries: int = 64) -> List[SocketEntry]:
    """Best-effort listening sockets via ss/netstat when /proc/net is absent.

    Every subprocess call is timeout-bounded (never hangs) and any failure
    (missing command, permission, timeout, unparsable output) yields [].
    """
    out: List[SocketEntry] = []
    if max_entries <= 0:
        return out
    try:
        import subprocess
    except Exception:
        return out
    for argv in (["ss", "-H", "-ltn"], ["netstat", "-ltn"]):
        try:
            cp = subprocess.run(  # noqa: S603 - fixed argv, no shell
                argv, capture_output=True, text=True, timeout=2.0, check=False,
            )
        except Exception:
            continue
        if getattr(cp, "returncode", 1) != 0:
            continue
        parsed_any = False
        for line in str(getattr(cp, "stdout", "") or "").splitlines()[:256]:
            cols = line.split()
            if len(cols) < 4:
                continue
            addr = cols[-2] if len(cols) >= 5 else cols[-1]
            ip, _, port = addr.rpartition(":")
            if not port.isdigit():
                continue
            parsed_any = True
            entry = SocketEntry(
                local_ip=ip.strip("[]") or "0.0.0.0",
                local_port=int(port),
                state="LISTEN",
                ipv6=":" in ip,
            )
            if entry not in out:
                out.append(entry)
                if len(out) >= max_entries:
                    return out
        if parsed_any:
            return out
    return out


def probe_listening_sockets(max_entries: int = 64) -> List[SocketEntry]:
    try:
        max_entries = int(max_entries)
    except (TypeError, ValueError, OverflowError):
        max_entries = 64
    max_entries = max(0, min(1024, max_entries))
    if max_entries == 0:
        return []
    out: List[SocketEntry] = []
    seen = set()
    proc_read = False
    for path, is_v6 in (("/proc/net/tcp", False), ("/proc/net/tcp6", True)):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.read().splitlines()[1:]
            proc_read = True
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            local_addr, state_hex = parts[1], parts[3]
            if state_hex != "0A" or ":" not in local_addr:
                continue
            ip_hex, port_hex = local_addr.split(":", 1)
            try:
                port = int(port_hex, 16)
            except ValueError:
                continue
            ip_str = _decode_proc_ip(ip_hex, ipv6=is_v6)
            key = (ip_str, port, is_v6)
            if key not in seen:
                seen.add(key)
                out.append(SocketEntry(local_ip=ip_str, local_port=port, state="LISTEN", ipv6=is_v6))
                if len(out) >= max_entries:
                    return out
    if not proc_read:
        # /proc unavailable (non-Linux, container, restricted): best-effort CLI.
        out.extend(_fallback_listening_sockets(max_entries - len(out)))
    return out


_SENSITIVE_ENV_RE_KEYS = (
    "SECRET", "PASSWORD", "PRIVATE_KEY", "TOKEN", "API_KEY", "APIKEY",
    "AWS_SECRET", "ACCESS_KEY", "CREDENTIAL", "AUTH",
)
_SECRET_VALUE_PREFIXES = ("AKIA", "ASIA", "ghp_", "gho_", "xox", "-----BEGIN")
_MAX_PATH_ENTRIES = 64


def _audit_path_hygiene(env: Optional[Dict[str, str]] = None) -> List[HygieneFinding]:
    """Bounded check: world-writable directories on PATH are privilege risks."""
    findings: List[HygieneFinding] = []
    source = env if env is not None else os.environ
    try:
        raw_path = str(dict(source).get("PATH", "") or "")
    except Exception:
        return findings
    bad: List[str] = []
    for entry in raw_path.split(os.pathsep)[:_MAX_PATH_ENTRIES]:
        candidate = entry if entry else "."
        try:
            st = os.stat(candidate)
        except (OSError, ValueError, TypeError):
            continue
        try:
            if stat.S_ISDIR(st.st_mode) and (st.st_mode & stat.S_IWOTH):
                bad.append(candidate)
        except Exception:
            continue
    if bad:
        findings.append(
            HygieneFinding(
                code="HYG-PATH-WORLD-WRITABLE",
                severity="medium",
                message=f"{len(bad)} world-writable PATH entr(ies) allow binary hijacking",
                evidence=", ".join(bad[:6]),
                remediation="Remove world-writable dirs from PATH or chmod 755 them.",
            )
        )
    return findings


def audit_environment_hygiene(
    cwd: str = ".",
    environ: Optional[Dict[str, str]] = None,
) -> List[HygieneFinding]:
    findings: List[HygieneFinding] = []
    env = environ if environ is not None else dict(os.environ)
    try:
        items = list(env.items())[:512]
    except Exception:
        items = []

    exposed_keys = []
    leaked_values = []
    for k, v in items:
        ku = str(k).upper()
        sv = str(v)[:1024] if v is not None else ""
        if any(tok in ku for tok in _SENSITIVE_ENV_RE_KEYS) and v and len(sv) >= 8:
            exposed_keys.append(str(k))
        elif sv.startswith(_SECRET_VALUE_PREFIXES) or "PRIVATE KEY" in sv.upper():
            leaked_values.append(str(k))
    if leaked_values:
        findings.append(
            HygieneFinding(
                code="HYG-ENV-SECRET-VALUES",
                severity="medium",
                message=f"{len(leaked_values)} env var(s) carry credential-shaped values",
                evidence=", ".join(sorted(leaked_values)[:6]),
                remediation="Load secrets from a vault at runtime instead of the environment.",
            )
        )
    if exposed_keys:
        findings.append(
            HygieneFinding(
                code="HYG-ENV-SECRETS",
                severity="low",
                message=f"{len(exposed_keys)} sensitive environment variable(s) present in process env",
                evidence=", ".join(sorted(exposed_keys)[:6]),
                remediation="Avoid passing sensitive env vars to untrusted subprocesses.",
            )
        )

    if hasattr(os, "getuid"):
        try:
            if os.getuid() == 0:
                findings.append(
                    HygieneFinding(
                        code="HYG-ROOT-USER",
                        severity="medium",
                        message="Process is executing with UID 0 (root) privileges",
                        evidence="uid=0",
                        remediation="Run agent workloads under an unprivileged user account.",
                    )
                )
        except Exception:
            pass

    try:
        st = os.stat(str(cwd) if cwd else ".")
        mode = st.st_mode
        if (mode & stat.S_IWOTH) and not (mode & stat.S_ISVTX):
            findings.append(
                HygieneFinding(
                    code="HYG-WORLD-WRITABLE-CWD",
                    severity="high",
                    message="Working directory is world-writable without sticky bit",
                    evidence=f"path={cwd} mode={oct(mode & 0o777)}",
                    remediation="Restrict directory permissions with chmod 755 or 700.",
                )
            )
    except Exception:
        pass

    for sock in probe_listening_sockets():
        if sock.local_ip in ("0.0.0.0", "::"):
            findings.append(
                HygieneFinding(
                    code="HYG-WILDCARD-LISTEN",
                    severity="low",
                    message=f"Socket listening on all interfaces ({sock.local_ip}:{sock.local_port})",
                    evidence=f"{sock.local_ip}:{sock.local_port}",
                    remediation="Bind local agent/webhook services to 127.0.0.1.",
                )
            )

    try:
        findings.extend(_audit_path_hygiene(env if environ is not None else None))
    except Exception:
        pass
    return findings


def collect_system_health(cwd: str = ".") -> SystemHealthReport:
    mem = probe_memory()
    disk = probe_disk(cwd)
    proc = probe_process()
    socks = probe_listening_sockets()
    findings = audit_environment_hygiene(cwd=cwd)

    load = (0.0, 0.0, 0.0)
    if hasattr(os, "getloadavg"):
        try:
            la = os.getloadavg()
            load = (float(la[0]), float(la[1]), float(la[2]))
        except OSError:
            pass

    cpus = os.cpu_count() or 1
    score = 100.0
    if mem.used_pct >= 95.0:
        score -= 30.0
    elif mem.used_pct >= 85.0:
        score -= 10.0

    if disk.used_pct >= 95.0:
        score -= 25.0
    elif disk.used_pct >= 88.0:
        score -= 10.0

    if proc.fd_soft_limit > 0 and proc.open_fds > proc.fd_soft_limit * 0.8:
        score -= 20.0

    for f in findings:
        if f.code.startswith("HYG-"):
            if f.severity == "high":
                score -= 15.0
            elif f.severity == "medium":
                score -= 8.0
            elif f.severity == "low":
                score -= 2.0

    score = max(0.0, min(100.0, score))
    status = "healthy" if score >= 80.0 else ("degraded" if score >= 50.0 else "critical")
    return SystemHealthReport(
        ts=time.time(),
        hostname=platform.node(),
        os_name=platform.system(),
        arch=platform.machine(),
        cpu_count=cpus,
        load_avg=load,
        memory=mem,
        disk=disk,
        process=proc,
        listening_sockets=socks,
        findings=findings,
        health_score=score,
        status=status,
    )


def _self_test() -> int:
    rep = collect_system_health(".")
    d = rep.to_dict()
    assert rep.cpu_count >= 1
    assert rep.process.pid == os.getpid()
    assert rep.process.active_threads >= 1
    assert 0.0 <= rep.health_score <= 100.0
    assert rep.status in ("healthy", "degraded", "critical")
    assert isinstance(d["memory"], dict) and isinstance(d["disk"], dict)

    hyg = audit_environment_hygiene(".", environ={"MY_SECRET_TOKEN": "supersecret12345"})
    assert any(f.code == "HYG-ENV-SECRETS" for f in hyg), hyg

    with tempfile.TemporaryDirectory() as tmp:
        os.chmod(tmp, 0o777)
        hyg2 = audit_environment_hygiene(tmp, environ={})
        assert any(f.code == "HYG-WORLD-WRITABLE-CWD" for f in hyg2)

    # adversarial: malformed/hostile inputs must degrade, never raise
    rep_none = collect_system_health(None)  # type: ignore[arg-type]
    assert rep_none.status in ("healthy", "degraded", "critical")
    assert rep_none.process.pid == os.getpid()
    assert 0.0 <= rep_none.health_score <= 100.0
    deep = collect_system_health("/nonexistent/__nope__/" + "x" * 9_000)
    assert deep.disk.total_gb == 0.0 and isinstance(deep.to_dict(), dict)
    assert audit_environment_hygiene(None, environ=None) is not None
    assert audit_environment_hygiene(".", environ={"WEIRD": None, 7: "x" * 5000}) is not None  # type: ignore[dict-item]
    assert probe_listening_sockets(0) == []
    assert probe_listening_sockets(-5) == []
    assert isinstance(probe_listening_sockets("bogus"), list)  # type: ignore[arg-type]
    assert isinstance(probe_listening_sockets(10 ** 9), list)

    # credential-shaped values leak even with innocuous key names
    hyg3 = audit_environment_hygiene(".", environ={"ODD_NAME": "AKIAIOSFODNN7EXAMPLE"})
    assert any(f.code == "HYG-ENV-SECRET-VALUES" for f in hyg3), hyg3

    # world-writable PATH directory is flagged (bounded)
    with tempfile.TemporaryDirectory() as tmp2:
        os.chmod(tmp2, 0o777)
        hyg4 = audit_environment_hygiene(
            ".", environ={"PATH": tmp2 + os.pathsep + "/usr/bin"})
        assert any(f.code == "HYG-PATH-WORLD-WRITABLE" for f in hyg4), hyg4
        hyg5 = audit_environment_hygiene(".", environ={"PATH": ""})
        assert not any(f.code == "HYG-PATH-WORLD-WRITABLE" for f in hyg5)

    print("sysprobe._self_test: OK PASS (memory, disk, process, sockets, hygiene)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
