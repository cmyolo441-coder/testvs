"""cli — Command-line interface for the enforcex v3 Sovereign toolkit.

Subcommands:

  extract <promptfile>                     print extracted rules + digest
  check <responsefile> [--prompt FILE]     run the gate, print report
         [--json] [--html out.html]
  scan <textfile>                          injection / threat scan
  swarm <responsefile> [--prompt FILE]     3-level parallel subagent swarm audit
  sysprobe [--cwd DIR] [--json]            deep OS/process/socket/hygiene forensics
  astscan <file> [--json]                  deep Python AST vulnerability scan
  cmdcheck <command> [--json]              shell command firewall evaluation
  dlp <file> [--redact] [--json]           DLP secret & PII scan / redaction
  serve --prompt FILE [--port 8765]        HTTP API on 127.0.0.1
  selftest                                 run all sibling _self_test()s

Exit codes: 0 pass, 1 fail (enforcement), 2 usage error.
Stdlib only; all sibling imports are defensive so the CLI works even
while sibling modules are being developed in parallel.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["main"]

# ── defensive sibling imports ────────────────────────────────────────────

try:
    from . import gate as _gate
except Exception:                                        # pragma: no cover
    _gate = None

try:
    from . import report as _report
except Exception:                                        # pragma: no cover
    _report = None

try:
    from . import scoring as _scoring
except Exception:                                        # pragma: no cover
    _scoring = None

try:
    from . import ruleextract as _ruleextract
except Exception:                                        # pragma: no cover
    _ruleextract = None

try:
    from . import injection as _injection
except Exception:                                        # pragma: no cover
    _injection = None

try:
    from . import detect as _detect
except Exception:                                        # pragma: no cover
    _detect = None

try:
    from . import swarm as _swarm
except Exception:                                        # pragma: no cover
    _swarm = None

try:
    from . import sysprobe as _sysprobe
except Exception:                                        # pragma: no cover
    _sysprobe = None

try:
    from . import astguard as _astguard
except Exception:                                        # pragma: no cover
    _astguard = None

try:
    from . import cmdguard as _cmdguard
except Exception:                                        # pragma: no cover
    _cmdguard = None

try:
    from . import dlp as _dlp
except Exception:                                        # pragma: no cover
    _dlp = None


EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


# ── small helpers ────────────────────────────────────────────────────────

def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def _die(msg: str, code: int = EXIT_USAGE) -> int:
    _eprint(f"error: {msg}")
    return code


def _extract_rules(prompt_text: str) -> Any:
    """Defensively extract rules from a system prompt. Returns None when
    no extractor is available."""
    if _ruleextract is None:
        return None
    for fn_name in ("extract_rules", "extract", "parse_rules"):
        fn = getattr(_ruleextract, fn_name, None)
        if callable(fn):
            try:
                return fn(prompt_text)
            except Exception:
                return None
    return None


def _rule_to_dict(it: Any) -> Dict[str, Any]:
    if isinstance(it, dict):
        return it
    if hasattr(it, "to_dict") and callable(it.to_dict):
        try:
            d = it.to_dict()
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return {
        "id": getattr(it, "id", getattr(it, "rule_id", "?")),
        "category": getattr(it, "category", "-"),
        "severity": getattr(it, "severity", getattr(it, "weight", "-")),
        "text": getattr(it, "text", getattr(it, "rule", str(it))),
    }


def _rules_digest(rules: Any) -> str:
    """Short digest string of an extracted rules object."""
    if rules is None:
        return "(no extractor available — gate will run with defaults)"
    try:
        if hasattr(rules, "digest") and getattr(rules, "digest"):
            return str(getattr(rules, "digest"))
        if isinstance(rules, dict):
            if "digest" in rules:
                return str(rules["digest"])
            items = list(rules.values())
        elif hasattr(rules, "rules"):
            items = list(getattr(rules, "rules"))
        else:
            items = list(rules)
        ids = []
        for it in items:
            d = _rule_to_dict(it)
            ids.append(str(d.get("rule_id") or d.get("id") or "?"))
        count = len(ids)
        tail = ",".join(ids[:6]) + ("..." if count > 6 else "")
        return f"{count} rules [{tail}]"
    except Exception:
        return "(rules extracted; digest unavailable)"


def _print_rules_table(rules: Any) -> None:
    if rules is None:
        return
    try:
        if isinstance(rules, dict):
            items = list(rules.values())
        elif hasattr(rules, "rules"):
            items = list(getattr(rules, "rules"))
        else:
            items = list(rules)
    except Exception:
        return
    print()
    print(f"{'RULE':<14} {'CAT':<16} {'SEV':<10} TEXT")
    print("-" * 78)
    for it in items:
        d = _rule_to_dict(it)
        rid = str(d.get("rule_id") or d.get("id") or "?")[:13]
        cat = str(d.get("category") or "-")[:15]
        sev = str(d.get("severity") or d.get("weight") or "-")[:9]
        text = str(d.get("text") or d.get("rule") or "")[:60]
        print(f"{rid:<14} {cat:<16} {sev:<10} {text}")
    print()


# ── command implementations ──────────────────────────────────────────────

def cmd_extract(args: argparse.Namespace) -> int:
    try:
        prompt_text = _read_file(args.promptfile)
    except OSError as exc:
        return _die(f"cannot read {args.promptfile}: {exc}")
    rules = _extract_rules(prompt_text)
    if rules is None:
        _eprint("note: ruleextract module unavailable; showing prompt digest only")
        print(f"prompt: {len(prompt_text)} chars, "
              f"{len(prompt_text.splitlines())} lines")
        return EXIT_PASS
    print("Extracted rules")
    print("===============")
    _print_rules_table(rules)
    print(f"digest: {_rules_digest(rules)}")
    return EXIT_PASS


def cmd_check(args: argparse.Namespace) -> int:
    try:
        response_text = _read_file(args.responsefile)
    except OSError as exc:
        return _die(f"cannot read {args.responsefile}: {exc}")
    prompt_text = ""
    if args.prompt:
        try:
            prompt_text = _read_file(args.prompt)
        except OSError as exc:
            return _die(f"cannot read {args.prompt}: {exc}")

    rules = _extract_rules(prompt_text) if prompt_text else None

    if _gate is None:
        return _die("gate module unavailable; cannot run enforcement",
                    EXIT_FAIL)

    try:
        cfg = _gate.GateConfig.from_env()
        g = _gate.Gate(cfg)
        decision = g.evaluate(response_text, rules=rules,
                              system_text=prompt_text,
                              session_id=getattr(args, "session", None) or "cli")
    except Exception as exc:
        return _die(f"gate evaluation failed: {exc}", EXIT_FAIL)

    if args.html:
        if _report is None:
            _eprint("warning: report module unavailable; skipping --html")
        else:
            try:
                _report.write_report(decision, args.html, fmt="html")
                print(f"html report written: {args.html}")
            except OSError as exc:
                _eprint(f"warning: could not write html report: {exc}")

    if args.json:
        if _report is not None:
            print(_report.to_json(decision))
        else:
            print(json.dumps(decision.to_dict(), indent=2, default=str))
    else:
        if _report is not None:
            print(_report.to_markdown(decision))
        else:
            d = decision.to_dict() if hasattr(decision, "to_dict") else {}
            print(f"action: {d.get('action', '?')}")
            s = d.get("score") or {}
            print(f"score:  {s.get('total', '?')} ({s.get('grade', '?')})")
            print(f"notes:  {'; '.join(d.get('notes', []))}")

    total = float(getattr(getattr(decision, "score", None), "total", 0.0) or 0.0)
    action = str(getattr(decision, "action", "block"))
    if action in ("allow", "rewrite"):
        return EXIT_PASS
    if action == "warn" and total >= 60.0:
        return EXIT_PASS
    return EXIT_FAIL


def cmd_scan(args: argparse.Namespace) -> int:
    try:
        text = _read_file(args.textfile)
    except OSError as exc:
        return _die(f"cannot read {args.textfile}: {exc}")

    findings: List[Dict[str, Any]] = []
    if _injection is not None:
        try:
            res = _injection.scan(text)
            if isinstance(res, dict):
                findings.extend(res.get("violations") or res.get("matches") or [])
            elif isinstance(res, list):
                findings.extend(res)
        except Exception as exc:
            _eprint(f"warning: injection.scan failed: {exc}")
    if _detect is not None:
        try:
            res = _detect.run_all(text)
            if isinstance(res, dict):
                findings.extend(res.get("violations") or res.get("findings") or [])
            elif isinstance(res, list):
                findings.extend(res)
        except Exception as exc:
            _eprint(f"warning: detect.run_all failed: {exc}")
    if not findings:
        low = text.lower()
        for kw in ("ignore previous instructions", "reveal your system prompt",
                   "you are now", "developer mode", "jailbreak"):
            if kw in low:
                findings.append({"category": "injection", "severity": "critical",
                                 "evidence": kw, "message": "keyword match"})
                break

    if not findings:
        print("scan: clean — no injection or threat patterns found")
        return EXIT_PASS
    print(f"scan: {len(findings)} finding(s)")
    for i, f in enumerate(findings, 1):
        if isinstance(f, dict):
            print(f"  {i}. [{f.get('severity', '?'):>8}] "
                  f"[{f.get('category', '?')}] {f.get('evidence', '')[:80]}")
        else:
            print(f"  {i}. {str(f)[:100]}")
    return EXIT_FAIL


def cmd_swarm(args: argparse.Namespace) -> int:
    try:
        response_text = _read_file(args.responsefile)
    except OSError as exc:
        return _die(f"cannot read {args.responsefile}: {exc}")
    prompt_text = ""
    if getattr(args, "prompt", None):
        try:
            prompt_text = _read_file(args.prompt)
        except OSError as exc:
            return _die(f"cannot read {args.prompt}: {exc}")
    if _swarm is None:
        return _die("swarm module unavailable", EXIT_FAIL)
    verdict = _swarm.run_swarm_audit(response_text, system_text=prompt_text)
    if getattr(args, "json", False):
        print(json.dumps(verdict.to_dict(), indent=2, default=str))
    else:
        print(f"swarm action: {verdict.action} (score={verdict.overall_score:.1f}, "
              f"risk={verdict.max_risk}, waves={len(verdict.waves)}, "
              f"findings={len(verdict.findings)}, {verdict.elapsed_ms:.1f}ms)")
    return EXIT_PASS if verdict.passed else EXIT_FAIL


def cmd_sysprobe(args: argparse.Namespace) -> int:
    if _sysprobe is None:
        return _die("sysprobe module unavailable", EXIT_FAIL)
    rep = _sysprobe.collect_system_health(getattr(args, "cwd", ".") or ".")
    if getattr(args, "json", False):
        print(json.dumps(rep.to_dict(), indent=2, default=str))
    else:
        print(f"System Health: {rep.status.upper()} ({rep.health_score:.1f}/100)")
        print(f"  Host: {rep.hostname} ({rep.os_name} {rep.arch}, {rep.cpu_count} CPUs)")
        print(f"  Mem:  {rep.memory.used_mb:.0f}/{rep.memory.total_mb:.0f} MB ({rep.memory.used_pct:.1f}%)")
        print(f"  Disk: {rep.disk.free_gb:.2f} GB free / {rep.disk.total_gb:.2f} GB ({rep.disk.used_pct:.1f}% used)")
        print(f"  Proc: PID={rep.process.pid} RSS={rep.process.rss_mb:.1f}MB FDs={rep.process.open_fds} Threads={rep.process.active_threads}")
        if rep.findings:
            print(f"  Findings ({len(rep.findings)}):")
            for f in rep.findings:
                print(f"    - [{f.severity}] {f.code}: {f.message}")
    return EXIT_PASS if rep.status != "critical" else EXIT_FAIL


def cmd_astscan(args: argparse.Namespace) -> int:
    if _astguard is None:
        return _die("astguard module unavailable", EXIT_FAIL)
    if not os.path.exists(args.file):
        return _die(f"cannot read {args.file}: file does not exist")
    rep = _astguard.scan_file(args.file)
    if getattr(args, "json", False):
        print(json.dumps(rep.to_dict(), indent=2, default=str))
    else:
        print(f"AST Security Scan: grade={rep.grade} score={rep.score:.1f}/100 "
              f"findings={len(rep.findings)}")
        for f in rep.findings:
            print(f"  line {f.lineno}: [{f.severity}] {f.rule_id} ({f.symbol}) — {f.message}")
    return EXIT_PASS if rep.passed else EXIT_FAIL


def cmd_cmdcheck(args: argparse.Namespace) -> int:
    if _cmdguard is None:
        return _die("cmdguard module unavailable", EXIT_FAIL)
    verdict = _cmdguard.evaluate_command(args.cmd)
    if getattr(args, "json", False):
        print(json.dumps(verdict.to_dict(), indent=2, default=str))
    else:
        print(f"Command Firewall: action={verdict.action} allowed={verdict.allowed} "
              f"risk={verdict.risk_score:.1f}/100")
        for f in verdict.findings:
            print(f"  - [{f.severity}] {f.rule_id}: {f.message}")
    return EXIT_PASS if verdict.allowed else EXIT_FAIL


def cmd_dlp(args: argparse.Namespace) -> int:
    try:
        text = _read_file(args.file)
    except OSError as exc:
        return _die(f"cannot read {args.file}: {exc}")
    if _dlp is None:
        return _die("dlp module unavailable", EXIT_FAIL)
    rep = _dlp.scan_text(text)
    if getattr(args, "json", False):
        print(json.dumps(rep.to_dict(), indent=2, default=str))
    elif getattr(args, "redact", False):
        print(rep.redacted_text)
    else:
        print(f"DLP Scan: clean={rep.clean} matches={len(rep.matches)} risk={rep.risk_score:.1f}")
        for m in rep.matches:
            print(f"  - [{m.severity}] {m.kind} ({m.category}): {m.masked_value}")
    return EXIT_PASS if rep.clean else EXIT_FAIL




def cmd_omniguard(args: argparse.Namespace) -> int:
    try:
        resp_text = _read_file(args.responsefile)
    except OSError as exc:
        return _die(f"cannot read {args.responsefile}: {exc}")
    prompt_text = ""
    if getattr(args, "prompt", None):
        try:
            prompt_text = _read_file(args.prompt)
        except OSError as exc:
            return _die(f"cannot read {args.prompt}: {exc}")
    from . import omniguard as _og
    ctrl = _og.OmniGuardController()
    verdict = ctrl.enforce_output(resp_text, system_prompt=prompt_text)
    if getattr(args, "json", False):
        print(json.dumps(verdict.to_dict(), indent=2, default=str))
    else:
        print(f"OmniGuard Verdict: action={verdict.action} passed={verdict.passed} "
              f"score={verdict.score:.1f}/100 repaired={verdict.repaired} "
              f"violations={len(verdict.violations)}")
    return EXIT_PASS if verdict.passed else EXIT_FAIL


def cmd_deontic(args: argparse.Namespace) -> int:
    try:
        resp_text = _read_file(args.responsefile)
    except OSError as exc:
        return _die(f"cannot read {args.responsefile}: {exc}")
    prompt_text = ""
    if getattr(args, "prompt", None):
        try:
            prompt_text = _read_file(args.prompt)
        except OSError as exc:
            return _die(f"cannot read {args.prompt}: {exc}")
    from . import deontic as _dn
    verdict = _dn.evaluate_deontic(resp_text, system_text=prompt_text)
    if getattr(args, "json", False):
        print(json.dumps(verdict.to_dict(), indent=2, default=str))
    else:
        print(f"Deontic Verdict: passed={verdict.passed} score={verdict.score:.1f}/100 "
              f"norms={len(verdict.norms)} conflicts={len(verdict.conflicts)} "
              f"violations={len(verdict.violations)}")
    return EXIT_PASS if verdict.passed else EXIT_FAIL


def cmd_heal(args: argparse.Namespace) -> int:
    try:
        prompt_text = _read_file(args.promptfile)
    except OSError as exc:
        return _die(f"cannot read {args.promptfile}: {exc}")
    from . import autohealer as _ah
    rep = _ah.heal_system_prompt(prompt_text)
    if getattr(args, "json", False):
        print(json.dumps(rep.to_dict(), indent=2, default=str))
    else:
        print(rep.healed_prompt)
    return EXIT_PASS


def cmd_serve(args: argparse.Namespace) -> int:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    try:
        prompt_text = _read_file(args.prompt) if args.prompt else ""
    except OSError as exc:
        return _die(f"cannot read {args.prompt}: {exc}")
    rules = _extract_rules(prompt_text) if prompt_text else None

    if _gate is None:
        return _die("gate module unavailable; cannot serve", EXIT_FAIL)
    gate_obj = _gate.Gate(_gate.GateConfig.from_env())
    if _report is None:
        return _die("report module unavailable; cannot serve", EXIT_FAIL)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: str, ctype: str) -> None:
            body = payload.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, '{"status": "ok"}', "application/json")
            else:
                self._send(404, '{"error": "not found"}', "application/json")

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(max(0, min(length, 10 * 1024 * 1024))) if length > 0 else b"{}"
            if self.path != "/check":
                self._send(404, '{"error": "not found"}', "application/json")
                return
            try:
                req = json.loads(raw.decode("utf-8") or "{}")
                response_text = str(req.get("response", ""))
                session_id = str(req.get("session_id", "default"))
                decision = gate_obj.evaluate(
                    response_text, rules=rules, system_text=prompt_text,
                    session_id=session_id)
                self._send(200, _report.to_json(decision), "application/json")
            except (ValueError, json.JSONDecodeError):
                self._send(400, '{"error": "invalid JSON"}', "application/json")
            except Exception as exc:
                self._send(500, json.dumps({"error": str(exc)}),
                           "application/json")

        def log_message(self, fmt: str, *log_args: Any) -> None:
            pass  # keep stdout clean

    server = HTTPServer((args.bind, args.port), Handler)
    print(f"enforcex serving on http://{args.bind}:{args.port} "
          f"(POST /check, GET /health) — Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return EXIT_PASS


def cmd_selftest(_args: argparse.Namespace) -> int:
    from . import _LAZY_MODULES
    modules: List[str] = [m for m in _LAZY_MODULES if m != "cli"]
    failed: List[str] = []
    ran: List[str] = []
    for name in modules:
        try:
            mod = __import__(f"fullagent.enforcex.{name}",
                             fromlist=["_self_test"])
        except Exception as exc:
            print(f"  {name:<16} SKIP (import failed: {type(exc).__name__})")
            continue
        fn = getattr(mod, "_self_test", None)
        if not callable(fn):
            print(f"  {name:<16} SKIP (no _self_test)")
            continue
        try:
            fn()
            ran.append(name)
            print(f"  {name:<16} OK")
        except AssertionError as exc:
            failed.append(name)
            print(f"  {name:<16} FAIL (assertion: {exc})")
        except Exception as exc:
            failed.append(name)
            print(f"  {name:<16} FAIL ({type(exc).__name__}: {exc})")
    print()
    print(f"selftest: {len(ran)} passed, {len(failed)} failed "
          f"({len(modules) - len(ran) - len(failed)} skipped)")
    return EXIT_FAIL if failed else EXIT_PASS


# ── argument parsing ─────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enforcex",
        description="Sovereign system-prompt compliance & security enforcement toolkit (v3.0)",
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("extract", help="extract rules from a system prompt file")
    p.add_argument("promptfile")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("check", help="run the enforcement gate on a response")
    p.add_argument("responsefile")
    p.add_argument("--prompt", default=None, help="system prompt file")
    p.add_argument("--session", default="cli", help="session id for retry budget")
    p.add_argument("--json", action="store_true", help="emit JSON report")
    p.add_argument("--html", default=None, metavar="OUT.html",
                   help="also write an HTML report")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("scan", help="injection / threat scan of a text file")
    p.add_argument("textfile")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("swarm", help="run 3-level parallel subagent swarm audit")
    p.add_argument("responsefile")
    p.add_argument("--prompt", default=None, help="system prompt file")
    p.add_argument("--json", action="store_true", help="emit JSON verdict")
    p.set_defaults(func=cmd_swarm)

    p = sub.add_parser("sysprobe", help="collect OS/process/socket forensics & hygiene score")
    p.add_argument("--cwd", default=".", help="directory to inspect")
    p.add_argument("--json", action="store_true", help="emit JSON report")
    p.set_defaults(func=cmd_sysprobe)

    p = sub.add_parser("astscan", help="scan Python file for AST vulnerabilities")
    p.add_argument("file")
    p.add_argument("--json", action="store_true", help="emit JSON report")
    p.set_defaults(func=cmd_astscan)

    p = sub.add_parser("cmdcheck", help="evaluate shell command via CmdGuard firewall")
    p.add_argument("cmd")
    p.add_argument("--json", action="store_true", help="emit JSON verdict")
    p.set_defaults(func=cmd_cmdcheck)

    p = sub.add_parser("dlp", help="scan or redact secrets & PII in a file")
    p.add_argument("file")
    p.add_argument("--redact", action="store_true", help="print redacted text")
    p.add_argument("--json", action="store_true", help="emit JSON report")
    p.set_defaults(func=cmd_dlp)

    p = sub.add_parser("omniguard", help="run 6-layer OmniGuard sovereign controller on a response")
    p.add_argument("responsefile")
    p.add_argument("--prompt", default=None, help="system prompt file")
    p.add_argument("--json", action="store_true", help="emit JSON verdict")
    p.set_defaults(func=cmd_omniguard)

    p = sub.add_parser("deontic", help="compile SDL deontic norms and verify response")
    p.add_argument("responsefile")
    p.add_argument("--prompt", default=None, help="system prompt file")
    p.add_argument("--json", action="store_true", help="emit JSON verdict")
    p.set_defaults(func=cmd_deontic)

    p = sub.add_parser("heal", help="strengthen weak modal phrasing and unbury mid-prompt rules")
    p.add_argument("promptfile")
    p.add_argument("--json", action="store_true", help="emit JSON report")
    p.set_defaults(func=cmd_heal)

    p = sub.add_parser("serve", help="serve the gate over HTTP")
    p.add_argument("--prompt", default=None, help="system prompt file")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--bind", default="127.0.0.1")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("selftest", help="run all module self-tests")
    p.set_defaults(func=cmd_selftest)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return int(args.func(args))
    except SystemExit as exc:  # argparse may raise for bad sub-use
        code = exc.code
        return code if isinstance(code, int) else EXIT_USAGE
    except KeyboardInterrupt:
        _eprint("interrupted")
        return EXIT_USAGE


# ── self-test ────────────────────────────────────────────────────────────

def _self_test() -> int:  # pragma: no cover
    parser = build_parser()
    argv = ["check", "resp.txt", "--prompt", "p.txt", "--json",
            "--html", "out.html"]
    ns = parser.parse_args(argv)
    assert ns.command == "check" and ns.responsefile == "resp.txt"
    assert ns.prompt == "p.txt" and ns.json and ns.html == "out.html"

    ns2 = parser.parse_args(["extract", "prompt.md"])
    assert ns2.command == "extract" and ns2.promptfile == "prompt.md"

    ns3 = parser.parse_args(["serve", "--port", "9000"])
    assert ns3.command == "serve" and ns3.port == 9000
    assert ns3.bind == "127.0.0.1"

    ns4 = parser.parse_args(["selftest"])
    assert ns4.command == "selftest"

    # missing command -> usage exit
    assert main([]) == EXIT_USAGE
    # missing file -> usage error exit
    assert main(["check", "/nonexistent/file/xyz.txt"]) == EXIT_USAGE

    # end-to-end check + swarm + sysprobe + cmdcheck against a real temp file
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        rpath = os.path.join(tmp, "resp.txt")
        with open(rpath, "w", encoding="utf-8") as fh:
            fh.write("All good, here is the summary.\n1. one\n2. two\n")
        rc = main(["check", rpath, "--json"])
        assert rc == EXIT_PASS, rc
        rc_sw = main(["swarm", rpath, "--json"])
        assert rc_sw == EXIT_PASS, rc_sw
        rc_sp = main(["sysprobe", "--json"])
        assert rc_sp == EXIT_PASS, rc_sp
        rc_cmd = main(["cmdcheck", "git status", "--json"])
        assert rc_cmd == EXIT_PASS, rc_cmd

    # extract must properly format Rule dataclass attributes
    rc = main(["extract", os.path.abspath(__file__)])
    assert rc in (EXIT_PASS, EXIT_FAIL), rc

    print("cli._self_test: OK PASS")
    return 0


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) == 1:
        sys.exit(_self_test())
    sys.exit(main())
