"""sovereign — Unified Sovereign Runtime Bridge & FullAgent Integration Layer for enforcex v4.0.

Self-test: ``python3 -m fullagent.enforcex.sovereign``
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Sequence, Tuple

from . import (
    astguard,
    autohealer,
    canary,
    circuit,
    cmdguard,
    deontic,
    dlp,
    formalverify,
    omniguard,
    orchestrator,
    profiler,
    ruleextract,
    swarm,
    sysprobe,
    toolguard,
)

__all__ = [
    "SovereignRuntime",
    "register",
    "handle_enforcex",
]


class SovereignRuntime:
    def __init__(self, workspace_root: Optional[str] = None) -> None:
        self.workspace_root = workspace_root
        self.profiler = profiler.ExecutionProfiler()
        self.breaker = circuit.CircuitBreaker("enforcex.sovereign", failure_threshold=5, recovery_timeout_s=10.0)
        self.orchestrator = orchestrator.Orchestrator()
        self.swarm = swarm.build_default_security_swarm()
        self.tool_guard = toolguard.ToolGuard(
            toolguard.ToolPolicy(workspace_root=workspace_root)
        )
        self.canary_engine = canary.CanaryEngine()
        self.redactor = dlp.VaultedRedactor()
        self.verifier = formalverify.FormalVerifier()
        self.omni = omniguard.OmniGuardController()
        self.audits_run = 0
        self.blocks_triggered = 0

    def omniguard_enforce(
        self,
        response_text: str,
        *,
        system_text: str = "",
        user_input: str = "",
    ) -> Dict[str, Any]:
        with self.profiler.span("omniguard_enforce"):
            self.audits_run += 1
            verdict = self.omni.enforce_output(
                response_text,
                system_prompt=system_text,
                user_input=user_input,
            )
            if not verdict.passed:
                self.blocks_triggered += 1
            return verdict.to_dict()

    def deontic_audit(
        self,
        response_text: str,
        *,
        system_text: str = "",
        user_input: str = "",
    ) -> Dict[str, Any]:
        with self.profiler.span("deontic_audit"):
            self.audits_run += 1
            verdict = deontic.evaluate_deontic(
                response_text,
                system_text=system_text,
                user_input=user_input,
            )
            if not verdict.passed:
                self.blocks_triggered += 1
            return verdict.to_dict()

    def heal_prompt(self, system_text: str) -> Dict[str, Any]:
        with self.profiler.span("heal_prompt"):
            return autohealer.heal_system_prompt(system_text).to_dict()

    def swarm_audit(
        self,
        response_text: str = "",
        *,
        user_input: str = "",
        system_text: str = "",
        code: str = "",
        command: str = "",
    ) -> Dict[str, Any]:
        with self.profiler.span("swarm_audit"):
            self.audits_run += 1
            verdict = self.swarm.execute({
                "response_text": response_text,
                "user_input": user_input,
                "system_text": system_text,
                "code": code,
                "command": command,
            })
            if not verdict.passed:
                self.blocks_triggered += 1
            return verdict.to_dict()

    def full_check(
        self,
        response_text: str,
        *,
        user_input: str = "",
        system_text: str = "",
        session_id: str = "default",
    ) -> Dict[str, Any]:
        with self.profiler.span("full_check"):
            self.audits_run += 1
            rules = ruleextract.extract_rules(system_text) if system_text else None
            outcome = self.orchestrator.enforce(
                orchestrator.OrchestrationRequest(
                    user_input=user_input,
                    response_text=response_text,
                    system_text=system_text,
                    rules=rules,
                    session_id=session_id,
                )
            )
            if outcome.blocked:
                self.blocks_triggered += 1
            return outcome.to_dict()

    def system_probe(self, cwd: str = ".") -> Dict[str, Any]:
        with self.profiler.span("system_probe"):
            return sysprobe.collect_system_health(cwd).to_dict()

    def ast_audit(self, code_or_path: str) -> Dict[str, Any]:
        with self.profiler.span("ast_audit"):
            if os.path.isfile(code_or_path):
                return astguard.scan_file(code_or_path).to_dict()
            return astguard.scan_code(code_or_path).to_dict()

    def command_firewall(self, command: str) -> Dict[str, Any]:
        with self.profiler.span("command_firewall"):
            v = cmdguard.evaluate_command(command)
            if not v.allowed:
                self.blocks_triggered += 1
            return v.to_dict()

    def dlp_scan(self, text: str, *, include_pii: bool = True) -> Dict[str, Any]:
        with self.profiler.span("dlp_scan"):
            return dlp.scan_text(text, include_pii=include_pii).to_dict()

    def tool_firewall(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self.profiler.span("tool_firewall"):
            v = self.tool_guard.check_call(tool_name, args or {})
            if not v.allowed:
                self.blocks_triggered += 1
            return v.to_dict()

    def verify_trace(self, trace: Sequence[Any]) -> Dict[str, Any]:
        with self.profiler.span("verify_trace"):
            return self.verifier.verify(trace).to_dict()

    def status(self) -> Dict[str, Any]:
        from . import __version__, _LAZY_MODULES
        prof_rep = self.profiler.report()
        sys_rep = sysprobe.collect_system_health(".")
        return {
            "version": __version__,
            "modules_count": len(_LAZY_MODULES),
            "audits_run": self.audits_run,
            "blocks_triggered": self.blocks_triggered,
            "circuit_state": self.breaker.state,
            "system_health_score": sys_rep.health_score,
            "system_status": sys_rep.status,
            "canary_id": self.canary_engine.canary.visible_tag,
            "profiler": {
                "total_spans": prof_rep.total_spans,
                "total_ms": round(prof_rep.total_ms, 3),
                "hotspots": prof_rep.hotspots,
            },
        }


def register(agent: Any) -> None:
    try:
        from ..tools import RISK_SAFE, Tool
    except Exception:
        return

    rt = getattr(agent, "enforcex_runtime", None)
    if rt is None:
        rt = SovereignRuntime()
        try:
            agent.enforcex_runtime = rt
        except Exception:
            pass

    tools = getattr(agent, "tools", None)
    if not isinstance(tools, dict):
        return

    def _h_swarm(
        response_text: str = "",
        user_input: str = "",
        system_text: str = "",
        code: str = "",
        command: str = "",
    ) -> str:
        res = rt.swarm_audit(
            response_text=response_text,
            user_input=user_input,
            system_text=system_text,
            code=code,
            command=command,
        )
        return json.dumps(res, indent=2, default=str)

    def _h_check(response_text: str, user_input: str = "", system_text: str = "") -> str:
        res = rt.full_check(response_text, user_input=user_input, system_text=system_text)
        return json.dumps(res, indent=2, default=str)

    def _h_omniguard(response_text: str, system_text: str = "", user_input: str = "") -> str:
        res = rt.omniguard_enforce(response_text, system_text=system_text, user_input=user_input)
        return json.dumps(res, indent=2, default=str)

    def _h_deontic(response_text: str, system_text: str = "", user_input: str = "") -> str:
        res = rt.deontic_audit(response_text, system_text=system_text, user_input=user_input)
        return json.dumps(res, indent=2, default=str)

    def _h_heal(system_text: str) -> str:
        res = rt.heal_prompt(system_text)
        return json.dumps(res, indent=2, default=str)

    def _h_sysprobe(cwd: str = ".") -> str:
        return json.dumps(rt.system_probe(cwd), indent=2, default=str)

    def _h_astaudit(target: str) -> str:
        return json.dumps(rt.ast_audit(target), indent=2, default=str)

    def _h_cmdfirewall(command: str) -> str:
        return json.dumps(rt.command_firewall(command), indent=2, default=str)

    def _h_dlpscan(text: str, include_pii: bool = True) -> str:
        return json.dumps(rt.dlp_scan(text, include_pii=include_pii), indent=2, default=str)

    def _h_toolfirewall(tool_name: str, args_json: str = "{}") -> str:
        try:
            parsed = json.loads(args_json or "{}")
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception:
            parsed = {}
        return json.dumps(rt.tool_firewall(tool_name, parsed), indent=2, default=str)

    def _h_traceverify(trace_json: str) -> str:
        try:
            parsed = json.loads(trace_json or "[]")
            if not isinstance(parsed, list):
                parsed = []
        except Exception:
            parsed = []
        return json.dumps(rt.verify_trace(parsed), indent=2, default=str)

    def _h_status() -> str:
        return json.dumps(rt.status(), indent=2, default=str)

    tools["EnforcexOmniGuard"] = Tool(
        name="EnforcexOmniGuard",
        description="Run the 6-Layer OmniGuard Sovereign Controller (Deontic + SMT + Sheaf + SelfRefine + PCR Certificate + ZK Proof).",
        parameters={
            "type": "object",
            "properties": {
                "response_text": {"type": "string"},
                "system_text": {"type": "string"},
                "user_input": {"type": "string"},
            },
            "required": ["response_text"],
        },
        handler=_h_omniguard,
        risk=RISK_SAFE,
    )
    tools["EnforcexDeonticAudit"] = Tool(
        name="EnforcexDeonticAudit",
        description="Compile system prompt into Standard Deontic Logic (O/F/P) norms, resolve conflicts, and verify response.",
        parameters={
            "type": "object",
            "properties": {
                "response_text": {"type": "string"},
                "system_text": {"type": "string"},
                "user_input": {"type": "string"},
            },
            "required": ["response_text"],
        },
        handler=_h_deontic,
        risk=RISK_SAFE,
    )
    tools["EnforcexPromptHeal"] = Tool(
        name="EnforcexPromptHeal",
        description="Heal weak modals ('try to', 'maybe') and buried mid-prompt rules into strict sovereign directives.",
        parameters={
            "type": "object",
            "properties": {"system_text": {"type": "string"}},
            "required": ["system_text"],
        },
        handler=_h_heal,
        risk=RISK_SAFE,
    )
    tools["EnforcexSwarmAudit"] = Tool(
        name="EnforcexSwarmAudit",
        description="Run the 3-level parallel EnforceX subagent swarm on text, code, or commands.",
        parameters={
            "type": "object",
            "properties": {
                "response_text": {"type": "string"},
                "user_input": {"type": "string"},
                "system_text": {"type": "string"},
                "code": {"type": "string"},
                "command": {"type": "string"},
            },
        },
        handler=_h_swarm,
        risk=RISK_SAFE,
    )
    tools["EnforcexCheck"] = Tool(
        name="EnforcexCheck",
        description="Run the full 10-stage EnforceX Orchestrator pipeline on a response and system prompt.",
        parameters={
            "type": "object",
            "properties": {
                "response_text": {"type": "string"},
                "user_input": {"type": "string"},
                "system_text": {"type": "string"},
            },
            "required": ["response_text"],
        },
        handler=_h_check,
        risk=RISK_SAFE,
    )
    tools["EnforcexSysProbe"] = Tool(
        name="EnforcexSysProbe",
        description="Collect deep OS/process/memory/disk/socket forensics and security hygiene score.",
        parameters={
            "type": "object",
            "properties": {"cwd": {"type": "string"}},
        },
        handler=_h_sysprobe,
        risk=RISK_SAFE,
    )
    tools["EnforcexASTAudit"] = Tool(
        name="EnforcexASTAudit",
        description="Scan Python code or a file path for AST security vulnerabilities, SQLi, eval/exec, and sandbox escapes.",
        parameters={
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
        },
        handler=_h_astaudit,
        risk=RISK_SAFE,
    )
    tools["EnforcexCmdFirewall"] = Tool(
        name="EnforcexCmdFirewall",
        description="Analyze a shell command for destructive actions, RCE pipes, reverse shells, and privilege escalation.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        handler=_h_cmdfirewall,
        risk=RISK_SAFE,
    )
    tools["EnforcexDLPScan"] = Tool(
        name="EnforcexDLPScan",
        description="Scan text for API keys, private keys, DB connection URIs, and PII.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "include_pii": {"type": "boolean"},
            },
            "required": ["text"],
        },
        handler=_h_dlpscan,
        risk=RISK_SAFE,
    )
    tools["EnforcexToolFirewall"] = Tool(
        name="EnforcexToolFirewall",
        description="Pre-validate a tool call against path jail, protected files, SSRF, and command firewall policies.",
        parameters={
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "args_json": {"type": "string"},
            },
            "required": ["tool_name"],
        },
        handler=_h_toolfirewall,
        risk=RISK_SAFE,
    )
    tools["EnforcexTraceVerify"] = Tool(
        name="EnforcexTraceVerify",
        description="Verify an execution trace (JSON array of steps) against Linear Temporal Logic (LTL) safety invariants.",
        parameters={
            "type": "object",
            "properties": {"trace_json": {"type": "string"}},
            "required": ["trace_json"],
        },
        handler=_h_traceverify,
        risk=RISK_SAFE,
    )
    tools["EnforcexStatus"] = Tool(
        name="EnforcexStatus",
        description="Return EnforceX v4 Sovereign status, module count, canary ID, system health score, and profiler hotspots.",
        parameters={"type": "object", "properties": {}},
        handler=_h_status,
        risk=RISK_SAFE,
    )


_HANDLE_ARG_MAX = 262_144
_ARG_CONTROL_RE = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")


def _normalize_handler_arg(arg: Any) -> Tuple[str, str]:
    """Coerce arbitrary caller input into (subcommand, rest).  Never raises."""
    if arg is None:
        text = ""
    elif isinstance(arg, (bytes, bytearray)):
        try:
            text = bytes(arg).decode("utf-8", "replace")
        except Exception:
            text = ""
    elif isinstance(arg, str):
        text = arg
    else:
        try:
            text = str(arg)
        except Exception:
            text = ""
    try:
        text = _ARG_CONTROL_RE.sub(" ", text[:_HANDLE_ARG_MAX])
    except Exception:
        text = ""
    parts = text.strip().split(None, 1)
    sub = parts[0].lower().strip("/.:@") if parts else "status"
    rest = parts[1].strip() if len(parts) > 1 else ""
    return sub, rest


def handle_enforcex(ui: Any, arg: str = "") -> str:
    sub, rest = _normalize_handler_arg(arg)
    try:
        agent = getattr(ui, "agent", None)
        rt = getattr(agent, "enforcex_runtime", None) or SovereignRuntime()
    except Exception:
        rt = None

    if rt is None:
        msg = "EnforceX: runtime unavailable — fail-closed, no audit performed"
    else:
        try:
            if sub == "probe":
                data = rt.system_probe(rest or ".")
                msg = (
                    f"EnforceX SysProbe: status={data.get('status', 'unknown')} "
                    f"score={data.get('health_score', 0)}/100 | "
                    f"CPU={data.get('cpu_count', 0)} | "
                    f"Mem={data.get('memory', {}).get('used_pct', 0)}% | "
                    f"DiskFree={data.get('disk', {}).get('free_gb', 0)}GB | "
                    f"FDs={data.get('process', {}).get('open_fds', 0)} | "
                    f"Findings={len(data.get('findings', []) or [])}"
                )
            elif sub == "swarm":
                data = rt.swarm_audit(rest or "sample check")
                msg = (
                    f"EnforceX Swarm: action={data.get('action', 'block')} "
                    f"score={data.get('overall_score', 0)}/100 "
                    f"waves={len(data.get('waves', []) or [])} "
                    f"findings={data.get('findings_count', 'n/a')} ({data.get('elapsed_ms', 0)}ms)"
                )
            elif sub == "omni":
                data = rt.omniguard_enforce(rest or "sample check")
                msg = (
                    f"EnforceX OmniGuard: action={data.get('action', 'block')} "
                    f"passed={data.get('passed', False)} "
                    f"score={data.get('score', 0)}/100 repaired={data.get('repaired', False)} "
                    f"violations={data.get('violations_count', 'n/a')}"
                )
            elif sub == "heal":
                data = rt.heal_prompt(rest or "Try to use JSON if possible.")
                msg = (
                    f"EnforceX AutoHealer: patches={data.get('patches_count', 0)} "
                    f"deontic_gain=+{data.get('deontic_gain', 0)}"
                )
            elif sub == "cmd":
                data = rt.command_firewall(rest)
                msg = (
                    f"EnforceX CmdFirewall: action={data.get('action', 'block')} "
                    f"allowed={data.get('allowed', False)} risk={data.get('risk_score', 'n/a')}"
                )
            elif sub == "dlp":
                data = rt.dlp_scan(rest)
                msg = (
                    f"EnforceX DLP: clean={data.get('clean', False)} "
                    f"matches={data.get('matches_count', 0)} "
                    f"redacted={str(data.get('redacted_text', ''))[:120]}"
                )
            elif sub == "ast":
                data = rt.ast_audit(rest)
                msg = (
                    f"EnforceX ASTGuard: passed={data.get('passed', False)} "
                    f"grade={data.get('grade', '?')} score={data.get('score', 0)} "
                    f"findings={data.get('findings_count', 'n/a')}"
                )
            else:
                st = rt.status()
                msg = (
                    f"EnforceX v{st.get('version', '?')} Sovereign Suite "
                    f"({st.get('modules_count', 0)} modules) | "
                    f"Health={st.get('system_health_score', 0)}/100 ({st.get('system_status', 'unknown')}) | "
                    f"Circuit={st.get('circuit_state', 'OPEN')} Canary={st.get('canary_id', '')} | "
                    f"Audits={st.get('audits_run', 0)} Blocks={st.get('blocks_triggered', 0)}"
                )
        except Exception as exc:
            msg = (
                f"EnforceX '{sub}' failed closed: {type(exc).__name__}: {str(exc)[:120]} "
                "(verdict: block; retry with a valid argument)"
            )

    if not isinstance(msg, str):
        msg = "EnforceX: fail-closed (unrenderable verdict)"
    try:
        printer = getattr(ui, "print_info", None)
        if callable(printer):
            printer(msg)
    except Exception:
        pass
    return msg


def _self_test() -> int:
    from types import SimpleNamespace

    rt = SovereignRuntime()
    st = rt.status()
    assert st["modules_count"] >= 88
    assert st["circuit_state"] == "CLOSED"

    sw = rt.swarm_audit("1. Valid item one.\n2. Valid item two.", code="def f(x):\n    return x + 1\n")
    assert sw["passed"] is True

    og = rt.omniguard_enforce(
        "Sure, I can do that!\n---\n- Item one.\n- Item two.",
        system_text="Always use a bulleted list and never use horizontal rules (---).",
    )
    assert og["passed"] is True and og["repaired"] is True

    cf = rt.command_firewall("rm -rf /")
    assert cf["allowed"] is False and cf["action"] == "block"

    fake_agent = SimpleNamespace(tools={})
    register(fake_agent)
    for tname in (
        "EnforcexOmniGuard", "EnforcexDeonticAudit", "EnforcexPromptHeal",
        "EnforcexSwarmAudit", "EnforcexCheck", "EnforcexSysProbe",
        "EnforcexASTAudit", "EnforcexCmdFirewall", "EnforcexDLPScan",
        "EnforcexToolFirewall", "EnforcexTraceVerify", "EnforcexStatus",
    ):
        assert tname in fake_agent.tools, tname

    fake_agent.tools["EnforcexSwarmAudit"].handler(response_text="1. one\n2. two")
    out = fake_agent.tools["EnforcexStatus"].handler()
    parsed = json.loads(out)
    assert parsed["audits_run"] >= 1

    # adversarial: handle_enforcex normalizes hostile input and fails closed
    fake_agent.enforcex_runtime = rt
    seen_msgs: list = []
    ui = SimpleNamespace(agent=fake_agent, print_info=seen_msgs.append)
    for hostile in (
        None, "", "   ", 12345, b"\xff\xfe status", "cmd", "dlp", "ast",
        "/PROBE /nonexistent/__nope__", "\x00cmd \x00rm -rf /",
        "status " + "x" * 50_000, "swarm " + "A" * 30_000, "x" * 2_000_000,
    ):
        out = handle_enforcex(ui, hostile)  # type: ignore[arg-type]
        assert isinstance(out, str) and out, repr(hostile)
    assert seen_msgs and all(isinstance(m, str) for m in seen_msgs)

    assert _normalize_handler_arg("  /CMD\r\n rm -rf /  ") == ("cmd", "rm -rf /")
    assert _normalize_handler_arg(None) == ("status", "")
    assert _normalize_handler_arg(b"\x00heal") == ("heal", "")

    out = handle_enforcex(ui, "cmd rm -rf /")
    assert "CmdFirewall" in out and "allowed=False" in out, out

    class _Boom:
        def system_probe(self, cwd):
            raise RuntimeError("boom")

    boom_ui = SimpleNamespace(agent=SimpleNamespace(enforcex_runtime=_Boom()))
    out = handle_enforcex(boom_ui, "probe /")
    assert "failed closed" in out and "RuntimeError" in out, out
    bad_ui = SimpleNamespace(agent=None, print_info=lambda m: (_ for _ in ()).throw(ValueError("no console")))
    assert isinstance(handle_enforcex(bad_ui, "status"), str)

    print("sovereign._self_test: OK PASS (runtime facade + 12 FullAgent tools registered)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
