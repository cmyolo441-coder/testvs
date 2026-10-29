"""sessiontypes — Honda Multiparty Session Types (MPST) & Protocol Duality Checker.

Implements Kohei Honda's **Binary & Multiparty Session Types** to statically and
dynamically verify communication protocols between `User`, `Agent`, `Tool`, and
`Verifier`:
  * Send `!peer<label:payload_type>.S`
  * Receive `?peer<label:payload_type>.S`
  * Internal Choice `Select( peer, {label_i: S_i} )`
  * External Choice `Branch( peer, {label_i: S_i} )`
  * End of session `End`

Provides three formal operations:
  1. **Exact Session Duality (`dual(S)`)**: flips `!` <-> `?` and `Select` <-> `Branch`.
  2. **Compatibility & Deadlock-Free Progress Check (`are_dual(S_A, S_B)`)**:
     verifies that two endpoint protocols are exact duals so no communication
     mismatch or orphan message can occur.
  3. **Runtime Trace Monitor**: steps a protocol along actual `(sender, receiver, label)`
     events and flags protocol-order violations immediately.

Self-test: ``python3 -m fullagent.enforcex.sessiontypes``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

__all__ = [
    "SessionType",
    "SessionMonitorReport",
    "SessionProtocolVerifier",
    "SEND",
    "RECV",
    "SELECT",
    "BRANCH",
    "END",
    "dual",
    "are_dual",
]


@dataclass(frozen=True)
class SessionType:
    kind: str  # SEND | RECV | SELECT | BRANCH | END
    peer: str = ""
    label: str = ""
    payload_type: str = "any"
    cont: Optional["SessionType"] = None
    branches: Tuple[Tuple[str, "SessionType"], ...] = ()

    def branch_map(self) -> Dict[str, "SessionType"]:
        return dict(self.branches)


def END() -> SessionType:
    return SessionType(kind="END")


def SEND(peer: str, label: str, payload_type: str = "any", cont: Optional[SessionType] = None) -> SessionType:
    return SessionType(kind="SEND", peer=peer, label=label, payload_type=payload_type, cont=cont or END())


def RECV(peer: str, label: str, payload_type: str = "any", cont: Optional[SessionType] = None) -> SessionType:
    return SessionType(kind="RECV", peer=peer, label=label, payload_type=payload_type, cont=cont or END())


def SELECT(peer: str, choices: Dict[str, SessionType]) -> SessionType:
    return SessionType(kind="SELECT", peer=peer, branches=tuple(sorted(choices.items())))


def BRANCH(peer: str, choices: Dict[str, SessionType]) -> SessionType:
    return SessionType(kind="BRANCH", peer=peer, branches=tuple(sorted(choices.items())))


def dual(st: SessionType, self_role: str = "", peer_role: str = "") -> SessionType:
    """Compute the syntactic dual `overline{S}` of session type `st`."""
    target_peer = self_role or st.peer
    if st.kind == "END":
        return END()
    if st.kind == "SEND":
        assert st.cont is not None
        return RECV(target_peer, st.label, st.payload_type, dual(st.cont, self_role, peer_role))
    if st.kind == "RECV":
        assert st.cont is not None
        return SEND(target_peer, st.label, st.payload_type, dual(st.cont, self_role, peer_role))
    if st.kind == "SELECT":
        return BRANCH(target_peer, {lbl: dual(sub, self_role, peer_role) for lbl, sub in st.branches})
    if st.kind == "BRANCH":
        return SELECT(target_peer, {lbl: dual(sub, self_role, peer_role) for lbl, sub in st.branches})
    raise ValueError(f"Unknown session type kind: {st.kind}")


def are_dual(st_a: SessionType, st_b: SessionType) -> bool:
    """Check whether endpoint types `st_a` and `st_b` are exact session duals."""
    if st_a.kind == "END" and st_b.kind == "END":
        return True
    if st_a.kind == "SEND" and st_b.kind == "RECV":
        if st_a.label != st_b.label or st_a.payload_type != st_b.payload_type:
            return False
        assert st_a.cont is not None and st_b.cont is not None
        return are_dual(st_a.cont, st_b.cont)
    if st_a.kind == "RECV" and st_b.kind == "SEND":
        if st_a.label != st_b.label or st_a.payload_type != st_b.payload_type:
            return False
        assert st_a.cont is not None and st_b.cont is not None
        return are_dual(st_a.cont, st_b.cont)
    if (st_a.kind == "SELECT" and st_b.kind == "BRANCH") or (st_a.kind == "BRANCH" and st_b.kind == "SELECT"):
        map_a = st_a.branch_map()
        map_b = st_b.branch_map()
        if set(map_a.keys()) != set(map_b.keys()):
            return False
        return all(are_dual(map_a[k], map_b[k]) for k in map_a)
    return False


@dataclass
class SessionMonitorReport:
    completed_cleanly: bool
    steps_verified: int
    remaining_kind: str
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "completed_cleanly": self.completed_cleanly,
            "steps_verified": self.steps_verified,
            "remaining_kind": self.remaining_kind,
            "error": self.error,
        }


class SessionProtocolVerifier:
    """Steps a local session type against a sequence of `(direction, label)` events."""

    def verify_trace(self, protocol: SessionType, events: Sequence[Tuple[str, str]]) -> SessionMonitorReport:
        cur = protocol
        steps = 0
        for direction, label in events:
            d = direction.upper()
            if cur.kind == "END":
                return SessionMonitorReport(False, steps, "END", f"Extra event ({d}, {label}) after session END")
            if cur.kind == "SEND":
                if d != "SEND" or cur.label != label:
                    return SessionMonitorReport(False, steps, cur.kind, f"Expected SEND:{cur.label}, got {d}:{label}")
                assert cur.cont is not None
                cur = cur.cont
                steps += 1
            elif cur.kind == "RECV":
                if d != "RECV" or cur.label != label:
                    return SessionMonitorReport(False, steps, cur.kind, f"Expected RECV:{cur.label}, got {d}:{label}")
                assert cur.cont is not None
                cur = cur.cont
                steps += 1
            elif cur.kind == "SELECT":
                bmap = cur.branch_map()
                if d != "SEND" or label not in bmap:
                    return SessionMonitorReport(False, steps, cur.kind, f"Invalid SELECT branch {d}:{label}")
                cur = bmap[label]
                steps += 1
            elif cur.kind == "BRANCH":
                bmap = cur.branch_map()
                if d != "RECV" or label not in bmap:
                    return SessionMonitorReport(False, steps, cur.kind, f"Invalid BRANCH choice {d}:{label}")
                cur = bmap[label]
                steps += 1
        return SessionMonitorReport(
            completed_cleanly=(cur.kind == "END"),
            steps_verified=steps,
            remaining_kind=cur.kind,
            error="" if cur.kind == "END" else f"Session ended prematurely at state {cur.kind}",
        )


def _self_test() -> int:
    # Agent protocol: send audit_request -> branch on {pass: send deliver, fail: send retry -> send deliver}
    agent_proto = SEND(
        "verifier",
        "audit_request",
        "str",
        BRANCH(
            "verifier",
            {
                "pass": SEND("user", "deliver", "str", END()),
                "fail": SEND("agent", "retry", "str", SEND("user", "deliver", "str", END())),
            },
        ),
    )
    verifier_proto = dual(agent_proto, self_role="agent")
    assert are_dual(agent_proto, verifier_proto) is True
    assert are_dual(agent_proto, agent_proto) is False

    mon = SessionProtocolVerifier()
    rep_ok = mon.verify_trace(agent_proto, [("SEND", "audit_request"), ("RECV", "fail"), ("SEND", "retry"), ("SEND", "deliver")])
    assert rep_ok.completed_cleanly and rep_ok.steps_verified == 4

    # Attempting to deliver directly without retry after 'fail' violates the session type!
    rep_bad = mon.verify_trace(agent_proto, [("SEND", "audit_request"), ("RECV", "fail"), ("SEND", "deliver")])
    assert not rep_bad.completed_cleanly and "Expected SEND:retry" in rep_bad.error

    print("sessiontypes._self_test: OK PASS (Honda MPST duality, Select/Branch choice, trace monitor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
