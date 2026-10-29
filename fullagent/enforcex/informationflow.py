"""informationflow — Dual Bell-LaPadula (Confidentiality) & Biba (Integrity) Lattice Engine.

Assigns every information source and sink a dual security label `(C, I)`:
  * Confidentiality `C` (0=PUBLIC_OUTPUT < 1=TOOL_EXTERNAL < 2=USER_SESSION < 3=SYSTEM_SECRET):
    Bell-LaPadula enforces *No Read Up* and *No Write Down* (`C_source <= C_sink`),
    preventing high-confidentiality system secrets from flowing to public/external sinks.
  * Integrity `I` (0=UNTRUSTED_TOOL < 1=USER_INPUT < 2=SESSION_STATE < 3=SOVEREIGN_SYSTEM):
    Biba enforces *No Read Down* and *No Write Up* (`I_source >= I_sink`),
    preventing low-integrity tool/user input from corrupting high-integrity system directives.

Self-test: ``python3 -m fullagent.enforcex.informationflow``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

__all__ = [
    "SecurityLabel",
    "FlowViolation",
    "IFCVerificationReport",
    "InformationFlowLattice",
    "PREDEFINED_LABELS",
    "check_flow",
]


@dataclass(frozen=True)
class SecurityLabel:
    name: str
    confidentiality: int      # 0 (lowest/public) .. 3 (highest/secret)
    integrity: int            # 0 (lowest/untrusted) .. 3 (highest/sovereign)

    def join(self, other: "SecurityLabel") -> "SecurityLabel":
        """Least upper bound in the information-flow lattice:
        confidentiality increases (more secret), integrity decreases (more tainted)."""
        return SecurityLabel(
            name=f"({self.name}_join_{other.name})",
            confidentiality=max(self.confidentiality, other.confidentiality),
            integrity=min(self.integrity, other.integrity),
        )

    def meet(self, other: "SecurityLabel") -> "SecurityLabel":
        return SecurityLabel(
            name=f"({self.name}_meet_{other.name})",
            confidentiality=min(self.confidentiality, other.confidentiality),
            integrity=max(self.integrity, other.integrity),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "confidentiality": self.confidentiality,
            "integrity": self.integrity,
        }


PREDEFINED_LABELS: Dict[str, SecurityLabel] = {
    "SOVEREIGN_SYSTEM": SecurityLabel("SOVEREIGN_SYSTEM", confidentiality=3, integrity=3),
    "DEVELOPER_CONFIG": SecurityLabel("DEVELOPER_CONFIG", confidentiality=2, integrity=3),
    "SESSION_MEMORY":   SecurityLabel("SESSION_MEMORY",   confidentiality=2, integrity=2),
    "USER_INPUT":       SecurityLabel("USER_INPUT",       confidentiality=1, integrity=1),
    "UNTRUSTED_TOOL":   SecurityLabel("UNTRUSTED_TOOL",   confidentiality=1, integrity=0),
    "PUBLIC_SINK":      SecurityLabel("PUBLIC_SINK",      confidentiality=1, integrity=0),
}


@dataclass
class FlowViolation:
    law: str                  # BELL_LAPADULA_CONFIDENTIALITY | BIBA_INTEGRITY
    source: str
    sink: str
    message: str
    severity: str = "critical"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "law": self.law,
            "source": self.source,
            "sink": self.sink,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass
class IFCVerificationReport:
    allowed: bool = True
    violations: List[FlowViolation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "violations_count": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
        }


def check_flow(source: SecurityLabel, sink: SecurityLabel, *, declassified: bool = False) -> IFCVerificationReport:
    """Verify whether information may legally flow from `source` to `sink`."""
    violations: List[FlowViolation] = []

    # 1. Bell-LaPadula Star Property (No Write Down): source.confidentiality <= sink.confidentiality
    if source.confidentiality > sink.confidentiality and not declassified:
        violations.append(
            FlowViolation(
                law="BELL_LAPADULA_CONFIDENTIALITY",
                source=source.name,
                sink=sink.name,
                message=(
                    f"Illegal confidentiality flow (write-down): '{source.name}' "
                    f"(C={source.confidentiality}) cannot flow to '{sink.name}' (C={sink.confidentiality})"
                ),
            )
        )

    # 2. Biba Integrity Star Property (No Write Up): source.integrity >= sink.integrity
    if source.integrity < sink.integrity:
        violations.append(
            FlowViolation(
                law="BIBA_INTEGRITY",
                source=source.name,
                sink=sink.name,
                message=(
                    f"Illegal integrity flow (taint elevation): low-integrity '{source.name}' "
                    f"(I={source.integrity}) cannot overwrite high-integrity '{sink.name}' (I={sink.integrity})"
                ),
            )
        )

    return IFCVerificationReport(allowed=(len(violations) == 0), violations=violations)


class InformationFlowLattice:
    def resolve_label(self, name_or_label: Any) -> SecurityLabel:
        if isinstance(name_or_label, SecurityLabel):
            return name_or_label
        return PREDEFINED_LABELS.get(str(name_or_label).upper(), PREDEFINED_LABELS["USER_INPUT"])

    def verify_flow(self, source: Any, sink: Any, *, declassified: bool = False) -> IFCVerificationReport:
        return check_flow(self.resolve_label(source), self.resolve_label(sink), declassified=declassified)


def _self_test() -> int:
    lat = InformationFlowLattice()
    # 1. Untrusted tool trying to overwrite Sovereign System prompt -> Biba integrity violation
    r1 = lat.verify_flow("UNTRUSTED_TOOL", "SOVEREIGN_SYSTEM")
    assert not r1.allowed and any(v.law == "BIBA_INTEGRITY" for v in r1.violations)

    # 2. Sovereign system secret flowing to public sink -> Bell-LaPadula violation
    r2 = lat.verify_flow("SOVEREIGN_SYSTEM", "PUBLIC_SINK")
    assert not r2.allowed and any(v.law == "BELL_LAPADULA_CONFIDENTIALITY" for v in r2.violations)

    # 3. User input flowing to untrusted tool -> allowed (C=1<=1, I=1>=0)
    r3 = lat.verify_flow("USER_INPUT", "UNTRUSTED_TOOL")
    assert r3.allowed, r3.to_dict()

    print("informationflow._self_test: OK PASS (Bell-LaPadula confidentiality + Biba integrity lattice)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
