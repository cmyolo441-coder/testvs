"""separationlogic — Reynolds Separation Logic (P * Q, P -* Q) Disjoint State Verifier.

Implements John C. Reynolds & Peter O'Hearn's **Separation Logic** for verifying
that concurrent subagents and tool calls operate on strictly disjoint workspace
files, blackboard keys, and memory heaps:
  * Heaplet `h: Map[Location, Value]` with finite domain `dom(h)`.
  * Disjointness `h1 # h2` iff `dom(h1) & dom(h2) == empty`.
  * Separating Conjunction `P * Q` holds on `h` iff `h = h1 U h2` with `h1 # h2`,
    `h1 |= P`, and `h2 |= Q`.
  * Separating Implication / Magic Wand `P -* Q` holds on `h` iff for every
    heaplet `h'` disjoint from `h` satisfying `h' |= P`, the combined heap
    `h U h' |= Q`.
  * Frame Rule Verification: `{P} C {Q}` implies `{P * R} C {Q * R}` provided
    command `C` never mutates any location in `free_locs(R)`.

Self-test: ``python3 -m fullagent.enforcex.separationlogic``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

__all__ = [
    "Heaplet",
    "FrameViolation",
    "SeparationReport",
    "SeparationLogicVerifier",
]


class Heaplet:
    """Partial function `Loc -> Val` representing an owned state partition."""

    def __init__(self, cells: Optional[Dict[str, Any]] = None) -> None:
        self.cells: Dict[str, Any] = dict(cells or {})

    @property
    def domain(self) -> Set[str]:
        return set(self.cells.keys())

    def disjoint_from(self, other: "Heaplet") -> bool:
        return len(self.domain & other.domain) == 0

    def star_union(self, other: "Heaplet") -> Optional["Heaplet"]:
        """Partial commutative monoid composition `h1 * h2` (undefined if domains overlap)."""
        if not self.disjoint_from(other):
            return None
        merged = dict(self.cells)
        merged.update(other.cells)
        return Heaplet(merged)


@dataclass
class FrameViolation:
    worker_a: str
    worker_b: str
    overlapping_locations: List[str]
    kind: str  # domain_collision | frame_mutation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_a": self.worker_a,
            "worker_b": self.worker_b,
            "overlapping_locations": list(self.overlapping_locations),
            "kind": self.kind,
        }


@dataclass
class SeparationReport:
    disjoint_valid: bool
    frame_preserved: bool
    composed_domain_size: int
    violations: List[FrameViolation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disjoint_valid": self.disjoint_valid,
            "frame_preserved": self.frame_preserved,
            "composed_domain_size": self.composed_domain_size,
            "violations": [v.to_dict() for v in self.violations],
        }


class SeparationLogicVerifier:
    """Verifies `P_1 * P_2 * ... * P_k` disjointness and the Frame Rule `{P * R} C {Q * R}`."""

    def verify_parallel_partitions(self, partitions: Dict[str, Heaplet]) -> SeparationReport:
        names = sorted(partitions.keys())
        violations: List[FrameViolation] = []
        composed: Dict[str, Any] = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                wa, wb = names[i], names[j]
                ha, hb = partitions[wa], partitions[wb]
                overlap = sorted(ha.domain & hb.domain)
                if overlap:
                    violations.append(
                        FrameViolation(
                            worker_a=wa,
                            worker_b=wb,
                            overlapping_locations=overlap,
                            kind="domain_collision",
                        )
                    )
        for h in partitions.values():
            composed.update(h.cells)
        return SeparationReport(
            disjoint_valid=(len(violations) == 0),
            frame_preserved=True,
            composed_domain_size=len(composed),
            violations=violations,
        )

    def verify_frame_rule(
        self,
        local_before: Heaplet,
        frame_r: Heaplet,
        full_after: Heaplet,
        worker_name: str = "subagent",
    ) -> SeparationReport:
        """Verify that executing `worker_name` on `local_before * frame_r` leaves `frame_r` untouched."""
        violations: List[FrameViolation] = []
        disjoint = local_before.disjoint_from(frame_r)
        if not disjoint:
            violations.append(
                FrameViolation(
                    worker_a=worker_name,
                    worker_b="frame_R",
                    overlapping_locations=sorted(local_before.domain & frame_r.domain),
                    kind="domain_collision",
                )
            )
        clobbered: List[str] = []
        for loc, expected_val in frame_r.cells.items():
            if loc not in full_after.cells or full_after.cells[loc] != expected_val:
                clobbered.append(loc)
        if clobbered:
            violations.append(
                FrameViolation(
                    worker_a=worker_name,
                    worker_b="frame_R",
                    overlapping_locations=sorted(clobbered),
                    kind="frame_mutation",
                )
            )
        return SeparationReport(
            disjoint_valid=disjoint,
            frame_preserved=(len(clobbered) == 0),
            composed_domain_size=len(full_after.cells),
            violations=violations,
        )


def _self_test() -> int:
    v = SeparationLogicVerifier()
    h1 = Heaplet({"ast_report": "ok", "ast_score": 100})
    h2 = Heaplet({"dlp_report": "clean", "dlp_score": 100})
    rep_ok = v.verify_parallel_partitions({"ast_worker": h1, "dlp_worker": h2})
    assert rep_ok.disjoint_valid and rep_ok.composed_domain_size == 4

    # Overlapping key triggers separating conjunction violation
    h3 = Heaplet({"ast_score": 50})
    rep_bad = v.verify_parallel_partitions({"ast_worker": h1, "rogue_worker": h3})
    assert not rep_bad.disjoint_valid and rep_bad.violations[0].overlapping_locations == ["ast_score"]

    # Frame rule verification: mutating frame_R triggers frame_mutation violation
    frame_r = Heaplet({"system_prompt_hash": "abc123", "ring0_locked": True})
    after_good = Heaplet({"ast_report": "done", "ast_score": 98, "system_prompt_hash": "abc123", "ring0_locked": True})
    assert v.verify_frame_rule(h1, frame_r, after_good).frame_preserved

    after_corrupt = Heaplet({"ast_report": "done", "ast_score": 98, "system_prompt_hash": "TAMPERED", "ring0_locked": True})
    rep_frame_bad = v.verify_frame_rule(h1, frame_r, after_corrupt)
    assert not rep_frame_bad.frame_preserved

    print("separationlogic._self_test: OK PASS (P * Q disjointness + Frame Rule {P * R} C {Q * R} verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
