r"""sheafcheck — Cellular Sheaf Cohomology & Global-Local Section Consistency Verifier.

Models a multi-section document or response as a 1-dimensional cellular sheaf
over a graph $G = (V, E)$ where:
  * Each vertex $v \in V$ is a section/paragraph carrying a stalk of local
    attribute assertions (`key -> polarity/value`, e.g. `port=8080`, `auth=enabled`,
    `status=passed`).
  * Each edge $e = (u, v) \in E$ carries restriction maps $F_{u \to e}, F_{v \to e}$
    onto the shared vocabulary between sections $u$ and $v$.
  * The Sheaf Coboundary operator $\delta s$ measures the disagreement across
    every edge; a global section exists (zero 0-cohomology obstruction) iff
    $\|\delta s\|^2 = 0$.

Detects subtle cross-section contradictions (e.g. Section 1 says "authentication
is disabled" while Section 3 says "authentication is required", or conflicting
numbers for the same metric across paragraphs).

Self-test: ``python3 -m fullagent.enforcex.sheafcheck``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

__all__ = [
    "SectionStalk",
    "CoboundaryDefect",
    "SheafCohomologyReport",
    "SheafConsistencyAnalyzer",
    "extract_stalks",
    "verify_sheaf_consistency",
]

_NEG_WORDS = {"no", "not", "never", "disabled", "off", "false", "none", "without", "forbidden"}
_POS_WORDS = {"yes", "enabled", "on", "true", "required", "mandatory", "active", "always"}

# Patterns for extracting subject-value or subject-polarity assertions
_KV_RE = re.compile(r"\b([a-z][a-z0-9_]{2,24})\s*(?:=|:|\bis\s+set\s+to\b)\s*([a-z0-9_.-]+)\b", re.I)
_PROP_RE = re.compile(
    r"\b([a-z][a-z0-9_ -]{2,28}?)\s+(?:is|are|must\s+be|remains?)\s+(enabled|disabled|required|forbidden|optional|on|off|true|false|active|inactive)\b",
    re.I,
)


@dataclass
class SectionStalk:
    section_index: int
    heading: str
    assertions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_index": self.section_index,
            "heading": self.heading,
            "assertions": dict(self.assertions),
        }


@dataclass
class CoboundaryDefect:
    section_u: int
    section_v: int
    attribute: str
    value_u: str
    value_v: str
    residual: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_u": self.section_u,
            "section_v": self.section_v,
            "attribute": self.attribute,
            "value_u": self.value_u,
            "value_v": self.value_v,
            "residual": self.residual,
        }


@dataclass
class SheafCohomologyReport:
    consistent: bool = True
    laplacian_energy: float = 0.0
    stalks: List[SectionStalk] = field(default_factory=list)
    defects: List[CoboundaryDefect] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consistent": self.consistent,
            "laplacian_energy": round(self.laplacian_energy, 4),
            "stalks_count": len(self.stalks),
            "defects_count": len(self.defects),
            "defects": [d.to_dict() for d in self.defects],
        }


def _canonical_val(v: str) -> str:
    low = v.strip().lower()
    if low in ("enabled", "on", "true", "active", "required", "yes"):
        return "POSITIVE"
    if low in ("disabled", "off", "false", "inactive", "forbidden", "no"):
        return "NEGATIVE"
    return low


def extract_stalks(text: str) -> List[SectionStalk]:
    """Split `text` into paragraph/section stalks and extract local assertions."""
    if not text or not isinstance(text, str):
        return []
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    stalks: List[SectionStalk] = []
    for idx, blk in enumerate(blocks):
        lines = blk.splitlines()
        heading = lines[0].lstrip("#").strip()[:40] if lines and lines[0].startswith("#") else f"section_{idx}"
        assertions: Dict[str, str] = {}
        for m in _KV_RE.finditer(blk):
            key = m.group(1).strip().lower()
            val = _canonical_val(m.group(2))
            assertions[key] = val
        for m in _PROP_RE.finditer(blk):
            subj = re.sub(r"\s+", "_", m.group(1).strip().lower())
            # drop leading articles
            subj = re.sub(r"^(?:the|a|an|this|our)\_", "", subj)
            val = _canonical_val(m.group(2))
            assertions[subj] = val
        stalks.append(SectionStalk(section_index=idx, heading=heading, assertions=assertions))
    return stalks


def verify_sheaf_consistency(text: str, *, system_text: str = "") -> SheafCohomologyReport:
    """Compute sheaf coboundary defects across all sections of `system_text` + `text`."""
    combined = f"{system_text.strip()}\n\n{text.strip()}" if system_text else (text or "")
    stalks = extract_stalks(combined)
    defects: List[CoboundaryDefect] = []

    for i in range(len(stalks)):
        for j in range(i + 1, len(stalks)):
            u, v = stalks[i], stalks[j]
            shared_keys = set(u.assertions.keys()) & set(v.assertions.keys())
            for k in sorted(shared_keys):
                vu, vv = u.assertions[k], v.assertions[k]
                if vu != vv:
                    defects.append(
                        CoboundaryDefect(
                            section_u=u.section_index,
                            section_v=v.section_index,
                            attribute=k,
                            value_u=vu,
                            value_v=vv,
                            residual=1.0,
                        )
                    )

    energy = float(sum(d.residual ** 2 for d in defects))
    return SheafCohomologyReport(
        consistent=(len(defects) == 0),
        laplacian_energy=energy,
        stalks=stalks,
        defects=defects,
    )


class SheafConsistencyAnalyzer:
    def evaluate(self, response_text: str, *, system_text: str = "") -> SheafCohomologyReport:
        return verify_sheaf_consistency(response_text, system_text=system_text)


def _self_test() -> int:
    # 1. Consistent multi-section document has 0 coboundary energy
    good_doc = (
        "# Overview\nThe tls_mode is enabled and port = 8443.\n\n"
        "# Deployment\nVerify that tls_mode is active and port = 8443."
    )
    r_ok = verify_sheaf_consistency(good_doc)
    assert r_ok.consistent and r_ok.laplacian_energy == 0.0, r_ok.to_dict()

    # 2. Contradictory sections produce non-zero coboundary defect
    bad_doc = (
        "# Section 1\nThe authentication is enabled and port = 8080.\n\n"
        "# Section 2\nIn production, authentication is disabled and port = 9090."
    )
    r_bad = verify_sheaf_consistency(bad_doc)
    assert not r_bad.consistent and len(r_bad.defects) == 2, r_bad.to_dict()

    print("sheafcheck._self_test: OK PASS (cellular sheaf stalks, coboundary defects, Laplacian energy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
