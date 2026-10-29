"""byzantine — $3f+1$ Byzantine Fault Tolerant (PBFT) Multi-Verifier Quorum Engine.

Adjudicates compliance verdicts across $N = 3f + 1$ verifier subagents where up
to $f$ verifiers may be Byzantine (faulty, compromised, equivocating, or
emitting extreme outlier scores):
  * Equivocation Detection & Slashing: if a verifier submits two conflicting
    votes for the same turn/sequence, both votes are discarded and the verifier
    is slashed into `slashed_verifiers`.
  * Byzantine Trimmed Median / Mean: sorts honest scores, trims the top $f$ and
    bottom $f$ extremes, and requires a $2f + 1$ supermajority quorum for finality.

Self-test: ``python3 -m fullagent.enforcex.byzantine``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "VerifierVote",
    "BFTConsensusCertificate",
    "ByzantineQuorumEngine",
    "adjudicate_byzantine_quorum",
]


@dataclass(frozen=True)
class VerifierVote:
    verifier_id: str
    seq: int
    passed: bool
    score: float


@dataclass
class BFTConsensusCertificate:
    seq: int
    quorum_reached: bool
    passed: bool
    trimmed_score: float
    total_verifiers: int
    fault_tolerance_f: int
    quorum_size: int
    slashed_verifiers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "quorum_reached": self.quorum_reached,
            "passed": self.passed,
            "trimmed_score": round(self.trimmed_score, 2),
            "total_verifiers": self.total_verifiers,
            "fault_tolerance_f": self.fault_tolerance_f,
            "quorum_size": self.quorum_size,
            "slashed_verifiers": list(self.slashed_verifiers),
        }


def _as_int(value: Any, default: int) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _as_score(value: Any) -> float:
    """Finite score clamped to [0, 100]; NaN/inf/garbage collapse to 0."""
    try:
        s = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(s):
        return 0.0
    return max(0.0, min(100.0, s))


def _coerce_vote(raw: Any, seq: int) -> Optional[VerifierVote]:
    """Coerce a VerifierVote / dict / tuple into a normalized vote.

    Votes carrying a different sequence number are ignored (return None);
    structurally invalid entries are ignored as well.  Never raises.
    """
    try:
        if isinstance(raw, VerifierVote):
            v_seq = _as_int(raw.seq, seq)
            if v_seq != seq:
                return None
            return VerifierVote(
                str(raw.verifier_id), v_seq, bool(raw.passed), _as_score(raw.score)
            )
        if isinstance(raw, dict):
            v_seq = _as_int(raw.get("seq", seq), seq)
            if v_seq != seq:
                return None
            vid = raw.get("verifier_id", raw.get("id", ""))
            if vid is None or vid == "":
                return None
            return VerifierVote(str(vid), v_seq, bool(raw.get("passed", False)),
                                _as_score(raw.get("score", 0.0)))
        if isinstance(raw, (tuple, list)) and len(raw) >= 4:
            v_seq = _as_int(raw[1], seq)
            if v_seq != seq:
                return None
            if raw[0] is None or raw[0] == "":
                return None
            return VerifierVote(str(raw[0]), v_seq, bool(raw[2]), _as_score(raw[3]))
    except Exception:  # defensive: malformed entries are dropped, not fatal
        return None
    return None


def adjudicate_byzantine_quorum(votes: Sequence[VerifierVote], seq: int = 1) -> BFTConsensusCertificate:
    seq_i = _as_int(seq, 1)
    by_id: Dict[str, List[VerifierVote]] = {}
    if votes is not None and not isinstance(votes, (str, bytes, dict)):
        try:
            raw_votes = list(votes)
        except TypeError:
            raw_votes = []
        for raw in raw_votes:
            v = _coerce_vote(raw, seq_i)
            if v is not None:
                by_id.setdefault(v.verifier_id, []).append(v)

    slashed: List[str] = []
    honest: List[VerifierVote] = []
    for vid, vlist in sorted(by_id.items()):
        # Equivocation check: multiple distinct votes for the same seq
        distinct = {(x.passed, round(x.score, 2)) for x in vlist}
        if len(distinct) > 1:
            slashed.append(vid)
        else:
            honest.append(vlist[0])

    n = len(honest)
    total = len(by_id)
    # PBFT fault bound is derived from the total participant set (slashed
    # equivocators are exactly the faulty nodes), not from survivors only —
    # otherwise slashing a liar could *lower* the quorum and let a single
    # honest vote finalize a decision.
    f = max(0, (total - 1) // 3) if total >= 4 else 0
    quorum_req = (2 * f + 1) if total >= 4 else (max(1, (total // 2) + 1) if total > 0 else 1)

    if not honest:
        return BFTConsensusCertificate(
            seq=seq_i,
            quorum_reached=False,
            passed=False,
            trimmed_score=0.0,
            total_verifiers=0,
            fault_tolerance_f=f,
            quorum_size=quorum_req,
            slashed_verifiers=slashed,
        )

    pass_votes = sum(1 for v in honest if v.passed)
    fail_votes = n - pass_votes
    quorum_ok = n >= quorum_req and (pass_votes >= quorum_req or fail_votes >= quorum_req)
    passed = quorum_ok and pass_votes >= quorum_req

    # Byzantine trimmed mean: drop lowest f and highest f scores when n > 2f
    scores = sorted(_as_score(v.score) for v in honest)
    if f > 0 and len(scores) > 2 * f:
        trimmed = scores[f:len(scores) - f]
    else:
        trimmed = scores
    t_mean = sum(trimmed) / len(trimmed) if trimmed else 0.0

    return BFTConsensusCertificate(
        seq=seq_i,
        quorum_reached=quorum_ok,
        passed=passed,
        trimmed_score=t_mean,
        total_verifiers=n,
        fault_tolerance_f=f,
        quorum_size=quorum_req,
        slashed_verifiers=slashed,
    )


class ByzantineQuorumEngine:
    def adjudicate(self, votes: Sequence[VerifierVote], seq: int = 1) -> BFTConsensusCertificate:
        return adjudicate_byzantine_quorum(votes, seq=seq)


def _self_test() -> int:
    # 4 honest verifiers (score ~95, pass=True), 1 Byzantine liar (score=0, pass=False),
    # and 1 equivocating Byzantine node sending two conflicting votes
    votes = [
        VerifierVote("v1", 1, True, 96.0),
        VerifierVote("v2", 1, True, 94.0),
        VerifierVote("v3", 1, True, 95.0),
        VerifierVote("v4", 1, True, 95.0),
        VerifierVote("byz_liar", 1, False, 0.0),
        VerifierVote("byz_equiv", 1, True, 100.0),
        VerifierVote("byz_equiv", 1, False, 10.0),
    ]
    cert = adjudicate_byzantine_quorum(votes, seq=1)
    assert "byz_equiv" in cert.slashed_verifiers
    assert cert.quorum_reached and cert.passed
    # Trimmed mean drops the 0.0 outlier and the 96.0 top, averaging [94, 95, 95] = 94.67
    assert 94.0 <= cert.trimmed_score <= 96.0, cert.to_dict()

    # Empty / None / garbage vote sets never raise and never finalize.
    for bad in (None, [], [None, 42, "x", {}, []]):
        c = adjudicate_byzantine_quorum(bad, seq=1)  # type: ignore[arg-type]
        assert not c.quorum_reached and not c.passed and c.total_verifiers == 0
    assert adjudicate_byzantine_quorum([]).quorum_size == 1

    # Votes for another sequence are ignored.
    c_other = adjudicate_byzantine_quorum(
        [VerifierVote("v1", 2, True, 90.0), VerifierVote("v2", 2, False, 10.0)], seq=1
    )
    assert c_other.total_verifiers == 0 and not c_other.quorum_reached

    # A single survivor must not finalize after its peer equivocates.
    c_slash = adjudicate_byzantine_quorum(
        [
            VerifierVote("only", 1, True, 99.0),
            VerifierVote("liar", 1, True, 99.0),
            VerifierVote("liar", 1, False, 1.0),
        ]
    )
    assert "liar" in c_slash.slashed_verifiers
    assert c_slash.total_verifiers == 1 and not c_slash.quorum_reached and not c_slash.passed

    # Equivocation with identical `passed` but conflicting scores is slashed.
    c_score = adjudicate_byzantine_quorum(
        [VerifierVote("e", 1, True, 10.0), VerifierVote("e", 1, True, 90.0),
         VerifierVote("a", 1, True, 80.0), VerifierVote("b", 1, True, 81.0),
         VerifierVote("c", 1, True, 82.0)]
    )
    assert "e" in c_score.slashed_verifiers

    # Duplicate identical votes count once and are not equivocation.
    c_dup = adjudicate_byzantine_quorum(
        [VerifierVote("a", 1, True, 80.0), VerifierVote("a", 1, True, 80.0),
         VerifierVote("b", 1, True, 81.0), VerifierVote("c", 1, True, 82.0),
         VerifierVote("d", 1, True, 83.0)]
    )
    assert c_dup.total_verifiers == 4 and c_dup.slashed_verifiers == []

    # Dict / tuple votes are accepted; NaN scores cannot poison the mean.
    c_coerced = adjudicate_byzantine_quorum([
        {"verifier_id": "v1", "seq": 1, "passed": True, "score": float("nan")},
        ("v2", 1, True, 90.0),
        ("v3", 1, True, 91.0),
        ("v4", 1, True, 92.0),
    ])
    assert c_coerced.total_verifiers == 4 and c_coerced.passed
    assert 0.0 <= c_coerced.trimmed_score <= 100.0

    # 3f+1 = 7 nodes tolerate f=2: with 2 Byzantine dissenters, 5 honest
    # pass votes still reach the 2f+1 = 5 quorum.
    votes7 = [VerifierVote(f"h{i}", 1, True, 90.0) for i in range(5)] + [
        VerifierVote("b1", 1, False, 5.0), VerifierVote("b2", 1, False, 6.0)
    ]
    c7 = adjudicate_byzantine_quorum(votes7)
    assert c7.fault_tolerance_f == 2 and c7.quorum_size == 5
    assert c7.passed and c7.quorum_reached

    # Determinism: identical inputs give byte-identical reports.
    assert cert.to_dict() == adjudicate_byzantine_quorum(votes, seq=1).to_dict()
    assert ByzantineQuorumEngine().adjudicate(votes).to_dict() == cert.to_dict()

    print("byzantine._self_test: OK PASS (equivocation slashing, f-trimmed mean, 2f+1 PBFT finality)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
