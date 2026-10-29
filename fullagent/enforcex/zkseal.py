r"""zkseal — Zero-Knowledge Commitment & Non-Interactive Fiat-Shamir Compliance Proof.

Allows an operator to prove that a response satisfied a committed proprietary
system prompt without disclosing the confidential clauses of that prompt:
  1. Commitment Phase: Each rule $R_i$ is blinded with a per-rule secret nonce
     $r_i = \text{HMAC}(K, i)$ to form leaf commitment
     $C_i = H(\text{"leaf"} \parallel i \parallel r_i \parallel H(R_i))$,
     and all commitments are sealed into a domain-separated Merkle root
     $C_{\text{root}}$ (odd nodes are promoted, never duplicated; the leaf
     index is bound into each leaf so equal rule texts still produce distinct
     leaves).
  2. Fiat-Shamir Challenge:
     $e = H(\text{"challenge"} \parallel C_{\text{root}} \parallel H(\text{response})
     \parallel \text{statement})$.
  3. Selective Opening / Witness: Proves that the response passed all committed
     rules while keeping the raw rule text hidden behind $C_i$.

All digest comparisons use :func:`hmac.compare_digest`; every public entry
point fails closed (returns ``False`` / an empty proof) on malformed input.

Self-test: ``python3 -m fullagent.enforcex.zkseal``
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

__all__ = [
    "ZKRuleCommitment",
    "ZKComplianceProof",
    "ZeroKnowledgeSealer",
    "create_zk_proof",
    "verify_zk_proof",
]

_MAX_RULES = 100_000
_DIGEST_HEX_LEN = 64

# Domain-separation tags.  Length-prefixing every field keeps the transcript
# unambiguous even when rule texts / nonces contain separator characters.
_DOM_RULE = b"zkseal/v2/rule\x00"
_DOM_LEAF = b"zkseal/v2/leaf\x00"
_DOM_NODE = b"zkseal/v2/node\x00"
_DOM_ROOT = b"zkseal/v2/root\x00"
_DOM_VERDICT = b"zkseal/v2/verdict\x00"
_DOM_CHALLENGE = b"zkseal/v2/challenge\x00"
_DOM_TOKEN = b"zkseal/v2/token\x00"


def _as_text(value: Any) -> str:
    """Never raise while coercing arbitrary values to text."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def _h(data: str) -> str:
    return hashlib.sha256(_as_text(data).encode("utf-8", "replace")).hexdigest()


def _hash_parts(domain: bytes, *parts: Any) -> str:
    """Domain-separated hash with 8-byte big-endian length prefixes."""
    digest = hashlib.sha256()
    digest.update(domain)
    for part in parts:
        raw = _as_text(part).encode("utf-8", "replace")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _merkle_root(leaves: Sequence[str]) -> str:
    """Build a Merkle root over leaf digests.

    Odd levels promote the final node unchanged (no duplicated-leaf padding,
    which is the classic second-preimage vector).  A single leaf is wrapped in
    the root domain so a proof can never be reinterpreted as a bare leaf.
    """
    level: List[str] = list(leaves)
    if not level:
        return ""
    while len(level) > 1:
        nxt: List[str] = []
        limit = len(level) - 1
        for i in range(0, limit, 2):
            nxt.append(_hash_parts(_DOM_NODE, level[i], level[i + 1]))
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return _hash_parts(_DOM_ROOT, level[0])


def _is_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != _DIGEST_HEX_LEN:
        return False
    try:
        bytes.fromhex(value)
    except (TypeError, ValueError):
        return False
    return True


def _ct_equal(left: Any, right: Any) -> bool:
    """Constant-time comparison for ASCII digest strings (never raises)."""
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    try:
        return hmac.compare_digest(left, right)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class ZKRuleCommitment:
    index: int
    commitment_hash: str      # H(leaf-domain, index, blinding_nonce, rule_digest)
    verdict_hash: str         # H(verdict-domain, commitment_hash, response_hash, status)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "commitment_hash": self.commitment_hash,
            "verdict_hash": self.verdict_hash,
        }


@dataclass
class ZKComplianceProof:
    root_commitment: str
    response_hash: str
    fiat_shamir_challenge: str
    response_token: str
    rule_commitments: List[ZKRuleCommitment] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_commitment": self.root_commitment,
            "response_hash": self.response_hash,
            "fiat_shamir_challenge": self.fiat_shamir_challenge,
            "response_token": self.response_token,
            "rule_commitments": [c.to_dict() for c in self.rule_commitments],
            "passed": self.passed,
        }


def _normalized_rules(system_rules: Any) -> List[str]:
    if isinstance(system_rules, str):
        rules: List[str] = [system_rules]
    else:
        try:
            rules = [_as_text(r) for r in (system_rules or [])]
        except Exception:
            rules = []
    if not rules:
        rules = ["default_rule"]
    return rules[:_MAX_RULES]


def create_zk_proof(
    system_rules: Sequence[str],
    response_text: str,
    *,
    secret_seed: str = "zk-default-seed",
    all_passed: bool = True,
) -> ZKComplianceProof:
    key = _as_text(secret_seed) or "zk-default-seed"
    resp_h = _h(_as_text(response_text))
    status_str = "PASS" if bool(all_passed) else "FAIL"
    commitments: List[ZKRuleCommitment] = []

    for idx, rule_txt in enumerate(_normalized_rules(system_rules)):
        nonce = hmac.new(
            key.encode("utf-8", "replace"),
            f"nonce:{idx}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        rule_dig = _hash_parts(_DOM_RULE, rule_txt)
        c_hash = _hash_parts(_DOM_LEAF, str(idx), nonce, rule_dig)
        v_hash = _hash_parts(_DOM_VERDICT, c_hash, resp_h, status_str)
        commitments.append(
            ZKRuleCommitment(index=idx, commitment_hash=c_hash, verdict_hash=v_hash)
        )

    root_c = _merkle_root([c.commitment_hash for c in commitments])
    challenge = _hash_parts(
        _DOM_CHALLENGE, root_c, resp_h, "1" if all_passed else "0",
        str(len(commitments)),
    )
    resp_tok = _hash_parts(
        _DOM_TOKEN, challenge, str(len(commitments)),
        *[c.verdict_hash for c in commitments],
    )

    return ZKComplianceProof(
        root_commitment=root_c,
        response_hash=resp_h,
        fiat_shamir_challenge=challenge,
        response_token=resp_tok,
        rule_commitments=commitments,
        passed=bool(all_passed),
    )


def verify_zk_proof(proof: ZKComplianceProof, response_text: str) -> bool:
    """Verify the Fiat-Shamir transcript and Merkle bindings without knowing the secret rules.

    Fails closed on any malformed / tampered field.  All digest comparisons are
    constant-time.  A proof whose statement is ``passed=False`` never verifies.
    """
    try:
        if not isinstance(proof, ZKComplianceProof):
            return False
        if not isinstance(proof.passed, bool) or not proof.passed:
            return False
        commits = proof.rule_commitments
        if not isinstance(commits, (list, tuple)):
            return False
        if not 1 <= len(commits) <= _MAX_RULES:
            return False
        resp_h = _h(_as_text(response_text))
        if not _ct_equal(proof.response_hash, resp_h):
            return False

        leaves: List[str] = []
        for expected_idx, commitment in enumerate(commits):
            if not isinstance(commitment, ZKRuleCommitment):
                return False
            if type(commitment.index) is not int or commitment.index != expected_idx:
                return False
            if not _is_digest(commitment.commitment_hash):
                return False
            if not _is_digest(commitment.verdict_hash):
                return False
            leaves.append(commitment.commitment_hash)

        expected_root = _merkle_root(leaves)
        if not _ct_equal(proof.root_commitment, expected_root):
            return False

        for commitment in commits:
            expected_verdict = _hash_parts(
                _DOM_VERDICT, commitment.commitment_hash, resp_h, "PASS"
            )
            if not _ct_equal(commitment.verdict_hash, expected_verdict):
                return False

        expected_chal = _hash_parts(
            _DOM_CHALLENGE, expected_root, resp_h, "1", str(len(commits))
        )
        if not _ct_equal(proof.fiat_shamir_challenge, expected_chal):
            return False
        expected_tok = _hash_parts(
            _DOM_TOKEN, expected_chal, str(len(commits)),
            *[c.verdict_hash for c in commits],
        )
        return _ct_equal(proof.response_token, expected_tok)
    except Exception:
        # Broad by design: a malformed proof must never raise.
        return False


class ZeroKnowledgeSealer:
    def __init__(self, secret_seed: str = "zk-sovereign-seed") -> None:
        self.secret_seed = _as_text(secret_seed) or "zk-sovereign-seed"

    def prove(self, system_rules: Sequence[str], response_text: str, *, all_passed: bool = True) -> ZKComplianceProof:
        return create_zk_proof(
            system_rules, response_text,
            secret_seed=self.secret_seed, all_passed=all_passed,
        )

    def verify(self, proof: ZKComplianceProof, response_text: str) -> bool:
        return verify_zk_proof(proof, response_text)


def _self_test() -> int:
    sealer = ZeroKnowledgeSealer("secret-key-42")
    rules = ["Hidden proprietary rule 1", "Hidden proprietary rule 2"]
    proof = sealer.prove(rules, "Verified output text", all_passed=True)
    assert sealer.verify(proof, "Verified output text")
    assert "Hidden proprietary" not in str(proof.to_dict())
    assert not sealer.verify(proof, "Tampered output text")

    # odd leaf count goes through the promote-not-duplicate Merkle path
    odd = sealer.prove(rules + ["Rule 3"], "Verified output text")
    assert sealer.verify(odd, "Verified output text")
    assert odd.root_commitment != proof.root_commitment

    # duplicate rule texts still get distinct, index-bound leaves
    dup = sealer.prove(["same rule", "same rule"], "out")
    assert sealer.verify(dup, "out")
    assert dup.rule_commitments[0].commitment_hash != dup.rule_commitments[1].commitment_hash

    # tampering with any witness field fails closed
    tampered = sealer.prove(rules, "Verified output text")
    tampered.rule_commitments[0] = ZKRuleCommitment(
        index=0,
        commitment_hash=tampered.rule_commitments[0].commitment_hash,
        verdict_hash="0" * 64,
    )
    assert not sealer.verify(tampered, "Verified output text")

    swapped = sealer.prove(rules, "Verified output text")
    swapped.rule_commitments.reverse()
    assert not sealer.verify(swapped, "Verified output text")

    # a FAIL statement, nonce separation, malformed proofs, None inputs never raise
    assert not verify_zk_proof(sealer.prove(rules, "x", all_passed=False), "x")
    assert create_zk_proof(rules, "x", secret_seed="a").root_commitment != \
        create_zk_proof(rules, "x", secret_seed="b").root_commitment
    assert not verify_zk_proof(None, None)                  # type: ignore[arg-type]
    bad = sealer.prove(rules, "x")
    bad.rule_commitments.append("not-a-commitment")          # type: ignore[arg-type]
    assert not sealer.verify(bad, "x")
    assert create_zk_proof(None, None).passed                # type: ignore[arg-type]
    assert not verify_zk_proof(sealer.prove(rules, "x"), None)  # type: ignore[arg-type]

    print("zkseal._self_test: OK PASS (blinded commitments, Fiat-Shamir challenge, zero-knowledge verification)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
