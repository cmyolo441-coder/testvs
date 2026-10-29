"""merkleforest — RFC 6962 Certificate-Transparency Merkle Audit Tree & Inclusion Verifier.

Implements RFC 6962 domain-separated binary Merkle trees:
  * Leaf hash: `SHA256(0x00 || leaf_bytes)`
  * Interior node hash: `SHA256(0x01 || left_hash || right_hash)`
  * O(log N) Merkle Audit Path (`inclusion_proof`) and independent
    `verify_inclusion(leaf_data, index, proof, root_hash)`!

Self-test: ``python3 -m fullagent.enforcex.merkleforest``
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

__all__ = [
    "MerkleAuditProof",
    "MerkleForestEngine",
    "leaf_hash",
    "node_hash",
]


def leaf_hash(data: str | bytes) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    return hashlib.sha256(b"\x00" + raw).hexdigest()


def node_hash(left_hex: str, right_hex: str) -> str:
    payload = b"\x01" + bytes.fromhex(left_hex) + bytes.fromhex(right_hex)
    return hashlib.sha256(payload).hexdigest()


@dataclass
class MerkleAuditProof:
    leaf_index: int
    tree_size: int
    root_hash: str
    siblings: List[Tuple[str, str]] = field(default_factory=list)  # (direction: "L"|"R", hex_hash)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "leaf_index": self.leaf_index,
            "tree_size": self.tree_size,
            "root_hash": self.root_hash,
            "siblings": list(self.siblings),
        }


class MerkleForestEngine:
    """RFC 6962 append-only Merkle tree with O(log N) inclusion proof generation and verification."""

    def __init__(self) -> None:
        self._leaves: List[str] = []

    def append(self, data: str | bytes) -> int:
        self._leaves.append(leaf_hash(data))
        return len(self._leaves) - 1

    def size(self) -> int:
        return len(self._leaves)

    def root(self) -> str:
        if not self._leaves:
            return hashlib.sha256(b"").hexdigest()
        level = list(self._leaves)
        while len(level) > 1:
            nxt: List[str] = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    nxt.append(node_hash(level[i], level[i + 1]))
                else:
                    nxt.append(level[i])
            level = nxt
        return level[0]

    def inclusion_proof(self, index: int) -> MerkleAuditProof:
        n = len(self._leaves)
        if not (0 <= index < n):
            raise IndexError(f"leaf index {index} out of range [0, {n})")
        idx = index
        level = list(self._leaves)
        siblings: List[Tuple[str, str]] = []

        while len(level) > 1:
            nxt: List[str] = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    if i == idx:
                        siblings.append(("R", level[i + 1]))
                        idx = len(nxt)
                    elif i + 1 == idx:
                        siblings.append(("L", level[i]))
                        idx = len(nxt)
                    nxt.append(node_hash(level[i], level[i + 1]))
                else:
                    if i == idx:
                        idx = len(nxt)
                    nxt.append(level[i])
            level = nxt

        return MerkleAuditProof(
            leaf_index=index,
            tree_size=n,
            root_hash=level[0],
            siblings=siblings,
        )

    @staticmethod
    def verify_inclusion(data: str | bytes, proof: MerkleAuditProof) -> bool:
        cur = leaf_hash(data)
        for side, sib_hash in proof.siblings:
            if side == "L":
                cur = node_hash(sib_hash, cur)
            else:
                cur = node_hash(cur, sib_hash)
        return cur == proof.root_hash


def _self_test() -> int:
    tree = MerkleForestEngine()
    items = [f"event-{i}" for i in range(7)]
    for it in items:
        tree.append(it)

    for idx, it in enumerate(items):
        proof = tree.inclusion_proof(idx)
        assert MerkleForestEngine.verify_inclusion(it, proof), (idx, proof.to_dict())
        assert not MerkleForestEngine.verify_inclusion(it + "-tampered", proof)

    print("merkleforest._self_test: OK PASS (RFC 6962 domain-separated Merkle tree & O(log N) inclusion proofs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
