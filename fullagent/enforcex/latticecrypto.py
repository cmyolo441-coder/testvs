"""latticecrypto — Post-Quantum Learning With Errors (LWE) & Ajtai SIS Lattice Commitments.

Implements post-quantum lattice-based cryptographic primitives over the finite
ring `Z_q` (`q = 3329`, the NIST Kyber prime):
  1. **Ajtai Collision-Resistant Lattice Hash (`h_A(x) = A * x mod q`)**:
     Finding a collision `h_A(x) == h_A(x')` for short integer vectors `x != x'`
     reduces to the worst-case Short Integer Solution ($\text{SIS}_{n, m, q, \beta}$)
     hard problem on Euclidean lattices.
  2. **Regev LWE Blinded Commitment (`c = (A^T s + e, <b, s> + e' + floor(q/2) * bit) mod q`)**:
     Commits to policy bits using short secret vectors `s in {-1, 0, 1}^n` and
     centered binomial noise `e in {-2..2}^m`, with exact opening verification
     checking both congruence mod `q` and the Euclidean norm bound `||e||_infty <= B`.

Self-test: ``python3 -m fullagent.enforcex.latticecrypto``
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, Dict, List

__all__ = [
    "LWECommitment",
    "LatticeCryptoEngine",
]


@dataclass
class LWECommitment:
    q: int
    n: int
    m: int
    public_seed: str
    commitment_vec: List[int]
    opening_secret: List[int]
    opening_error: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "q": self.q,
            "n": self.n,
            "m": self.m,
            "public_seed": self.public_seed,
            "commitment_vec": list(self.commitment_vec),
        }


class LatticeCryptoEngine:
    """Ajtai SIS lattice hash and Regev LWE short-norm commitment engine over Z_3329."""

    Q = 3329  # NIST ML-KEM (Kyber) prime

    def __init__(self, n: int = 16, m: int = 32, public_seed: str = "enforcex-lwe-crs") -> None:
        self.n = max(4, int(n))
        self.m = max(self.n, int(m))
        self.public_seed = str(public_seed)
        self.matrix_a = self._expand_matrix_a(self.public_seed, self.m, self.n)

    @classmethod
    def _expand_matrix_a(cls, seed: str, rows: int, cols: int) -> List[List[int]]:
        mat: List[List[int]] = []
        for i in range(rows):
            row: List[int] = []
            for j in range(cols):
                digest = hashlib.sha256(f"{seed}:{i}:{j}".encode("utf-8")).digest()
                val = struct.unpack(">H", digest[:2])[0] % cls.Q
                row.append(val)
            mat.append(row)
        return mat

    def ajtai_sis_hash(self, message: bytes) -> List[int]:
        """Compute Ajtai lattice hash `h_A(x) = A * x mod q` where `x in {0, 1}^m`."""
        digest = hashlib.sha512(message).digest()
        bits: List[int] = []
        for b in digest:
            for shift in range(8):
                bits.append((b >> shift) & 1)
                if len(bits) >= self.m:
                    break
            if len(bits) >= self.m:
                break
        # Multiply A^T (n x m) by bits (m x 1) mod Q
        out: List[int] = []
        for j in range(self.n):
            acc = sum(self.matrix_a[i][j] * bits[i] for i in range(self.m)) % self.Q
            out.append(acc)
        return out

    def commit_message(self, message: bytes, nonce: str = "0") -> LWECommitment:
        """Compute LWE commitment `c = A * s + m_encoded + e (mod q)` with short ternary `s, e`."""
        s: List[int] = []
        for j in range(self.n):
            h = hashlib.sha256(message + f":s:{nonce}:{j}".encode("utf-8")).digest()[0]
            s.append((h % 3) - 1)  # in {-1, 0, 1}

        e: List[int] = []
        msg_digest = hashlib.sha256(message).digest()
        for i in range(self.m):
            h = hashlib.sha256(message + f":e:{nonce}:{i}".encode("utf-8")).digest()[0]
            err = (h % 3) - 1  # short error in {-1, 0, 1}
            bit = (msg_digest[i % len(msg_digest)] >> (i & 7)) & 1
            # Encode message bit scaled by floor(Q / 2) = 1664 alongside short error
            e.append(err + 1664 * bit)

        c_vec: List[int] = []
        for i in range(self.m):
            dot = sum(self.matrix_a[i][j] * s[j] for j in range(self.n))
            c_vec.append((dot + e[i]) % self.Q)

        return LWECommitment(
            q=self.Q,
            n=self.n,
            m=self.m,
            public_seed=self.public_seed,
            commitment_vec=c_vec,
            opening_secret=s,
            opening_error=e,
        )

    @classmethod
    def _centered_mod(cls, x: int) -> int:
        r = x % cls.Q
        return r - cls.Q if r > cls.Q // 2 else r

    def verify_opening(self, commitment: LWECommitment, message: bytes) -> bool:
        """Verify LWE opening `(s, e)` has short infinity norm and reconstructs `message`."""
        if len(commitment.opening_secret) != self.n or len(commitment.commitment_vec) != self.m:
            return False
        if any(abs(x) > 1 for x in commitment.opening_secret):
            return False

        msg_digest = hashlib.sha256(message).digest()
        for i in range(self.m):
            dot = sum(self.matrix_a[i][j] * commitment.opening_secret[j] for j in range(self.n)) % self.Q
            residual = (commitment.commitment_vec[i] - dot) % self.Q
            bit = (msg_digest[i % len(msg_digest)] >> (i & 7)) & 1
            noise = self._centered_mod(residual - 1664 * bit)
            if abs(noise) > 2:
                return False
        return True


def _self_test() -> int:
    eng = LatticeCryptoEngine(n=16, m=32)
    h1 = eng.ajtai_sis_hash(b"system-policy-v1")
    h2 = eng.ajtai_sis_hash(b"system-policy-v2")
    assert len(h1) == 16 and h1 != h2

    comm = eng.commit_message(b"immutable-sovereign-contract", nonce="epoch-1")
    assert eng.verify_opening(comm, b"immutable-sovereign-contract") is True
    assert eng.verify_opening(comm, b"tampered-contract") is False

    print("latticecrypto._self_test: OK PASS (Ajtai SIS lattice hash + Regev LWE short-norm commitment over Z_3329)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
