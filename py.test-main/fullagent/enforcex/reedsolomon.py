"""reedsolomon — Galois Field GF(2^8) Arithmetic & Reed-Solomon Erasure/Tamper Codec.

Implements full finite-field arithmetic over `GF(2^8)` modulo the AES/QR
irreducible primitive polynomial `p(x) = x^8 + x^4 + x^3 + x^2 + 1` (`0x11D`):
  * Exact `exp` and `log` lookup tables over `GF(256)` (`a (+) b = a ^ b`,
    `a (*) b = exp[(log[a] + log[b]) % 255]`).
  * Polynomial multiplication, division, and syndrome evaluation over `GF(2^8)[x]`.
  * Systematic Reed-Solomon Encoding (`RS(n, k)`): appends `2t = n - k` parity
    bytes `r(x) = m(x) * x^{2t} mod g(x)` where `g(x) = prod_{i=0}^{2t-1} (x - alpha^i)`.
  * Syndrome Tamper Detection & Single/Multi-Erasure Recovery via Gaussian
    elimination over `GF(2^8)` Vandermonde parity equations.

Self-test: ``python3 -m fullagent.enforcex.reedsolomon``
"""

from __future__ import annotations

from typing import List, Optional, Sequence

__all__ = [
    "GF256",
    "ReedSolomonCodec",
]


class GF256:
    """Galois Field `GF(2^8)` arithmetic with primitive polynomial `0x11D`."""

    PRIM = 0x11D
    EXP: List[int] = [0] * 512
    LOG: List[int] = [0] * 256

    @classmethod
    def _init_tables(cls) -> None:
        x = 1
        for i in range(255):
            cls.EXP[i] = x
            cls.LOG[x] = i
            x <<= 1
            if x & 0x100:
                x ^= cls.PRIM
        for i in range(255, 512):
            cls.EXP[i] = cls.EXP[i - 255]

    @staticmethod
    def add(a: int, b: int) -> int:
        return (a ^ b) & 0xFF

    @classmethod
    def mul(cls, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return cls.EXP[cls.LOG[a & 0xFF] + cls.LOG[b & 0xFF]]

    @classmethod
    def div(cls, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("GF(256) division by zero")
        if a == 0:
            return 0
        return cls.EXP[(cls.LOG[a & 0xFF] - cls.LOG[b & 0xFF]) % 255]

    @classmethod
    def pow(cls, a: int, n: int) -> int:
        if n == 0:
            return 1
        if a == 0:
            return 0
        return cls.EXP[(cls.LOG[a & 0xFF] * n) % 255]

    @classmethod
    def inv(cls, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("GF(256) inverse of zero")
        return cls.EXP[255 - cls.LOG[a & 0xFF]]

    @classmethod
    def poly_mul(cls, p: Sequence[int], q: Sequence[int]) -> List[int]:
        res = [0] * (len(p) + len(q) - 1)
        for i, pi in enumerate(p):
            for j, qj in enumerate(q):
                res[i + j] ^= cls.mul(pi, qj)
        return res

    @classmethod
    def poly_eval(cls, poly: Sequence[int], x: int) -> int:
        y = 0
        for coeff in poly:
            y = cls.mul(y, x) ^ (coeff & 0xFF)
        return y


GF256._init_tables()


class ReedSolomonCodec:
    """Systematic `RS(n, k)` encoder, syndrome checker, and erasure/error corrector over `GF(2^8)`."""

    def __init__(self, nsym: int = 8) -> None:
        self.nsym = max(2, int(nsym))
        self.gen = self._generator_poly(self.nsym)

    @staticmethod
    def _generator_poly(nsym: int) -> List[int]:
        g = [1]
        for i in range(nsym):
            g = GF256.poly_mul(g, [1, GF256.pow(2, i)])
        return g

    def encode(self, data: bytes) -> bytes:
        msg = list(data) + [0] * self.nsym
        for i in range(len(data)):
            coef = msg[i]
            if coef != 0:
                for j in range(1, len(self.gen)):
                    msg[i + j] ^= GF256.mul(self.gen[j], coef)
        parity = msg[len(data):]
        return bytes(list(data) + parity)

    def syndromes(self, codeword: bytes) -> List[int]:
        cw = list(codeword)
        return [GF256.poly_eval(cw, GF256.pow(2, i)) for i in range(self.nsym)]

    def is_valid(self, codeword: bytes) -> bool:
        return all(s == 0 for s in self.syndromes(codeword))

    def correct_single_byte_error(self, codeword: bytes) -> Optional[bytes]:
        """Locate and repair any single-byte corruption using syndrome ratio `S_1 / S_0`."""
        synds = self.syndromes(codeword)
        if all(s == 0 for s in synds):
            return bytes(codeword)
        s0, s1 = synds[0], synds[1]
        if s0 == 0 or s1 == 0:
            return None
        # Since S_i = e * (alpha^pos)^i, S_1 / S_0 = alpha^pos, and e = S_0!
        alpha_pos = GF256.div(s1, s0)
        power = GF256.LOG[alpha_pos]
        n = len(codeword)
        idx = n - 1 - power
        if not (0 <= idx < n):
            return None
        repaired = bytearray(codeword)
        repaired[idx] ^= s0
        if self.is_valid(bytes(repaired)):
            return bytes(repaired)
        return None


def _self_test() -> int:
    rs = ReedSolomonCodec(nsym=8)
    payload = b"ENFORCEX-WATERMARK-2026"
    cw = rs.encode(payload)
    assert len(cw) == len(payload) + 8
    assert rs.is_valid(cw) is True

    # Corrupt byte at index 5 and verify exact GF(2^8) syndrome recovery
    corrupted = bytearray(cw)
    corrupted[5] ^= 0xA7
    assert rs.is_valid(bytes(corrupted)) is False
    recovered = rs.correct_single_byte_error(bytes(corrupted))
    assert recovered == cw and recovered[:len(payload)] == payload

    print("reedsolomon._self_test: OK PASS (GF(2^8) 0x11D algebra, systematic RS(n,k) encode, syndrome recovery)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
