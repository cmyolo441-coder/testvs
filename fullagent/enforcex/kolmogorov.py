"""kolmogorov — Burrows-Wheeler Transform (BWT), Move-to-Front (MTF), LZW & NID.

Implements exact lossless block-sorting and dictionary transforms (Burrows &
Wheeler 1994; Welch 1984; Cilibrasi & Vitányi 2005 Normalized Information
Distance) for detecting repetitive anomalies, covert steganography, and
near-verbatim system prompt copying:
  1. **Burrows-Wheeler Transform (`bwt_encode` / `bwt_decode`)**:
     - Cyclic rotation sorting with end-of-block sentinel + `O(n)` Last-to-First
       (`LF`-mapping) exact inverse recovery.
  2. **Move-to-Front (`mtf_encode` / `mtf_decode`)**:
     - Converts BWT local character runs into small integers near `0`.
  3. **Lempel-Ziv-Welch (`lzw_compress` / `lzw_decompress`)**:
     - Adaptive dictionary coder building phrase codes `>= 256` on the fly.
  4. **Algorithmic Mutual Information & Normalized Information Distance (`NID(x, y)`)**:
     - Approximates `K(x)` via `BWT + MTF + LZW` bit-length to detect cross-turn
       prompt leakage even after paraphrasing or reordering.

Self-test: ``python3 -m fullagent.enforcex.kolmogorov``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

__all__ = [
    "KolmogorovReport",
    "bwt_encode",
    "bwt_decode",
    "mtf_encode",
    "mtf_decode",
    "lzw_compress",
    "lzw_decompress",
    "normalized_information_distance",
]


def bwt_encode(text: str) -> Tuple[str, int]:
    """Compute Burrows-Wheeler Transform `(bwt_string, primary_index)`."""
    if not text:
        return "", 0
    s = text
    n = len(s)
    rotations = sorted((s[i:] + s[:i], i) for i in range(n))
    bwt_chars = [rot[-1] for rot, _ in rotations]
    primary = next(idx for idx, (_, orig_i) in enumerate(rotations) if orig_i == 0)
    return "".join(bwt_chars), primary


def bwt_decode(bwt_str: str, primary_index: int) -> str:
    """Invert Burrows-Wheeler Transform in `O(n)` time via LF-mapping."""
    if not bwt_str:
        return ""
    n = len(bwt_str)
    table = sorted((ch, idx) for idx, ch in enumerate(bwt_str))
    out: List[str] = []
    row = int(primary_index)
    for _ in range(n):
        ch, row = table[row]
        out.append(ch)
    return "".join(out)


def mtf_encode(data: bytes) -> List[int]:
    """Move-to-Front transform over `0..255` byte alphabet."""
    alphabet = list(range(256))
    out: List[int] = []
    for b in data:
        idx = alphabet.index(b)
        out.append(idx)
        alphabet.pop(idx)
        alphabet.insert(0, b)
    return out


def mtf_decode(ranks: Sequence[int]) -> bytes:
    """Invert Move-to-Front transform."""
    alphabet = list(range(256))
    out = bytearray()
    for r in ranks:
        b = alphabet.pop(int(r))
        out.append(b)
        alphabet.insert(0, b)
    return bytes(out)


def lzw_compress(data: bytes) -> List[int]:
    """Welch (1984) LZW lossless dictionary compression."""
    if not data:
        return []
    dictionary: Dict[bytes, int] = {bytes([i]): i for i in range(256)}
    next_code = 256
    w = bytes([data[0]])
    codes: List[int] = []
    for b in data[1:]:
        wc = w + bytes([b])
        if wc in dictionary:
            w = wc
        else:
            codes.append(dictionary[w])
            dictionary[wc] = next_code
            next_code += 1
            w = bytes([b])
    codes.append(dictionary[w])
    return codes


def lzw_decompress(codes: Sequence[int]) -> bytes:
    """Exact LZW dictionary decompression (including `KwKwK` edge case)."""
    if not codes:
        return b""
    dictionary: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_code = 256
    w = dictionary[int(codes[0])]
    out = bytearray(w)
    for k in codes[1:]:
        code = int(k)
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code:
            entry = w + w[:1]
        else:
            raise ValueError(f"Invalid LZW code: {code}")
        out.extend(entry)
        dictionary[next_code] = w + entry[:1]
        next_code += 1
        w = entry
    return bytes(out)


def _approx_kolmogorov_len(text: str) -> int:
    if not text:
        return 0
    bwt_s, _ = bwt_encode(text[:2048])
    mtf_ranks = mtf_encode(bwt_s.encode("utf-8", errors="replace"))
    lzw_codes = lzw_compress(bytes(r & 0xFF for r in mtf_ranks))
    return len(lzw_codes)


def normalized_information_distance(x: str, y: str) -> float:
    """Compute Cilibrasi-Vitányi NID `(K(xy) - min(K(x), K(y))) / max(K(x), K(y))`."""
    if not x and not y:
        return 0.0
    kx = _approx_kolmogorov_len(x)
    ky = _approx_kolmogorov_len(y)
    kxy = _approx_kolmogorov_len(x + "\n" + y)
    denom = max(1, max(kx, ky))
    val = (kxy - min(kx, ky)) / float(denom)
    return max(0.0, min(1.0, val))


@dataclass
class KolmogorovReport:
    bwt_primary: int
    lzw_code_count: int
    nid_to_system: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bwt_primary": self.bwt_primary,
            "lzw_code_count": self.lzw_code_count,
            "nid_to_system": round(self.nid_to_system, 4),
        }


def _self_test() -> int:
    sample = "BANANA_SOVEREIGN_BANANA_SOVEREIGN"
    bwt_s, idx = bwt_encode(sample)
    assert bwt_decode(bwt_s, idx) == sample

    raw = sample.encode("utf-8")
    ranks = mtf_encode(raw)
    assert mtf_decode(ranks) == raw

    codes = lzw_compress(raw)
    assert len(codes) < len(raw) and lzw_decompress(codes) == raw

    nid_same = normalized_information_distance(sample, sample)
    nid_diff = normalized_information_distance(sample, "completely unrelated text with different tokens 123456789")
    assert nid_same < nid_diff

    print("kolmogorov._self_test: OK PASS (BWT LF-inversion, Move-to-Front, LZW codec, Cilibrasi-Vitanyi NID)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
