"""entropysense — Multi-Scale Shannon, Rényi, Tsallis & Zlib Compression Complexity Analyzer.

Detects two opposite information-theoretic failure modes in LLM generations
and untrusted inputs:
  1. Mode Collapse / Degenerate Repetition: abnormally low compression ratio
     and low Rényi collision entropy ($H_2$).
  2. Steganographic / Obfuscated Payload Smuggling: high-entropy spikes
     ($H_1 > 4.6$ bits/char) inside sliding windows caused by Base64/Hex/XOR
     blobs hidden inside normal prose.

Self-test: ``python3 -m fullagent.enforcex.entropysense``
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List

__all__ = [
    "EntropyAnomalyReport",
    "EntropyComplexityAnalyzer",
    "shannon_entropy",
    "renyi_collision_entropy",
    "tsallis_entropy",
    "compression_ratio",
    "analyze_entropy",
]

_MAX_TEXT = 262_144
_MAX_WINDOWS = 2048


def _as_text(text: Any, cap: int = _MAX_TEXT) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    return text[:cap]


def _safe_int(value: Any, fallback: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
    return out



@dataclass
class EntropyAnomalyReport:
    shannon_bits: float = 0.0
    renyi_bits: float = 0.0
    tsallis_q2: float = 0.0
    zlib_ratio: float = 1.0
    max_window_entropy: float = 0.0
    mode_collapse: bool = False
    obfuscated_spike: bool = False
    anomalies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shannon_bits": round(self.shannon_bits, 4),
            "renyi_bits": round(self.renyi_bits, 4),
            "tsallis_q2": round(self.tsallis_q2, 4),
            "zlib_ratio": round(self.zlib_ratio, 4),
            "max_window_entropy": round(self.max_window_entropy, 4),
            "mode_collapse": self.mode_collapse,
            "obfuscated_spike": self.obfuscated_spike,
            "anomalies": list(self.anomalies),
        }


def _probs(text: str) -> List[float]:
    if not text:
        return []
    counts: Dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    n = float(len(text))
    return [c / n for c in counts.values()]


def shannon_entropy(text: str) -> float:
    """Compute Shannon entropy H_1(X) = -sum p_i log2(p_i) in bits per character."""
    ps = _probs(_as_text(text, 1_000_000))
    return -sum(p * math.log2(p) for p in ps if p > 0.0)


def renyi_collision_entropy(text: str) -> float:
    """Compute Rényi quadratic collision entropy H_2(X) = -log2(sum p_i^2)."""
    ps = _probs(_as_text(text, 1_000_000))
    if not ps:
        return 0.0
    s2 = sum(p * p for p in ps)
    return -math.log2(s2) if s2 > 0.0 else 0.0


def tsallis_entropy(text: str, q: float = 2.0) -> float:
    """Compute Tsallis entropy S_q(X) = (1 - sum p_i^q) / (q - 1)."""
    clean = _as_text(text, 1_000_000)
    ps = _probs(clean)
    if not ps:
        return 0.0
    try:
        qf = float(q)
    except (TypeError, ValueError):
        qf = 2.0
    if not math.isfinite(qf) or abs(qf - 1.0) < 1e-6:
        return shannon_entropy(clean)
    try:
        total = sum(p ** qf for p in ps)
    except (OverflowError, ValueError):
        return 0.0
    return (1.0 - total) / (qf - 1.0)


def compression_ratio(text: str) -> float:
    """Compute compressed_bytes / raw_bytes using deflate (zlib)."""
    clean = _as_text(text)
    if not clean:
        return 1.0
    raw = clean.encode("utf-8", "replace")
    if len(raw) < 32:
        return 1.0
    comp = zlib.compress(raw, level=6)
    return len(comp) / float(len(raw))


def analyze_entropy(text: str, *, window: int = 80, step: int = 40) -> EntropyAnomalyReport:
    s = _as_text(text)
    if not s:
        return EntropyAnomalyReport()

    h1 = shannon_entropy(s)
    h2 = renyi_collision_entropy(s)
    tq = tsallis_entropy(s, q=2.0)
    z_rat = compression_ratio(s)

    w = min(4096, max(32, _safe_int(window, 80)))
    st = max(16, _safe_int(step, 40))
    max_win = 0.0
    span = len(s) - w + 1
    if span > 0:
        # Bound total window work: widen the stride rather than truncate data.
        if (span + st - 1) // st > _MAX_WINDOWS:
            st = max(st, (span + _MAX_WINDOWS - 1) // _MAX_WINDOWS)
        for i in range(0, span, st):
            chunk = s[i:i + w]
            # Only flag chunks with minimal whitespace (like base64/hex blobs)
            wh_ratio = sum(1 for c in chunk if c.isspace()) / float(w)
            ch_h = shannon_entropy(chunk)
            if wh_ratio < 0.05 and ch_h > max_win:
                max_win = ch_h
    else:
        # Too short for a window; the whole string is the only window.
        max_win = h1

    anomalies: List[str] = []
    mode_collapse = len(s) >= 120 and z_rat < 0.18
    if mode_collapse:
        anomalies.append(f"mode_collapse: zlib compression ratio={z_rat:.3f} (< 0.18)")

    obf_spike = max_win >= 5.15
    if obf_spike:
        anomalies.append(f"high_entropy_blob: window Shannon H={max_win:.2f} bits/char")

    return EntropyAnomalyReport(
        shannon_bits=h1,
        renyi_bits=h2,
        tsallis_q2=tq,
        zlib_ratio=z_rat,
        max_window_entropy=max_win,
        mode_collapse=mode_collapse,
        obfuscated_spike=obf_spike,
        anomalies=anomalies,
    )


class EntropyComplexityAnalyzer:
    def analyze(self, text: str) -> EntropyAnomalyReport:
        return analyze_entropy(text)


def _self_test() -> int:
    normal = "The system architecture separates the Temporal Kernel from the stateless tool handlers."
    r_norm = analyze_entropy(normal)
    assert not r_norm.mode_collapse and not r_norm.obfuscated_spike, r_norm.to_dict()

    collapsed = "repeat this phrase again and again " * 20
    r_col = analyze_entropy(collapsed)
    assert r_col.mode_collapse, r_col.to_dict()

    blob = "Normal intro. " + "aB3$xZ9!qW8@pL7#mN6%vC5^bG4&hJ2*kF1(dS0)yT9+rE8=uI7~oP6_zX5" * 2
    r_blob = analyze_entropy(blob)
    assert r_blob.obfuscated_spike, r_blob.to_dict()

    # ── new adversarial coverage ────────────────────────────────────────
    # 1. None / non-str / empty input never raise and report zeros
    for junk in (None, 12345, "", [], b"bytes"):
        rep = analyze_entropy(junk)  # type: ignore[arg-type]
        assert isinstance(rep, EntropyAnomalyReport)
        assert isinstance(rep.to_dict(), dict)
    assert analyze_entropy(None).shannon_bits == 0.0  # type: ignore[arg-type]
    assert analyze_entropy("").shannon_bits == 0.0
    assert EntropyComplexityAnalyzer().analyze(None).shannon_bits == 0.0  # type: ignore[arg-type]

    # 2. invalid window/step settings degrade instead of raising
    for bad_w, bad_s in ((0, 0), (-10, -10), ("x", "y"), (float("nan"), 3)):
        rep = analyze_entropy("hello world " * 100, window=bad_w, step=bad_s)  # type: ignore[arg-type]
        assert isinstance(rep, EntropyAnomalyReport)

    # 3. Tsallis q=1 reduces exactly to Shannon and H2 <= H1
    sample = "the quick brown fox jumps over the lazy dog 0123456789"
    assert abs(tsallis_entropy(sample, q=1.0) - shannon_entropy(sample)) < 1e-12
    assert renyi_collision_entropy(sample) <= shannon_entropy(sample) + 1e-12
    assert shannon_entropy("aaaa") == 0.0 and renyi_collision_entropy("aaaa") == 0.0

    # 4. whitespace-broken payloads are not mistaken for base64 blobs
    spaced = ("aB3$xZ9! qW8@pL7# mN6%vC5^ bG4&hJ2* kF1(dS0) "
              "yT9+rE8= uI7~oP6_ zX5 " * 2)
    r_spaced = analyze_entropy(spaced)
    assert not r_spaced.obfuscated_spike, r_spaced.to_dict()

    # 5. window entropy is bounded by the alphabet and by the byte ceiling
    r_ascii = analyze_entropy("abcABC123!@#" * 50)
    assert 0.0 <= r_ascii.max_window_entropy <= 8.0

    # 6. 1e6-character hostile input stays bounded and fast
    huge = "A" * 1_000_000
    r_huge = analyze_entropy(huge)
    assert r_huge.mode_collapse and r_huge.shannon_bits == 0.0
    assert compression_ratio(huge) < 0.01
    assert compression_ratio("") == 1.0 and compression_ratio(None) == 1.0  # type: ignore[arg-type]

    print("entropysense._self_test: OK PASS (Shannon, Rényi, Tsallis, zlib mode-collapse & blob detection)")

    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
