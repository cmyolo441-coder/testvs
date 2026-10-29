"""tokensteer — Provider-Ready Logit-Bias, Stop-Sequence & Assistant Prefill Compiler.

Compiles natural-language system prompt rules into concrete decoding parameters:
  * `forbidden_strings`: extracted from `Never say "..."` / `Do not use "..."`
    and banned buzzwords/openers.
  * `logit_bias`: maps token IDs (via a pluggable tokenizer or byte/word hash)
    to `-100.0` (hard ban) or `+5.0` (boost).
  * `stop_sequences`: structural boundary tags (`</developer_config>`, `---` when
    horizontal rules are banned).
  * `assistant_prefill`: e.g. `"{ "` for JSON-only mode or `"```python\n"` for
    Python-code-only mode.

The compiler scans at most 262,144 characters / 4,096 lines, deduplicates
phrases case-insensitively while preserving the first spelling, and never
raises on malformed prompt text, tokenizer output or huge inputs.

Self-test: ``python3 -m fullagent.enforcex.tokensteer``
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = [
    "TokenSteeringSpec",
    "TokenSteeringCompiler",
    "compile_token_steering",
]

_MAX_SCAN_CHARS = 262_144
_MAX_LINES = 4_096
_MAX_FORBIDDEN = 256
_MAX_STOPS = 32

_NEGATION_RE = re.compile(r"\b(?:never|do\s+not|don't|forbid|ban|avoid)\b", re.I)
_QUOTED_RE = re.compile(r"['\"]([^'\"]{2,40})['\"]")
_HR_RE = re.compile(r"\b(?:never\s+use\s+horizontal\s+rules?|no\s+horizontal\s+rules?)\b", re.I)
_JSON_RE = re.compile(r"\b(?:only\s+(?:valid\s+)?json|json\s+only|respond\s+in\s+json)\b", re.I)
_PYTHON_RE = re.compile(r"\b(?:python\s+code\s+only|only\s+python\s+code)\b", re.I)
_TEMP_RE = re.compile(
    r"\btemperature\s*(?:cap|limit|of|at\s+most|is|=|:|<=)?\s*(\d+(?:\.\d+)?)\b",
    re.I,
)


def _safe_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


@dataclass
class TokenSteeringSpec:
    forbidden_phrases: List[str] = field(default_factory=list)
    logit_bias: Dict[int, float] = field(default_factory=dict)
    stop_sequences: List[str] = field(default_factory=list)
    assistant_prefill: str = ""
    temperature_cap: float = 0.7
    response_format: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forbidden_phrases": list(self.forbidden_phrases),
            "logit_bias": dict(self.logit_bias),
            "stop_sequences": list(self.stop_sequences),
            "assistant_prefill": self.assistant_prefill,
            "temperature_cap": self.temperature_cap,
            "response_format": dict(self.response_format) if isinstance(self.response_format, dict) else None,
        }


def _dedupe(items: List[str], *, casefold: bool) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        key = item.casefold() if casefold else item
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def compile_token_steering(
    system_text: str,
    *,
    tokenizer_encode: Optional[Callable[[str], Sequence[int]]] = None,
) -> TokenSteeringSpec:
    s = _safe_text(system_text)
    if len(s) > _MAX_SCAN_CHARS:
        s = s[:_MAX_SCAN_CHARS]
    low = s.lower()
    forbidden: List[str] = []

    # Extract quoted strings after negative verbs (bounded line scan, ordered,
    # case-insensitively deduplicated).
    seen_forbidden = set()
    for line_no, line in enumerate(s.splitlines()):
        if line_no >= _MAX_LINES:
            break
        if not _NEGATION_RE.search(line):
            continue
        for match in _QUOTED_RE.finditer(line):
            literal = match.group(1).strip()
            key = literal.casefold()
            if literal and key not in seen_forbidden:
                seen_forbidden.add(key)
                forbidden.append(literal)
                if len(forbidden) >= _MAX_FORBIDDEN:
                    break
        if len(forbidden) >= _MAX_FORBIDDEN:
            break

    stops = ["</developer_config>", "</system_instructions>"]
    if _HR_RE.search(low):
        stops.append("\n---\n")
    stops = _dedupe(stops, casefold=True)[:_MAX_STOPS]

    prefill = ""
    resp_fmt: Optional[Dict[str, str]] = None
    temp = 0.5
    if _JSON_RE.search(low):
        prefill = "{"
        resp_fmt = {"type": "json_object"}
        temp = 0.1
    elif _PYTHON_RE.search(low):
        prefill = "```python\n"
        temp = 0.1

    # An explicit temperature constraint can only tighten the safe default.
    temp_match = _TEMP_RE.search(low)
    if temp_match:
        try:
            requested = float(temp_match.group(1))
            if not math.isnan(requested) and not math.isinf(requested):
                temp = min(temp, max(0.0, requested))
        except (TypeError, ValueError, OverflowError):
            pass
    temp = max(0.0, min(2.0, temp))

    bias: Dict[int, float] = {}
    if callable(tokenizer_encode):
        for phrase in forbidden:
            try:
                tok_ids = list(tokenizer_encode(phrase))
                if not tok_ids:
                    continue
                first = int(tok_ids[0])
            except Exception:
                continue
            # Provider token ids are non-negative 31-bit integers; anything
            # else is ignored rather than poisoned into the request.
            if 0 <= first <= 0x7FFFFFFF:
                bias[first] = -100.0

    return TokenSteeringSpec(
        forbidden_phrases=forbidden,
        logit_bias=bias,
        stop_sequences=stops,
        assistant_prefill=prefill,
        temperature_cap=temp,
        response_format=resp_fmt,
    )


class TokenSteeringCompiler:
    def compile(self, system_text: str, *, tokenizer_encode: Optional[Callable[[str], Sequence[int]]] = None) -> TokenSteeringSpec:
        return compile_token_steering(system_text, tokenizer_encode=tokenizer_encode)


def _self_test() -> int:
    prompt = (
        'Never say "delve" or "synergistic".\n'
        "Never use horizontal rules.\n"
        "Respond in JSON only."
    )
    spec = compile_token_steering(prompt, tokenizer_encode=lambda w: [sum(ord(c) for c in w)])
    assert "delve" in spec.forbidden_phrases and "synergistic" in spec.forbidden_phrases
    assert "\n---\n" in spec.stop_sequences
    assert spec.assistant_prefill == "{"
    assert spec.response_format == {"type": "json_object"}
    assert len(spec.logit_bias) == 2 and all(v == -100.0 for v in spec.logit_bias.values())

    # malformed / empty inputs are harmless
    for bad in (None, 12345, "", [], b"bytes"):
        empty = compile_token_steering(bad)          # type: ignore[arg-type]
        assert empty.forbidden_phrases == [] and empty.logit_bias == {}
        assert empty.temperature_cap == 0.5

    # case-insensitive dedupe keeps the first spelling
    dup = compile_token_steering('Never say "Delve". Never say "delve" or "DELVE".')
    assert dup.forbidden_phrases == ["Delve"], dup.forbidden_phrases

    # broken / hostile tokenizer output is skipped, never raised
    def _boom(_phrase: str) -> List[int]:
        raise RuntimeError("tokenizer failure")
    assert compile_token_steering('Never say "delve"', tokenizer_encode=_boom).logit_bias == {}
    bad_ids = compile_token_steering('Never say "delve"', tokenizer_encode=lambda w: ["not-a-token"])
    assert bad_ids.logit_bias == {}
    neg = compile_token_steering('Never say "delve"', tokenizer_encode=lambda w: [-5])
    assert neg.logit_bias == {}

    # explicit temperature can tighten, never bypass, the safety cap
    assert compile_token_steering("Use temperature 0.2 for this.").temperature_cap == 0.2
    assert compile_token_steering("temperature 0").temperature_cap == 0.0
    assert compile_token_steering("temperature 1.9").temperature_cap == 0.5

    # huge inputs are capped and compile quickly
    huge = ('Never say "delve". ' * 30_000) + ("x" * 300_000)
    huge_spec = compile_token_steering(huge)
    assert huge_spec.forbidden_phrases == ["delve"]
    assert len(huge_spec.stop_sequences) <= _MAX_STOPS

    # to_dict returns detached copies
    blob = spec.to_dict()
    blob["logit_bias"][999999] = 1.0
    blob["response_format"]["type"] = "text"
    assert 999999 not in spec.logit_bias
    assert spec.response_format == {"type": "json_object"}

    print("tokensteer._self_test: OK PASS (logit_bias, stop sequences, JSON prefill, temperature cap)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
