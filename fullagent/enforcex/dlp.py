"""dlp — Enterprise Data Loss Prevention (DLP), Secret Scrubber & Reversible Token Vault.

Scans untrusted text for secrets and PII with a single normalized pass:

  * normalization: NFKC + zero-width stripping (U+200B–U+200F, U+2060, U+FEFF)
    so homoglyph / zero-width obfuscation cannot hide a credential;
  * coverage: PEM private keys, AWS AKIA/ASIA, GitHub, Slack, Stripe, OpenAI,
    Anthropic, Google API keys, GCP service-account markers, Azure connection
    strings, JWTs, database URIs, IPv4/IPv6, IBAN (mod-97 checked), SSN,
    credit cards (Luhn checked), e-mail, phone numbers;
  * limits: scans at most 262144 characters, redacts the unscanned tail as a
    placeholder, reports at most 512 findings, masks every value to <= 10 chars.

Self-test: ``python3 -m fullagent.enforcex.dlp``
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

__all__ = [
    "DLPMatch",
    "DLPReport",
    "VaultedRedactor",
    "iban_valid",
    "luhn_valid",
    "scan_text",
    "redact_text",
]

_MAX_SCAN = 262_144
_MAX_FINDINGS = 512
_MAX_RAW_HITS = 100_000
_TRUNCATED = "<REDACTED:TRUNCATED>"
_OVERFLOW = "<REDACTED:OVERFLOW>"

_ZERO_WIDTH = frozenset(
    [0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0xFEFF, 0x180E, 0x00AD]
)


@dataclass
class DLPMatch:
    kind: str
    category: str
    severity: str
    start: int
    end: int
    masked_value: str
    fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "category": self.category,
            "severity": self.severity,
            "start": self.start,
            "end": self.end,
            "masked_value": self.masked_value,
            "fingerprint": self.fingerprint,
        }

    def to_violation(self) -> Dict[str, Any]:
        return {
            "rule_id": f"DLP-{self.kind}",
            "category": f"dlp_{self.category}",
            "severity": self.severity,
            "evidence": self.masked_value,
            "message": f"Sensitive {self.category} detected ({self.kind})",
            "remediation": f"Redact or vault {self.kind} before logging or sending externally.",
        }


@dataclass
class DLPReport:
    clean: bool = True
    matches: List[DLPMatch] = field(default_factory=list)
    redacted_text: str = ""
    risk_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clean": self.clean,
            "matches_count": len(self.matches),
            "matches": [m.to_dict() for m in self.matches],
            "redacted_text": self.redacted_text,
            "risk_score": round(self.risk_score, 2),
        }


# ── validators ───────────────────────────────────────────────────────────

def luhn_valid(number_str: str) -> bool:
    try:
        digits = [int(c) for c in (number_str or "") if c.isdigit()]
    except (TypeError, ValueError):
        return False
    if not (13 <= len(digits) <= 19) or len(set(digits)) == 1:
        return False
    total = 0
    for i, d in enumerate(digits[::-1]):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_valid(iban: str) -> bool:
    """ISO 13616 IBAN check: mod-97 of the rearranged alphanumeric string == 1."""
    if not isinstance(iban, str):
        return False
    compact = re.sub(r"\s+", "", iban).upper()
    if not (15 <= len(compact) <= 34) or not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    remainder = 0
    for ch in rearranged:
        value = str(ord(ch) - 55) if ch.isalpha() else ch
        for digit in value:
            remainder = (remainder * 10 + int(digit)) % 97
    return remainder == 1


def _ipv6_valid(candidate: str) -> bool:
    try:
        ipaddress.IPv6Address(candidate)
        return True
    except (ipaddress.AddressValueError, ValueError):
        return False


def _true(_: str) -> bool:
    return True


_Validator = Callable[[str], bool]

_PRIVATE_PEM = (
    r"-----BEGIN (?:[A-Z0-9 ]{0,40})?PRIVATE KEY-----"
    r"[\s\S]{0,32768}?-----END (?:[A-Z0-9 ]{0,40})?PRIVATE KEY-----"
)

_PATTERNS: List[Tuple[str, str, str, "re.Pattern[str]", _Validator]] = [
    ("PRIVATE_KEY_PEM", "secret", "critical", re.compile(_PRIVATE_PEM), _true),
    ("PRIVATE_KEY_HEADER", "secret", "critical",
     re.compile(r"-----BEGIN (?:[A-Z0-9 ]{0,40})?PRIVATE KEY-----"), _true),
    ("ANTHROPIC_KEY", "secret", "critical",
     re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,200}\b"), _true),
    ("OPENAI_KEY", "secret", "critical",
     re.compile(r"\bsk-(?:proj-|live-|test-)?[A-Za-z0-9]{20,200}\b"), _true),
    ("AWS_ACCESS_KEY", "secret", "critical",
     re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), _true),
    ("GITHUB_TOKEN", "secret", "critical",
     re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{30,255})\b"), _true),
    ("GOOGLE_API_KEY", "secret", "high",
     re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), _true),
    ("GCP_SERVICE_ACCOUNT", "secret", "high",
     re.compile(r"[\"']?type[\"']?\s*:\s*[\"']service_account[\"']", re.I), _true),
    ("SLACK_TOKEN", "secret", "high",
     re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{16,200}\b"), _true),
    ("SLACK_WEBHOOK", "secret", "high",
     re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]{20,200}"), _true),
    ("STRIPE_SECRET", "secret", "critical",
     re.compile(r"\b(?:sk|rk)_(?:live|test)_[0-9a-zA-Z]{20,99}\b"), _true),
    ("JWT_TOKEN", "secret", "high",
     re.compile(r"\beyJ[A-Za-z0-9_-]{8,2000}\.[A-Za-z0-9_-]{8,4000}\.[A-Za-z0-9_-]{8,4000}\b"), _true),
    ("AZURE_CONN_STRING", "secret", "critical",
     re.compile(r"\bDefaultEndpointsProtocol=[A-Za-z]{2,16};AccountName=[^;\s\"']{1,128};"
                r"AccountKey=[A-Za-z0-9+/=]{20,256}(?:;[^\s\"']{1,256})?"), _true),
    ("DB_CONNECTION_URI", "infra", "critical",
     re.compile(r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|rediss|"
                r"amqp|amqps|mssql|sqlserver)://[^\s:@/]{0,128}:[^\s@/]{1,256}@[^\s\"'<>]{1,256}"), _true),
    ("US_SSN", "pii", "high",
     re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), _true),
    ("EMAIL_ADDRESS", "pii", "low",
     re.compile(r"\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,190}\.[A-Za-z]{2,12}\b"), _true),
    ("IPV4_ADDRESS", "pii", "low",
     re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
                r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"), _true),
    ("IPV6_ADDRESS", "pii", "low",
     re.compile(r"(?<![0-9A-Za-z:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![0-9A-Za-z:])"),
     _ipv6_valid),
    ("PHONE_NUMBER", "pii", "low",
     re.compile(r"(?<!\d)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?"
                r"\d{3,4}[\s.-]\d{3,4}(?:[\s.-]\d{2,4})?(?!\d)"), _true),
    ("IBAN", "pii", "medium",
     re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), iban_valid),
]

_CC_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

_RISK_WEIGHTS = {"critical": 45.0, "high": 25.0, "medium": 12.0, "low": 5.0}


def _mask(raw: str) -> str:
    """Constant-size masked preview (<= 10 chars)."""
    s = re.sub(r"\s+", " ", raw or "").strip()
    return "***" if len(s) <= 8 else f"{s[:4]}...{s[-3:]}"


def _normalize_with_map(text: str) -> Tuple[str, List[int]]:
    """NFKC + zero-width stripped copy plus a normalized→original index map."""
    out: List[str] = []
    index: List[int] = []
    for i, ch in enumerate(text):
        if ord(ch) in _ZERO_WIDTH:
            continue
        folded = unicodedata.normalize("NFKC", ch)
        for c in folded:
            out.append(c)
            index.append(i)
    return "".join(out), index


def _collect_hits(text: str, include_pii: bool) -> Tuple[List[Tuple[int, int, str, str, str]], bool]:
    """Return non-overlapping (start, end, kind, category, severity) hits.

    Spans index the original ``text``. Second value is True when the defensive
    hit cap was exceeded (caller must fail closed).
    """
    normalized, index_map = _normalize_with_map(text)
    if not normalized:
        return [], False
    raw_hits: List[Tuple[int, int, str, str, str]] = []
    overflow = False
    for kind, category, severity, pattern, validator in _PATTERNS:
        if not include_pii and category == "pii":
            continue
        for match in pattern.finditer(normalized):
            start, end = match.start(), match.end()
            if start >= end:
                continue
            if validator is not _true and not validator(match.group(0)):
                continue
            raw_hits.append((
                index_map[start], index_map[end - 1] + 1,
                kind, category, severity,
            ))
            if len(raw_hits) >= _MAX_RAW_HITS:
                overflow = True
                break
        if overflow:
            break

    if include_pii and not overflow:
        for match in _CC_CANDIDATE_RE.finditer(normalized):
            candidate = match.group(0)
            if luhn_valid(candidate):
                raw_hits.append((
                    index_map[match.start()], index_map[match.end() - 1] + 1,
                    "CREDIT_CARD", "pii", "high",
                ))

    raw_hits.sort(key=lambda hit: (hit[0], -(hit[1] - hit[0])))
    filtered: List[Tuple[int, int, str, str, str]] = []
    last_end = -1
    for hit in raw_hits:
        if hit[0] >= last_end:
            filtered.append(hit)
            last_end = hit[1]
    return filtered, overflow


def _matches_for(text: str,
                 hits: List[Tuple[int, int, str, str, str]]) -> List[DLPMatch]:
    matches: List[DLPMatch] = []
    for start, end, kind, category, severity in hits[:_MAX_FINDINGS]:
        raw_value = text[start:end]
        fingerprint = hashlib.sha256(
            raw_value.encode("utf-8", "replace")).hexdigest()[:12]
        matches.append(DLPMatch(
            kind=kind, category=category, severity=severity,
            start=start, end=end,
            masked_value=_mask(raw_value),
            fingerprint=fingerprint,
        ))
    return matches


def scan_text(text: str, *, include_pii: bool = True) -> DLPReport:
    if not text or not isinstance(text, str):
        return DLPReport(clean=True, redacted_text=str(text or ""))

    source = text[:_MAX_SCAN]
    hits, overflow = _collect_hits(source, include_pii)
    if overflow:
        # Fail closed: never emit text we could not fully classify.
        return DLPReport(clean=False, matches=[], redacted_text=_OVERFLOW,
                         risk_score=100.0)

    matches = _matches_for(source, hits)
    out_parts: List[str] = []
    cursor = 0
    severity_total = 0.0
    for idx, (start, end, kind, _category, severity) in enumerate(hits, 1):
        out_parts.append(source[cursor:start])
        out_parts.append(f"<REDACTED:{kind}:{idx}>")
        cursor = end
        severity_total += _RISK_WEIGHTS.get(severity, 10.0)
    out_parts.append(source[cursor:])
    redacted = "".join(out_parts)
    if len(source) < len(text):
        redacted += _TRUNCATED

    return DLPReport(
        clean=len(hits) == 0,
        matches=matches,
        redacted_text=redacted,
        risk_score=min(100.0, severity_total),
    )


def redact_text(text: str, *, include_pii: bool = True) -> str:
    return scan_text(text, include_pii=include_pii).redacted_text


class VaultedRedactor:
    def __init__(self, *, include_pii: bool = True) -> None:
        self.include_pii = include_pii
        self._lock = threading.Lock()
        self._vault: Dict[str, str] = {}
        self._counter = 0

    def scrub(self, text: str) -> Tuple[str, List[DLPMatch]]:
        if not text or not isinstance(text, str):
            return str(text or ""), []
        source = text[:_MAX_SCAN]
        hits, overflow = _collect_hits(source, self.include_pii)
        if overflow:
            return _OVERFLOW, []
        matches = _matches_for(source, hits)
        with self._lock:
            out_parts: List[str] = []
            cursor = 0
            for start, end, kind, _category, _severity in hits:
                self._counter += 1
                token = f"<VAULT:{kind}:{self._counter}>"
                self._vault[token] = source[start:end]
                out_parts.append(source[cursor:start])
                out_parts.append(token)
                cursor = end
            out_parts.append(source[cursor:])
            scrubbed = "".join(out_parts)
            if len(source) < len(text):
                scrubbed += _TRUNCATED
            return scrubbed, matches

    def restore(self, text: str) -> str:
        if not text or not isinstance(text, str) or "<VAULT:" not in text:
            return str(text or "")
        with self._lock:
            out = text
            for token, raw_val in self._vault.items():
                if token in out:
                    out = out.replace(token, raw_val)
            return out


def _self_test() -> int:
    sample = (
        "Connect to postgres://admin:s3cr3tPass@db.internal:5432/prod "
        "using key sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ123456 "
        "and AWS AKIAIOSFODNN7EXAMPLE or contact alice@example.com."
    )
    rep = scan_text(sample)
    assert not rep.clean, rep.to_dict()
    kinds = {m.kind for m in rep.matches}
    assert "DB_CONNECTION_URI" in kinds, kinds
    assert "OPENAI_KEY" in kinds, kinds
    assert "AWS_ACCESS_KEY" in kinds, kinds
    assert "EMAIL_ADDRESS" in kinds, kinds
    assert "s3cr3tPass" not in rep.redacted_text
    assert "sk-proj-" not in rep.redacted_text

    assert luhn_valid("4532 0151 1283 0366")
    assert not luhn_valid("4532 0151 1283 0367")

    vr = VaultedRedactor()
    scrubbed, matches = vr.scrub(sample)
    assert len(matches) == 4 and "<VAULT:" in scrubbed
    assert vr.restore(scrubbed) == sample

    # ── new adversarial coverage ────────────────────────────────────────
    # 1. zero-width + fullwidth homoglyphs cannot hide credentials
    hidden = "AKIA\u200bIOSFODNN7EXAMPLE and ＡＫＩＡIOSFODNN7EXAMPLX"
    hidden_rep = scan_text(hidden)
    assert sum(1 for m in hidden_rep.matches if m.kind == "AWS_ACCESS_KEY") >= 2, hidden_rep.to_dict()

    # 2. JWT, Azure connection string and GitHub/Slack/Stripe tokens
    multi = (
        "token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U "
        "DefaultEndpointsProtocol=https;AccountName=acct;AccountKey="
        "AbCdEf0123456789AbCdEf0123456789AbCdEf==;EndpointSuffix=core.windows.net "
        "ghp_" "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        "xoxb-" "123456789012" "-abcdefghijklmnop "
        "sk_live_" "ABCDEFGHIJKLMNOPQRSTUVWX"
    )
    multi_kinds = {m.kind for m in scan_text(multi).matches}
    for kind in ("JWT_TOKEN", "AZURE_CONN_STRING", "GITHUB_TOKEN",
                 "SLACK_TOKEN", "STRIPE_SECRET"):
        assert kind in multi_kinds, (kind, multi_kinds)

    # 3. IBAN is mod-97 validated and credit cards are Luhn validated
    assert iban_valid("GB82 WEST 1234 5698 7654 32")
    assert not iban_valid("GB82 WEST 1234 5698 7654 33")
    assert any(m.kind == "IBAN" for m in scan_text("acct GB82WEST12345698765432").matches)
    assert not any(m.kind == "CREDIT_CARD" for m in scan_text("num 4532015112830367").matches)

    # 4. IPv4 / IPv6 / phone are covered and bounded
    net_rep = scan_text("host 10.0.0.7 and 2001:db8::1 call +1 (555) 123-4567")
    net_kinds = {m.kind for m in net_rep.matches}
    assert {"IPV4_ADDRESS", "IPV6_ADDRESS", "PHONE_NUMBER"} <= net_kinds, net_kinds

    # 5. include_pii=False leaves PII alone but still removes secrets
    secrets_only = scan_text("mail alice@example.com key AKIAIOSFODNN7EXAMPLE",
                             include_pii=False)
    assert {m.kind for m in secrets_only.matches} == {"AWS_ACCESS_KEY"}
    assert "alice@example.com" in secrets_only.redacted_text

    # 6. masked values and findings stay capped; huge input stays linear
    huge_rep = scan_text("x" * 1_000_000 + " AKIAIOSFODNN7EXAMPLE")
    assert len(huge_rep.redacted_text) <= _MAX_SCAN + len(_TRUNCATED)
    assert huge_rep.redacted_text.endswith(_TRUNCATED)
    many = " ".join(["alice@example.com"] * 900)
    many_rep = scan_text(many)
    assert len(many_rep.matches) <= _MAX_FINDINGS
    assert all(len(m.masked_value) <= 10 for m in many_rep.matches)

    # 7. truncated PEM header is still critical, invalid IPv6 is not flagged
    pem_rep = scan_text("-----BEGIN RSA PRIVATE KEY-----\nMIIEow...")
    assert any(m.kind == "PRIVATE_KEY_HEADER" for m in pem_rep.matches)
    assert not any(m.kind == "IPV6_ADDRESS" for m in scan_text("time 12:30:45").matches)

    # 8. non-str input and empty input fail soft
    assert scan_text(None).clean  # type: ignore[arg-type]
    assert scan_text(12345).clean  # type: ignore[arg-type]
    assert redact_text("") == ""

    print("dlp._self_test: OK PASS (keys, DB URI, Luhn CC, PII, reversible vault)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
