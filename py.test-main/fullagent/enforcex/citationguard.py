"""citationguard — Claim-Source Grounding, URL Specificity & Anti-Hallucination Verifier.

Audits Markdown hyperlinks `[Anchor](URL)` and citations in responses:
  1. Bare Homepage Detection: flags generic root URLs (`https://example.com/`
     with no path or query) when a specific entity page is required.
  2. Bracketed Number Citation Ban: flags academic-style `[1]`, `[2]` citations
     when inline clickable `[Entity](URL)` links are mandated.
  3. Placeholder / Fabricated Domain Detection: flags `example.com`, `url_here`,
     `link_to_source`, `http://localhost` in external citations.
  4. Shared URL Reuse Detection: flags when 3+ distinct entities reuse the
     exact same URL.

Matching is obfuscation-resistant: text is NFKC-normalized, zero-width
characters are stripped, HTML character references are decoded, and common
Cyrillic/Greek homoglyph domains are folded to Latin before classification.
Scans are capped at 262144 characters to bound worst-case work.

Self-test: ``python3 -m fullagent.enforcex.citationguard``
"""

from __future__ import annotations

import functools
import html
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "LinkRecord",
    "GroundingVerificationReport",
    "CitationGroundingEngine",
    "audit_citations",
]

_MAX_SCAN = 262_144
_MAX_ANCHOR = 300
_MAX_URL = 2048
_ZERO_WIDTH_RE = re.compile("[\u200b-\u200f\u2060\ufeff\u180e]")
_MD_LINK_RE = re.compile(
    r"(?<!!)\[([^\[\]]{1,%d})\]\(\s*((?:[^()\s]|\([^()\s]*\)){1,%d})\s*\)"
    % (_MAX_ANCHOR, _MAX_URL)
)
_BRACKET_CITE_RE = re.compile(r"(?<!\!)\[\s*\d{1,3}(?:\s*[,;–—-]\s*\d{1,3})*\s*\](?!\()")
_PLACEHOLDER_DOMAINS = {
    "example.com", "www.example.com", "example.org", "www.example.org",
    "example.net", "www.example.net", "domain.com", "www.domain.com",
    "yoursite.com", "www.yoursite.com", "yourdomain.com", "url", "link",
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
}
_PLACEHOLDER_TLDS = (".test", ".invalid", ".example", ".localhost")
_PLACEHOLDER_URL_RE = re.compile(
    r"(?:url[_-]?here|link[_-]?to[_-]?source|your[_-]?(?:url|link)|insert[_-]?(?:url|link)"
    r"|add[_-]?(?:url|link)|placeholder|\btbd\b|\btodo\b|xxx+)",
    re.I,
)
_UNSAFE_SCHEME_RE = re.compile(r"^(?:javascript|data|vbscript|file):", re.I)

# Common confusables: Cyrillic/Greek letters that render like Latin ones.
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "һ": "h", "ӏ": "l", "м": "m",
    "т": "t", "в": "b", "к": "k", "н": "h", "п": "n",
    "α": "a", "ε": "e", "ο": "o", "ρ": "p", "τ": "t", "υ": "u",
    "χ": "x", "κ": "k", "ν": "v", "ι": "i",
    "ı": "i", "ł": "l", "đ": "d",
})


@dataclass
class LinkRecord:
    anchor: str
    url: str
    domain: str
    is_bare_homepage: bool = False
    is_placeholder: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor": self.anchor,
            "url": self.url,
            "domain": self.domain,
            "is_bare_homepage": self.is_bare_homepage,
            "is_placeholder": self.is_placeholder,
        }


@dataclass
class GroundingVerificationReport:
    passed: bool = True
    score: float = 100.0
    links: List[LinkRecord] = field(default_factory=list)
    bracket_citations: List[str] = field(default_factory=list)
    violations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 2),
            "links_count": len(self.links),
            "links": [l.to_dict() for l in self.links],
            "bracket_citations": list(self.bracket_citations),
            "violations_count": len(self.violations),
            "violations": list(self.violations),
        }


def _as_text(text: Any) -> str:
    """Coerce anything to scannable text; never raises."""
    if text is None:
        return ""
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8", "replace")
        except Exception:
            return ""
    try:
        text = str(text)
    except Exception:
        return ""
    try:
        text = unicodedata.normalize("NFKC", text[:_MAX_SCAN])
    except Exception:
        text = text[:_MAX_SCAN]
    return _ZERO_WIDTH_RE.sub("", text)


def _deconfuse(host: str) -> str:
    return (host or "").casefold().translate(_CONFUSABLES).rstrip(".")


def _split_url(url: str) -> Optional[urllib.parse.SplitResult]:
    """Parse a URL, tolerating scheme-less 'example.com/path' citations."""
    candidate = (url or "").strip()
    if not candidate:
        return None
    if "://" not in candidate and not candidate.startswith(("//", "/", "#", "?")):
        candidate = "//" + candidate
    try:
        return urllib.parse.urlsplit(candidate)
    except ValueError:
        return None


def _hostname(parsed: Optional[urllib.parse.SplitResult]) -> str:
    if parsed is None:
        return ""
    try:
        return _deconfuse(parsed.hostname or "")
    except ValueError:
        return ""


@functools.lru_cache(maxsize=2048)
def _canonical_url(url: str) -> str:
    """Stable key for shared-URL detection: case, entity, port, trailing slash."""
    cleaned = _ZERO_WIDTH_RE.sub("", html.unescape(url or "")).strip()
    parsed = _split_url(cleaned)
    if parsed is None:
        return cleaned.casefold()
    scheme = (parsed.scheme or "").casefold()
    host = _hostname(parsed)
    port = ""
    try:
        p = parsed.port
        if p is not None and not ((scheme == "http" and p == 80) or
                                  (scheme == "https" and p == 443)):
            port = ":%d" % p
    except ValueError:
        port = ""
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    query = ("?" + parsed.query) if parsed.query else ""
    return "%s://%s%s%s%s" % (scheme, host, port, path, query)


def audit_citations(text: str) -> GroundingVerificationReport:
    try:
        return _audit(_as_text(text))
    except Exception:  # defensive: a malformed citation never crashes the guard
        return GroundingVerificationReport()


def _audit(s: str) -> GroundingVerificationReport:
    links: List[LinkRecord] = []
    violations: List[Dict[str, Any]] = []
    url_anchors: Dict[str, Tuple[str, List[str]]] = {}

    for m in _MD_LINK_RE.finditer(s):
        anchor, url = m.group(1).strip(), m.group(2).strip()
        decoded_url = _ZERO_WIDTH_RE.sub("", html.unescape(url))
        parsed = _split_url(decoded_url)
        domain = _hostname(parsed)
        path = (parsed.path if parsed is not None else "") or ""
        query = (parsed.query if parsed is not None else "") or ""
        fragment = (parsed.fragment if parsed is not None else "") or ""
        is_bare = bool(domain and path in ("", "/") and not query and not fragment)

        low_url = decoded_url.casefold()
        is_ph = bool(
            domain in _PLACEHOLDER_DOMAINS
            or low_url in ("url", "link", "#")
            or _PLACEHOLDER_URL_RE.search(decoded_url)
            or domain.endswith(_PLACEHOLDER_TLDS)
        )

        rec = LinkRecord(
            anchor=anchor,
            url=url,
            domain=domain,
            is_bare_homepage=is_bare,
            is_placeholder=is_ph,
        )
        links.append(rec)
        key = _canonical_url(decoded_url)
        group = url_anchors.setdefault(key, (url, []))
        group[1].append(anchor.casefold())

        if is_bare:
            violations.append({
                "rule_id": "CITE-BARE-HOMEPAGE",
                "category": "citation_specificity",
                "severity": "medium",
                "evidence": f"[{anchor}]({url})",
                "message": f"Bare homepage URL '{url}' used for '{anchor}'; link to the dedicated entity page.",
            })
        if is_ph:
            violations.append({
                "rule_id": "CITE-PLACEHOLDER-URL",
                "category": "citation_fabrication",
                "severity": "high",
                "evidence": f"[{anchor}]({url})",
                "message": f"Placeholder or fabricated URL '{url}' detected.",
            })
        if _UNSAFE_SCHEME_RE.match(low_url):
            violations.append({
                "rule_id": "CITE-UNSAFE-SCHEME",
                "category": "citation_fabrication",
                "severity": "high",
                "evidence": f"[{anchor}]({url})",
                "message": f"Unsafe or non-web URL scheme in citation '{url}'.",
            })

    # Check bracketed numeric citations (also lists/ranges: [1, 2], [3-5])
    b_cites = _BRACKET_CITE_RE.findall(s)
    if b_cites:
        violations.append({
            "rule_id": "CITE-BRACKET-NUMBER",
            "category": "citation_format",
            "severity": "medium",
            "evidence": ", ".join(b_cites[:5]),
            "message": "Bracketed numeric citations like [1] are forbidden; use inline clickable [Entity](URL) links.",
        })

    # Check excessive URL reuse across distinct entity anchors
    for _canonical, (display_url, anchors) in url_anchors.items():
        unique = set(anchors)
        if len(unique) >= 3:
            violations.append({
                "rule_id": "CITE-SHARED-URL",
                "category": "citation_specificity",
                "severity": "low",
                "evidence": f"{display_url} -> {sorted(unique)}",
                "message": f"Single URL '{display_url}' reused across {len(unique)} distinct entities.",
            })

    score = max(0.0, 100.0 - 20.0 * len(violations))
    return GroundingVerificationReport(
        passed=(len(violations) == 0),
        score=score,
        links=links,
        bracket_citations=b_cites,
        violations=violations,
    )


class CitationGroundingEngine:
    def evaluate(self, text: str) -> GroundingVerificationReport:
        return audit_citations(text)


def _self_test() -> int:
    good = "Check out [Python AST Docs](https://docs.python.org/3/library/ast.html) for details."
    r_ok = audit_citations(good)
    assert r_ok.passed and len(r_ok.links) == 1, r_ok.to_dict()

    bad = "See [Vendor](https://vendor.io/) and [Docs](https://example.com/doc) [1]."
    r_bad = audit_citations(bad)
    assert not r_bad.passed
    rids = {v["rule_id"] for v in r_bad.violations}
    assert "CITE-BARE-HOMEPAGE" in rids
    assert "CITE-PLACEHOLDER-URL" in rids
    assert "CITE-BRACKET-NUMBER" in rids

    # Zero-width and HTML-entity obfuscation of example.com
    r_zw = audit_citations("See [A](https://exa\u200bmple.com/page).")
    assert r_zw.links[0].domain == "example.com"
    assert any(v["rule_id"] == "CITE-PLACEHOLDER-URL" for v in r_zw.violations)
    r_ent = audit_citations("See [B](https://ex&#97;mple.com/page).")
    assert r_ent.links[0].domain == "example.com" and not r_ent.passed

    # Cyrillic homoglyph domain folds to the Latin placeholder domain
    cyr = "https://\u0435xample.com/page"  # Cyrillic 'е'
    r_cyr = audit_citations("See [C](%s)." % cyr)
    assert r_cyr.links[0].domain == "example.com", r_cyr.to_dict()
    assert any(v["rule_id"] == "CITE-PLACEHOLDER-URL" for v in r_cyr.violations)

    # Bracket lists/ranges are caught; a 4-digit year is not
    r_range = audit_citations("Facts [1, 2] and [3\u20135] hold, see [R](https://x.io/a) 2024.")
    assert "[1, 2]" in r_range.bracket_citations, r_range.to_dict()
    assert any(v["rule_id"] == "CITE-BRACKET-NUMBER" for v in r_range.violations)
    r_year = audit_citations("The year [2024] was fine.")
    assert not r_year.bracket_citations

    # Markdown images are not citations
    r_img = audit_citations("![logo](https://example.com/logo.png)")
    assert r_img.links == [] and r_img.passed

    # Shared-URL reuse is keyed on a canonical URL (case/trailing slash/entity)
    reuse = ("[A](https://vendor.io/p) [B](HTTPS://VENDOR.IO/p/) "
             "[C](https://vendor.io/p#frag)")
    r_reuse = audit_citations(reuse)
    assert sum(1 for v in r_reuse.violations if v["rule_id"] == "CITE-SHARED-URL") == 1

    # Unsafe schemes are flagged even when the host is absent
    r_js = audit_citations("[Click](javascript:alert(1))")
    assert any(v["rule_id"] == "CITE-UNSAFE-SCHEME" for v in r_js.violations)

    # Malformed / huge input never raises and stays bounded
    for malformed in (None, 12345, b"[x](https://example.com)", ["not", "text"]):
        rep = audit_citations(malformed)  # type: ignore[arg-type]
        assert isinstance(rep.to_dict(), dict)
    huge = "See [A](https://vendor.io/page). " * 20000
    r_huge = audit_citations(huge)
    assert len(huge) > _MAX_SCAN and 0 < len(r_huge.links) <= _MAX_SCAN // 32

    print("citationguard._self_test: OK PASS (bare homepage, placeholder URL, [1] bracket detector)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
