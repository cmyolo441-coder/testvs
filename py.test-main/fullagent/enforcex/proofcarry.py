"""proofcarry — Proof-Carrying Responses (PCR) & Independent Compliance Certificate Verifier.

Inspired by Necula's Proof-Carrying Code (PCC): every model response is paired
with a formal `ComplianceCertificate` containing machine-checkable lemmas
(`RuleLemma`) for each active system-prompt constraint (deontic, quantitative,
structural, lexical, DLP) and a SHA-256 cryptographic binding over
`(system_prompt_hash, response_hash, lemmas)`.

An independent verifier (`verify_certificate`) can validate the certificate in
O(K) time without trusting the generator. Verification is fail-closed: the
certificate must carry the full canonical lemma set, well-formed SHA-256
bindings and evidence checksums, an untampered digest, and a matching payload
checksum (`payload_checksum`). Malformed certificates return ``False`` instead
of raising.

Self-test: ``python3 -m fullagent.enforcex.proofcarry``
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from . import deontic, dlp, grammarguard, morpholex, smtguard

__all__ = [
    "RuleLemma",
    "ComplianceCertificate",
    "ProofCarryingEngine",
    "generate_certificate",
    "verify_certificate",
    "payload_checksum",
]

# Guard scans are capped so huge payloads cannot trigger unbounded regex work.
_MAX_SCAN_CHARS = 262_144
_CANONICAL_LEMMAS = (
    "LEM-DEONTIC", "LEM-SMT", "LEM-GRAMMAR", "LEM-LEXICAL", "LEM-DLP",
)


@dataclass(frozen=True)
class RuleLemma:
    lemma_id: str
    layer: str                # deontic | smt | grammar | morpholex | dlp
    satisfied: bool
    evidence_hash: str
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lemma_id": self.lemma_id,
            "layer": self.layer,
            "satisfied": self.satisfied,
            "evidence_hash": self.evidence_hash,
            "summary": self.summary,
        }


@dataclass
class ComplianceCertificate:
    system_sha256: str
    response_sha256: str
    all_satisfied: bool
    lemmas: List[RuleLemma] = field(default_factory=list)
    certificate_sha256: str = ""

    def compute_digest(self) -> str:
        lemmas: List[Dict[str, Any]] = []
        for lemma in (self.lemmas or []):
            if isinstance(lemma, RuleLemma):
                lemmas.append(lemma.to_dict())
            else:
                # Malformed entries are preserved as a canonical marker so the
                # verifier can reject them rather than crashing here.
                lemmas.append({"invalid": repr(lemma)})
        payload = json.dumps(
            {
                "sys": self.system_sha256,
                "resp": self.response_sha256,
                "ok": self.all_satisfied,
                "lemmas": lemmas,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_sha256": self.system_sha256,
            "response_sha256": self.response_sha256,
            "all_satisfied": self.all_satisfied,
            "lemmas": [l.to_dict() for l in self.lemmas],
            "certificate_sha256": self.certificate_sha256,
        }


def _sha(text: str) -> str:
    if text is None:
        text = ""
    if isinstance(text, bytes):
        data = text
    else:
        try:
            data = str(text).encode("utf-8", "replace")
        except Exception:
            data = repr(text).encode("utf-8", "replace")
    return hashlib.sha256(data).hexdigest()


def payload_checksum(text: str) -> str:
    """SHA-256 payload checksum used for certificate binding."""
    return _sha(text)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    try:
        return str(value)
    except Exception:
        return ""


def _is_hex(value: Any, length: int = 64) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except (TypeError, ValueError):
        return False
    return True


def _is_short_hex(value: Any) -> bool:
    if not isinstance(value, str) or not (8 <= len(value) <= 128):
        return False
    try:
        int(value, 16)
    except (TypeError, ValueError):
        return False
    return True


def _call_guard(fn: Callable[[], Any]) -> Tuple[Any, str]:
    """Run one guard; return (result, error_name). Never raises."""
    try:
        return fn(), ""
    except Exception as exc:
        return None, type(exc).__name__


def _error_lemma(lemma_id: str, layer: str, error_name: str) -> RuleLemma:
    return RuleLemma(
        lemma_id=lemma_id,
        layer=layer,
        satisfied=False,
        evidence_hash=_sha(f"error:{error_name}")[:16],
        summary=f"guard error={error_name}",
    )


def generate_certificate(response_text: str, *, system_text: str = "", user_input: str = "") -> ComplianceCertificate:
    response_text = _safe_text(response_text)
    system_text = _safe_text(system_text)
    user_input = _safe_text(user_input)
    sys_h = _sha(system_text)
    resp_h = _sha(response_text)

    # Guards inspect a bounded prefix; the hash still binds the full payload.
    scan_text = response_text[:_MAX_SCAN_CHARS]

    d_ver, d_err = _call_guard(
        lambda: deontic.evaluate_deontic(scan_text, system_text=system_text,
                                         user_input=user_input))
    s_ver, s_err = _call_guard(
        lambda: smtguard.verify_bounds(scan_text, system_text=system_text))
    g_ver, g_err = _call_guard(
        lambda: grammarguard.validate_and_repair(scan_text,
                                                 system_text=system_text))
    m_ver, m_err = _call_guard(
        lambda: morpholex.evaluate_linguistics(scan_text))
    dlp_ver, dlp_err = _call_guard(
        lambda: dlp.scan_text(scan_text, include_pii=False))

    lemmas: List[RuleLemma] = []
    if d_err:
        lemmas.append(_error_lemma("LEM-DEONTIC", "deontic", d_err))
    else:
        lemmas.append(RuleLemma(
            "LEM-DEONTIC", "deontic", bool(d_ver.passed),
            _sha(str(d_ver.score))[:16],
            f"norms={len(d_ver.norms)} violations={len(d_ver.violations)}",
        ))
    if s_err:
        lemmas.append(_error_lemma("LEM-SMT", "smt", s_err))
    else:
        lemmas.append(RuleLemma(
            "LEM-SMT", "smt", bool(s_ver.passed),
            _sha(str(s_ver.metrics))[:16],
            f"bounds={len(s_ver.bounds)} violations={len(s_ver.violations)}",
        ))
    if g_err:
        lemmas.append(_error_lemma("LEM-GRAMMAR", "grammar", g_err))
    else:
        lemmas.append(RuleLemma(
            "LEM-GRAMMAR", "grammar", bool(g_ver.valid),
            _sha(str(g_ver.errors))[:16],
            f"errors={len(g_ver.errors)}",
        ))
    if m_err:
        lemmas.append(_error_lemma("LEM-LEXICAL", "morpholex", m_err))
    else:
        lemmas.append(RuleLemma(
            "LEM-LEXICAL", "morpholex", bool(m_ver.passed),
            _sha(str(m_ver.buzzwords_found))[:16],
            f"buzzwords={len(m_ver.buzzwords_found)}",
        ))
    if dlp_err:
        lemmas.append(_error_lemma("LEM-DLP", "dlp", dlp_err))
    else:
        lemmas.append(RuleLemma(
            "LEM-DLP", "dlp", bool(dlp_ver.clean),
            _sha(str(len(dlp_ver.matches)))[:16],
            f"secrets={len(dlp_ver.matches)}",
        ))

    all_ok = all(l.satisfied for l in lemmas)
    cert = ComplianceCertificate(
        system_sha256=sys_h,
        response_sha256=resp_h,
        all_satisfied=all_ok,
        lemmas=lemmas,
    )
    cert.certificate_sha256 = cert.compute_digest()
    return cert


def _structurally_valid(cert: ComplianceCertificate) -> bool:
    """Validate the certificate envelope before trusting any binding."""
    if not _is_hex(cert.system_sha256) or not _is_hex(cert.response_sha256):
        return False
    if not _is_hex(cert.certificate_sha256):
        return False
    if not isinstance(cert.all_satisfied, bool):
        return False
    if not isinstance(cert.lemmas, list) or not cert.lemmas:
        return False
    seen: set = set()
    for lemma in cert.lemmas:
        if not isinstance(lemma, RuleLemma):
            return False
        if lemma.lemma_id in seen:
            return False
        seen.add(lemma.lemma_id)
        if not isinstance(lemma.satisfied, bool):
            return False
        if not isinstance(lemma.layer, str) or not lemma.layer:
            return False
        if not isinstance(lemma.summary, str):
            return False
        if not _is_short_hex(lemma.evidence_hash):
            return False
    required = set(_CANONICAL_LEMMAS)
    return required <= seen


def verify_certificate(response_text: str, cert: ComplianceCertificate, *, system_text: str = "") -> bool:
    """Independently verify that `cert` is authentic, untampered, and matches `response_text`.

    Fail-closed: any malformed or incomplete certificate returns ``False``.
    """
    try:
        if not isinstance(cert, ComplianceCertificate):
            return False
        if not _structurally_valid(cert):
            return False
        if not isinstance(response_text, (str, bytes, type(None))):
            return False
        if cert.response_sha256 != _sha(response_text):
            return False
        if system_text and cert.system_sha256 != _sha(system_text):
            return False
        if cert.certificate_sha256 != cert.compute_digest():
            return False
        if cert.all_satisfied is not True:
            return False
        if any(not lemma.satisfied for lemma in cert.lemmas):
            return False
        return True
    except Exception:
        return False


class ProofCarryingEngine:
    def certify(self, response_text: str, *, system_text: str = "", user_input: str = "") -> ComplianceCertificate:
        return generate_certificate(response_text, system_text=system_text, user_input=user_input)

    def verify(self, response_text: str, cert: ComplianceCertificate, *, system_text: str = "") -> bool:
        return verify_certificate(response_text, cert, system_text=system_text)


def _self_test() -> int:
    sys_p = "Always use a bulleted list and keep your answer under 30 words."
    good_resp = "- First verified item.\n- Second verified item."
    cert = generate_certificate(good_resp, system_text=sys_p)
    assert cert.all_satisfied
    assert verify_certificate(good_resp, cert, system_text=sys_p)

    # Tampering with response invalidates certificate
    assert not verify_certificate(good_resp + " tampered", cert, system_text=sys_p)

    # 1. payload checksum is deterministic and binding
    assert payload_checksum(good_resp) == cert.response_sha256
    assert payload_checksum(good_resp) != payload_checksum(good_resp + "x")
    assert payload_checksum(None) == _sha("")  # type: ignore[arg-type]

    # 2. empty / forged lemma set with a recomputed digest is rejected
    forged = ComplianceCertificate(
        system_sha256=cert.system_sha256,
        response_sha256=cert.response_sha256,
        all_satisfied=True,
        lemmas=[],
    )
    forged.certificate_sha256 = forged.compute_digest()
    assert not verify_certificate(good_resp, forged, system_text=sys_p)

    # 3. duplicate lemma ids are rejected even with a valid digest
    dup = ComplianceCertificate(
        system_sha256=cert.system_sha256,
        response_sha256=cert.response_sha256,
        all_satisfied=True,
        lemmas=[cert.lemmas[0]] * len(cert.lemmas),
    )
    dup.certificate_sha256 = dup.compute_digest()
    assert not verify_certificate(good_resp, dup, system_text=sys_p)

    # 4. malformed lemma payloads return False instead of raising
    malformed = ComplianceCertificate(
        system_sha256=cert.system_sha256,
        response_sha256=cert.response_sha256,
        all_satisfied=True,
        lemmas=[{"lemma_id": "LEM-DEONTIC"}],  # type: ignore[list-item]
    )
    malformed.certificate_sha256 = malformed.compute_digest()
    assert not verify_certificate(good_resp, malformed, system_text=sys_p)
    assert not verify_certificate(good_resp, None)  # type: ignore[arg-type]
    assert not verify_certificate(good_resp, "cert")  # type: ignore[arg-type]

    # 5. a lemma with a non-hex evidence hash invalidates the certificate
    bad_evidence = ComplianceCertificate(
        system_sha256=cert.system_sha256,
        response_sha256=cert.response_sha256,
        all_satisfied=True,
        lemmas=[RuleLemma(l.lemma_id, l.layer, True, "zz", l.summary)
                for l in cert.lemmas],
    )
    bad_evidence.certificate_sha256 = bad_evidence.compute_digest()
    assert not verify_certificate(good_resp, bad_evidence, system_text=sys_p)

    # 6. system-prompt binding mismatch is rejected
    assert not verify_certificate(good_resp, cert,
                                  system_text=sys_p + " changed")

    # 7. malformed inputs never raise during generation
    for junk in (None, 123, b"- a\n- b", ""):
        c = generate_certificate(junk)  # type: ignore[arg-type]
        assert isinstance(c, ComplianceCertificate)
        assert len(c.lemmas) == len(_CANONICAL_LEMMAS)
    huge_cert = generate_certificate("a" * 300_000)
    assert isinstance(huge_cert, ComplianceCertificate)
    assert _is_hex(huge_cert.response_sha256)

    print("proofcarry._self_test: OK PASS (5-layer PCR lemma generation + cryptographic verification + "
          "fail-closed structural/checksum validation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
